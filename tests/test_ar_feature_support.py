"""Do the autoregressive channels stay inside the range the weights were fitted on?

`docs/feature_engineering.md` §4 ("Which ones to prefer") warns in prose that
`cumulative_transactions`, `cumulative_count` and `period_since_first_transaction`
"grow without bound and, over a long holdout, drift past the range the model ever
saw in calibration". Nothing asserted it, so a config could opt into all three and
the only symptom was a forecast that came out several hundred percent high.

Measuring it sharpens the warning. The quantity these tests use is the
**support-escape fraction**: of all the (customer, period) cells a rollout feeds the
model over the holdout, what share carry a value for this channel outside
`[min, max]` of the same channel over the whole calibration window? Calibration is
exactly the data the weights and the standardisation statistics were fitted on, so a
cell outside that interval is one where the shared `nn.Linear` covariate projection
extrapolates rather than interpolates.

By that measure the three warned-about features do NOT behave alike, and the
difference is structural rather than a property of any one panel:

* `period_since_first_transaction` and `period_since_last_transaction` are capped in
  calibration by the calibration window length -- neither can exceed `T_CAL` there,
  because there are only that many periods to count. Over the holdout they keep
  counting, so they necessarily leave the fitted range. Tenure is the extreme case:
  under `require_calibration_activity=True` every retained customer's clock is
  already running when the holdout opens, so the entire cohort marches past the
  calibration maximum together.
* `cumulative_transactions` and `cumulative_count` grow without bound in principle,
  but their calibration maximum is set by the panel's heaviest buyer, which is
  typically far above where the median customer ends the holdout. In practice almost
  nothing escapes.

These are **characterisation tests**: they describe the pipeline as it behaves today
and pass on the current package, so no code changes with them. What they buy is a
tripwire -- if a channel is ever clipped, reparametrised or dropped, the thresholds
here fail loudly and have to be re-stated deliberately rather than drifting.

They stay dataset-agnostic (`CLAUDE.md`'s dataset-agnostic priority) and need no
trained checkpoint, so they measure the input-side precondition for the failure
rather than the forecast error itself.

Run:  pytest -q tests/test_ar_feature_support.py
"""

import numpy as np
import pandas as pd
import pytest

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.ar_features import compute_ar_feature_columns
from panelclv.data_preparation.target_channel import holdout_actuals

PANEL_SEED = 20260901
N_CUSTOMERS = 120
# Two calibration years then one holdout year, at the weekly frequency the panels use.
# The ratio matters: the escape fraction grows with holdout length relative to
# calibration length, which is the point of the warning being tested.
N_CAL_PERIODS = 104
N_HOLD_PERIODS = 52

# Capped in calibration by the window length, so they must leave the range over a
# holdout. Tenure escapes for the whole cohort; recency only for customers who go
# quiet for longer than anyone did during calibration.
WINDOW_CAPPED = ("period_since_first_transaction", "period_since_last_transaction")
# Unbounded in principle (the docs group these with tenure) but bounded in practice
# by the heaviest calibration buyer.
RUNNING_COUNTERS = ("cumulative_transactions", "cumulative_count")
# Bounded and stationary by construction.
BOUNDED = ("has_transacted_before", "active_in_last_4_periods", "transaction_rate")

ALL_FEATURES = WINDOW_CAPPED + RUNNING_COUNTERS + BOUNDED


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _panel() -> pd.DataFrame:
    """One row per customer per week: a Poisson panel with a spread of activity.

    Rates are drawn per customer so the panel holds both frequent and near-dormant
    buyers, which is what makes the bounded channels (a rate, a flag) genuinely vary
    rather than sit at a constant.
    """
    rng = np.random.default_rng(PANEL_SEED)
    rates = rng.gamma(shape=1.2, scale=0.05, size=N_CUSTOMERS)
    weeks = pd.date_range("1999-01-04", periods=N_CAL_PERIODS + N_HOLD_PERIODS, freq="7D")
    iso = [(d.isocalendar()[0], d.isocalendar()[1]) for d in weeks]

    frames = []
    for i, rate in enumerate(rates):
        counts = rng.poisson(rate, size=len(weeks))
        # Guarantee at least one calibration transaction, so `require_calibration_
        # activity` keeps the customer and the tenure clock starts inside calibration.
        # This mirrors the real cohort rule rather than working around it.
        if counts[:N_CAL_PERIODS].sum() == 0:
            counts[rng.integers(0, N_CAL_PERIODS)] = 1
        frames.append(
            pd.DataFrame(
                {
                    "Id": f"C{i:03d}",
                    "year": [y for y, _ in iso],
                    "week": [w for _, w in iso],
                    "Transactions": counts,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _config(ar_features) -> PanelConfig:
    """The panel's column roles, with `ar_features` as the only varying part."""
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        training_start="1999-01-04",
        training_end="2000-12-25",
        validation_start="2000-01-03",
        holdout_start="2001-01-01",
        holdout_end="2001-12-31",
        time_cols=("year", "week"),
        clip_target_upper=6,
        require_calibration_activity=True,
        time_features={"add_year_idx": True, "add_week_sin_cos": True},
        ar_features=tuple(ar_features),
        known_future=(),
        static=(),
        observed_past=(),
        embedded_cols={"Transactions": "auto"},
    )


@pytest.fixture(scope="module")
def prepared() -> dict:
    """`prepare_dataset` output carrying every feature under test."""
    return panel_dataset.prepare_dataset(_panel(), _config(ALL_FEATURES), verbose=False)


@pytest.fixture(scope="module")
def trajectories(prepared) -> dict:
    """`{feature: (N, T_CAL + T_HOLD)}` in raw units, plus the calibration length.

    Built from the TRUE counts across both windows. That is the best case for the
    model: during a real rollout the holdout half is rebuilt from *sampled* counts,
    which can only add noise on top of whatever drift is measured here. So a support
    gap found on this array is a lower bound on the gap the rollout actually feeds.
    """
    target_idx = int(prepared["target_idx"])
    calibration_counts = np.asarray(prepared["calibration"])[:, :, target_idx].astype(np.int64)
    holdout_counts = holdout_actuals(prepared).astype(np.int64)
    full_counts = np.concatenate([calibration_counts, holdout_counts], axis=1)
    out = compute_ar_feature_columns(full_counts, ALL_FEATURES)
    out["__t_cal__"] = calibration_counts.shape[1]
    return out


def _escape_fraction(trajectory: np.ndarray, t_cal: int) -> float:
    """Share of holdout cells falling outside the calibration `[min, max]`.

    The interval is taken over every calibration cell -- the same cells
    `standardize_covariates` fits its mean and standard deviation on -- so "outside"
    means "no training signal anywhere near this value".
    """
    calibration, holdout = trajectory[:, :t_cal], trajectory[:, t_cal:]
    return float(((holdout < calibration.min()) | (holdout > calibration.max())).mean())


# --------------------------------------------------------------------------- #
# 1. The three families behave differently, and the difference is structural.
# --------------------------------------------------------------------------- #


def test_tenure_leaves_the_calibration_support_for_the_whole_cohort(trajectories):
    """`period_since_first_transaction` is the channel that actually breaks.

    It cannot exceed `T_CAL` during calibration (there are only that many periods to
    count) and it keeps counting through the holdout, so every calibration-active
    customer ends the holdout beyond anything the weights were fitted on. On the real
    electronics panel this is 100% of customers and ~89% of holdout cells.
    """
    t_cal = trajectories["__t_cal__"]
    tenure = trajectories["period_since_first_transaction"]

    assert _escape_fraction(tenure, t_cal) > 0.5, (
        "most holdout cells should carry a tenure outside the calibration range"
    )
    calibration_max = tenure[:, :t_cal].max()
    assert (tenure[:, -1] > calibration_max).mean() > 0.75, (
        "by the final holdout period nearly every customer should be past the "
        "calibration tenure maximum"
    )


def test_recency_leaves_the_calibration_support_only_for_customers_who_go_quiet(trajectories):
    """`period_since_last_transaction` escapes too, but partially.

    It is capped by the window length in the same way, yet a customer who buys during
    the holdout resets it to 0, so only the ones quiet for longer than anyone was in
    calibration drift out. That partial escape is why recency is a milder problem than
    tenure, not a harmless one.
    """
    t_cal = trajectories["__t_cal__"]
    escaped = _escape_fraction(trajectories["period_since_last_transaction"], t_cal)
    assert 0.0 < escaped < 0.5, (
        f"recency should escape for some but not most holdout cells, got {escaped:.1%}"
    )


@pytest.mark.parametrize("feature", RUNNING_COUNTERS)
def test_running_counters_stay_in_support_because_a_heavy_buyer_sets_the_maximum(
    feature, trajectories
):
    """The cumulative counters are unbounded in principle and bounded in practice.

    `docs/feature_engineering.md` §4 groups them with tenure. Measured, they behave
    quite differently: the calibration maximum is set by the panel's heaviest buyer and
    sits far above where an ordinary customer ends the holdout, so almost nothing
    escapes (0.04% on the real electronics panel). This test records that difference so
    the doc's grouping is not mistaken for an equal hazard.
    """
    t_cal = trajectories["__t_cal__"]
    escaped = _escape_fraction(trajectories[feature], t_cal)
    assert escaped < 0.10, (
        f"{feature!r} was expected to stay almost entirely within the calibration "
        f"range, got {escaped:.1%} of holdout cells outside it"
    )


@pytest.mark.parametrize("feature", BOUNDED)
def test_bounded_ar_features_never_leave_the_calibration_support(feature, trajectories):
    """A bounded, stationary channel is served by interpolation throughout.

    These carry much of the same recency/frequency information (§4) with no
    extrapolation hazard at all, which is what makes them the documented preference.
    """
    t_cal = trajectories["__t_cal__"]
    escaped = _escape_fraction(trajectories[feature], t_cal)
    assert escaped == 0.0, (
        f"{feature!r} is documented as bounded and stationary, but {escaped:.1%} of "
        f"holdout cells fell outside its calibration range"
    )


# --------------------------------------------------------------------------- #
# 2. Why this is a level shift, not a Monte Carlo artefact.
# --------------------------------------------------------------------------- #


def test_tenure_advances_regardless_of_the_sampled_counts(trajectories):
    """Once a customer is active, tenure is a deterministic ramp.

    This is why the resulting bias is NOT exposure bias and cannot be found by
    comparing a sampled rollout against a teacher-forced one: the channel takes the
    same value on every Monte Carlo path, so every path is displaced in the same
    direction by the same amount. Only the forecast's *level* moves -- which is exactly
    what `bias_percent` measures and what `rmse` on a 97%-zero panel largely hides.
    """
    t_cal = trajectories["__t_cal__"]
    holdout = trajectories["period_since_first_transaction"][:, t_cal:]
    steps = np.diff(holdout, axis=1)
    assert (steps == 1).all(), (
        "tenure should increment by exactly 1 per holdout period for a fully "
        f"calibration-active cohort; saw increments {sorted(np.unique(steps))}"
    )


def test_tenure_drift_is_one_sided(trajectories):
    """The drift is entirely upward, so its effect on the forecast has a fixed sign.

    A channel that drifted symmetrically would push some customers up and others down
    and largely cancel in the aggregate. This one cannot: no holdout cell falls below
    the calibration minimum, so whatever slope the model fitted at the top of the
    tenure range is applied to the whole cohort in the same direction.
    """
    t_cal = trajectories["__t_cal__"]
    tenure = trajectories["period_since_first_transaction"]
    below = (tenure[:, t_cal:] < tenure[:, :t_cal].min()).mean()
    assert below == 0.0, "tenure drifted below its calibration minimum, which it cannot do"


# --------------------------------------------------------------------------- #
# 3. Silence accumulates, so the holdout is scored mostly in the tail.
# --------------------------------------------------------------------------- #


def test_silence_is_right_shifted_over_the_holdout(trajectories):
    """The holdout is spent at longer silences than calibration was.

    `period_since_last_transaction` carries its end-of-calibration value into the
    holdout and keeps counting for anyone who does not buy, and it is one-sided (a
    purchase resets it to 0, nothing pushes it negative). So even on a *stationary*
    panel the holdout's silence distribution sits to the right of calibration's.

    This is what makes the long-silence region the one that decides the aggregate
    forecast: it is a corner of the training data but the bulk of what gets scored.
    """
    t_cal = trajectories["__t_cal__"]
    silence = trajectories["period_since_last_transaction"]
    calibration, holdout = silence[:, :t_cal], silence[:, t_cal:]
    assert holdout.mean() > calibration.mean(), (
        f"holdout mean silence ({holdout.mean():.1f}) should exceed calibration's "
        f"({calibration.mean():.1f})"
    )


def test_the_long_silence_tail_is_over_represented_in_the_holdout(trajectories):
    """Calibration's top silence decile covers far more than a decile of the holdout.

    By construction the 90th percentile of calibration silence leaves 10% of
    calibration cells above it. The holdout puts a much larger share there, so the
    fitted response in that thinly-populated region is applied to a large fraction of
    the cells the forecast is scored on.

    The effect is far stronger on a panel whose purchase rate falls between the two
    windows -- the real electronics panel moves this share from 7.7% to 49.9%, because
    its holdout rate is 1.6x lower than its calibration rate. The synthetic panel here
    is stationary, so it shows only the structural part of the shift; the threshold is
    set to catch that part without pretending to reproduce the non-stationary case.
    """
    t_cal = trajectories["__t_cal__"]
    silence = trajectories["period_since_last_transaction"]
    calibration, holdout = silence[:, :t_cal], silence[:, t_cal:]

    p90 = np.quantile(calibration, 0.90)
    calibration_share = float((calibration > p90).mean())
    holdout_share = float((holdout > p90).mean())

    assert holdout_share > 1.3 * calibration_share, (
        "the holdout should spend a disproportionate share of its cells above the "
        f"calibration 90th-percentile silence; got {holdout_share:.1%} of holdout vs "
        f"{calibration_share:.1%} of calibration"
    )


# --------------------------------------------------------------------------- #
# 4. Standardisation does not close the gap -- the docs say so; assert it.
# --------------------------------------------------------------------------- #


def test_standardisation_does_not_return_a_drifting_channel_to_the_fitted_range(prepared, trajectories):
    """Recentring on calibration moves the origin, not the drift.

    §4: "Standardisation does not rescue an unbounded counter." The rollout applies the
    calibration-fitted `(mean, std)` to every AR value it recomputes, so this checks the
    transform the rollout actually uses, in the units the model actually sees.
    """
    t_cal = trajectories["__t_cal__"]
    tenure = trajectories["period_since_first_transaction"]
    mean, std = prepared["covariate_stats"]["period_since_first_transaction"]

    z_calibration = (tenure[:, :t_cal] - mean) / std
    z_holdout = (tenure[:, t_cal:] - mean) / std

    assert z_holdout.max() > z_calibration.max(), (
        "after standardisation the holdout tenure should still exceed every value seen "
        f"in calibration (calibration max z={z_calibration.max():.2f}, "
        f"holdout max z={z_holdout.max():.2f})"
    )
    # The statistics really are the calibration ones: standardising the calibration
    # window with them returns a centred, unit-scale column.
    assert z_calibration.mean() == pytest.approx(0.0, abs=1e-3)
    assert z_calibration.std() == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# 5. The holdout tensor carries no AR values -- the rollout must supply them.
# --------------------------------------------------------------------------- #


def test_holdout_tensor_ar_columns_are_placeholders(prepared):
    """`prepare_dataset` leaves the holdout's AR columns constant, by design.

    AR features are functions of the target's own past, and the holdout target is never
    revealed to the model, so there is nothing legitimate to put here -- the rollout
    recomputes each column from the SAMPLED history instead
    (`models.monte_carlo_forecasting.simulate_recurrent_path`).

    Pinning it matters because the placeholder is a *plausible* value, not a sentinel:
    a rollout that failed to overwrite these columns would feed "tenure 0, never
    transacted" for every customer at every step and still run without error. This test
    plus the overwrite loop in the simulator are what make that silent failure
    impossible to reach unnoticed.
    """
    seq_cols = list(prepared["seq_cols"])
    holdout = np.asarray(prepared["holdout"])
    for feature in ALL_FEATURES:
        column = holdout[:, :, seq_cols.index(feature)]
        assert np.unique(column).size == 1, (
            f"holdout column {feature!r} was expected to be a single constant "
            f"placeholder, found {np.unique(column).size} distinct values"
        )


def test_ar_feature_names_travel_with_the_dataset(prepared):
    """`data['ar_features']` is what tells the rollout which columns to overwrite.

    `forecast_recurrent` reads it with `data.get("ar_features", [])`, so a dataset that
    dropped the key would roll out with the placeholders above still in place and report
    a forecast rather than an error.
    """
    assert list(prepared["ar_features"]) == list(ALL_FEATURES)
    stats = prepared["covariate_stats"]
    missing = [f for f in ALL_FEATURES if f not in stats]
    assert not missing, (
        f"every AR feature needs calibration (mean, std) for the rollout to restandardise "
        f"the values it recomputes; missing: {missing}"
    )
