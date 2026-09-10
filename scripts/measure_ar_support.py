"""How far outside its fitted range does each AR encoding put the holdout?

`docs/feature_engineering.md` §4 ranks AR features by their **support-escape
fraction**: the share of holdout cells whose value falls outside the `[min, max]`
the channel took anywhere in calibration. That number is the right diagnostic for
choosing *whether* to carry a counter, and the wrong one for choosing *how to
encode* it, because it is invariant to any order-preserving transform --
`log1p(recency)` escapes on exactly the cells `recency` does, to the last cell.

What separates two encodings of the same counter is **distance**. Theorem 1 of Xu
et al. (2021, ICLR, arXiv:2009.11848) says a ReLU network's prediction converges to
a linear function of its input away from the training region, so the error it makes
out there grows with how far out there the input is. `standardize_covariates` fits
its mean and standard deviation on calibration, so the units the model actually sees
are calibration z-scores, and the quantity that matters is

    z_beyond = (holdout max - calibration max) / calibration sd

reported both as that worst single cell and as the average excess over the cells
that actually escape, since a channel can escape rarely but very far (CDNOW's
`cumulative_transactions` escapes on 0.18% of cells and by 9.2 z at the extreme).
A negative worst-case figure means the holdout stayed inside the fitted range
throughout, which is what a genuinely stationary channel looks like.

This script reports those numbers, for the raw counters and for each re-encoding, on
whichever panels are available. It reads only the panel and the config -- no trained
checkpoint -- so it measures the input-side precondition for the failure rather than
a forecast error, and runs in seconds.

The AR states are built from the TRUE counts across both windows. During a real
rollout the holdout half is rebuilt from *sampled* counts, which can only add noise,
so every distance here is a lower bound on what the rollout feeds.

`--linearity` answers the companion question: which coordinate makes the response
a straight line? Pareto/NBD's lifetime is exponential with a gamma-mixed rate, so
survival is Lomax and `log S(t) = -s log(1 + t/beta)` is linear in `log(1 + gap)`,
not in the gap. Under Hypothesis 1 of the same paper -- encode the non-linearity in
the representation so the network only has to learn a linear step -- that makes
log1p the aligned coordinate for recency. The report fits the empirical log-hazard
in both coordinates and extrapolates each past the calibration ceiling, so the
choice rests on a measurement rather than on the derivation alone.

Usage (src-layout: the package is importable only once it is installed, or with
`PYTHONPATH=src`, which is how the test suite finds it too):
    PYTHONPATH=src python scripts/measure_ar_support.py              # support table
    PYTHONPATH=src python scripts/measure_ar_support.py --linearity  # + coordinate check
    PYTHONPATH=src python scripts/measure_ar_support.py --panel cdnow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.ar_features import compute_ar_feature_columns
from panelclv.data_preparation.target_channel import holdout_actuals

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"

# The encodings compared, grouped by what they do to the underlying counter. The two
# raw clocks are the baseline the rest are trying to beat.
ENCODINGS: tuple[str, ...] = (
    "period_since_last_transaction",
    "log_period_since_last_transaction",
    "saturating_recency_8_periods",
    "saturating_recency_26_periods",
    "recency_over_tenure",
    "active_in_last_8_periods",
    "period_since_first_transaction",
    "log_period_since_first_transaction",
    "saturating_tenure_8_periods",
    "saturating_tenure_26_periods",
    "cumulative_transactions",
    "transaction_rate",
    "has_transacted_before",
)


def electronics_config() -> PanelConfig:
    """Windows copied from `run_ar_encoding_ablation.py`, so the two agree."""
    return PanelConfig(
        id_col="Id", target_col="Transactions", frequency="weekly",
        time_cols=("year", "week"),
        training_start="1999-01-01", validation_start="2000-01-01",
        training_end="2000-12-31",
        holdout_start="2001-01-01", holdout_end="2001-12-31",
        clip_target_upper=6, require_calibration_activity=True,
        time_features={"add_year_idx": True, "add_week_sin_cos": True},
        ar_features=(), embedded_cols={"Transactions": "auto"},
    )


def cdnow_config() -> PanelConfig:
    return PanelConfig(
        id_col="Id", target_col="Transactions", frequency="weekly",
        time_cols=("year", "week"),
        training_start="1997-01-01", validation_start="1997-08-06",
        training_end="1997-09-30",
        holdout_start="1997-10-01", holdout_end="1998-06-30",
        clip_target_upper=4, ar_features=(), embedded_cols={"Transactions": "auto"},
    )


PANELS = {
    "electronics": (CLEAN / "electronics_customer_week_panel.csv", electronics_config),
    "cdnow": (CLEAN / "cdnow_customer_week_panel.csv", cdnow_config),
}


def counts_for(panel: str) -> tuple[np.ndarray, int]:
    """`(N, T_CAL + T_HOLD)` true counts, and the calibration length."""
    path, config = PANELS[panel]
    prepared = panel_dataset.prepare_dataset(pd.read_csv(path), config(), verbose=False)
    target_idx = int(prepared["target_idx"])
    calibration = np.asarray(prepared["calibration"])[:, :, target_idx].astype(np.int64)
    holdout = holdout_actuals(prepared).astype(np.int64)
    return np.concatenate([calibration, holdout], axis=1), calibration.shape[1]


def support_table(panel: str) -> pd.DataFrame:
    """Escape fraction and standardised distance for every encoding on one panel."""
    counts, t_cal = counts_for(panel)
    columns = compute_ar_feature_columns(counts, ENCODINGS)
    rows = []
    for name in ENCODINGS:
        calibration, holdout = columns[name][:, :t_cal], columns[name][:, t_cal:]
        lo, hi, sd = calibration.min(), calibration.max(), calibration.std()
        sd = sd if sd > 0 else 1.0
        # How far each escaping cell sits outside, on whichever side it left by.
        excess = np.maximum(holdout - hi, lo - holdout)
        escaping = excess > 0
        rows.append({
            "encoding": name,
            "escape %": float(escaping.mean()) * 100,
            "z worst": float((holdout.max() - hi) / sd),
            "z avg": float(excess[escaping].mean() / sd) if escaping.any() else 0.0,
            "cal range": f"[{lo:.3g}, {hi:.3g}]",
        })
    return pd.DataFrame(rows)


def linearity_report(panel: str) -> None:
    """Which recency coordinate straightens the log-hazard, and what each predicts.

    Bins every (customer, period) cell by the silence preceding it, takes the
    empirical probability of a transaction in the next period, and fits the log of
    that against the gap and against log(1 + gap). Then extrapolates each fit past
    the calibration ceiling and compares both against what the holdout actually did
    out there -- the whole question, since that region is where the forecast is
    decided and where no training cell exists.
    """
    counts, t_cal = counts_for(panel)
    states = compute_ar_feature_columns(
        counts, ("period_since_last_transaction", "has_transacted_before")
    )
    gap = states["period_since_last_transaction"][:, :-1]
    active = states["has_transacted_before"][:, :-1] == 1
    transacted_next = (counts[:, 1:] > 0).astype(float)
    in_calibration = np.zeros_like(gap, dtype=bool)
    in_calibration[:, : t_cal - 1] = True

    def hazard_by_gap(mask: np.ndarray) -> pd.DataFrame:
        """Empirical P(transaction next period) per silence value, cells >= 30."""
        frame = pd.DataFrame({"gap": gap[mask], "y": transacted_next[mask]})
        table = frame.groupby("gap").agg(p=("y", "mean"), n=("y", "size"))
        return table[(table.n >= 30) & (table.p > 0)]

    def weighted_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
        """Weighted least squares of y on x; returns (intercept, slope, R^2)."""
        weight = w / w.sum()
        xm, ym = (weight * x).sum(), (weight * y).sum()
        cov = (weight * (x - xm) * (y - ym)).sum()
        vx = (weight * (x - xm) ** 2).sum()
        vy = (weight * (y - ym) ** 2).sum()
        slope = cov / vx
        return ym - slope * xm, slope, cov**2 / (vx * vy)

    fitted = hazard_by_gap(active & in_calibration)
    gaps = fitted.index.values.astype(float)
    log_hazard = np.log(fitted.p.values)
    weights = fitted.n.values.astype(float)

    print(f"\n  coordinate check (calibration cells only, then extrapolated):")
    ceiling = gap[active & in_calibration].max()
    tail = active & ~in_calibration & (gap > ceiling)
    mean_tail_gap = gap[tail].mean() if tail.any() else float("nan")
    truth = transacted_next[tail].mean() if tail.any() else float("nan")
    for label, x, x_tail in (
        ("gap", gaps, mean_tail_gap),
        ("log(1 + gap)", np.log1p(gaps), np.log1p(mean_tail_gap)),
    ):
        intercept, slope, r2 = weighted_fit(x, log_hazard, weights)
        print(f"    log-hazard vs {label:13s} R^2={r2:.3f}  slope={slope:+.3f}  "
              f"-> hazard at gap {mean_tail_gap:.0f}: {np.exp(intercept + slope * x_tail):.4f}")
    print(f"    the holdout's own rate beyond a gap of {ceiling:.0f}:            {truth:.4f}")
    print("    (Pareto/NBD predicts a straight line in log(1 + gap), slope = -s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=sorted(PANELS), action="append",
                        help="panel to measure; repeatable, defaults to all present")
    parser.add_argument("--linearity", action="store_true",
                        help="also fit the log-hazard in both recency coordinates")
    args = parser.parse_args()

    for panel in args.panel or sorted(PANELS):
        path, _ = PANELS[panel]
        if not path.exists():
            print(f"\n===== {panel}: {path} not present, skipped =====")
            continue
        counts, t_cal = counts_for(panel)
        print(f"\n===== {panel}  (N={counts.shape[0]}, T_CAL={t_cal}, "
              f"T_HOLD={counts.shape[1] - t_cal}) =====")
        table = support_table(panel)
        print(table.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        if args.linearity:
            linearity_report(panel)


if __name__ == "__main__":
    main()
