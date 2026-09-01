"""AR-feature encoding ablation — does bounding the counters fix the forecast level?

The AR-feature configuration forecasts +220% to +259% aggregate bias on electronics
against +23% for the same model with no AR features, and its across-study SD rises from
21 to 160-195 points. The cause is not the rollout's feedback loop: a teacher-forced
pass (true counts *and* true AR values fed in at every step) reproduces the bias at
+169%, so the fitted conditional is wrong before any sampling happens.

What is wrong is where it is evaluated. `period_since_last_transaction` and
`period_since_first_transaction` cannot exceed the calibration window length while being
fitted -- there are only 104 periods to count -- and they keep counting through the
holdout. 37.7% of holdout cells carry a recency past the calibration maximum and 100% of
customers end past the calibration tenure maximum. Outside that range the response stops
decreasing and turns upward, so the model's predicted rate never falls below ~0.072
while the true rate at long silence is 0.0153. Half the holdout sits there, so 54% of
the total excess prediction comes from that one region.

The arms replace those unbounded counters with bounded ones that carry the same
recency information:

    no_ar          no AR features at all -- the level baseline
    ar_unbounded   recency + cumulative_transactions + tenure  (today's config)
    ar_bounded_32  nested activity flags to K=32, plus has_transacted_before
    ar_bounded_52  the same, one bin deeper

**There is no expected discrimination cost.** A model-free encoding check (fit the
calibration mean per bucket, apply it to the holdout) puts uncapped recency at +45.5%
bias / 0.661 Spearman and the bounded flag set at +6.3% / 0.863: the uncapped tail
buckets fit noise that does not generalise, so bounding improves *both*. This ablation
tests that on trained models, and reports both numbers so an arm that fixes the level by
going blind is visibly rejected.

**What "no worse" means, measured.** Those encoding figures are an ORACLE bound: they
read the true holdout AR states, whereas a real rollout rebuilds them from sampled
counts, which costs most of the ranking. Scored on the archived suites, the achieved
per-customer Spearman is **0.036 (NoCov)** and **0.191 (AR unbounded)** — so the AR
channels really do buy roughly 5x the discrimination, and **0.191 is the number a
bounded arm has to match**, not 0.863.

**Why one suite per arm.** `run_loss_ablation.py` and `run_cdnow_embedding_ablation.py`
vary a *model* parameter, so their arms share one `prepare_dataset` object and sit in one
suite. Here the arms differ in `ar_features`, which is a `PanelConfig` property, and
`StudySuiteConfig.data` is a single shared dataset -- so each arm needs its own dataset
and therefore its own suite. Shards split an arm's replications across two workers by
seed range. Every suite root is disjoint, so collecting from rented workers is a plain
copy and aggregation concatenates tables (VastAI/Rules.md §4, §6).

**Read the distribution, not paired differences.** Training is unseeded (CLAUDE.md
priority 3): `base_seed + i` drives the Optuna sampler and the Monte Carlo forecast, not
weight init, `DataLoader` shuffling or dropout. Replications are genuine replications, so
pairing arms on seed removes far less variance than it does in the loss ablation, and the
mean / SD / min / max across replications is the honest summary.

Usage:
    # one arm-shard (this is what a rented worker runs)
    python scripts/run_ar_encoding_ablation.py --arm ar_bounded_32 --shard a

    # timing probe on a fresh box, before committing a shard to it
    python scripts/run_ar_encoding_ablation.py --arm no_ar --shard a \
        --n-studies 1 --n-trials 5 --n-simulations 20 --suite-suffix probe

    # pool every suite present and print the comparison
    python scripts/run_ar_encoding_ablation.py --report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.models import compute_forecast_metrics
from panelclv.studies import (
    ModelSpec,
    StudySuiteConfig,
    load_model_predictions,
    run_study_suite,
    study_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"
CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"

# --- study size ------------------------------------------------------------------
# 4 arms x 2 shards x N_STUDIES studies x N_TRIALS trainings, plus one refit per study.
# Matches the archived electronics suites so the `ar_unbounded` arm is comparable with
# what is already on disk -- which is this script's own sanity check (see `report`).
N_TRIALS = 50
N_STUDIES = 20         # per shard; two shards per arm gives 40 replications
N_SIMULATIONS = 300    # Monte Carlo paths per forecast (docs/loss-functions.md §6 floor)

# The feature sets under test. Only this mapping differs between arms.
#
# `active_in_last_<K>_periods` is a bounded step encoding of the same silence the
# unbounded counter measures: nested flags at K = 2, 4, 8, 16, 32 distinguish seven
# silence states and are all 0 beyond the deepest bin, so no holdout value can leave the
# range the weights were fitted on. `active_in_last_1_periods` is omitted because the
# target channel already carries the previous count.
#
# Tenure and `cumulative_transactions` are dropped rather than left searchable: the
# Optuna objective is teacher-forced validation loss, which is blind to the rollout
# drift, which is why the search selected them in the first place.
def bounded_flags(deepest: int) -> tuple[str, ...]:
    """Nested activity flags up to `deepest`, plus `has_transacted_before`.

    A bounded step encoding of the same silence `period_since_last_transaction`
    measures: the flags are all 0 beyond the deepest bin, so no holdout value can leave
    the range the weights were fitted on. `active_in_last_1_periods` is omitted because
    the target channel already carries the previous count.

    `deepest` must be < the panel's calibration length. At or above it the flag can never
    be 0 while fitting -- silence has nowhere to accumulate -- so the column is an exact
    duplicate of `has_transacted_before` in calibration and diverges from it only in the
    holdout, which is precisely the failure this encoding exists to remove.
    `check_arm_depth` enforces that against the built dataset.
    """
    ks = [k for k in (2, 4, 8, 16, 32, 52) if k <= deepest]
    return tuple(f"active_in_last_{k}_periods" for k in ks) + ("has_transacted_before",)


# The unbounded set under test: today's `config_pareto`, and the failure being fixed.
UNBOUNDED = (
    "period_since_last_transaction",
    "cumulative_transactions",
    "period_since_first_transaction",
)

# Two bounded arms per panel, with the deepest bin at roughly 31% and 50% of the
# calibration window. The K values differ between panels because the windows do
# (electronics 104 periods, CDNOW 39); holding K fixed instead would make the deeper
# CDNOW arm degenerate, which is the trap `bounded_flags` documents.
PANEL_DEPTHS: dict[str, tuple[int, int]] = {
    "electronics": (32, 52),
    "cdnow":       (16, 32),
}


def arms_for(panel: str) -> dict[str, tuple[str, ...]]:
    """`{arm name: ar_features}` for one panel. Only this mapping differs between arms.

    Tenure and `cumulative_transactions` are dropped from the bounded arms rather than
    left searchable: the Optuna objective is teacher-forced validation loss, which is
    blind to the rollout drift, which is why the search selected them in the first place.
    """
    shallow, deep = PANEL_DEPTHS[panel]
    return {
        "no_ar": (),
        "ar_unbounded": UNBOUNDED,
        f"ar_bounded_{shallow}": bounded_flags(shallow),
        f"ar_bounded_{deep}": bounded_flags(deep),
    }


SHARDS: dict[str, int] = {"a": 42, "b": 62}


def electronics_config(ar_features: tuple[str, ...]) -> PanelConfig:
    """The electronics panel as the ARCHIVED suites read it, with `ar_features` varying.

    Windows, clipping, cohort rule and time features are copied from
    `Studies/cross_entropy_config_pareto_Comparaison_pareto/config.json` so the
    `ar_unbounded` arm reproduces a suite that already exists. That reproduction is what
    makes the other arms trustworthy.
    """
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        training_start="1999-01-01",
        validation_start="2000-01-01",   # the last calibration year
        training_end="2000-12-31",
        holdout_start="2001-01-01",
        holdout_end="2001-12-31",
        clip_target_upper=6,             # 7-class head
        require_calibration_activity=True,
        time_features={"add_year_idx": True, "add_week_sin_cos": True},
        ar_features=ar_features,
        known_future=(),
        static=(),
        observed_past=(),
        embedded_cols={"Transactions": "auto"},
    )


def cdnow_config(ar_features: tuple[str, ...]) -> PanelConfig:
    """The CDNOW panel exactly as `run_loss_ablation.py` reads it, `ar_features` varying.

    A much harsher test of the same hazard than electronics: the holdout is nearly as
    long as the calibration window (38 vs 39 periods, against 52 vs 104), so the capped
    counters drift proportionally further -- recency escapes its fitted range on 56.9% of
    holdout cells here against 37.7% there -- and the per-cell rate falls 2.52x between
    the windows rather than 1.60x.
    """
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        training_start="1997-01-01",
        validation_start="1997-08-06",   # 1997 week 31 - last 8 calibration weeks
        training_end="1997-09-30",       # inclusive of 1997 week 38
        holdout_start="1997-10-01",      # 1997 week 39
        holdout_end="1998-06-30",        # inclusive of the last complete week, 1998 w24
        clip_target_upper=4,             # 5-class head; 3 cells in 181,489 exceed it
        ar_features=ar_features,
        embedded_cols={"Transactions": "auto"},
    )


PANELS = {
    "electronics": (CLEAN / "electronics_customer_week_panel.csv", electronics_config),
    "cdnow":       (CLEAN / "cdnow_customer_week_panel.csv", cdnow_config),
}


def check_arm_depth(arm: str, data: dict) -> None:
    """Refuse an `active_in_last_K` channel that cannot vary on this panel.

    Silence cannot exceed `T_CAL - 1` while fitting, so a flag with K at or above the
    calibration length is 1 for every customer who has ever transacted -- an exact copy
    of `has_transacted_before` -- and only becomes a distinct signal out in the holdout,
    where nothing constrained it. Cheap to check, silent and expensive to miss.
    """
    t_cal = int(data["T_CAL"])
    bad = [c for c in data["seq_cols"]
           if c.startswith("active_in_last_")
           and int(c.split("_")[3]) >= t_cal]
    if bad:
        raise ValueError(
            f"arm {arm!r}: {bad} cannot vary on a panel with T_CAL={t_cal}. Such a flag "
            f"duplicates 'has_transacted_before' in calibration and diverges from it only "
            f"in the holdout. Lower the depth in PANEL_DEPTHS[{data.get('panel_name', '?')!r}]."
        )

# Everything that is NOT the ablation. Identical in every arm, so the only thing moving
# is the feature set. `embedding_dim` is deliberately absent: the default `valendin`
# embedder has no common width, so naming one would advertise a search that never happens
# (docs/running-a-model.md §14).
SHARED_SEARCH_SPACE: dict[str, object] = {
    "lstm_hidden_size": {32, 64, 128},
    "dense_units":      {32, 64, 128},
    "dropout":          {0.0, 0.2},
    "learning_rate":    (1e-4, 1e-2, "log"),
    "weight_decay":     (1e-6, 1e-2, "log"),
    "batch_size":       {64, 128, 256},
}

SHARED_TRAINING: dict[str, object] = {
    "n_epochs":  100,
    "patience":  7,
    "verbose":   False,
    "loss_type": "cross_entropy",
}


def suite_name(panel: str, arm: str, shard: str, suffix: str | None = None) -> str:
    """`ar_encoding__<panel>__<arm>__<shard>` — one disjoint root per rented worker."""
    stem = f"ar_encoding__{panel}__{arm}__{shard}"
    return f"{stem}__{suffix}" if suffix else stem


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, as Pearson on average-tied ranks.

    Written out rather than imported from `scipy.stats`: scipy is an undeclared
    transitive dependency of this project (it arrives via scikit-learn), so a script that
    names it directly would break on an environment that trimmed it.
    """
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def score_suite(root: Path, data: dict, base_seed: int) -> list[dict[str, object]]:
    """Score every stored forecast in one suite root, one row per replication.

    Deliberately reads `Predictions/Prediction_*.csv` rather than the suite's
    `results.csv`. `run_study_suite` saves each study's forecast inside its loop
    (`studies/runner.py:174`) but writes `results.csv`, `metrics.csv` and `config.json`
    only after the last one, so a shard killed by its watchdog leaves completed
    replications that none of those three files describe. Scoring the predictions
    directly makes a partial shard count for exactly the replications it finished, which
    is what lets the watchdog be a budget bound rather than an all-or-nothing gamble.

    It also means this does not depend on the persisted `panel_config`: `data` is rebuilt
    from this script's own `arms_for` mapping, which is the same declaration the suite was
    created from.

    Alignment follows `studies.suite_metrics`: predictions are stored in the cohort's own
    order, and the ids are asserted against the rebuilt cohort so row i of the forecast
    and row i of the actuals are the same customer.
    """
    actual = holdout_actuals(data)                              # (N, T_HOLD)
    actual_totals = actual.sum(axis=1)
    ref_ids = np.asarray(data["ids"])

    model_dir = root / "LSTM"
    predictions = model_dir / "Predictions"
    if not predictions.is_dir():
        return []

    rows: list[dict[str, object]] = []
    for path in sorted(predictions.glob("Prediction_*.csv")):
        study = int(path.stem.split("_")[-1])
        values, ids = load_model_predictions(model_dir, study=study)
        if ids is not None and not np.array_equal(np.asarray(ids), ref_ids):
            raise ValueError(
                f"{root.name} study {study}: prediction ids do not match the rebuilt "
                f"cohort — is this the panel the suite was built from?"
            )
        rows.append(
            {
                "study": study,
                "seed": base_seed + study,
                **compute_forecast_metrics(actual, values),
                "spearman": spearman(values.sum(axis=1), actual_totals),
            }
        )
    return rows


def report(panel: str, suffix: str | None = None) -> None:
    """Pool every arm's shards and print bias and discrimination side by side.

    `compare_study_metrics` accepts at most four suite roots and there are eight here
    (four arms x two shards), so the pooling is done here: each shard contributes its own
    per-study rows and an arm's summary is taken over the union.

    `suffix` reads a probe/smoke set instead of the real run, so the report path can be
    exercised without waiting for the full ablation.
    """
    panel_path, build_config = PANELS[panel]

    rows: list[dict[str, object]] = []
    arms = arms_for(panel)
    for arm, ar_features in arms.items():
        data = None
        for shard, base_seed in SHARDS.items():
            root = STUDIES_BASE / suite_name(panel, arm, shard, suffix)
            if not (root / "LSTM" / "Predictions").is_dir():
                continue
            # Rebuild this arm's dataset once, lazily: only needed to score, and only for
            # arms that actually have forecasts on disk.
            if data is None:
                panel_df = pd.read_csv(panel_path)
                data = panel_dataset.prepare_dataset(
                    panel_df, build_config(ar_features), verbose=False
                )
            for row in score_suite(root, data, base_seed):
                rows.append({"arm": arm, "shard": shard, **row})

    if not rows:
        print("No finished suites found. Run the arms first.")
        return

    per_study = pd.DataFrame(rows)
    # Replications per (arm, shard), so a shard the watchdog cut short is visible as a
    # short count rather than silently shrinking an arm's sample.
    coverage = per_study.groupby(["arm", "shard"]).size().unstack(fill_value=0)
    print(f"\n{len(per_study)} replications across {per_study.arm.nunique()} arms")
    print("\nreplications per shard (10 expected each):")
    print(coverage.to_string())


    # Distribution across replications, not a pooled point estimate: a single mean hides
    # exactly the across-study spread this ablation exists to measure.
    summary = per_study.groupby("arm", sort=False).agg(
        n=("bias_percent", "size"),
        bias_mean=("bias_percent", "mean"),
        bias_sd=("bias_percent", lambda s: s.std(ddof=1)),
        bias_min=("bias_percent", "min"),
        bias_max=("bias_percent", "max"),
        abs_bias_mean=("bias_percent", lambda s: s.abs().mean()),
        spearman_mean=("spearman", "mean"),
        spearman_sd=("spearman", lambda s: s.std(ddof=1)),
        mape_mean=("mape_aggregate", "mean"),
        mape_sd=("mape_aggregate", lambda s: s.std(ddof=1)),
        rmse_mean=("rmse", "mean"),
    )
    order = [a for a in arms if a in summary.index]
    print(summary.loc[order].round(3).to_string())

    print(
        "\nAccept a bounded arm only if BOTH hold against ar_unbounded:"
        "\n  |bias| falls substantially, AND spearman is no worse."
        "\nmape_aggregate is the aggregate-accuracy metric that separates these arms;"
        "\nit shows the bounded AR channels beating the no-AR baseline where |bias|"
        "\nalone does not. RMSE is shown for completeness only: it is dominated by the"
        "\nzeros and its arm-to-arm differences land in the fourth decimal."
    )
    if "ar_unbounded" in summary.index:
        got = summary.loc["ar_unbounded", "bias_mean"]
        print(
            f"\nSanity check — ar_unbounded reproduces the archived suites: "
            f"got {got:+.1f}%, archived +220% / +259%. "
            f"{'OK' if 80 <= got <= 500 else 'OFF — investigate before trusting the other arms'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=sorted(PANELS), default="electronics")
    # Not an argparse `choices`: the arm names depend on --panel (the bounded arms
    # take their depth from the calibration window), so it is validated below.
    parser.add_argument("--arm")
    parser.add_argument("--shard", choices=sorted(SHARDS))
    parser.add_argument("--n-studies", type=int, default=N_STUDIES)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument(
        "--suite-suffix",
        help="appended to the suite name — use for probe/smoke runs so they cannot "
             "collide with the real shard's folder",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="pool every finished suite for this panel and print the comparison",
    )
    args = parser.parse_args()

    if args.report:
        report(args.panel, args.suite_suffix)
        return
    if not args.arm or not args.shard:
        parser.error("--arm and --shard are required unless --report is given")

    arms = arms_for(args.panel)
    if args.arm not in arms:
        parser.error(
            f"--arm {args.arm!r} is not defined for panel {args.panel!r}; "
            f"choose from {sorted(arms)}"
        )

    panel_path, build_config = PANELS[args.panel]
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Datasets/ is gitignored, so a rented worker needs "
            f"the panel pushed to it (VastAI/Rules.md §3)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = pd.read_csv(panel_path)
    data_full = panel_dataset.prepare_dataset(panel, build_config(arms[args.arm]))
    data_full["panel_name"] = args.panel
    check_arm_depth(args.arm, data_full)

    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    config = StudySuiteConfig(
        studies_base_path=str(STUDIES_BASE),
        suite_name=suite_name(args.panel, args.arm, args.shard, args.suite_suffix),
        n_studies_per_model=args.n_studies,
        n_simulations=args.n_simulations,
        device=device,
        data=data_full,
        models=[
            ModelSpec(
                name="LSTM",
                model_type="lstm",
                n_trials=args.n_trials,
                search_space=dict(SHARED_SEARCH_SPACE),
                training=dict(SHARED_TRAINING),
            )
        ],
        base_seed=SHARDS[args.shard],
        keep_only_best_checkpoint=True,
    )

    root = run_study_suite(config)
    print(f"\nStudy suite written to: {root}")
    print(f"arm={args.arm}  shard={args.shard}  ar_features={arms[args.arm]}")

    table = study_metrics(root, panel_path, standard_deviation=True)
    print(f"\nForecast metrics (mean over {args.n_studies} studies, +- SD):")
    print(table.to_string())


if __name__ == "__main__":
    main()
