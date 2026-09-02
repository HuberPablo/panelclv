"""Cluster ablation — does the same history help more as a category than as a counter?

`run_ar_encoding_ablation.py` established two facts about the (t_x, x, T) triple on the
electronics panel, and this ablation sits exactly between them:

* handing the neural model those three continuous AR channels raises per-customer
  discrimination roughly 5x — achieved Spearman **0.191** against **0.036** with no
  covariates at all. The information is real and the model can use it.
* it also wrecks the level. Aggregate bias goes from **+23%** (no AR) to **+220% / +259%**,
  with the across-study SD rising from 21 to 160-195. The cause is diagnosed there:
  `period_since_last_transaction` and `period_since_first_transaction` cannot exceed the
  calibration window length while being fitted, they keep counting through the holdout,
  and 37.7% of holdout cells land past the calibration maximum where the fitted response
  turns the wrong way.

A **behavioural cluster** carries the same information in a form that **cannot escape its
support**. Customers are partitioned by that identical (t_x, x, T) triple at the last
calibration period, and the model reads the group index as a learned embedding. A cluster
label takes exactly the same K values in the holdout as in calibration, by construction —
there is no counter to run off the end (`docs/feature_engineering.md` §4).

So the question is sharp and falsifiable:

    Does a cluster embedding buy the AR discrimination without the AR level failure?

Three outcomes, all informative:

* **Discrimination up, bias flat.** The counters' problem was their encoding, not their
  content, and a bounded categorical summary is the way to hand a sequence model
  per-customer state. The strongest result available here.
* **Discrimination flat.** The model cannot use the triple in categorical form — which,
  since it demonstrably can use it in continuous form, is a fact about the embedding path
  rather than about the information.
* **Discrimination up AND bias up.** The level failure was never about support escape;
  the diagnosis in the AR ablation needs revisiting.

Note what the second and third outcomes cost: nothing is wasted, because the arms are
constructed so each answer rules something out.

## The arms

    no_cluster          nothing target-derived — the level baseline (= `no_ar`)
    cluster_4           k-means into 4 clusters on (t_x, x, T)
    cluster_8           ... 8
    cluster_16          ... 16
    ar_unbounded        the three continuous channels — reproduces the archived suites,
                        and is this script's sanity check
    ar_plus_cluster_8   both, to see whether the category adds anything the channels
                        do not already carry

K is an ARM, never an Optuna knob. If the search tuned K per trial the arms would stop
being comparable and no difference could be attributed. Three values are declared because
K is the sensitive knob of this design — the same criticism `docs/p-slstm.md` §11 makes of
never sweeping P-sLSTM's patch size, which it would be careless to repeat here.

You are not obliged to run all six. Each invocation runs ONE arm-shard, and `--report`
pools whatever finished, so the table declares the space and the budget decides how much
of it gets explored. `no_cluster`, `cluster_8` and `ar_unbounded` are the minimum that
answers the question.

Both panels `run_ar_encoding_ablation.py` uses are registered, and `--panel` selects one.
CDNOW is the sharper test: its holdout is nearly as long as its calibration window, so the
unbounded counters drift further there (recency escapes on 56.9% of holdout cells against
37.7% on electronics) and a bounded encoding has more to fix.

## Why one suite per arm

The arms differ in `PanelConfig` (`cluster_features` / `ar_features`), and
`StudySuiteConfig.data` is a single shared dataset — so each arm needs its own
`prepare_dataset` output and therefore its own suite. Same constraint, same reason, as
`run_ar_encoding_ablation.py`; `run_loss_ablation.py` and
`run_cdnow_embedding_ablation.py` vary a *model* parameter instead and so share one.

Shards split an arm's replications across two workers by disjoint seed range. Every suite
root is disjoint, so collecting from rented workers is a plain copy.

## Two things that make this a clean comparison

**Nothing can drop the cluster column.** `ModelSpec` has no `removable_features` field and
neither `studies` nor `trials` passes one, so the Optuna covariate-subset search is
unreachable from a study suite. Every trial of a cluster arm really does carry the label.

**Cluster labels are deterministic.** k-means runs with a fixed `random_state`
(`data_preparation.cluster_features`), so an arm's replications differ only in the things
`base_seed + i` drives — the Optuna sampler and the Monte Carlo forecast — plus the
unseeded training noise every replication has. Cluster assignment is not a hidden extra
variance component.

**Read the distribution, not paired differences.** Training is unseeded (`CLAUDE.md`
priority 3): weight init, `DataLoader` shuffling and dropout draw on a global torch RNG the
package never sets. Replications are genuine replications, so mean / SD / min / max across
them is the honest summary and a gap smaller than the spread is not a result.

Usage:
    # one arm-shard (this is what a rented worker runs)
    python scripts/run_cluster_ablation.py --arm cluster_8 --shard a

    # timing probe on a fresh box, before committing a shard to it
    python scripts/run_cluster_ablation.py --arm cluster_8 --shard a \
        --n-studies 1 --n-trials 5 --n-simulations 20 --suite-suffix probe

    # pool every suite present and print the comparison
    python scripts/run_cluster_ablation.py --report
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
# Per arm-shard: N_STUDIES studies x N_TRIALS trainings, plus one refit per study.
# Matched to `run_ar_encoding_ablation.py` so the `ar_unbounded` arm here and the one
# there are the same experiment, which is what makes the sanity check meaningful.
N_TRIALS = 50
N_STUDIES = 20         # per shard; two shards per arm gives 40 replications
N_SIMULATIONS = 300    # Monte Carlo paths per forecast (docs/loss-functions.md §6 floor)

# The (t_x, x, T) triple as continuous channels. Exactly what a cluster arm partitions
# on, which is the whole point: the arms differ in ENCODING, not in information.
AR_TRIPLE = (
    "period_since_last_transaction",     # t_x — recency
    "cumulative_transactions",           # x  — active periods, NOT cumulative_count
    "period_since_first_transaction",    # T  — observation age
)


@dataclass(frozen=True)
class Arm:
    """One arm: the two `PanelConfig` fields that vary, and nothing else."""

    ar_features: tuple[str, ...] = ()
    cluster_features: tuple[str, ...] = ()


ARMS: dict[str, Arm] = {
    "no_cluster":        Arm(),
    "cluster_4":         Arm(cluster_features=("kmeans_4",)),
    "cluster_8":         Arm(cluster_features=("kmeans_8",)),
    "cluster_16":        Arm(cluster_features=("kmeans_16",)),
    "ar_unbounded":      Arm(ar_features=AR_TRIPLE),
    "ar_plus_cluster_8": Arm(ar_features=AR_TRIPLE, cluster_features=("kmeans_8",)),
}

# The three that answer the question on their own, if the budget is tight.
MINIMUM_ARMS = ("no_cluster", "cluster_8", "ar_unbounded")

# Shard -> base seed. Study i of a shard draws `base_seed + i`, so at N_STUDIES = 20
# shard "a" covers seeds 43-62 and shard "b" covers 63-82: disjoint, and an arm's 40
# replications are 40 distinct seeds rather than two runs of the same twenty. The gap
# between the base seeds must stay >= N_STUDIES or half the replications become
# duplicates in silence.
SHARDS: dict[str, int] = {"a": 42, "b": 62}


def electronics_config(arm: Arm) -> PanelConfig:
    """The electronics panel as the ARCHIVED suites read it, with the arm varying.

    Windows, clipping, cohort rule and time features are copied from
    `run_ar_encoding_ablation.electronics_config`, which took them from
    `Studies/cross_entropy_config_pareto_Comparaison_pareto/config.json`. Holding them
    identical is what lets the `ar_unbounded` arm reproduce a suite that already exists,
    and that reproduction is what makes the cluster arms trustworthy.
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
        ar_features=arm.ar_features,
        cluster_features=arm.cluster_features,
        known_future=(),
        static=(),
        observed_past=(),
        embedded_cols={"Transactions": "auto"},
        # NOTE: the cluster column is NOT listed above. `prepare_dataset` embeds it
        # automatically at cardinality K — a group index is categorical by definition and
        # standardising it would impose an ordering the labels do not have.
    )


def cdnow_config(arm: Arm) -> PanelConfig:
    """The CDNOW panel exactly as `run_ar_encoding_ablation.cdnow_config` reads it.

    The harsher test of the same hazard, and therefore the sharper test of this
    ablation's claim. CDNOW's holdout is nearly as long as its calibration window (38 vs
    39 periods, against 52 vs 104 on electronics), so the unbounded counters drift
    proportionally further — recency escapes its fitted range on 56.9% of holdout cells
    here against 37.7% there. If a bounded categorical encoding fixes the level anywhere,
    it should show most clearly here.
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
        ar_features=arm.ar_features,
        cluster_features=arm.cluster_features,
        embedded_cols={"Transactions": "auto"},
    )


PANELS = {
    "electronics": (CLEAN / "electronics_customer_week_panel.csv", electronics_config),
    "cdnow":       (CLEAN / "cdnow_customer_week_panel.csv", cdnow_config),
}

# NOTE: unlike `run_ar_encoding_ablation.PANEL_DEPTHS`, K does NOT vary by panel. That
# script's K is a WINDOW LENGTH — `active_in_last_K_periods` degenerates if K approaches
# the calibration window, so it must shrink for CDNOW's 39 periods. Here K is a NUMBER OF
# GROUPS, which depends on cohort size and heterogeneity rather than window length, and
# CDNOW's cohort is the larger of the two. The same K values are therefore comparable
# across both panels, which is what lets the arms be read side by side.

# Everything that is NOT the ablation. Identical in every arm, so the only thing moving is
# the feature encoding. `embedding_dim` is deliberately absent: the default `valendin`
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
    """`cluster__<panel>__<arm>__<shard>` — one disjoint root per rented worker."""
    stem = f"cluster__{panel}__{arm}__{shard}"
    return f"{stem}__{suffix}" if suffix else stem


# `spearman` and `score_suite` are duplicated from `run_ar_encoding_ablation.py` rather
# than imported from it. A rented worker is sent ONE script (VastAI/Rules.md §3), so a
# cross-script import would make this file unrunnable exactly where it is meant to run.
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, as Pearson on average-tied ranks.

    Written out rather than imported from `scipy.stats`: scipy is an undeclared
    transitive dependency of this project (it arrives via scikit-learn), so a script that
    named it directly would break on an environment that trimmed it.
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
    replications that none of those three files describe. Scoring the predictions directly
    makes a partial shard count for exactly the replications it finished.

    Alignment follows `studies.suite_metrics`: predictions are stored in the cohort's own
    order, and the ids are asserted against the rebuilt cohort so row i of the forecast and
    row i of the actuals are the same customer.
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
    """Pool every arm's shards and print level and discrimination side by side.

    `compare_study_metrics` accepts at most four suite roots and there are up to twelve
    here (six arms x two shards), so the pooling is done here: each shard contributes its
    own per-study rows and an arm's summary is taken over the union.
    """
    panel_path, build_config = PANELS[panel]

    rows: list[dict[str, object]] = []
    for arm_name, arm in ARMS.items():
        data = None
        for shard, base_seed in SHARDS.items():
            root = STUDIES_BASE / suite_name(panel, arm_name, shard, suffix)
            if not (root / "LSTM" / "Predictions").is_dir():
                continue
            # Rebuild this arm's dataset once, lazily: only needed to score, and only for
            # arms that actually have forecasts on disk.
            if data is None:
                data = panel_dataset.prepare_dataset(
                    pd.read_csv(panel_path), build_config(arm), verbose=False
                )
            for row in score_suite(root, data, base_seed):
                rows.append({"arm": arm_name, "shard": shard, **row})

    if not rows:
        print("No finished suites found. Run the arms first.")
        return

    per_study = pd.DataFrame(rows)
    coverage = per_study.groupby(["arm", "shard"]).size().unstack(fill_value=0)
    print(f"\n{len(per_study)} replications across {per_study.arm.nunique()} arms")
    print(f"\nreplications per shard ({N_STUDIES} expected each):")
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
        rmse_mean=("rmse", "mean"),
    )
    order = [a for a in ARMS if a in summary.index]
    summary = summary.loc[order]
    print(summary.round(3).to_string())

    missing = [a for a in MINIMUM_ARMS if a not in summary.index]
    if missing:
        verb = "has" if len(missing) == 1 else "have"
        print(
            f"\nIncomplete: {', '.join(missing)} {verb} no finished suite. The "
            f"comparison below needs all of {', '.join(MINIMUM_ARMS)}."
        )

    print(
        "\nA cluster arm SUCCEEDS only if BOTH hold:"
        "\n  spearman_mean rises materially above `no_cluster` (~0.036 archived), AND"
        "\n  abs_bias_mean stays near `no_cluster` rather than climbing toward"
        "\n  `ar_unbounded`."
        "\nA rise in discrimination bought with a rise in bias is the AR result again,"
        "\nnot an improvement on it. Compare gaps against bias_sd / spearman_sd: a"
        "\ndifference smaller than the across-study spread is not a result."
        "\nRMSE is shown for completeness; on a ~97%-zero panel it is dominated by the"
        "\nzeros and cannot separate these arms."
    )

    if "ar_unbounded" in summary.index:
        got = float(summary.loc["ar_unbounded", "bias_mean"])
        ok = 80 <= got <= 500
        print(
            f"\nSanity check — ar_unbounded reproduces the archived suites: "
            f"got {got:+.1f}%, archived +220% / +259%. "
            f"{'OK' if ok else 'OFF — investigate before trusting the cluster arms'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=sorted(PANELS), default="electronics")
    parser.add_argument("--arm", choices=sorted(ARMS))
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

    panel_path, build_config = PANELS[args.panel]
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Datasets/ is gitignored, so a rented worker needs "
            f"the panel pushed to it (VastAI/Rules.md §3)."
        )

    arm = ARMS[args.arm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_full = panel_dataset.prepare_dataset(
        pd.read_csv(panel_path), build_config(arm)
    )

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
    print(
        f"arm={args.arm}  shard={args.shard}  "
        f"ar_features={arm.ar_features}  cluster_features={arm.cluster_features}"
    )

    table = study_metrics(root, panel_path, standard_deviation=True)
    print(f"\nForecast metrics (mean over {args.n_studies} studies, +- SD):")
    print(table.to_string())


if __name__ == "__main__":
    main()
