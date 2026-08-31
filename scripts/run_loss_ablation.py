"""Loss ablation — cross-entropy vs. squared EMD vs. their sum (R1 and R2).

Implements the two ranked recommendations of `docs/loss-functions.md`: everything about
the model is held fixed and only the training objective moves.

    LSTM_ce       loss_type="cross_entropy"   the package default
    LSTM_emd      loss_type="emd"             squared EMD = the discrete CRPS (R1)
    LSTM_ce_emd   loss_type="ce_emd"          CE + lambda*EMD, lambda searched (R2)

**Why these three are comparable at all.** Squared EMD is identically the Ranked
Probability Score, the discrete case of the CRPS, and is *strictly proper* — so like
cross-entropy it is minimised by the truth, and the rollout still samples from an
unbiased estimate of the predictive distribution. That is what separates this ablation
from one over `weighted_ce` / `focal`, which are improper here and tilt the very
distribution the simulator draws from (`docs/loss-functions.md` §5). The softmax head,
the sampler, the architecture and the metrics are untouched in all three arms.

**Read it on bias, not RMSE.** On a 97%-zero panel, predicting zero everywhere beats a
trained model on RMSE — that metric is dominated by the zeros and cannot separate these
arms. The quantity EMD acts on is the K-1 CDF residuals that the forecast's aggregate
bias is built from, so `bias_percent` and `mape_aggregate` are where an effect would
show.

**Never compare the `objective` column across arms.** Each arm's Optuna objective is its
own loss, and the three are on different scales; a lower validation loss under `emd`
says nothing against one under `cross_entropy`. Only the forecast metrics are comparable.

**Detectability.** The archived electronics suite's across-studies SD of `bias_percent`
is 27.5 (LSTM). At `N_STUDIES = 10` the standard error of an arm's mean is ~9 points, so
an effect much smaller than that is not resolvable by comparing means — which is why the
script also prints the *paired per-seed* differences, where the seed-to-seed variance
cancels.

Run on BOTH panels: they exercise the ordinal argument differently and may disagree.
CDNOW has no tail to speak of (a 5-class head, three clipped cells in the whole panel);
electronics has a heavy one. A result on one is not a result on the other.

Usage:
    python scripts/run_loss_ablation.py --panel cdnow
    python scripts/run_loss_ablation.py --panel electronics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite, study_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"
CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"

# --- study size ----------------------------------------------------------------
# 3 arms x N_STUDIES x N_TRIALS trainings, plus one refit per arm per study.
N_TRIALS = 20          # Optuna trials per study
N_STUDIES = 10         # independent studies per arm — see "Detectability" above
N_SIMULATIONS = 300    # Monte Carlo paths per forecast (the doc's floor)


def cdnow_config() -> PanelConfig:
    """The CDNOW panel, exactly as `run_cdnow_embedding_ablation.py` reads it."""
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        training_start="1997-01-01",
        validation_start="1997-08-06",   # 1997 week 31 — last 8 calibration weeks
        training_end="1997-09-30",       # inclusive of 1997 week 38
        holdout_start="1997-10-01",      # 1997 week 39
        holdout_end="1998-06-30",        # inclusive of the last complete week, 1998 w24
        clip_target_upper=4,             # 5-class head; 3 cells in 181,489 exceed it
        ar_features=("period_since_last_transaction", "has_transacted_before"),
        embedded_cols={"Transactions": "auto"},
    )


def electronics_config() -> PanelConfig:
    """The electronics panel as the ARCHIVED suites read it, deliberately unchanged.

    Same windows, clipping and feature set as
    `Studies/cross_entropy_cfg_2yTrain_1yPred_NoCov_Comparaison`, so this ablation's
    baseline arm is comparable with what is already on disk. It therefore inherits that
    config's two known weaknesses — no AR features and no declared covariates, which are
    `docs/loss-functions.md` R5.1 and R5.2. Both arms share them, so neither confounds
    the loss comparison; fixing them is a different experiment.
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
        time_features={"add_year_idx": True, "add_week_sin_cos": True},
        embedded_cols={"Transactions": "auto"},
    )


PANELS = {
    "cdnow":       (CLEAN / "cdnow_customer_week_panel.csv", cdnow_config),
    "electronics": (CLEAN / "electronics_customer_week_panel.csv", electronics_config),
}

# Everything that is NOT the ablation. Identical in every arm, so the only thing moving
# is the objective. `embedder` is left at the registry default (`valendin`) and
# `embedding_dim` is deliberately absent: that strategy has no common width, so naming
# one would advertise a search that never happens (docs/running-a-model.md §14).
SHARED_SEARCH_SPACE: dict[str, object] = {
    "lstm_hidden_size": {64, 128},
    "dense_units":      {64},
    "dropout":          {0.1, 0.3},
    "learning_rate":    (1e-4, 3e-3, "log"),
    "weight_decay":     (1e-6, 1e-3, "log"),
    "batch_size":       {256},
}

SHARED_TRAINING: dict[str, object] = {
    "n_epochs": 60,
    "patience": 6,
    "verbose":  False,
}

# One entry per arm: (suffix, loss_type, extra training knobs).
#
# lambda's range INCLUDES 0, where `ce_emd` is exactly `cross_entropy` (asserted by
# `test_ce_emd_at_zero_weight_is_exactly_cross_entropy`). So the composite arm cannot
# lose to the baseline except through search noise — and a search that keeps choosing
# lambda near 0 is itself the answer, retiring the line of enquiry cleanly.
ARMS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("ce",     "cross_entropy", {}),
    ("emd",    "emd",           {}),
    ("ce_emd", "ce_emd",        {"emd_weight": (0.0, 10.0)}),
)


def build_models() -> list[ModelSpec]:
    """One ModelSpec per arm: same type, same budget, same knobs, different loss."""
    return [
        ModelSpec(
            name=f"LSTM_{suffix}",
            model_type="lstm",
            n_trials=N_TRIALS,
            search_space=dict(SHARED_SEARCH_SPACE),
            training={**SHARED_TRAINING, "loss_type": loss_type, **extra},
        )
        for suffix, loss_type, extra in ARMS
    ]


def report(root: Path, panel_path: Path) -> None:
    """The ablation table, the paired differences, and the lambdas the search chose."""
    table = study_metrics(root, panel_path, standard_deviation=True)
    print(f"\nForecast metrics by loss (mean over {N_STUDIES} studies, +- SD):")
    print(table.to_string())
    print("\nRead bias_percent and mape_aggregate; RMSE cannot separate these arms.")

    results = pd.read_csv(root / "results.csv")

    # Paired per-seed differences against the baseline arm. Every arm ran study i under
    # the same seed, so differencing on `seed` removes the study-to-study variance that
    # swamps a comparison of means at this sample size.
    base = results[results.model == "LSTM_ce"].set_index("seed")
    for arm in ("LSTM_emd", "LSTM_ce_emd"):
        other = results[results.model == arm].set_index("seed")
        common = base.index.intersection(other.index)
        if common.empty:
            continue
        # The comparison is on the MAGNITUDE of the bias: an arm that moves bias from
        # +40% to -30% has not helped, and a signed difference would score it as a large
        # improvement. Negative delta = this arm is closer to unbiased on that seed.
        delta = (
            other.loc[common, "bias_percent"].abs() - base.loc[common, "bias_percent"].abs()
        )
        print(
            f"\n{arm} vs LSTM_ce, paired on seed — change in |bias_percent|:"
            f"\n  mean {delta.mean():+.2f}  SD {delta.std(ddof=1):.2f}  "
            f"n={len(common)}  improved on {int((delta < 0).sum())}/{len(common)} seeds"
        )

    # R2's own stopping rule: if the search keeps picking lambda ~ 0, the composite is
    # just cross-entropy and the recommendation retires itself.
    if "param_emd_weight" in results.columns:
        lam = results.loc[results.model == "LSTM_ce_emd", "param_emd_weight"].dropna()
        if not lam.empty:
            print(
                f"\nlambda chosen by the search across {len(lam)} studies: "
                f"min {lam.min():.3f}  median {lam.median():.3f}  max {lam.max():.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=sorted(PANELS), required=True)
    args = parser.parse_args()

    panel_path, build_config = PANELS[args.panel]
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = pd.read_csv(panel_path)

    # Built once and shared by every arm: the arms must differ in the loss and nothing
    # else, and a per-arm rebuild is a place for them to drift.
    data_full = panel_dataset.prepare_dataset(panel, build_config())

    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    config = StudySuiteConfig(
        studies_base_path=str(STUDIES_BASE),
        suite_name=f"loss_ablation_{args.panel}",
        n_studies_per_model=N_STUDIES,
        n_simulations=N_SIMULATIONS,
        device=device,
        data=data_full,
        models=build_models(),
        # Study j of every arm draws seed base_seed + j, so the arms are paired.
        base_seed=42,
        keep_only_best_checkpoint=True,
        # NOTE: no `refit_kwargs`. The runner forwards each arm's own loss to its refit
        # (ADR-0008 is where the forecast comes from); setting `loss_type` here would
        # override all three arms with one loss and silently flatten the ablation.
    )

    root = run_study_suite(config)
    print(f"\nStudy suite written to: {root}")
    report(root, panel_path)


if __name__ == "__main__":
    main()
