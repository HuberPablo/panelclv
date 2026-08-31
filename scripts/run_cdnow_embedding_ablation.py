"""Embedding ablation for the LSTM on CDNOW — one arm per embedding strategy.

The question: **how much does the way features become a vector matter to the
forecast?** Everything downstream of the embedder is held fixed, so any difference
in holdout RMSE / bias / MAPE is attributable to the embedding alone.

The arms are the strategies `registry` registers, plus the one knob that exists in
only one of them:

    LSTM_valendin        one raw Embedding(n, sqrt(n)+1) per feature, concatenated,
                         no normalisation and no projection (the published design)
    LSTM_projected_32    every feature embedded, normalised and projected to a
    LSTM_projected_64    common width, the context summed and the target embedding
    LSTM_projected_128   concatenated last — one arm per width

`embedder` is an ordinary search-space key on the `lstm` registry entry, so an arm
is a *pin*, not new code: nothing in `src/` changes to run this. All four arms are
`model_type="lstm"` and differ only in that pin (and in `embedding_dim`, which the
Valendin strategy has no use for — the suggester skips it there rather than spending
trials on a number that never reaches the model).

**What makes it an ablation rather than four unrelated studies.** Every arm gets:

- the same dataset object, built once and shared;
- the same remaining search space and the same trial budget, so no arm can win by
  being allowed to search harder;
- the same seeds — study `j` of every arm uses `base_seed + j` — so the arms are
  paired replications and the across-studies spread is comparable.

Read the result with `study_metrics`, which scores every study's stored forecast
with the same function the runner used and reports the mean and the study-to-study
standard deviation per arm. A gap smaller than that spread is not a result.

Usage:
    1. python scripts/build_cdnow_panel.py     (writes the panel this reads)
    2. python scripts/run_cdnow_embedding_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite, study_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "Datasets" / "Dataset_clean" / "cdnow_customer_week_panel.csv"
STUDIES_BASE = REPO_ROOT / "Studies"
SUITE_NAME = "cdnow_embedding_ablation"

# --- study size: the whole cost of the run is these three numbers ---------------
# 4 arms x N_STUDIES x N_TRIALS trainings, plus one refit per arm per study.
N_TRIALS = 15          # Optuna trials per study
N_STUDIES = 3          # independent studies (seeds) per arm — the spread comes from these
N_SIMULATIONS = 300    # Monte Carlo rollout paths per forecast

LOSS_TYPE = "cross_entropy"


def build_panel_config() -> PanelConfig:
    """Column roles and window dates for the CDNOW weekly panel.

    The windows follow the standard CDNOW calibration/holdout split, snapped to the
    package's week grid: calibration is 1997 weeks 0-38 (39 weeks, the classic
    39-week calibration period) and the holdout is 1997 w39 through the last complete
    week of the data, 1998 w24 (38 weeks). The last eight calibration weeks are the
    temporal validation window (ADR-0001) — early stopping and trial selection see
    them, the weights never train on them.

    CDNOW carries no covariates, so the features are the target's own history plus
    engineered ones: `week_sin`/`week_cos` for the annual cycle and two leak-free
    autoregressive signals recomputed from the sampled count during the rollout.
    That mix is deliberate for this ablation. `prepare_dataset` standardises the
    numeric channels on the calibration window, so the arms are not separated by
    feature scaling: what separates them is that Valendin carries each standardised
    covariate as one raw channel beside the target embedding, while Projected embeds
    every feature to a common width and sums them. On this panel that is a 7-wide
    input against a 2 x `embedding_dim`-wide one.

    `embedded_cols` names only the target: its cardinality IS the softmax head size,
    and `"auto"` reads it off the data that `clip_target_upper` has already capped, so
    the head and the cap cannot drift apart.
    """
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
        # Weekly CDNOW counts are overwhelmingly 0 or 1; the cap sets the head size.
        # Check the tail `build_cdnow_panel.py` prints and raise this if the data
        # carries meaningful mass above it.
        clip_target_upper=4,
        time_features={"add_week_sin_cos": True},
        ar_features=("period_since_last_transaction", "has_transacted_before"),
        embedded_cols={"Transactions": "auto"},
    )


# The hyperparameters that are NOT the ablation. Identical in every arm, so the only
# thing that varies across arms is the embedder (and its width, where it has one).
SHARED_SEARCH_SPACE: dict[str, object] = {
    "lstm_hidden_size": {64, 128},
    "dense_units":      {64},
    "dropout":          {0.1, 0.3},
    "learning_rate":    (1e-4, 3e-3, "log"),
    "weight_decay":     (1e-6, 1e-3, "log"),
    "batch_size":       {256},
}

SHARED_TRAINING: dict[str, object] = {
    "n_epochs":  60,
    "patience":  6,
    "loss_type": LOSS_TYPE,
    "verbose":   False,
}

# One entry per arm: (embedder strategy, projection width or None). `None` is not a
# width of zero — it means the strategy has no common width, so the parameter is left
# out of the arm's search space entirely and the registry's suggester never offers it.
ARMS: tuple[tuple[str, int | None], ...] = (
    ("valendin", None),
    ("projected", 32),
    ("projected", 64),
    ("projected", 128),
)


def arm_name(embedder: str, width: int | None) -> str:
    """Folder name for one arm — this is what labels the row in the results table."""
    return f"LSTM_{embedder}" if width is None else f"LSTM_{embedder}_{width}"


def build_models() -> list[ModelSpec]:
    """One ModelSpec per arm: same type, same budget, same knobs, pinned embedder."""
    specs: list[ModelSpec] = []
    for embedder, width in ARMS:
        # A scalar in the search-space mini-language is a pin: the value is registered
        # as a one-choice categorical, so it still reaches `best_params` and the run's
        # record of what it used stays complete.
        search_space = {"embedder": embedder, **SHARED_SEARCH_SPACE}
        if width is not None:
            search_space["embedding_dim"] = width
        specs.append(
            ModelSpec(
                name=arm_name(embedder, width),
                model_type="lstm",
                n_trials=N_TRIALS,
                search_space=search_space,
                training=dict(SHARED_TRAINING),
            )
        )
    return specs


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{PANEL_PATH} not found — build it first with "
            "`python scripts/build_cdnow_panel.py`."
        )
    panel = pd.read_csv(PANEL_PATH)

    # Built once and shared by every arm: the arms must differ in the embedder and in
    # nothing else, and a per-arm rebuild is a place for them to drift.
    data_full = panel_dataset.prepare_dataset(panel, build_panel_config())

    # `run_study_suite` requires the base directory to exist, and failing on that after
    # every arm has trained is the expensive way to find out.
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    config = StudySuiteConfig(
        studies_base_path=str(STUDIES_BASE),
        suite_name=SUITE_NAME,
        n_studies_per_model=N_STUDIES,
        n_simulations=N_SIMULATIONS,
        device=device,
        data=data_full,
        models=build_models(),
        # Study j of every arm draws seed base_seed + j, so the arms are paired.
        base_seed=42,
        # 4 arms x 3 studies x 15 trials is 180 checkpoints kept for inspection
        # otherwise; the refit only ever warm-starts from each study's winner.
        keep_only_best_checkpoint=True,
    )

    root = run_study_suite(config)
    print(f"\nStudy suite written to: {root}")

    # --- the ablation table -------------------------------------------------
    # Mean over the N_STUDIES independent studies per arm, with the study-to-study
    # standard deviation beside it: a difference between arms that is smaller than
    # this spread is a difference between seeds.
    table = study_metrics(root, PANEL_PATH, standard_deviation=True)
    print("\nHoldout metrics by embedding strategy (mean over "
          f"{N_STUDIES} studies, +- study-to-study SD):")
    print(table.to_string())


if __name__ == "__main__":
    main()
