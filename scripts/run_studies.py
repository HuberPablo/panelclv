"""Entry point for a study suite — copy, edit the data loading + models, run.

Runs ``n_studies_per_model`` independent Optuna studies for each model over one
shared dataset and archives everything under ``Studies/<study_name>/``. The config
below is the same set of arguments you already pass to ``run_optuna_study``, so
there is nothing new to learn — paste your ``search_space`` / ``training`` blocks
verbatim.

Usage:
    1. Edit ``load_panel()`` to read your panel CSV.
    2. Edit the ``PanelConfig`` and the ``models=[...]`` list.
    3. Pick a ``study_name``.
    4. ``python scripts/run_studies.py``

Output lands in ``<repo root>/Studies/<study_name>/``, resolved from this file's
location rather than from an absolute path, so the script runs unchanged on a
laptop or on a rented GPU box with no edit and no patching step.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import dynamic_panel_dataset
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite

# --- EDIT ME: tags that name this run (mirrors the notebook constants) ----------
LOSS_TYPE = "cross_entropy"
CONFIG_NAME = "baseline"

# Repo root = the parent of scripts/. Everything the suite writes goes under it,
# so the only machine-specific thing left is where you cloned the repo.
STUDIES_BASE = Path(__file__).resolve().parents[1] / "Studies"


def load_panel() -> pd.DataFrame:
    """EDIT ME — return the customer-period panel DataFrame to model."""
    return pd.read_csv("Datasets/Dataset_clean/electronics_customer_week_panel.csv")


def build_models() -> list[ModelSpec]:
    """The two neural search spaces (passed verbatim to run_optuna_study) + Pareto.

    The sets / tuples in ``search_space`` override the range each model's registry
    entry declares for that hyperparameter (a ``{...}`` set is a categorical choice,
    a ``(lo, hi, "log"|"int")`` tuple is a range); anything left out keeps the
    entry's own range. ``training`` holds what is not searched. The suite assigns
    ``study_name`` / ``summary_dir`` / ``storage`` / ``checkpoint_dir`` / seed per
    study, so those are intentionally omitted here.
    """
    lstm = ModelSpec(
        name="LSTM",
        model_type="lstm",
        n_trials=100,
        search_space={
            "batch_size":      {64, 128, 256},
            "learning_rate":   (1e-4, 1e-2, "log"),
            "embedding_dim":   {64, 128, 256},
            "lstm_hidden_size": {32, 64, 128},
            "dense_units":     {32, 64, 128},
            "dropout":         {0.0, 0.2, 0.4},
        },
        training={
            # n_epochs / patience are training control, but may still carry a spec.
            "n_epochs":  {100},
            "patience":  {5, 7, 9},
            "verbose":   False,
            "loss_type": LOSS_TYPE,
            # "class_weights": class_weights,  # used by 'weighted_ce' / 'focal'
        },
    )

    transformer = ModelSpec(
        name="Transformer",
        model_type="transformer",
        n_trials=100,
        search_space={
            "batch_size":         {64, 128, 256},
            "d_model":            {32, 64, 128},
            "nhead":              {2, 4, 8},
            "num_encoder_layers": (1, 3, "int"),
            "dropout":            {0.0, 0.1, 0.2, 0.3},

            "learning_rate": (1e-4, 3e-3, "log"),
            "weight_decay":  (1e-6, 1e-2, "log"),
        },
        training={
            "n_epochs":  100,
            "patience":  7,
            "loss_type": LOSS_TYPE,
        },
    )

    pareto = ModelSpec(
        name="ParetoNBD",
        model_type="pareto_nbd",
        # no Optuna; a single deterministic fit on BTYDplus's default MCMC settings
    )

    return [lstm, transformer, pareto]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. data: exactly as in the README quickstart -----------------------
    panel = load_panel()
    cfg = PanelConfig(
        id_col="Id", target_col="Transactions", frequency="weekly",
        training_start="1999-01-01", training_end="2000-12-31",
        validation_start="2000-07-01",
        holdout_start="2001-01-01", holdout_end="2001-12-31",
        time_cols=("year", "week"), clip_target_upper=6,
    )
    data_full = dynamic_panel_dataset.prepare_dataset(panel, cfg)

    # --- 2. the suite config ------------------------------------------------
    # run_study_suite requires the base directory to already exist, and failing on
    # that *after* every model has trained is the expensive way to find out.
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    config = StudySuiteConfig(
        studies_base_path=str(STUDIES_BASE),
        study_name=f"{LOSS_TYPE}_{CONFIG_NAME}",     # new folder created under it
        n_studies_per_model=5,                       # X independent studies per model
        n_simulations=600,
        device=device,
        data=data_full,                              # shared by all models
        models=build_models(),
    )

    root = run_study_suite(config)
    print(f"Study suite written to: {root}")


if __name__ == "__main__":
    main()
