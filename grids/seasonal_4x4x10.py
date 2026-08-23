"""A 4x4x10 Pareto/NBD grid with fixed within-year seasonality.

Four mean purchase rates x four churn rates, ten replicate panels per cell = 160
datasets of 1000 customers over 156 weeks (three years: two of calibration, one of
holdout). Seasonality is held fixed across the whole grid rather than varied, so the
only thing moving between cells is the (rate, churn) regime.

Copy this file to declare another grid; the module name is the grid's name and every
path derives from it.
"""

from __future__ import annotations

from panelclv.configs.panel_config import PanelConfig
from panelclv.studies import ModelSpec

from . import GridSpec

# The window dates must land inside the calendar the generator produces:
# start_year=1999 and n_weeks=156 gives 1999-2001, so two calibration years and one
# holdout year. Change n_weeks or start_year above and these move with them.
PANEL = PanelConfig(
    id_col="Id",
    target_col="Transactions",
    frequency="weekly",
    training_start="1999-01-01",
    training_end="2000-12-31",
    validation_start="2000-01-01",      # last calibration year = the temporal val split
    holdout_start="2001-01-01",
    holdout_end="2001-12-31",
    time_cols=("year", "week"),
    clip_target_upper=6,                # also the softmax head size (7 classes: 0..6)
    require_calibration_activity=True,
    time_features={"add_year_idx": True, "add_week_sin_cos": True},
    # Pure Pareto/NBD panels carry no covariates: the target's own past is the input.
    known_future=(),
    static=(),
    observed_past=(),
    embedded_cols={"Transactions": "auto"},
)

# 20 Optuna trials per dataset, not the 100 a single-panel suite uses: this budget is
# paid 160 times over, once per dataset, so it buys a tuned model per dataset rather
# than an exhaustively tuned one.
LSTM = ModelSpec(
    name="LSTM",
    model_type="lstm",
    n_trials=10,
    search_space={
        "batch_size":       {64, 128, 256},
        "learning_rate":    (1e-4, 1e-2, "log"),
        "embedding_dim":    {64, 128, 256},
        "lstm_hidden_size": {32, 64, 128},
        "dense_units":      {32, 64, 128},
        "dropout":          {0.0, 0.2, 0.4},
    },
    training={
        "n_epochs": 100,
        "patience": 7,
        "loss_type": "cross_entropy",
        "verbose": False,
    },
)

# Same trial budget as the LSTM so the two are compared on equal search effort. The
# search space is the one scripts/run_studies.py declares for the single-panel suite.
TRANSFORMER = ModelSpec(
    name="Transformer",
    model_type="transformer",
    n_trials=20,
    search_space={
        "batch_size":         {64, 128, 256},
        "d_model":            {32, 64, 128},
        "nhead":              {2, 4, 8},
        "num_encoder_layers": (1, 3, "int"),
        "dropout":            {0.0, 0.1, 0.2, 0.3},
        "learning_rate":      (1e-4, 3e-3, "log"),
        "weight_decay":       (1e-6, 1e-2, "log"),
    },
    training={
        "n_epochs": 100,
        "patience": 7,
        "loss_type": "cross_entropy",
        "verbose": False,
    },
)

# The Valendin reference implementation. Its architecture is FROZEN (ADR-0004) — the
# published memory_units=128 / dense_units=128 and raw sqrt(n)+1 embeddings — so no
# search space is declared here at all: it inherits the registry entry's, which offers
# only the three training hyperparameters. Naming a width would quietly unfreeze the
# benchmark. Same trial budget as the others so search effort stays comparable.
VALENDIN = ModelSpec(
    name="ValendinLSTM",
    model_type="valendin_lstm",
    n_trials=20,
    training={
        "n_epochs": 100,
        "patience": 7,
        "loss_type": "cross_entropy",
        "verbose": False,
    },
)

# No Optuna: one deterministic MCMC fit on BTYDplus's default settings. On synthetic
# Pareto/NBD data this is the *correct* model by construction, so it is the ceiling the
# neural models are measured against.
PARETO = ModelSpec(name="ParetoNBD", model_type="pareto_nbd")

GRID = GridSpec(
    name="seasonal_4x4x10",

    # --- axes ---------------------------------------------------------------
    mean_transaction_rates=[0.01, 0.05, 0.10, 0.30],
    churn_rates=[0.20, 0.40, 0.60, 0.80],
    n_weeks_for_churn_rate=52,

    # --- panels -------------------------------------------------------------
    n_customers=1000,
    n_weeks=156,
    n_datasets=10,
    base_seed=42,
    start_year=1999,
    r=2.0,
    s=2.0,

    # --- seasonality: spring, two summer peaks, year-end --------------------
    seasonal_peaks=[12, 25, 30, 47],
    seasonal_amplitude=1.5,
    seasonal_width=3,

    # --- training -----------------------------------------------------------
    panel=PANEL,
    models=(LSTM, TRANSFORMER, VALENDIN, PARETO),
    n_simulations=200,

    # EDIT ME: how many vast.ai workers each model's 160 datasets are split across.
    # 0 = run on the orchestrator instead of renting (VastAI/Rules.md §5).
    workers={"transformer": 7, "valendin_lstm": 2, "lstm": 1, "pareto_nbd": 0},
)
