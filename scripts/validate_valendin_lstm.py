"""Validate `benchmarks/valendin_lstm.py` against the reference notebook's own run.

The counterpart to `scripts/validate_pareto_benchmark.py`, for the neural benchmark:
it rebuilds the dataset `Original_paper_model/banking_transactions_demo.ipynb` builds,
trains `ValendinLSTMModel` on it under the notebook's protocol, and reports the two
numbers the notebook publishes.

    .../thesis_rocm/bin/python scripts/validate_valendin_lstm.py

The notebook's published targets are approximate by its author's own account — its
markdown says the validation loss "should end up around 0.44" and the prediction bias
"should be less than 1%", and repeated runs of the reference code land in a band
rather than on a point. So this script reports and judges against a tolerance band,
not an equality.

What is reproduced from the notebook
------------------------------------
- Cohort      : accounts whose FIRST transaction falls on or before `training_end`.
- Time grid   : week = (dayofyear // 7).clip(upper=51), so 52 classes; the 52nd week
                rolls into the 51st. Calibration 1993-1995, holdout 1996-1998, giving
                156 calibration weeks and a 155-step training sequence.
- Features    : [week, transactions], in that order — the notebook concatenates
                [emb_week, emb_trans], and the embedder concatenates in column order.
- Target      : next-step transaction count, i.e. the sequence shifted by one.
- Split       : a random 10% of CUSTOMERS held out for validation.
- Optimiser   : Adam at its defaults (lr=1e-3, no weight decay, no gradient clipping),
                sparse categorical cross-entropy, batch 32, early stopping on
                validation loss with patience 5, restoring the best epoch.

Note the split. This project's own studies use a temporal validation split (ADR-0001),
a deliberate departure — but a reproduction has to run the protocol it reproduces, or
the validation loss is not the same quantity the notebook reports. The departure
applies to our studies, not to this check.

Everything after the architecture is the package's shared infrastructure: the training
loop is `training.fit_model` and the holdout rollout is the Monte Carlo simulator in
`models.monte_carlo_forecasting`, exactly as for every other model.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# src-layout: the package lives under <repo>/src, so add that to the path as a
# fallback for running this script without an editable install.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from panelclv.benchmarks.valendin_lstm import ValendinLSTMModel  # noqa: E402
from panelclv.models.monte_carlo_forecasting import (  # noqa: E402
    compute_forecast_metrics,
    forecast_recurrent,
)
from panelclv.training.training_utils import fit_model  # noqa: E402

# --- the notebook's constants ------------------------------------------------
TRAINING_START = "1993-01-01"
TRAINING_END = "1995-12-31"
HOLDOUT_START = "1996-01-01"
HOLDOUT_END = "1998-12-31"

# The notebook's own week grid, re-implemented here rather than imported from
# `panelclv.data_preparation.period_calendar`, and deliberately so — note it is not even
# the package's convention: this is `dayofyear // 7` (week 0 holds six days), the package
# uses `(dayofyear - 1) // 7`. Two reasons to keep it. A reproduction has to run the grid
# it reproduces, and a gate that imports the code it gates stops being a gate — a future
# edit to the shared convention would move both the benchmark and this check together and
# still land in the band.
WEEKS_PER_YEAR = 52
VALIDATION_SPLIT = 0.1
BATCH_SIZE_TRAIN = 32
MAX_EPOCHS = 150
PATIENCE = 5

# The notebook's published targets, and the band we accept around them.
PUBLISHED_VAL_LOSS = 0.44
VAL_LOSS_TOLERANCE = 0.06      # "around 0.44"
PUBLISHED_BIAS_LIMIT_PCT = 1.0
BIAS_LIMIT_TOLERANCE_PCT = 2.0  # run-to-run spread; see the module docstring


def build_banking_dataset(csv_path: Path, seed: int = 0) -> dict:
    """Rebuild the notebook's dataset from the raw transaction log.

    Returns the dict the package's simulator consumes (`calibration`, `holdout`,
    `seq_cols`, `target_col`, `embedded_cols`) plus the training tensors.
    """
    df = pd.read_csv(csv_path, usecols=["account_id", "date"],
                     parse_dates=["date"], date_format="%y%m%d")

    # Cohort: first transaction on or before the end of calibration. Customers first
    # seen only in the holdout are unknown at forecast time.
    first_seen = df.groupby("account_id")["date"].min()
    cohort = sorted(first_seen[first_seen <= pd.Timestamp(TRAINING_END)].index)
    df = df[df["account_id"].isin(cohort)]
    print(f"accounts: {len(cohort):,}   transactions: {len(df):,}")

    # The full (year, week) calendar spanning calibration + holdout. Building the grid
    # explicitly is what gives every customer the same length with zeros for silence.
    calendar = pd.DataFrame(
        {"date": pd.date_range(TRAINING_START, HOLDOUT_END, freq="D")}
    )
    calendar["year"] = calendar["date"].dt.year
    calendar["week"] = (calendar["date"].dt.dayofyear // 7).clip(upper=WEEKS_PER_YEAR - 1)
    weeks = (calendar.groupby(["year", "week"], as_index=False)["date"].min()
                     .sort_values("date").reset_index(drop=True))
    n_weeks = len(weeks)
    n_cal = int((weeks["date"] < pd.Timestamp(HOLDOUT_START)).sum())
    print(f"weeks: {n_weeks} ({n_cal} calibration + {n_weeks - n_cal} holdout)")

    # Per-customer weekly counts, as one (N, n_weeks) matrix. Done with a pivot rather
    # than the notebook's per-customer loop — same numbers, minutes faster.
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["week"] = (df["date"].dt.dayofyear // 7).clip(upper=WEEKS_PER_YEAR - 1)
    counts = (df.groupby(["account_id", "year", "week"]).size()
                .rename("transactions").reset_index())
    week_index = {(int(y), int(w)): i for i, (y, w)
                  in enumerate(zip(weeks["year"], weeks["week"]))}
    account_index = {a: i for i, a in enumerate(cohort)}

    matrix = np.zeros((len(cohort), n_weeks), dtype=np.int64)
    rows = counts["account_id"].map(account_index).to_numpy()
    cols = [week_index[(int(y), int(w))] for y, w in zip(counts["year"], counts["week"])]
    matrix[rows, cols] = counts["transactions"].to_numpy()
    assert matrix.sum() == len(df), "lost transactions while gridding"

    max_trans = int(matrix.max()) + 1        # +1: 0 is a valid weekly count
    print(f"max transactions per account-week: {int(matrix.max())} "
          f"-> {max_trans} count classes")

    # Feature 0 is the week index, feature 1 the transaction count — the order the
    # notebook concatenates its embeddings in.
    week_col = np.tile(weeks["week"].to_numpy(dtype=np.int64), (len(cohort), 1))
    features = np.stack([week_col, matrix], axis=-1)        # (N, n_weeks, 2)

    calibration = features[:, :n_cal, :]
    holdout = features[:, n_cal:, :]

    # Training pairs: predict each step's count from the previous step.
    samples = torch.from_numpy(calibration[:, :-1, :]).float()   # (N, 155, 2)
    targets = torch.from_numpy(matrix[:, 1:n_cal]).long()        # (N, 155)

    # A random 10% of customers for validation — the notebook shuffles then slices.
    order = list(range(len(cohort)))
    random.Random(seed).shuffle(order)
    n_valid = round(len(order) * VALIDATION_SPLIT)
    valid_idx, train_idx = order[-n_valid:], order[:-n_valid]
    print(f"train customers: {len(train_idx):,}   validation: {len(valid_idx):,}")

    return {
        "calibration": calibration,
        "holdout": holdout,
        "seq_cols": ["week", "transaction"],
        "target_col": "transaction",
        # Where the count channel sits on the feature axis. `prepare_dataset` records
        # this for a real panel and the shared simulator reads it; this dict is built
        # by hand (see the header — the gate reproduces the notebook's own grid), so it
        # states the same thing itself. 1, matching the stack order above.
        "target_idx": 1,
        "embedded_cols": {"week": WEEKS_PER_YEAR, "transaction": max_trans},
        "ids": cohort,
        "id_col": "account_id",
        "max_trans": max_trans,
        "seq_len": samples.shape[1],
        "train_ds": TensorDataset(samples[train_idx], targets[train_idx]),
        "valid_ds": TensorDataset(samples[valid_idx], targets[valid_idx]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=_REPO / "Datasets" / "trans.csv",
                        help="raw transaction log (account_id, date)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--n-simulations", type=int, default=30,
                        help="Monte Carlo paths averaged for the holdout forecast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"transaction log not found: {args.data}")

    torch.manual_seed(args.seed)
    data = build_banking_dataset(args.data, seed=args.seed)

    model = ValendinLSTMModel(
        seq_cols=data["seq_cols"],
        embedded_cols=data["embedded_cols"],
        target_col=data["target_col"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nValendinLSTMModel: {n_params:,} parameters, "
          f"LSTM input {model.backbone.lstm.input_size}\n")

    # Adam at its Keras defaults: no weight decay, no gradient clipping.
    result = fit_model(
        model,
        DataLoader(data["train_ds"], batch_size=BATCH_SIZE_TRAIN, shuffle=True),
        DataLoader(data["valid_ds"], batch_size=len(data["valid_ds"])),
        max_trans=data["max_trans"],
        n_epochs=args.epochs,
        patience=PATIENCE,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=None,
        device=args.device,
        checkpoint_dir=_REPO / "checkpoints" / "valendin_validation",
        model_name="valendin_lstm_validation",
        loss_type="cross_entropy",
        verbose=True,
    )
    # The notebook's EarlyStopping uses restore_best_weights=True, and `fit_model`
    # matches it: the model it returns holds the best epoch's weights, not the last
    # one's (ADR-0007), so the rollout below scores the model the protocol selected.
    val_loss = result.best_val_loss
    print(f"\nbest validation loss: {val_loss:.4f} (epoch {result.best_epoch + 1})  "
          f"(published: around {PUBLISHED_VAL_LOSS})")

    # Holdout rollout through the shared simulator: warm up on calibration, then step
    # the holdout one week at a time, feeding each sampled count back in. True holdout
    # counts are never fed in — they are only used to score.
    rollout = model.to_rollout()
    forecast = forecast_recurrent(
        rollout, data, n_simulations=args.n_simulations,
        device=args.device, seed=args.seed, return_simulations=False,
    )
    metrics = compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])
    print(f"\nholdout ({args.n_simulations} simulated paths):")
    for name, value in metrics.items():
        print(f"  {name:22s} {value:10.4f}")

    # Bias per holdout year. An autoregressive rollout feeds its own samples back, so a
    # small over-prediction can compound over 156 steps; a bias that grows year on year
    # is drift, while a flat one is a level difference between the two windows.
    actual_t = np.asarray(forecast["actual"]).sum(axis=0)
    pred_t = np.asarray(forecast["prediction_mean"]).sum(axis=0)
    print("\n  bias by holdout year (drift check):")
    for year in range(len(actual_t) // WEEKS_PER_YEAR):
        lo, hi = year * WEEKS_PER_YEAR, (year + 1) * WEEKS_PER_YEAR
        a, p = actual_t[lo:hi].sum(), pred_t[lo:hi].sum()
        print(f"    year {year + 1}: actual {a:9,.0f}  predicted {p:9,.0f}  "
              f"bias {100.0 * (p - a) / a:+6.2f}%")

    bias = abs(metrics["bias_percent"])
    loss_ok = abs(val_loss - PUBLISHED_VAL_LOSS) <= VAL_LOSS_TOLERANCE
    bias_ok = bias <= PUBLISHED_BIAS_LIMIT_PCT + BIAS_LIMIT_TOLERANCE_PCT
    print(f"\n  validation loss {val_loss:.4f} vs {PUBLISHED_VAL_LOSS} "
          f"+/- {VAL_LOSS_TOLERANCE}: {'PASS' if loss_ok else 'REVIEW'}")
    print(f"  |bias| {bias:.2f}% vs published <{PUBLISHED_BIAS_LIMIT_PCT}% "
          f"(+{BIAS_LIMIT_TOLERANCE_PCT}% run-to-run): {'PASS' if bias_ok else 'REVIEW'}")

    ok = loss_ok and bias_ok
    print("\nRESULT:", "PASS — matches the reference notebook within its own run-to-run spread"
          if ok else "REVIEW — outside the expected band, inspect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
