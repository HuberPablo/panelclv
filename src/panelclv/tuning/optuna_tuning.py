"""Optuna tuning for the multinomial LSTM / Transformer baselines.

One file, two search spaces (`suggest_lstm_params` / `suggest_transformer_params`),
one shared `objective`. Each trial samples an architecture + training HPs (and,
optionally, a covariate subset), trains via `training_utils.fit_model` — which
always optimises classification cross-entropy and owns the loss curve, early
stopping, and per-epoch pruning reports — then returns a score to Optuna. What
that returned score IS depends on `selection_metric` (see below); the training
objective is CE either way.

data_builder contract
---------------------
The caller supplies a `data_builder`, so Optuna never touches the raw dataframe
or re-runs data prep per trial:

    train_loader, val_loader, metadata = data_builder(
        feature_config=feature_config,   # list of column names to DROP this trial
        batch_size=batch_size,
    )

`metadata` must contain `seq_cols` (list[str] matching the input tensor's last
axis), `embedded_cols` ({col: cardinality}), and `target_col`
(the AR target, default "Transactions"); optionally `seq_len` (Transformer
fixed-length mask cache; the LSTM ignores it) and `val_score_start` (the temporal
validation boundary — the objective forwards it to `fit_model` so `val_loss` is the
cross-entropy on the validation window only). `experiments.make_loaders` /
`make_data_builder` produce a contract-compliant builder from a `prepare_dataset`
dict; the train/val split is temporal (a time window over all customers).

Feature-group selection
-----------------------
`removable_features` lists covariates Optuna may drop. An entry is one column
(its own on/off toggle) or a group toggled as a unit — e.g. `("week_sin",
"week_cos")`, since a cyclical pair is meaningless split. Per trial,
`suggest_covariate_selection` samples the toggles and hands the dropped set to
`data_builder`; `select_features` then slices the precomputed `(N,T,F)` tensors
(no data re-prep) and rebuilds `samples`/`targets`/`target_idx`/`embedded_cols`
for the reduced layout. The target is never removable. Each trial records its
`selected_features` / `dropped_features` user-attrs so the summary CSV/JSON is
self-documenting and the winner can be rebuilt with `select_features_for_trial`.

ar_features stay in lockstep: the autoregressive target-derived columns
(recency / frequency / tenure / rate) live in `data["ar_features"]`, and
`select_features` filters that list to the surviving columns. So if a trial
drops an AR covariate, it is removed from `ar_features` too — otherwise the
Monte-Carlo rollout would try to look it up by `seq_cols.index(name)` and raise.

Selection metric (what Optuna minimises)
----------------------------------------
- "val_loss" (default): teacher-forced next-step validation cross-entropy.
  Cheap and on one scale across architectures, but blind to the autoregressive
  sampling rollout the real forecast uses, so it can favour feature sets that
  drift at forecast time.
- "rollout_composite": after training, score the trial with a LEAK-FREE
  validation Monte-Carlo rollout (`weekly_aggregate_rollout_metrics`). The
  temporal validation window (the last `n_val_periods` weeks of CALIBRATION, i.e.
  after `validation_start`) is carved off as a pseudo-holdout for ALL customers
  (the real `data["holdout"]` is never read in tuning); the model warms up on the
  prefix and autoregressively rolls the pseudo-holdout over `rollout_n_simulations`
  paths. Metrics are computed on the
  WEEKLY AGGREGATE (sum over customers per step) and combined into one
  scale-normalised composite:

      rmse_norm = aggregate_RMSE / mean_weekly_volume
      mape_norm = masked_clipped_MAPE / 100   (weeks below a volume floor skipped)
      bias_norm = |aggregate_bias_percent| / 100
      score     = w_rmse*rmse_norm + w_mape*mape_norm + w_bias*bias_norm

  Defaults w_rmse=1.0, w_mape=0.5, w_bias=0.3; lower is better, so the study
  stays direction="minimize". The normalisation makes the score comparable
  across datasets of very different volume. CE is still logged as `val_loss`,
  and every sub-metric (`rollout_rmse/mape/bias_percent/score`) as a user-attr.

  Two caveats: (a) the composite is on a DIFFERENT scale than CE, so a rollout
  run needs its OWN fresh study/storage — never a val_loss study's DB; (b) the
  pruner still acts on per-epoch CE, so it only prunes clearly bad-CE trials
  early — surviving trials are always rolled out and scored. Requires
  `rollout_data` (the full prepare_dataset dict; its `n_val_periods` sets the
  default horizon); `rollout_horizon` is validated up front (0 < horizon < T_CAL,
  with a short-warm-up warning).
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import optuna
import torch

# Model definitions and the Monte Carlo simulator live in `panelclv.models`; the
# training loop lives in `panelclv.training`. After the subpackage split these are
# cross-package imports, so they are absolute rather than relative.
from panelclv.models.multinomial_lstm import MultinomialLSTMModel, InferenceMultinomialLSTMModel
from panelclv.models.multinomial_transformer import (
    MultinomialTransformerModel,
    InferenceMultinomialTransformerModel,
)
from panelclv.models.monte_carlo_forecasting import (
    run_monte_carlo_forecast,
    run_monte_carlo_forecast_transformer,
)
from panelclv.training.training_utils import fit_model


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


DataBuilder = Callable[..., tuple[Any, Any, dict[str, Any]]]


# ---------------------------------------------------------------------------
# Feature-group selection
# ---------------------------------------------------------------------------


def _as_group(item: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one `removable_features` entry to a tuple of column names.

    An entry is either a single column name (``"Gender"`` → one toggle) or a
    group of names that must toggle together (``("week_sin", "week_cos")`` →
    one toggle covering both). Grouping exists because some encodings are
    atomic: a cyclical sin/cos pair is meaningless with only one half present.
    """
    return (item,) if isinstance(item, str) else tuple(item)


def validate_removable_features(
    removable: Sequence[str | Sequence[str]],
    seq_cols: Sequence[str],
    target_col: str,
) -> None:
    """Fail fast if `removable_features` references unknown or illegal columns.

    Every removable column must (a) exist in the dataset's `seq_cols` and
    (b) not be the autoregressive target — dropping the target is nonsensical
    (it is both the model input and the prediction). Checked once, before the
    study starts, so a typo surfaces immediately rather than mid-search.
    """
    known = set(seq_cols)
    for item in removable:
        for col in _as_group(item):
            if col == target_col:
                raise ValueError(
                    f"removable_features may not include the target column "
                    f"{target_col!r} (it is the AR input and the prediction)."
                )
            if col not in known:
                raise ValueError(
                    f"removable_features references {col!r}, which is not in "
                    f"seq_cols={list(seq_cols)}."
                )


def suggest_covariate_selection(
    trial: optuna.Trial,
    removable: Sequence[str | Sequence[str]],
) -> list[str]:
    """Let Optuna decide which removable covariates to drop this trial.

    For each entry in `removable` (a single column, or a group toggled as a
    unit) we sample one boolean `use_<cols>`: True keeps the column(s), False
    drops them. The search space is therefore exactly the covariates the caller
    opted into — nothing is hardcoded, so flags for features the dataset does
    not have can never appear. Returns the flat list of DROPPED column names
    (empty when every removable feature is kept, or when `removable` is empty,
    in which case the feature set is fixed and only model/training HPs vary).
    """
    dropped: list[str] = []
    for item in removable:
        cols = _as_group(item)
        keep = trial.suggest_categorical("use_" + "+".join(cols), [True, False])
        if not keep:
            dropped.extend(cols)
    return dropped


def select_features(data: dict[str, Any], drop_cols: Sequence[str]) -> dict[str, Any]:
    """Return a copy of a `prepare_dataset` output with `drop_cols` removed.

    Feature selection is pure column slicing on the precomputed tensors, so it
    is cheap and deterministic (no re-running data prep per trial). We slice the
    feature axis of `calibration`/`holdout`, rebuild `samples`/`targets` and
    `target_idx` for the reduced layout, and filter `embedded_cols`
    in lockstep — the model validator requires embedded_cols ⊆ seq_cols. All
    other keys (ids, N, T_*, panels, ...) pass through unchanged.

    The same primitive is reused at forecast time: slice `data` to the best
    trial's feature set so the Monte Carlo simulator sees matching `seq_cols`
    and `target_idx`.
    """
    drop = set(drop_cols)
    seq_cols = list(data["seq_cols"])
    target_col = data["target_col"]
    if target_col in drop:
        raise ValueError(f"cannot drop the target column {target_col!r}")
    unknown = drop - set(seq_cols)
    if unknown:
        raise ValueError(f"drop_cols not in seq_cols: {sorted(unknown)}")

    keep = [c for c in seq_cols if c not in drop]
    idx = [seq_cols.index(c) for c in keep]            # feature-axis positions to retain
    target_idx = keep.index(target_col)

    calibration = data["calibration"][:, :, idx]
    holdout = data["holdout"][:, :, idx]

    embedded = data.get("embedded_cols") or {}
    kept_embedded = {c: v for c, v in embedded.items() if c in keep}

    # Keep ar_features in lockstep with the surviving columns. If a trial drops a
    # target-derived AR feature, it must leave this list too — otherwise the
    # Monte Carlo rollout would look it up via seq_cols.index(name) and raise.
    ar_features = [c for c in data.get("ar_features", []) if c in keep]

    out = dict(data)
    out.update(
        calibration=calibration,
        holdout=holdout,
        # samples/targets mirror prepare_dataset: predict step t+1 from step t.
        samples=calibration[:, :-1, :],
        targets=calibration[:, 1:, target_idx:target_idx + 1],
        seq_cols=keep,
        target_idx=target_idx,
        embedded_cols=kept_embedded if kept_embedded else None,
        ar_features=ar_features,
        F=len(keep),
    )
    return out


def select_features_for_trial(
    data: dict[str, Any],
    trial: "optuna.trial.FrozenTrial | optuna.Trial",
) -> dict[str, Any]:
    """Slice a `prepare_dataset` output to the feature set a given trial used.

    `objective` records each trial's dropped columns in the `dropped_features`
    user attribute; this reads them back and applies `select_features`, so the
    write and the read sit in one module and cannot drift. The intended use is
    after a study, to rebuild the winning model and run the forecast on matching
    columns (otherwise the checkpoint — trained on the sliced layout — will not
    load into a full-feature model):

        data_best = select_features_for_trial(data_full, study.best_trial)
        # build the inference model from data_best["seq_cols"]/["embedded_cols"]
        # and pass data_best (not data_full) to the Monte Carlo forecast.

    A trial with no `dropped_features` attribute (e.g. a study run without
    `removable_features`) dropped nothing, so `data` is returned with every
    column intact. `select_features` raises if a recorded column is absent from
    `data` — a guard against pairing a trial with the wrong dataset.
    """
    raw = trial.user_attrs.get("dropped_features", "")
    dropped = raw.split(",") if raw else []
    return select_features(data, dropped)


# ---------------------------------------------------------------------------
# Per-model search spaces
# ---------------------------------------------------------------------------


# Hardcoded fallback search spaces. These are used per-parameter ONLY when the
# caller's `data_info` does not specify that parameter, so the historical
# behaviour (caller passes no search keys) is reproduced exactly. Each value is a
# "spec" in the mini-language `_suggest_param` understands (see its docstring):
#   set            -> categorical over those values
#   (lo, hi)       -> float, uniform
#   (lo, hi,'log') -> float, log scale
#   (lo, hi,'int') -> integer
#   (lo, hi, step) -> float on a step grid
#   scalar         -> fixed (not searched)
LSTM_SEARCH_DEFAULTS: dict[str, Any] = {
    "embedding_dim":    {64, 128, 256},
    "lstm_hidden_size":  {32, 64, 128},
    "dense_units":   {32, 64, 128},
    "dropout":       (0.0, 0.4),
    "learning_rate": (1e-4, 3e-3, "log"),
    "weight_decay":  (1e-6, 1e-2, "log"),
    "batch_size":    {64, 128, 256},
}

TRANSFORMER_SEARCH_DEFAULTS: dict[str, Any] = {
    "d_model":            {32, 64, 128},
    "nhead":              {2, 4, 8},
    "num_encoder_layers": (1, 3, "int"),
    "dropout":            (0.0, 0.4),
    "learning_rate":      (1e-4, 3e-3, "log"),
    "weight_decay":       (1e-6, 1e-2, "log"),
    "batch_size":         {64, 128, 256},
}

# `data_info` keys that are NOT search-space parameters — training control and
# loss/logging settings. `n_epochs`/`patience` are special: they are training
# control, but the caller may still hand them a search spec (e.g. patience over
# {5,7,9}), so they are resolved through `_suggest_param` like a hyperparameter.
# This whitelist is what `validate_data_info` checks against so a typo'd key
# (e.g. "hiddendim") raises up front instead of being silently ignored.
_NON_SEARCH_DATA_INFO_KEYS: frozenset[str] = frozenset({
    "n_epochs", "patience",          # training control (scalar, or a search spec)
    "checkpoint_dir", "verbose",     # bookkeeping
    "loss_type", "class_weights", "focal_gamma",   # loss configuration
    "grad_clip", "log_wandb", "seed",              # optimiser / logging / RNG
})


def _suggest_param(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """Turn one `data_info` spec into a value, sampling from `trial` if needed.

    The spec mini-language lets the caller describe a search dimension (or a
    fixed value) declaratively in the notebook, instead of editing this module:

    - **scalar** (`int`/`float`/`str`/`bool`) -> returned as-is, FIXED. No trial
      parameter is registered, so it never appears in `best_params`.
    - **set / frozenset** -> `suggest_categorical` over the values (sorted for a
      deterministic, reproducible category order).
    - **list** -> `suggest_categorical` in the given order.
    - **tuple** -> a numeric RANGE:
        `(lo, hi)`          float, uniform
        `(lo, hi, "log")`   float, log scale (for LR / weight decay)
        `(lo, hi, "int")`   integer
        `(lo, hi, step)`    float on a step grid (numeric 3rd element)

    A clear `ValueError` is raised for malformed specs (e.g. an empty set or an
    unknown range mode) so mistakes surface immediately, not as a silent default.
    """
    # bool is a subclass of int — check it within the scalar branch so a fixed
    # boolean flag is returned verbatim rather than mis-read as a number.
    if isinstance(spec, (bool, int, float, str)):
        return spec
    if isinstance(spec, (set, frozenset)):
        if not spec:
            raise ValueError(f"{name}: empty set of choices")
        return trial.suggest_categorical(name, sorted(spec))
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"{name}: empty list of choices")
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, tuple):
        if len(spec) == 2:
            low, high = spec
            return trial.suggest_float(name, float(low), float(high))
        if len(spec) == 3:
            low, high, mode = spec
            if mode == "log":
                return trial.suggest_float(name, float(low), float(high), log=True)
            if mode == "int":
                return trial.suggest_int(name, int(low), int(high))
            if isinstance(mode, (int, float)) and not isinstance(mode, bool):
                # numeric 3rd element = grid step
                return trial.suggest_float(name, float(low), float(high), step=float(mode))
            raise ValueError(
                f"{name}: unknown range mode {mode!r}; use 'log', 'int', or a "
                f"numeric step"
            )
        raise ValueError(
            f"{name}: tuple spec must be (lo, hi), (lo, hi, 'log'|'int'), or "
            f"(lo, hi, step); got {spec!r}"
        )
    raise ValueError(
        f"{name}: unsupported spec {spec!r} (type {type(spec).__name__}). Use a "
        f"set/list (categorical), a tuple (range), or a scalar (fixed)."
    )


def _merge_specs(
    defaults: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Per-parameter override: caller's spec wins, else the hardcoded default.

    Only keys present in `defaults` are pulled from `overrides`; non-search
    settings in `data_info` (checkpoint_dir, loss_type, ...) are ignored here.
    """
    overrides = overrides or {}
    return {name: overrides.get(name, default) for name, default in defaults.items()}


def validate_data_info(model_type: str, data_info: dict[str, Any]) -> None:
    """Fail fast on unrecognised `data_info` keys, before any training runs.

    The search space is now driven by `data_info`, so a typo'd hyperparameter
    name (`"hiddendim"`) would otherwise be silently dropped and the default
    range used instead — exactly the kind of silent miss this guard prevents.
    """
    if model_type == "lstm":
        search_keys = set(LSTM_SEARCH_DEFAULTS)
    elif model_type == "transformer":
        search_keys = set(TRANSFORMER_SEARCH_DEFAULTS)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")
    allowed = search_keys | set(_NON_SEARCH_DATA_INFO_KEYS)
    unknown = [k for k in data_info if k not in allowed]
    if unknown:
        raise ValueError(
            f"Unrecognised data_info key(s) for model_type={model_type!r}: "
            f"{sorted(unknown)}. Allowed search params: {sorted(search_keys)}; "
            f"allowed settings: {sorted(_NON_SEARCH_DATA_INFO_KEYS)}."
        )


def suggest_lstm_params(
    trial: optuna.Trial, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Sample the LSTM hyperparameters, honouring `overrides` (from data_info)."""
    specs = _merge_specs(LSTM_SEARCH_DEFAULTS, overrides)
    return {name: _suggest_param(trial, name, spec) for name, spec in specs.items()}


def suggest_transformer_params(
    trial: optuna.Trial, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Sample the Transformer hyperparameters, honouring `overrides`.

    `d_model` and `nhead` are resolved first so the divisibility constraint can
    prune incompatible draws before the remaining params are sampled.
    """
    specs = _merge_specs(TRANSFORMER_SEARCH_DEFAULTS, overrides)
    d_model = _suggest_param(trial, "d_model", specs["d_model"])
    nhead = _suggest_param(trial, "nhead", specs["nhead"])
    if d_model % nhead != 0:
        # Cleaner than narrowing the categorical domain per trial; Optuna's
        # samplers handle pruned trials gracefully.
        raise optuna.TrialPruned(
            f"d_model={d_model} is not divisible by nhead={nhead}"
        )
    params = {"d_model": d_model, "nhead": nhead}
    for name, spec in specs.items():
        if name in ("d_model", "nhead"):
            continue
        params[name] = _suggest_param(trial, name, spec)
    return params


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _build_lstm(
    params: dict[str, Any], metadata: dict[str, Any]
) -> MultinomialLSTMModel:
    return MultinomialLSTMModel(
        seq_cols=metadata["seq_cols"],
        embedded_cols=metadata["embedded_cols"],
        target_col=metadata.get("target_col", "Transactions"),
        embedding_dim=params["embedding_dim"],
        lstm_hidden_size=params["lstm_hidden_size"],
        dense_units=params["dense_units"],
        dropout=params["dropout"],
    )


def _build_transformer(
    params: dict[str, Any], metadata: dict[str, Any]
) -> MultinomialTransformerModel:
    return MultinomialTransformerModel(
        seq_cols=metadata["seq_cols"],
        embedded_cols=metadata["embedded_cols"],
        target_col=metadata.get("target_col", "Transactions"),
        seq_len=metadata.get("seq_len"),
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_encoder_layers=params["num_encoder_layers"],
        dropout=params["dropout"],
    )


# ---------------------------------------------------------------------------
# Rollout-based selection (optional; default objective stays val cross-entropy)
# ---------------------------------------------------------------------------


# The one literal accepted as the rollout selection mode. Anything else (besides
# "val_loss") is a typo and is rejected up front rather than silently ignored.
ROLLOUT_METRIC = "rollout_composite"

_EPS = 1e-8


def weekly_aggregate_rollout_metrics(
    actual: np.ndarray,
    pred_mean: np.ndarray,
    *,
    weight_rmse: float = 1.0,
    weight_mape: float = 0.5,
    weight_bias: float = 0.3,
    mape_clip: float = 300.0,
    min_actual_for_mape: float = 5.0,
) -> dict[str, float]:
    """Weekly-aggregate forecast metrics + a normalized composite score.

    Both inputs are ``(N, V)`` — per-customer counts over the V-step validation
    horizon: ``actual`` are the true pseudo-holdout counts, ``pred_mean`` the
    Monte-Carlo mean. Everything is computed on the WEEKLY AGGREGATE (sum over
    customers per step), matching the thesis's aggregate RMSE / MAPE / bias and
    the ``mape_aggregate_style`` reported elsewhere.

    The composite is normalized by the mean weekly volume so its scale is
    comparable across datasets (a raw RMSE of 30 means something very different
    on a panel averaging 50/week vs 5000/week):

        rmse_norm = rmse / mean_actual
        mape_norm = clipped_masked_mape / 100
        bias_norm = abs(bias_percent) / 100
        score     = w_rmse*rmse_norm + w_mape*mape_norm + w_bias*bias_norm

    Lower is better, so the study stays ``direction="minimize"``.
    """
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred_mean, dtype=np.float64)

    actual_agg = actual.sum(axis=0)          # (V,) true weekly totals
    pred_agg = pred.sum(axis=0)              # (V,) predicted weekly totals

    diff = pred_agg - actual_agg
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # Aggregate bias over the whole horizon: signed total over/under-prediction.
    actual_total = float(actual_agg.sum())
    pred_total = float(pred_agg.sum())
    bias_percent = 100.0 * (pred_total - actual_total) / max(actual_total, _EPS)

    # MAPE only on weeks with enough real volume to be meaningful (a near-zero
    # denominator otherwise explodes), then clip so one bad week can't dominate.
    mask = actual_agg >= min_actual_for_mape
    if mask.any():
        wk_mape = 100.0 * np.abs(diff[mask]) / actual_agg[mask]
        mape = float(min(np.mean(wk_mape), mape_clip))
    else:
        # No week clears the threshold — MAPE is undefined; fall back to the clip
        # so the composite still has a finite, bounded MAPE term.
        mape = float(mape_clip)

    mean_actual = max(float(actual_agg.mean()), _EPS)
    rmse_norm = rmse / mean_actual
    mape_norm = mape / 100.0
    bias_norm = abs(bias_percent) / 100.0
    score = (
        weight_rmse * rmse_norm
        + weight_mape * mape_norm
        + weight_bias * bias_norm
    )

    return {
        "rollout_rmse": rmse,
        "rollout_mae": mae,
        "rollout_mape": mape,
        "rollout_bias_percent": bias_percent,
        "rollout_score": float(score),
    }


def _validation_rollout_score(
    *,
    model_type: str,
    params: dict[str, Any],
    drop_cols: Sequence[str],
    checkpoint_path: str,
    rollout_data: dict[str, Any],
    horizon: int,
    n_simulations: int,
    seed: int,
    device: str | torch.device | None,
    metric_kwargs: dict[str, float],
) -> dict[str, float]:
    """Evaluate one trained trial with a validation-horizon MC rollout.

    Leak-free pseudo-holdout: the real ``data["holdout"]`` is never touched.
    Instead we carve the last ``horizon`` weeks off the CALIBRATION window — the
    temporal validation window (after ``validation_start``) — for ALL customers and
    treat them as a holdout the trial has not been selected on:

        calib_prefix  = calibration[:, :-horizon]   # warm-up context
        pseudo_holdout= calibration[:, -horizon:]    # scored target

    The split is temporal, not customer-wise, so every customer contributes both a
    warm-up prefix and a scored suffix (matching how the teacher-forced ``val_loss``
    path scores the same window).

    Because each trial may have dropped a different covariate subset, we re-slice
    ``rollout_data`` with this trial's ``drop_cols`` first (so F and seq_cols
    match the checkpoint), build the matching INFERENCE model in sampling mode,
    load the trial's best weights, and reuse the existing MC forecaster.

    This is a faithful proxy for the final autoregressive evaluation in the
    things that drive selection — sampling-drift and seasonality under the
    model's own fed-back samples — but it is deliberately NOT identical to it:
    the pseudo-holdout sits INSIDE calibration, so any known-future covariate
    keeps in-range values here, whereas the real holdout may extrapolate beyond
    the training range (e.g. a trend index taking an unseen value). Capturing
    that would require reading holdout-period covariates, which is exactly the
    leak we refuse. So selection tracks rollout quality without peeking ahead.
    """
    d = select_features(rollout_data, list(drop_cols))
    seq_cols = d["seq_cols"]
    embedded_cols = d["embedded_cols"]
    target_col = d.get("target_col", "Transactions")

    calib = np.asarray(d["calibration"])                  # (N, T_CAL, F) — all customers
    T_CAL = calib.shape[1]
    if not 0 < horizon < T_CAL:
        raise ValueError(
            f"rollout_horizon={horizon} must be in (0, T_CAL={T_CAL}); the "
            f"calibration window is too short to carve a pseudo-holdout."
        )

    calib_prefix = calib[:, :-horizon, :]                  # warm-up
    pseudo_holdout = calib[:, -horizon:, :]                # scored target

    # Minimal data dict shaped like prepare_dataset's output, just enough for the
    # MC forecaster (it reads calibration / holdout / seq_cols / target_col /
    # ar_features). actual targets are extracted by the forecaster from holdout.
    # ar_features comes from the SLICED dict `d`, so a trial that dropped an AR
    # column doesn't leave it dangling here (select_features filters it).
    roll_data = {
        "calibration": calib_prefix,
        "holdout": pseudo_holdout,
        "seq_cols": seq_cols,
        "target_col": target_col,
        "ar_features": list(d.get("ar_features", [])),
    }

    if model_type == "lstm":
        model = InferenceMultinomialLSTMModel(
            seq_cols=seq_cols, embedded_cols=embedded_cols, target_col=target_col,
            embedding_dim=params["embedding_dim"], lstm_hidden_size=params["lstm_hidden_size"],
            dense_units=params["dense_units"], dropout=params["dropout"],
        )
        forecaster = run_monte_carlo_forecast
    else:
        model = InferenceMultinomialTransformerModel(
            seq_cols=seq_cols, embedded_cols=embedded_cols, target_col=target_col,
            d_model=params["d_model"], nhead=params["nhead"],
            num_encoder_layers=params["num_encoder_layers"],
            dropout=params["dropout"],
        )
        forecaster = run_monte_carlo_forecast_transformer

    state = torch.load(checkpoint_path, map_location="cpu")
    # Transformer training caches a fixed-length mask buffer the inference model
    # regenerates on the fly; drop it so strict load_state_dict succeeds.
    state.pop("_cached_mask", None)
    model.load_state_dict(state)
    model.eval()

    forecast = forecaster(
        model, roll_data, n_simulations=n_simulations, device=device, seed=seed,
    )
    return weekly_aggregate_rollout_metrics(
        forecast["actual"], forecast["prediction_mean"], **metric_kwargs,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial,
    model_type: str,
    data_builder: DataBuilder,
    data_info: dict[str, Any],
    device: str | torch.device | None = None,
    removable_features: Sequence[str | Sequence[str]] = (),
    selection_metric: str = "val_loss",
    rollout_cfg: dict[str, Any] | None = None,
) -> float:
    """Objective: validation cross-entropy, or an autoregressive rollout score.

    `data_info` carries BOTH the search-space overrides (per-parameter specs in
    the `_suggest_param` mini-language — set=categorical, tuple=range, scalar=
    fixed; anything omitted falls back to the model's hardcoded default range)
    and the non-search settings (checkpoint dir, loss config, ...). Its keys are
    validated up front by `validate_data_info`. `removable_features`
    lists covariates Optuna may drop this trial (see `suggest_covariate_selection`);
    the chosen drop-set is handed to `data_builder` as `feature_config`.

    `selection_metric` decides what is RETURNED to Optuna (the training loop
    always optimises cross-entropy regardless):

        "val_loss"          — teacher-forced validation cross-entropy (default,
                              unchanged historical behaviour).
        "rollout_composite" — after training, run a validation-horizon Monte
                              Carlo rollout (leak-free pseudo-holdout carved from
                              the calibration tail) and return its normalized
                              composite score. Cross-entropy is still logged as
                              the `val_loss` user attribute. Requires
                              `rollout_cfg` (assembled by `run_optuna_study`).
    """
    if model_type == "lstm":
        params = suggest_lstm_params(trial, data_info)
    elif model_type == "transformer":
        params = suggest_transformer_params(trial, data_info)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")

    # Which covariates to drop this trial (empty list ⇒ fixed feature set).
    drop_cols = suggest_covariate_selection(trial, removable_features)

    train_loader, val_loader, metadata = data_builder(
        feature_config=drop_cols,
        batch_size=params["batch_size"],
    )

    # Record the actual feature set so trials.csv / best.json are self-documenting.
    trial.set_user_attr("selected_features", ",".join(metadata["seq_cols"]))
    trial.set_user_attr("dropped_features", ",".join(sorted(drop_cols)))
    trial.set_user_attr("target_col", metadata.get("target_col", "Transactions"))

    model = (_build_lstm if model_type == "lstm" else _build_transformer)(params, metadata)

    # `focal_gamma` is either a scalar (fixed) or `(low, high, step)`
    # (Optuna-tuned on a step grid). Missing / None → 2.0.
    loss_type = data_info.get("loss_type", "cross_entropy")
    focal_gamma_spec = data_info.get("focal_gamma", 2.0)
    if loss_type == "focal" and isinstance(focal_gamma_spec, (tuple, list)):
        low, high, step = focal_gamma_spec
        focal_gamma = trial.suggest_float(
            "focal_gamma", float(low), float(high), step=float(step),
        )
    elif isinstance(focal_gamma_spec, (int, float)):
        focal_gamma = float(focal_gamma_spec)
    else:
        focal_gamma = 2.0

    result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_trans=model.num_target_classes,
        # n_epochs / patience are training control, but the caller may still hand
        # them a search spec (e.g. patience over {5,7,9}); resolve through the same
        # mini-language so a scalar stays fixed and a set/tuple is searched.
        n_epochs=_suggest_param(trial, "n_epochs", data_info.get("n_epochs", 50)),
        patience=_suggest_param(trial, "patience", data_info.get("patience", 5)),
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        grad_clip=data_info.get("grad_clip", 1.0),
        device=device,
        checkpoint_dir=data_info.get("checkpoint_dir", "./checkpoints"),
        model_name=f"{model_type}_trial_{trial.number}",
        trial=trial,
        log_wandb=data_info.get("log_wandb", False),
        verbose=data_info.get("verbose", False),
        loss_type=loss_type,
        class_weights=data_info.get("class_weights"),
        focal_gamma=focal_gamma,
        # Temporal split: score CE only on the validation suffix (periods after
        # validation_start). make_loaders puts this in metadata; 0 ⇒ score all steps.
        val_score_start=metadata.get("val_score_start", 0),
    )
    trial.set_user_attr("checkpoint_path", str(result.checkpoint_path))
    trial.set_user_attr("best_epoch", result.best_epoch)
    trial.set_user_attr("best_val_f1", result.best_val_f1)
    # Always log cross-entropy, so it stays inspectable even when it is no longer
    # the selection criterion.
    trial.set_user_attr("val_loss", float(result.best_val_loss))
    trial.set_user_attr("selection_metric", selection_metric)

    if selection_metric == "val_loss":
        return result.best_val_loss

    # selection_metric == ROLLOUT_METRIC: score this trained trial with a
    # validation-horizon autoregressive rollout and return the composite.
    # NOTE: the pruner (MedianPruner) still acts on the per-epoch CROSS-ENTROPY
    # fit_model reports, not on this rollout score — so it only prunes clearly
    # bad-CE trials early. A trial that survives training is always rolled out
    # and scored here; the two metrics are intentionally kept on separate jobs
    # (cheap per-epoch pruning vs one rollout at the end) to stay simple.
    metrics = _validation_rollout_score(
        model_type=model_type,
        params=params,
        drop_cols=drop_cols,
        checkpoint_path=str(result.checkpoint_path),
        rollout_data=rollout_cfg["rollout_data"],
        horizon=rollout_cfg["horizon"],
        n_simulations=rollout_cfg["n_simulations"],
        seed=rollout_cfg["seed"],
        device=device,
        metric_kwargs=rollout_cfg["metric_kwargs"],
    )
    for key, val in metrics.items():
        trial.set_user_attr(key, float(val))
    trial.set_user_attr("rollout_horizon", rollout_cfg["horizon"])
    trial.set_user_attr("rollout_n_simulations", rollout_cfg["n_simulations"])
    return metrics["rollout_score"]


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------


def run_optuna_study(
    model_type: str,
    data_builder: DataBuilder,
    data_info: dict[str, Any],
    device: str | torch.device | None = None,
    n_trials: int = 50,
    study_name: str | None = None,
    storage: str | None = None,
    direction: str = "minimize",
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | bool | None = True,
    summary_dir: str | Path = "./optuna_summaries",
    append_timestamp: bool = True,
    removable_features: Sequence[str | Sequence[str]] = (),
    selection_metric: str = "val_loss",
    rollout_data: dict[str, Any] | None = None,
    rollout_horizon: int | None = None,
    rollout_n_simulations: int = 100,
    rollout_seed: int = 42,
    rollout_mape_clip: float = 300.0,
    rollout_min_actual_for_mape: float = 5.0,
    rollout_weight_rmse: float = 1.0,
    rollout_weight_mape: float = 0.5,
    rollout_weight_bias: float = 0.3,
    keep_only_best_checkpoint: bool = False,
) -> optuna.Study:
    """Runs an Optuna study and saves a JSON / CSV summary of all trials.

    By default the objective is validation cross-entropy (lower is better).
    Use the returned study to inspect `study.best_trial` and the saved
    checkpoint path stored as a user attribute on each trial.

    Rollout-based selection
    -----------------------
    The default `selection_metric="val_loss"` is teacher-forced next-step
    cross-entropy — convenient and cheap, but blind to the autoregressive
    sampling rollout the final forecast actually uses, so it can pick feature
    sets / architectures that drift badly at forecast time. Pass
    `selection_metric="rollout_composite"` to instead select on a validation
    Monte-Carlo rollout that mirrors the real forecasting regime:

        - The temporal validation window (the last `n_val_periods` weeks of the
          CALIBRATION window, i.e. everything after `validation_start`) is carved
          off as a leak-free pseudo-holdout for ALL customers (the real
          `data["holdout"]` is never used in tuning). `rollout_horizon` defaults
          to that window; pass an int to override it.
        - Each trained trial is warmed up on the prefix and autoregressively
          rolls the pseudo-holdout (`rollout_n_simulations` paths), then scored
          by `weekly_aggregate_rollout_metrics` (normalized RMSE + MAPE + bias).
        - Cross-entropy (over the same validation window) is still logged as the
          `val_loss` user attr.

    This mode requires `rollout_data` (the full `prepare_dataset` dict — the
    objective re-slices it per trial's feature subset; its `n_val_periods` sets the
    default horizon). Because the returned score is on a different scale than
    cross-entropy, a rollout run must use its OWN fresh study (don't point it at a
    `val_loss` study's storage).

    When `append_timestamp` is True (default) the effective run name is
    `f"{study_name}_{YYYYMMDD_HHMM}"`; that name is used for the Optuna study,
    a per-run checkpoint subfolder under `data_info["checkpoint_dir"]`, and the
    summary files, so separate runs never overwrite each other. The resolved
    name is available afterwards as `study.study_name`. Pass False to keep a
    stable name (e.g. to resume via `storage=`).

    `pruner` controls early stopping of unpromising trials. `True` (default) uses
    the standard `MedianPruner` on the per-epoch cross-entropy `fit_model` reports;
    `False` disables pruning (`NopPruner`) so every trial trains fully. Prefer
    `False` with `selection_metric="rollout_composite"`: the pruner acts on CE,
    not the rollout score, so leaving it on can cut a trial before it is rolled
    out and scored (and bias the sampler toward the low-CE region the rollout
    metric is meant to look past). You may also pass a concrete `optuna` pruner
    instance for full control; it is used as-is.

    `removable_features` lists the covariates Optuna is allowed to drop. Each
    entry is a column name (its own toggle) or a group of names toggled together
    (e.g. `("week_sin", "week_cos")` for a cyclical pair). Columns not listed are
    always included; the target is never removable. Leave it empty (default) to
    keep the feature set fixed and tune only model/training hyperparameters. The
    chosen feature set per trial is recorded in the `selected_features` /
    `dropped_features` user attributes (so the summary CSV/JSON is self-documenting).

    `keep_only_best_checkpoint` (default False) trades inspectability for disk:
    every trial writes a `.pth` (one per trial, they accumulate fast over a long
    study), but only the best trial's checkpoint is needed afterwards
    (`build_inference_from_trial` / `refit_best_trial` both load the best trial).
    Set it True to delete all non-best trial checkpoints once the study completes
    and the summary is written. The best trial's file (and its recorded
    `checkpoint_path`) is preserved, so the downstream workflow is unaffected; you
    only lose the ability to rebuild a NON-winning trial from its weights.
    """
    if model_type not in {"lstm", "transformer"}:
        raise ValueError(f"model_type must be 'lstm' or 'transformer', got {model_type!r}")

    # Resolve the selection mode up front: a typo'd metric should fail loudly
    # here, not silently fall back to cross-entropy after hours of tuning.
    if selection_metric not in {"val_loss", ROLLOUT_METRIC}:
        raise ValueError(
            f"selection_metric must be 'val_loss' or {ROLLOUT_METRIC!r}, "
            f"got {selection_metric!r}"
        )
    rollout_cfg: dict[str, Any] | None = None
    if selection_metric == ROLLOUT_METRIC:
        if rollout_data is None:
            raise ValueError(
                f"selection_metric={ROLLOUT_METRIC!r} requires rollout_data "
                "(the prepare_dataset dict)."
            )
        # The rollout scores the SAME temporal validation window the teacher-forced
        # path uses: the last `n_val_periods` weeks of calibration (= the window after
        # validation_start), for ALL customers. By default the horizon IS that window,
        # so the two selection metrics stay comparable; an explicit `rollout_horizon`
        # overrides it (e.g. to probe a different carve), still leak-free since the
        # real `data["holdout"]` is never read.
        T_CAL = int(np.asarray(rollout_data["calibration"]).shape[1])
        horizon = (
            int(rollout_horizon)
            if rollout_horizon is not None
            else int(rollout_data["n_val_periods"])
        )
        # Validate the horizon UP FRONT (before any training) so a misconfigured
        # rollout fails in seconds, not after the first trial finishes. The
        # pseudo-holdout is the last `horizon` periods of calibration, so the horizon
        # must leave a non-empty warm-up prefix: 0 < horizon < T_CAL.
        if not 0 < horizon < T_CAL:
            raise ValueError(
                f"rollout horizon={horizon} must be in (0, T_CAL={T_CAL}): "
                f"the validation rollout is carved from the calibration window, so "
                f"it cannot be >= the full calibration length (no warm-up would "
                f"remain). Pick a horizon well below {T_CAL} (e.g. <= {T_CAL // 2})."
            )
        # Even when valid, a horizon that consumes most of calibration leaves too
        # little warm-up for the model to represent each customer's history, so the
        # rollout score becomes unreliable. Warn (don't error) past the halfway mark.
        warmup = T_CAL - horizon
        if warmup < horizon:
            warnings.warn(
                f"rollout horizon={horizon} leaves only {warmup} warm-up "
                f"period(s) of {T_CAL} (shorter than the scored horizon). The "
                f"validation rollout metric may be unreliable; consider a horizon "
                f"<= {T_CAL // 2} (move validation_start earlier).",
                stacklevel=2,
            )
        rollout_cfg = {
            "rollout_data": rollout_data,
            "horizon": horizon,
            "n_simulations": int(rollout_n_simulations),
            "seed": int(rollout_seed),
            "metric_kwargs": {
                "weight_rmse": rollout_weight_rmse,
                "weight_mape": rollout_weight_mape,
                "weight_bias": rollout_weight_bias,
                "mape_clip": rollout_mape_clip,
                "min_actual_for_mape": rollout_min_actual_for_mape,
            },
        }

    # Validate data_info keys once, up front: the search space is now driven by
    # data_info, so a typo'd hyperparameter name must raise here rather than be
    # silently ignored (which would quietly fall back to the default range).
    validate_data_info(model_type, data_info)

    # Validate removable_features once, before any training. We probe the
    # data_builder with an empty drop-set (keep everything) purely to learn the
    # real column layout, then check every removable name against it — a typo
    # raises here instead of midway through the search.
    if removable_features:
        _, _, _probe_meta = data_builder(feature_config=[], batch_size=1)
        validate_removable_features(
            removable_features,
            _probe_meta["seq_cols"],
            _probe_meta.get("target_col", "Transactions"),
        )

    if study_name is None:
        study_name = f"{model_type}_multinomial"
    # Make every run unique: STUDY_NAME_YYYYMMDD_HHMM. This keeps separate runs
    # (e.g. cross_entropy vs focal) from clobbering each other's checkpoints and
    # summary files. Pass append_timestamp=False to keep a stable name when
    # resuming a study via `storage=`.
    run_name = (
        f"{study_name}_{datetime.now():%Y%m%d_%H%M}"
        if append_timestamp else study_name
    )
    # Isolate this run's checkpoints in a per-run subfolder so trial-number
    # filenames never overwrite a previous study's. Copy data_info rather than
    # mutating the caller's dict. `fit_model` mkdir's the dir, so no setup here.
    base_ckpt = Path(data_info.get("checkpoint_dir", "./checkpoints"))
    data_info = {**data_info, "checkpoint_dir": str(base_ckpt / run_name)}

    # Resolve the pruner. `True` (default) / `None` keep the historical
    # early-stopping behaviour (MedianPruner on the per-epoch CE fit_model
    # reports); `False` disables pruning entirely (NopPruner). A concrete
    # BasePruner instance is honoured as-is. Disabling is recommended for
    # selection_metric="rollout_composite": there the pruner acts on CE, not the
    # rollout score, so it can cut a trial before it is ever rolled out and
    # scored — see the module docstring.
    if pruner is True or pruner is None:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    elif pruner is False:
        pruner = optuna.pruners.NopPruner()
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=data_info.get("seed", 42))

    study = optuna.create_study(
        study_name=run_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=storage is not None,
    )

    study.optimize(
        lambda trial: objective(
            trial, model_type, data_builder, data_info, device,
            removable_features=removable_features,
            selection_metric=selection_metric,
            rollout_cfg=rollout_cfg,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df.to_csv(summary_dir / f"{run_name}_trials.csv", index=False)

    best = study.best_trial
    summary = {
        "study_name": run_name,
        "model_type": model_type,
        "selection_metric": selection_metric,
        # best.value is whatever the objective returned: cross-entropy for
        # "val_loss", the composite rollout score for "rollout_composite".
        "best_objective_value": best.value,
        "best_params": best.params,
        "best_user_attrs": dict(best.user_attrs),
        "n_trials": len(study.trials),
    }
    with open(summary_dir / f"{run_name}_best.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Optional disk cleanup. Each completed trial recorded its checkpoint under the
    # "checkpoint_path" user attr; the downstream rebuild (build_inference_from_trial
    # / refit_best_trial) only ever reloads the BEST trial, so once the summary above
    # is written every other trial's .pth is dead weight. Delete them when asked.
    if keep_only_best_checkpoint:
        best_ckpt = best.user_attrs.get("checkpoint_path")
        if best_ckpt:
            best_ckpt = str(Path(best_ckpt))
            removed = 0
            for trial in study.trials:
                path = trial.user_attrs.get("checkpoint_path")
                # Skip trials with no checkpoint (pruned/failed before torch.save)
                # and, crucially, the winning checkpoint itself.
                if not path or str(Path(path)) == best_ckpt:
                    continue
                try:
                    Path(path).unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    pass  # never let cleanup crash an otherwise-finished study
            if data_info.get("verbose"):
                print(
                    f"[run_optuna_study] keep_only_best_checkpoint: removed "
                    f"{removed} non-best trial checkpoint(s); kept {best_ckpt}"
                )
        else:
            # No recorded winner path → deleting "non-best" files could remove the
            # one we must keep. Skip rather than risk it.
            warnings.warn(
                "keep_only_best_checkpoint=True but the best trial has no recorded "
                "checkpoint_path; skipping checkpoint cleanup to avoid deleting the "
                "winning checkpoint.",
                stacklevel=2,
            )

    return study
