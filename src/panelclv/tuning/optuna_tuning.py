"""Optuna tuning for the multinomial LSTM / Transformer baselines.

One file, two search spaces (`suggest_lstm_params` / `suggest_transformer_params`),
one shared `objective`. Each trial samples an architecture + training HPs (and,
optionally, a covariate subset), trains via `training_utils.fit_model` — which
optimises classification cross-entropy and owns the loss curve, early stopping,
and per-epoch pruning reports — then returns that same cross-entropy, scored on
the temporal validation window (ADR-0001), to Optuna. Selection and training
therefore minimise one number.

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
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import optuna
import torch

# Model definitions and the Monte Carlo simulator live in `panelclv.models`; the
# training loop lives in `panelclv.training`. After the subpackage split these are
# cross-package imports, so they are absolute rather than relative.
from panelclv.models.embedders import ProjectedEmbedder
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
    `target_idx` for the reduced layout, and filter `embedded_cols`,
    `ar_features` and `covariate_stats` in lockstep — the model validator requires
    embedded_cols ⊆ seq_cols. All other keys (ids, N, T_*, panels, ...) pass
    through unchanged. Note the tensors are sliced, never recomputed, so the
    standardization `prepare_dataset` already applied is preserved as-is.

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

    # Standardization stats are keyed by column name, so they need no reindexing —
    # just drop the entries for columns this trial removed, keeping the dict an
    # accurate description of the sliced layout.
    stats = data.get("covariate_stats") or {}
    kept_stats = {c: v for c, v in stats.items() if c in keep}

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
        covariate_stats=kept_stats,
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

# The Valendin benchmark's architecture is FROZEN (ADR-0004): its widths are the
# published `memory_units = 128` / `dense_units = 128`, and its embeddings are raw
# sqrt(n)+1 vectors, so none of them appear here. Only training hyperparameters are
# searched — Optuna over fixed sizes is the deliberate departure ADR-0004 records, and
# searching a width would quietly unfreeze the reference implementation.
VALENDIN_SEARCH_DEFAULTS: dict[str, Any] = {
    "learning_rate": (1e-4, 3e-3, "log"),
    "weight_decay":  (1e-6, 1e-2, "log"),
    "batch_size":    {64, 128, 256},
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


# Search space per model type. The four dispatch sites below read this rather than
# each carrying its own `if model_type == ...` cascade, so a type registered here is
# recognised by all of them or by none — never by some.
_SEARCH_DEFAULTS: dict[str, dict[str, Any]] = {
    "lstm": LSTM_SEARCH_DEFAULTS,
    "transformer": TRANSFORMER_SEARCH_DEFAULTS,
    "valendin_lstm": VALENDIN_SEARCH_DEFAULTS,
}


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
    if model_type not in _SEARCH_DEFAULTS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; "
            f"registered types: {sorted(_SEARCH_DEFAULTS)}"
        )
    search_keys = set(_SEARCH_DEFAULTS[model_type])
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


def suggest_valendin_params(
    trial: optuna.Trial, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Sample the Valendin benchmark's TRAINING hyperparameters only.

    Its architecture is frozen at the published sizes (ADR-0004), so there is nothing
    architectural to search — `VALENDIN_SEARCH_DEFAULTS` carries learning rate, weight
    decay and batch size and nothing else.
    """
    specs = _merge_specs(VALENDIN_SEARCH_DEFAULTS, overrides)
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
        embedder=ProjectedEmbedder(
            seq_cols=metadata["seq_cols"],
            embedded_cols=metadata["embedded_cols"],
            target_col=metadata.get("target_col", "Transactions"),
            embedding_dim=params["embedding_dim"],
        ),
        lstm_hidden_size=params["lstm_hidden_size"],
        dense_units=params["dense_units"],
        dropout=params["dropout"],
    )


def _build_valendin(
    params: dict[str, Any], metadata: dict[str, Any]
) -> "ValendinLSTMModel":
    """Build the frozen benchmark. `params` carries no architecture, by design."""
    from panelclv.benchmarks.valendin_lstm import ValendinLSTMModel

    return ValendinLSTMModel(
        seq_cols=metadata["seq_cols"],
        embedded_cols=metadata["embedded_cols"],
        target_col=metadata.get("target_col", "Transactions"),
    )


def _build_transformer(
    params: dict[str, Any], metadata: dict[str, Any]
) -> MultinomialTransformerModel:
    return MultinomialTransformerModel(
        # The Transformer projects the embedder's width onto d_model, and has always
        # embedded at d_model, so that is the width the ProjectedEmbedder uses.
        embedder=ProjectedEmbedder(
            seq_cols=metadata["seq_cols"],
            embedded_cols=metadata["embedded_cols"],
            target_col=metadata.get("target_col", "Transactions"),
            embedding_dim=params["d_model"],
        ),
        seq_len=metadata.get("seq_len"),
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_encoder_layers=params["num_encoder_layers"],
        dropout=params["dropout"],
    )


# Parameter suggester and training-model builder per registered type. Every dispatch
# site goes through these two helpers, so a type is either wired everywhere or nowhere.
# Before this table two sites fell through to the Transformer on an unrecognised type,
# which would have trained the wrong architecture under the right name.
_SUGGESTERS = {
    "lstm": suggest_lstm_params,
    "transformer": suggest_transformer_params,
    "valendin_lstm": suggest_valendin_params,
}

_BUILDERS = {
    "lstm": _build_lstm,
    "transformer": _build_transformer,
    "valendin_lstm": _build_valendin,
}


def _require_registered(model_type: str, table: dict, what: str) -> Any:
    """Look `model_type` up, or say plainly that it is not registered."""
    try:
        return table[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model_type {model_type!r} — no {what} registered. "
            f"Registered types: {sorted(table)}"
        ) from None


def _suggest_params_for(
    model_type: str, trial: optuna.Trial, data_info: dict[str, Any]
) -> dict[str, Any]:
    """Sample this model type's hyperparameters."""
    return _require_registered(model_type, _SUGGESTERS, "parameter suggester")(
        trial, data_info
    )


def _build_model_for(
    model_type: str, params: dict[str, Any], metadata: dict[str, Any]
):
    """Build this model type's TRAINING model."""
    return _require_registered(model_type, _BUILDERS, "model builder")(params, metadata)


def _build_inference_model_for(
    model_type: str, params: dict[str, Any], metadata: dict[str, Any]
):
    """Build the matching INFERENCE model and the simulator that drives it.

    Returns ``(inference_model, forecaster)``. Constructor arguments mirror the
    training model's, since the rollout loads that model's ``state_dict`` into this one.
    """
    seq_cols = metadata["seq_cols"]
    embedded_cols = metadata["embedded_cols"]
    target_col = metadata.get("target_col", "Transactions")

    if model_type == "lstm":
        return (
            InferenceMultinomialLSTMModel(
                embedder=ProjectedEmbedder(
                    seq_cols=seq_cols, embedded_cols=embedded_cols,
                    target_col=target_col, embedding_dim=params["embedding_dim"],
                ),
                lstm_hidden_size=params["lstm_hidden_size"],
                dense_units=params["dense_units"], dropout=params["dropout"],
            ),
            run_monte_carlo_forecast,
        )
    if model_type == "transformer":
        return (
            InferenceMultinomialTransformerModel(
                embedder=ProjectedEmbedder(
                    seq_cols=seq_cols, embedded_cols=embedded_cols,
                    target_col=target_col, embedding_dim=params["d_model"],
                ),
                d_model=params["d_model"], nhead=params["nhead"],
                num_encoder_layers=params["num_encoder_layers"],
                dropout=params["dropout"],
            ),
            run_monte_carlo_forecast_transformer,
        )
    if model_type == "valendin_lstm":
        from panelclv.benchmarks.valendin_lstm import InferenceValendinLSTMModel

        # Frozen architecture, so no params are read: the sizes are the published ones.
        return (
            InferenceValendinLSTMModel(
                seq_cols=seq_cols, embedded_cols=embedded_cols, target_col=target_col,
            ),
            run_monte_carlo_forecast,   # stateful rollout, as for the LSTM
        )
    raise ValueError(
        f"Unknown model_type {model_type!r} — no inference model registered. "
        f"Registered types: {sorted(_BUILDERS)}"
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
) -> float:
    """Objective: teacher-forced validation cross-entropy.

    `data_info` carries BOTH the search-space overrides (per-parameter specs in
    the `_suggest_param` mini-language — set=categorical, tuple=range, scalar=
    fixed; anything omitted falls back to the model's hardcoded default range)
    and the non-search settings (checkpoint dir, loss config, ...). Its keys are
    validated up front by `validate_data_info`. `removable_features`
    lists covariates Optuna may drop this trial (see `suggest_covariate_selection`);
    the chosen drop-set is handed to `data_builder` as `feature_config`.

    What is RETURNED to Optuna is the cross-entropy the training loop already
    minimises, scored on the temporal validation window only (ADR-0001), so
    selection and training agree on the number they are looking at.
    """
    params = _suggest_params_for(model_type, trial, data_info)

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

    model = _build_model_for(model_type, params, metadata)

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
    # Logged as well as returned, so a trial's score is readable straight off
    # the summary CSV without joining back to the Optuna value column.
    trial.set_user_attr("val_loss", float(result.best_val_loss))

    return result.best_val_loss


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
    keep_only_best_checkpoint: bool = False,
) -> optuna.Study:
    """Runs an Optuna study and saves a JSON / CSV summary of all trials.

    The objective is validation cross-entropy (lower is better), scored on the
    temporal validation window (ADR-0001). Use the returned study to inspect
    `study.best_trial` and the saved checkpoint path stored as a user attribute
    on each trial.

    When `append_timestamp` is True (default) the effective run name is
    `f"{study_name}_{YYYYMMDD_HHMM}"`; that name is used for the Optuna study,
    a per-run checkpoint subfolder under `data_info["checkpoint_dir"]`, and the
    summary files, so separate runs never overwrite each other. The resolved
    name is available afterwards as `study.study_name`. Pass False to keep a
    stable name (e.g. to resume via `storage=`).

    `pruner` controls early stopping of unpromising trials. `True` (default) uses
    the standard `MedianPruner` on the per-epoch cross-entropy `fit_model` reports;
    `False` disables pruning (`NopPruner`) so every trial trains fully. You may
    also pass a concrete `optuna` pruner instance for full control; it is used
    as-is.

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
    # Reject an unregistered type here rather than after the first trial trains.
    if model_type not in _BUILDERS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; "
            f"registered types: {sorted(_BUILDERS)}"
        )

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
    # BasePruner instance is honoured as-is.
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
        # best.value is what the objective returned: validation cross-entropy.
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
