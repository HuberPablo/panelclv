"""The one table that declares every model this package knows (ADR-0006).

Adding a model means adding one entry to ``MODEL_REGISTRY``. Everything that used
to be a separate enumeration of the model set — the valid-types list, the neural
list, the per-model search defaults, the suggesters, the builders, the suite's
forecaster map — is now read off this table, so a type is registered everywhere or
nowhere. There is no state in which the tuner knows a model and the forecaster does
not.

**"Neural" is a predicate, not a list.** ``is_neural`` answers "does this entry have
a training builder", because the second list is the copy that already drifted once:
a stale one in the archive reader classified the Valendin benchmark as a single
deterministic fit and silently collapsed its across-study spread to one study.

**Fields are optional** so ``pareto_nbd`` sits in the table beside the neural
families rather than being a hand-written addend to every enumeration. Its entry is
purely declarative: it has no search space, no builder and no rollout, because it is
a closed-form MCMC fit that never trains and never simulates. The suite runner keeps
a separate deterministic path for it.

**Entries hold direct references.** An earlier design used dotted paths so a
``model_type`` could be validated without importing torch; torch is a hard
dependency and that goal was dropped, so the indirection bought nothing.

Why its own subpackage: the table must name benchmark classes, and
``benchmarks/valendin_lstm.py`` already imports ``models.embedders`` — putting the
table in ``models/`` would close that loop. ``studies/`` is blocked the other way,
since ``tuning`` is what needs the registry while ``studies`` already imports
``tuning``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import optuna

from panelclv.benchmarks.valendin_lstm import ValendinLSTMModel
from panelclv.models.embedders import ProjectedEmbedder
from panelclv.models.monte_carlo_forecasting import (
    run_monte_carlo_forecast,
    run_monte_carlo_forecast_transformer,
)
from panelclv.models.multinomial_lstm import MultinomialLSTMModel
from panelclv.models.multinomial_transformer import MultinomialTransformerModel


# ---------------------------------------------------------------------------
# The search-space mini-language
# ---------------------------------------------------------------------------


def suggest_param(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """Turn one search spec into a value, sampling from `trial` if needed.

    The spec mini-language lets a caller describe a search dimension (or a fixed
    value) declaratively in a notebook, instead of editing this module:

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
    """Per-parameter override: caller's spec wins, else the entry's default.

    Only keys the entry declares are pulled from `overrides`, and in the entry's
    order — which is the order Optuna registers the parameters in.
    """
    overrides = overrides or {}
    return {name: overrides.get(name, default) for name, default in defaults.items()}


# ---------------------------------------------------------------------------
# Suggesters — how an entry's search space is sampled
# ---------------------------------------------------------------------------


def _suggest_every_param(
    trial: optuna.Trial, specs: dict[str, Any]
) -> dict[str, Any]:
    """Sample every declared parameter, in declaration order. The common case."""
    return {name: suggest_param(trial, name, spec) for name, spec in specs.items()}


def _suggest_transformer_params(
    trial: optuna.Trial, specs: dict[str, Any]
) -> dict[str, Any]:
    """Sample the Transformer's parameters, pruning indivisible (d_model, nhead).

    Multi-head attention splits `d_model` across `nhead` heads, so the two are not
    independent. They are resolved FIRST and the trial is pruned on a bad pair
    before anything else is sampled — cleaner than narrowing the categorical domain
    per trial, and it keeps the pruned trial from registering parameters it never
    used.
    """
    d_model = suggest_param(trial, "d_model", specs["d_model"])
    nhead = suggest_param(trial, "nhead", specs["nhead"])
    if d_model % nhead != 0:
        # Optuna's samplers handle pruned trials gracefully.
        raise optuna.TrialPruned(
            f"d_model={d_model} is not divisible by nhead={nhead}"
        )
    params = {"d_model": d_model, "nhead": nhead}
    for name, spec in specs.items():
        if name in ("d_model", "nhead"):
            continue
        params[name] = suggest_param(trial, name, spec)
    return params


# ---------------------------------------------------------------------------
# Builders — how an entry's training model is constructed from sampled params
# ---------------------------------------------------------------------------
#
# `recipe` is what the calibration split hands over (`trials.CalibrationSplit`):
# `seq_cols` (matching the input tensor's last axis), `embedded_cols`
# ({col: cardinality}), `target_col`, and optionally `seq_len`.


def _build_lstm(
    params: dict[str, Any], recipe: dict[str, Any]
) -> MultinomialLSTMModel:
    return MultinomialLSTMModel(
        embedder=ProjectedEmbedder(
            seq_cols=recipe["seq_cols"],
            embedded_cols=recipe["embedded_cols"],
            target_col=recipe.get("target_col", "Transactions"),
            embedding_dim=params["embedding_dim"],
        ),
        lstm_hidden_size=params["lstm_hidden_size"],
        dense_units=params["dense_units"],
        dropout=params["dropout"],
    )


def _build_transformer(
    params: dict[str, Any], recipe: dict[str, Any]
) -> MultinomialTransformerModel:
    return MultinomialTransformerModel(
        # The Transformer projects the embedder's width onto d_model, and has always
        # embedded at d_model, so that is the width the ProjectedEmbedder uses.
        embedder=ProjectedEmbedder(
            seq_cols=recipe["seq_cols"],
            embedded_cols=recipe["embedded_cols"],
            target_col=recipe.get("target_col", "Transactions"),
            embedding_dim=params["d_model"],
        ),
        seq_len=recipe.get("seq_len"),
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_encoder_layers=params["num_encoder_layers"],
        dropout=params["dropout"],
    )


def _build_valendin(
    params: dict[str, Any], recipe: dict[str, Any]
) -> ValendinLSTMModel:
    """Build the frozen benchmark. `params` carries no architecture, by design."""
    return ValendinLSTMModel(
        seq_cols=recipe["seq_cols"],
        embedded_cols=recipe["embedded_cols"],
        target_col=recipe.get("target_col", "Transactions"),
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """Everything the package needs to know about one model type.

    Every field is optional, so a model that only needs to be *named* (Pareto/NBD)
    registers as ``ModelEntry()`` rather than forcing the enumerations apart.

    Parameters
    ----------
    search_space
        Per-parameter specs in the ``suggest_param`` mini-language, used when the
        caller does not override that parameter. Its keys are also the allowlist a
        caller's overrides are validated against, so the two cannot drift.
    suggest
        ``(trial, specs) -> params``: how this entry's space is sampled — the other
        half of declaring a search space, not a separate concern. Nearly always
        ``_suggest_every_param``; the Transformer needs its own because two of its
        parameters constrain each other.
    build
        ``(params, recipe) -> nn.Module``: the TRAINING model. Its presence is what
        makes a type neural.
    rollout
        The Monte Carlo forecaster this model must be simulated through. The two
        differ because the architectures carry history differently (see
        ``models.monte_carlo_forecasting``); pairing the wrong one produces a
        forecast that is wrong rather than an error, so it is declared here rather
        than chosen by whoever calls the simulator.
    """

    search_space: dict[str, Any] | None = None
    suggest: Callable[[optuna.Trial, dict[str, Any]], dict[str, Any]] | None = None
    build: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None
    rollout: Callable[..., Any] | None = None


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "lstm": ModelEntry(
        search_space={
            "embedding_dim":   {64, 128, 256},
            "lstm_hidden_size": {32, 64, 128},
            "dense_units":     {32, 64, 128},
            "dropout":         (0.0, 0.4),
            "learning_rate":   (1e-4, 3e-3, "log"),
            "weight_decay":    (1e-6, 1e-2, "log"),
            "batch_size":      {64, 128, 256},
        },
        suggest=_suggest_every_param,
        build=_build_lstm,
        rollout=run_monte_carlo_forecast,
    ),
    "transformer": ModelEntry(
        search_space={
            # d_model and nhead lead, because `_suggest_transformer_params` resolves
            # them first to prune indivisible pairs.
            "d_model":            {32, 64, 128},
            "nhead":              {2, 4, 8},
            "num_encoder_layers": (1, 3, "int"),
            "dropout":            (0.0, 0.4),
            "learning_rate":      (1e-4, 3e-3, "log"),
            "weight_decay":       (1e-6, 1e-2, "log"),
            "batch_size":         {64, 128, 256},
        },
        suggest=_suggest_transformer_params,
        build=_build_transformer,
        rollout=run_monte_carlo_forecast_transformer,
    ),
    # The Valendin benchmark's architecture is FROZEN (ADR-0004): its widths are the
    # published `memory_units = 128` / `dense_units = 128` and its embeddings are raw
    # sqrt(n)+1 vectors, so none of them appear in the search space. Only training
    # hyperparameters are searched — Optuna over fixed sizes is the deliberate
    # departure ADR-0004 records, and searching a width would quietly unfreeze the
    # reference implementation. It is a stateful LSTM, so it rolls out through the
    # same simulator as ours; only the architecture differs.
    "valendin_lstm": ModelEntry(
        search_space={
            "learning_rate": (1e-4, 3e-3, "log"),
            "weight_decay":  (1e-6, 1e-2, "log"),
            "batch_size":    {64, 128, 256},
        },
        suggest=_suggest_every_param,
        build=_build_valendin,
        rollout=run_monte_carlo_forecast,
    ),
    # Declarative only. The Pareto/NBD benchmark is a single hierarchical-Bayes MCMC
    # fit: no Optuna study, no training, one prediction. Its entry exists so every
    # model-type enumeration derives from this table rather than adding it by hand.
    "pareto_nbd": ModelEntry(),
}

# The enumeration of model types IS the table's keys. Nothing restates it.
MODEL_TYPES: tuple[str, ...] = tuple(MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------


def entry(model_type: str) -> ModelEntry:
    """This type's registry entry, or a `ValueError` naming the registered types."""
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model_type {model_type!r}; "
            f"registered types: {sorted(MODEL_REGISTRY)}"
        ) from None


def is_neural(model_type: str) -> bool:
    """True when this type trains — i.e. when its entry carries a builder.

    Read off the entry rather than listed separately: the neural set is exactly the
    set of trainable models, so restating it is a copy that can drift.
    """
    return entry(model_type).build is not None


def _require(model_type: str, field: str, what: str) -> Any:
    """Fetch an optional field, or say plainly that this type does not have one."""
    value = getattr(entry(model_type), field)
    if value is None:
        raise ValueError(
            f"model_type {model_type!r} has no {what}: its registry entry is "
            f"declarative only (it does not train or roll out)."
        )
    return value


def validate_model_knobs(
    model_type: str, search_space: dict[str, Any], training: dict[str, Any]
) -> None:
    """Fail fast on a hyperparameter that is missing from, or misplaced in, the space.

    Both misplacements are silent otherwise, and both surface only after a whole
    search has run: a typo'd hyperparameter (`"hiddendim"`) is dropped and the entry's
    default range used instead; a real one left in `training` is dropped and never
    searched at all. The allowlist is the entry's own search space, so it cannot fall
    out of step with what the suggester samples.

    This polices `training` only for keys that belong in the *other* dict. Whether a
    training control is one the tuner actually reads is the tuner's own question, and
    `optuna_tuning` checks it against the keys it consumes.
    """
    declared = _require(model_type, "search_space", "search space")
    unknown = [k for k in search_space if k not in declared]
    if unknown:
        raise ValueError(
            f"Unrecognised search_space key(s) for model_type={model_type!r}: "
            f"{sorted(unknown)}. This model searches: {sorted(declared)}. "
            f"Training controls (n_epochs, loss_type, ...) go in `training`."
        )
    misplaced = [k for k in training if k in declared]
    if misplaced:
        raise ValueError(
            f"Hyperparameter(s) {sorted(misplaced)} passed as training controls for "
            f"model_type={model_type!r}; they belong in `search_space`, which is what "
            f"the search reads."
        )


def suggest_params(
    model_type: str, trial: optuna.Trial, search_space: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Sample this type's hyperparameters, with the caller's specs taking priority."""
    e = entry(model_type)
    specs = _merge_specs(
        _require(model_type, "search_space", "search space"), search_space
    )
    return e.suggest(trial, specs)


def build_model(
    model_type: str, params: dict[str, Any], recipe: dict[str, Any]
) -> Any:
    """Build this type's TRAINING model from sampled params + the split's recipe."""
    return _require(model_type, "build", "training builder")(params, recipe)


def rollout_for(model_type: str) -> Callable[..., Any]:
    """The Monte Carlo forecaster this model must be simulated through.

    Read here rather than chosen by the caller: the two rollouts step different
    architectures, and the wrong pairing yields a wrong forecast, not an error.
    """
    return _require(model_type, "rollout", "rollout function")
