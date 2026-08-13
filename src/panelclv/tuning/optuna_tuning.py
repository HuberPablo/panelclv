"""Optuna tuning for the models the registry declares.

One shared `objective` over every registered model type: what each one searches, how
that space is sampled and how the model is built all come from its registry entry
(ADR-0006), so this file holds the *search*, not a list of architectures. Each trial
samples an architecture + training HPs (and, optionally, a covariate subset), trains
via `training.loop.fit_model` — which optimises classification cross-entropy and
owns the loss curve, early stopping, and per-epoch pruning reports — then returns
that same cross-entropy, scored on the temporal validation window (ADR-0001), to
Optuna. Selection and training therefore minimise one number.

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
cross-entropy on the validation window only). `trials.split_calibration` /
`trials.make_data_builder` produce a contract-compliant builder from a
`prepare_dataset` dict; the train/val split is temporal (a time window over all customers).

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

# Which architectures exist, what they search and how they are built is declared once,
# in the registry (ADR-0006). This module searches; it does not enumerate models. The
# training loop lives in `panelclv.training`. After the subpackage split both are
# cross-package imports, so they are absolute rather than relative.
from panelclv.data_preparation.target_channel import target_index
from panelclv.registry import (
    build_model,
    suggest_param,
    suggest_params,
    validate_model_knobs,
)
from panelclv.training.loop import fit_model


# The `training` keys this module reads — every one appears as a `training.get(...)`
# in `objective` or `run_optuna_study` below. Kept next to the reads it describes so
# the two cannot drift, and checked up front because a misspelled control
# (`"paitence"`) is otherwise dropped in silence and discovered only in the loss
# curve. The search-space half of the same question belongs to the registry entry,
# which declares what each model searches.
TRAINING_CONTROLS: frozenset[str] = frozenset({
    "n_epochs", "patience",          # training control (scalar, or a search spec)
    "checkpoint_dir", "verbose",     # bookkeeping
    "loss_type", "class_weights", "focal_gamma",   # loss configuration
    "grad_clip", "log_wandb", "seed",              # optimiser / logging / RNG
})


def _validate_training(model_type: str, training: dict[str, Any]) -> None:
    """Fail fast on a training control this module would silently ignore."""
    unknown = [k for k in training if k not in TRAINING_CONTROLS]
    if unknown:
        raise ValueError(
            f"Unrecognised training key(s) for model_type={model_type!r}: "
            f"{sorted(unknown)}. Recognised controls: {sorted(TRAINING_CONTROLS)}; "
            f"hyperparameters go in `search_space`."
        )


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


# The closure signature `run_optuna_study` calls once per trial (see the module
# docstring): data_builder(feature_config, batch_size) -> (train_loader, val_loader,
# metadata). Declared once, here, and imported by `trials.loaders`, which builds one.
# The loaders are typed `Any` so this module does not have to name torch's DataLoader.
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
    target_idx = target_index(keep, target_col)   # position on the REDUCED axis

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
    after a study, to refit the winning model and run the forecast on matching
    columns (otherwise the checkpoint — trained on the sliced layout — will not
    warm-start a full-feature model):

        data_best = select_features_for_trial(data_full, study.best_trial)
        # refit from data_best["seq_cols"]/["embedded_cols"] and pass data_best
        # (not data_full) to the Monte Carlo forecast.

    A trial with no `dropped_features` attribute (e.g. a study run without
    `removable_features`) dropped nothing, so `data` is returned with every
    column intact. `select_features` raises if a recorded column is absent from
    `data` — a guard against pairing a trial with the wrong dataset.
    """
    raw = trial.user_attrs.get("dropped_features", "")
    dropped = raw.split(",") if raw else []
    return select_features(data, dropped)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial,
    model_type: str,
    data_builder: DataBuilder,
    search_space: dict[str, Any],
    training: dict[str, Any],
    device: str | torch.device | None = None,
    removable_features: Sequence[str | Sequence[str]] = (),
) -> float:
    """Objective: teacher-forced validation cross-entropy.

    `search_space` overrides the registry entry's default spec for a hyperparameter
    (per-parameter specs in the `suggest_param` mini-language — set=categorical,
    tuple=range, scalar=fixed; anything omitted falls back to the entry's own
    range), and is validated up front against that entry's keys. `training` carries
    the controls that are not searched — checkpoint dir, loss config, epochs — and
    is read with defaults. `removable_features` lists covariates Optuna may drop
    this trial (see `suggest_covariate_selection`); the chosen drop-set is handed to
    `data_builder` as `feature_config`.

    What is RETURNED to Optuna is the cross-entropy the training loop already
    minimises, scored on the temporal validation window only (ADR-0001), so
    selection and training agree on the number they are looking at.
    """
    params = suggest_params(model_type, trial, search_space)

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

    model = build_model(model_type, params, metadata)

    # `focal_gamma` is either a scalar (fixed) or `(low, high, step)`
    # (Optuna-tuned on a step grid). Missing / None → 2.0.
    loss_type = training.get("loss_type", "cross_entropy")
    focal_gamma_spec = training.get("focal_gamma", 2.0)
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
        num_target_classes=model.num_target_classes,
        # n_epochs / patience are training control, but the caller may still hand
        # them a search spec (e.g. patience over {5,7,9}); resolve through the same
        # mini-language so a scalar stays fixed and a set/tuple is searched.
        n_epochs=suggest_param(trial, "n_epochs", training.get("n_epochs", 50)),
        patience=suggest_param(trial, "patience", training.get("patience", 5)),
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        grad_clip=training.get("grad_clip", 1.0),
        device=device,
        checkpoint_dir=training.get("checkpoint_dir", "./checkpoints"),
        model_name=f"{model_type}_trial_{trial.number}",
        trial=trial,
        log_wandb=training.get("log_wandb", False),
        verbose=training.get("verbose", False),
        loss_type=loss_type,
        class_weights=training.get("class_weights"),
        focal_gamma=focal_gamma,
        # Temporal split: score CE only on the validation suffix (periods after
        # validation_start). split_calibration puts this in its recipe; 0 ⇒ score all steps.
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
    search_space: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
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

    The two knob dicts are deliberately separate. `search_space` overrides the
    registry entry's default spec for a hyperparameter, one key per parameter in
    the `registry.suggest_param` mini-language (a `{...}` set is a categorical, a
    `(lo, hi, "log"|"int")` tuple a range, a scalar is pinned); anything left out
    keeps the entry's own range, and a key the model does not have raises. `training`
    carries what is not searched — `n_epochs`, `patience`, `checkpoint_dir`,
    `verbose`, `loss_type`, `class_weights`, `focal_gamma`, `grad_clip`, `log_wandb`,
    `seed`. `n_epochs` / `patience` sit there because they are training control, but
    they may still be handed a search spec (e.g. patience over `{5, 7, 9}`) and are
    resolved through the same mini-language.

    When `append_timestamp` is True (default) the effective run name is
    `f"{study_name}_{YYYYMMDD_HHMM}"`; that name is used for the Optuna study,
    a per-run checkpoint subfolder under `training["checkpoint_dir"]`, and the
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
    (`trials.refit_best_trial` warm-starts from the best trial).
    Set it True to delete all non-best trial checkpoints once the study completes
    and the summary is written. The best trial's file (and its recorded
    `checkpoint_path`) is preserved, so the downstream workflow is unaffected; you
    only lose the ability to rebuild a NON-winning trial from its weights.
    """
    search_space = dict(search_space or {})
    training = dict(training or {})

    # Reject an unregistered type, a key in the wrong knob dict, and a control this
    # module does not read — all here rather than after the first trial has trained,
    # because every one of them is otherwise ignored in silence.
    validate_model_knobs(model_type, search_space, training)
    _validate_training(model_type, training)

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
    # filenames never overwrite a previous study's. `training` was already copied
    # above, so the caller's dict is untouched. `fit_model` mkdir's the dir.
    base_ckpt = Path(training.get("checkpoint_dir", "./checkpoints"))
    training["checkpoint_dir"] = str(base_ckpt / run_name)

    # Resolve the pruner. `True` (default) / `None` keep the historical
    # early-stopping behaviour (MedianPruner on the per-epoch CE fit_model
    # reports); `False` disables pruning entirely (NopPruner). A concrete
    # BasePruner instance is honoured as-is.
    if pruner is True or pruner is None:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    elif pruner is False:
        pruner = optuna.pruners.NopPruner()
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=training.get("seed", 42))

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
            trial, model_type, data_builder, search_space, training, device,
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
    # "checkpoint_path" user attr; the downstream refit (`trials.refit_best_trial`)
    # only ever warm-starts from the BEST trial, so once the summary above is written
    # every other trial's .pth is dead weight. Delete them when asked.
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
            if training.get("verbose"):
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
