# 05 — `backpropagation.md` says `emd_weight` is in the registry search space; it raises

**Status:** ready-for-agent

A reader following this sentence writes a config that fails.

## Doc claim

`docs/backpropagation.md:294-296`:

> `λ` itself is tuned — it is **declared in the registry search space** and resolved by
> Optuna like any other hyperparameter, so a study chooses how ordinal it wants to be.

## Code reality

`emd_weight` is a **training control**, not a search-space key:

- `src/panelclv/tuning/optuna_tuning.py:84` lists it in `TRAINING_CONTROLS` beside
  `loss_type` / `class_weights` / `focal_gamma`.
- `src/panelclv/tuning/optuna_tuning.py:333-337` samples it *inside* the `ce_emd` branch,
  by handing `training.get("emd_weight", 1.0)` to the registry's `suggest_param` helper —
  which is why a range works there, and why it is easy to mistake for a search-space key.
- It appears in no `ModelEntry.search_space` (`src/panelclv/registry/model_registry.py:327-383`).
- `scripts/run_loss_ablation.py:143` does it correctly, as `training={"emd_weight": (0.0, 10.0)}`.

Putting it where the doc says does not silently do nothing — `validate_model_knobs` rejects
it, by design (`src/panelclv/registry/model_registry.py:441-447`).

## Evidence

```console
$ PYTHONPATH=src ~/Desktop/Thesis/venvs/thesis_rocm/bin/python -c "
from panelclv.registry import validate_model_knobs
validate_model_knobs('lstm', {'emd_weight': (0.0, 10.0)}, {})"
ValueError: Unrecognised search_space key(s) for model_type='lstm': ['emd_weight'].
This model searches: ['batch_size', 'dense_units', 'dropout', 'embedder',
'embedding_dim', 'learning_rate', 'lstm_hidden_size', 'weight_decay'].
Training controls (n_epochs, loss_type, ...) go in `training`.
```

## Fix

Rewrite `docs/backpropagation.md:294-296` to say λ is a **training control** searched through
the same mini-language, sampled only when `loss_type="ce_emd"`, and cite
`tuning/optuna_tuning.py:333-337` plus the working example at `scripts/run_loss_ablation.py:143`.

Worth saying explicitly *why* it lives there rather than in the search space: it is a
property of the objective, not of the architecture, so it is not one of the knobs
`validate_model_knobs` polices per model type.

## Related

`docs/backpropagation.md` is one of the three docs `tests/test_docs_are_current.py` does not
lint — see issue `17`.
