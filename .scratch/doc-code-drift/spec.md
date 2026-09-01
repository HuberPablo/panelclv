# Doc ↔ code drift audit

**Date:** 2026-09-01
**Scope:** every prose surface in the repo, checked against what the code actually does.

## Why

The prose layer here is load-bearing for a thesis: `CLAUDE.md`, `CONTEXT.md`, `README.md`,
eight ADRs, four topic docs under `docs/`, plus long module docstrings that carry the
reasoning a reader is meant to follow. Only a slice of that is gated.
`tests/test_docs_are_current.py` checks *paths and symbols* in six of the surfaces, and its
own docstring names the hole it leaves:

> What is NOT checked, and cannot be, is a claim that is false without naming anything that
> moved: this chapter once asserted the package applies "no per-feature standardisation"
> while `standardize_covariates` had been running for a day. Only a reader catches that one.

This audit is that reader. It found 30 divergences; that same sentence's bug is one of them,
still present in a second place in the same file (issue `06`).

## How each finding was verified

Claims were settled by **executing or tracing the code**, never by reading an adjacent
comment or docstring — comments are part of what is under audit. Concretely:

- The full suite was run first, to establish that nothing below is already caught:
  `PYTHONPATH=src ~/Desktop/Thesis/venvs/thesis_rocm/bin/python -m pytest -q`
  → **336 passed** in 32 s. Every finding here is invisible to it.
- Behavioural claims were probed in a live interpreter against the real package
  (`PYTHONPATH=src`, the `thesis_rocm` venv, nothing installed or changed). Issues `01`,
  `05`, `18` carry the probe and its output in an **Evidence** block.
- Structural claims were traced through the actual control flow, with `file:line` for both
  the doc side and the code side.

Nothing in `src/`, `tests/`, `docs/` or the notebooks was modified by this audit.

## A note on trust — findings that were checked and thrown out

An audit pass reported roughly eleven findings against `docs/loss-functions.md`: stale line
numbers throughout §3, "202 lines, four selectable strings", "no CDNOW suite exists",
"recommendation R2 asks for code that already exists". **Every one is false.** Checked
against the file on disk:

- `wc -l src/panelclv/models/losses.py` → **277**, matching the doc's "277 lines, five
  selectable strings, one factory" (`docs/loss-functions.md:158`).
- Every §3 line reference sampled — `41, 68, 76, 91, 99, 130, 138, 212, 228, 249, 259-273`
  — resolves exactly, `ce_emd` row included.
- §7 already reads "`ls Studies/` **now returns 19 entries**, of which exactly one is CDNOW:
  `Studies/loss_ablation_cdnow`" (`:998-1000`).
- R2 is annotated "**(implemented:** the dispatch arm is `models/losses.py:272-273`, the
  module `CrossEntropyPlusEMDLoss` at `99-130`**)**" (`:869-870`) and carries a Measured
  block dated 2026-08-31.

`docs/loss-functions.md` is the most current doc in the repo. Those findings are **not**
in the issue list and should not be re-chased.

## What is in `issues/`

21 tickets, ordered by impact. Each carries the doc claim with `file:line`, the code reality
with `file:line`, and — where the fix direction is genuinely open — both options rather than
a verdict.

| # | Ticket | Kind |
|---|---|---|
| 01 | Training is unseeded; `seed` reaches only the Optuna sampler | behaviour |
| 02 | A Valendin forecast is written into a directory named `lstm_…` | behaviour |
| 03 | The frozen benchmark's widths are constructor-overridable | behaviour |
| 04 | `scipy` is a hard import and an undeclared dependency | packaging |
| 05 | `backpropagation.md` puts `emd_weight` in the registry search space | doc false |
| 06 | `feature_engineering.md` §11 "No per-feature scaling" | doc false |
| 07 | ADR-0004 and the benchmark's docstring disagree on split and tuning | doc false |
| 08 | `running-a-model.md` §3 gives the `prepare_dataset` order wrong | doc false |
| 09 | The refit docstring's "typically the `best_epoch`" | doc false |
| 10 | The embedder seam (ADR-0005) never reached the feature docs | doc false |
| 11 | `studies/config.py`: "Coerced to 1" — nothing coerces | doc false |
| 12 | The root `__init__` calls `benchmarks` non-neural | doc false |
| 13 | The cohort rule is stated more narrowly than it behaves | doc imprecise |
| 14 | `CONTEXT.md`: "both are standardised the same way" | doc imprecise |
| 15 | ADR precision batch (0002, 0003, 0004, 0007) | doc imprecise |
| 16 | The refit's lr / batch size cite an ADR that does not say it | over-attribution |
| 17 | The doc-currency gate's scope and its "seven prose surfaces" | coverage |
| 18 | Subpackage `__init__` promises vs. what they actually export | doc false |
| 19 | Triage labels and the `Status:` vocabulary | convention |
| 20 | README and `running-a-model` accuracy batch | doc false |
| 21 | `week_*` column names on a non-weekly panel | naming |

Four are behaviour or packaging rather than prose (`01`–`04`). Those are written as
**divergences, not change requests**: each lays out both directions — change the code, or
correct the doc — and leaves the call open. `01` in particular, because seeding the training
loop would move every archived study's numbers off their current baseline.

## Verified clean — do not re-audit

Checked by execution or by tracing control flow, and found accurate:

- **`docs/loss-functions.md`** — fully current, see the trust note above.
- **`docs/p-slstm.md`** — accurate, including its own "P-sLSTM is not part of `panelclv`"
  and the note that `compare.py:44` points at a dead scratch path.
- **`docs/agents/domain.md`** — accurate.
- **ADR-0001** — the temporal split is enforced in exactly one place
  (`trials/loaders.py:102-129`); `compute_class_weights` defaults to `training_only=True`
  and slices the training prefix (`models/losses.py:186-189`), so the validation window's
  class mix does not reach the loss.
- **ADR-0006's core invariant** — `MODEL_TYPES = tuple(MODEL_REGISTRY)`
  (`registry/model_registry.py:386`) is the only enumeration of the model set in `src/`, and
  `is_neural` is a predicate over `entry(...).build`, not a second list. (Issue `02` and the
  prose copy at `studies/config.py:55-57` are the only leaks.)
- **`predictions` is a leaf** — `prediction_csv.py` and `run_directory.py` import stdlib +
  numpy/pandas only; zero `panelclv` imports, deferred ones included.
- **No cross-subpackage duplicate exports** — checked at runtime across all eleven
  subpackages' public surfaces: zero name collisions.
- **ADR-0004's covariate ruling** — tested. `ValendinLSTMModel(seq_cols=[…, "cov"], …)`
  raises with the ADR-0004 message; `ValendinEmbedder`'s width is
  `sum(sqrt(n)+1) + len(covariates)`, measured 11 → 12 when one covariate is added, so the
  benchmark's arithmetic with no covariate is unchanged.
- **`Embedder` refuses a head/target mismatch** — tested on both strategies: raises when
  `target_col` is missing from `seq_cols` or from `embedded_cols`.
- **The AR-feature leak discipline** — placeholders zeroed, filled on the calibration window
  from the *clipped* target, holdout columns left at zero and overwritten at each rollout
  step from `ARFeatureState.update(sample)`, re-standardised with `covariate_stats`. No leak.
- **`to_rollout()`** — the only construction path for all three rollout classes, sharing the
  backbone by reference; no rollout class is named outside `models/` and `benchmarks/`
  except in two tests.
- **`scripts/`** — every script is referenced from a doc, a test or another script;
  `run_loss_ablation.py` is documented at `docs/loss-functions.md:834`.
