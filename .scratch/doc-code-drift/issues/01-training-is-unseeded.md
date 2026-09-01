# 01 — Training is unseeded; `training["seed"]` reaches only the Optuna sampler

**Status:** done

Reproducibility is priority #2 in this repo, and it does not hold on the production path.

## Doc claim

`CLAUDE.md:34-35`:

> 2. **Reproducibility** — same config and seed gives the same result, and results
>    never depend on the order notebook cells were run in.

`src/panelclv/studies/config.py:116-118`:

> ``base_seed`` — Study ``i`` uses ``base_seed + i`` for its sampler **and training**, so the
> studies are genuine independent replications.

`src/panelclv/tuning/optuna_tuning.py:85` classes `seed` as an RNG control:

> ```python
> "grad_clip", "log_wandb", "seed",              # optimiser / logging / RNG
> ```

## Code reality

`seed` reaches exactly one place — the TPE sampler:

- `src/panelclv/tuning/optuna_tuning.py:493` —
  `sampler = optuna.samplers.TPESampler(seed=training.get("seed", 42))`

There is **no `torch.manual_seed` anywhere** in `training/`, `tuning/`, `trials/` or
`studies/`. The only two seeding sites in the package are the forecaster
(`models/monte_carlo_forecasting.py:374`) and the synthetic-panel / Pareto MCMC generators
(`benchmarks/pareto_nbd.py:348`, `data_preparation/pareto_nbd_simulation.py:220`).
Neither `fit_model` nor `refit_full_calibration` takes a seed argument
(`training/loop.py:203-222`, `:400-412`). So weight initialisation, dropout masks and
`DataLoader` shuffling are all drawn from the *global* torch RNG, which the package never
sets.

Two consequences:

1. Two runs of the same study, same config, same `seed`, in one process, differ.
2. Because it is the **global** RNG, the result also depends on what ran before it — which
   is the second half of the `CLAUDE.md:34-35` claim ("results never depend on the order
   notebook cells were run in").

## Evidence

`tests/test_golden_end_to_end.py` asserts bit-identical determinism and passes — but only
because *the test itself* seeds torch at `tests/test_golden_end_to_end.py:207`
(`torch.manual_seed(TORCH_SEED)`). Nothing in `src/` does that, so the property is the
harness's, not the package's.

Run directly against the package — one fixed dataset, two identical `run_optuna_study`
calls in one process, `training={"seed": 42}` both times:

```console
$ PYTHONPATH=src ~/Desktop/Thesis/venvs/thesis_rocm/bin/python - <<'PY'
# ... 24-customer x 78-week synthetic panel, prepare_dataset, then twice:
#   run_optuna_study(model_type="lstm", search_space=<all six knobs pinned>,
#                    training={"n_epochs": 2, "patience": 2, "seed": 42, ...},
#                    n_trials=2, pruner=False)
PY
run 1 trial values: [1.6184914112091064, 1.5807276964187622]
run 2 trial values: [1.6029781103134155, 1.5970462560653687]
IDENTICAL: False
```

Every architectural hyperparameter was pinned, so the two runs sampled the identical
architecture; the difference is entirely initialisation and dropout.

## Fix options

**(a) Seed the training path.** Thread a `seed` through `fit_model` and
`refit_full_calibration` and call `torch.manual_seed` (plus a generator on the
`DataLoader`) at the top of each. `run_optuna_study` would derive a per-trial seed from
`training["seed"]` and the trial number, so trials stay distinct while the study repeats.

- Cost: every archived study under `Studies/` was produced with unseeded weights, so none
  of them will reproduce against the new baseline. That has to be stated somewhere before
  the thesis leans on any stored number as re-runnable.
- Also worth checking before committing to this: torch on ROCm is not bit-deterministic
  across all kernels without `torch.use_deterministic_algorithms(True)`, which may not be
  free here.

**(b) Correct the docs.** Say plainly that only the *search* is seeded, that training is
not, and that this is why a study suite reports a distribution across replications rather
than one number — which is arguably the honest framing already, given `base_seed + i`
exists to make studies independent. `CLAUDE.md:34`, `studies/config.py:116-118` and
`optuna_tuning.py:85`'s "RNG" comment would all change.

Whichever direction is taken, `studies/config.py:116-118`'s "and training" is wrong today
and must go either way.

## Related

Issue `11` is the other false claim in the same `StudySuiteConfig` docstring.

## Comments

Closed by fix option **(b)** — the prose is corrected, no behaviour moves. Decided in a
grilling session; two facts found there set the direction, neither of them in the ticket
above:

- **The notebooks already seed deliberately.** `notebooks/Study.ipynb` cell 2 and
  `notebooks/Pareto_Datasets.ipynb` call `torch.manual_seed` + `torch.cuda.manual_seed_all`
  under a comment naming this exact gap. The package's de facto position was "the entry
  point owns the global RNG" — undocumented, and honoured only by the notebooks.
- **The archive is already unreproducible, so (a) would have cost less than the ticket
  says.** `scripts/run_studies.py`, `scripts/run_loss_ablation.py`,
  `scripts/run_cdnow_embedding_ablation.py`, `scripts/run_pnbd_grid.py` and both
  `grids/*.py` never seed torch. `Studies/loss_ablation_cdnow`, `seasonal_4x4x10__*` and
  `pnbd_study_*` were all produced that way and do not reproduce today under any seed.

The correction ran wider than the three sites listed above, because the same claim was
false in five more places:

| Site | Change |
|---|---|
| `CLAUDE.md:28-40` | Reproducibility demoted 2 → 3 and restated as the seeded/unseeded boundary. The "results never depend on the order notebook cells were run in" clause is deleted, not reworded: `_run_monte_carlo` seeds the *global* torch RNG, so a forecast advances it and cell order is part of the result by design. |
| `studies/config.py` `base_seed` | "for its sampler and training" → the sampler and the Monte Carlo forecast; "It does not seed training." |
| `studies/config.py` `n_studies_per_model` | issue `11`'s line, closed here with its option (a) — see that ticket. |
| `tuning/optuna_tuning.py:85` | comment "RNG" → "Optuna sampler seed". |
| `docs/running-a-model.md:724-726` | "the runner owns the seed" now says *which* seed. |
| both notebooks' cell 2 | comment block rewritten: says what the two lines buy (a fresh-kernel top-to-bottom run) and what they do not (cell re-execution; anything under `scripts/` or `grids/`). Code untouched. |
| `tests/test_golden_end_to_end.py:22-25` | quoted the deleted sentence as what it proves. Restated: it pins bit-identical behaviour *given* a seeded RNG that the test sets itself at `:207`, so it guards the pipeline's wiring, not a determinism the package provides. |

The priority order was cited **by number** in five live files, and demoting reproducibility
would have swapped 2 and 3 under all of them. Rather than renumber, every citation now names
the priority instead — `tests/test_output_paths.py:3`, `predictions/run_directory.py:7`,
`docs/running-a-model.md:91`, `tests/test_ar_feature_support.py:37` and the golden test. The
list can be reordered from here without a rename. (`predictions/`'s claim is *true* — a run
folder does derive from config and seed — so only its citation moved.)

**Decisions taken and not taken.** The promise is scoped to same process / same machine /
same build; `torch.use_deterministic_algorithms(True)` was not added, because ROCm's cost for
it is unmeasured and `VastAI/Rules.md:266-274` already declines to promise bit-identity. The
`Studies/` archive is left as it is, with no new provenance fields. No ADR.

**What carries it instead:** `tests/test_training_is_unseeded.py` — one parametrised case per
module under `training/`, `tuning/`, `trials/`, `studies/`, asserting none of them sets the
global RNG. This is the class of claim `test_docs_are_current.py:16-18` says it cannot catch
(false without naming anything that moved), so it is pinned by a test instead. Verified to
fail by temporarily seeding `fit_model`, then reverted.

**Fix option (a) remains open.** Seeding the training path is still the change that would make
the stronger claim true; it is now a deliberate act that must move `CLAUDE.md`, the `base_seed`
docstring and that test together, rather than something that can drift in.

`pytest -q`: **366 passed** (350 before, plus this test's 16 cases), 31.99s.
