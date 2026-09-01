# 01 — Training is unseeded; `training["seed"]` reaches only the Optuna sampler

**Status:** needs-triage

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
