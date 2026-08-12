# Benchmarks are frozen reference implementations

The thesis compares a new architecture against published ones, so the published ones
have to stay published — a benchmark that quietly drifts toward our own design
flatters the comparison it exists to make. `benchmarks/` therefore holds models we
reproduce rather than develop: the Valendin et al. LSTM and Pareto/NBD. Their
architectures are frozen; improving them is out of scope by definition. New LSTM
variants are new classes in `models/`.

**Frozen means the numbers, not the surrounding code.** A benchmark's *architecture* — its
layers, widths, activations and the arithmetic that produces a forecast — may not change.
Everything else in the file may: the plumbing that hands it data, the way its rollout model
is paired with its training model, its imports and its documentation. The test is not
whether the file was edited but whether `scripts/validate_pareto_benchmark.py` and
`scripts/validate_valendin_lstm.py` still land in their bands afterwards. Those two scripts
are the executable definition of this ADR, and any change touching `benchmarks/` is gated
on them.

Everything around the architecture — data preparation, embeddings as configured, the
training loop, tuning, the simulator, evaluation — is shared infrastructure applied
identically to every model. That is what makes a comparison isolate architecture.

## Deviations from Valendin et al.

Their code is in `Original_paper_model/`. These departures describe **our** model in
`models/`, not the benchmark — `benchmarks/valendin_lstm.py` is the published
architecture, and the whole point of it being a separate module is that it does not
inherit our choices:

- **Validation split** — temporal rather than a random 10% of customers (ADR-0001).
- **Tuning** — Optuna over architecture and feature subset; they use fixed sizes.
- **Covariates** — their model reads week and transaction count only; ours accepts
  arbitrary covariates through a projection.

The covariate line is settled for the benchmark: **it takes none**. The published model
has two inputs, `week` and `transaction`, with nowhere for a covariate to enter, so
`ValendinEmbedder` raises on a non-embedded column rather than dropping it — silently
ignoring a requested feature would make the benchmark differ from what the caller asked
for without saying so. A covariate run is our model's job.

Anything else that differs is a bug, not a decision. Two were found and are now fixed:
their embeddings are raw `sqrt(n)+1` vectors concatenated together, where ours were
projected to a common width through LayerNorm and summed. Rather than change our model,
both strategies became embedders (ADR-0005) and the published one backs the benchmark.

## Which Pareto/NBD

**Pareto/NBD means the hierarchical-Bayes MCMC estimator** — a pure NumPy port of R's
BTYDplus, the one Valendin et al. actually fit. A frequentist-MLE variant (via
`lifetimes`) was carried alongside it for a while. Keeping both was reproducing an
estimator the paper does not use, and the pair invited comparing our benchmark against
itself rather than against the published one, so the MLE variant was retired.

Study results produced before that decision are stored under `ParetoNBD_MLE`, never
`ParetoNBD`: the two are different models, and a stored result that does not name its
estimator cannot be defended later.

## Consequences

**The validation scripts re-implement what they check, deliberately.** Each carries its own
`WEEKS_PER_YEAR`, cohort filter and week index rather than importing the package's. A gate
that imports the code it gates stops being a gate: a future bug in a shared cohort filter
would move the benchmark and its own check in lockstep and still pass. These copies are
insulation and are not to be deduplicated.

Retired reference implementations move to the repo-root `archive/` rather than out of
the repo, so a stored result stays traceable to the code that produced it. `src/` holds
only shippable package code — the convention `Original_paper_model/` already set.
