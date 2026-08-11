# Benchmarks are frozen reference implementations

The thesis compares a new architecture against published ones, so the published ones
have to stay published — a benchmark that quietly drifts toward our own design
flatters the comparison it exists to make. `benchmarks/` therefore holds models we
reproduce rather than develop: the Valendin et al. LSTM and Pareto/NBD. Their
architectures are frozen; improving them is out of scope by definition. New LSTM
variants are new classes in `models/`.

Everything around the architecture — data preparation, embeddings as configured, the
training loop, tuning, the simulator, evaluation — is shared infrastructure applied
identically to every model. That is what makes a comparison isolate architecture.

## Deviations from Valendin et al.

Their code is in `Original_paper_model/`. These departures are deliberate:

- **Validation split** — temporal rather than a random 10% of customers (ADR-0001).
- **Tuning** — Optuna over architecture and feature subset; they use fixed sizes.
- **Covariates** — their model reads week and transaction count only; ours accepts
  arbitrary covariates through a projection.

Anything else that differs is a bug, not a decision. Two were found and are being
fixed: their embeddings are raw `sqrt(n)+1` vectors concatenated together, where ours
were projected to a common width through LayerNorm and summed.

## Consequences

`benchmarks/` is no longer torch-free, so the neural benchmark is imported lazily —
callers who only want Pareto/NBD do not pay for torch.
