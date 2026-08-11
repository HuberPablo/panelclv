# Build a faithful `ValendinLSTMModel` in `benchmarks/`

Status: ready-for-agent
Blocked by: 05

`MultinomialLSTMModel` is not the paper's architecture, so renaming it would give a
benchmark that quietly differs from what it claims to reproduce (ADR-0004). Comparing
`Original_paper_model/banking_transactions_demo.ipynb` against `models/multinomial_lstm.py`
found two departures nobody chose:

- **Embeddings** — theirs are raw `sqrt(n)+1` vectors; ours pass through LayerNorm and
  a projection to a common width.
- **Combination** — theirs concatenates every embedding into a roughly 12-dimensional
  LSTM input; ours sums the context and concatenates the target, giving 256.

Build `benchmarks/valendin_lstm.py` using the Valendin embedder from ticket 05, and
leave `MultinomialLSTMModel` in `models/` as our own variant. The result is a simpler
model than the current one; the training loop, simulator and evaluation are already
shared.

Import it lazily from `benchmarks/__init__.py` so Pareto-only callers do not pay for
torch.

Deliberate departures that stay: temporal validation split, Optuna tuning, and the
covariate path. Everything else should match.

Done when: the architecture matches the notebook layer for layer, and a run on the
banking demo data reproduces the published numbers. Ask Pablo for those numbers if
they are not in the notebook output.
