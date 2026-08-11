# Migrate the notebooks onto the current model and benchmark APIs

Status: ready-for-human

Ticket 03 scoped the MLE removal to `src/` and `scripts/`, so the notebooks were left
alone and now call an API that no longer exists. Each of these raises `TypeError` at the
call, not at import, so a notebook runs happily until it reaches the benchmark cell.

- `pareto_nbd_benchmark=True` / `pareto_paper_benchmark=True` → a single
  `pareto_benchmark=True`. In `Data_integration_LSTM.ipynb`,
  `Data_integration_LSTM_v2.ipynb`, `Data_integration_TRANSFORMER.ipynb`,
  `Data_integration_TRANSFORMER_v2.ipynb`, `Study.ipynb`, `august test.ipynb`.
- `pareto_forecast(data_best, variant="paper")` → drop the argument; there is one
  estimator. In `Study.ipynb` and `august test.ipynb`.
- `pareto_kwargs={"penalizer_coef": 0.01}` → drop it; the knob was MLE-only. Commented
  out in `Study.ipynb` and `august test.ipynb`, so it only bites if uncommented.

Note the two `pareto_nbd_benchmark=True` sites are not a rename but a **deletion**: they
requested the MLE estimator, which is gone. A notebook that printed both `Pareto/NBD` and
`Pareto/NBD (HB)` rows now gets one row, and its stored output showing two is stale.

## Also: the embedder seam changed the model constructors

Ticket 05 moved `seq_cols` / `embedded_cols` / `target_col` / the width off the model
constructors and onto an embedder the model is given, so these raise `TypeError`:

- `Data_integration_LSTM.ipynb` — `InferenceMultinomialLSTMModel(seq_cols=..., embedded_cols=..., target_col=..., embedding_dim=...)`
- `Data_integration_TRANSFORMER.ipynb` — `InferenceMultinomialTransformerModel(seq_cols=..., embedded_cols=..., target_col=..., d_model=...)`

Both become `embedder=ProjectedEmbedder(seq_cols=..., embedded_cols=..., target_col=...,
embedding_dim=<embedding_dim or d_model>)` with the remaining arguments unchanged; see the
updated call sites in `scripts/main_plot.py` for the exact shape.

Any checkpoint those notebooks reload also predates the seam and needs
`scripts/migrations/rename_embedder_checkpoint_keys.py` run over it first.

Marked for a human for the same reason as ticket 09: the notebooks are experiment records
as much as code, and which ones are still live is Pablo's call. Worth doing in one pass
with ticket 09, which touches the same cells.

Done when: every notebook still in use calls the current benchmark API, and the ones that
are not in use are said so out loud.
