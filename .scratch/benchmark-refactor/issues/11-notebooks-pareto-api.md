# Migrate the notebooks onto the current model and benchmark APIs

Status: done

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

## Comments

Done in one pass with ticket 09, as this ticket suggested. Read that ticket's comment for
the live/dead verdict on the notebooks; only the four live ones were migrated.

Applied to the live notebooks:

- `pareto_nbd_benchmark=True` / `pareto_paper_benchmark=True` -> a single
  `pareto_benchmark=True`, in `Data_integration_LSTM_v2.ipynb`,
  `Data_integration_TRANSFORMER_v2.ipynb` and `Study.ipynb`. The commented-out
  `#pareto_nbd_benchmark=True` overlay hints were renamed too — they are instructions to
  the reader, and would have raised the moment anyone uncommented one.
- `pareto_forecast(data_best, variant="paper")` -> the argument dropped.
- `pareto_kwargs={"penalizer_coef": 0.01}` -> `{"mcmc": 2500, "burnin": 500}`, the real
  defaults of `compute_pareto_predictions`. The old line also claimed `penalizer_coef=0.01`
  was the default, which it was not.
- Prose that described "the two Pareto/NBD comparators (MLE + hierarchical-Bayes)" now
  describes the one estimator.
- `_pareto_from_data(data_best, "mle")` / `(data_best, "paper")` -> one
  `_pareto_from_data(data_best)`, in `Study.ipynb` cell 39 and
  `Data_integration_LSTM_v2.ipynb` cell 26. The aggregate-total sanity check called the
  internal fitter directly for each variant; today's signature is
  `_pareto_from_data(data, **fit_kwargs)`, so the positional variant argument raised
  `TypeError`, and one of the two variants it asked for no longer exists. Found by the
  review, not by the first pass — this ticket's own list did not mention these sites.

The stale stored output this ticket predicted was real, and there was a second kind it did
not predict. Both were cleared in `Study.ipynb`; the contents remain in git history.

- Cell 38 held a `metrics_table` printout with both a `Pareto/NBD` and a `Pareto/NBD (HB)`
  row, which the migrated cell can no longer produce.
- Cell 64 held a saved `ValueError` traceback that quoted the **pre-refactor**
  `plot_suite_forecast(..., pareto_nbd_benchmark, pareto_paper_benchmark, **plot_kwargs)`
  signature and echoed the cell's own pre-migration source line. Migrating the source
  alone left the cell contradicting itself, presenting two dead keywords as real API to
  anyone reading the stored output. The run had failed anyway, so nothing was lost.

The second one is why `tests/test_notebooks_current_api.py` scans stored outputs — stream
text, rich data, and an error's `evalue` and `traceback` — and not just cell source. A
saved traceback quotes signatures back at the reader and survives every source-level
search, so source-only scanning would have declared this ticket done while a live notebook
still displayed the retired API.

**The blacklist was not enough.** A retired-name list only catches renames someone thought
to add to it, and it missed the `_pareto_from_data` breakage above entirely: the dead call
mentions no retired name, it just passes an argument the function no longer takes. So the
test now also **binds every `panelclv` call in a live notebook against
`inspect.signature`** — placeholders only, nothing evaluated, still fully static. That
check needs no advance warning of a rename and reports exactly what the cell would raise.
Run against `notebooks/archive/august test.ipynb` (the pre-migration `Study.ipynb`,
byte-for-byte) it independently rediscovers every defect this ticket listed, plus the
embedder-seam constructors and a stale `val_idx` argument in the v1 notebooks. Both checks
are kept: binding cannot see commented-out lines, prose or stored outputs, and the
blacklist cannot see an argument-shape change.

The `Pareto/NBD (HB)` strings in cells 42-44 were left alone — there they are user-chosen
dict keys labelling prediction CSV paths, not API.

**Not done, deliberately:** the embedder-seam constructor fixes. Both call sites
(`InferenceMultinomialLSTMModel`, `InferenceMultinomialTransformerModel`) were only in the
two v1 notebooks, which are now archived. What they would need, and the checkpoint
migration that goes with it, is written down in `notebooks/archive/README.md` so reviving
one is not a rediscovery job.
