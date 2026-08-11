# Migrate notebooks off `INPUT_SPEC`

Status: done

`configs/transformations_spec.py` no longer exists — `PanelConfig` absorbed the column
roles, time features, autoregressive features and embedding declarations it used to
carry. All seven notebooks still reference `INPUT_SPEC`, so they run a legacy path the
package no longer supports.

Marked for a human because the notebooks are experiment records as much as code, and
which ones are still live is Pablo's call. `Data_integration_*_v2.ipynb` were the
current pair as of the last documentation pass; `august test.ipynb`, `Study.ipynb` and
`Pareto_Datasets.ipynb` postdate it.

Done when: every notebook still in use builds its dataset from `PanelConfig`, and the
ones that are not in use are said so out loud.

## Comments

Resolved together with ticket 11, which touches the same cells.

**The premise was stale.** All seven notebooks did reference `INPUT_SPEC`, but every one
of the eleven hits was the same leftover comment line — `# One config object replaces
DATA_CONFIG + TIME_FEATURES + FEATURE_SCHEMA + INPUT_SPEC.` — never code. The migration
itself had already happened: every notebook builds through `prepare_dataset(panel, cfg)`
with a `PanelConfig`, every `panelclv` import resolves against the current package, and
every call matches the current signature. The comment is now rewritten to describe what
`PanelConfig` does rather than what it replaced.

**Which notebooks are live** (Pablo's call, the reason this was `ready-for-human`):

- Live, in `notebooks/`: `Data_integration_LSTM_v2.ipynb`,
  `Data_integration_TRANSFORMER_v2.ipynb`, `Study.ipynb`, `Pareto_Datasets.ipynb`.
- Dead, moved to `notebooks/archive/` with a README giving the reason for each:
  `Data_integration_LSTM.ipynb` and `Data_integration_TRANSFORMER.ipynb` (superseded by
  the `_v2` pair), `august test.ipynb` (byte-identical to `Study.ipynb` — all 73 source
  cells match exactly), `dataset_building.ipynb` (imports nothing from `panelclv`).

`README.md`, `docs/feature_engineering.md` and `data_preparation/__init__.py` each
pointed at a notebook that moved, and now point at the archive.

**Standing check.** `tests/test_notebooks_current_api.py` asserts that no live notebook
mentions a retired name (`INPUT_SPEC`, `FEATURE_SCHEMA`, `pareto_nbd_benchmark`,
`pareto_paper_benchmark`, `variant="paper"`, `penalizer_coef`) in its cell source **or in
a stored output**, and that every `panelclv` import in one resolves. Notebooks are JSON, so
nothing else in the suite reads them — this is what stops the next rename from going
unnoticed for months. It is static: no cell is executed. `notebooks/archive/` is
deliberately excluded, being frozen by design.

It also binds every `panelclv` call in a live notebook against `inspect.signature`, which
is the check that generalises — it catches a rename nobody thought to blacklist.

Both additions came out of the review of this pass, each after it found a real defect the
earlier, weaker check had passed over; see ticket 11's comment for what they were.
