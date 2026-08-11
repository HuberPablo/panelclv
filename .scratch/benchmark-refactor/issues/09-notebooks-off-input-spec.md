# Migrate notebooks off `INPUT_SPEC`

Status: ready-for-human

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
