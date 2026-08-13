# Archived notebooks

These four notebooks are **not in use**. They are kept as experiment records, not as
code: nothing here is migrated to the current package API, and none of them is expected
to run. The live notebooks are the four left in `notebooks/`.

Recorded 2026-08-11, closing `.scratch/benchmark-refactor/issues/09-notebooks-off-input-spec.md`
and `11-notebooks-pareto-api.md`, both of which asked for the dead ones to be said so out
loud.

| Notebook | Why it is dead |
| --- | --- |
| `Data_integration_LSTM.ipynb` | Superseded by `Data_integration_LSTM_v2.ipynb`, which adds the temporal validation window. Last touched 2026-06-21. |
| `Data_integration_TRANSFORMER.ipynb` | Superseded by `Data_integration_TRANSFORMER_v2.ipynb`, same reason. Last touched 2026-06-18. |
| `august test.ipynb` | Byte-identical to `Study.ipynb` — all 73 source cells match exactly. Keeping both meant applying every fix twice. |
| `dataset_building.ipynb` | Oldest of the eight (2026-06-01) and imports nothing from `panelclv`; panel construction now lives in `data_preparation`. |

## What would need doing if one is revived

Beyond the migrations applied to the live notebooks (`pareto_benchmark`, no
`variant="paper"`, no `penalizer_coef`), the two v1 notebooks predate the embedder seam
of ADR-0002 / ticket 05 and construct inference models the old way:

- `Data_integration_LSTM.ipynb` — `InferenceMultinomialLSTMModel(seq_cols=..., embedded_cols=..., target_col=..., embedding_dim=...)`
- `Data_integration_TRANSFORMER.ipynb` — `InferenceMultinomialTransformerModel(seq_cols=..., embedded_cols=..., target_col=..., d_model=...)`

Neither construction survives. A rollout model is now obtained from the trained model
that holds the weights — `trained.to_rollout()` (ADR-0007) — and the trained model takes
`embedder=ProjectedEmbedder(seq_cols=..., embedded_cols=..., target_col=...,
embedding_dim=<embedding_dim or d_model>)` in place of those columns; see
`docs/running-a-model.md` for the current call shape. Any checkpoint they reload also
predates the seam, so its `state_dict` keys need renaming before it will load — see the
same document.

The move does not break their data paths: all four open with a bootstrap cell that walks
up to the repo root — three by locating `pyproject.toml` and calling `os.chdir`,
`dataset_building.ipynb` by locating `Datasets/` — so relative paths resolve the same way
from here as they did from `notebooks/`.
