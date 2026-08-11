# Extract the embedder as a swappable component

Status: done

How features become a vector is currently welded into `_MultinomialLSTMBackbone.__init__`
and `_encode`: the module builds its own embedding stack and hard-codes sum-then-concat.
The Valendin reference needs a different strategy from the models under development,
so the two cannot share a hard-coded one (ADR-0005).

Extract an embedder object a model is given: a forward pass producing `(B, T, D)` and
an `output_dim` the model consumes without knowing the strategy. First two
implementations:

- **Valendin** — `Embedding(n, sqrt(n)+1)` per feature, concatenated, no
  normalisation, no projection, no covariate path.
- **Projected** — the current behaviour: embed, LayerNorm, project to a common width,
  LayerNorm, sum the context, concatenate with the target embedding.

The Transformer needs the same seam, so this is shared infrastructure rather than LSTM
work.

Done when: both models take an embedder, the seam is covered by shape tests that need
no training, and forecasts from the projected embedder match the current ones.

## Comments
Done in `79e7a3d`. `models/embedders.py` holds the seam: `forward((B, T, F)) -> (B, T,
output_dim)` plus an `output_dim` the model wires its first layer to. `ProjectedEmbedder`
is the existing behaviour, `ValendinEmbedder` the published one. The column bookkeeping
and validation that both backbones duplicated now live once on the `Embedder` base.

Both models take an embedder instead of `seq_cols`/`embedded_cols`/`target_col`/width;
the LSTM's `input_size` and the Transformer's `input_projection` are sized from
`output_dim`. Six construction sites in `tuning`, `experiments` and `scripts` updated.

"Forecasts from the projected embedder match the current ones" was **verified, not
assumed**: logits captured from both models before the change are reproduced bit for bit
(max abs diff 0.0) by loading the pre-refactor weights under one key rename,
`backbone.*` -> `backbone.embedder.*` for the embedding modules. 26 shape tests in
`tests/test_embedders.py`, none of which train anything.

**Consequence:** checkpoints written before this commit no longer load as-is. There are
**three** key renames, not one — the third is easy to miss, because the LSTM called its
covariate module `covariate_proj` while the Transformer called the same thing
`covariate_projection`, and `ProjectedEmbedder` unified them:

    backbone._emb_modules.*         -> backbone.embedder._emb_modules.*
    backbone.covariate_proj.*       -> backbone.embedder.covariate_proj.*
    backbone.covariate_projection.* -> backbone.embedder.covariate_proj.*   (Transformer)

`scripts/migrations/rename_embedder_checkpoint_keys.py` does all three (dry-runs by
default, idempotent). Verified on real files from `checkpoints/`: a migrated checkpoint
loads `strict=True` into the refactored model with its tensors bit-identical. 345
checkpoints under `checkpoints/` are affected; the migration has **not** been run on them,
since that rewrites Pablo's artifacts in place.

`ValendinEmbedder` *raises* if `seq_cols` carries a non-embedded column rather than
dropping it. Confirmed by Pablo as intended: the benchmark takes no covariates. Recorded
in ADR-0004 so it stays settled.
