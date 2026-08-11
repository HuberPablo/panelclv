# Extract the embedder as a swappable component

Status: ready-for-agent

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
