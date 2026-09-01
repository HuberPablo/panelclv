# 10 — The embedder seam (ADR-0005) never reached the feature docs, and one rationale names a symbol that does not exist

**Status:** ready-for-agent

Three places describe the projected embedding stack as *the* thing the model does. It is one
of two strategies, and it is not the default.

## The stale symbol

`src/panelclv/data_preparation/panel_dataset.py:463-465`, justifying standardisation:

> Why this is needed: the models push every non-embedded column through ONE shared
> `nn.Linear(len(covariate_cols), embedding_dim)` (see
> **`_MultinomialLSTMBackbone.covariate_proj`**).

There is no such attribute. `grep -rn covariate_proj src/` returns four hits, all in
`src/panelclv/models/embedders.py` (`:155`, `:160`, `:164`, `:184-186`) —
it is `ProjectedEmbedder.covariate_proj`. `_MultinomialLSTMBackbone`
(`src/panelclv/models/multinomial_lstm.py:96-138`) has no covariate path at all; it consumes
whatever width the embedder hands it, which is the whole point of ADR-0005.

`tests/test_docs_are_current.py` cannot catch this — it scans `.md` files only, and this is a
module docstring.

## The rationale is false under the default configuration

Both `_EMBEDDERS` entries are legal, and the pinned default in **both** developed models is
the published one, not the projected one:

- `src/panelclv/registry/model_registry.py:332` — `lstm` entry: `"embedder": "valendin"`
- `src/panelclv/registry/model_registry.py:350` — `transformer` entry: `"embedder": "valendin"`

`ValendinEmbedder.forward` (`src/panelclv/models/embedders.py:245-257`) concatenates each
covariate as its own raw channel:

```python
chunks = [
    self._emb_modules[self._emb_index[col]](x[:, :, i].long())
    if col in self.embedded_cols
    else x[:, :, i : i + 1].float()
    for i, col in enumerate(self.seq_cols)
]
return torch.cat(chunks, dim=-1)
```

No shared `Linear`, no `LayerNorm`, no parameter of its own for a covariate. So the "one
shared `Linear` mixes the columns, therefore raw magnitude decides the gradient" argument —
which is the stated reason standardisation exists — does not apply to the default path.

Standardisation is still *right* under `ValendinEmbedder` (a raw channel of std 27 sitting
beside embedding outputs of order 1 is its own problem, and `embedders.py:208-211` says so),
but that is a different argument from the one written down.

## `docs/feature_engineering.md` has the same gap

`docs/feature_engineering.md:276-286`:

> Inside the model, the two paths are:
> - **Embedded columns** → `nn.Embedding(cardinality, √cardinality + 1)` → `LayerNorm` →
>   `Linear(→ embedding_dim)` → `LayerNorm` … all other embeddings are **summed** …
> - **Continuous columns** → concatenated → `Linear(n → embedding_dim)` → `LayerNorm`

That is `ProjectedEmbedder` (`src/panelclv/models/embedders.py:117-197`) and nothing else.
The word **"embedder" does not appear anywhere in that chapter**, and neither does ADR-0005;
`grep -ni "embedder\|ADR-0005" docs/feature_engineering.md` returns nothing. The same false
rationale is repeated at `:304-305`.

## ADR-0005 has the mirror-image framing problem

`docs/adr/0005-embedder-seam.md:4-5`:

> theirs concatenates raw square-root-sized embeddings, **ours projects each feature to a
> common width and sums the context**.

True when written; the registry default has since been flipped to `"valendin"`, so "our"
models run the published strategy unless a `ModelSpec` overrides it — and `embedding_dim` is
then never even sampled (`src/panelclv/registry/model_registry.py:169-170`). The ADR states
no default explicitly, so this is framing drift rather than a false invariant, but it is what
sends a reader to the wrong class.

## Fix

1. `panel_dataset.py:463-465` — point at `ProjectedEmbedder.covariate_proj`, and rewrite the
   rationale so it covers both strategies: under `ProjectedEmbedder` the columns are summed
   through one shared `Linear` so raw magnitude decides the gradient; under `ValendinEmbedder`
   a covariate is one raw channel concatenated beside embedding outputs of order 1. Both need
   the inputs on a comparable scale, for related but distinct reasons.
2. `docs/feature_engineering.md:276-286` — say up front that "inside the model" means "inside
   the *embedder*", name both strategies, point at ADR-0005, and mark which is the current
   registry default. Fix `:304-305` the same way.
3. `docs/adr/0005-embedder-seam.md:4-5` — keep the sentence (it is the historical framing that
   motivated the seam) but add that the developed models now default to the published
   strategy, which is exactly the swap the seam was built to make possible.

## Related

Issue `07` — ADR-0004 has the mirror-image problem about which *protocol* the benchmark
inherits.
