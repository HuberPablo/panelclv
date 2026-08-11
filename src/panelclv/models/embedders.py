"""How features become a vector, as a swappable component (ADR-0005).

A sequence model does not build its own embedding stack. It is handed an
**embedder**: an object that maps the raw feature tensor to the vector the model's
first layer consumes, and that advertises how wide that vector is. The model wires
itself to `output_dim` and never learns which strategy produced it.

    forward(x: (B, T, F)) -> (B, T, output_dim)

`F` is `len(seq_cols)`, and column `k` of the last axis holds `seq_cols[k]` at every
`(B, T)` — the layout `prepare_dataset` produces. Columns named in `embedded_cols`
carry integer class indices in `[0, cardinality)`; everything else is a numerical
covariate.

Two strategies live here because the thesis needs both, and they genuinely differ:

- ``ValendinEmbedder`` — the published architecture: one raw `Embedding(n, sqrt(n)+1)`
  per feature, concatenated. No normalisation, no projection, no covariate path.
- ``ProjectedEmbedder`` — our own: every feature is embedded, normalised and projected
  to one common width, the context is summed, and the target embedding is concatenated
  last.

Welding either into a model would force the frozen reference (ADR-0004) and the models
under development to share code that one of them must never change. Keeping them apart
means a new strategy is a new class here, not a new branch inside a model.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _emb_size(num_categories: int) -> int:
    """Square-root heuristic for embedding width — the paper's `emb_size`."""
    return int(num_categories ** 0.5) + 1


class Embedder(nn.Module):
    """Base class carrying the column bookkeeping both strategies need.

    Subclasses build their own modules and implement `forward`; this class only
    resolves which columns are embedded, which are covariates, and validates the
    invariants the *model* depends on:

    - the target is present in `seq_cols` (so it can be read out of `x`), and
    - the target is embedded (its cardinality is the softmax head size).

    The fuller spec validation — pinned-vs-"auto" types, cardinalities covering the
    data, `embedded_cols ⊆ seq_cols` — already happened upstream in
    `PanelConfig._validate_embedded_cols` and `prepare_dataset`, and `select_features`
    only filters that resolved set, so none of it is re-derived here.

    Attributes
    ----------
    output_dim : int
        Width of the vector `forward` returns. Set by each subclass before any
        forward pass, because a model wires its first layer to it at construction.
    num_target_classes : int
        Cardinality of the target column — the size of the model's softmax head.
    """

    output_dim: int

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
    ) -> None:
        super().__init__()

        if not isinstance(embedded_cols, dict):
            raise ValueError(
                "embedded_cols must be a {column: cardinality} dict "
                "(use PanelConfig.embedded_cols / prepare_dataset's data['embedded_cols'])"
            )
        embedded_cols = dict(embedded_cols)

        if target_col not in seq_cols:
            raise ValueError(
                f"target_col {target_col!r} not in seq_cols={list(seq_cols)}"
            )
        if target_col not in embedded_cols:
            raise ValueError(
                f"target_col {target_col!r} must appear in embedded_cols "
                f"(its cardinality drives the output head size)"
            )

        self.seq_cols: list[str] = list(seq_cols)
        self.target_col: str = target_col
        self.embedded_cols: dict[str, int] = embedded_cols
        self.num_target_classes: int = int(embedded_cols[target_col])

        # Embedded columns keep seq_cols order. ModuleList + an index map rather than
        # ModuleDict, because ModuleDict rejects keys containing dots and real column
        # names often have them (e.g. "high.season").
        self._emb_cols: list[str] = [c for c in self.seq_cols if c in embedded_cols]
        self._emb_index: dict[str, int] = {c: i for i, c in enumerate(self._emb_cols)}

        # Everything in seq_cols that is not embedded is a numerical covariate.
        self.covariate_cols: list[str] = [
            c for c in self.seq_cols if c not in embedded_cols
        ]

    def _check_shape(self, x: torch.Tensor) -> None:
        if x.shape[-1] != len(self.seq_cols):
            raise ValueError(
                f"Expected x.shape[-1] == {len(self.seq_cols)} (= len(seq_cols)), "
                f"got {x.shape[-1]}"
            )


class ProjectedEmbedder(Embedder):
    """Project every feature to a common width, sum the context, append the target.

    Per embedded column: `Embedding(n, sqrt(n)+1)` -> LayerNorm -> `Linear` to
    `embedding_dim` -> LayerNorm. The LayerNorm before the projection is why the raw
    embedding is square-root-sized rather than already `embedding_dim` wide.

    Numerical covariates go through one shared `Linear` -> LayerNorm to the same width.

    The context — every non-target embedding plus the projected covariates — is
    **summed**, then the target embedding is **concatenated** onto it, so the model
    always receives the target in a fixed trailing slot:

        output_dim = embedding_dim * 2   with context
        output_dim = embedding_dim       target-only panel (nothing to concatenate)
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        embedding_dim: int = 128,
    ) -> None:
        super().__init__(seq_cols, embedded_cols, target_col)

        self.embedding_dim: int = embedding_dim
        self._emb_modules = nn.ModuleList(
            nn.Sequential(
                nn.Embedding(int(self.embedded_cols[c]), _emb_size(int(self.embedded_cols[c]))),
                nn.LayerNorm(_emb_size(int(self.embedded_cols[c]))),
                nn.Linear(_emb_size(int(self.embedded_cols[c])), embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            for c in self._emb_cols
        )

        if self.covariate_cols:
            self.covariate_proj: nn.Module | None = nn.Sequential(
                nn.Linear(len(self.covariate_cols), embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
        else:
            self.covariate_proj = None

        # Context is everything except the target embedding.
        n_context_embs = len(self._emb_cols) - 1
        self.has_context: bool = (n_context_embs > 0) or (self.covariate_proj is not None)
        self.output_dim: int = embedding_dim * (2 if self.has_context else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)

        target_emb: torch.Tensor | None = None
        context_repr: torch.Tensor | None = None
        numeric_covariate_chunks: list[torch.Tensor] = []

        for i, col in enumerate(self.seq_cols):
            if col in self._emb_index:
                emb = self._emb_modules[self._emb_index[col]](x[:, :, i].long())
                if col == self.target_col:
                    target_emb = emb
                else:
                    context_repr = emb if context_repr is None else context_repr + emb
            else:
                numeric_covariate_chunks.append(x[:, :, i:i + 1])

        if self.covariate_proj is not None:
            numeric_covariates = torch.cat(numeric_covariate_chunks, dim=-1).float()
            numeric_covariate_repr = self.covariate_proj(numeric_covariates)
            context_repr = (
                numeric_covariate_repr
                if context_repr is None
                else context_repr + numeric_covariate_repr
            )

        # `target_emb` is always assigned: the base class guarantees the target is an
        # embedded column, so the loop above always hits it.
        if context_repr is None:
            return target_emb  # type: ignore[return-value]
        return torch.cat([context_repr, target_emb], dim=-1)


class ValendinEmbedder(Embedder):
    """The published strategy: raw `Embedding(n, sqrt(n)+1)` per feature, concatenated.

    Nothing follows the embedding — no LayerNorm, no projection to a common width.
    The result is a narrow input (roughly 12 dimensions on the paper's banking data,
    where the features are 52 weeks and the transaction-count classes), which is a
    large part of what makes the reference model smaller than ours.

    There is no covariate path: the paper's model reads week and transaction count
    only, both categorical. A numerical covariate is rejected rather than dropped,
    because silently ignoring a requested feature would make the benchmark differ
    from what the caller asked for without saying so.

        output_dim = sum of sqrt(n)+1 over every embedded column
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
    ) -> None:
        super().__init__(seq_cols, embedded_cols, target_col)

        if self.covariate_cols:
            raise ValueError(
                f"ValendinEmbedder has no covariate path, but seq_cols carries "
                f"{len(self.covariate_cols)} non-embedded column(s): "
                f"{self.covariate_cols}. The published model reads embedded features "
                f"only. Either drop them from seq_cols or use ProjectedEmbedder."
            )

        self._emb_modules = nn.ModuleList(
            nn.Embedding(int(self.embedded_cols[c]), _emb_size(int(self.embedded_cols[c])))
            for c in self._emb_cols
        )
        self.output_dim: int = sum(
            _emb_size(int(self.embedded_cols[c])) for c in self._emb_cols
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)

        # Concatenated in seq_cols order — nothing is summed, so every feature keeps
        # its own slot in the vector the LSTM sees.
        chunks = [
            self._emb_modules[self._emb_index[col]](x[:, :, i].long())
            for i, col in enumerate(self.seq_cols)
        ]
        return torch.cat(chunks, dim=-1)
