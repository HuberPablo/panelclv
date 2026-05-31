"""Multinomial LSTM with dynamically configured embeddings.

Autoregressive sequence model that predicts a categorical distribution
P(y = 0), ..., P(y = K-1) over transaction-count classes at every time
step. Architecture is rebuilt from a schema, so categorical embeddings
are opt-in rather than hard-coded.


Constructor inputs
------------------
seq_cols : list[str]
    Ordered column names matching the LAST axis of the input tensor.
    Example: ["Transactions", "week_sin", "week_cos", "Gender"].
input_spec : dict
    {"embedded_cols": {col: num_categories, ...}} — every column listed
    here is embedded with `nn.Embedding(num_categories, hidden_dim)`.
    Values in those columns must be integer class indices in
    [0, num_categories). Anything in `seq_cols` but not here is treated as
    a numerical covariate.
target_col : str = "Transactions"
    Which embedded column is the autoregressive target. Its cardinality
    sets the size of the output multinomial head (max_trans).
hidden_dim, memory_units, dense_units, dropout
    Standard LSTM hyper-parameters.


Mandatory vs optional inputs
----------------------------
Mandatory : `target_col` must appear in BOTH `seq_cols` AND
            `input_spec["embedded_cols"]`. Cardinality there drives
            the output head size.
Optional  : any number of additional categorical embeddings (week_idx,
            month_idx, cohort_idx, ...) and any number of numerical
            covariates. The smallest legal model has only the target
            column — input shape (B, T, 1).


Input tensor (forward x)
------------------------
Shape  : (B, T, F)  where F = len(seq_cols)
dtype  : float32 (categorical columns are cast to long internally)
Layout : column k holds the value for `seq_cols[k]` at every (B, T).


Output
------
Training (`MultinomialLSTMModel.forward`):
    raw logits of shape (B, T, max_trans).
    Use with `nn.CrossEntropyLoss` — integer class targets of shape
    (B, T) with values in [0, max_trans).

Inference (`InferenceMultinomialLSTMModel.forward`) — returns (out, state):
    mode = "sample"   → (B, T, 1) class indices drawn from
                        Categorical(softmax(logits)).
    mode = "expected" → (B, T, 1) E[Y] = sum_k k * P(y=k).
    mode = "probs"    → (B, T, max_trans) full P(y=k) tensor.
    state is the LSTM hidden state, chainable across AR steps.


Architecture
------------
    emb_target           embedding of the AR target column
    context_sum    sum of (all other embeddings) + projection of covariates
    LSTM input  =  [context_sum, emb_target]   if any context exists
                =  emb_target                   otherwise
The LSTM `input_size` is `2 * hidden_dim` when context is present and
`hidden_dim` otherwise — no zero-padding tricks.


Validation
----------
Raises ValueError on:
    - `target_col` missing from `seq_cols` or from `input_spec["embedded_cols"]`,
    - `input_spec` referencing columns absent from `seq_cols`,
    - x's last axis size != len(seq_cols).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributions as dist
from torch import nn


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _emb_size(n: int) -> int:
    """Square-root heuristic for embedding dimensionality (legacy compatible)."""
    return int(n ** 0.5) + 1


def _cat_embedding(num_categories: int, out_dim: int) -> nn.Sequential:
    inner = _emb_size(num_categories)
    return nn.Sequential(
        nn.Embedding(num_categories, inner),
        nn.LayerNorm(inner),
        nn.Linear(inner, out_dim),
        nn.LayerNorm(out_dim),
    )


def _validate_spec_against_seq(
    seq_cols: Sequence[str],
    input_spec: dict[str, Any],
    target_col: str,
) -> dict[str, int]:
    """Return the validated, dict-form embedded_cols mapping."""
    if not isinstance(input_spec, dict) or "embedded_cols" not in input_spec:
        raise ValueError(
            "input_spec must be a dict containing 'embedded_cols' "
            "(see configs.transformations_spec.validate_input_spec)"
        )
    embedded_cols = dict(input_spec["embedded_cols"])

    if target_col not in seq_cols:
        raise ValueError(
            f"target_col {target_col!r} not in seq_cols={list(seq_cols)}"
        )
    if target_col not in embedded_cols:
        raise ValueError(
            f"target_col {target_col!r} must appear in input_spec['embedded_cols'] "
            f"(its cardinality drives the output head size)"
        )
    unknown = [c for c in embedded_cols if c not in seq_cols]
    if unknown:
        raise ValueError(
            f"input_spec['embedded_cols'] references columns not in seq_cols: {unknown}"
        )
    return embedded_cols


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class _MultinomialLSTMBackbone(nn.Module):
    """Embeddings + LSTM + dense head, producing raw logits over `max_trans`."""

    def __init__(
        self,
        seq_cols: Sequence[str],
        input_spec: dict[str, Any],
        target_col: str = "Transactions",
        hidden_dim: int = 128,
        memory_units: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        embedded_cols = _validate_spec_against_seq(seq_cols, input_spec, target_col)
        max_trans = int(embedded_cols[target_col])

        self.seq_cols: list[str] = list(seq_cols)
        self.target_col: str = target_col
        self.max_trans: int = max_trans
        self.hidden_dim: int = hidden_dim

        # Embeddings — kept in seq_cols order. We use ModuleList + an index
        # map (instead of ModuleDict) because nn.ModuleDict rejects keys
        # containing dots, and real column names often have them
        # (e.g. "high.season").
        self._emb_cols: list[str] = [c for c in self.seq_cols if c in embedded_cols]
        self._emb_modules = nn.ModuleList(
            _cat_embedding(int(embedded_cols[c]), hidden_dim) for c in self._emb_cols
        )
        self._emb_index: dict[str, int] = {c: i for i, c in enumerate(self._emb_cols)}

        # Numerical covariates: everything in seq_cols but not embedded.
        self.covariate_cols: list[str] = [
            c for c in self.seq_cols if c not in embedded_cols
        ]
        if self.covariate_cols:
            self.covariate_proj: nn.Module | None = nn.Sequential(
                nn.Linear(len(self.covariate_cols), hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.covariate_proj = None

        # Context is everything except the target embedding.
        n_context_embs = len(self._emb_cols) - 1
        self.has_context: bool = (n_context_embs > 0) or (self.covariate_proj is not None)

        lstm_input_size = hidden_dim * (2 if self.has_context else 1)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=memory_units,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dense = nn.Linear(memory_units, dense_units)
        self.output_layer = nn.Linear(dense_units, max_trans)

    # ------------------------------------------------------------------

    def _check_shape(self, x: torch.Tensor) -> None:
        if x.shape[-1] != len(self.seq_cols):
            raise ValueError(
                f"Expected x.shape[-1] == {len(self.seq_cols)} (= len(seq_cols)), "
                f"got {x.shape[-1]}"
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)

        emb_target: torch.Tensor | None = None
        context_sum: torch.Tensor | None = None
        cov_chunks: list[torch.Tensor] = []

        for i, col in enumerate(self.seq_cols):
            if col in self._emb_index:
                emb = self._emb_modules[self._emb_index[col]](x[:, :, i].long())
                if col == self.target_col:
                    emb_target = emb
                else:
                    context_sum = emb if context_sum is None else context_sum + emb
            else:
                cov_chunks.append(x[:, :, i:i + 1])

        if self.covariate_proj is not None:
            covs = torch.cat(cov_chunks, dim=-1).float()
            proj = self.covariate_proj(covs)
            context_sum = proj if context_sum is None else context_sum + proj

        # `emb_target` is always assigned because the validator guarantees the
        # target column is in self._emb_index.
        if context_sum is None:
            return emb_target  # type: ignore[return-value]
        return torch.cat([context_sum, emb_target], dim=-1)

    def forward(self, x: torch.Tensor, hidden=None):
        h = self._encode(x)
        lstm_out, hidden = self.lstm(h, hidden)
        lstm_out = self.dropout(lstm_out)
        return self.output_layer(self.dense(lstm_out)), hidden


# ---------------------------------------------------------------------------
# Training-time wrapper
# ---------------------------------------------------------------------------


class MultinomialLSTMModel(nn.Module):
    """Training-mode LSTM returning raw logits.

    Forward output shape: (B, T, max_trans). Use with `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        input_spec: dict[str, Any],
        target_col: str = "Transactions",
        hidden_dim: int = 128,
        memory_units: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            seq_cols=seq_cols,
            input_spec=input_spec,
            target_col=target_col,
            hidden_dim=hidden_dim,
            memory_units=memory_units,
            dense_units=dense_units,
            dropout=dropout,
        )
        # Hoist commonly accessed fields for convenience.
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.max_trans: int = self.backbone.max_trans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.backbone(x)
        return logits


# ---------------------------------------------------------------------------
# Inference-time wrapper (sample / expected / probs)
# ---------------------------------------------------------------------------


class InferenceMultinomialLSTMModel(nn.Module):
    """Inference-mode LSTM. Returns one of:

        - mode="sample":   a class index drawn from Categorical(softmax(logits)).
                           Output shape (B, T, 1) float.
        - mode="expected": E[Y] = sum_k k * P(y=k). Shape (B, T, 1).
        - mode="probs":    the full P(y=k) tensor. Shape (B, T, max_trans).

    The second tuple element is the LSTM hidden state, suitable for chaining
    autoregressive steps.
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        input_spec: dict[str, Any],
        target_col: str = "Transactions",
        hidden_dim: int = 128,
        memory_units: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
        mode: str = "sample",
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            seq_cols=seq_cols,
            input_spec=input_spec,
            target_col=target_col,
            hidden_dim=hidden_dim,
            memory_units=memory_units,
            dense_units=dense_units,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.max_trans: int = self.backbone.max_trans
        self.mode: str = mode

    def forward(
        self,
        x: torch.Tensor,
        state=None,
        only_last: bool = False,  # kept for API parity with the Transformer
        mode: str | None = None,
    ):
        mode = mode or self.mode
        logits, state = self.backbone(x, state)
        probs = torch.softmax(logits, dim=-1)

        if mode == "sample":
            out = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        elif mode == "expected":
            k = torch.arange(self.max_trans, device=probs.device, dtype=probs.dtype)
            out = (probs * k).sum(dim=-1, keepdim=True)
        elif mode == "probs":
            out = probs
        else:
            raise ValueError(f"Unknown inference mode: {mode!r}")

        if only_last:
            out = out[:, -1:, :]
        return out, state
