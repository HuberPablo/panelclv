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
embedded_cols : dict
    {col: num_categories, ...} — every column listed here is embedded with
    `nn.Embedding(num_categories, hidden_dim)`. Values in those columns must
    be integer class indices in [0, num_categories). Anything in `seq_cols`
    but not here is treated as a numerical covariate.
target_col : str = "Transactions"
    Which embedded column is the autoregressive target. Its cardinality
    sets the size of the output multinomial head (max_trans).
hidden_dim, memory_units, dense_units, dropout
    Standard LSTM hyper-parameters.


Mandatory vs optional inputs
----------------------------
Mandatory : `target_col` must appear in BOTH `seq_cols` AND
            `embedded_cols`. Cardinality there drives the output head size.
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

Inference (`InferenceMultinomialLSTMModel.forward`) — returns (sample, state):
    sample → (B, T, 1) count classes drawn from Categorical(softmax(logits)).
    state  → the LSTM hidden state, chainable across AR steps.


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
    - `target_col` missing from `seq_cols` or from `embedded_cols`,
    - x's last axis size != len(seq_cols).
(That `embedded_cols ⊆ seq_cols` and the cardinalities are valid is guaranteed
upstream by PanelConfig + prepare_dataset, so it is not re-checked here.)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributions as dist
from torch import nn


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# dynamic Embeddings (for columns in embedded_cols + transaction) :
# Automatic embedding size
def _emb_size(n: int) -> int:
    """Square-root heuristic for embedding dimensionality (legacy compatible)."""
    return int(n ** 0.5) + 1

# Embedding structure
def _cat_embedding(num_categories: int, out_dim: int) -> nn.Sequential:
    inner = _emb_size(num_categories)
    return nn.Sequential(
        nn.Embedding(num_categories, inner),
        nn.LayerNorm(inner),
        nn.Linear(inner, out_dim),
        nn.LayerNorm(out_dim),
    )


# -----------------------------------------

# Safeguard for seq_cols and embedded_cols :
def _validate_embedded_cols(
    seq_cols: Sequence[str],
    embedded_cols: dict[str, int],
    target_col: str,
) -> dict[str, int]:
    """Assert the MODEL-critical invariants and return embedded_cols as a dict.

    Only the facts the model itself depends on are checked here: that the input
    is a dict, and that the target is a present, embedded column (its cardinality
    is the softmax head size). The fuller spec validation — pinned-vs-"auto" types,
    cardinalities covering the data, and `embedded_cols ⊆ seq_cols` — already
    happened upstream in `PanelConfig._validate_embedded_cols` (static) and
    `prepare_dataset`/`resolve_embedded_cols` (data-dependent), and `select_features`
    only ever filters that resolved set, so it stays a subset by construction. We
    don't re-derive any of that.
    """
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
    return embedded_cols


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class _MultinomialLSTMBackbone(nn.Module):
    """Embeddings + LSTM + dense head, producing raw logits over `max_trans`."""

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        hidden_dim: int = 128,
        memory_units: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        embedded_cols = _validate_embedded_cols(seq_cols, embedded_cols, target_col)
        max_trans = int(embedded_cols[target_col])

        self.seq_cols: list[str] = list(seq_cols)
        self.target_col: str = target_col
        self.max_trans: int = max_trans

        # Embeddings — kept in seq_cols order. We use ModuleList + an index
        # map (instead of ModuleDict) because nn.ModuleDict rejects keys
        # containing dots, and real column names often have them
        # (e.g. "high.season").
        self._emb_cols: list[str] = [c for c in self.seq_cols if c in embedded_cols]

        # Automatic Embedding for columns in embedded_cols
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
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        hidden_dim: int = 128,
        memory_units: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            seq_cols=seq_cols,
            embedded_cols=embedded_cols,
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
    """Inference-mode LSTM. Returns (sample, state):

        sample : (B, T, 1) float — a count class drawn from
                 Categorical(softmax(logits)) at each step.
        state  : the LSTM hidden state, suitable for chaining autoregressive steps.

    The autoregressive Monte Carlo simulator threads `state` across steps and
    averages many sampled paths; sampling is the only inference behaviour the
    forecast needs, so it is hardcoded here (no mode switch).
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        hidden_dim: int = 128,
        memory_units: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            seq_cols=seq_cols,
            embedded_cols=embedded_cols,
            target_col=target_col,
            hidden_dim=hidden_dim,
            memory_units=memory_units,
            dense_units=dense_units,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.max_trans: int = self.backbone.max_trans

    def forward(self, x: torch.Tensor, state=None):
        logits, state = self.backbone(x, state)
        probs = torch.softmax(logits, dim=-1)
        sample = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        return sample, state
