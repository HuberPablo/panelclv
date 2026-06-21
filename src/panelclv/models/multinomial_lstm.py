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
    a categorical embedding block: `nn.Embedding` into a square-root-heuristic
    width, LayerNorm, then a linear projection to `embedding_dim` followed by a
    final LayerNorm. Values in those columns must be integer class indices in
    [0, num_categories). Anything in `seq_cols` but not here is treated as a
    numerical covariate.
target_col : str = "Transactions"
    Which embedded column is the autoregressive target. Its cardinality
    sets the size of the output multinomial head (num_target_classes).
embedding_dim
    Width of categorical embeddings and numerical covariate projections.
lstm_hidden_size
    Width of the LSTM hidden state and cell state.
dense_units
    Width of the dense prediction layer after the LSTM.
dropout
    Dropout applied to LSTM outputs before the prediction head.


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
    raw logits of shape (B, T, num_target_classes).
    Use with `nn.CrossEntropyLoss` — integer class targets of shape
    (B, T) with values in [0, num_target_classes).

Inference (`InferenceMultinomialLSTMModel.forward`) — returns (sample, state):
    sample → (B, T, 1) count classes drawn from Categorical(softmax(logits)).
    state  → the LSTM hidden state, chainable across AR steps.


Architecture
------------
    target_emb
        Embedding of the autoregressive target column.

    context_repr
        Sum of all non-target categorical embeddings plus the projected
        numerical covariates.

    encoded_input
        [context_repr, target_emb] if context exists,
        otherwise target_emb only. This is the LSTM input.

    lstm_out
        Per-step output of the LSTM (width = lstm_hidden_size).

    dense_out
        lstm_out passed through the dense layer (width = dense_units).

    logits
        Raw output scores over num_target_classes transaction-count classes.

The LSTM `input_size` is `2 * embedding_dim` when context is present and
`embedding_dim` otherwise.


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
    """Square-root heuristic for embedding dimensionality."""
    return int(n ** 0.5) + 1

# Embedding structure
def _cat_embedding(num_categories: int, embedding_dim: int) -> nn.Sequential:
    raw_embedding_dim = _emb_size(num_categories)
    return nn.Sequential(
        nn.Embedding(num_categories, raw_embedding_dim), # raw because at the end the dim will be embedding_dim, but we want to apply layernorm before the projection to embedding_dim
        nn.LayerNorm(raw_embedding_dim),
        nn.Linear(raw_embedding_dim, embedding_dim), #out dim = embedding_dim
        nn.LayerNorm(embedding_dim),
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
    """Embeddings + LSTM + dense head, producing raw logits over `num_target_classes`."""

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        embedding_dim: int = 128,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        embedded_cols = _validate_embedded_cols(seq_cols, embedded_cols, target_col)
        num_target_classes = int(embedded_cols[target_col])

        self.seq_cols: list[str] = list(seq_cols)
        self.target_col: str = target_col
        self.num_target_classes: int = num_target_classes

        # Embeddings — kept in seq_cols order. We use ModuleList + an index
        # map (instead of ModuleDict) because nn.ModuleDict rejects keys
        # containing dots, and real column names often have them
        # (e.g. "high.season").
        self._emb_cols: list[str] = [c for c in self.seq_cols if c in embedded_cols]

        # Automatic Embedding for columns in embedded_cols
        self._emb_modules = nn.ModuleList(
            _cat_embedding(int(embedded_cols[c]), embedding_dim) for c in self._emb_cols
        )
        self._emb_index: dict[str, int] = {c: i for i, c in enumerate(self._emb_cols)}

        # Numerical covariates: everything in seq_cols but not embedded.
        self.covariate_cols: list[str] = [
            c for c in self.seq_cols if c not in embedded_cols
        ]
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

        lstm_input_size = embedding_dim * (2 if self.has_context else 1)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dense = nn.Linear(lstm_hidden_size, dense_units)
        self.output_layer = nn.Linear(dense_units, num_target_classes)

    # ------------------------------------------------------------------

    def _check_shape(self, x: torch.Tensor) -> None:
        if x.shape[-1] != len(self.seq_cols):
            raise ValueError(
                f"Expected x.shape[-1] == {len(self.seq_cols)} (= len(seq_cols)), "
                f"got {x.shape[-1]}"
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
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

        # `target_emb` is always assigned because the validator guarantees the
        # target column is in self._emb_index.
        if context_repr is None:
            return target_emb  # type: ignore[return-value]
        return torch.cat([context_repr, target_emb], dim=-1)

    def forward(self, x: torch.Tensor, state=None):
        # Build the LSTM input: concatenate the target embedding with the summed
        # context (other embeddings + covariate projection). With no context the
        # LSTM input is just the target embedding.
        encoded_input = self._encode(x)

        # encoded_input: (B, T, input_size) where input_size = 2 * embedding_dim
        # if context exists else embedding_dim. lstm_out: (B, T, lstm_hidden_size).
        # `state` is the LSTM recurrent (hidden, cell) state, threaded across AR steps.
        lstm_out, state = self.lstm(encoded_input, state)
        lstm_out = self.dropout(lstm_out)

        dense_out = self.dense(lstm_out)
        logits = self.output_layer(dense_out)

        return logits, state


# ---------------------------------------------------------------------------
# Training-time wrapper
# ---------------------------------------------------------------------------


class MultinomialLSTMModel(nn.Module):
    """Training-mode LSTM returning raw logits.

    Forward output shape: (B, T, num_target_classes). Use with `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        embedding_dim: int = 128,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            seq_cols=seq_cols,
            embedded_cols=embedded_cols,
            target_col=target_col,
            embedding_dim=embedding_dim,
            lstm_hidden_size=lstm_hidden_size,
            dense_units=dense_units,
            dropout=dropout,
        )
        # Hoist commonly accessed fields for convenience.
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.num_target_classes: int = self.backbone.num_target_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.backbone(x)
        return logits


# ---------------------------------------------------------------------------
# Inference-time wrapper (sampling)
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
        embedding_dim: int = 128,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            seq_cols=seq_cols,
            embedded_cols=embedded_cols,
            target_col=target_col,
            embedding_dim=embedding_dim,
            lstm_hidden_size=lstm_hidden_size,
            dense_units=dense_units,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.num_target_classes: int = self.backbone.num_target_classes

    def forward(self, x: torch.Tensor, state=None):
        logits, state = self.backbone(x, state)
        probs = torch.softmax(logits, dim=-1)
        sample = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        return sample, state
