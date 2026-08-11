"""Multinomial LSTM with dynamically configured embeddings.

Autoregressive sequence model that predicts a categorical distribution
P(y = 0), ..., P(y = K-1) over transaction-count classes at every time
step. Architecture is rebuilt from a schema, so categorical embeddings
are opt-in rather than hard-coded.


Constructor inputs
------------------
embedder : Embedder
    How features become a vector (ADR-0005). The model consumes
    `embedder.output_dim` and takes its softmax head size from
    `embedder.num_target_classes`; it does not know the strategy. Pass
    `ProjectedEmbedder` for the behaviour this model has always had, or
    `ValendinEmbedder` for the published one. The embedder also owns `seq_cols`,
    `embedded_cols` and `target_col`, so the model no longer takes them.
lstm_hidden_size
    Width of the LSTM hidden state and cell state.
dense_units
    Width of the dense prediction layer after the LSTM.
dropout
    Dropout applied to LSTM outputs before the prediction head.


Mandatory vs optional inputs
----------------------------
Both are now the embedder's concern: it requires the target to be a present,
embedded column, and accepts any number of further embeddings and covariates. The
smallest legal model has only the target column — input shape (B, T, 1).


Input tensor (forward x)
------------------------
Shape  : (B, T, F)  where F = len(embedder.seq_cols)
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
    encoded_input
        `embedder(x)` — whatever strategy the embedder implements, width
        `embedder.output_dim`. This is the LSTM input.

    lstm_out
        Per-step output of the LSTM (width = lstm_hidden_size).

    dense_out
        lstm_out passed through the dense layer (width = dense_units).

    logits
        Raw output scores over num_target_classes transaction-count classes.

The LSTM `input_size` is `embedder.output_dim`, so swapping the embedder changes
the model's width without changing a line here.


Validation
----------
The embedder validates its own columns. This module raises ValueError only when
x's last axis does not match `len(seq_cols)` — reported by the embedder, since it
owns the layout.
"""

from __future__ import annotations

import torch
import torch.distributions as dist
from torch import nn

from .embedders import Embedder


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class _MultinomialLSTMBackbone(nn.Module):
    """Embedder + LSTM + dense head, producing raw logits over `num_target_classes`."""

    def __init__(
        self,
        embedder: Embedder,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embedder = embedder
        # Read off the embedder rather than recomputing: it owns the column layout,
        # and its output_dim is the only thing this model needs to know about the
        # embedding strategy.
        self.seq_cols: list[str] = embedder.seq_cols
        self.target_col: str = embedder.target_col
        self.num_target_classes: int = embedder.num_target_classes

        self.lstm = nn.LSTM(
            input_size=embedder.output_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dense = nn.Linear(lstm_hidden_size, dense_units)
        self.output_layer = nn.Linear(dense_units, self.num_target_classes)

    def forward(self, x: torch.Tensor, state=None):
        # The embedder turns (B, T, F) into (B, T, embedder.output_dim); which
        # features were summed, concatenated or projected is its business.
        encoded_input = self.embedder(x)

        # lstm_out: (B, T, lstm_hidden_size). `state` is the LSTM recurrent
        # (hidden, cell) state, threaded across autoregressive rollout steps.
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
        embedder: Embedder,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            embedder=embedder,
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
        embedder: Embedder,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialLSTMBackbone(
            embedder=embedder,
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
