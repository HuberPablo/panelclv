"""Multinomial LSTM with attention over its own past hidden states.

The LSTM of `multinomial_lstm`, with one addition: at every step the head also reads
a context vector — single-head scaled dot-product attention from the current LSTM
output over all LSTM outputs up to and including this step. The LSTM carries the
history in a fixed-size state; attention lets the head look back at any earlier week
directly (for instance, the same week a year ago) instead of relying on the state to
have kept it. This is the LSTM-with-attention design of the recurrent-attention
literature, and the counterpart of the Transformer, which uses attention *instead of*
recurrence.

It is a classifier driving a simulator, like every model here: a softmax over
transaction-count classes at each step, trained by cross-entropy against a class
index, forecast by sampling one period at a time.


Constructor inputs
------------------
embedder, lstm_hidden_size, dense_units, dropout
    As in `MultinomialLSTMModel`. The attention adds no width of its own: its query,
    key and value all live at `lstm_hidden_size`.


Input / output
--------------
Input  : (B, T, F) float, F = len(embedder.seq_cols).
Training (`MultinomialLSTMAttentionModel.forward`): logits (B, T, num_target_classes).
Rollout (`RolloutMultinomialLSTMAttentionModel.forward`): (sample (B, T, 1), state).


Architecture
------------
    encoded  = embedder(x)                           (B, T, embedder.output_dim)
    h        = LSTM(encoded)                         (B, T, H)
    context  = causal attention of h over h          (B, T, H)
    logits   = output(dense([h, context]))           (B, T, K)

Attention at step t sees steps 0..t only, so the model is causal and trains on whole
sequences like the plain LSTM.


State
-----
The rollout state is `(lstm_state, memory)`: the LSTM's (hidden, cell) pair and every
LSTM output produced so far, `memory` of shape (B, T_seen, H). A call extends
`memory` with its own outputs and attends over the result, so stepping one period at
a time reproduces a full forward pass exactly. That is what lets it forecast through
`forecast_recurrent`, whose simulator threads the state without looking inside it.
"""

from __future__ import annotations

import math

import torch
import torch.distributions as dist
from torch import nn

from .embedders import Embedder


class _LSTMAttentionBackbone(nn.Module):
    """Embedder + LSTM + causal attention over past outputs + dense head -> logits."""

    def __init__(
        self,
        embedder: Embedder,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.seq_cols: list[str] = embedder.seq_cols
        self.target_col: str = embedder.target_col
        self.num_target_classes: int = embedder.num_target_classes

        self.lstm = nn.LSTM(
            input_size=embedder.output_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Single-head attention: learned query, key and value projections of the
        # LSTM outputs, all at the hidden width.
        self.query = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.key = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.value = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.dense = nn.Linear(2 * lstm_hidden_size, dense_units)
        self.output_layer = nn.Linear(dense_units, self.num_target_classes)

    def forward(self, x: torch.Tensor, state=None):
        lstm_state, memory = state if state is not None else (None, None)

        # h: (B, T, H), the LSTM outputs of this call's steps.
        h, lstm_state = self.lstm(self.embedder(x), lstm_state)
        h = self.dropout(h)

        # Keys and values span every step seen so far: the earlier calls' outputs
        # held in `memory`, then this call's. (B, P + T, H) with P = steps before.
        seen = h if memory is None else torch.cat([memory, h], dim=1)
        past = seen.shape[1] - h.shape[1]

        # scores: (B, T, P + T). Query step i sits at absolute position past + i and
        # may attend to positions 0 .. past + i, never later ones.
        scores = self.query(h) @ self.key(seen).transpose(1, 2)
        scores = scores / math.sqrt(h.shape[-1])
        T, S = h.shape[1], seen.shape[1]
        future = torch.ones(T, S, dtype=torch.bool, device=h.device).triu(past + 1)
        scores = scores.masked_fill(future, float("-inf"))
        context = torch.softmax(scores, dim=-1) @ self.value(seen)   # (B, T, H)

        logits = self.output_layer(self.dense(torch.cat([h, context], dim=-1)))
        return logits, (lstm_state, seen)


class MultinomialLSTMAttentionModel(nn.Module):
    """Training-mode LSTM with attention, returning raw logits (B, T, K)."""

    def __init__(
        self,
        embedder: Embedder,
        lstm_hidden_size: int = 64,
        dense_units: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _LSTMAttentionBackbone(
            embedder=embedder,
            lstm_hidden_size=lstm_hidden_size,
            dense_units=dense_units,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.num_target_classes: int = self.backbone.num_target_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.backbone(x)
        return logits

    def to_rollout(self) -> "RolloutMultinomialLSTMAttentionModel":
        """The rollout model over this model's own backbone, shared not copied (ADR-0007)."""
        return RolloutMultinomialLSTMAttentionModel(self.backbone)


class RolloutMultinomialLSTMAttentionModel(nn.Module):
    """Rollout-mode LSTM with attention. Returns (sample, state):

        sample : (B, T, 1) float — a count class drawn from Categorical(softmax(logits)).
        state  : (lstm_state, memory), chainable across autoregressive steps.

    Built only by `MultinomialLSTMAttentionModel.to_rollout()` (ADR-0007).
    """

    def __init__(self, backbone: _LSTMAttentionBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.seq_cols: list[str] = backbone.seq_cols
        self.target_col: str = backbone.target_col
        self.num_target_classes: int = backbone.num_target_classes

    def forward(self, x: torch.Tensor, state=None):
        logits, state = self.backbone(x, state)
        probs = torch.softmax(logits, dim=-1)
        sample = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        return sample, state
