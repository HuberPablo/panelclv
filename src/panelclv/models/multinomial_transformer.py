"""Multinomial Transformer with dynamically configured embeddings.

Mirror of `multinomial_lstm.py`, swapping the LSTM for a causal Transformer
encoder with sinusoidal positional encoding. Same dynamic input contract.


Constructor inputs
------------------
embedder : Embedder
    How features become a vector (ADR-0005). Its `output_dim` is projected to
    `d_model` by `input_projection`, and its `num_target_classes` sets the softmax
    head size. The embedder owns `seq_cols`, `embedded_cols` and `target_col`, so
    the model no longer takes them.
d_model
    Width of token embeddings/projections and the Transformer encoder.
nhead
    Number of self-attention heads (must divide `d_model`).
num_encoder_layers
    Number of stacked causal Transformer encoder layers.
dropout
    Dropout applied in the positional encoding, attention, and feed-forward
    sublayers.


Architecture
------------
    target_emb
        Embedding of the autoregressive target column.

    context_repr
        Sum of all non-target categorical embeddings plus the projected
        numerical covariates.

    combined_input_repr
        [context_repr, target_emb] if context exists,
        otherwise target_emb only.

    token_repr
        combined_input_repr projected to d_model.

    positioned_repr
        token_repr plus sinusoidal positional encoding.

    encoder_out
        Output of the causal Transformer encoder.

    logits
        Raw output scores over num_target_classes transaction-count classes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from .embedders import Embedder


# ---------------------------------------------------------------------------

# Positional Encoding
class SinePositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    # d_model: the dimensionality of the input embeddings (and thus of the output encodings)
    # dropout: applied to the sum of input and positional encoding
    # max_len: maximum sequence length for which to precompute encodings. = Number of positions to encode

#  The lookup table: one row per possible position (up to max_len = 5000),
#  each row a d_model-wide vector. This is the thing that gets added to the inputs.
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) # Regularize the signal by randomly zeroing out some of the summed inputs during training. (not use at inference obviously)

        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model) -> Preallocate the positional encoding matrix with zeros. Each row corresponds to a position in the sequence, and each column corresponds to a dimension in the embedding space
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)# [o,1,2,...,max_len-1] -> shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class _MultinomialTransformerBackbone(nn.Module):
    """Embeddings + Transformer encoder + dense head, producing raw logits."""

    def __init__(
        self,
        embedder: Embedder,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.embedder = embedder
        # Read off the embedder rather than recomputing: it owns the column layout,
        # and its output_dim is the only thing this model needs to know about the
        # embedding strategy.
        self.seq_cols: list[str] = embedder.seq_cols
        self.target_col: str = embedder.target_col
        self.num_target_classes: int = embedder.num_target_classes
        self.d_model: int = d_model

        # Setup the positional encoding  -----------------------------------------------
        self.positional_encoding = SinePositionalEncoding(d_model, dropout=dropout)

        # Project whatever width the embedder produces onto d_model, which is the
        # only width the encoder stack understands. A ProjectedEmbedder at
        # embedding_dim=d_model gives (B, T, 2*d_model) -> (B, T, d_model); a
        # ValendinEmbedder gives its much narrower concatenation. Either way the
        # encoder below is unchanged.
        self.input_projection = nn.Linear(embedder.output_dim, d_model)
        
        # Setup the Transformer encoder and output head -----------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            # Pre-LN (norm_first=True) is incompatible with the nested-tensor fast
            # path, which only speeds up padded batches anyway. Our sequences are
            # fixed-length (no padding), so disable it explicitly — this silences the
            # "enable_nested_tensor is True, but ..." warning with no behavior change.
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.num_target_classes),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Standard causal mask: -inf above the diagonal, 0 elsewhere."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return torch.zeros(sz, sz, device=device).masked_fill(mask, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        only_last: bool = False,
    ) -> torch.Tensor:
        # The embedder turns (B, T, F) into (B, T, embedder.output_dim); which
        # features were summed, concatenated or projected is its business.
        combined_input_repr = self.embedder(x)

        token_repr = self.input_projection(combined_input_repr)

        positioned_repr = self.positional_encoding(token_repr)

        if mask is None:
            mask = self.generate_causal_mask(
                positioned_repr.shape[1],
                positioned_repr.device,
            )

        encoder_out = self.transformer_encoder(positioned_repr, mask)

        if only_last:
            encoder_out = encoder_out[:, -1:, :]

        normalized_out = self.norm(encoder_out)

        logits = self.output_linear(normalized_out)

        return logits


# ---------------------------------------------------------------------------
# Training-time wrapper
# ---------------------------------------------------------------------------


class MultinomialTransformerModel(nn.Module):
    """Training-mode Transformer returning raw logits.

    Forward output shape: (B, T, num_target_classes). Use with `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        embedder: Embedder,
        seq_len: int | None = None,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialTransformerBackbone(
            embedder=embedder,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.num_target_classes: int = self.backbone.num_target_classes

        if seq_len is not None:
            # Cache a causal mask for the common fixed-length training case.
            # persistent=False keeps it OUT of state_dict: it is fully
            # recomputable from seq_len, and persisting it would otherwise leak
            # an "_cached_mask" key into checkpoints that the inference model
            # (which has no such buffer) then rejects on load.
            self.register_buffer(
                "_cached_mask",
                self.backbone.generate_causal_mask(seq_len, torch.device("cpu")),
                persistent=False,
            )
            self._cached_seq_len: int | None = seq_len
        else:
            self._cached_mask = None
            self._cached_seq_len = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._cached_mask is not None
            and x.shape[1] == self._cached_seq_len
        ):
            mask = self._cached_mask.to(x.device)
        else:
            mask = None  # backbone builds one for the actual sequence length
        return self.backbone(x, mask=mask, only_last=False)


# ---------------------------------------------------------------------------
# Inference-time wrapper (sampling)
# ---------------------------------------------------------------------------


class InferenceMultinomialTransformerModel(nn.Module):
    """Inference-mode Transformer. Returns (sample, None):

        sample : (B, T, 1) float — a count class drawn from
                 Categorical(softmax(logits)) at each step.
        None   : the Transformer is stateless across calls (no hidden state to
                 thread), so the second tuple element is always None — kept for
                 call-signature parity with the inference LSTM.

    Sampling is the only inference behaviour the forecast needs, so it is
    hardcoded here (no mode switch).
    """

    def __init__(
        self,
        embedder: Embedder,
        seq_len: int | None = None,  # accepted for API symmetry; unused here
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialTransformerBackbone(
            embedder=embedder,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.num_target_classes: int = self.backbone.num_target_classes

        self._seq_len_hint = seq_len

    def forward(
        self,
        x: torch.Tensor,
        state=None,  # unused; kept for API parity with the LSTM inference model
        only_last: bool = False,
    ):
        logits = self.backbone(x, mask=None, only_last=only_last)
        probs = torch.softmax(logits, dim=-1)
        sample = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        return sample, None
