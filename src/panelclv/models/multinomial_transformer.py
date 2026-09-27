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
import torch.nn.functional as F

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


# Customers per piece when `forward_cached` reads the calibration window; bounds the
# GPU memory of that first call without changing its result.
_WARMUP_CHUNK = 512


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

    def forward_cached(self, x: torch.Tensor, cache: dict | None = None):
        """The same computation as `forward` in eval mode, one call per new period.

        Returns the logits at the LAST position of `x` — (B, 1, K) — and a cache that
        lets the next call process only its own periods. The cache holds, per encoder
        layer, the attention keys and values of every position seen so far, (B, H, P,
        d_model // H), plus the count P. A position's keys and values depend only on
        positions at or before it (the mask is causal), so they never change once
        computed, and attending from the new positions over cached plus new keys is
        exactly what the full causal forward pass computes for those positions.

        This turns the rollout's re-read of the whole growing context at every holdout
        step (O(t) per step) into one step's work (O(1) per step), with the same
        logits. It re-states `nn.TransformerEncoderLayer`'s pre-LN forward (norm_first,
        GELU, dropout inactive in eval) from that layer's own modules, so it holds only
        for the layer this backbone builds; `tests/test_transformer_cache.py` pins the
        equality.

        x : (B, T_new, F) — the whole calibration window on the first call, then one
            period at a time.
        """
        if cache is None and x.shape[0] > _WARMUP_CHUNK:
            # The first call reads the whole calibration window, whose attention
            # scores for every customer at once can outgrow the GPU although the cache
            # they leave behind fits easily. Customers never attend to each other, so
            # the window is read in customer chunks and the pieces concatenated.
            parts = [self.forward_cached(x[i:i + _WARMUP_CHUNK])
                     for i in range(0, x.shape[0], _WARMUP_CHUNK)]
            layers = range(len(parts[0][1]["k"]))
            return torch.cat([p[0] for p in parts]), {
                "pos": parts[0][1]["pos"],
                "k": [torch.cat([p[1]["k"][i] for p in parts]) for i in layers],
                "v": [torch.cat([p[1]["v"][i] for p in parts]) for i in layers],
            }
        past = 0 if cache is None else cache["pos"]
        h = self.input_projection(self.embedder(x))            # (B, T_new, d_model)
        # Positions continue from where the cache stopped, as in the full sequence.
        h = self.positional_encoding.dropout(
            h + self.positional_encoding.pe[:, past: past + h.shape[1]]
        )
        B, T, D = h.shape
        # New position i sits at absolute position past + i and may attend to
        # positions 0 .. past + i (True = attend). A single new period sees them all.
        mask = (
            torch.ones(T, past + T, dtype=torch.bool, device=h.device).tril(past)
            if T > 1 else None
        )
        keys, values = [], []
        for i, layer in enumerate(self.transformer_encoder.layers):
            attn = layer.self_attn
            heads = attn.num_heads
            # Pre-LN self-attention block: h + out_proj(attention(norm1(h))).
            q, k, v = F.linear(layer.norm1(h), attn.in_proj_weight,
                               attn.in_proj_bias).chunk(3, dim=-1)
            q, k, v = (t.view(B, T, heads, D // heads).transpose(1, 2) for t in (q, k, v))
            if cache is not None:
                k = torch.cat([cache["k"][i], k], dim=2)       # (B, H, past + T, dh)
                v = torch.cat([cache["v"][i], v], dim=2)
            keys.append(k)
            values.append(v)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            h = h + layer.dropout1(attn.out_proj(a.transpose(1, 2).reshape(B, T, D)))
            # Pre-LN feed-forward block: h + linear2(activation(linear1(norm2(h)))).
            h = h + layer.dropout2(layer.linear2(layer.dropout(
                layer.activation(layer.linear1(layer.norm2(h))))))
        logits = self.output_linear(self.norm(h[:, -1:, :]))
        return logits, {"pos": past + T, "k": keys, "v": values}


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
            # an "_cached_mask" key into checkpoints that a warm-start into a
            # freshly built model then rejects on load.
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

    def to_rollout(self) -> "RolloutMultinomialTransformerModel":
        """The rollout model paired with this one, over this model's own backbone.

        Same handover as the LSTM's (ADR-0007), including the consequence of sharing
        documented there: the two models are one set of weights in one mode, so the
        simulator's `.eval()` reaches this model's backbone too.

        The cached causal mask does not travel. The rollout window grows a period at
        a time, so a mask pinned to the training length would never fit, and the
        backbone builds one per call instead.
        """
        return RolloutMultinomialTransformerModel(self.backbone)


# ---------------------------------------------------------------------------
# Rollout-time wrapper (sampling)
# ---------------------------------------------------------------------------


class RolloutMultinomialTransformerModel(nn.Module):
    """Rollout-mode Transformer. Returns (sample, state):

        sample : (B, 1, 1) float — a count class drawn from Categorical(softmax(logits))
                 at the last position of the input.
        state  : the attention key/value cache of every position seen so far
                 (`_MultinomialTransformerBackbone.forward_cached`). Pass it back with
                 the next period to continue the sequence without re-reading it.

    Sampling is the only rollout behaviour the forecast needs, so it is
    hardcoded here (no mode switch).

    Built only by `MultinomialTransformerModel.to_rollout()`, which hands over the
    trained backbone it already holds (ADR-0007).
    """

    def __init__(self, backbone: _MultinomialTransformerBackbone) -> None:
        super().__init__()
        # Shared, not copied: these are the trained weights themselves.
        self.backbone = backbone
        self.seq_cols: list[str] = backbone.seq_cols
        self.target_col: str = backbone.target_col
        self.num_target_classes: int = backbone.num_target_classes

    def forward(self, x: torch.Tensor, state=None):
        logits, state = self.backbone.forward_cached(x, state)
        probs = torch.softmax(logits, dim=-1)
        sample = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        return sample, state
