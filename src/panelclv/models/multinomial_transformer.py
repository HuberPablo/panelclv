"""Multinomial Transformer with dynamically configured embeddings.

Mirror of `multinomial_lstm.py`, swapping the LSTM for a causal Transformer
encoder with sinusoidal positional encoding. Same dynamic input contract.


Constructor inputs
------------------
seq_cols : list[str]
    Ordered column names matching the LAST axis of the input tensor.
embedded_cols : dict
    {col: num_categories, ...} — every column listed here is embedded with
    a categorical embedding block: `nn.Embedding` into a square-root-heuristic
    width, LayerNorm, then a linear projection to `d_model` followed by a final
    LayerNorm. Values in those columns must be integer class indices in
    [0, num_categories). Anything in `seq_cols` but not here is treated as a
    numerical covariate (single shared linear projection to `d_model`).
target_col : str = "Transactions"
    Which embedded column is the autoregressive target. Its cardinality
    sets the size of the output multinomial head (num_target_classes). Must
    appear in both `seq_cols` and `embedded_cols`.
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

from typing import Sequence

import numpy as np
import torch
import torch.distributions as dist
from torch import nn


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _emb_size(n: int) -> int:
    """Square-root heuristic for embedding dimensionality."""
    return int(n ** 0.5) + 1


def _cat_embedding(num_categories: int, d_model: int) -> nn.Sequential:
    raw_embedding_dim = _emb_size(num_categories)
    return nn.Sequential(
        nn.Embedding(num_categories, raw_embedding_dim),
        nn.LayerNorm(raw_embedding_dim),
        nn.Linear(raw_embedding_dim, d_model),
        nn.LayerNorm(d_model),
    )


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
# Positional encoding
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
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        embedded_cols = _validate_embedded_cols(seq_cols, embedded_cols, target_col)
        num_target_classes = int(embedded_cols[target_col])

        self.seq_cols: list[str] = list(seq_cols)
        self.target_col: str = target_col
        self.num_target_classes: int = num_target_classes

        self.d_model: int = d_model

        # Dynamic embeddings — ModuleList + index map (dot-safe alternative to
        # ModuleDict, since real column names often contain '.'). -----------------------------------------------
        self._emb_cols: list[str] = [c for c in self.seq_cols if c in embedded_cols]
        self._emb_modules = nn.ModuleList(
            _cat_embedding(int(embedded_cols[c]), d_model) for c in self._emb_cols
        )
        self._emb_index: dict[str, int] = {c: i for i, c in enumerate(self._emb_cols)}

        # Numerical covariates: everything in seq_cols but not embedded.
        self.covariate_cols: list[str] = [
            c for c in self.seq_cols if c not in embedded_cols
        ]
        if self.covariate_cols:
            self.covariate_projection: nn.Module | None = nn.Sequential(
                nn.Linear(len(self.covariate_cols), d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.covariate_projection = None

        n_context_embs = len(self._emb_cols) - 1
        self.has_context: bool = (n_context_embs > 0) or (self.covariate_projection is not None)

        
        # Setup the positional encoding  -----------------------------------------------
        self.positional_encoding = SinePositionalEncoding(d_model, dropout=dropout)

        # Project the combined input representation to d_model.
        # With context:
        #     combined_input_repr = [context_repr, target_emb]
        #     shape: (B, T, 2 * d_model) -> (B, T, d_model)
        # Without context:
        #     combined_input_repr = target_emb
        #     shape: (B, T, d_model) -> (B, T, d_model)
        proj_in = d_model * (2 if self.has_context else 1)
        self.input_projection = nn.Linear(proj_in, d_model)
        
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
            nn.Linear(d_model, num_target_classes),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Standard causal mask: -inf above the diagonal, 0 elsewhere."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return torch.zeros(sz, sz, device=device).masked_fill(mask, float("-inf"))

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

        if self.covariate_projection is not None:
            numeric_covariates = torch.cat(numeric_covariate_chunks, dim=-1).float()
            numeric_covariate_repr = self.covariate_projection(numeric_covariates)
            context_repr = (
                numeric_covariate_repr
                if context_repr is None
                else context_repr + numeric_covariate_repr
            )

        if context_repr is None:
            return target_emb  # type: ignore[return-value]
        return torch.cat([context_repr, target_emb], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        only_last: bool = False,
    ) -> torch.Tensor:
        combined_input_repr = self._encode(x)

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
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        seq_len: int | None = None,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialTransformerBackbone(
            seq_cols=seq_cols,
            embedded_cols=embedded_cols,
            target_col=target_col,
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
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
        seq_len: int | None = None,  # accepted for API symmetry; unused here
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialTransformerBackbone(
            seq_cols=seq_cols,
            embedded_cols=embedded_cols,
            target_col=target_col,
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
