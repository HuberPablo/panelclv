"""Multinomial Transformer with dynamically configured embeddings.

Mirror of `multinomial_lstm.py`, swapping the LSTM for a causal Transformer
encoder with sinusoidal positional encoding. Same dynamic input contract:

    seq_cols    list[str]      Ordered column names matching x's last axis.
    input_spec  dict           {"embedded_cols": {col: num_categories, ...}}

Columns named in `input_spec["embedded_cols"]` are embedded; everything else
in `seq_cols` is a numerical covariate (single shared linear projection).
The AR target column (default `"Transactions"`) must appear in both lists.

Architecture
------------
    emb_target            embedding of the AR target column
    context_sum    sum of (all other embeddings) + projection of covariates
    h0          =  [context_sum, emb_target]   if context is present
                =  emb_target                   otherwise
    h          ->  Linear(in -> d_model) -> + positional -> encoder
                ->  output head over `max_trans` classes
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
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
# Positional encoding (unchanged from the legacy file)
# ---------------------------------------------------------------------------


class SinePositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
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
        input_spec: dict[str, Any],
        target_col: str = "Transactions",
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        embedded_cols = _validate_spec_against_seq(seq_cols, input_spec, target_col)
        max_trans = int(embedded_cols[target_col])

        self.seq_cols: list[str] = list(seq_cols)
        self.target_col: str = target_col
        self.max_trans: int = max_trans
        self.d_model: int = d_model

        # Dynamic embeddings — ModuleList + index map (dot-safe alternative to
        # ModuleDict, since real column names often contain '.').
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

        self.positional_encoding = SinePositionalEncoding(d_model, dropout=dropout)

        # Project [context_sum, emb_target] (2*d_model) or just emb_target
        # (d_model) down to d_model.
        proj_in = d_model * (2 if self.has_context else 1)
        self.input_projection = nn.Linear(proj_in, d_model)

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
            nn.Linear(d_model, max_trans),
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

        if self.covariate_projection is not None:
            covs = torch.cat(cov_chunks, dim=-1).float()
            proj = self.covariate_projection(covs)
            context_sum = proj if context_sum is None else context_sum + proj

        if context_sum is None:
            return emb_target  # type: ignore[return-value]
        return torch.cat([context_sum, emb_target], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        only_last: bool = False,
    ) -> torch.Tensor:
        h = self._encode(x)
        h = self.input_projection(h)
        h = self.positional_encoding(h)

        if mask is None:
            mask = self.generate_causal_mask(h.shape[1], h.device)

        h = self.transformer_encoder(h, mask)
        if only_last:
            h = h[:, -1:, :]
        h = self.norm(h)
        return self.output_linear(h)


# ---------------------------------------------------------------------------
# Training-time wrapper
# ---------------------------------------------------------------------------


class MultinomialTransformerModel(nn.Module):
    """Training-mode Transformer returning raw logits.

    Forward output shape: (B, T, max_trans). Use with `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        input_spec: dict[str, Any],
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
            input_spec=input_spec,
            target_col=target_col,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.max_trans: int = self.backbone.max_trans

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
# Inference-time wrapper (sample / expected / probs)
# ---------------------------------------------------------------------------


class InferenceMultinomialTransformerModel(nn.Module):
    """Inference-mode Transformer.

    Same modes as the inference LSTM:
        "sample"   -> class index from Categorical(softmax(logits)), shape (B, T, 1)
        "expected" -> E[Y] = sum_k k * P(y=k), shape (B, T, 1)
        "probs"    -> full P(y=k), shape (B, T, max_trans)

    The second tuple element is always None (the Transformer is stateless
    across calls).
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        input_spec: dict[str, Any],
        target_col: str = "Transactions",
        seq_len: int | None = None,  # accepted for API symmetry; unused here
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.0,
        mode: str = "sample",
    ) -> None:
        super().__init__()
        self.backbone = _MultinomialTransformerBackbone(
            seq_cols=seq_cols,
            input_spec=input_spec,
            target_col=target_col,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.max_trans: int = self.backbone.max_trans
        self.mode: str = mode
        self._seq_len_hint = seq_len

    def forward(
        self,
        x: torch.Tensor,
        state=None,                # unused; kept for API parity with the LSTM
        only_last: bool = False,
        mode: str | None = None,
    ):
        mode = mode or self.mode
        logits = self.backbone(x, mask=None, only_last=only_last)
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

        return out, None
