"""The Valendin et al. (2022, IJRM) LSTM — a frozen reference implementation.

A transcription of the reference notebook's Keras model
(`Original_paper_model/banking_transactions_demo.ipynb`), layer for layer:

    week  ──► Embedding(52, 8) ──┐
                                 ├─► concat (12) ──► LSTM(128) ──► Dense(128) ──► Dense(K)
    trans ──► Embedding(K,  4) ──┘

That is the whole model. There is no normalisation, no projection to a common
width, no dropout, and no covariate path — the published architecture reads week
and transaction count only, both categorical.

Why this is not `models.MultinomialLSTMModel`
---------------------------------------------
Ours departs from the paper in two ways nobody chose: its embeddings pass through
LayerNorm and a projection to a common width, and it sums the context and
concatenates the target, giving a 256-wide LSTM input where the paper's is 12.
Renaming ours would give a benchmark that quietly differs from what it claims to
reproduce, so the two live side by side (ADR-0004): this module is frozen, and
`models.MultinomialLSTMModel` is free to develop.

For the same reason this module does **not** reuse
`models.multinomial_lstm._MultinomialLSTMBackbone`. Sharing it would mean the frozen
reference silently followed every change to the model under development. What *is*
shared is the infrastructure around the architecture — the embedder seam (ADR-0005),
the training loop, the Monte Carlo simulator and evaluation — applied identically to
every model, which is what makes the comparison isolate architecture.

Deliberate departures that stay
-------------------------------
Temporal validation split (ADR-0001) and Optuna tuning. Everything else matches.

Faithfulness check
------------------
The reference notebook's `model.summary()` reports these parameter counts, which
`tests/test_valendin_lstm.py` pins:

    embed_week    416      Embedding(52, 8)
    embed_trans    48      Embedding(12, 4)
    concat          -      (None, 155, 12)
    lstm        72192      LSTM(128) over a 12-wide input
    dense       16512      Dense(128)
    softmax      1548      Dense(12)

The LSTM is the one line that cannot match exactly: Keras carries a single bias
vector per gate, PyTorch carries two (`b_ih` and `b_hh`), so ours has 4 * 128 = 512
more parameters. That is a framework convention, not an architectural choice, and it
is the only difference.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributions as dist
from torch import nn

from panelclv.models.embedders import ValendinEmbedder

# The paper's sizes: `memory_units = 128`, `dense_units = 128` in the notebook.
# Read directly by the backbone rather than reaching it as a constructor argument.
# ADR-0004 freezes a benchmark's widths, and a default is only a default: an argument
# that can be passed is an architecture that can be changed, which is the same
# unfreezing the registry refuses to allow the search space to do.
_MEMORY_UNITS = 128
_DENSE_UNITS = 128


class _ValendinLSTMBackbone(nn.Module):
    """Embeddings -> LSTM -> Dense -> softmax head, exactly as published.

    The dense layer has no activation (the notebook leaves `activation=` commented
    out, so Keras applies the linear default), and the head emits raw logits — the
    published `Dense(..., activation='softmax')` is folded into the cross-entropy
    loss at training time and into sampling at rollout time, which is the same
    function computed in a numerically stabler order.
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
    ) -> None:
        super().__init__()

        # The embedder is the seam (ADR-0005), and this is the strategy that makes
        # the model the paper's: raw sqrt(n)+1 embeddings concatenated.
        #
        # The published model reads week and transaction count, both categorical,
        # with nowhere for a covariate to enter — so THIS class refuses one (ADR-0004).
        # The check lives here rather than in the embedder because it is a fact about
        # the benchmark's inputs, not about the embedding strategy: our own models may
        # use the same strategy on a panel that carries covariates. Dropping them
        # silently would make the benchmark differ from what the caller asked for
        # without saying so, which is the failure this guards.
        covariates = [c for c in seq_cols if c not in embedded_cols]
        if covariates:
            raise ValueError(
                f"The Valendin benchmark reads embedded features only, but seq_cols "
                f"carries {len(covariates)} non-embedded column(s): {covariates}. "
                f"The published model has no covariate path (ADR-0004). Either drop "
                f"them from seq_cols, or run one of the developed models in "
                f"`models/` — they accept covariates under either embedding strategy."
            )
        self.embedder = ValendinEmbedder(seq_cols, embedded_cols, target_col)

        self.seq_cols: list[str] = self.embedder.seq_cols
        self.target_col: str = self.embedder.target_col
        self.num_target_classes: int = self.embedder.num_target_classes

        self.lstm = nn.LSTM(
            input_size=self.embedder.output_dim,
            hidden_size=_MEMORY_UNITS,
            batch_first=True,
        )
        self.dense = nn.Linear(_MEMORY_UNITS, _DENSE_UNITS)
        self.output_layer = nn.Linear(_DENSE_UNITS, self.num_target_classes)

    def forward(self, x: torch.Tensor, state=None):
        encoded_input = self.embedder(x)                 # (B, T, 12) on the demo data
        lstm_out, state = self.lstm(encoded_input, state)
        dense_out = self.dense(lstm_out)
        logits = self.output_layer(dense_out)
        return logits, state


class ValendinLSTMModel(nn.Module):
    """Training-mode Valendin LSTM returning raw logits.

    Forward output shape: (B, T, num_target_classes). Use with `nn.CrossEntropyLoss`,
    which is the notebook's `sparse_categorical_crossentropy`.
    """

    def __init__(
        self,
        seq_cols: Sequence[str],
        embedded_cols: dict[str, int],
        target_col: str = "Transactions",
    ) -> None:
        super().__init__()
        self.backbone = _ValendinLSTMBackbone(
            seq_cols=seq_cols,
            embedded_cols=embedded_cols,
            target_col=target_col,
        )
        self.seq_cols: list[str] = self.backbone.seq_cols
        self.target_col: str = self.backbone.target_col
        self.num_target_classes: int = self.backbone.num_target_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.backbone(x)
        return logits

    def to_rollout(self) -> "RolloutValendinLSTMModel":
        """The rollout model paired with this one, over this model's own backbone.

        The benchmark declares its own pairing here, inside the frozen file, for the
        same reason `models/` does (ADR-0007): the pair is never assembled by a
        caller. ADR-0004 freezes the published *numbers*, not the surrounding code,
        and `scripts/validate_valendin_lstm.py` is the gate that proves they did not
        move.
        """
        return RolloutValendinLSTMModel(self.backbone)


class RolloutValendinLSTMModel(nn.Module):
    """Rollout-mode Valendin LSTM. Returns (sample, state).

    Same contract as `models.RolloutMultinomialLSTMModel`, so the shared Monte
    Carlo simulator drives this benchmark with no special-casing:

        sample : (B, T, 1) float — a count class drawn from Categorical(softmax(logits)).
        state  : the LSTM hidden state, threaded across autoregressive steps. This is
                 the notebook's `stateful=True` prediction LSTM, whose state it also
                 manages by hand across steps.

    Built only by `ValendinLSTMModel.to_rollout()`, which hands over the trained
    backbone it already holds — so the two can never be built with different sizes.
    """

    def __init__(self, backbone: _ValendinLSTMBackbone) -> None:
        super().__init__()
        # Shared, not copied: these are the trained weights themselves.
        self.backbone = backbone
        self.seq_cols: list[str] = backbone.seq_cols
        self.target_col: str = backbone.target_col
        self.num_target_classes: int = backbone.num_target_classes

    def forward(self, x: torch.Tensor, state=None):
        logits, state = self.backbone(x, state)
        probs = torch.softmax(logits, dim=-1)
        sample = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        return sample, state
