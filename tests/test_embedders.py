"""Shape and strategy tests for the embedder seam (ADR-0005).

How features become a vector is a swappable component: an embedder maps a raw
`(B, T, F)` feature tensor to `(B, T, output_dim)`, and the model wires its first
layer to `output_dim` without knowing the strategy. Everything here runs on random
tensors — the seam is testable on shapes alone, with nothing trained.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from panelclv.models.embedders import (  # noqa: E402
    ProjectedEmbedder,
    ValendinEmbedder,
    _emb_size,
)

# A panel with both kinds of column: two categorical (one of them the target) and
# two numerical covariates. Column order matches the last axis of `x`.
SEQ_COLS = ["Transactions", "week_idx", "spend_norm", "tenure_norm"]
EMBEDDED_COLS = {"Transactions": 6, "week_idx": 52}
TARGET = "Transactions"
B, T = 3, 7


def make_x(seq_cols=SEQ_COLS, embedded_cols=EMBEDDED_COLS, seed=0):
    """A valid input tensor: class indices in embedded columns, floats elsewhere."""
    g = torch.Generator().manual_seed(seed)
    cols = []
    for name in seq_cols:
        if name in embedded_cols:
            cols.append(torch.randint(0, embedded_cols[name], (B, T), generator=g).float())
        else:
            cols.append(torch.randn(B, T, generator=g))
    return torch.stack(cols, dim=-1)


# ---------------------------------------------------------------------------
# The contract both embedders satisfy
# ---------------------------------------------------------------------------


def _both_embedders():
    return [
        ProjectedEmbedder(SEQ_COLS, EMBEDDED_COLS, TARGET, embedding_dim=16),
        ValendinEmbedder(SEQ_COLS, EMBEDDED_COLS, TARGET),
    ]


@pytest.mark.parametrize("embedder", _both_embedders())
def test_forward_returns_declared_output_dim(embedder):
    """The forward pass produces exactly the width the embedder advertises."""
    x = make_x(embedder.seq_cols)
    out = embedder(x)
    assert out.shape == (B, T, embedder.output_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("embedder", _both_embedders())
def test_output_dim_is_an_int_known_before_any_forward(embedder):
    """A model wires its first layer to `output_dim` at construction time."""
    assert isinstance(embedder.output_dim, int)
    assert embedder.output_dim > 0


@pytest.mark.parametrize("embedder", _both_embedders())
def test_rejects_wrong_feature_count(embedder):
    """A tensor whose last axis is not len(seq_cols) is a caller error, not a silent one."""
    x = make_x(embedder.seq_cols)
    with pytest.raises(ValueError, match="seq_cols"):
        embedder(x[..., :-1])


# ---------------------------------------------------------------------------
# ProjectedEmbedder — our own strategy
# ---------------------------------------------------------------------------


def test_projected_output_dim_doubles_when_context_exists():
    """Context present -> [context, target] concatenated, so twice the width."""
    emb = ProjectedEmbedder(SEQ_COLS, EMBEDDED_COLS, TARGET, embedding_dim=16)
    assert emb.output_dim == 32


def test_projected_output_dim_is_single_width_for_a_target_only_panel():
    """The smallest legal panel is the target alone: no context to concatenate."""
    emb = ProjectedEmbedder(["Transactions"], {"Transactions": 6}, TARGET, embedding_dim=16)
    assert emb.output_dim == 16
    assert emb(make_x(["Transactions"], {"Transactions": 6})).shape == (B, T, 16)


def test_projected_matches_a_hand_built_reference():
    """Pin the strategy: embed -> LayerNorm -> project -> LayerNorm, sum context,
    concatenate the target embedding last.

    Written out by hand so a change to the combination rule fails here rather than
    silently altering every forecast.
    """
    emb = ProjectedEmbedder(SEQ_COLS, EMBEDDED_COLS, TARGET, embedding_dim=16).eval()
    x = make_x()

    with torch.no_grad():
        got = emb(x)

        # Re-run the pipeline by hand from the embedder's own modules.
        target_vec = emb._emb_modules[emb._emb_index["Transactions"]](x[:, :, 0].long())
        week_vec = emb._emb_modules[emb._emb_index["week_idx"]](x[:, :, 1].long())
        covariates = torch.cat([x[:, :, 2:3], x[:, :, 3:4]], dim=-1).float()
        context = week_vec + emb.covariate_proj(covariates)
        want = torch.cat([context, target_vec], dim=-1)

    assert torch.allclose(got, want, atol=1e-6)


def test_projected_target_embedding_occupies_the_trailing_half():
    """The model's first layer relies on the target sitting last; pin the order."""
    emb = ProjectedEmbedder(SEQ_COLS, EMBEDDED_COLS, TARGET, embedding_dim=16).eval()
    x = make_x()
    with torch.no_grad():
        got = emb(x)
        target_vec = emb._emb_modules[emb._emb_index["Transactions"]](x[:, :, 0].long())
    assert torch.allclose(got[..., 16:], target_vec, atol=1e-6)


# ---------------------------------------------------------------------------
# ValendinEmbedder — the paper's strategy
# ---------------------------------------------------------------------------


def test_valendin_output_dim_is_the_sum_of_raw_embedding_widths():
    """Raw sqrt(n)+1 vectors concatenated: no projection to a common width.

    With the paper's banking cardinalities (52 weeks, and counts 0..max) this is the
    roughly 12-dimensional LSTM input the notebook builds.
    """
    emb = ValendinEmbedder(["Transactions", "week_idx"], EMBEDDED_COLS, TARGET)
    assert emb.output_dim == _emb_size(6) + _emb_size(52)
    assert emb.output_dim == 3 + 8


def test_valendin_concatenates_in_seq_cols_order():
    """Column order in, embedding order out — nothing is summed."""
    emb = ValendinEmbedder(["Transactions", "week_idx"], EMBEDDED_COLS, TARGET).eval()
    x = make_x(["Transactions", "week_idx"])
    with torch.no_grad():
        got = emb(x)
        first = emb._emb_modules[0](x[:, :, 0].long())
        second = emb._emb_modules[1](x[:, :, 1].long())
    assert torch.allclose(got, torch.cat([first, second], dim=-1), atol=1e-6)


def test_valendin_applies_no_normalisation_or_projection():
    """The strategy is a bare nn.Embedding per feature — nothing after it.

    If a LayerNorm or Linear ever creeps in, the benchmark stops being the published
    architecture (ADR-0004), so it is pinned structurally.
    """
    emb = ValendinEmbedder(["Transactions", "week_idx"], EMBEDDED_COLS, TARGET)
    for block in emb._emb_modules:
        assert isinstance(block, torch.nn.Embedding)


def test_valendin_concatenates_a_numerical_covariate_untouched():
    """A covariate rides along as its own channel, with nothing applied to it.

    The paper's own model has no covariate to carry — that restriction belongs to the
    benchmark (ADR-0004) and is enforced in `benchmarks/valendin_lstm.py`, not here.
    The strategy itself is "concatenate, add no arithmetic", and a covariate is the
    case where the arithmetic added is literally none: `prepare_dataset` has already
    standardised the channel, so it appears in the output exactly as it arrived.
    """
    emb = ValendinEmbedder(SEQ_COLS, EMBEDDED_COLS, TARGET).eval()
    x = make_x(SEQ_COLS)

    # Two embedded columns contribute sqrt(n)+1 each; two covariates one channel each.
    assert emb.output_dim == _emb_size(6) + _emb_size(52) + 2

    with torch.no_grad():
        got = emb(x)
    assert got.shape == (B, T, emb.output_dim)

    # The covariates occupy the trailing slots, in seq_cols order, bit-for-bit.
    assert torch.equal(got[:, :, -2], x[:, :, 2])   # spend_norm
    assert torch.equal(got[:, :, -1], x[:, :, 3])   # tenure_norm


def test_valendin_width_is_unchanged_without_covariates():
    """The published width, pinned: the benchmark's arithmetic must not have moved.

    Adding a covariate path is only safe for the frozen reference (ADR-0004) because
    it contributes nothing when there is no covariate. This is that guarantee.
    """
    emb = ValendinEmbedder(["Transactions", "week_idx"], EMBEDDED_COLS, TARGET)
    assert emb.output_dim == _emb_size(6) + _emb_size(52)
    assert emb.output_dim == 3 + 8


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def test_target_must_be_embedded_because_its_cardinality_is_the_head_size():
    with pytest.raises(ValueError, match="embedded_cols"):
        ProjectedEmbedder(SEQ_COLS, {"week_idx": 52}, TARGET, embedding_dim=16)


def test_target_must_appear_in_seq_cols():
    with pytest.raises(ValueError, match="seq_cols"):
        ProjectedEmbedder(["week_idx"], {"week_idx": 52}, TARGET, embedding_dim=16)


@pytest.mark.parametrize("embedder", _both_embedders())
def test_num_target_classes_comes_from_the_target_cardinality(embedder):
    """The softmax head size travels with the embedder, not the model."""
    assert embedder.num_target_classes == EMBEDDED_COLS[TARGET]


def test_emb_size_matches_the_papers_heuristic():
    """`int(sqrt(n)) + 1`, the notebook's `emb_size`."""
    for n in (1, 4, 6, 52, 100):
        assert _emb_size(n) == int(np.sqrt(n)) + 1


# ---------------------------------------------------------------------------
# Both models consume the seam
# ---------------------------------------------------------------------------

# A model must work with either strategy without knowing which it was given. These
# construct and forward once on random tensors — no training, no optimiser.

PROJECTED = ("projected", SEQ_COLS, EMBEDDED_COLS)
VALENDIN = ("valendin", ["Transactions", "week_idx"], EMBEDDED_COLS)


def build_embedder(kind, seq_cols, embedded_cols, width):
    if kind == "projected":
        return ProjectedEmbedder(seq_cols, embedded_cols, TARGET, embedding_dim=width)
    return ValendinEmbedder(seq_cols, embedded_cols, TARGET)


@pytest.mark.parametrize("kind,seq_cols,embedded_cols", [PROJECTED, VALENDIN])
def test_lstm_accepts_either_embedder(kind, seq_cols, embedded_cols):
    from panelclv.models import MultinomialLSTMModel

    embedder = build_embedder(kind, seq_cols, embedded_cols, width=16)
    model = MultinomialLSTMModel(embedder=embedder, lstm_hidden_size=8,
                                 dense_units=4, dropout=0.0).eval()
    with torch.no_grad():
        logits = model(make_x(seq_cols, embedded_cols))

    assert logits.shape == (B, T, EMBEDDED_COLS[TARGET])
    # The LSTM wires its input to whatever width the embedder advertised.
    assert model.backbone.lstm.input_size == embedder.output_dim


@pytest.mark.parametrize("kind,seq_cols,embedded_cols", [PROJECTED, VALENDIN])
def test_transformer_accepts_either_embedder(kind, seq_cols, embedded_cols):
    from panelclv.models import MultinomialTransformerModel

    embedder = build_embedder(kind, seq_cols, embedded_cols, width=16)
    model = MultinomialTransformerModel(embedder=embedder, seq_len=T, d_model=16,
                                        nhead=2, num_encoder_layers=1, dropout=0.0).eval()
    with torch.no_grad():
        logits = model(make_x(seq_cols, embedded_cols))

    assert logits.shape == (B, T, EMBEDDED_COLS[TARGET])
    # Whatever the embedder's width, it is projected onto d_model for the encoder.
    assert model.backbone.input_projection.in_features == embedder.output_dim
    assert model.backbone.input_projection.out_features == 16


@pytest.mark.parametrize("kind,seq_cols,embedded_cols", [PROJECTED, VALENDIN])
def test_the_rollout_model_carries_the_embedder_it_was_trained_with(kind, seq_cols, embedded_cols):
    """Whichever strategy the seam supplies, the handover keeps it (ADR-0005/0007).

    The rollout model is the trained model's own (`to_rollout()`), so the embedder
    travels with the backbone rather than being reconstructed — and the sampling head
    reads the same width whichever strategy was chosen.
    """
    from panelclv.models import MultinomialLSTMModel

    embedder = build_embedder(kind, seq_cols, embedded_cols, width=16)
    trained = MultinomialLSTMModel(
        embedder=embedder, lstm_hidden_size=8, dense_units=4, dropout=0.0)

    rollout = trained.to_rollout()
    assert rollout.backbone.embedder is embedder

    rollout.eval()
    with torch.no_grad():
        sample, state = rollout(make_x(seq_cols, embedded_cols))
    assert sample.shape == (B, T, 1)
    assert state is not None


def test_swapping_the_embedder_changes_the_model_width_only():
    """The point of the seam: a narrower strategy gives a smaller model, same code."""
    from panelclv.models import MultinomialLSTMModel

    projected = MultinomialLSTMModel(
        embedder=ProjectedEmbedder(["Transactions", "week_idx"], EMBEDDED_COLS,
                                   TARGET, embedding_dim=128),
        lstm_hidden_size=8, dense_units=4)
    valendin = MultinomialLSTMModel(
        embedder=ValendinEmbedder(["Transactions", "week_idx"], EMBEDDED_COLS, TARGET),
        lstm_hidden_size=8, dense_units=4)

    # 2*128 vs 3+8: the published architecture's LSTM input is far narrower.
    assert projected.backbone.lstm.input_size == 256
    assert valendin.backbone.lstm.input_size == 11
    assert sum(p.numel() for p in valendin.parameters()) < sum(
        p.numel() for p in projected.parameters())
