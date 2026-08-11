"""Faithfulness tests for the Valendin et al. LSTM benchmark.

The reference notebook (`Original_paper_model/banking_transactions_demo.ipynb`)
prints a `model.summary()` with an exact parameter count per layer. Those counts are
the specification this module reproduces, so they are pinned here — a benchmark that
drifts from the published architecture is worse than no benchmark (ADR-0004).

Nothing is trained here; parameter shapes are enough to catch architectural drift.
"""

import pytest

torch = pytest.importorskip("torch")

from panelclv.benchmarks.valendin_lstm import (  # noqa: E402
    InferenceValendinLSTMModel,
    ValendinLSTMModel,
)

# The reference notebook's banking demo: 52 week-of-year classes and 12 transaction
# count classes (0..11), sequence length 155. Column order matters — the notebook
# concatenates [emb_week, emb_trans], so week comes first in seq_cols.
SEQ_COLS = ["week", "transaction"]
EMBEDDED_COLS = {"week": 52, "transaction": 12}
TARGET = "transaction"
SEQ_LEN = 155

# Straight from the notebook's model.summary() output.
PUBLISHED_PARAMS = {
    "embed_week": 416,      # Embedding(52, 8)
    "embed_trans": 48,      # Embedding(12, 4)
    "dense": 16512,         # Dense(128) from a 128-wide LSTM output
    "softmax": 1548,        # Dense(12) from a 128-wide dense output
}
PUBLISHED_LSTM_PARAMS = 72192   # Keras LSTM(128) over a 12-wide input
# Keras keeps one bias vector per gate, PyTorch keeps two (b_ih and b_hh), so ours
# carries 4 * 128 more. A framework convention, not an architectural difference.
TORCH_LSTM_BIAS_SURPLUS = 4 * 128


@pytest.fixture
def model():
    return ValendinLSTMModel(SEQ_COLS, EMBEDDED_COLS, TARGET)


# ---------------------------------------------------------------------------
# Layer for layer against the published summary
# ---------------------------------------------------------------------------


def test_embedding_widths_follow_the_papers_heuristic(model):
    """Embedding(52, 8) and Embedding(12, 4) — `int(sqrt(n)) + 1`."""
    week, trans = model.backbone.embedder._emb_modules
    assert (week.num_embeddings, week.embedding_dim) == (52, 8)
    assert (trans.num_embeddings, trans.embedding_dim) == (12, 4)
    assert week.weight.numel() == PUBLISHED_PARAMS["embed_week"]
    assert trans.weight.numel() == PUBLISHED_PARAMS["embed_trans"]


def test_concatenated_input_is_twelve_wide(model):
    """`concat (Concatenate) (None, 155, 12)` — 8 + 4, nothing projected."""
    assert model.backbone.embedder.output_dim == 12
    assert model.backbone.lstm.input_size == 12


def test_lstm_matches_the_published_size_up_to_the_bias_convention(model):
    """`lstm (LSTM) (None, 155, 128) 72192`."""
    lstm = model.backbone.lstm
    assert lstm.hidden_size == 128
    assert lstm.num_layers == 1

    n = sum(p.numel() for p in lstm.parameters())
    assert n == PUBLISHED_LSTM_PARAMS + TORCH_LSTM_BIAS_SURPLUS


def test_dense_and_head_match_the_published_sizes(model):
    """`dense (Dense) 16512` and `softmax (Dense) 1548`."""
    assert sum(p.numel() for p in model.backbone.dense.parameters()) == \
        PUBLISHED_PARAMS["dense"]
    assert sum(p.numel() for p in model.backbone.output_layer.parameters()) == \
        PUBLISHED_PARAMS["softmax"]


def test_there_is_nothing_between_the_embeddings_and_the_lstm(model):
    """No LayerNorm, no projection, no dropout — the departures ADR-0004 records.

    This is the whole reason the benchmark is a separate module from
    `models.MultinomialLSTMModel`, so it is pinned structurally rather than trusted.
    """
    for block in model.backbone.embedder._emb_modules:
        assert isinstance(block, torch.nn.Embedding)

    kinds = {type(m) for m in model.backbone.modules()}
    assert torch.nn.LayerNorm not in kinds
    assert torch.nn.Dropout not in kinds


def test_total_parameter_count_matches_the_published_total(model):
    published_total = sum(PUBLISHED_PARAMS.values()) + PUBLISHED_LSTM_PARAMS
    assert sum(p.numel() for p in model.parameters()) == \
        published_total + TORCH_LSTM_BIAS_SURPLUS


# ---------------------------------------------------------------------------
# It behaves like every other model in the package
# ---------------------------------------------------------------------------


def make_x(batch=2, seq_len=SEQ_LEN):
    g = torch.Generator().manual_seed(0)
    return torch.stack([
        torch.randint(0, 52, (batch, seq_len), generator=g).float(),
        torch.randint(0, 12, (batch, seq_len), generator=g).float(),
    ], dim=-1)


def test_forward_returns_logits_over_the_count_classes(model):
    """A categorical head, not a point regressor — one logit per count class."""
    model.eval()
    with torch.no_grad():
        logits = model(make_x())
    assert logits.shape == (2, SEQ_LEN, 12)
    assert model.num_target_classes == 12


def test_inference_model_samples_and_threads_state(model):
    """The rollout contract: (sample, state), state chainable across steps."""
    inference = InferenceValendinLSTMModel(SEQ_COLS, EMBEDDED_COLS, TARGET)
    inference.load_state_dict(model.state_dict(), strict=True)
    inference.eval()

    x = make_x(seq_len=4)
    with torch.no_grad():
        sample, state = inference(x)
        # Feeding the state back is how the simulator steps through the holdout.
        next_sample, _ = inference(x[:, -1:, :], state)

    assert sample.shape == (2, 4, 1)
    assert next_sample.shape == (2, 1, 1)
    # Samples are count CLASS indices, never fractional quantities.
    assert torch.equal(sample, sample.round())
    assert (sample >= 0).all() and (sample < 12).all()


def test_rejects_a_numerical_covariate():
    """The published model has no covariate path; asking for one is an error."""
    with pytest.raises(ValueError, match="covariate"):
        ValendinLSTMModel(
            ["week", "transaction", "balance_norm"],
            EMBEDDED_COLS,
            TARGET,
        )


# ---------------------------------------------------------------------------
# ADR-0004: importing the subpackage must not drag in torch
# ---------------------------------------------------------------------------


def test_benchmark_names_resolve_from_the_subpackage():
    import panelclv.benchmarks as bench

    assert bench.ValendinLSTMModel is ValendinLSTMModel
    assert bench.InferenceValendinLSTMModel is InferenceValendinLSTMModel
    assert "ValendinLSTMModel" in dir(bench)


def test_unknown_attribute_still_raises_attribute_error():
    import panelclv.benchmarks as bench

    with pytest.raises(AttributeError):
        bench.NoSuchModel
