"""`MultinomialLSTMAttentionModel`: an LSTM whose head also reads attention over its past.

Two properties make it usable as a forecaster, and both are tested here:

- **Causal.** Changing the input at step t must not change the logits before t. The
  model trains on whole sequences with a class-index target at every step, so a
  future-looking attention would learn to copy the answer.
- **The stateful rollout is the same model.** `forecast_recurrent` warms the model up
  on the calibration window and then feeds one period at a time, threading `state`.
  Stepping through a sequence that way must reproduce the logits of one full forward
  pass, or the forecast would come from a different function than the one trained.
"""

import pytest

torch = pytest.importorskip("torch")

from panelclv.models.embedders import ValendinEmbedder  # noqa: E402
from panelclv.models.multinomial_lstm_attention import (  # noqa: E402
    MultinomialLSTMAttentionModel,
)
from panelclv.registry.model_registry import MODEL_REGISTRY  # noqa: E402

N_CLASSES = 4
B, T = 3, 9


def _model() -> MultinomialLSTMAttentionModel:
    torch.manual_seed(0)
    embedder = ValendinEmbedder(
        seq_cols=["Transactions", "week"],
        embedded_cols={"Transactions": N_CLASSES, "week": 52},
        target_col="Transactions",
    )
    return MultinomialLSTMAttentionModel(
        embedder=embedder, lstm_hidden_size=8, dense_units=8, dropout=0.0
    ).eval()


def _inputs() -> torch.Tensor:
    """(B, T, 2): a count class and a week index per step, as floats."""
    g = torch.Generator().manual_seed(1)
    counts = torch.randint(0, N_CLASSES, (B, T, 1), generator=g)
    weeks = torch.randint(0, 52, (B, T, 1), generator=g)
    return torch.cat([counts, weeks], dim=-1).float()


def test_logits_have_one_softmax_row_per_step():
    logits = _model()(_inputs())
    assert logits.shape == (B, T, N_CLASSES)


def test_attention_is_causal():
    model, x = _model(), _inputs()
    changed = x.clone()
    changed[:, 5, 0] = (changed[:, 5, 0] + 1) % N_CLASSES     # alter step 5 only
    with torch.no_grad():
        before, after = model(x), model(changed)
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-6)
    assert not torch.allclose(before[:, 5:], after[:, 5:])


def test_stepping_with_state_reproduces_the_full_forward_pass():
    """Warm up on the first 6 steps, then one step at a time — as the rollout does."""
    model, x = _model(), _inputs()
    with torch.no_grad():
        full = model(x)
        logits, state = model.backbone(x[:, :6], None)
        stepped = [logits]
        for t in range(6, T):
            logits, state = model.backbone(x[:, t:t + 1], state)
            stepped.append(logits)
    assert torch.allclose(torch.cat(stepped, dim=1), full, atol=1e-5)


def test_rollout_samples_a_class_and_threads_state():
    rollout = _model().to_rollout()
    sample, state = rollout(_inputs())
    assert sample.shape == (B, T, 1)
    assert ((sample >= 0) & (sample < N_CLASSES)).all()
    sample, state = rollout(_inputs()[:, :1], state)
    assert sample.shape == (B, 1, 1)


def test_the_registry_builds_it_and_forecasts_it_recurrently():
    entry = MODEL_REGISTRY["lstm_attention"]
    params = {"embedder": "valendin", "lstm_hidden_size": 8, "dense_units": 8,
              "dropout": 0.0}
    recipe = {"seq_cols": ["Transactions", "week"],
              "embedded_cols": {"Transactions": N_CLASSES, "week": 52},
              "target_col": "Transactions"}
    assert isinstance(entry.build(params, recipe), MultinomialLSTMAttentionModel)
    assert entry.rollout is MODEL_REGISTRY["lstm"].rollout
