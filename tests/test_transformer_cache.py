"""The Transformer's cached rollout computes what the full forward pass computes.

`forward_cached` re-states the pre-LN encoder layer from its modules so the rollout
can process one new period against cached keys and values instead of re-reading the
whole growing context. That is only legitimate if it is the same function, so both
the logits and the sampled paths are checked against the uncached computation.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from panelclv.models.embedders import ValendinEmbedder  # noqa: E402
from panelclv.models.monte_carlo_forecasting import simulate_attention_path  # noqa: E402
from panelclv.models.multinomial_transformer import MultinomialTransformerModel  # noqa: E402

N_CLASSES = 4
B, T = 5, 12


def _model(layers: int = 2) -> MultinomialTransformerModel:
    torch.manual_seed(0)
    embedder = ValendinEmbedder(
        seq_cols=["Transactions", "week"],
        embedded_cols={"Transactions": N_CLASSES, "week": 52},
        target_col="Transactions",
    )
    return MultinomialTransformerModel(
        embedder=embedder, d_model=16, nhead=4, num_encoder_layers=layers, dropout=0.1,
    ).eval()


def _inputs(t: int = T) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    counts = torch.randint(0, N_CLASSES, (B, t, 1), generator=g)
    weeks = torch.randint(0, 52, (B, t, 1), generator=g)
    return torch.cat([counts, weeks], dim=-1).float()


@pytest.mark.parametrize("layers", [1, 3])
def test_cached_steps_reproduce_the_full_forward_logits(layers):
    """Warm up on 7 periods, then one at a time; every step matches the full pass."""
    model, x = _model(layers), _inputs()
    backbone = model.backbone
    with torch.no_grad():
        full = backbone(x)                                    # (B, T, K), causal
        logits, cache = backbone.forward_cached(x[:, :7])
        assert torch.allclose(logits[:, 0], full[:, 6], atol=1e-5)
        for t in range(7, T):
            logits, cache = backbone.forward_cached(x[:, t:t + 1], cache)
            assert torch.allclose(logits[:, 0], full[:, t], atol=1e-5), t
    assert cache["pos"] == T


def test_the_first_call_is_the_same_when_read_in_customer_chunks(monkeypatch):
    import panelclv.models.multinomial_transformer as mt

    model, x = _model(), _inputs()
    with torch.no_grad():
        whole_logits, whole = model.backbone.forward_cached(x)
        monkeypatch.setattr(mt, "_WARMUP_CHUNK", 2)            # B = 5 -> 2 + 2 + 1
        chunk_logits, chunked = model.backbone.forward_cached(x)
    assert torch.allclose(whole_logits, chunk_logits, atol=1e-6)
    for a, b in zip(whole["k"] + whole["v"], chunked["k"] + chunked["v"]):
        assert torch.allclose(a, b, atol=1e-6)


def _uncached_path(model, calibration, holdout, target_idx):
    """The rollout as it was before the cache: re-read the growing context each step."""
    backbone = model.backbone
    context, path = calibration, []
    with torch.no_grad():
        for t in range(holdout.shape[1]):
            probs = torch.softmax(backbone(context, only_last=True), dim=-1)
            sample = torch.distributions.Categorical(probs=probs).sample()[:, -1].float()
            path.append(sample)
            nxt = holdout[:, t:t + 1].clone()
            nxt[:, 0, target_idx] = sample
            context = torch.cat([context, nxt], dim=1)
    return torch.stack(path, dim=1).numpy()


def test_the_rollout_samples_the_same_path_as_rereading_the_context():
    model, x = _model(), _inputs()
    calibration, holdout = x[:, :8], x[:, 8:]
    torch.manual_seed(7)
    cached = simulate_attention_path(model.to_rollout(), calibration, holdout,
                                     seq_cols=["Transactions", "week"], target_idx=0,
                                     device="cpu")
    torch.manual_seed(7)
    uncached = _uncached_path(model, calibration, holdout, target_idx=0)
    np.testing.assert_array_equal(cached, uncached)
