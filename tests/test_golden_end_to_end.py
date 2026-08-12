"""One pinned run of the whole pipeline: panel -> tensors -> training -> rollout -> metrics.

Every other test in this suite checks a part in isolation. Nothing checks that the parts
still compose, which is the failure this file exists for: a change to the embedder seam,
the feature contract or the simulator does not raise — it quietly returns a slightly
different forecast, and no unit test notices.

Two distinct properties are asserted, and they fail for different reasons:

- **Determinism** — the same config and seed produce bit-identical predictions. This is
  priority #2 in ``CLAUDE.md`` ("same config and seed gives the same result"), and
  nothing else in the suite tests it. Asserted as exact equality, because an unseeded
  RNG or an order-dependent step breaks it on any machine.
- **Regression** — the pinned numbers below still come out. This is asserted with a
  relative tolerance rather than exact equality: the run is bit-reproducible within one
  environment (verified across processes), but CPU float reduction order is not
  guaranteed identical across BLAS builds, and this repo runs on ROCm locally, on Colab
  and on VastAI. A tolerance of 1e-6 is far tighter than any real behaviour change and
  loose enough not to cry wolf on a different host.

The model here is deliberately tiny and undertrained — two epochs on 23 customers. The
numbers are not good and are not meant to be. This test pins *what the pipeline
computes*, not how well it forecasts.

**When these numbers change**, that is the test doing its job. Do not re-baseline
without knowing which change moved them and why. To regenerate after a deliberate
change, run with ``PANELCLV_PRINT_GOLDEN=1`` and paste the printed block below.
"""

import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from panelclv.configs.panel_config import PanelConfig  # noqa: E402
from panelclv.data_preparation import dynamic_panel_dataset  # noqa: E402
from panelclv.experiments import make_loaders  # noqa: E402
from panelclv.models.embedders import ProjectedEmbedder  # noqa: E402
from panelclv.models.monte_carlo_forecasting import (  # noqa: E402
    compute_forecast_metrics,
    run_monte_carlo_forecast,
)
from panelclv.models.multinomial_lstm import (  # noqa: E402
    InferenceMultinomialLSTMModel,
    MultinomialLSTMModel,
)
from panelclv.training import fit_model  # noqa: E402

# --------------------------------------------------------------------------------------
# The golden fixture: a synthetic panel, small enough for CI, real enough to exercise
# every stage. Poisson counts at a per-customer gamma rate — the Pareto/NBD generative
# story minus the death process, which keeps the panel dense enough that a 2-epoch model
# has something to learn.
# --------------------------------------------------------------------------------------

N_CUSTOMERS = 24
N_PERIODS = 78                 # 52 calibration + 26 holdout weeks
PANEL_SEED = 20260811          # fixed: `default_rng` streams are stable across numpy versions
TORCH_SEED = 1234
FORECAST_SEED = 7
N_SIMULATIONS = 8

# Pinned outcomes. See the module docstring before touching these.
GOLDEN_METRICS = {
    "rmse": 2.0019012702059444,
    "bias_percent": 247.03757225433526,
    "mape_aggregate_style": 247.03757225433526,
}
# Cohort and window shapes. One synthetic customer never transacts in the calibration
# window and is dropped by `require_calibration_activity`, so N is 23 and not 24 — that
# drop is part of what this test pins.
GOLDEN_SHAPES = {
    "n_customers": 23,
    "t_calibration": 52,
    "t_holdout": 25,
    "n_features": 5,
    "val_start_idx": 39,
    "n_target_classes": 5,
}


def _golden_panel() -> pd.DataFrame:
    """A customer-period panel: one row per customer per week, counts and calendar."""
    rng = np.random.default_rng(PANEL_SEED)
    rates = rng.gamma(shape=1.2, scale=0.5, size=N_CUSTOMERS)
    weeks = pd.date_range("2000-01-03", periods=N_PERIODS, freq="7D")
    iso = [(d.isocalendar()[0], d.isocalendar()[1]) for d in weeks]

    frames = [
        pd.DataFrame(
            {
                "Id": f"C{i:03d}",
                "year": [y for y, _ in iso],
                "week": [w for _, w in iso],
                "Transactions": rng.poisson(rate, size=N_PERIODS),
            }
        )
        for i, rate in enumerate(rates)
    ]
    return pd.concat(frames, ignore_index=True)


def _golden_config() -> PanelConfig:
    """Every column role, window and declaration the pipeline reads, in one object.

    Deliberately exercises the parts most likely to break silently: two autoregressive
    features (the leakage-prone path), derived time features, and an embedded target
    whose cardinality sets the softmax head size.
    """
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        training_start="2000-01-03",
        training_end="2001-01-01",
        validation_start="2000-10-02",
        holdout_start="2001-01-02",
        holdout_end="2001-06-30",
        time_cols=("year", "week"),
        clip_target_upper=4,
        time_features={"add_year_idx": True, "add_week_sin_cos": True},
        ar_features=("period_since_last_transaction", "cumulative_transactions"),
        embedded_cols={"Transactions": "auto"},
    )


def run_golden_pipeline(tmp_path) -> dict:
    """Panel -> tensors -> train -> rollout -> metrics, seeded end to end.

    Exported (not underscore-private) because `scripts/trace_golden_reachability.py`
    runs this exact function under a tracer: the reachability evidence is only worth
    anything if it traces the same code path this test pins.
    """
    data = dynamic_panel_dataset.prepare_dataset(_golden_panel(), _golden_config(), verbose=False)
    train_loader, val_loader, metadata = make_loaders(data, batch_size=8)
    n_classes = int(data["embedded_cols"]["Transactions"])

    def _build(cls):
        # Seeded immediately before construction so the training model and the inference
        # model start from identical weights — the constructor-arguments-must-match
        # invariant, made concrete.
        torch.manual_seed(TORCH_SEED)
        embedder = ProjectedEmbedder(
            seq_cols=metadata["seq_cols"],
            embedded_cols=metadata["embedded_cols"],
            target_col=metadata["target_col"],
            embedding_dim=8,
        )
        return cls(embedder=embedder, lstm_hidden_size=8, dense_units=8, dropout=0.0)

    fit = fit_model(
        _build(MultinomialLSTMModel),
        train_loader,
        val_loader,
        max_trans=n_classes,          # class COUNT, not the maximum class index
        n_epochs=2,
        patience=2,
        device="cpu",                 # CPU keeps the run reproducible and GPU-free
        checkpoint_dir=str(tmp_path),
        model_name="golden",
        verbose=False,
        val_score_start=metadata["val_score_start"],
    )

    inference_model = _build(InferenceMultinomialLSTMModel)
    inference_model.load_state_dict(torch.load(fit.checkpoint_path, map_location="cpu"))

    forecast = run_monte_carlo_forecast(
        inference_model,
        data,
        n_simulations=N_SIMULATIONS,
        seed=FORECAST_SEED,
        device="cpu",
        return_simulations=False,
    )
    metrics = compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])
    return {"data": data, "fit": fit, "forecast": forecast, "metrics": metrics}


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    return run_golden_pipeline(tmp_path_factory.mktemp("golden"))


def test_golden_shapes_are_pinned(golden):
    """The cohort and windows the pipeline builds, before any model sees them.

    A change here means data preparation changed what it feeds the model — which shifts
    every downstream number, so it is worth failing on its own rather than as a confusing
    metrics mismatch.
    """
    data = golden["data"]
    assert {
        "n_customers": data["samples"].shape[0],
        "t_calibration": data["T_CAL"],
        "t_holdout": data["T_HOLD"],
        "n_features": len(data["seq_cols"]),
        "val_start_idx": data["val_start_idx"],
        "n_target_classes": int(data["embedded_cols"]["Transactions"]),
    } == GOLDEN_SHAPES


def test_golden_feature_axis_is_pinned(golden):
    """Feature order is part of the contract: the tensor's last axis is positional.

    A reordered `seq_cols` loads a checkpoint that silently reads the wrong channel.
    """
    assert golden["data"]["seq_cols"] == [
        "Transactions",
        "week_sin",
        "week_cos",
        "period_since_last_transaction",
        "cumulative_transactions",
    ]


def test_golden_metrics_are_pinned(golden):
    """The three scores `compute_forecast_metrics` is the single authority for."""
    if os.environ.get("PANELCLV_PRINT_GOLDEN"):
        print("\nGOLDEN_METRICS = {")
        for key, value in golden["metrics"].items():
            print(f'    "{key}": {value!r},')
        print("}")
    assert golden["metrics"] == pytest.approx(GOLDEN_METRICS, rel=1e-6)


def test_pipeline_is_deterministic(tmp_path):
    """Same config, same seed, bit-identical forecast — asserted, not assumed.

    Run twice in one process rather than compared against a stored array: this isolates
    *reproducibility* from *regression*, so a failure here means an unseeded RNG or an
    order-dependent step, never a deliberate behaviour change.
    """
    first = run_golden_pipeline(tmp_path / "a")
    second = run_golden_pipeline(tmp_path / "b")

    np.testing.assert_array_equal(
        first["forecast"]["prediction_mean"], second["forecast"]["prediction_mean"]
    )
    assert first["metrics"] == second["metrics"]
    assert first["fit"].best_val_loss == second["fit"].best_val_loss


def test_forecast_never_reads_the_holdout(golden):
    """The rollout's own output, not the truth, is what it feeds back.

    `actual` is returned for scoring only. If the simulator ever fed it in, a 2-epoch
    model would score implausibly well — so a forecast that matches the truth exactly is
    evidence of leakage, not of skill.
    """
    forecast = golden["forecast"]
    assert forecast["prediction_mean"].shape == forecast["actual"].shape
    assert not np.array_equal(forecast["prediction_mean"], forecast["actual"])
