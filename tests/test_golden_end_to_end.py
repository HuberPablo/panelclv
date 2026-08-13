"""One pinned run of the whole pipeline, for every model family the package ships.

Every other test in this suite checks a part in isolation. Nothing checks that the parts
still compose, which is the failure this file exists for: a change to the embedder seam,
the feature contract or the simulator does not raise — it quietly returns a slightly
different forecast, and no unit test notices.

**Four arms**, one per model family, because the families share a pipeline but not a
stepper, and a net that covers one covers the others only by assumption:

    lstm           the developed recurrent model, `forecast_recurrent`
    transformer    the developed attention model, `forecast_attention`
    valendin_lstm  the frozen published benchmark (ADR-0004), recurrent rollout
    pareto_nbd     the frozen Pareto/NBD benchmark — an MCMC fit, no training, no rollout

The first three are one shape: panel -> tensors -> two epochs -> rollout -> metrics, so
they are a fixture parameter rather than three pipelines. `pareto_nbd` is not, and is
asserted differently — see `test_pareto_fit_is_shaped_and_finite`.

Three distinct properties are asserted, and they fail for different reasons:

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
- **Shape and finiteness** — what the Pareto arm gets *instead* of pinned numbers. Its
  200-draw, single-chain fit records which code runs, not whether it has converged, so
  pinning its values would pin sampler noise and no later reader could tell a real
  regression from MCMC drift.

The models here are deliberately tiny and undertrained — two epochs on 23 customers. The
numbers are not good and are not meant to be. This test pins *what the pipeline
computes*, not how well it forecasts.

**When these numbers change**, that is the test doing its job. Do not re-baseline
without knowing which change moved them and why. To regenerate after a deliberate
change, run with ``PANELCLV_PRINT_GOLDEN=1`` (and ``-s``, so pytest does not swallow the
print) and paste the printed per-arm block into ``GOLDEN_METRICS`` below.
"""

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from panelclv.benchmarks import (  # noqa: E402
    ValendinLSTMModel,
    compute_pareto_predictions,
)
from panelclv.configs.panel_config import PanelConfig  # noqa: E402
from panelclv.data_preparation import panel_dataset  # noqa: E402
from panelclv.trials import split_calibration  # noqa: E402
from panelclv.models.embedders import ProjectedEmbedder  # noqa: E402
from panelclv.models.monte_carlo_forecasting import (  # noqa: E402
    compute_forecast_metrics,
    forecast_recurrent,
    forecast_attention,
)
from panelclv.models.multinomial_lstm import MultinomialLSTMModel  # noqa: E402
from panelclv.models.multinomial_transformer import (  # noqa: E402
    MultinomialTransformerModel,
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

# Pinned outcomes, one block per rollout arm. See the module docstring before touching
# these. The Pareto arm is absent on purpose — it pins no values.
GOLDEN_METRICS = {
    "lstm": {
        "rmse": 2.0019012702059444,
        "bias_percent": 247.03757225433526,
        "mape_aggregate": 247.03757225433526,
    },
    "transformer": {
        "rmse": 1.8498824874546234,
        "bias_percent": 211.56069364161849,
        "mape_aggregate": 211.56069364161849,
    },
    "valendin_lstm": {
        "rmse": 1.869680860932991,
        "bias_percent": 216.257225433526,
        "mape_aggregate": 216.257225433526,
    },
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


def _valendin_config() -> PanelConfig:
    """The golden config stripped to what the published architecture can read.

    `ValendinEmbedder` has no covariate path — the paper's model consumes embedded
    features only — so the derived time features and autoregressive columns the other
    arms carry have to go. Constructing this config is the ADR-0004 constraint made
    concrete.
    """
    return replace(_golden_config(), time_features=None, ar_features=())


# --------------------------------------------------------------------------------------
# The four scenarios. `scripts/trace_golden_reachability.py` imports `SCENARIOS` and runs
# every one of them under a tracer: the reachability evidence is only worth anything if
# it traces the same code paths this test pins, so the two cannot be allowed to drift.
# Every scenario therefore takes a temporary directory and returns a dict.
# --------------------------------------------------------------------------------------


def _projected_embedder(metadata) -> ProjectedEmbedder:
    """The default embedding strategy (ADR-0005), as both developed arms build it."""
    return ProjectedEmbedder(
        seq_cols=metadata["seq_cols"],
        embedded_cols=metadata["embedded_cols"],
        target_col=metadata["target_col"],
        embedding_dim=8,
    )


def _fit_and_roll(data, train_cls, build, forecaster, tmp_path) -> dict:
    """The tail every rollout arm shares: fit two epochs, hand over, roll out, score.

    `build(cls, metadata)` constructs one model of class `cls` from the split's recipe.
    The rollout model is not built here at all: the trained model hands over its own
    (ADR-0007), sharing the backbone `fit_model` just left the selected weights in — so
    the arm exercises the production handover rather than a file round-trip.
    """
    split = split_calibration(data, batch_size=8)
    n_classes = int(data["embedded_cols"]["Transactions"])

    torch.manual_seed(TORCH_SEED)
    trained = build(train_cls, split.recipe)

    fit = fit_model(
        trained,
        split.train_loader,
        split.val_loader,
        num_target_classes=n_classes,  # class COUNT, not the maximum class index
        n_epochs=2,
        patience=2,
        device="cpu",                 # CPU keeps the run reproducible and GPU-free
        checkpoint_dir=str(tmp_path),
        model_name="golden",
        verbose=False,
        val_score_start=split.recipe["val_score_start"],
    )

    rollout_model = trained.to_rollout()

    forecast = forecaster(
        rollout_model,
        data,
        n_simulations=N_SIMULATIONS,
        seed=FORECAST_SEED,
        device="cpu",
        return_simulations=False,
    )
    metrics = compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])
    return {"data": data, "fit": fit, "forecast": forecast, "metrics": metrics}


def run_lstm_pipeline(tmp_path) -> dict:
    """The recurrent arm: panel -> tensors -> train -> rollout -> metrics, seeded end to end."""
    data = panel_dataset.prepare_dataset(_golden_panel(), _golden_config(), verbose=False)

    def build(cls, metadata):
        return cls(
            embedder=_projected_embedder(metadata),
            lstm_hidden_size=8,
            dense_units=8,
            dropout=0.0,
        )

    return _fit_and_roll(
        data,
        MultinomialLSTMModel,
        build,
        forecast_recurrent,
        tmp_path,
    )


def run_transformer_pipeline(tmp_path) -> dict:
    """The attention arm: same panel and same contract, growing-window stepper.

    `forecast_attention` is what every Transformer study runs through in production,
    and the recurrent/attention crossing fails silently rather than raising — which
    is why this arm exists.
    """
    data = panel_dataset.prepare_dataset(_golden_panel(), _golden_config(), verbose=False)

    def build(cls, metadata):
        return cls(
            embedder=_projected_embedder(metadata),
            seq_len=metadata["seq_len"],   # caches the causal mask for the training length
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            dropout=0.0,
        )

    return _fit_and_roll(
        data,
        MultinomialTransformerModel,
        build,
        forecast_attention,
        tmp_path,
    )


def run_valendin_pipeline(tmp_path) -> dict:
    """The published benchmark, end to end on the stripped config.

    `scripts/validate_valendin_lstm.py` needs the gitignored `Datasets/`, so this arm is
    the only coverage of a full Valendin rollout that runs on a fresh clone.
    """
    data = panel_dataset.prepare_dataset(
        _golden_panel(), _valendin_config(), verbose=False
    )

    def build(cls, metadata):
        # No embedder argument: the benchmark owns its own published embedding strategy.
        return cls(
            seq_cols=metadata["seq_cols"],
            embedded_cols=metadata["embedded_cols"],
            target_col=metadata["target_col"],
        )

    return _fit_and_roll(
        data,
        ValendinLSTMModel,
        build,
        forecast_recurrent,
        tmp_path,
    )


def run_pareto_fit(tmp_path) -> dict:
    """The Pareto/NBD benchmark: an MCMC fit on the calibration panel, no model to train.

    Fed exactly as `studies/runner.py` feeds it in production — `prepare_dataset`'s own
    `train_panel` and cohort, rather than a panel this test slices itself, so the
    benchmark sees the same calibration window and the same customers as the three
    neural arms.

    Short chains on purpose (200 draws, 50 burn-in, one chain): this records which code
    runs and that it runs reproducibly, not that the sampler has converged. `tmp_path` is
    accepted and ignored, so every scenario has the same signature.
    """
    data = panel_dataset.prepare_dataset(_golden_panel(), _golden_config(), verbose=False)
    predictions, ids = compute_pareto_predictions(
        data["train_panel"],
        holdout_length=int(data["T_HOLD"]),
        id_col=data["id_col"],
        target_col=data["target_col"],
        customer_ids=data["ids"],   # same row order as the neural arms' forecasts
        period_in_days=7.0,
        mcmc=200,
        burnin=50,
        thin=10,
        chains=1,
        seed=42,
    )
    return {"predictions": predictions, "ids": ids}


SCENARIOS = {
    "lstm": run_lstm_pipeline,
    "transformer": run_transformer_pipeline,
    "valendin_lstm": run_valendin_pipeline,
    "pareto_nbd": run_pareto_fit,
}

# The three arms that train a model and roll it forward. `pareto_nbd` does neither, so it
# shares no assertion with them.
ROLLOUT_ARMS = ("lstm", "transformer", "valendin_lstm")


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """`scenario(name)` -> that arm's result, run at most once per module.

    Memoised rather than eagerly built so that selecting a subset with `-k` pays only for
    the arms that subset actually asks for.
    """
    results: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in results:
            results[name] = SCENARIOS[name](tmp_path_factory.mktemp(name))
        return results[name]

    return get


@pytest.fixture(scope="module", params=ROLLOUT_ARMS)
def rollout(request, scenario):
    """One trained-and-rolled-out arm, as `(arm name, result)`."""
    return request.param, scenario(request.param)


@pytest.fixture(scope="module")
def golden(scenario):
    """The recurrent arm, for the assertions that are about data preparation only."""
    return scenario("lstm")


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


def test_rollout_metrics_are_pinned(rollout):
    """The three scores `compute_forecast_metrics` is the single authority for."""
    arm, result = rollout
    if os.environ.get("PANELCLV_PRINT_GOLDEN"):
        print(f'\n    "{arm}": {{')
        for key, value in result["metrics"].items():
            print(f'        "{key}": {value!r},')
        print("    },")
    assert result["metrics"] == pytest.approx(GOLDEN_METRICS[arm], rel=1e-6)


def test_rollout_is_deterministic(rollout, tmp_path):
    """Same config, same seed, bit-identical forecast — asserted, not assumed.

    Run a second time in this process and compared against the fixture's run, rather than
    against a stored array: this isolates *reproducibility* from *regression*, so a
    failure here means an unseeded RNG or an order-dependent step, never a deliberate
    behaviour change.
    """
    arm, first = rollout
    second = SCENARIOS[arm](tmp_path)

    np.testing.assert_array_equal(
        first["forecast"]["prediction_mean"], second["forecast"]["prediction_mean"]
    )
    assert first["metrics"] == second["metrics"]
    assert first["fit"].best_val_loss == second["fit"].best_val_loss


def test_rollout_never_reads_the_holdout(rollout):
    """The rollout's own output, not the truth, is what it feeds back.

    `actual` is returned for scoring only. If the simulator ever fed it in, a 2-epoch
    model would score implausibly well — so a forecast that matches the truth exactly is
    evidence of leakage, not of skill.
    """
    _, result = rollout
    forecast = result["forecast"]
    assert forecast["prediction_mean"].shape == forecast["actual"].shape
    assert not np.array_equal(forecast["prediction_mean"], forecast["actual"])


def test_pareto_fit_is_shaped_and_finite(scenario):
    """The Pareto arm's whole assertion set, and deliberately weaker than the others.

    One expected count per customer per holdout period, all finite and non-negative. No
    value is pinned: the fit is one short chain, so its numbers are sampler noise at this
    length and pinning them would leave the next reader unable to tell a real regression
    from MCMC drift.
    """
    result = scenario("pareto_nbd")
    predictions = result["predictions"]

    # Same cohort as the neural arms, because the benchmark is handed prepare_dataset's
    # own train_panel and customer order rather than a panel this test slices itself.
    assert len(result["ids"]) == GOLDEN_SHAPES["n_customers"]
    assert predictions.shape == (GOLDEN_SHAPES["n_customers"], GOLDEN_SHAPES["t_holdout"])
    assert np.all(np.isfinite(predictions))


def test_pareto_fit_is_deterministic(scenario, tmp_path):
    """One seed, one sampler path: the MCMC fit is reproducible even though it is random.

    This is the property worth asserting on an unconverged chain — that the chain is the
    *same* chain — and it is why the arm can be useful without pinning any value.
    """
    first = scenario("pareto_nbd")
    second = run_pareto_fit(tmp_path)

    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    assert first["ids"] == second["ids"]
