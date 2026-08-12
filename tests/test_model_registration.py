"""Every registered neural model type is wired into all of its dispatch sites.

CLAUDE.md: "Adding a model touches three places, and missing the second fails only
after training completes." These tests make that failure immediate and cheap.

They assert more than list membership. `tuning/optuna_tuning.py` dispatches on
`model_type` at four sites, and two of them historically fell through to the
Transformer on an unrecognised type — so a model registered in some lists but not all
would train the wrong architecture under the right name. Membership alone would not
catch that, so the built model's *class* is checked per type.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from panelclv.studies.config import NEURAL_MODEL_TYPES, VALID_MODEL_TYPES  # noqa: E402
from panelclv.studies.runner import _FORECASTERS  # noqa: E402
from panelclv.tuning import optuna_tuning as tuning  # noqa: E402

# The architecture each model type must build. Getting a different class here means a
# dispatch site fell through — the silent failure these tests exist for.
EXPECTED_CLASS = {
    "lstm": "MultinomialLSTMModel",
    "transformer": "MultinomialTransformerModel",
    "valendin_lstm": "ValendinLSTMModel",
}

# A small panel every model type can be built against.
METADATA = {
    "seq_cols": ["Transactions", "week_idx"],
    "embedded_cols": {"Transactions": 6, "week_idx": 52},
    "target_col": "Transactions",
    "seq_len": 8,
}


class _FixedTrial:
    """Stands in for an optuna.Trial, returning the low end of every search range.

    Enough to drive `suggest_*_params` without an Optuna study; the values do not
    matter, only that every declared parameter can be sampled.
    """

    def suggest_categorical(self, name, choices):
        return sorted(choices)[0]

    def suggest_float(self, name, low, high, **kwargs):
        return low

    def suggest_int(self, name, low, high, **kwargs):
        return low


def test_every_neural_type_is_a_valid_model_type():
    for model_type in NEURAL_MODEL_TYPES:
        assert model_type in VALID_MODEL_TYPES


@pytest.mark.parametrize("model_type", NEURAL_MODEL_TYPES)
def test_neural_type_has_a_forecaster(model_type):
    """studies/runner.py — the entry missing this fails only after training."""
    assert model_type in _FORECASTERS
    assert callable(_FORECASTERS[model_type])


@pytest.mark.parametrize("model_type", NEURAL_MODEL_TYPES)
def test_neural_type_has_a_search_space_and_suggester(model_type):
    """tuning/optuna_tuning.py — data_info validation and parameter sampling."""
    # Validation must recognise the type rather than raising "Unknown model_type".
    tuning.validate_data_info(model_type, {})

    params = tuning._suggest_params_for(model_type, _FixedTrial(), {})
    assert isinstance(params, dict) and params, f"{model_type} sampled no parameters"


@pytest.mark.parametrize("model_type", NEURAL_MODEL_TYPES)
def test_neural_type_builds_its_own_architecture(model_type):
    """The check membership alone cannot make: the right class comes out.

    A dispatch site that falls through to another family would pass every list-based
    assertion above and still train the wrong model.
    """
    params = tuning._suggest_params_for(model_type, _FixedTrial(), {})
    model = tuning._build_model_for(model_type, params, METADATA)
    assert type(model).__name__ == EXPECTED_CLASS[model_type]


@pytest.mark.parametrize("model_type", NEURAL_MODEL_TYPES)
def test_neural_type_builds_a_matching_inference_model(model_type):
    """The rollout loads the trained model's state_dict into the inference model.

    CLAUDE.md's invariant: their constructor arguments must match. A mismatch here
    surfaces only after a full training run, so it is pinned.
    """
    params = tuning._suggest_params_for(model_type, _FixedTrial(), {})
    trained = tuning._build_model_for(model_type, params, METADATA)
    inference, forecaster = tuning._build_inference_model_for(
        model_type, params, METADATA
    )
    inference.load_state_dict(trained.state_dict(), strict=True)
    assert callable(forecaster)


def test_unknown_model_type_is_rejected_everywhere():
    """No dispatch site may quietly fall through to another architecture."""
    for call in (
        lambda: tuning.validate_data_info("nonesuch", {}),
        lambda: tuning._suggest_params_for("nonesuch", _FixedTrial(), {}),
        lambda: tuning._build_model_for("nonesuch", {}, METADATA),
        lambda: tuning._build_inference_model_for("nonesuch", {}, METADATA),
    ):
        with pytest.raises(ValueError, match="model_type"):
            call()


@pytest.mark.parametrize("model_type", NEURAL_MODEL_TYPES)
def test_study_entry_point_accepts_every_registered_type(model_type):
    """`run_optuna_study` carried its own hardcoded {"lstm", "transformer"} guard.

    Every other site agreed on the registered set while this one did not, so a newly
    registered model passed all of them and was rejected at the entry point instead —
    caught only by running a real suite. Pinned so the guard cannot drift again.
    """
    import inspect

    source = inspect.getsource(tuning.run_optuna_study)
    assert "must be 'lstm' or 'transformer'" not in source, (
        "run_optuna_study hardcodes a model-type list instead of using the registry"
    )


def test_valendin_architecture_is_not_searched():
    """ADR-0004 freezes the benchmark's architecture, so only training knobs are tuned.

    Optuna searching `memory_units` or `dense_units` would silently unfreeze the
    reference implementation, which is the drift ADR-0004 exists to prevent.
    """
    search_space = set(tuning.VALENDIN_SEARCH_DEFAULTS)
    assert search_space == {"learning_rate", "weight_decay", "batch_size"}

    params = tuning._suggest_params_for("valendin_lstm", _FixedTrial(), {})
    model = tuning._build_model_for("valendin_lstm", params, METADATA)
    # The published sizes, whatever Optuna sampled.
    assert model.backbone.lstm.hidden_size == 128
    assert model.backbone.dense.out_features == 128


@pytest.mark.parametrize("model_type", VALID_MODEL_TYPES)
def test_read_side_agrees_on_which_types_run_per_study(model_type, tmp_path):
    """The archive reader classifies each registered type the way the runner ran it.

    `studies.analysis` decides whether a model's folder holds one prediction per study
    or a single deterministic fit. It used to answer from its own copy of the neural
    list, which drifted: `valendin_lstm` was missing, so every study index resolved to
    `Prediction_1.csv` and the benchmark's across-study spread silently collapsed to
    one study. Nothing crashed — the reported distribution was just wrong.
    """
    from panelclv.studies.analysis import _is_deterministic_model

    model_dir = tmp_path / "SomeModel"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))

    # Neural types are Optuna-tuned once per study; anything else is a single fit.
    assert _is_deterministic_model(model_dir) is (model_type not in NEURAL_MODEL_TYPES)
