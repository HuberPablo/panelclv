"""Every model type is declared once, in the registry table, and read from there.

ADR-0006: adding a model means adding one entry to ``registry.MODEL_REGISTRY``.
Before it, the model set was enumerated seven times and one of those copies had
already drifted. So these tests do not check that several lists agree — there is
one table — they check that the table is *complete* and that every derivation off
it lands on the right thing.

They assert more than key membership. The tuning path dispatches on ``model_type``
and two of its sites historically fell through to the Transformer on an
unrecognised type, so a half-registered model would train the wrong architecture
under the right name. Membership alone would not catch that; the built model's
*class* is checked per type, which is the assertion this file was written for and
which the registry does not make redundant.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from panelclv.registry import (  # noqa: E402
    MODEL_REGISTRY,
    MODEL_TYPES,
    build_model,
    entry,
    is_neural,
    rollout_for,
    suggest_params,
    validate_model_knobs,
)
from panelclv.tuning import optuna_tuning as tuning  # noqa: E402

# The architecture each neural model type must build. Getting a different class
# here means a dispatch site fell through — the silent failure these tests exist
# for.
EXPECTED_CLASS = {
    "lstm": "MultinomialLSTMModel",
    "transformer": "MultinomialTransformerModel",
    "valendin_lstm": "ValendinLSTMModel",
}

# Types with a training builder, read off the table rather than restated: a second
# list here would be the copy ADR-0006 exists to make unwritable.
NEURAL_TYPES = [t for t in MODEL_TYPES if is_neural(t)]

# A small panel every model type can be built against.
RECIPE = {
    "seq_cols": ["Transactions", "week_idx"],
    "embedded_cols": {"Transactions": 6, "week_idx": 52},
    "target_col": "Transactions",
    "seq_len": 8,
}


class _FixedTrial:
    """Stands in for an optuna.Trial, returning the low end of every search range.

    Enough to drive the suggesters without an Optuna study; the values do not
    matter, only that every declared parameter can be sampled.
    """

    def suggest_categorical(self, name, choices):
        return sorted(choices)[0]

    def suggest_float(self, name, low, high, **kwargs):
        return low

    def suggest_int(self, name, low, high, **kwargs):
        return low


def test_model_types_are_the_tables_keys():
    """The enumeration IS the table — not a list kept in step with it."""
    assert MODEL_TYPES == tuple(MODEL_REGISTRY)


def test_neural_is_read_off_the_entry():
    """"Neural" is the predicate "this entry has a training builder", not a list.

    This is the copy that drifted once (a stale neural list in the archive reader
    collapsed the Valendin benchmark's across-study spread to one study), so the
    derivation is pinned rather than assumed.
    """
    for model_type in MODEL_TYPES:
        assert is_neural(model_type) is (entry(model_type).build is not None)
    assert set(NEURAL_TYPES) == set(EXPECTED_CLASS)


def test_pareto_entry_is_declarative():
    """Pareto/NBD sits in the table with empty fields, not outside it.

    It is a valid model type with no search space, no builder and no rollout; its
    entry exists so every enumeration still derives from one place, and the runner
    keeps a separate deterministic path for it.
    """
    pareto = entry("pareto_nbd")
    assert (pareto.search_space, pareto.suggest, pareto.build, pareto.rollout) == (
        None, None, None, None,
    )
    assert is_neural("pareto_nbd") is False


@pytest.mark.parametrize("model_type", NEURAL_TYPES)
def test_neural_entry_is_complete(model_type):
    """Every field a neural model needs is filled in, in one place."""
    e = entry(model_type)
    assert e.search_space, f"{model_type} declares no search space"
    assert callable(e.suggest)
    assert callable(e.build)
    assert callable(e.rollout)

    # Validation recognises the type rather than raising "Unknown model_type".
    validate_model_knobs(model_type, {}, {})
    params = suggest_params(model_type, _FixedTrial(), {})
    assert isinstance(params, dict) and params, f"{model_type} sampled no parameters"


@pytest.mark.parametrize("model_type", NEURAL_TYPES)
def test_neural_type_builds_its_own_architecture(model_type):
    """The check membership alone cannot make: the right class comes out.

    A dispatch site that falls through to another family would pass every
    table-based assertion above and still train the wrong model.
    """
    params = suggest_params(model_type, _FixedTrial(), {})
    model = build_model(model_type, params, RECIPE)
    assert type(model).__name__ == EXPECTED_CLASS[model_type]


@pytest.mark.parametrize("model_type", NEURAL_TYPES)
def test_the_trained_model_hands_over_its_own_rollout_model(model_type):
    """Every neural type answers ``to_rollout()``, and shares its backbone (ADR-0007).

    There used to be a second construction to keep in step, and a mismatch surfaced
    only after a full training run. Now the pairing is the trained class's own — so
    what is worth pinning is that the handover exists for every registered type and
    that it hands over the *same* weights object, not a copy that could drift from it.
    """
    params = suggest_params(model_type, _FixedTrial(), {})
    trained = build_model(model_type, params, RECIPE)

    rollout = trained.to_rollout()
    assert rollout.backbone is trained.backbone
    assert rollout.num_target_classes == trained.num_target_classes


@pytest.mark.parametrize("model_type", NEURAL_TYPES)
def test_rollout_is_declared_by_the_entry(model_type):
    """The suite reads the rollout off the table instead of its own forecaster map.

    A type registered for tuning but missing from that map used to fail only after
    the study had trained; there is no second map to be missing from now.
    """
    assert rollout_for(model_type) is entry(model_type).rollout


def test_unknown_model_type_is_rejected_everywhere():
    """No lookup may quietly fall through to another architecture."""
    for call in (
        lambda: entry("nonesuch"),
        lambda: validate_model_knobs("nonesuch", {}, {}),
        lambda: suggest_params("nonesuch", _FixedTrial(), {}),
        lambda: build_model("nonesuch", {}, RECIPE),
        lambda: rollout_for("nonesuch"),
    ):
        with pytest.raises(ValueError, match="model_type"):
            call()


def test_declarative_entry_is_rejected_where_it_has_nothing_to_offer():
    """`pareto_nbd` is a valid type that cannot be tuned, built or rolled out.

    It must fail loudly at each of those, rather than returning ``None`` for a
    caller to trip over later.
    """
    for call in (
        lambda: validate_model_knobs("pareto_nbd", {}, {}),
        lambda: suggest_params("pareto_nbd", _FixedTrial(), {}),
        lambda: build_model("pareto_nbd", {}, RECIPE),
        lambda: rollout_for("pareto_nbd"),
    ):
        with pytest.raises(ValueError, match="pareto_nbd"):
            call()


def test_a_knob_in_the_wrong_dict_is_rejected():
    """Both misplacements are silent otherwise, and both waste a whole search.

    The one allowlist is the entry's own search space, so it cannot drift from what
    the suggester actually samples — which is what replaced the hand-maintained list
    of keys that were *not* search parameters.
    """
    # A parameter the model does not have: a typo, not a knob.
    with pytest.raises(ValueError, match="hiddendim"):
        validate_model_knobs("lstm", {"hiddendim": {32, 64}}, {})

    # A training control in the search space.
    with pytest.raises(ValueError, match="n_epochs"):
        validate_model_knobs("lstm", {"n_epochs": 10}, {})

    # A real hyperparameter left in `training`, where nothing would search it.
    with pytest.raises(ValueError, match="dropout"):
        validate_model_knobs("lstm", {}, {"dropout": {0.0, 0.2}})

    # And a misspelled training control, which the tuner polices against the keys it
    # actually reads — the half of the old allowlist that is not per-model.
    with pytest.raises(ValueError, match="paitence"):
        tuning._validate_training("lstm", {"paitence": 7})


def test_the_embedder_is_a_pinnable_search_dimension():
    """Which embedding strategy a model is built with is chosen by `params`.

    The seam (ADR-0005) only pays off if a strategy can be selected per study rather
    than per model type — otherwise the ablation needs a second registry entry, which
    is the duplication ADR-0006 exists to prevent. So both arms are pinned here and
    the built embedder is checked by class: a dispatch that fell through to a default
    would still train, and would silently report the wrong architecture.
    """
    for name, expected in (("projected", "ProjectedEmbedder"),
                           ("valendin", "ValendinEmbedder")):
        params = suggest_params("lstm", _FixedTrial(), {"embedder": name})
        assert params["embedder"] == name
        model = build_model("lstm", params, RECIPE)
        assert type(model.backbone.embedder).__name__ == expected

    # `embedding_dim` is the ProjectedEmbedder's common projection width. The
    # Valendin strategy has no such knob, so it must not be registered as a search
    # dimension that spends the trial budget without reaching the model.
    valendin = suggest_params("lstm", _FixedTrial(), {"embedder": "valendin"})
    assert "embedding_dim" not in valendin
    projected = suggest_params("lstm", _FixedTrial(), {"embedder": "projected"})
    assert "embedding_dim" in projected


def test_a_pinned_scalar_reaches_the_recorded_params():
    """Pinning a hyperparameter to a scalar must still record it on the trial.

    The scalar branch used to return the value without registering anything, so the
    key was simply absent from `study.best_trial.params` — and anything reading that
    dict raised `KeyError` after every trial had already trained. Registering it as a
    one-element categorical keeps the pin exact (one choice is no choice) while
    leaving a complete record of what was used.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest_params("lstm", trial, {"dropout": 0.0, "dense_units": 32})
        # The pinned values are what comes back, not a sampled substitute.
        assert params["dropout"] == 0.0
        assert params["dense_units"] == 32
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=2)

    assert study.best_trial.params["dropout"] == 0.0
    assert study.best_trial.params["dense_units"] == 32


def test_study_entry_point_accepts_every_registered_type():
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
    assert set(entry("valendin_lstm").search_space) == {
        "learning_rate", "weight_decay", "batch_size",
    }

    params = suggest_params("valendin_lstm", _FixedTrial(), {})
    model = build_model("valendin_lstm", params, RECIPE)
    # The published sizes, whatever Optuna sampled.
    assert model.backbone.lstm.hidden_size == 128
    assert model.backbone.dense.out_features == 128


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_read_side_agrees_on_which_types_run_per_study(model_type, tmp_path):
    """The archive reader classifies each registered type the way the runner ran it.

    `studies.analysis` decides whether a model's folder holds one prediction per study
    or a single deterministic fit. It used to answer from its own copy of the neural
    list, which drifted: `valendin_lstm` was missing, so every study index resolved to
    `Prediction_1.csv` and the benchmark's across-study spread silently collapsed to
    one study. Nothing crashed — the reported distribution was just wrong.
    """
    from panelclv.studies.suite_reader import _is_deterministic_model

    model_dir = tmp_path / "SomeModel"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))

    # Neural types are Optuna-tuned once per study; anything else is a single fit.
    assert _is_deterministic_model(model_dir) is (not is_neural(model_type))
