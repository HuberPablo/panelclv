"""The shipped entry point's config can actually build the models it lists.

`scripts/run_studies.py` is a live entry point (`CLAUDE.md`, "scripts/"), and it is
the file a new run is started by copying. Nothing else in the suite reads it, so a
config that no longer builds a model sits there looking correct until someone runs
it — and then fails *inside the first Optuna trial*, after `prepare_dataset` has
already succeeded, which reads as a package bug rather than a config typo.

That is not hypothetical: the config shipped without `embedded_cols` from the day it
was written, so every neural model in it raised `ValueError: embedded_cols must be a
{column: cardinality} dict` on trial 0. The Pareto/NBD baseline has no embedder and
would have run fine, which is exactly why the gap survived — the script is not
all-or-nothing broken.

These tests are static: no panel is read, nothing is trained.
"""

import importlib.util
from pathlib import Path

import pytest

from panelclv.configs.panel_config import normalize_embedded_cols
from panelclv.registry import is_neural

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_studies.py"


@pytest.fixture(scope="module")
def run_studies():
    """`scripts/run_studies.py` imported as a module.

    Loaded by path because `scripts/` is not a package and is deliberately outside
    the wheel. Importing runs only module-level code — the constants and the
    `STUDIES_BASE` path — because the script guards `main()` behind `__main__`.
    """
    spec = importlib.util.spec_from_file_location("run_studies", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_entry_point_embeds_its_target(run_studies):
    """The target column is declared in `embedded_cols`.

    The target's own column is one of the model's inputs and its cardinality *is*
    the softmax head size (`CLAUDE.md`), so a config that omits it can build no
    neural model. `Embedder` enforces this at construction; this test moves the
    failure to the suite, where it costs seconds instead of a rented GPU box.
    """
    config = run_studies.build_panel_config()
    embedded = normalize_embedded_cols(config.embedded_cols)

    assert config.target_col in embedded, (
        f"build_panel_config() lists models that need an embedder, but its "
        f"embedded_cols {dict(embedded)} does not declare the target "
        f"{config.target_col!r} — every neural trial will raise on trial 0"
    )


def test_the_entry_point_lists_a_neural_model(run_studies):
    """Guards the test above from going vacuous.

    If `build_models()` ever drops to the Pareto/NBD baseline alone, the embedding
    requirement stops applying and the assertion above would pass for the wrong
    reason. This says out loud that it is still load-bearing.
    """
    assert any(is_neural(spec.model_type) for spec in run_studies.build_models())
