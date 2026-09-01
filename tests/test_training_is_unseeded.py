"""The package does not seed the training path, and `CLAUDE.md` says so.

`CLAUDE.md`'s reproducibility priority draws a boundary: every seed the package owns —
the Optuna sampler, the Monte Carlo forecast, the synthetic panels — derives from one
config value, while model training does not. Weight initialisation, `DataLoader`
shuffling and dropout draw on the *global* torch RNG, which nothing under `training/`,
`tuning/`, `trials/` or `studies/` sets. An entry point that wants a repeatable run
sets it itself, as `notebooks/Study.ipynb` does.

That is a claim about an absence, which is exactly the kind `test_docs_are_current.py`
says it cannot catch: nothing is renamed when it stops being true. So it is pinned
here instead. If a future change seeds the training path deliberately, this test, the
reproducibility priority in `CLAUDE.md` and the `base_seed` docstring in
`studies/config.py` move together — that is the point of the failure, not an obstacle
to it.

`models/monte_carlo_forecasting.py` seeds legitimately (a forecast takes its own seed)
and sits outside the scanned subpackages.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The four subpackages that stand between a config and a trained model. The forecaster
# and the data generators are deliberately not here: they take a seed and use it.
UNSEEDED_SUBPACKAGES = ["training", "tuning", "trials", "studies"]

# Every way the global RNG gets set. Matched as plain text, the same deliberately loose
# style `test_docs_are_current.py` uses — a false positive is a comment mentioning one
# of these, which is cheap to fix, while a miss defeats the test.
SEEDING_CALLS = re.compile(
    r"\b(?:torch\.manual_seed|torch\.cuda\.manual_seed(?:_all)?|torch\.random\.manual_seed"
    r"|np\.random\.seed|numpy\.random\.seed|random\.seed)\s*\("
)


def _modules():
    """Every module under the four subpackages, one parametrised case each."""
    for name in UNSEEDED_SUBPACKAGES:
        for path in sorted((REPO_ROOT / "src" / "panelclv" / name).rglob("*.py")):
            relative = path.relative_to(REPO_ROOT)
            yield pytest.param(relative, id=str(relative))


@pytest.mark.parametrize("relative_path", list(_modules()))
def test_training_path_does_not_seed_the_global_rng(relative_path):
    """No module on the training path sets the global torch or numpy RNG."""
    source = (REPO_ROOT / relative_path).read_text()
    offenders = [
        f"{relative_path}:{i}: {line.strip()}"
        for i, line in enumerate(source.splitlines(), start=1)
        if SEEDING_CALLS.search(line)
    ]
    assert not offenders, (
        "The training path seeds the global RNG:\n  "
        + "\n  ".join(offenders)
        + "\n\nThat contradicts the reproducibility priority in `CLAUDE.md` and the "
        "`base_seed` docstring in `src/panelclv/studies/config.py`, both of which say "
        "training is not seeded. If seeding it is the intent, update those two and this "
        "test in the same commit — and note that no study archived under `Studies/` was "
        "produced with a seeded training path, so none of them reproduce against the new "
        "baseline."
    )
