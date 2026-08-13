"""`prepare_dataset` returns its keys under the names its docstring documents.

The output dict is the package's widest interface: every model, every rollout and the
Pareto/NBD benchmark reads it, and a notebook or a new script reads it by hand. Its keys
are therefore public API — but they are written as string literals in one dict literal,
where a typo is invisible to the compiler, to every caller that happens not to read that
key, and to a reviewer scanning an aligned block of `"key": value` lines.

That is not hypothetical. `N` — the customer count — shipped as `"N  "`, with two
trailing spaces, from the first version of this function until it was found by running
the package on an unfamiliar panel: `data["N"]` raised `KeyError` on the first line
someone outside the package wrote. Nothing inside `src/` reads the key, so no test, no
gate and no production run had ever touched it. The module docstring documented it as
`N` the whole time.

So there are two assertions here, and they fail for different reasons: the first pins
the key that was wrong, the second pins the property whose absence let it stay wrong.
"""

import pytest

from panelclv.data_preparation import panel_dataset

# The golden panel, imported rather than re-declared — the same reuse
# `test_study_suite_end_to_end.py` makes of it, for the same reason: one synthetic panel
# definition in the suite, not one per file.
from test_golden_end_to_end import _golden_config, _golden_panel


@pytest.fixture(scope="module")
def data() -> dict:
    """One prepared dataset. Module-scoped: both tests read the same finished dict."""
    return panel_dataset.prepare_dataset(_golden_panel(), _golden_config(), verbose=False)


def test_the_customer_count_is_returned_as_N(data):
    """`data["N"]` is the number of customers, spelled as the docstring spells it.

    Asserted against `ids` and the tensors rather than a literal, so the test states the
    contract — N is how many customers the tensors hold — instead of restating a number.
    """
    assert data["N"] == len(data["ids"])
    assert data["N"] == data["calibration"].shape[0] == data["holdout"].shape[0]


def test_no_output_key_carries_stray_whitespace(data):
    """No key is padded. This is the check that would have caught `"N  "` in review.

    A padded key is not a cosmetic problem: it is unreachable by the name the docs give
    it, and it fails as a `KeyError` in the caller rather than as anything the package
    can see.
    """
    padded = [key for key in data if key != key.strip()]
    assert padded == [], f"output keys with leading/trailing whitespace: {padded!r}"
