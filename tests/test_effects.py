"""`evaluation.effects` implements the statistical protocol's decision rule."""

import numpy as np

from panelclv.evaluation import effect


def test_a_clear_shift_is_supported_and_no_shift_is_not():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 20)
    assert effect(a + 3.0, a[::-1], paired=False, metric="m", panel="p").supported
    assert not effect(rng.normal(0.0, 1.0, 20), a, paired=False, metric="m", panel="p").supported


def test_paired_and_independent_estimate_the_same_delta():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 10)
    b = a + rng.normal(0.5, 0.1, 10)
    paired = effect(b, a, paired=True, metric="m", panel="p")
    independent = effect(b, a, paired=False, metric="m", panel="p")
    assert np.isclose(paired.delta, independent.delta)


def test_pairing_removes_unit_to_unit_spread():
    # Panels differ wildly in difficulty (MAPE 50 to 1,250), B is better by ~6 on each.
    a = np.array([50.0, 180.0, 900.0, 1250.0, 70.0, 400.0, 620.0, 95.0, 310.0, 1100.0])
    b = a - np.array([5, 6, 7, 7, 5, 6, 6, 5, 6, 7], float)
    paired = effect(b, a, paired=True, metric="mape", panel="grid")
    independent = effect(b, a, paired=False, metric="mape", panel="grid")
    assert paired.supported and not independent.supported
    assert paired.hi - paired.lo < independent.hi - independent.lo


def test_paired_drops_nan_pairs_together():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([2.0, np.nan, 5.0, 6.0])
    e = effect(b, a, paired=True, metric="m", panel="p")
    assert e.n_a == e.n_b == 2
