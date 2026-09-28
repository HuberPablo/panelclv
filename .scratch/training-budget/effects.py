"""Compatibility shim: the scripts under `.scratch/` that import `effects` get the package's
single implementation, `panelclv.evaluation.effects` (`docs/statistical-protocol.md`).

What stays here is what is specific to `docs/training-budget.md`: the measured refit noise
per panel, which `effect` fills in automatically so every printed effect carries its
magnitude reference.

The old standalone `effect(b, a, metric, panel)` had no `paired` argument and always meant
independent replications (separate Optuna searches per condition). This shim keeps that
default, so existing callers are unchanged; a caller comparing the same studies scored two
ways passes ``paired=True``.

    from effects import effect, REFIT_NOISE
    effect(better, baseline, metric="spearman", panel="electronics")
"""
from __future__ import annotations

from panelclv.evaluation.effects import Effect, table  # noqa: F401  (re-exported)
from panelclv.evaluation.effects import effect as _effect

# Mean |movement| when the SAME checkpoint is refit a second time and nothing else
# changes: 80 archived family-N winners paired with their re-scored refits
# (`refit_noise.py`, results/refit_noise.csv). Printed beside every effect so a reader can
# tell a large difference from a small one — never used as a threshold.
REFIT_NOISE: dict[str, dict[str, float]] = {
    "cdnow":        {"mape_aggregate": 5.83, "bias_percent": 12.58, "spearman": 0.0159},
    "electronics":  {"mape_aggregate": 3.63, "bias_percent": 5.94,  "spearman": 0.0105},
    "gift":         {"mape_aggregate": 5.89, "bias_percent": 14.99, "spearman": 0.0116},
    "multichannel": {"mape_aggregate": 7.71, "bias_percent": 12.56, "spearman": 0.0152},
}


def effect(b, a, metric: str, panel: str, *, paired: bool = False) -> Effect:
    """`panelclv.evaluation.effects.effect`, with this document's refit noise filled in.

    ``paired`` defaults to False: the callers written against the old standalone version
    all compare independent replications.
    """
    return _effect(b, a, paired=paired, metric=metric, panel=panel,
                   refit_noise=REFIT_NOISE.get(panel, {}).get(metric))
