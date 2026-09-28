"""The one place a comparative claim is computed (`docs/statistical-protocol.md`).

Every claim compares two conditions, A (the baseline) and B (the change), on one metric:

    delta = mean(B) - mean(A), with its 95% percentile-bootstrap interval,
    and the claim is supported exactly when that interval excludes zero.

Only the resampling depends on how the experiment was built:

* ``paired=False`` — A and B are independent replications (n new Optuna searches per
  condition on a real panel). Each side is resampled separately.
* ``paired=True`` — replication i of A and of B share a unit (the same generated
  panel, or the same searches scored two ways). The units are resampled, carrying the
  pair (A_i, B_i) together; delta is the mean of the per-unit differences, which is the
  same number as the difference of means.

A claim about one per-replication statistic (e.g. each study's rank correlation between
a criterion and the holdout) is the paired case against zero: ``effect(rhos, zeros,
paired=True, ...)``.

    from panelclv.evaluation import effect
    effect(b, a, paired=False, metric="spearman", panel="electronics")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap

N_RESAMPLES = 10_000
CONFIDENCE = 0.95
# The bootstrap draws from its own generator so a rerun reproduces a doc's intervals to
# the last digit; it never touches the global RNG training draws on.
SEED = 0


@dataclass
class Effect:
    """One comparison, in the form every doc reports it."""

    metric: str
    panel: str
    paired: bool
    mean_a: float
    mean_b: float
    delta: float
    lo: float
    hi: float
    n_a: int
    n_b: int
    refit_noise: float | None = None

    @property
    def supported(self) -> bool:
        """The whole criterion: does the 95% interval exclude zero?"""
        return bool(self.lo > 0 or self.hi < 0)

    def equivalent_within(self, margin: float) -> bool:
        """True when the interval lies entirely inside +/- margin.

        The margin must be named before the result is looked at; the refit noise is the
        usual choice. Worded as "no difference larger than +/- margin is detectable at
        this n", never as "no difference".
        """
        return bool(self.lo > -margin and self.hi < margin)

    def __str__(self) -> str:
        mark = "supported" if self.supported else "not supported"
        noise = f", refit noise {self.refit_noise:.4g}" if self.refit_noise else ""
        return (f"{self.panel}/{self.metric}: {self.mean_a:.4g} -> {self.mean_b:.4g}, "
                f"delta {self.delta:+.4g} [{self.lo:+.4g}, {self.hi:+.4g}] "
                f"(n {self.n_a}/{self.n_b}, {'paired' if self.paired else 'independent'}, "
                f"{mark}{noise})")


def effect(b, a, *, paired: bool, metric: str, panel: str,
           refit_noise: float | None = None) -> Effect:
    """Delta = mean(b) - mean(a) with its 95% percentile-bootstrap interval.

    `b` is the new condition, `a` the baseline. With ``paired=True`` the two must be
    aligned unit by unit (same length, same order); NaN pairs are dropped together.
    """
    b, a = np.asarray(b, float), np.asarray(a, float)
    rng = np.random.default_rng(SEED)
    if paired:
        if b.shape != a.shape:
            raise ValueError(f"paired samples must align: {b.shape} vs {a.shape}")
        keep = ~(np.isnan(a) | np.isnan(b))
        a, b = a[keep], b[keep]
        diff = b - a
        # A constant difference (every unit moved by exactly the same amount) has a
        # degenerate bootstrap distribution; its interval is that point.
        if np.ptp(diff) == 0:
            lo = hi = float(diff[0])
        else:
            ci = bootstrap((diff,), np.mean, n_resamples=N_RESAMPLES,
                           confidence_level=CONFIDENCE, method="percentile",
                           random_state=rng).confidence_interval
            lo, hi = float(ci.low), float(ci.high)
    else:
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        ci = bootstrap((b, a), lambda x, y, axis=-1: x.mean(axis) - y.mean(axis),
                       n_resamples=N_RESAMPLES, confidence_level=CONFIDENCE,
                       method="percentile", random_state=rng).confidence_interval
        lo, hi = float(ci.low), float(ci.high)
    return Effect(metric=metric, panel=panel, paired=paired,
                  mean_a=float(a.mean()), mean_b=float(b.mean()),
                  delta=float(b.mean() - a.mean()), lo=lo, hi=hi,
                  n_a=len(a), n_b=len(b), refit_noise=refit_noise)


def table(effects: list[Effect], label: str = "comparison") -> str:
    """The markdown table every doc prints: one row per effect."""
    out = [f"| {label} | n (A / B) | mean A | mean B | Δ | 95% CI | supported | refit noise |",
           "| --- | :---: | ---: | ---: | ---: | :---: | :---: | ---: |"]
    for e in effects:
        noise = f"{e.refit_noise:.4g}" if e.refit_noise else "—"
        out.append(f"| {e.panel} | {e.n_a} / {e.n_b} | {e.mean_a:.4g} | {e.mean_b:.4g} | "
                   f"{e.delta:+.4g} | {e.lo:+.4g} to {e.hi:+.4g} | "
                   f"{'yes' if e.supported else 'no'} | {noise} |")
    return "\n".join(out)
