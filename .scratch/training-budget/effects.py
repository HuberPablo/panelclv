"""The one place a comparison in `docs/training-budget.md` is computed.

The document's standard, stated in its "How claims are made" front matter:

  * one metric per claim — Spearman for discrimination, MAPE for level, |bias| secondary,
    RMSE descriptive;
  * report the difference between condition means, delta = mean(B) - mean(A);
  * with a 95% BOOTSTRAP CONFIDENCE INTERVAL for that difference;
  * a result is statistically supported when the interval excludes zero;
  * the measured refit noise is printed beside it as a magnitude reference, never as a
    second significance hurdle.

Conditions are independent samples — replications are not paired across arms, see the
seeding protocol in the same front matter — so the bootstrap resamples each condition
separately.

    from effects import effect, REFIT_NOISE
    effect(better, baseline, metric="spearman", panel="electronics")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap

# Mean |movement| when the SAME checkpoint is refit a second time and nothing else
# changes: 80 archived family-N winners paired with their re-scored refits
# (`refit_noise.py`, results/refit_noise.csv). Printed beside every effect so a reader can
# tell a large difference from a small one.
REFIT_NOISE: dict[str, dict[str, float]] = {
    "cdnow":        {"mape_aggregate": 5.83, "bias_percent": 12.58, "spearman": 0.0159},
    "electronics":  {"mape_aggregate": 3.63, "bias_percent": 5.94,  "spearman": 0.0105},
    "gift":         {"mape_aggregate": 5.89, "bias_percent": 14.99, "spearman": 0.0116},
    "multichannel": {"mape_aggregate": 7.71, "bias_percent": 12.56, "spearman": 0.0152},
}

# Which metric may carry which claim. RMSE is absent on purpose: on these panels every
# arm sits within 0.004 of the all-zero forecast, so it ranks nothing.
PRIMARY_METRIC = {"discrimination": "spearman", "level": "mape_aggregate"}


@dataclass
class Effect:
    """One comparison, in the form the document reports it."""

    metric: str
    panel: str
    mean_a: float
    mean_b: float
    delta: float
    lo: float
    hi: float
    n_a: int
    n_b: int
    refit_noise: float | None

    @property
    def supported(self) -> bool:
        """The whole criterion: does the 95% interval exclude zero?"""
        return (self.lo > 0) or (self.hi < 0)

    def equivalent_within(self, margin: float | None = None) -> bool:
        """True when the interval lies entirely inside +/- margin (default: the refit noise).

        Deliberately uses the 95% interval rather than the conventional 90% TOST one, so
        the document keeps a single interval convention; that makes this conservative.
        """
        m = self.refit_noise if margin is None else margin
        return m is not None and self.lo > -m and self.hi < m

    def __str__(self) -> str:
        mark = "supported" if self.supported else "not supported"
        rel = ""
        if self.refit_noise:
            rel = (f", refit noise {self.refit_noise:.4g}"
                   f"{' — below it' if abs(self.delta) < self.refit_noise else ''}")
        return (f"{self.panel}/{self.metric}: {self.mean_a:.4g} -> {self.mean_b:.4g}, "
                f"delta {self.delta:+.4g} [{self.lo:+.4g}, {self.hi:+.4g}] "
                f"({mark}{rel})")


def effect(b, a, metric: str, panel: str, n_resamples: int = 10000,
           seed: int = 0) -> Effect:
    """Delta = mean(b) - mean(a) with its 95% bootstrap CI. `b` is the new condition."""
    b, a = np.asarray(b, float), np.asarray(a, float)
    res = bootstrap((b, a), lambda x, y: x.mean() - y.mean(), n_resamples=n_resamples,
                    method="percentile", random_state=seed, vectorized=False)
    return Effect(metric=metric, panel=panel, mean_a=a.mean(), mean_b=b.mean(),
                  delta=b.mean() - a.mean(),
                  lo=res.confidence_interval.low, hi=res.confidence_interval.high,
                  n_a=len(a), n_b=len(b),
                  refit_noise=REFIT_NOISE.get(panel, {}).get(metric))


def table(effects: list[Effect], label: str = "comparison") -> str:
    """The markdown table the document prints."""
    out = [f"| {label} | mean A | mean B | delta | 95% CI | supported | refit noise |",
           "| --- | ---: | ---: | ---: | :---: | :---: | ---: |"]
    for e in effects:
        out.append(f"| {e.panel} | {e.mean_a:.4g} | {e.mean_b:.4g} | {e.delta:+.4g} | "
                   f"{e.lo:+.4g} to {e.hi:+.4g} | {'yes' if e.supported else 'no'} | "
                   f"{e.refit_noise:.4g} |" if e.refit_noise else
                   f"| {e.panel} | {e.mean_a:.4g} | {e.mean_b:.4g} | {e.delta:+.4g} | "
                   f"{e.lo:+.4g} to {e.hi:+.4g} | {'yes' if e.supported else 'no'} | — |")
    return "\n".join(out)
