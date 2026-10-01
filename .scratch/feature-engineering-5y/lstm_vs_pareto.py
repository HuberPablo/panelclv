"""Every neural cell on electronic_5y against Pareto/NBD. Backs "The LSTM against
Pareto/NBD" and the Pareto/NBD sentences of `docs/feature-engineering.md` §4.

Pareto/NBD is 20 seeded MCMC fits on the same cohort and windows; each neural cell is 20
studies with unseeded training. The two share no unit, so each comparison is the
protocol's independent bootstrap (`docs/statistical-protocol.md` §2):

    delta = mean(cell) - mean(Pareto/NBD), 95% percentile interval, n = 20 / 20,

through the package's `effect` (via the training-budget shim). Supported when the interval
excludes zero. Reads the per-forecast scores `effects_5y.py` writes (run it first) and
writes `results/lstm_vs_pareto.csv` with one row per (model, rule, input, metric).

    PYTHONPATH=src python .scratch/feature-engineering-5y/lstm_vs_pareto.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / ".scratch" / "training-budget"))
from effects import effect                                               # noqa: E402

# Signed bias answers "which way does each miss", |bias| "which misses by less".
METRICS = ("rmse_customer_total", "abs_bias", "mape_aggregate", "spearman", "bias_percent")
FMT = {"rmse_customer_total": "{:+.3f}", "abs_bias": "{:+.1f}", "mape_aggregate": "{:+.1f}",
       "spearman": "{:+.3f}", "bias_percent": "{:+.1f}"}

per = pd.read_csv(HERE / "results" / "per_forecast.csv")
pn = per[per.model == "ParetoNBD"]
assert len(pn) == 20, f"{len(pn)} Pareto/NBD fits in per_forecast.csv; run effects_5y.py"

rows = []
for (model, feature, arm), g in per[per.model != "ParetoNBD"].groupby(
        ["model", "feature", "arm"], sort=False):
    for k in METRICS:
        e = effect(g[k], pn[k], k, "electronic_5y")
        rows.append(dict(model=model, rule=arm, input=feature, metric=k, n_a=e.n_a,
                         n_b=e.n_b, mean_pnbd=e.mean_a, mean_cell=e.mean_b, delta=e.delta,
                         lo=e.lo, hi=e.hi, supported=e.supported))
out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "lstm_vs_pareto.csv", index=False)

print("Pareto/NBD, 20 fits:", {k: round(pn[k].mean(), 4) for k in METRICS})
for model in ("LSTM", "LSTMAttention", "Transformer"):
    print(f"\n### {model} − Pareto/NBD (n = 20 / 20, independent)\n")
    print("| rule | input | Δ RMSE (customer total) | Δ \\|bias\\| | Δ MAPE | Δ Spearman | Δ bias % |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    m = out[out.model == model]
    for (arm, feature), g in m.groupby(["rule", "input"], sort=False):
        cells = []
        for k in METRICS:
            r = g[g.metric == k].iloc[0]
            f = FMT[k]
            txt = f"{f.format(r.delta)} [{f.format(r.lo)}, {f.format(r.hi)}]"
            cells.append(f"**{txt}**" if r.supported else txt)
        inp = "none" if feature == "none" else f"`{feature}`"
        print(f"| `{arm}` | {inp} | " + " | ".join(cells) + " |")
