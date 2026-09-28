"""The tests behind `docs/insights-study.md` §4.1, §5.4 and §8, under the statistical protocol.

Every comparison goes through `panelclv.evaluation.effects.effect`
(`docs/statistical-protocol.md`); this script only decides what is paired with what.

§4.1  P-sLSTM against the LSTM on electronics, 8 runs each (`.scratch/p-slstm/`). Run k of
      both models called `torch.manual_seed(k)`, but two different architectures consume
      that random stream differently, so seed k is not a shared experimental unit: the
      runs are independent replications, `paired=False` (as in `docs/p-slstm.md`). The
      per-seed validation CE was never stored (only a range for six of the eight seeds),
      so the CE comparison cannot be re-tested.

§5.4  Selection criteria on the `selection_rescore` studies (`scripts/run_selection_rescore.py`).
      One study (one Optuna search) is one replication; each gives one rank correlation
      between a criterion and a holdout target over its trials. Per panel and per model:
      the criterion's own correlation against 0, and the rollout criterion against
      validation CE paired by study (both scored on the same trials). Same inputs and
      method as `.scratch/training-budget/selection_analysis.py`, recomputed here.

§8    Family G (`Studies/_archive_50sim/`), 20 independent studies per (model, panel, arm):
      the LSTM against the Transformer under `ar_unbounded`, and `kmeans_8` against no
      cluster under `ar_bounded` for the LSTM, on |bias %|, independent, per panel.

    PYTHONPATH=src .../python .scratch/statistical-protocol/grid_insights_study_effects.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.evaluation.effects import effect, table

REPO = Path(__file__).resolve().parents[2]

# --- §4.1 ---------------------------------------------------------------------------------
runs = json.loads((REPO / ".scratch/p-slstm/comparison-results.json").read_text())
print("## §4.1 — P-sLSTM against the LSTM, electronics, independent (8 / 8)\n")
eff = []
for metric in ("mape_aggregate", "bias_percent", "rmse"):
    a = np.array(runs["lstm"][metric]["runs"])
    b = np.array(runs["p_slstm"][metric]["runs"])
    if metric == "bias_percent":          # calibration accuracy is judged on |bias|
        a, b, metric = np.abs(a), np.abs(b), "abs_bias"
    eff.append(effect(b, a, paired=False, metric=metric, panel=f"electronics: {metric}"))
print(table(eff, "metric (LSTM → P-sLSTM)"))
b = np.array(runs["p_slstm"]["bias_percent"]["runs"])
a = np.array(runs["lstm"]["bias_percent"]["runs"])
print("\nsigned bias:\n")
print(table([effect(b, a, paired=False, metric="bias", panel="electronics: bias")], "metric"))

# --- §5.4 ---------------------------------------------------------------------------------
files = sorted((REPO / "Studies").glob("selection_rescore__*/selection_rescore.csv"))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d["panel"] = d.suite.str.split("__").str[2]
CE = lambda x: x.val_loss                                   # noqa: E731
ROLL_MAPE = lambda x: x.val_mape_aggregate                  # noqa: E731
ROLL_SPEAR = lambda x: -x.val_spearman                      # noqa: E731
TARGETS = {"holdout MAPE": lambda x: x.hold_mape_aggregate,
           "holdout |bias|": lambda x: x.hold_bias_percent.abs(),
           "holdout Spearman": lambda x: -x.hold_spearman}   # lower is better throughout
print("\n## §5.4 — within-study rank correlation, per panel and model\n")
for (panel, model), dp in d.groupby(["panel", "model"]):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]

    def rho(crit, target):
        return np.array([crit(g).corr(target(g), method="spearman") for g in studies])

    out = []
    for tname, t in TARGETS.items():
        ce = rho(CE, t)
        out.append(effect(ce, np.zeros_like(ce), paired=True, metric=tname, panel=f"val CE vs {tname}"))
    for crit, cname, tname in ((ROLL_MAPE, "val rollout MAPE", "holdout MAPE"),
                               (ROLL_SPEAR, "val rollout Spearman", "holdout Spearman")):
        r, ce = rho(crit, TARGETS[tname]), rho(CE, TARGETS[tname])
        out.append(effect(r, np.zeros_like(r), paired=True, metric=tname, panel=f"{cname} vs {tname}"))
        e = effect(r, ce, paired=True, metric=tname, panel=f"{cname} minus val CE, on {tname}")
        out.append(e)
        print(f"{panel}/{model}: {cname} beats val CE on {tname} in {int((r > ce).sum())} of {len(r)} studies")
    print(f"\n### {panel} / {model} (n = {len(studies)} studies)\n")
    print(table(out, "mean rho (A = 0) or Δ rho (A = val CE)") + "\n")

# --- §8 -----------------------------------------------------------------------------------
G = REPO / "Studies/_archive_50sim"


def fam_g(model, panel, arm):
    r = pd.read_csv(G / f"real_panel_arms__{model}__{panel}__{arm}-valendin__a" / "results.csv")
    return r.bias_percent.abs()


print("## §8 — family G, |bias %|, independent, 20 vs 20\n")
out = []
for panel in ("cdnow", "electronics"):
    for cluster in ("no_cluster", "kmeans_8"):
        arm = f"ar_unbounded-{cluster}"
        out.append(effect(fam_g("Transformer", panel, arm), fam_g("LSTM", panel, arm), paired=False,
                          metric="abs_bias", panel=f"{panel} {arm}: LSTM → Transformer"))
for panel in ("electronics", "cdnow"):
    out.append(effect(fam_g("LSTM", panel, "ar_bounded-kmeans_8"), fam_g("LSTM", panel, "ar_bounded-no_cluster"),
                      paired=False, metric="abs_bias", panel=f"{panel} LSTM ar_bounded: no_cluster → kmeans_8"))
print(table(out))
