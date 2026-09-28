"""The tests behind `docs/insights-cluster-ablation.md` §3-§5, under the statistical protocol.

Family F: six LSTM arms on two real panels, 40 replications per arm and panel, each one
complete Optuna search, refit and forecast (two shards of 20 seeds). Training is unseeded,
so study i of one arm shares nothing with study i of another beyond a seed label: every
comparison is between INDEPENDENT replications, `effect(..., paired=False)`
(`docs/statistical-protocol.md` §2), and each panel is tested on its own.

* Level (|bias %|, MAPE) is read from each suite's `results.csv`.
* Spearman is recomputed from the stored forecasts with the scoring code of
  `scripts/run_cluster_ablation.py --report` (imported, not copied), electronics only:
  CDNOW's stored forecasts are 38 holdout weeks wide and today's week rule (ADR-0009)
  rebuilds a 39-week holdout, so they cannot be scored.
* Pareto/NBD on these electronics windows is a single fit here (n = 1). Seeded refits of
  the same panel and windows are being produced under
  `Studies/real_panel_benchmarks__ParetoNBD__electronics__rNN`; until they exist, no
  comparison with Pareto/NBD is tested.

    PYTHONPATH=src .../python .scratch/statistical-protocol/grid_cluster_ablation_effects.py
"""
import sys
from pathlib import Path

import pandas as pd

from panelclv.evaluation.effects import effect, table

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_cluster_ablation as rca  # noqa: E402  (the family's own scoring code)

PANELS = ("electronics", "cdnow")


def level(panel: str, arm: str) -> pd.DataFrame:
    """The 40 replications of one arm on one panel, both shards, from results.csv."""
    frames = [pd.read_csv(REPO / "Studies" / rca.suite_name(panel, arm, shard) / "results.csv")
              for shard in rca.SHARDS]
    d = pd.concat(frames, ignore_index=True)
    d["abs_bias"] = d.bias_percent.abs()
    return d


def spearman(arm: str) -> pd.Series:
    """Per-study Spearman of one electronics arm, scored from its stored forecasts."""
    panel_path, build = rca.PANELS["electronics"]
    data = rca.panel_dataset.prepare_dataset(pd.read_csv(panel_path), build(rca.ARMS[arm]),
                                             verbose=False)
    rows = []
    for shard, base_seed in rca.SHARDS.items():
        rows += rca.score_suite(REPO / "Studies" / rca.suite_name("electronics", arm, shard), data, base_seed)
    return pd.DataFrame(rows).spearman


def compare(base: str, arms, metric: str, values) -> list:
    """Effect of each arm against `base` on one metric, per panel, independent."""
    out = []
    for panel in values:
        a = values[panel][base]
        for arm in arms:
            e = effect(values[panel][arm], a, paired=False, metric=metric, panel=f"{panel}: {arm}")
            out.append(e)
    return out


lv = {p: {arm: level(p, arm) for arm in rca.ARMS} for p in PANELS}
CLUSTER_AND_AR = ["cluster_4", "cluster_8", "cluster_16", "ar_unbounded", "ar_plus_cluster_8"]

print("## §4 — each arm against `no_cluster`, |bias %| (independent, 40 vs 40)\n")
print(table(compare("no_cluster", CLUSTER_AND_AR, "abs_bias",
                    {p: {a: lv[p][a].abs_bias for a in lv[p]} for p in PANELS}), "panel: arm"))
print("\n## §4 — each arm against `no_cluster`, MAPE (independent, 40 vs 40)\n")
print(table(compare("no_cluster", CLUSTER_AND_AR, "mape",
                    {p: {a: lv[p][a].mape_aggregate for a in lv[p]} for p in PANELS}), "panel: arm"))

print("\n## §5.2 — `ar_plus_cluster_8` against `ar_unbounded` (independent, 40 vs 40)\n")
for metric, col in (("abs_bias", "abs_bias"), ("mape", "mape_aggregate")):
    print(f"{metric}:\n")
    print(table(compare("ar_unbounded", ["ar_plus_cluster_8"], metric,
                        {p: {a: lv[p][a][col] for a in lv[p]} for p in PANELS}), "panel: arm") + "\n")

print("## §5.1 — electronics Spearman, each arm against `no_cluster` (independent)\n")
sp = {arm: spearman(arm) for arm in ["no_cluster", "cluster_4", "cluster_8", "cluster_16",
                                     "ar_unbounded", "ar_plus_cluster_8"]}
for arm, v in sp.items():
    print(f"{arm}: n {len(v)}, mean {v.mean():.3f}, sd {v.std(ddof=1):.3f}")
print()
print(table(compare("no_cluster", CLUSTER_AND_AR, "spearman", {"electronics": sp}), "panel: arm"))

# --- §5.1 against Pareto/NBD ----------------------------------------------------------------
# Twenty seeded Pareto/NBD refits on the same electronics panel and windows
# (`scripts/run_real_panel_benchmarks.py`, default calibration). Each refit is one
# replication; scored against the family's own rebuilt cohort, with the ids checked, so
# row i of every forecast is the same customer. Independent of the LSTM studies.
panel_path, build = rca.PANELS["electronics"]
data = rca.panel_dataset.prepare_dataset(pd.read_csv(panel_path), build(rca.ARMS["no_cluster"]), verbose=False)
actual_totals = rca.holdout_actuals(data).sum(axis=1)
pnbd = []
for root in sorted((REPO / "Studies").glob("real_panel_benchmarks__ParetoNBD__electronics__r*")):
    values, ids = rca.load_model_predictions(root / "ParetoNBD", study=1)
    if ids is not None and not (len(ids) == len(data["ids"]) and (pd.Series(ids).values == pd.Series(data["ids"]).values).all()):
        raise ValueError(f"{root.name}: forecast ids do not match the family-F cohort")
    pnbd.append(rca.spearman(values.sum(axis=1), actual_totals))
pnbd = pd.Series(pnbd)
print(f"\nPareto/NBD refits: n {len(pnbd)}, mean {pnbd.mean():.3f}, sd {pnbd.std(ddof=1):.3f}\n")
print(table([effect(sp[arm], pnbd, paired=False, metric="spearman", panel=f"electronics: Pareto/NBD → {arm}")
             for arm in ["no_cluster", "cluster_4", "cluster_8", "cluster_16", "ar_plus_cluster_8"]], "panel: arm"))
