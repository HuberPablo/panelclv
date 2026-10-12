"""Every results table of `docs/insights-real-panels.md`, rebuilt with intervals and marks.

Each metric cell is the mean over replications with its 95% percentile-bootstrap
interval: `effect(x, 0, paired=True)`, the protocol's one-statistic-per-replication case
(`docs/statistical-protocol.md` §2). Within a block (one panel, one window — rows that
forecast the same customers over the same holdout):

* **bold** is the best mean of its column: highest Spearman, lowest MAPE, |bias| and RMSE.
  A bias column prints signed bias but is ranked on |bias| per replication (protocol §4).
* <u>underline</u> is a row whose Δ against the best has an interval containing 0, i.e.
  not clearly different from it. Δ is `effect(row, best, paired=False)`: every pair of
  conditions here is independent replications. A single Pareto/NBD fit (CDNOW's
  pre-ADR-0009 window) is a constant, so a comparison with it is paired against that
  constant, as `benchmarks_real_panels_effects.py` already does.
* Forecast CV is a diagnostic, not a performance metric, and is never ranked.

Sources (all per replication, all already cached or stored):

    §3    scripts/measure_forecast_collapse.py's `collect` (cached in results/census_per_forecast.csv)
    §4.1  benchmarks_real_panels_effects.load_all (families E, H, N, Pareto/NBD); the CDNOW
          Transformer `no_ar` suite has no results.csv and is scored here from its stored
          forecasts against the rebuilt 38-week holdout
    §4.3  load_all (families O, N, Pareto/NBD)
    §4.4  Studies/_archive_50sim/*/results.csv (family G), Pareto/NBD as in §4.1
    §5.1  Studies/cluster__*/results.csv (family F level); Spearman scored from stored
          forecasts with run_cluster_ablation's own scorer (electronics only)
    §5.2  .scratch/training-budget/results/factorial.csv (family U)
    §6    load_all (families O, P, N, Pareto/NBD 2y and 3y)
    §7    .scratch/feature-engineering-5y/results/per_forecast.csv (families Y and Pareto/NBD)
    §12   the same sources, every configuration per (panel, calibration window)

Writes `results/tables.md`.

    PYTHONPATH=src:scripts python .scratch/insights-tables/build_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / ".scratch" / "statistical-protocol"))

import benchmarks_real_panels_effects as bre                              # noqa: E402
import measure_forecast_collapse as mfc                                   # noqa: E402
import run_cluster_ablation as rca                                        # noqa: E402
import run_real_panel_benchmarks as bm                                    # noqa: E402
from panelclv.configs.panel_config import PanelConfig                     # noqa: E402
from panelclv.data_preparation import panel_dataset                       # noqa: E402
from panelclv.data_preparation.target_channel import holdout_actuals     # noqa: E402
from panelclv.evaluation.effects import effect                            # noqa: E402
from panelclv.models import compute_forecast_metrics                      # noqa: E402
from panelclv.studies import load_model_predictions                       # noqa: E402

S = REPO / "Studies"
OUT = HERE / "results" / "tables.md"
SPEARMAN_F = HERE / "results" / "family_f_spearman.csv"
CENSUS = HERE / "results" / "census_per_forecast.csv"
PARETO = pd.read_csv(REPO / ".scratch/statistical-protocol/results/pareto_seeded_scores.csv")

# metric -> (decimals, higher is better). `bias_percent` is shown signed, ranked on |bias|.
METRIC = {"spearman": (3, True), "mape_aggregate": (1, False), "bias_percent": (1, False),
          "abs_bias": (1, False), "rmse": (4, False), "rmse_customer_total": (3, False),
          "cv": (2, None)}


# ---------------------------------------------------------------------------------------
# Intervals and marks
# ---------------------------------------------------------------------------------------

def clean(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[~np.isnan(x)]


def level(x) -> tuple[float, float, float]:
    """Mean and its 95% interval; a single value (one fit) is its own interval."""
    x = clean(x)
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    e = effect(x, np.zeros_like(x), paired=True, metric="level", panel="")
    return e.mean_b, e.lo, e.hi


def differs(b, a) -> bool:
    """Does Δ = mean(b) − mean(a) have an interval excluding 0?"""
    b, a = clean(b), clean(a)
    if len(a) == 1 and len(b) == 1:
        return bool(a[0] != b[0])
    if len(a) == 1:            # a is one fit: paired against that constant
        return effect(b, np.full_like(b, a[0]), paired=True, metric="", panel="").supported
    if len(b) == 1:
        return effect(np.full_like(a, b[0]), a, paired=True, metric="", panel="").supported
    return effect(b, a, paired=False, metric="", panel="").supported


def marks(block: dict, metric: str) -> dict:
    """{row key: 'best' | 'tie' | ''} for one metric within one block of rows.

    `block` maps a row key to its per-replication values of `metric`; for signed bias
    pass the signed values and the ranking reads their absolute value.
    """
    if METRIC[metric][1] is None:
        return {k: "" for k in block}
    vals = {k: clean(v) for k, v in block.items()}
    vals = {k: (np.abs(v) if metric == "bias_percent" else v) for k, v in vals.items() if len(v)}
    if not vals:
        return {k: "" for k in block}
    pick = max if METRIC[metric][1] else min
    best = pick(vals, key=lambda k: vals[k].mean())
    out = {k: "" for k in block}
    for k, v in vals.items():
        out[k] = "best" if k == best else ("" if differs(v, vals[best]) else "tie")
    return out


def cell(x, metric: str, mark: str = "") -> str:
    x = clean(x)
    if not len(x):
        return "n/a"
    d = METRIC[metric][0]
    sign = "+" if metric == "bias_percent" else ""
    m, lo, hi = level(x)
    f = lambda v: f"{v:{sign}.{d}f}".replace("-", "−")              # noqa: E731
    s = f(m) if len(x) == 1 else f"{f(m)} [{f(lo)}, {f(hi)}]"
    return {"best": f"**{s}**", "tie": f"<u>{s}</u>", "": s}[mark]


def render(rows: list[dict], metrics: list[str], lead: list[str]) -> list[str]:
    """Rows carry their `lead` cells, a `block` id and per-metric value arrays."""
    m = {}
    for blk in dict.fromkeys(r["block"] for r in rows):
        members = [i for i, r in enumerate(rows) if r["block"] == blk]
        for met in metrics:
            for i, mk in marks({i: rows[i][met] for i in members}, met).items():
                m[(i, met)] = mk
    return ["| " + " | ".join([str(r[c]) for c in lead]
                              + [cell(r[met], met, m[(i, met)]) for met in metrics]) + " |"
            for i, r in enumerate(rows)]


def pareto(panel: str, cal: str) -> pd.DataFrame:
    p = PARETO[(PARETO.panel == panel) & (PARETO.cal == cal)]
    assert len(p) == 20, (panel, cal, len(p))
    return p


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------

def census() -> list[str]:
    """§3: every stored electronics forecast; Spearman ranked, CV descriptive."""
    if CENSUS.exists():
        f = pd.read_csv(CENSUS)
    else:
        data = bm.build_data("electronics", "2y")
        f = mfc.collect(holdout_actuals(data), np.asarray(data["ids"]))
        f = f.drop_duplicates(subset=["model", "per_customer", "calendar", "digest"])
        CENSUS.parent.mkdir(exist_ok=True)
        f.to_csv(CENSUS, index=False)
    print(f"census: {len(f)} distinct forecasts over {f.suite.nunique()} suites")
    # The HB Pareto/NBD row is the benchmark: its 20 seeded fits plus one earlier fit.
    f["model"] = f.model.replace({"ParetoNBD": "Pareto/NBD (benchmark)"})
    g = f.groupby(["per_customer", "model", "calendar"], sort=False)
    rows = [{"block": 0, "input": k[0], "model": k[1], "calendar": k[2], "n": len(v),
             "median_cv": v.cv.median(), "spearman": v.spearman.to_numpy()} for k, v in g]
    rows.sort(key=lambda r: (r["median_cv"], r["model"]))
    for r in rows:
        r["cv"] = f"{r['median_cv']:.2f}"
    head = ["| per-customer input beside the count | model | calendar | forecasts | median CV "
            "| mean Spearman [95% CI] |", "| --- | --- | --- | ---: | ---: | ---: |"]
    return head + render(rows, ["spearman"], ["input", "model", "calendar", "n", "cv"])


def cdnow_transformer_no_ar() -> pd.DataFrame:
    """The 19-of-20 suite without a results.csv, scored against its own 38-week holdout."""
    suite = S / "real_panel_arms__Transformer__cdnow__no_ar-no_cluster-valendin__a"
    pc = PanelConfig.from_dict(json.loads((suite / "config.json").read_text())["panel_config"])
    data = panel_dataset.prepare_dataset(pd.read_csv(rca.PANELS["cdnow"][0]), pc, verbose=False)
    actual, ids = holdout_actuals(data), np.asarray(data["ids"])
    rows = []
    for path in sorted((suite / "Transformer" / "Predictions").glob("Prediction_*.csv")):
        study = int(path.stem.split("_")[-1])
        values, got = load_model_predictions(suite / "Transformer", study=study)
        assert np.array_equal(np.asarray(got), ids)
        rows.append({"study": study, **compute_forecast_metrics(actual, values)})
    return pd.DataFrame(rows).assign(spearman=np.nan)


def table_41(d: pd.DataFrame) -> list[str]:
    tn = cdnow_transformer_no_ar()
    print("CDNOW Transformer no_ar:", len(tn), "studies; study 14 bias",
          round(tn.set_index("study").bias_percent.get(14, np.nan), 1), "; other 18:",
          tn[tn.study != 14][["bias_percent", "mape_aggregate"]].agg(["mean", "std", "median"])
          .round(1).to_dict())
    spec = {"electronics": [
        ("2y", "ValendinLSTM", "ValendinLSTM (benchmark), count + week", "100 / 500"),
        ("2y", "Pareto/NBD", "Pareto/NBD (benchmark)", "—"),
        *[("2y", f"LSTM {a} (ar_encoding)", f"LSTM {a} (`ar_encoding`)", "50 / 300")
          for a in bre.AR_ENC_ARMS["electronics"]],
        *[("2y", f"{m} {a} (real_panel_arms)", f"{m} {a} (`real_panel_arms`)", "50 / 200")
          for m, a in bre.ARMS_ROWS["electronics"]]],
        "cdnow": [
        ("2y-arms", "Pareto/NBD, arms window (one fit)",
         "Pareto/NBD (benchmark), one fit (`real_panel_arms__ParetoNBD__cdnow`)", "—"),
        *[("2y-arms", f"LSTM {a} (ar_encoding)", f"LSTM {a} (`ar_encoding`)", "50 / 300")
          for a in bre.AR_ENC_ARMS["cdnow"]],
        ("2y-arms", "LSTM no_ar (real_panel_arms)", "LSTM no_ar (`real_panel_arms`)", "50 / 200"),
        ("2y-arms", "LSTM ar_bounded (real_panel_arms)", "LSTM ar_bounded (`real_panel_arms`)", "50 / 200"),
        ("2y-arms", "TN", "Transformer no_ar (`real_panel_arms`)", "50 / 200"),
        ("2y-arms", "Transformer ar_bounded (real_panel_arms)",
         "Transformer ar_bounded (`real_panel_arms`)", "50 / 200")]}
    rows = []
    for panel, items in spec.items():
        for cal, lab, name, budget in items:
            x = tn if lab == "TN" else d[(d.panel == panel) & (d.cal == cal) & (d.label == lab)]
            n = len(x)
            rows.append({"block": panel, "panel": panel, "name": name,
                         "n": (f"{n} fit" + "s" * (n > 1)) if "Pareto" in lab else n, "budget": budget,
                         **{k: x[k].to_numpy() for k in
                            ("bias_percent", "mape_aggregate", "rmse", "spearman")}})
    head = ["| panel | model and features | n | trials / paths | bias % | MAPE | RMSE | Spearman |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    return head + render(rows, ["bias_percent", "mape_aggregate", "rmse", "spearman"],
                         ["panel", "name", "n", "budget"])


ENC = {"bounded32": "LSTM + bounded32", "log": "LSTM + log", "ratio": "LSTM + ratio",
       "bounded32ratio": "LSTM + bounded32 + ratio"}


def block_rows(d, panel, cal, with_valendin=True) -> list[dict]:
    """One (panel, window) block: the four encodings, ValendinLSTM, Pareto/NBD."""
    items = [(f"LSTM + {e}", name) for e, name in ENC.items()]
    if with_valendin:
        items.append(("ValendinLSTM", "ValendinLSTM (benchmark)"))
    items.append(("Pareto/NBD", "Pareto/NBD (benchmark)"))
    rows = []
    for lab, name in items:
        x = d[(d.panel == panel) & (d.cal == cal) & (d.label == lab)]
        if not len(x):
            continue
        rows.append({"block": (panel, cal), "panel": panel, "cal": cal, "name": name,
                     "n": f"{len(x)} fits" if lab == "Pareto/NBD" else len(x),
                     **{k: x[k].to_numpy() for k in
                        ("bias_percent", "mape_aggregate", "rmse", "spearman")}})
    return rows


def table_43(d) -> list[str]:
    rows = [r for p in bm.WINDOWS for r in block_rows(d, p, "2y")]
    head = ["| panel | model | n | bias % | MAPE | RMSE | Spearman |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    return head + render(rows, ["bias_percent", "mape_aggregate", "rmse", "spearman"],
                         ["panel", "name", "n"])


def table_6(d) -> list[str]:
    rows = []
    for p in ("electronics", "gift", "multichannel"):
        for cal in ("2y", "3y"):
            rows += block_rows(d, p, cal)
    for r in rows:
        r["cal"] = {"2y": "2-year", "3y": "3-year"}[r["cal"]]
    head = ["| panel | calibration | model | n | bias % | MAPE | RMSE | Spearman |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    return head + render(rows, ["bias_percent", "mape_aggregate", "rmse", "spearman"],
                         ["panel", "cal", "name", "n"])


def table_44() -> list[str]:
    """Family G, `ar_unbounded`: LSTM and Transformer against Pareto/NBD, signed bias."""
    arch = S / "_archive_50sim"
    one_fit = pd.read_csv(S / "real_panel_arms__ParetoNBD__cdnow" / "results.csv").bias_percent
    rows = []
    for panel in ("cdnow", "electronics"):
        pn = one_fit if panel == "cdnow" else pareto("electronics", "2y").bias_percent
        for cluster in ("no_cluster", "kmeans_8"):
            for model in ("LSTM", "Transformer"):
                r = pd.read_csv(arch / f"real_panel_arms__{model}__{panel}__ar_unbounded-"
                                       f"{cluster}-valendin__a" / "results.csv")
                r = r[r.model == model]
                rows.append({"block": (panel, cluster), "panel": panel, "cluster": f"`{cluster}`",
                             "name": model, "n": len(r), "bias_percent": r.bias_percent.to_numpy()})
            rows.append({"block": (panel, cluster), "panel": panel, "cluster": f"`{cluster}`",
                         "name": "Pareto/NBD (benchmark)" + (", one fit" if panel == "cdnow" else ""),
                         "n": f"{len(pn)} fit" + ("s" if len(pn) > 1 else ""),
                         "bias_percent": pn.to_numpy()})
    head = ["| panel | cluster | model | n | bias % |", "| --- | --- | --- | ---: | ---: |"]
    return head + render(rows, ["bias_percent"], ["panel", "cluster", "name", "n"])


def family_f_spearman() -> pd.DataFrame:
    if SPEARMAN_F.exists():
        return pd.read_csv(SPEARMAN_F)
    panel_path, build = rca.PANELS["electronics"]
    raw = pd.read_csv(panel_path)
    out = []
    for arm in rca.ARMS:
        data = panel_dataset.prepare_dataset(raw, build(rca.ARMS[arm]), verbose=False)
        for shard, base in rca.SHARDS.items():
            for rec in rca.score_suite(S / rca.suite_name("electronics", arm, shard), data, base):
                out.append({"arm": arm, "shard": shard, **rec})
    out = pd.DataFrame(out)
    SPEARMAN_F.parent.mkdir(exist_ok=True)
    out.to_csv(SPEARMAN_F, index=False)
    return out


def table_51() -> list[str]:
    sp = family_f_spearman()
    one_fit = pd.read_csv(S / "real_panel_arms__ParetoNBD__cdnow" / "results.csv")
    rows = []
    for panel, title in (("electronics", "electronics"), ("cdnow", "CDNOW")):
        for arm in rca.ARMS:
            lv = pd.concat([pd.read_csv(S / rca.suite_name(panel, arm, sh) / "results.csv")
                            for sh in rca.SHARDS])
            s = sp[sp.arm == arm].spearman.to_numpy() if panel == "electronics" else []
            rows.append({"block": panel, "panel": title, "name": f"`{arm}`", "n": len(lv),
                         "abs_bias": lv.bias_percent.abs().to_numpy(),
                         "mape_aggregate": lv.mape_aggregate.to_numpy(), "spearman": s})
        pn = pareto("electronics", "2y") if panel == "electronics" else one_fit
        rows.append({"block": panel, "panel": title,
                     "name": "Pareto/NBD (benchmark)" + (", one fit" if panel == "cdnow" else ""),
                     "n": f"{len(pn)} fit" + ("s" if len(pn) > 1 else ""),
                     "abs_bias": pn.bias_percent.abs().to_numpy(),
                     "mape_aggregate": pn.mape_aggregate.to_numpy(),
                     "spearman": pn.spearman.to_numpy() if panel == "electronics" else []})
    head = ["| panel | arm | n | \\|bias\\| | MAPE | Spearman |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    lines = head + render(rows, ["abs_bias", "mape_aggregate", "spearman"], ["panel", "name", "n"])
    # Δ against `no_cluster`, every arm, per panel (independent, 40 / 40).
    dl = ["", "| panel | arm | Δ \\|bias\\| [95% CI] | Δ MAPE [95% CI] | Δ Spearman [95% CI] |",
          "| --- | --- | ---: | ---: | ---: |"]
    for panel in ("electronics", "CDNOW"):
        rs = [r for r in rows if r["panel"] == panel and "Pareto" not in r["name"]]
        base = rs[0]
        for r in rs[1:]:
            cells = []
            for met in ("abs_bias", "mape_aggregate", "spearman"):
                if not len(clean(r[met])):
                    cells.append("n/a")
                    continue
                e = effect(clean(r[met]), clean(base[met]), paired=False, metric=met, panel=panel)
                dd = METRIC[met][0]
                s = (f"{e.delta:+.{dd}f} [{e.lo:+.{dd}f}, {e.hi:+.{dd}f}]").replace("-", "−")
                cells.append(f"**{s}**" if e.supported else s)
            dl.append(f"| {panel} | {r['name']} | " + " | ".join(cells) + " |")
    return lines + dl


def table_52() -> list[str]:
    f = pd.read_csv(REPO / ".scratch/training-budget/results/factorial.csv")
    cells = [("archive", "no_cluster"), ("archive", "kmeans_8"),
             ("floored", "no_cluster"), ("floored", "kmeans_8")]
    out = ["| panel | model | archive / no_cluster | archive / kmeans_8 | floored / no_cluster "
           "| floored / kmeans_8 | Pareto/NBD |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for panel in bm.WINDOWS:
        block = {}
        for model in ("ValendinLSTM", "LSTM"):
            for c in cells:
                x = f[(f.panel == panel) & (f.model == model) & (f.training == c[0])
                      & (f.cluster == c[1])].spearman.to_numpy()
                assert len(x) == 20, (panel, model, c, len(x))
                block[(model, c)] = x
        block[("Pareto/NBD", None)] = pareto(panel, "2y").spearman.to_numpy()
        mk = marks(block, "spearman")
        for model in ("ValendinLSTM", "LSTM"):
            out.append(f"| {panel} | {model} | "
                       + " | ".join(cell(block[(model, c)], "spearman", mk[(model, c)])
                                    for c in cells) + " | |")
        out.append(f"| {panel} | Pareto/NBD (benchmark), 20 fits | | | | | "
                   + cell(block[("Pareto/NBD", None)], "spearman", mk[("Pareto/NBD", None)])
                   + " |")
    return out


def table_7() -> list[str]:
    per = pd.read_csv(REPO / ".scratch/feature-engineering-5y/results/per_forecast.csv")
    pn = per[per.model == "ParetoNBD"]
    assert len(pn) == 20
    mets = ["rmse_customer_total", "bias_percent", "mape_aggregate", "spearman", "cv"]
    out = []
    for model in ("LSTM", "LSTMAttention", "Transformer"):
        rows = []
        for arm in ("searched", "nofloor", "from20", "from30"):
            for feat in ("none", "ar_bounded_52", "kmeans_8"):
                x = per[(per.model == model) & (per.arm == arm) & (per.feature == feat)]
                if not len(x):
                    continue
                rows.append({"block": 0, "rule": f"`{arm}`",
                             "input": "none" if feat == "none" else f"`{feat}`",
                             **{k: x[k].to_numpy() for k in mets}})
        rows.append({"block": 0, "rule": "Pareto/NBD (benchmark), 20 fits", "input": "—",
                     **{k: pn[k].to_numpy() for k in mets}})
        out += [f"{model}:", "", "| rule | input | RMSE (customer total) | bias % | MAPE | "
                "Spearman | forecast CV |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        out += render(rows, mets, ["rule", "input"]) + [""]
    return out


MET4 = ("bias_percent", "mape_aggregate", "rmse", "spearman")


def candidates(d: pd.DataFrame) -> list[dict]:
    """Every configuration that forecast a block's customers over its holdout.

    A block is one panel and one calibration window. Each candidate carries its
    architecture, a configuration label and its per-replication scores.
    """
    out = []

    def add(block, arch, config, x):
        out.append({"block": block, "arch": arch, "config": config, "n": len(x),
                    **{k: (x[k].to_numpy() if k in x else np.array([])) for k in MET4}})

    # Benchmark windows, 2 years (CDNOW: 39 weeks): families N, O, U.
    fac = pd.read_csv(REPO / ".scratch/training-budget/results/factorial.csv")
    for p in bm.WINDOWS:
        blk = (p, "2y")
        add(blk, "ValendinLSTM", "paper recipe (family N)",
            d[(d.panel == p) & (d.cal == "2y") & (d.label == "ValendinLSTM")])
        for e, name in ENC.items():
            add(blk, "LSTM", f"`{e}` (family O)",
                d[(d.panel == p) & (d.cal == "2y") & (d.label == f"LSTM + {e}")])
        for (m, tr, cl), x in fac[fac.panel == p].groupby(["model", "training", "cluster"]):
            add(blk, m, f"`{tr} / {cl}` (family U)", x)
    # Electronics 2 years: families E, H and F share the benchmark's windows.
    blk = ("electronics", "2y")
    for lab in d[(d.panel == "electronics") & d.group.isin(["E", "H"])].label.unique():
        arch, arm, fam = lab.split()[0], lab.split()[1], "E" if "ar_encoding" in lab else "H"
        add(blk, arch, f"`{arm}` (family {fam})", d[(d.panel == "electronics") & (d.label == lab)])
    sp = family_f_spearman()
    for arm in rca.ARMS:
        lv = pd.concat([pd.read_csv(S / rca.suite_name("electronics", arm, sh) / "results.csv")
                        for sh in rca.SHARDS])
        x = lv[["bias_percent", "mape_aggregate", "rmse"]].reset_index(drop=True)
        x["spearman"] = np.nan
        sx = sp[sp.arm == arm].spearman.to_numpy()
        x = x.reindex(range(max(len(x), len(sx))))
        x.loc[:len(sx) - 1, "spearman"] = sx
        add(blk, "LSTM", f"`{arm}` (family F)", x)
    # CDNOW's pre-ADR-0009 38-week window: families E, H, F and one Pareto/NBD fit.
    blk = ("cdnow", "38w")
    for lab in d[(d.panel == "cdnow") & (d.cal == "2y-arms")].label.unique():
        if "Pareto" in lab:
            continue
        arch, arm, fam = lab.split()[0], lab.split()[1], "E" if "ar_encoding" in lab else "H"
        add(blk, arch, f"`{arm}` (family {fam})", d[(d.panel == "cdnow") & (d.label == lab)])
    add(blk, "Transformer", "`no_ar` (family H)", cdnow_transformer_no_ar())
    for arm in rca.ARMS:
        lv = pd.concat([pd.read_csv(S / rca.suite_name("cdnow", arm, sh) / "results.csv")
                        for sh in rca.SHARDS])
        add(blk, "LSTM", f"`{arm}` (family F)", lv)
    one = pd.read_csv(S / "real_panel_arms__ParetoNBD__cdnow" / "results.csv")
    add(blk, "Pareto/NBD", "one fit", one)
    # Three years: family P.
    for p in ("electronics", "gift", "multichannel"):
        for e in ENC:
            add((p, "3y"), "LSTM", f"`{e}` (family P)",
                d[(d.panel == p) & (d.cal == "3y") & (d.label == f"LSTM + {e}")])
    # electronic_5y: family Y.
    per = pd.read_csv(REPO / ".scratch/feature-engineering-5y/results/per_forecast.csv")
    for (m, arm, feat), x in per[per.model != "ParetoNBD"].groupby(["model", "arm", "feature"]):
        add(("electronic_5y", "5y"), m, f"`{arm}`, input {feat}", x)
    # Pareto/NBD, 20 seeded fits, on every block that has them.
    for p, cal in [(p, "2y") for p in bm.WINDOWS] + [(p, "3y") for p in
                                                     ("electronics", "gift", "multichannel")]:
        add((p, cal), "Pareto/NBD", "20 seeded fits", pareto(p, cal))
    add(("electronic_5y", "5y"), "Pareto/NBD", "20 seeded fits", per[per.model == "ParetoNBD"])
    return out


BLOCKS = [("cdnow", "2y", "CDNOW", "39 weeks (2-year windows)"),
          ("cdnow", "38w", "CDNOW", "38 weeks (pre-ADR-0009)"),
          ("electronics", "2y", "electronics", "2 years"),
          ("electronics", "3y", "electronics", "3 years"),
          ("gift", "2y", "gift", "2 years"), ("gift", "3y", "gift", "3 years"),
          ("multichannel", "2y", "multichannel", "2 years"),
          ("multichannel", "3y", "multichannel", "3 years"),
          ("electronic_5y", "5y", "electronic_5y", "5 years")]
ARCH = ("LSTM", "ValendinLSTM", "LSTMAttention", "Transformer")


def table_best(d) -> list[str]:
    """§12: per block, each architecture's best configuration by MAPE and by Spearman."""
    cand = candidates(d)
    rows = []
    for panel, cal, title, window in BLOCKS:
        blk = [c for c in cand if c["block"] == (panel, cal)]
        for arch in ARCH:
            own = [c for c in blk if c["arch"] == arch]
            if not own:
                continue
            chosen = {}
            for met, pick in (("mape_aggregate", min), ("spearman", max)):
                scored = [c for c in own if len(clean(c[met]))]
                if scored:
                    c = pick(scored, key=lambda c: clean(c[met]).mean())
                    chosen.setdefault(id(c), [c, []])[1].append(
                        {"mape_aggregate": "MAPE", "spearman": "Spearman"}[met])
            for c, why in chosen.values():
                rows.append({**c, "block": (panel, cal), "panel": title, "window": window,
                             "why": " and ".join(why), "name": f"{arch}: {c['config']}"})
        for c in blk:
            if c["arch"] == "Pareto/NBD":
                rows.append({**c, "block": (panel, cal), "panel": title, "window": window,
                             "why": "benchmark",
                             "n": f"{c['n']} fit" + "s" * (c["n"] > 1),
                             "name": f"Pareto/NBD (benchmark), {c['config']}"})
    head = ["| panel | calibration | model: best configuration | chosen on | n | bias % | MAPE "
            "| RMSE | Spearman |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    return head + render(rows, list(MET4), ["panel", "window", "name", "why", "n"])


def main() -> None:
    d = bre.load_all()
    parts = {"§3": census(), "§4.1": table_41(d), "§4.3": table_43(d), "§4.4": table_44(),
             "§5.1": table_51(), "§5.2": table_52(), "§6": table_6(d), "§7": table_7(),
             "§12": table_best(d)}
    text = "\n\n".join(f"## {k}\n\n" + "\n".join(v) for k, v in parts.items()) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
