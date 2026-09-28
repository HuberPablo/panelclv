"""Every comparative claim of `docs/benchmarks-real-panels.md`, under the statistical protocol.

`docs/statistical-protocol.md`: Δ = mean(B) − mean(A), a 95% percentile-bootstrap interval
from 10,000 resamples, supported iff the interval excludes 0, computed only by
`panelclv.evaluation.effects.effect`. Nothing here re-implements a bootstrap. The doc
quotes exactly what this prints.

Pairing (protocol §2), one decision per data source:

* ValendinLSTM (family N), the developed LSTM (families E, H, O, P) and the Transformer
  (family H): every replication is its own Optuna search with unseeded training, so two
  conditions never share a unit — paired=False, unequal n allowed (100 vs 20, 80, 40).
* Pareto/NBD on the benchmark windows: 20 seeded MCMC fits per panel and calibration
  (`benchmarks_pareto_scores.py`) — its own condition, paired=False against anything.
* Pareto/NBD on CDNOW's pre-ADR-0009 window (`real_panel_arms__ParetoNBD__cdnow`, the
  38-week holdout from 1997-10-01 that `ar_encoding` and `real_panel_arms` forecast): one
  fit, never replicated. It is a fixed reference, so a CDNOW arm against it is the paired
  case against a constant, and its own fit-to-fit spread is NOT in the interval.
  Electronics' `real_panel_arms__ParetoNBD__electronics` is on the benchmark's 2-year
  windows (checked against config.json), so electronics uses the 20 seeded fits.

Per-forecast scores are cached in `results/benchmarks_per_forecast.csv` (delete it to
rescore from `Studies/`). Scoring is the runners' own: `compute_forecast_metrics` plus
per-customer Spearman of holdout totals.

    PYTHONPATH=src:scripts python .scratch/statistical-protocol/benchmarks_real_panels_effects.py
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
sys.path.insert(0, str(HERE))

import run_real_panel_ar as ar                                            # noqa: E402
import run_real_panel_benchmarks as bm                                    # noqa: E402
from benchmarks_pareto_scores import OUT as PARETO_SCORES                 # noqa: E402
from panelclv.data_preparation.target_channel import holdout_actuals     # noqa: E402
from panelclv.evaluation.effects import effect, table                     # noqa: E402
from panelclv.models import compute_forecast_metrics                      # noqa: E402
from panelclv.studies import load_model_predictions                       # noqa: E402

S = REPO / "Studies"
CACHE = HERE / "results" / "benchmarks_per_forecast.csv"
FE5Y = REPO / ".scratch" / "feature-engineering-5y" / "results" / "per_forecast.csv"
ENCODINGS = ("bounded32", "log", "ratio", "bounded32ratio")
AR_ENC_ARMS = {  # ar_encoding (family E) arms per panel, as the doc's table lists them
    "electronics": ("no_ar", "ar_bounded_32", "ar_bounded_52", "ar_saturating", "ar_log",
                    "ar_ratio", "ar_unbounded"),
    "cdnow": ("no_ar", "ar_bounded_16", "ar_bounded_32", "ar_saturating", "ar_log",
              "ar_ratio", "ar_unbounded"),
}
ARMS_ROWS = {  # real_panel_arms (family H) rows the doc's table lists: (model, arm)
    "electronics": (("LSTM", "no_ar"), ("LSTM", "ar_bounded"), ("Transformer", "no_ar"),
                    ("Transformer", "ar_bounded")),
    "cdnow": (("LSTM", "no_ar"), ("LSTM", "ar_bounded"), ("Transformer", "ar_bounded")),
}


# ---------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return bm.spearman(x, y)


def score_dir(model_dir: Path, actual: np.ndarray, ids: np.ndarray) -> list[dict]:
    """Every stored forecast in one model folder, one row per study."""
    rows = []
    for path in sorted((model_dir / "Predictions").glob("Prediction_*.csv")):
        study = int(path.stem.split("_")[-1])
        values, got = load_model_predictions(model_dir, study=study)
        if got is not None and not np.array_equal(np.asarray(got), ids):
            raise ValueError(f"{model_dir} study {study}: cohort mismatch")
        rows.append({"study": study, **compute_forecast_metrics(actual, values),
                     "spearman": spearman(values.sum(axis=1), actual.sum(axis=1))})
    return rows


def results_rows(suite: Path, model: str) -> list[dict]:
    """bias / MAPE / RMSE from a suite's own results.csv (CDNOW's pre-ADR-0009 suites,
    whose 38-week holdout today's week rule cannot rebuild, so Spearman is absent)."""
    r = pd.read_csv(suite / "results.csv")
    r = r[r.model == model]
    return [{"study": int(s.study), "bias_percent": s.bias_percent,
             "mape_aggregate": s.mape_aggregate, "rmse": s.rmse,
             "spearman": np.nan} for s in r.itertuples()]


def score_everything() -> pd.DataFrame:
    rows = []

    def add(group, panel, cal, label, recs, source):
        for rec in recs:
            rows.append({"group": group, "panel": panel, "cal": cal, "label": label,
                         "source": source, **rec})

    for cal in ("2y", "3y"):
        for panel in bm.CALIBRATIONS[cal]:
            data = bm.build_data(panel, cal)
            actual, ids = holdout_actuals(data), np.asarray(data["ids"])
            print(f"scoring {cal}/{panel}", flush=True)
            for r in range(bm.N_REPLICATIONS):
                p = bm.forecast_path(panel, r, cal)
                if p.exists():
                    add("N", panel, cal, "ValendinLSTM", score_dir(p.parents[1], actual, ids),
                        p.parents[2].name)
            for enc in ENCODINGS:
                for r in range(ar.REPLICATIONS["lstm"]):
                    p = ar.forecast_path("lstm", enc, panel, r, cal)
                    if p.exists():
                        add("O" if cal == "2y" else "P", panel, cal, f"LSTM + {enc}",
                            score_dir(p.parents[1], actual, ids), p.parents[2].name)
            if cal == "2y" and panel == "electronics":
                # Families E and H on electronics share the benchmark's windows and cohort.
                for arm in AR_ENC_ARMS["electronics"]:
                    for suite in sorted(S.glob(f"ar_encoding__electronics__{arm}__?")):
                        add("E", panel, cal, f"LSTM {arm} (ar_encoding)",
                            score_dir(suite / "LSTM", actual, ids), suite.name)
                for model, arm in ARMS_ROWS["electronics"]:
                    suite = S / f"real_panel_arms__{model}__electronics__{arm}-no_cluster-valendin__a"
                    add("H", panel, cal, f"{model} {arm} (real_panel_arms)",
                        score_dir(suite / model, actual, ids), suite.name)
    for arm in AR_ENC_ARMS["cdnow"]:
        for suite in sorted(S.glob(f"ar_encoding__cdnow__{arm}__?")):
            if (suite / "results.csv").exists():
                add("E", "cdnow", "2y-arms", f"LSTM {arm} (ar_encoding)",
                    results_rows(suite, "LSTM"), suite.name)
    for model, arm in ARMS_ROWS["cdnow"]:
        suite = S / f"real_panel_arms__{model}__cdnow__{arm}-no_cluster-valendin__a"
        add("H", "cdnow", "2y-arms", f"{model} {arm} (real_panel_arms)",
            results_rows(suite, model), suite.name)
    add("H", "cdnow", "2y-arms", "Pareto/NBD, arms window (one fit)",
        results_rows(S / "real_panel_arms__ParetoNBD__cdnow", "ParetoNBD"),
        "real_panel_arms__ParetoNBD__cdnow")
    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    if CACHE.exists():
        d = pd.read_csv(CACHE)
    else:
        d = score_everything()
        CACHE.parent.mkdir(exist_ok=True)
        d.to_csv(CACHE, index=False)
    # Families E and H on electronics: the doc's table reads bias / MAPE / RMSE from each
    # suite's own results.csv, and Spearman from the stored predictions. The two agree
    # everywhere but the Transformer no_ar suite, whose stored predictions score MAPE 51.3
    # and bias +10.2 against results.csv's 53.0 and +15.9; the effects follow the table.
    eh = (d.panel == "electronics") & d.group.isin(["E", "H"])
    for suite in d[eh].source.unique():
        model = d[eh & (d.source == suite)].label.iloc[0].split()[0]
        r = pd.read_csv(S / suite / "results.csv")
        r = r[r.model == model].set_index("study")
        m = eh & (d.source == suite)
        for col in ("bias_percent", "mape_aggregate", "rmse"):
            d.loc[m, col] = d.loc[m, "study"].map(r[col]).to_numpy()
    # The seeded Pareto/NBD fits come from their own scorer; 3y joins only when complete.
    pn = pd.read_csv(PARETO_SCORES)
    pn = pn[pn.cal.isin(["2y", "3y"])].assign(group="PNBD", label="Pareto/NBD",
                                              source="real_panel_benchmarks")
    d = pd.concat([d[d.group != "PNBD"], pn], ignore_index=True)
    d["abs_bias"] = d.bias_percent.abs()
    return d


# ---------------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------------

def show(title: str, effects: list, label: str = "comparison") -> None:
    print(f"\n### {title}\n")
    print(table(effects, label=label))


def eff(d, panel, cal, b, a, metric, *, name=None):
    """Δ = mean(b) − mean(a) for two labelled conditions of one panel and calibration."""
    xb = d[(d.panel == panel) & (d.cal == cal) & (d.label == b)][metric]
    xa = d[(d.panel == panel) & (d.cal == cal) & (d.label == a)][metric]
    e = effect(xb, xa, paired=False, metric=metric, panel=panel)
    e.panel = name or f"{panel}: {b} − {a} [{metric}]"
    return e


def main() -> None:
    d = load_all()
    has3y = (d[(d.label == "Pareto/NBD") & (d.cal == "3y")].groupby("panel").size() == 20)
    has3y = len(has3y) == 3 and bool(has3y.all())

    # 1. ValendinLSTM against Pareto/NBD, per panel (Summary and "Reading").
    for metric in ("spearman", "mape_aggregate", "bias_percent", "abs_bias"):
        show(f"ValendinLSTM − Pareto/NBD, 2y, {metric} (n = 20 / 20)",
             [eff(d, p, "2y", "ValendinLSTM", "Pareto/NBD", metric, name=p)
              for p in bm.WINDOWS], label="panel")

    # 1b. Direction of error: each benchmark's signed bias against 0 (protocol §4 — an
    #     over- or under-forecasting claim needs its own interval to exclude 0).
    rows = []
    for p in bm.WINDOWS:
        for lab in ("ValendinLSTM", "Pareto/NBD"):
            x = d[(d.panel == p) & (d.cal == "2y") & (d.label == lab)].bias_percent.to_numpy()
            rows.append(effect(x, np.zeros_like(x), paired=True, metric="bias_percent",
                               panel=f"{p} / {lab}"))
    show("2y, signed bias against 0", rows, label="panel / model")

    # 2. Families E and H on electronics against both benchmarks and their own no_ar arm.
    el = d[(d.panel == "electronics") & d.group.isin(["E", "H"])]
    labels = list(dict.fromkeys(el.label))
    for metric in ("mape_aggregate", "spearman", "abs_bias"):
        for ref in ("ValendinLSTM", "Pareto/NBD"):
            show(f"electronics, arms − {ref}, {metric}",
                 [eff(d, "electronics", "2y", lab, ref, metric, name=lab) for lab in labels],
                 label="arm")
    for metric in ("spearman", "mape_aggregate"):
        rows = [eff(d, "electronics", "2y", lab, "LSTM no_ar (ar_encoding)", metric, name=lab)
                for lab in labels if "(ar_encoding)" in lab and "no_ar" not in lab]
        rows += [eff(d, "electronics", "2y", f"{m} ar_bounded (real_panel_arms)",
                     f"{m} no_ar (real_panel_arms)", metric, name=f"{m} ar_bounded (real_panel_arms)")
                 for m in ("LSTM", "Transformer")]
        show(f"electronics, arm − its own family's no_ar, {metric}", rows, label="arm")
    # "The Transformer collapses less completely than the LSTM": count-only, same family.
    show("electronics, Transformer no_ar − LSTM no_ar (real_panel_arms)",
         [eff(d, "electronics", "2y", "Transformer no_ar (real_panel_arms)",
              "LSTM no_ar (real_panel_arms)", m, name=m) for m in ("spearman", "mape_aggregate")],
         label="metric")

    # 3. CDNOW's pre-ADR-0009 arms against the one Pareto/NBD fit on their window
    #    (paired against a constant) and against their own no_ar arm.
    cd = d[(d.panel == "cdnow") & (d.cal == "2y-arms")]
    ref = cd[cd.label.str.startswith("Pareto/NBD")].iloc[0]
    print(f"\ncdnow arms-window Pareto/NBD (one fit): bias {ref.bias_percent:+.2f}, "
          f"MAPE {ref.mape_aggregate:.2f}, RMSE {ref.rmse:.4f}")
    for metric in ("mape_aggregate", "abs_bias"):
        rows = []
        for lab in dict.fromkeys(cd.label):
            if lab.startswith("Pareto/NBD"):
                continue
            x = cd[cd.label == lab][metric].to_numpy()
            e = effect(x, np.full_like(x, ref[metric]), paired=True, metric=metric, panel=lab)
            rows.append(e)
        show(f"cdnow (arms window), arm − one Pareto/NBD fit ({ref[metric]:.2f}), {metric}",
             rows, label="arm")
    for metric in ("mape_aggregate", "bias_percent"):
        rows = [eff(d, "cdnow", "2y-arms", lab, "LSTM no_ar (ar_encoding)", metric, name=lab)
                for lab in dict.fromkeys(cd.label)
                if "(ar_encoding)" in lab and "no_ar" not in lab]
        rows.append(eff(d, "cdnow", "2y-arms", "LSTM ar_bounded (real_panel_arms)",
                        "LSTM no_ar (real_panel_arms)", metric,
                        name="LSTM ar_bounded (real_panel_arms)"))
        show(f"cdnow (arms window), arm − its own family's no_ar, {metric}", rows, label="arm")

    # 4. Family O: every encoding against both benchmarks, and the encodings against
    #    each other, on each panel.
    for metric in ("spearman", "mape_aggregate", "abs_bias"):
        for refl in ("ValendinLSTM", "Pareto/NBD"):
            rows = [eff(d, p, "2y", f"LSTM + {enc}", refl, metric, name=f"{p} / {enc}")
                    for p in bm.WINDOWS for enc in ENCODINGS]
            show(f"2y, encoding − {refl}, {metric} (n = 20 / 100)", rows, label="panel / encoding")
    pairs = (("log", "bounded32"), ("ratio", "bounded32"), ("bounded32ratio", "ratio"),
             ("bounded32ratio", "bounded32"))
    for metric in ("spearman", "mape_aggregate", "bias_percent"):
        rows = [eff(d, p, "2y", f"LSTM + {b}", f"LSTM + {a}", metric, name=f"{p}: {b} − {a}")
                for p in bm.WINDOWS for b, a in pairs]
        show(f"2y, encoding against encoding, {metric} (n = 100 / 100)", rows, label="panel: B − A")

    # 5. Three-year calibration: against its own Pareto/NBD, and each side against 2y.
    for metric in ("spearman", "mape_aggregate", "bias_percent"):
        rows = []
        for p in bm.WINDOWS_3Y:
            for lab in [f"LSTM + {e}" for e in ENCODINGS] + ["Pareto/NBD"]:
                xb = d[(d.panel == p) & (d.cal == "3y") & (d.label == lab)][metric]
                xa = d[(d.panel == p) & (d.cal == "2y") & (d.label == lab)][metric]
                if len(xb) and len(xa):
                    e = effect(xb, xa, paired=False, metric=metric, panel=f"{p} / {lab}")
                    rows.append(e)
        show(f"3y − 2y, same model, {metric}", rows, label="panel / model")
    if has3y:
        for metric in ("spearman", "mape_aggregate", "bias_percent", "abs_bias"):
            rows = [eff(d, p, "3y", f"LSTM + {enc}", "Pareto/NBD", metric, name=f"{p} / {enc}")
                    for p in bm.WINDOWS_3Y for enc in ENCODINGS]
            show(f"3y, encoding − Pareto/NBD (3y), {metric} (n = 20 / 100)", rows,
                 label="panel / encoding")
        # The best LSTM arm's gap to Pareto/NBD at each calibration, for the Spearman
        # "did three years close the gap" reading (each gap is its own interval).
        for cal in ("2y", "3y"):
            rows = []
            for p in bm.WINDOWS_3Y:
                g = d[(d.panel == p) & (d.cal == cal) & d.label.str.startswith("LSTM + ")]
                best = g.groupby("label").spearman.mean().idxmax()
                rows.append(eff(d, p, cal, best, "Pareto/NBD", "spearman",
                                name=f"{p} / {best} ({cal})"))
            show(f"{cal}: best-Spearman LSTM arm − Pareto/NBD, spearman", rows,
                 label="panel / arm")
    else:
        print("\n(3y Pareto/NBD fits incomplete: 3y-against-Pareto/NBD comparisons skipped)")

    # 6. electronic_5y: the searched benchmark and the epoch rules, from the
    #    feature-engineering scores (which also hold the 20 seeded Pareto/NBD fits).
    if FE5Y.exists():
        f = pd.read_csv(FE5Y)
        f["abs_bias"] = f.bias_percent.abs()

        def c5(model, feature, arm):
            return f[(f.model == model) & (f.feature == feature) & (f.arm == arm)]
        pn5 = f[f.model == "ParetoNBD"]
        if len(pn5) == 20:
            print("\n### electronic_5y Pareto/NBD, 20 fits: mean [95% CI of the mean], sd, range\n")
            for m in ("rmse_customer_total", "bias_percent", "mape_aggregate", "spearman", "cv"):
                x = pn5[m].to_numpy()
                e = effect(x, np.zeros_like(x), paired=True, metric=m, panel="electronic_5y")
                print(f"{m}: {x.mean():.4f} [{e.lo:.4f}, {e.hi:.4f}], sd {x.std(ddof=1):.4f}, "
                      f"{x.min():.4f} … {x.max():.4f}")
        contrasts = [("from20 − searched", ("none", "from20"), ("none", "searched")),
                     ("from30 − searched", ("none", "from30"), ("none", "searched")),
                     ("from30 − from20", ("none", "from30"), ("none", "from20")),
                     ("nofloor − searched", ("none", "nofloor"), ("none", "searched")),
                     ("from20 − nofloor", ("none", "from20"), ("none", "nofloor"))]
        for model in ("LSTM", "LSTMAttention", "Transformer"):
            rows = []
            for metric in ("mape_aggregate", "abs_bias", "bias_percent", "spearman"):
                for name, b, a in contrasts:
                    e = effect(c5(model, *b)[metric], c5(model, *a)[metric], paired=False,
                               metric=metric, panel=f"{name} [{metric}]")
                    rows.append(e)
            show(f"electronic_5y, {model}, epoch rules, input none (n = 20 / 20)", rows,
                 label="contrast")


if __name__ == "__main__":
    main()
