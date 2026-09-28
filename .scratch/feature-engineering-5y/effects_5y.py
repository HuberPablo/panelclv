"""Every claim `docs/feature_engineering.md` makes about inputs on electronic_5y.

Scores each stored forecast of the 5y runs once, through the runners' own `score`
(the single scoring authority plus per-customer Spearman), adds forecast CV, then
applies the standard of `docs/training-budget.md` "How claims are made" via
`.scratch/training-budget/effects.py`: a cell is a mean with a 95% bootstrap CI over its
20 studies; an effect is delta = mean(B) - mean(A) with a 95% bootstrap CI, supported
when the interval excludes zero. Replications are treated as independent samples
(training is unseeded), so each condition is resampled separately.

    python .scratch/feature-engineering-5y/effects_5y.py          # writes results/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / ".scratch" / "training-budget"))
import run_epoch_floor_5y as floor  # noqa: E402
import run_real_panel_benchmarks as benchmarks  # noqa: E402
from effects import effect  # noqa: E402
from panelclv.data_preparation.target_channel import holdout_actuals  # noqa: E402
from panelclv.studies.suite_reader import load_model_predictions  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
S = ROOT / "Studies"
ARMS = ("nofloor", "from20", "from30")
FEATURES = ("none", "ar_bounded_52", "kmeans_8")
MODELS = ("LSTM", "LSTMAttention", "Transformer")


def suite(model: str, feature: str, arm: str, r: int) -> tuple[Path, str]:
    """Directory of one forecast's model folder, and the model folder's name."""
    if model == "LSTM":
        if feature == "none" and arm == "searched":
            return S / f"real_panel_benchmarks_cal5y__ValendinLSTM__electronics__r{r:02d}", "ValendinLSTM"
        if feature == "none":
            return S / f"epoch_floor_cal5y__ValendinLSTM__electronics__{arm}__r{r:02d}", "ValendinLSTM"
        return S / f"epoch_floor_cal5y__LSTM__electronics__{feature}-{arm}__r{r:02d}", "LSTM"
    return S / f"attention_cal5y__{model}__electronics__{feature}-{arm}__r{r:02d}", model


def cv_of_totals(model_dir: Path) -> float:
    """Forecast CV: std / mean of per-customer predicted holdout totals."""
    values, _ = load_model_predictions(model_dir, study=1)
    tot = values.sum(axis=1)
    return float(tot.std() / tot.mean())


def score_all() -> pd.DataFrame:
    base = floor.build_data()
    actual = holdout_actuals(base)
    ref_ids = np.asarray(base["ids"])
    rows = []
    for model in MODELS:
        cells = [("none", "searched")] + [(f, a) for a in ARMS for f in FEATURES]
        for feature, arm in cells:
            for r in range(20):
                d, name = suite(model, feature, arm, r)
                md = d / name
                if not (md / "Predictions").is_dir():
                    print("missing", md)
                    continue
                rows.append({"model": model, "feature": feature, "arm": arm, "rep": r,
                             **benchmarks.score(md, actual, ref_ids), "cv": cv_of_totals(md)})
    pn = S / "real_panel_benchmarks_cal5y__ParetoNBD__electronics" / "ParetoNBD"
    rows.append({"model": "ParetoNBD", "feature": "-", "arm": "-", "rep": 0,
                 **benchmarks.score(pn, actual, ref_ids), "cv": cv_of_totals(pn)})
    df = pd.DataFrame(rows)
    df["abs_bias"] = df["bias_percent"].abs()
    return df


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    res = bootstrap((x,), np.mean, n_resamples=10000, method="percentile", random_state=0)
    return x.mean(), res.confidence_interval.low, res.confidence_interval.high


METRICS = ("rmse_customer_total", "bias_percent", "mape_aggregate", "spearman", "cv")


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, f, a), g in df[df.model != "ParetoNBD"].groupby(["model", "feature", "arm"], sort=False):
        row = {"model": m, "feature": f, "arm": a, "n": len(g)}
        for k in METRICS:
            mu, lo, hi = mean_ci(g[k])
            row.update({k: mu, f"{k}_lo": lo, f"{k}_hi": hi, f"{k}_sd": g[k].std(ddof=1)})
        out.append(row)
    return pd.DataFrame(out)


def effects(df: pd.DataFrame) -> pd.DataFrame:
    """Each input against no input, within one model and one epoch rule; and models
    against the LSTM within one input and rule."""
    out = []
    metrics = ("spearman", "mape_aggregate", "abs_bias", "rmse_customer_total", "cv")

    def add(kind, a_mask, b_mask, label_a, label_b, model, arm):
        for k in metrics:
            e = effect(df.loc[b_mask, k], df.loc[a_mask, k], metric=k, panel="electronic_5y")
            out.append({"kind": kind, "model": model, "arm": arm, "A": label_a, "B": label_b,
                        "metric": k, "mean_a": e.mean_a, "mean_b": e.mean_b,
                        "delta": e.delta, "lo": e.lo, "hi": e.hi, "supported": e.supported})

    for m in MODELS:
        for a in ARMS:
            base = (df.model == m) & (df.feature == "none") & (df.arm == a)
            for f in FEATURES[1:]:
                add("input", base, (df.model == m) & (df.feature == f) & (df.arm == a),
                    "none", f, m, a)
            add("input", (df.model == m) & (df.feature == "ar_bounded_52") & (df.arm == a),
                (df.model == m) & (df.feature == "kmeans_8") & (df.arm == a),
                "ar_bounded_52", "kmeans_8", m, a)
    for f in FEATURES:
        for a in ARMS:
            lstm = (df.model == "LSTM") & (df.feature == f) & (df.arm == a)
            for m in MODELS[1:]:
                add("model", lstm, (df.model == m) & (df.feature == f) & (df.arm == a),
                    "LSTM", m, f, a)
    return pd.DataFrame(out)





def input_support() -> pd.DataFrame:
    """Input side: escape fraction and z-distance of the raw clocks and the flags on
    electronic_5y (true counts in both windows, as `scripts/measure_ar_support.py`),
    and the share of already-active cells whose silence reaches the deepest bin."""
    from panelclv.data_preparation.ar_features import compute_ar_feature_columns
    base = floor.build_data()
    t = int(base["target_idx"])
    cal = np.asarray(base["calibration"])[:, :, t].astype(np.int64)
    counts = np.concatenate([cal, holdout_actuals(base).astype(np.int64)], axis=1)
    t_cal = cal.shape[1]
    names = ("period_since_last_transaction", "period_since_first_transaction",
             "cumulative_transactions", "transaction_rate", "has_transacted_before",
             "active_in_last_32_periods", "active_in_last_52_periods")
    cols = compute_ar_feature_columns(counts, names)
    rows = []
    for n in names:
        c, h = cols[n][:, :t_cal], cols[n][:, t_cal:]
        lo, hi, sd = c.min(), c.max(), c.std() or 1.0
        exc = np.maximum(h - hi, lo - h)
        rows.append({"feature": n, "escape_pct": 100 * (exc > 0).mean(),
                     "z_worst": (h.max() - hi) / sd})
    ever = cols["has_transacted_before"] == 1
    gap = cols["period_since_last_transaction"]
    for k in (32, 52):
        for w, sl in (("calibration", slice(0, t_cal)), ("holdout", slice(t_cal, None))):
            e, g = ever[:, sl], gap[:, sl]
            rows.append({"feature": f"silence>={k} among active, {w}",
                         "escape_pct": 100 * (g[e] >= k).mean(), "z_worst": np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    per = OUT / "per_forecast.csv"
    df = pd.read_csv(per) if per.exists() else score_all()
    df.to_csv(per, index=False)
    cell_table(df).to_csv(OUT / "cells.csv", index=False)
    effects(df).to_csv(OUT / "effects.csv", index=False)
    input_support().to_csv(OUT / "input_support.csv", index=False)
    print(df[df.model == "ParetoNBD"].T)
