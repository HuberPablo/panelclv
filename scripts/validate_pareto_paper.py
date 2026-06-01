"""Validate Models/pareto_paper.py (pure-Python HB Pareto/NBD) against the real
R BTYDplus package it ports.

Generates one synthetic Pareto/NBD cohort, runs BOTH estimators on the identical
calibration event log, and compares per-customer expected holdout transactions.
Because both are stochastic (MCMC + a sampled forecast), they agree only up to
Monte Carlo noise — we check the aggregate forecast and the per-customer
correlation, not bit-equality.

Run:
    .../thesis_rocm/bin/python scripts/validate_pareto_paper.py
(needs R on PATH with BTYDplus installed.)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# src-layout: the package lives under <repo>/src, so add that to the path as a
# fallback for running this script without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panelclv.benchmarks.pareto_paper import compute_pareto_paper_predictions  # noqa: E402

PERIOD_DAYS = 7.0
T_CAL_W = 78          # 1.5y calibration
H = 26               # 26w holdout
N_CUST = 600
SEED = 7


def make_synthetic_elog(n, t_cal_w, h, seed):
    """Simulate a Pareto/NBD cohort, return (event_log_df, weekly_panel_df).

    Each customer: lambda~Gamma(r,alpha), mu~Gamma(s,beta), lifetime~Exp(mu),
    purchases ~ Poisson process(lambda) until min(lifetime, end). Acquisition
    (first purchase) uniform over the first quarter of calibration.
    """
    rng = np.random.default_rng(seed)
    r, alpha, s, beta = 0.55, 10.0, 0.6, 12.0     # weekly-scale heterogeneity
    lam = rng.gamma(r, 1.0 / alpha, size=n)
    mu = rng.gamma(s, 1.0 / beta, size=n)
    life = rng.exponential(1.0 / mu)              # weeks alive after first purchase
    first = rng.uniform(0, t_cal_w * 0.25, size=n)

    total_w = t_cal_w + h
    rows = []
    for i in range(n):
        t = first[i]
        rows.append((i, t))                        # cohort-entry purchase
        end = min(first[i] + life[i], total_w)
        while True:
            t += rng.exponential(1.0 / lam[i])
            if t > end:
                break
            rows.append((i, t))
    raw = pd.DataFrame(rows, columns=["cust", "week"])
    raw = raw[raw["week"] <= total_w]

    # Bucket to WHOLE weeks first — the real thesis panels are weekly (one row per
    # customer per active week). Deriving BOTH the R event log and the Python panel
    # from the same weekly buckets guarantees identical sufficient statistics
    # (x, t.x, T.cal), so the comparison tests the sampler/forecast, not a data-prep
    # mismatch (R's elog2cbs counts occasions at daily resolution otherwise).
    raw["wk"] = np.floor(raw["week"]).astype(int)
    base = pd.Timestamp("1997-01-06")             # a Monday
    weekly = raw.groupby(["cust", "wk"]).size().rename("Transactions").reset_index()
    weekly["date"] = base + pd.to_timedelta(weekly["wk"] * 7, unit="D")

    # R event log: one row per (customer, active week).
    elog = weekly[["cust", "date"]].copy()
    # Python panel: one row per (customer, active week) with the weekly count.
    panel = weekly.rename(columns={"cust": "Id", "date": "period_start"})[
        ["Id", "period_start", "Transactions"]]
    return elog, panel


def run_r_btydplus(elog: pd.DataFrame, t_cal_w, h, seed):
    """Run BTYDplus elog2cbs -> pnbd.mcmc -> future tx; return (ids, expected_tx)."""
    with tempfile.TemporaryDirectory() as d:
        elog_csv = Path(d) / "elog.csv"
        out_csv = Path(d) / "out.csv"
        e = elog[["cust", "date"]].copy()
        e["date"] = e["date"].dt.strftime("%Y-%m-%d")
        e.to_csv(elog_csv, index=False)
        t_cal_date = (pd.Timestamp("1997-01-06")
                      + pd.to_timedelta(t_cal_w * 7, unit="D")).strftime("%Y-%m-%d")

        r_script = f"""
        suppressMessages(library(BTYDplus)); suppressMessages(library(data.table))
        set.seed({seed})
        elog <- fread("{elog_csv}"); elog$date <- as.Date(elog$date)
        cbs <- elog2cbs(elog, units="week", T.cal=as.Date("{t_cal_date}"))
        draws <- pnbd.mcmc.DrawParameters(cbs, mcmc=2500, burnin=500, thin=50,
                                          chains=2, trace=1e9)
        xs <- mcmc.DrawFutureTransactions(cbs, draws, T.star={h})
        exp_tx <- apply(xs, 2, mean)
        out <- data.frame(cust=cbs$cust, exp_tx=exp_tx)
        fwrite(out, "{out_csv}")
        cat("R_DONE\\n")
        """
        rscript = Path(d) / "run.R"
        rscript.write_text(r_script)
        res = subprocess.run(["Rscript", str(rscript)], capture_output=True, text=True)
        if "R_DONE" not in res.stdout:
            print("R STDERR:\n", res.stderr[-2000:]); raise RuntimeError("R failed")
        out = pd.read_csv(out_csv)
        return out["cust"].tolist(), out["exp_tx"].to_numpy()


def main():
    elog_full, panel_full = make_synthetic_elog(N_CUST, T_CAL_W, H, SEED)

    # Slice BOTH sides to the CALIBRATION window only — `compute_pareto_paper_predictions`
    # (like the real pipeline) is given the calibration panel and infers cal_end from
    # its max period, so the panel must not contain holdout weeks. R gets the same cut
    # via an explicit T.cal date. Cohort = customers with >=1 calibration purchase.
    base = pd.Timestamp("1997-01-06")
    cal_cut = base + pd.to_timedelta(T_CAL_W * 7, unit="D")
    panel = panel_full[panel_full["period_start"] <= cal_cut].copy()
    elog = elog_full[elog_full["date"] <= cal_cut].copy()
    print(f"cohort: {panel['Id'].nunique()} customers, "
          f"{len(elog)} calibration events, T_cal={T_CAL_W}w, H={H}w")

    # --- R reference ---
    r_ids, r_exp = run_r_btydplus(elog, T_CAL_W, H, SEED)
    r_map = dict(zip(r_ids, r_exp))

    # --- Python port (forecast aligned to R's customer order) ---
    py_pred, py_ids = compute_pareto_paper_predictions(
        panel, holdout_length=H, id_col="Id", target_col="Transactions",
        time_col="period_start", period_in_days=PERIOD_DAYS,
        mcmc=2500, burnin=500, thin=50, chains=2, seed=SEED,
    )
    py_total = py_pred.sum(axis=1)                 # per-customer expected holdout total
    py_map = dict(zip(py_ids, py_total))

    common = [c for c in r_ids if c in py_map]
    r_v = np.array([r_map[c] for c in common])
    p_v = np.array([py_map[c] for c in common])

    print("\n=== per-customer expected holdout transactions ===")
    print(f"  customers compared : {len(common)}")
    print(f"  R   aggregate total: {r_v.sum():10.2f}  (mean {r_v.mean():.4f}/cust)")
    print(f"  Py  aggregate total: {p_v.sum():10.2f}  (mean {p_v.mean():.4f}/cust)")
    agg_diff = 100.0 * (p_v.sum() - r_v.sum()) / r_v.sum()
    print(f"  aggregate diff     : {agg_diff:+.2f}%")
    corr = np.corrcoef(r_v, p_v)[0, 1]
    print(f"  per-customer corr  : {corr:.4f}")
    rmse = np.sqrt(np.mean((p_v - r_v) ** 2))
    print(f"  per-customer RMSE  : {rmse:.4f}")

    ok = abs(agg_diff) < 8.0 and corr > 0.95
    print("\nRESULT:", "PASS — Python port matches BTYDplus within MC noise"
          if ok else "REVIEW — larger gap than expected, inspect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
