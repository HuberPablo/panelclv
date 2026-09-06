"""Replicate Jerath, Fader & Hardie's CDNOW Pareto/NBD result, and score the same
fit the way this package scores a forecast.

Written for `docs/pareto-nbd-cdnow-replication.md`; every number in that document
comes from one run of this script.

The question it answers: JFH report a cumulative aggregate MAPE of 1.35% for the
Pareto/NBD over CDNOW weeks 40-78, which reads as "near-perfect", while this
package's archived Pareto/NBD runs sit at -53% to -64% aggregate bias. Are those
the same model behaving differently on different panels, or the same behaviour
described by two different metrics?

Five configurations, all on the identical cohort and the identical 39/39 week split:

    A  this package's HB sampler, fed a BTYDplus-faithful CBS built from the raw
       event log at daily resolution
    B  this package's `compute_pareto_predictions`, fed a weekly panel on JFH's
       week grid
    C  this package's `compute_pareto_predictions`, fed the committed repo panel
       (the package's own week calendar, one period shorter)
    D  a textbook MLE Pareto/NBD, scored the way this package scores: the sum of
       per-customer conditional expectations over the holdout
    E  the same MLE parameters, scored the way JFH score: the population-level
       aggregate tracking curve, cumulative from week 1

A and D are the same model under the same metric by two estimators. D and E are the
same estimator under two metrics. That pair of contrasts is the whole point.

The raw data is NOT in this repo. Fetch the canonical 1/10th sample first:

    curl -sSO https://www.brucehardie.com/datasets/CDNOW_sample.zip
    unzip -o CDNOW_sample.zip           # -> CDNOW_sample.txt

Run with the project venv's interpreter:

    <venv>/bin/python .scratch/pnbd-cdnow-replication/replicate.py --raw CDNOW_sample.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from panelclv.benchmarks.pareto_nbd import (          # noqa: E402
    _run_single_chain,
    compute_pareto_predictions,
)
from panelclv.data_preparation.period_calendar import week_start   # noqa: E402

# --- JFH's window, to the day ------------------------------------------------
# CDNOW's observation window opens 1997-01-01 and closes 1998-06-30: 546 days,
# exactly 78 seven-day weeks. Calibration is weeks 1-39, i.e. the first 273 days,
# so the holdout opens on 1997-10-01 and runs 39 weeks to the end of the data.
ORIGIN = pd.Timestamp("1997-01-01")
CAL_CUT = pd.Timestamp("1997-10-01")
H = 39
# The published estimates this replication is checked against (JFH Table 1).
PAPER_PARAMS = (0.55, 10.58, 0.61, 11.67)
PAPER_LL = -9595.0
PAPER_CUM_MAPE, PAPER_WK_MAPE = 1.35, 20.89


# ---------------------------------------------------------------------------
# 1. The cohort
# ---------------------------------------------------------------------------

def load_occasions(raw: Path) -> pd.DataFrame:
    """Raw CDNOW records -> one row per purchase *occasion*.

    The file has 6,919 records but only 6,696 distinct (customer, date) pairs: a
    customer can appear twice on one day. Fader & Hardie's transaction unit is the
    occasion, so same-day records collapse. §5 of the write-up shows what happens
    to the likelihood if they do not.
    """
    tx = pd.read_csv(raw, sep=r"\s+", header=None,
                     names=["master_id", "Id", "Date", "CDs", "Price"])
    tx["Date"] = pd.to_datetime(tx["Date"].astype(str), format="%Y%m%d")
    occ = tx[["Id", "Date"]].drop_duplicates().sort_values(["Id", "Date"])
    return occ.reset_index(drop=True)


def build_cbs(occ: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, t_x, T) at daily resolution, the BTYDplus `elog2cbs` convention.

    Time is measured per customer from that customer's own first purchase, in
    weeks, as a real number rather than a week index -- so a customer who first
    bought on a Thursday carries a fractional age.
    """
    cal = occ[occ["Date"] < CAL_CUT]
    g = cal.groupby("Id")["Date"]
    first, last, n = g.min(), g.max(), g.size()
    x = (n - 1).to_numpy(float)                          # repeat transactions
    t_x = ((last - first).dt.days / 7.0).to_numpy(float)  # recency, weeks
    T = ((CAL_CUT - first).dt.days / 7.0).to_numpy(float)  # age, weeks
    return x, t_x, T


# ---------------------------------------------------------------------------
# 2. Metrics — the two that are easy to confuse
# ---------------------------------------------------------------------------

def holdout_metrics(weekly: np.ndarray, actual: np.ndarray) -> dict:
    """This package's view: the holdout window alone.

    `bias` is `compute_forecast_metrics`'s `bias_percent` at the aggregate; the
    cumulative MAPE here accumulates from the FIRST HOLDOUT WEEK, so its
    denominator starts at zero-plus-one-week of transactions.
    """
    cp, ca = np.cumsum(weekly), np.cumsum(actual)
    return dict(total=float(weekly.sum()),
                bias=float(100 * (weekly.sum() - actual.sum()) / actual.sum()),
                cum_mape=float(100 * np.mean(np.abs(cp - ca) / ca)),
                wk_mape=float(100 * np.mean(np.abs(weekly - actual) / actual)))


def tracking_metrics(cum: np.ndarray, weekly: np.ndarray,
                     act_cum: np.ndarray, act_weekly: np.ndarray) -> dict:
    """JFH's view: one curve over all 78 weeks, scored on weeks 40-78.

    The difference that matters is the denominator. At week 40 the cumulative
    curve already carries the ~2,457 repeat transactions of the calibration
    window, so the same absolute shortfall is divided by a much larger number.
    """
    hold = slice(39, 78)
    return dict(cum_at_39=float(cum[38]), cum_at_78=float(cum[77]),
                cum_mape=float(100 * np.mean(
                    np.abs(cum[hold] - act_cum[hold]) / act_cum[hold])),
                wk_mape=float(100 * np.mean(
                    np.abs(weekly[hold] - act_weekly[hold]) / act_weekly[hold])))


# ---------------------------------------------------------------------------
# 3. Textbook MLE Pareto/NBD (Fader & Hardie derivation note 009)
# ---------------------------------------------------------------------------

def _A0(r, alpha, s, beta, x, t_x, T):
    """The integral term shared by the likelihood and P(alive).

    Two branches keep the Gaussian hypergeometric's argument inside [0, 1) by
    putting the larger of alpha/beta in the denominator.
    """
    d = abs(alpha - beta)
    if alpha >= beta:
        p1, p2 = r + s + x, s + 1.0
        lo, hi = alpha + t_x, alpha + T
    else:
        p1, p2 = r + s + x, r + x
        lo, hi = beta + t_x, beta + T
    return (hyp2f1(p1, p2, p1 + 1.0, d / lo) / lo ** p1
            - hyp2f1(p1, p2, p1 + 1.0, d / hi) / hi ** p1)


def loglik(params, x, t_x, T) -> float:
    r, alpha, s, beta = params
    if min(params) <= 0:
        return -np.inf
    term1 = np.exp(-(r + x) * np.log(alpha + T) - s * np.log(beta + T))
    term2 = (s / (r + s + x)) * _A0(r, alpha, s, beta, x, t_x, T)
    ll = (gammaln(r + x) - gammaln(r) + r * np.log(alpha) + s * np.log(beta)
          + np.log(np.maximum(term1 + term2, 1e-300)))
    return float(ll.sum())


def p_alive(params, x, t_x, T) -> np.ndarray:
    r, alpha, s, beta = params
    A0 = _A0(r, alpha, s, beta, x, t_x, T)
    scale = np.exp((r + x) * np.log(alpha + T) + s * np.log(beta + T))
    return 1.0 / (1.0 + (s / (r + s + x)) * scale * A0)


def expected_cum(params, x, t_x, T, t) -> np.ndarray:
    """E[Y(t) | x, t_x, T]: expected repeat transactions in the next `t` weeks.

    Note `s < 1` on this data, so `(s - 1)` is negative and so is the bracket;
    the ratio is positive.
    """
    r, alpha, s, beta = params
    core = ((r + x) * (beta + T) / ((alpha + T) * (s - 1.0))
            * (1.0 - ((beta + T) / (beta + T + t)) ** (s - 1.0)))
    return core * p_alive(params, x, t_x, T)


def E_X(t, r, alpha, s, beta) -> np.ndarray:
    """Population-level expected cumulative repeat transactions by time `t`.

    Unconditional: it does not read the customer's calibration behaviour, only
    how long they have been in the cohort. This is what an aggregate tracking
    curve is drawn from.
    """
    t = np.maximum(t, 0.0)
    return (r / alpha) * (beta / (s - 1.0)) * (1.0 - (beta / (beta + t)) ** (s - 1.0))


# ---------------------------------------------------------------------------
# 4. This package's HB sampler
# ---------------------------------------------------------------------------

def hb_forecast(x, t_x, T, seed, *, mcmc=2500, burnin=500, thin=50, chains=2):
    """Config A: the package's Gibbs sampler on a CBS this script built.

    Reuses `_run_single_chain` and reimplements only the forecast assembly, which
    is four lines and lets the holdout grid be JFH's rather than the panel's.
    """
    lam_all, tau_all = [], []
    for child in np.random.SeedSequence(seed).spawn(chains):
        d = _run_single_chain(x, t_x, T, mcmc=mcmc, burnin=burnin, thin=thin,
                              param_init=(1.0, 1.0, 1.0, 1.0),
                              rng=np.random.default_rng(child))
        lam_all.append(d["lambda"])
        tau_all.append(d["tau"])
    lam, tau = np.concatenate(lam_all, 0), np.concatenate(tau_all, 0)
    pred = np.empty((x.size, H))
    for t in range(1, H + 1):
        # Holdout week t spans [T + t-1, T + t] in customer time; because
        # T_i = (CAL_CUT - first_i)/7, that is the same calendar week for everyone.
        lo, hi = T + (t - 1), T + t
        pred[:, t - 1] = (lam * np.clip(np.minimum(tau, hi) - lo, 0.0, None)).mean(0)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True,
                    help="CDNOW_sample.txt from brucehardie.com (see module docstring)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("results.json"))
    args = ap.parse_args()

    occ = load_occasions(args.raw)
    x, t_x, T = build_cbs(occ)
    print(f"cohort {x.size} customers, {len(occ)} occasions, "
          f"sum(x) = {x.sum():.0f} calibration repeat transactions")

    # --- actuals, two shapes -------------------------------------------------
    hold = occ[occ["Date"] >= CAL_CUT].copy()
    hold["t"] = (hold["Date"] - CAL_CUT).dt.days // 7 + 1
    act_hold = hold.groupby("t").size().reindex(range(1, H + 1),
                                                fill_value=0).to_numpy(float)

    first = occ.groupby("Id")["Date"].min()
    rep = occ.merge(first.rename("first"), on="Id")
    rep = rep[rep["Date"] > rep["first"]]                # repeat transactions only
    rep = rep.assign(w=(rep["Date"] - ORIGIN).dt.days // 7)
    act_weekly = rep.groupby("w").size().reindex(range(78), fill_value=0).to_numpy(float)
    act_cum = np.cumsum(act_weekly)
    print(f"repeat sales: weeks 1-39 {act_cum[38]:.0f}, "
          f"weeks 40-78 {act_hold.sum():.0f}, total {act_cum[77]:.0f}")

    results: dict = {"cohort": {"customers": int(x.size), "occasions": int(len(occ)),
                                "cal_repeat": float(x.sum()),
                                "hold_repeat": float(act_hold.sum())}}

    # --- §5 check: does the likelihood agree with the paper's? ---------------
    ll_paper = loglik(PAPER_PARAMS, x, t_x, T)
    res = minimize(lambda lp: -loglik(np.exp(lp), x, t_x, T),
                   np.log(np.array(PAPER_PARAMS)), method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
    mle = tuple(np.exp(res.x))
    print(f"\nLL at the paper's estimates: {ll_paper:,.1f}   (paper {PAPER_LL:,.1f})")
    print(f"our MLE: r={mle[0]:.3f} alpha={mle[1]:.3f} s={mle[2]:.3f} "
          f"beta={mle[3]:.3f}  LL={-res.fun:,.1f}")
    results["mle"] = {"ll_at_paper_params": ll_paper, "params": list(mle),
                      "ll": float(-res.fun)}

    # --- configs D and E: one estimator, two metrics -------------------------
    for label, params in (("paper", PAPER_PARAMS), ("ours", mle)):
        cum_h = np.array([expected_cum(params, x, t_x, T, t).sum()
                          for t in range(1, H + 1)])
        results[f"D_{label}"] = holdout_metrics(
            np.diff(np.concatenate([[0.0], cum_h])), act_hold)

        t0 = ((first - ORIGIN).dt.days / 7.0).to_numpy(float)   # trial time, weeks
        cum = np.array([E_X((w + 1) - t0, *params).sum() for w in range(78)])
        results[f"E_{label}"] = tracking_metrics(
            cum, np.diff(np.concatenate([[0.0], cum])), act_cum, act_weekly)

    print(f"\nD (holdout metric)  paper params: bias {results['D_paper']['bias']:+.2f}%")
    print(f"E (tracking metric) paper params: cumMAPE "
          f"{results['E_paper']['cum_mape']:.2f}%  wkMAPE "
          f"{results['E_paper']['wk_mape']:.2f}%   "
          f"(paper {PAPER_CUM_MAPE} / {PAPER_WK_MAPE})")

    # --- configs B and C: the package's own entry point -----------------------
    occ_b = occ.assign(w=(occ["Date"] - ORIGIN).dt.days // 7)
    panel_b = (occ_b[occ_b["w"] <= 38].groupby(["Id", "w"]).size()
               .rename("Transactions").reset_index())
    panel_b["period_start"] = ORIGIN + pd.to_timedelta(panel_b["w"] * 7, unit="D")

    repo_panel = pd.read_csv(
        REPO_ROOT / "Datasets/Dataset_clean/cdnow_customer_week_panel.csv")
    repo_panel["period_start"] = week_start(repo_panel["year"], repo_panel["week"])
    repo_panel = repo_panel[repo_panel["Transactions"] > 0]
    periods = np.sort(repo_panel["period_start"].unique())
    cal_c = repo_panel[repo_panel["period_start"] <= periods[38]]
    H_c = len(periods) - 39
    hold_c = repo_panel[repo_panel["period_start"] > periods[38]].copy()
    hold_c["t"] = hold_c["period_start"].map({p: i for i, p in enumerate(periods[39:])})
    act_c = (hold_c.groupby("t")["Transactions"].sum()
             .reindex(range(H_c), fill_value=0).to_numpy(float))
    print(f"\nrepo panel: {H_c} holdout periods carrying {act_c.sum():.0f} "
          f"transactions (JFH: {H} weeks, {act_hold.sum():.0f})")

    # --- the three sampler configurations, across seeds ----------------------
    runs: dict[str, list] = {"A": [], "B": [], "C": []}
    for s in args.seeds:
        runs["A"].append(holdout_metrics(hb_forecast(x, t_x, T, s).sum(0), act_hold))
        pB, _ = compute_pareto_predictions(
            panel_b, holdout_length=H, id_col="Id", target_col="Transactions",
            time_col="period_start", period_in_days=7.0, seed=s)
        runs["B"].append(holdout_metrics(pB.sum(0), act_hold))
        pC, _ = compute_pareto_predictions(
            cal_c, holdout_length=H_c, id_col="Id", target_col="Transactions",
            time_col="period_start", period_in_days=7.0, seed=s)
        runs["C"].append(holdout_metrics(pC.sum(0), act_c))
        print(f"  seed {s}: A {runs['A'][-1]['bias']:+.2f}%  "
              f"B {runs['B'][-1]['bias']:+.2f}%  C {runs['C'][-1]['bias']:+.2f}%",
              flush=True)
    results["runs"] = runs

    print(f"\n{'cfg':<5}{'total':>9}{'bias%':>9}{'cumMAPE%':>11}{'wkMAPE%':>10}")
    for k in "ABC":
        m = lambda n: np.array([d[n] for d in runs[k]])           # noqa: E731
        print(f"{k:<5}{m('total').mean():9.1f}{m('bias').mean():+9.2f}"
              f"{m('cum_mape').mean():11.2f}{m('wk_mape').mean():10.2f}"
              f"   sd {m('bias').std():.2f} / {m('cum_mape').std():.2f}")

    args.out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
