"""Paper-faithful Pareto/NBD benchmark — hierarchical-Bayes MCMC.

This is the "high-fidelity" companion to `Models/pareto_nbd.py`. Where that module
fits the Pareto/NBD by **frequentist MLE** (`lifetimes.ParetoNBDFitter`), this one
reproduces the **hierarchical-Bayes MCMC** estimator that Valendin et al. (2022,
IJRM) actually use as their benchmark — the `pnbd.mcmc.DrawParameters` /
`mcmc.DrawFutureTransactions` routines from Platzer's R package **BTYDplus**.

It is a pure NumPy/SciPy port (no R, no PyMC) of BTYDplus 1.2.0's Gibbs sampler,
transcribed line-by-line from the package source so the two agree up to Monte
Carlo noise (see `scripts/validate_pareto_paper.py`, which cross-checks against the
installed R package).

Model (Schmittlein, Morrison & Colombo 1987)
--------------------------------------------
Per customer i, two latent rates with Gamma population priors:

    purchasing  lambda_i ~ Gamma(r, alpha)     events while "alive" ~ Poisson(lambda_i)
    dropout     mu_i     ~ Gamma(s, beta)      lifetime tau_i ~ Exponential(mu_i)

The four population parameters (r, alpha, s, beta) themselves get vague Gamma(1e-3,
1e-3) hyperpriors. A two-level Gibbs sampler alternates:

    level 1 (per customer, closed-form):
        lambda_i ~ Gamma(r + x_i,  alpha + min(tau_i, T_i))
        mu_i     ~ Gamma(s + 1,    beta  + tau_i)
        tau_i    ~ data-augmented churn time (alive vs. churned branch below)
    level 2 (population, slice-sampled):
        (r, alpha) | {lambda_i}      via slice sampling of the Gamma log-posterior
        (s, beta)  | {mu_i}          likewise

where x_i = repeat transactions, t.x_i = recency (time of last purchase), and
T_i = age (calibration length since first purchase) — the standard Pareto/NBD
sufficient statistics, here counted at the **occasion** level (one purchase event
per active period, matching BTYDplus's `elog2cbs`).

Forecast
--------
Given posterior draws of (lambda_i, tau_i), the expected number of holdout-period-t
transactions has a closed form: conditional on being alive until tau_i, holdout
purchases are Poisson(lambda_i) on [T_i, min(tau_i, T_i + H)], so

    E[count in period t | lambda_i, tau_i]
        = lambda_i * overlap( [T_i + t-1, T_i + t],  [T_i, tau_i] )

(zero once the period starts after the sampled churn time tau_i). Averaging this
over the posterior draws is the exact expectation of BTYDplus's
`mcmc.DrawFutureTransactions` simulator, with lower variance — so we use it directly.

Public API
----------
`compute_pareto_paper_predictions` mirrors `pareto_nbd.compute_pareto_predictions`
exactly (same inputs, same `(predictions (N, H), ids)` return), so it is a drop-in
second benchmark for the plots / metrics tables.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# 1. Customer-by-sufficient-statistic (CBS) summary — BTYDplus `elog2cbs`
# ---------------------------------------------------------------------------


def _build_cbs(
    train_panel: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    time_col: str,
    period_in_days: float,
) -> pd.DataFrame:
    """Reduce a customer-period panel to per-customer (x, t_x, T_cal).

    Mirrors BTYDplus's `elog2cbs(units="week")` at occasion granularity: an
    "occasion" is one *active period* (a period with >=1 transaction), so two
    transactions in the same week count as a single repeat event — the standard
    Pareto/NBD treatment, since within-period timing is unknown for panel data.

        x_i   = (# active periods) - 1     repeat purchase occasions
        t_x_i = (last_active - first) / P  recency, in periods
        T_i   = (cal_end   - first) / P    age, in periods   (NO +1 offset)

    `first` is each customer's first active period; `cal_end` is the last period
    in the calibration panel. Time is measured per customer **from its own first
    purchase**, exactly as the BTYDplus convention.
    """
    panel = train_panel.copy()
    panel[time_col] = pd.to_datetime(panel[time_col])
    cal_end = panel[time_col].max()

    # Active periods only (an occasion = a period with positive count).
    active = panel[panel[target_col] > 0]
    first_t = active.groupby(id_col)[time_col].min()
    last_t = active.groupby(id_col)[time_col].max()
    n_active = active.groupby(id_col)[time_col].nunique()

    cbs = pd.DataFrame(index=n_active.index)
    cbs["x"] = (n_active - 1).clip(lower=0).astype(float)        # repeat occasions
    cbs["t_x"] = (last_t - first_t).dt.days / period_in_days     # recency (periods)
    cbs["T_cal"] = (cal_end - first_t).dt.days / period_in_days  # age (periods)
    # Guard the degenerate single-period customer (first == cal_end): give a
    # minimal positive age so the Gamma rates stay finite. BTYDplus's weekly
    # bucketing yields T>=1 here; we floor at one period for the same reason.
    cbs["T_cal"] = cbs["T_cal"].clip(lower=1.0)
    cbs["t_x"] = np.minimum(cbs["t_x"], cbs["T_cal"])            # recency <= age
    return cbs


# ---------------------------------------------------------------------------
# 2. Level-2 update — slice sampling of the Gamma (shape, rate) posterior
#    Port of BTYDplus src/slice-sampling.cpp (post_gamma_parameters +
#    slice_sample_cpp), called with steps=50, w=0.1.
# ---------------------------------------------------------------------------


def _post_gamma(log_shape: float, log_rate: float,
                n: float, sum_x: float, sum_log_x: float,
                h1: float, h2: float, h3: float, h4: float) -> float:
    """Log-posterior of Gamma (shape, rate) given data x and Gamma hyperpriors.

    Exactly BTYDplus's `post_gamma_parameters`: Gamma likelihood of the per-customer
    rates {x} under (shape, rate), plus Gamma(h1, h2) prior on shape and Gamma(h3,
    h4) prior on rate. Parameterised in log-space (the slice sampler walks there).
    """
    shape = np.exp(log_shape)
    rate = np.exp(log_rate)
    return (
        n * (shape * np.log(rate) - gammaln(shape))
        + (shape - 1.0) * sum_log_x
        - rate * sum_x
        + (h1 - 1.0) * np.log(shape) - shape * h2
        + (h3 - 1.0) * np.log(rate) - rate * h4
    )


def _slice_sample_gamma(rates: np.ndarray, init: tuple[float, float],
                        hyper: tuple[float, float, float, float],
                        rng: np.random.Generator,
                        steps: int = 50, w: float = 0.1) -> tuple[float, float]:
    """Slice-sample (shape, rate) for the population Gamma over `rates`.

    Faithful port of `slice_sample_cpp` driving `post_gamma_parameters`: `steps`
    sweeps, each coordinate (log shape then log rate) updated by drawing a vertical
    level `logz = f(x) - Exp(1)`, stepping the interval out in increments of `w`
    while the endpoints stay above the level, then shrinking until a sampled point
    clears it. The data only enters through the sufficient statistics (n, sum,
    sum-log), computed once.
    """
    n = float(rates.size)
    sum_x = float(rates.sum())
    sum_log_x = float(np.log(rates).sum())
    h1, h2, h3, h4 = hyper

    x = [np.log(init[0]), np.log(init[1])]   # log-space (shape, rate)

    def f(vec):
        return _post_gamma(vec[0], vec[1], n, sum_x, sum_log_x, h1, h2, h3, h4)

    logy = f(x)
    for _ in range(steps):
        for j in (0, 1):
            logz = logy - rng.exponential(1.0)            # vertical level
            u = rng.uniform(0.0, 1.0) * w                 # step out
            L = list(x); R = list(x)
            L[j] = x[j] - u
            R[j] = x[j] + (w - u)
            while f(L) > logz:
                L[j] -= w
            while f(R) > logz:
                R[j] += w
            r0, r1 = L[j], R[j]                           # shrink in
            xs = list(x)
            for _cnt in range(10000):
                xs[j] = rng.uniform(r0, r1)
                logys = f(xs)
                if logys > logz:
                    break
                if xs[j] < x[j]:
                    r0 = xs[j]
                else:
                    r1 = xs[j]
            x = list(xs)
            logy = logys
    return float(np.exp(x[0])), float(np.exp(x[1]))


# ---------------------------------------------------------------------------
# 3. One MCMC chain — port of BTYDplus `run_single_chain`
# ---------------------------------------------------------------------------


def _draw_tau(x_recency: np.ndarray, T_cal: np.ndarray,
              lam: np.ndarray, mu: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """Data-augmented draw of the latent churn time tau (BTYDplus `draw_tau`).

    p_alive at calibration end is the Pareto/NBD survival probability; an alive
    customer's tau is the calibration end plus an Exp(mu) residual lifetime, a
    churned customer's tau is drawn from the Exponential of the dropout+purchase
    process truncated to (t.x, T_cal] by inverse-CDF.
    """
    N = lam.size
    mu_lam = mu + lam
    t_diff = T_cal - x_recency
    p_alive = 1.0 / (1.0 + (mu / mu_lam) * (np.expm1(mu_lam * t_diff)))
    alive = p_alive > rng.uniform(size=N)

    tau = np.empty(N)
    if alive.any():
        # left-truncated at T_cal: still-alive residual lifetime ~ Exp(mu)
        tau[alive] = T_cal[alive] + rng.exponential(1.0 / mu[alive])
    if (~alive).any():
        idx = ~alive
        mu_lam_tx = np.minimum(700.0, mu_lam[idx] * x_recency[idx])
        mu_lam_Tcal = np.minimum(700.0, mu_lam[idx] * T_cal[idx])
        rand = rng.uniform(size=int(idx.sum()))
        tau[idx] = -np.log(
            (1.0 - rand) * np.exp(-mu_lam_tx) + rand * np.exp(-mu_lam_Tcal)
        ) / mu_lam[idx]
    return tau


def _run_single_chain(x: np.ndarray, t_x: np.ndarray, T_cal: np.ndarray,
                      *, mcmc: int, burnin: int, thin: int,
                      param_init: tuple[float, float, float, float],
                      rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Run one Gibbs chain; return thinned posterior draws of lambda, tau, and the
    four population params. Transcribes `run_single_chain` (init + sweep order)."""
    N = x.size
    n_draws = (mcmc - 1) // thin + 1
    lam_draws = np.empty((n_draws, N))
    tau_draws = np.empty((n_draws, N))
    lvl2_draws = np.empty((n_draws, 4))         # r, alpha, s, beta

    # --- init exactly as BTYDplus (lines 86-91) ---------------------------
    r, alpha, s, beta = param_init
    lam = np.full(N, x.mean() / np.where(t_x == 0, T_cal, t_x).mean())
    tau = t_x + 0.5 / lam
    mu = 1.0 / tau

    hyper = (1e-3, 1e-3, 1e-3, 1e-3)            # Gamma(1e-3, 1e-3) hyperpriors

    for step in range(1, burnin + mcmc + 1):
        # record a thinned draw (R indexing: after burnin, every `thin`-th)
        if (step - burnin) > 0 and (step - 1 - burnin) % thin == 0:
            idx = (step - 1 - burnin) // thin
            lam_draws[idx] = lam
            tau_draws[idx] = tau
            lvl2_draws[idx] = (r, alpha, s, beta)

        # level 1 — closed-form Gibbs (vectorised over customers)
        lam = rng.gamma(shape=r + x, scale=1.0 / (alpha + np.minimum(tau, T_cal)))
        lam = np.where((lam == 0) | (np.log(np.maximum(lam, 1e-300)) < -30),
                       np.exp(-30.0), lam)
        mu = rng.gamma(shape=s + 1.0, scale=1.0 / (beta + tau))
        mu = np.where((mu == 0) | (np.log(np.maximum(mu, 1e-300)) < -30),
                      np.exp(-30.0), mu)
        tau = _draw_tau(t_x, T_cal, lam, mu, rng)

        # level 2 — slice sampling of the two population Gammas
        r, alpha = _slice_sample_gamma(lam, (r, alpha), hyper, rng)
        s, beta = _slice_sample_gamma(mu, (s, beta), hyper, rng)

    return {"lambda": lam_draws, "tau": tau_draws, "level2": lvl2_draws}


# ---------------------------------------------------------------------------
# 4. Public API — fit + forecast, same contract as compute_pareto_predictions
# ---------------------------------------------------------------------------


def compute_pareto_paper_predictions(
    train_panel: pd.DataFrame,
    holdout_length: int,
    *,
    id_col: str = "Id",
    target_col: str = "Transactions",
    time_col: str = "period_start",
    period_in_days: float = 7.0,
    customer_ids: Sequence | None = None,
    mcmc: int = 2500,
    burnin: int = 500,
    thin: int = 50,
    chains: int = 2,
    seed: int = 42,
    param_init: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[np.ndarray, list]:
    """Hierarchical-Bayes Pareto/NBD forecast (BTYDplus-faithful).

    Drop-in replacement for `pareto_nbd.compute_pareto_predictions`: same inputs
    and the same `(predictions (N, holdout_length), ids)` return, so it can back
    the `pareto_nbd_benchmark` plots/metrics. MCMC controls default to BTYDplus's
    own (`mcmc=2500, burnin=500, thin=50, chains=2`); `seed` makes the whole run
    reproducible.

    `param_init` seeds the population (r, alpha, s, beta); the (1,1,1,1) default
    matches BTYDplus's fallback and is washed out by burn-in.
    """
    cbs = _build_cbs(
        train_panel, id_col=id_col, target_col=target_col,
        time_col=time_col, period_in_days=period_in_days,
    )
    x = cbs["x"].to_numpy(dtype=float)
    t_x = cbs["t_x"].to_numpy(dtype=float)
    T_cal = cbs["T_cal"].to_numpy(dtype=float)
    ids = cbs.index.tolist()

    # Independent chains under sub-streams of one seed (reproducible, parallel-safe).
    seed_seq = np.random.SeedSequence(seed)
    lam_all, tau_all = [], []
    for child in seed_seq.spawn(chains):
        draws = _run_single_chain(
            x, t_x, T_cal, mcmc=mcmc, burnin=burnin, thin=thin,
            param_init=param_init, rng=np.random.default_rng(child),
        )
        lam_all.append(draws["lambda"])
        tau_all.append(draws["tau"])
    lam = np.concatenate(lam_all, axis=0)        # (D, N) pooled posterior draws
    tau = np.concatenate(tau_all, axis=0)        # (D, N)

    # --- expected per-period holdout counts (analytic given each draw) -----
    # Holdout period t (1-based) spans customer-time [T_cal + t-1, T_cal + t].
    # Conditional on (lambda, tau): Poisson(lambda) on the part of that period the
    # customer is still alive, i.e. lambda * overlap with [T_cal, tau].
    N = x.size
    predictions = np.empty((N, holdout_length))
    for t in range(1, holdout_length + 1):
        seg_lo = T_cal + (t - 1)                 # (N,)
        seg_hi = T_cal + t
        overlap = np.clip(np.minimum(tau, seg_hi) - seg_lo, 0.0, None)  # (D, N)
        predictions[:, t - 1] = (lam * overlap).mean(axis=0)            # avg draws

    # --- optional reorder to a requested customer order --------------------
    if customer_ids is not None:
        order_map = {cid: i for i, cid in enumerate(ids)}
        missing = [cid for cid in customer_ids if cid not in order_map]
        if missing:
            raise ValueError(
                f"customer_ids contains {len(missing)} ids absent from train_panel "
                f"(first few: {missing[:5]})"
            )
        order = [order_map[cid] for cid in customer_ids]
        predictions = predictions[order]
        ids = list(customer_ids)

    return predictions, ids
