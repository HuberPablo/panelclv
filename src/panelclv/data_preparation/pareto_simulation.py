"""Synthetic customer-period panels drawn from a Pareto/NBD process.

This is a *generator*, the mirror image of the Pareto/NBD *estimators* in
``panelclv/benchmarks/``. Those read a panel and infer the four population
parameters ``(r, alpha, s, beta)``; this one takes those four parameters as a
hand-picked ground truth and runs the model **forward** to synthesise a weekly
transaction panel in the exact schema ``prepare_dataset`` consumes
(``Id, year, week, Transactions``).

Why generate data this way
--------------------------
Because the data-generating process is known, each synthetic panel comes with a
ground truth the neural models never see: every customer's true purchase rate
``lambda``, dropout rate ``mu`` and (latent, unobserved) churn week ``tau``. The
Pareto/NBD benchmark is then the *correct* model by construction — a principled
ceiling to compare the LSTM / Transformer against — and the knobs
``(r, alpha, s, beta)`` let you dial the regime (busy vs. light buyers, loyal vs.
churny base) to stress-test the models.

The model (Schmittlein, Morrison & Colombo 1987), per customer i
----------------------------------------------------------------
    purchasing   lambda_i ~ Gamma(r, alpha)     events while alive ~ Poisson(lambda_i)
    dropout      mu_i     ~ Gamma(s, beta)       lifetime tau_i ~ Exponential(mu_i)

Time is measured in **weeks**, so ``lambda`` is purchases per week and ``alpha``,
``beta`` are in week units. Two readable summaries of the parameters:

    mean weekly purchase rate  = r / alpha
    P(dropped out by week t)   = 1 - (beta / (beta + t))**s      (Lomax survival)

The second identity is how a target churn rate maps to ``beta``: fixing a horizon
``t`` and a churn fraction ``c``, ``beta = t / ((1 - c)**(-1/s) - 1)``.

All customers share the same observation window (week 0 .. n_weeks). By default
(``birth_purchase=False``) purchases come only from the Poisson process, so some
customers buy zero times in calibration and are dropped by the pipeline's
``require_calibration_activity`` filter — mirroring how a real cohort is observed.
Set ``birth_purchase=True`` to instead plant a guaranteed acquisition purchase at
week 0 for everyone (the synchronised-cohort convention), which makes that filter
a no-op.

The exact, vectorised sampling trick
------------------------------------
We never materialise individual purchase timestamps. A Poisson process of rate
``lambda`` restricted to a disjoint interval yields an *independent* Poisson count
over that interval, and churn at ``tau`` simply censors the process there. So the
count in week w (the interval ``[w, w+1)``) is

    Transactions_{i,w} ~ Poisson( lambda_i * alive_fraction_{i,w} )

    alive_fraction_{i,w} = clip( min(tau_i, w+1) - w, 0, 1 )   # weeks alive in [w, w+1)

Exact (not an approximation), so the whole simulation is two Gamma draws, one
Exponential draw and one Poisson draw over an ``(N, W)`` matrix.

Grid studies (this module's second half)
-----------------------------------------
`generate_pnbd_study(...)` sweeps a grid of ``mean_transaction_rate`` x
``churn_rate`` (the human-facing axes — it derives ``alpha``/``beta`` for you),
generating ``n_datasets`` independent replicate panels per combination and laying
them out on disk for later training / benchmarking:

    <out_path>/<study_name>/
        study_config.json                 <- the whole grid + settings
        index.csv                         <- manifest: one row per dataset
        Dataset_{rate%}_{churn%}/         <- one folder per (rate, churn) combo,
            Dataset_1/                       integer percents, e.g. Dataset_10_20
                panel.csv                 <- Id, year, week, Transactions (train-ready)
                ground_truth.csv          <- per-customer lambda, mu, tau
                config.json               <- self-describing (params, seed, schema, ...)
            Dataset_2/
            ...

`list_pnbd_datasets(study_dir)` and `load_pnbd_dataset(study_dir, combo, dataset)`
read them back, so nothing about a dataset has to be remembered outside its folder.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from panelclv.data_preparation.period_calendar import WEEKS_PER_YEAR, year_and_week

# Schema every synthetic panel is written in — recorded in each config.json so a
# reader knows exactly how to feed it to `prepare_dataset` / `PanelConfig`.
_PANEL_SCHEMA = {
    "id_col": "Id",
    "target_col": "Transactions",
    "time_cols": ["year", "week"],
    "frequency": "weekly",
}


# ---------------------------------------------------------------------------
# 1. Core generator — the forward Pareto/NBD simulation
# ---------------------------------------------------------------------------


def seasonal_weekly_multiplier(
    peaks: Sequence[int], amplitude: float, width: float,
    period: int = WEEKS_PER_YEAR,
) -> np.ndarray:
    """Length-``period`` week-of-year purchase-rate multiplier.

    ``1 + amplitude * sum_k bump_k``, where each bump is a periodic Gaussian of
    std ``width`` weeks centred on a peak week (``width=0`` -> a single-week spike),
    then normalised to mean 1 so the annual mean rate is preserved (seasonality only
    *redistributes* purchases within the year). Distances are measured on the
    ``period``-week ring, so a peak near week 51 correctly bleeds into week 0.
    Returns all-ones when there is no seasonality (no peaks or zero amplitude).

    Public because reconstructing the pattern a stored study was generated with is a
    supported operation, not an internal detail: ``studies.synthetic_grid`` scores a
    forecast against this exact curve, and it must be the *same* function that made
    the data rather than a second copy that can drift from it.
    """
    peaks = list(peaks)
    if not peaks or amplitude == 0.0:
        return np.ones(period)
    woy = np.arange(period)
    bumps = np.zeros(period)
    for p in peaks:
        d = np.abs(woy - (int(p) % period))
        d = np.minimum(d, period - d)                     # circular distance on the ring
        bumps += np.exp(-0.5 * (d / width) ** 2) if width > 0 else (d == 0).astype(float)
    weight = 1.0 + amplitude * bumps
    return weight / weight.mean()                         # annual mean 1 -> mean-preserving


def simulate_pareto_nbd_panel(
    r: float,
    alpha: float,
    s: float,
    beta: float,
    *,
    n_customers: int,
    n_weeks: int,
    start_year: int = 1999,
    seed: int = 42,
    birth_purchase: bool = False,
    seasonal_peaks: Sequence[int] = (),
    seasonal_amplitude: float = 0.0,
    seasonal_width: float = 1.0,
    id_col: str = "Id",
    target_col: str = "Transactions",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one weekly transaction panel from a Pareto/NBD process.

    Parameters
    ----------
    r, alpha, s, beta
        Population parameters (shape/rate Gammas). ``lambda_i ~ Gamma(r, alpha)``
        sets the purchase-rate spread (mean ``r/alpha`` per week); ``mu_i ~
        Gamma(s, beta)`` sets the dropout-rate spread (churn by week t is
        ``1 - (beta/(beta+t))**s``).
    n_customers, n_weeks
        Panel size: ``n_customers`` customers, one row per customer per week for
        ``n_weeks`` weeks total. Split into calibration/holdout later via ``PanelConfig``.
    start_year
        Calendar year of week 0; weeks roll into ``start_year + 1`` every
        ``WEEKS_PER_YEAR``, so the ``(year, week)`` columns match the real panels.
    seed
        Seeds a ``numpy`` ``default_rng`` for a fully reproducible panel.
    birth_purchase
        ``False`` (default): purchases come *only* from the Poisson process, so a
        customer can have zero transactions in the calibration window — matching the
        real panels, where ``require_calibration_activity=True`` then selects the
        observed cohort and drops never-buyers. Because of that filtering the
        retained cohort is *smaller* than ``n_customers`` (more so for low-rate /
        high-churn regimes), so treat ``n_customers`` as the generated pool, not the
        final cohort size. ``True``: add one guaranteed cohort-entry purchase at week
        0, so every customer has >=1 transaction (a synchronised acquired cohort, the
        textbook BTYDplus ``pnbd.GenerateData`` convention) and the calibration-
        activity filter becomes a no-op.
    seasonal_peaks, seasonal_amplitude, seasonal_width
        Optional recurring within-year seasonality. ``seasonal_peaks`` are the
        high-season weeks-of-year (0..51); each gets a bump of height
        ``seasonal_amplitude`` (e.g. 0.8 ~ +80% at the peak) spread over
        ``seasonal_width`` weeks (0 = single-week spike, ~2 = a multi-week season).
        The per-week Poisson rate is multiplied by this pattern, which is keyed to
        ``week mod 52`` so it recurs identically every year (calibration and
        holdout). The multiplier is normalised to annual mean 1, so seasonality
        redistributes purchases within the year without changing the mean rate
        (``r/alpha``). Default (empty peaks / zero amplitude) = no seasonality.
        NOTE: seasonality makes the process no longer *pure* Pareto/NBD, so the
        Pareto/NBD benchmark becomes a misspecified baseline the neural models
        (which receive week_sin/week_cos) can beat.
    id_col, target_col
        Output column names (defaults match the pipeline's conventions).

    Returns
    -------
    panel : DataFrame
        Long panel ``[id_col, "year", "week", target_col]`` — one row per
        (customer, week), zeros included. Drops straight into ``prepare_dataset``.
    ground_truth : DataFrame
        One row per customer: latent ``lambda`` (weekly rate), ``mu`` (dropout
        rate), ``tau`` (churn week; ``>= n_weeks`` means still active at the end)
        and ``alive_weeks = min(tau, n_weeks)``. The models never see these.
    """
    if min(r, alpha, s, beta) <= 0:
        raise ValueError("r, alpha, s, beta must all be > 0")
    if n_customers <= 0 or n_weeks <= 0:
        raise ValueError("n_customers and n_weeks must be positive")
    if seasonal_amplitude < 0 or seasonal_width < 0:
        raise ValueError("seasonal_amplitude and seasonal_width must be >= 0")
    bad_peaks = [p for p in seasonal_peaks if not (0 <= int(p) < WEEKS_PER_YEAR)]
    if bad_peaks:
        raise ValueError(
            f"seasonal_peaks must be weeks-of-year in [0, {WEEKS_PER_YEAR}); got {bad_peaks}"
        )

    rng = np.random.default_rng(seed)
    N, W = int(n_customers), int(n_weeks)

    # --- 1. Draw each customer's latent rates and lifetime --------------------
    # numpy's Gamma takes (shape, scale); scale = 1/rate for the shape/rate priors.
    lam = rng.gamma(shape=r, scale=1.0 / alpha, size=N)   # purchases per week, while alive
    mu = rng.gamma(shape=s, scale=1.0 / beta, size=N)     # weekly dropout hazard
    tau = rng.exponential(scale=1.0 / mu)                 # churn week (from week-0 birth)

    # --- 2. Weeks-alive fraction per (customer, week) -------------------------
    # week w spans [w, w+1); the customer is alive for min(tau, w+1) - w of it,
    # clipped to [0, 1]: full weeks before tau -> 1, the week tau lands in -> a
    # fraction, weeks after churn -> 0.
    w_lo = np.arange(W)[None, :]                          # (1, W) week left edges
    alive_frac = np.clip(np.minimum(tau[:, None], w_lo + 1) - w_lo, 0.0, 1.0)  # (N, W)

    # --- 3. Exact per-week counts (+ optional seasonality + cohort entry) -----
    # Recurring within-year multiplier (all-ones when seasonality is off), applied
    # by week-of-year so the pattern repeats each year over the whole panel.
    season = seasonal_weekly_multiplier(seasonal_peaks, seasonal_amplitude, seasonal_width)
    # The flat week counter unpacked into the (year, week-of-year) layout the panel is
    # written in — one call, so the seasonal multiplier below and the `week` column are
    # indexed by the same week-of-year rather than by two copies of the same arithmetic.
    week_idx = np.arange(W)                               # (W,) flat 0-based week
    year, woy = year_and_week(week_idx, start_year)       # (W,) each; woy is 0..51
    counts = rng.poisson(lam[:, None] * alive_frac * season[woy][None, :])  # (N, W)
    if birth_purchase:
        counts[:, 0] += 1                                # week-0 acquisition, every customer

    # --- 4. Assemble the long panel in (year, week) layout --------------------
    ids = np.arange(1, N + 1)

    panel = pd.DataFrame({
        id_col: np.repeat(ids, W),                       # customer block-repeated
        "year": np.tile(year, N),
        "week": np.tile(woy, N),
        target_col: counts.reshape(-1).astype(np.int64),
    })

    ground_truth = pd.DataFrame({
        id_col: ids,
        "lambda": lam,
        "mu": mu,
        "tau": tau,
        "alive_weeks": np.minimum(tau, float(W)),
    })

    return panel, ground_truth


# ---------------------------------------------------------------------------
# 2. Grid study — sweep (mean rate, churn), replicate, and save for later use
# ---------------------------------------------------------------------------


def _pct(fraction: float) -> str:
    """Integer-percent, dot-free folder token: 0.10 -> "10", 0.2 -> "20"."""
    return str(int(round(fraction * 100)))


def _beta_for_churn(churn_rate: float, s: float, horizon_weeks: float) -> float:
    """The ``beta`` giving `churn_rate` dropout by `horizon_weeks` (inverse Lomax).

    Solves ``1 - (beta/(beta+t))**s = churn_rate`` for beta:
        beta = t / ((1 - churn_rate)**(-1/s) - 1)
    """
    return horizon_weeks / ((1.0 - churn_rate) ** (-1.0 / s) - 1.0)


def _auto_study_name(n_rate: int, n_churn: int, n_datasets: int, base_seed: int) -> str:
    """Default study name from the grid shape and the seed, e.g. ``pnbd_study_6x4x5_seed42``.

    Every character comes from the arguments that decide what the study contains, so
    regenerating a study finds its own folder instead of making a second one beside
    it — the wall clock used to sit here, and its time is now recorded in
    ``study_config.json`` as ``created_at`` instead.
    """
    return f"pnbd_study_{n_rate}x{n_churn}x{n_datasets}_seed{base_seed}"


def generate_pnbd_study(
    mean_transaction_rates: Sequence[float],
    churn_rates: Sequence[float],
    *,
    n_customers: int,
    n_weeks: int,
    n_datasets: int,
    out_path: str | Path,
    r: float = 2.0,
    s: float = 2.0,
    n_weeks_for_churn_rate: float = 521,
    birth_purchase: bool = False,
    seasonal_peaks: Sequence[int] = (),
    seasonal_amplitude: float = 0.0,
    seasonal_width: float = 1.0,
    study_name: str | None = None,
    base_seed: int = 42,
    start_year: int = 1999,
) -> tuple[Path, pd.DataFrame]:
    """Generate a full grid study of Pareto/NBD datasets and save it to disk.

    The grid axes are the two human-facing quantities: ``mean_transaction_rates``
    (weekly purchase rate while active) and ``churn_rates`` (dropout fraction by
    week ``n_weeks_for_churn_rate``). The driver converts each to a Pareto/NBD
    parameter internally — ``alpha = r / mean_rate`` and ``beta`` from the inverse
    Lomax survival — so the folder labels always match the values. It sweeps every
    ``(rate, churn)`` combination and generates ``n_datasets`` independent replicate
    panels each, under ``out_path/study_name/`` in the layout documented atop this
    module (folders named ``Dataset_{rate%}_{churn%}`` with integer percents).

    Parameters
    ----------
    mean_transaction_rates
        Grid axis 1: mean weekly purchases per active customer (=> ``alpha = r/rate``).
    churn_rates
        Grid axis 2: fraction of customers dropped out by ``n_weeks_for_churn_rate``
        (=> ``beta``). Each must be strictly between 0 and 1.
    n_customers, n_weeks
        Size of every generated panel (identical across the grid).
    n_datasets
        Number of replicate panels per combination (the "X" in ``Dataset_1 .. X``).
    out_path
        Base directory in which the study folder is created.
    r, s
        Shared Gamma **shapes** for the purchase / dropout priors (default 2.0 each).
    n_weeks_for_churn_rate
        Horizon (in weeks) at which ``churn_rates`` is defined. Note this can exceed
        ``n_weeks``: it only calibrates ``beta``; churn *observed inside* a panel of
        length ``n_weeks`` is correspondingly smaller.
    birth_purchase
        ``False`` (default): no forced week-0 purchase, so ``require_calibration_activity``
        selects the cohort (and the retained cohort is smaller than ``n_customers``).
        ``True``: guarantee a week-0 acquisition purchase for every customer. See
        ``simulate_pareto_nbd_panel`` for the full trade-off.
    seasonal_peaks, seasonal_amplitude, seasonal_width
        Optional recurring within-year seasonality, applied identically to every
        dataset in the study (fixed, not a grid axis). See
        ``simulate_pareto_nbd_panel`` for the meaning; default = no seasonality.
    study_name
        Folder name for this study. When omitted it is derived from the grid shape
        and ``base_seed``, so the same study always regenerates into the same folder.
    base_seed
        Seeds are assigned ``base_seed, base_seed+1, ...`` across all datasets in
        generation order, so the whole study is reproducible and every replicate
        is a distinct draw.
    start_year
        Calendar year of week 0 for the ``(year, week)`` columns.

    Returns
    -------
    study_dir : Path
        The created ``out_path/study_name`` directory.
    manifest : DataFrame
        One row per generated dataset (also written to ``study_dir/index.csv``).
    """
    mean_rates = list(mean_transaction_rates)
    churns = list(churn_rates)
    if not mean_rates or not churns:
        raise ValueError("mean_transaction_rates and churn_rates must both be non-empty")
    if any(m <= 0 for m in mean_rates):
        raise ValueError("mean_transaction_rates must all be > 0")
    if any(not (0.0 < c < 1.0) for c in churns):
        raise ValueError("churn_rates must all be strictly between 0 and 1")
    if n_datasets < 1:
        raise ValueError("n_datasets must be >= 1")

    # Human grid -> Pareto/NBD parameters (kept paired with their labels).
    alpha_by_rate = [(m, r / m) for m in mean_rates]
    beta_by_churn = [(c, _beta_for_churn(c, s, n_weeks_for_churn_rate)) for c in churns]

    study_name = study_name or _auto_study_name(
        len(mean_rates), len(churns), n_datasets, base_seed
    )
    study_dir = Path(out_path) / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seed = base_seed
    # product() varies churn fastest: all churns for rate[0], then rate[1], ...
    for (mrate, alpha), (churn, beta) in product(alpha_by_rate, beta_by_churn):
        combo = f"Dataset_{_pct(mrate)}_{_pct(churn)}"       # e.g. Dataset_10_20
        combo_dir = study_dir / combo
        combo_dir.mkdir(parents=True, exist_ok=True)

        for k in range(1, n_datasets + 1):
            panel, ground_truth = simulate_pareto_nbd_panel(
                r, alpha, s, beta,
                n_customers=n_customers, n_weeks=n_weeks,
                start_year=start_year, seed=seed, birth_purchase=birth_purchase,
                seasonal_peaks=seasonal_peaks, seasonal_amplitude=seasonal_amplitude,
                seasonal_width=seasonal_width,
                id_col=_PANEL_SCHEMA["id_col"], target_col=_PANEL_SCHEMA["target_col"],
            )
            ds_name = f"Dataset_{k}"
            ds_dir = combo_dir / ds_name
            ds_dir.mkdir(parents=True, exist_ok=True)

            cfg = {
                "study": study_name,
                "combo": combo,
                "dataset": ds_name,
                "replicate": k,
                "model": "pareto_nbd",
                # Target grid labels (what the folder name encodes).
                "mean_transaction_rate": mrate,
                "churn_rate": churn,
                "n_weeks_for_churn_rate": n_weeks_for_churn_rate,
                # The actual generative parameters they map to.
                "params": {"r": r, "alpha": alpha, "s": s, "beta": beta},
                "n_customers": int(n_customers),
                "n_weeks": int(n_weeks),
                "start_year": int(start_year),
                "seed": int(seed),
                "birth_purchase": bool(birth_purchase),
                "seasonal_peaks": list(seasonal_peaks),
                "seasonal_amplitude": float(seasonal_amplitude),
                "seasonal_width": float(seasonal_width),
                "schema": dict(_PANEL_SCHEMA),
                "files": {"panel": "panel.csv", "ground_truth": "ground_truth.csv"},
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            panel.to_csv(ds_dir / cfg["files"]["panel"], index=False)
            ground_truth.to_csv(ds_dir / cfg["files"]["ground_truth"], index=False)
            with open(ds_dir / "config.json", "w") as fh:
                json.dump(cfg, fh, indent=2)

            rows.append({
                "combo": combo, "dataset": ds_name, "replicate": k,
                "mean_transaction_rate": mrate, "churn_rate": churn,
                "alpha": alpha, "beta": beta, "r": r, "s": s,
                "n_customers": int(n_customers), "n_weeks": int(n_weeks),
                "seed": int(seed),
                "panel_path": str((ds_dir / cfg["files"]["panel"]).as_posix()),
            })
            seed += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(study_dir / "index.csv", index=False)

    # Study-level self-describing config (the whole grid at a glance).
    study_cfg = {
        "study_name": study_name,
        "model": "pareto_nbd",
        "grid": {"mean_transaction_rates": mean_rates, "churn_rates": churns},
        "r": r, "s": s,
        "n_weeks_for_churn_rate": n_weeks_for_churn_rate,
        "birth_purchase": bool(birth_purchase),
        "seasonal_peaks": list(seasonal_peaks),
        "seasonal_amplitude": float(seasonal_amplitude),
        "seasonal_width": float(seasonal_width),
        "alpha_values": [a for _, a in alpha_by_rate],
        "beta_values": [b for _, b in beta_by_churn],
        "n_customers": int(n_customers), "n_weeks": int(n_weeks),
        "n_datasets_per_combo": int(n_datasets),
        "n_combinations": len(mean_rates) * len(churns),
        "n_datasets_total": len(rows),
        "base_seed": int(base_seed), "start_year": int(start_year),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(study_dir / "study_config.json", "w") as fh:
        json.dump(study_cfg, fh, indent=2)

    return study_dir, manifest


# ---------------------------------------------------------------------------
# 3. Retrieval — read a study's datasets back for training
# ---------------------------------------------------------------------------


def list_pnbd_datasets(study_dir: str | Path) -> pd.DataFrame:
    """Return a manifest of every dataset in a study by scanning its config.json files.

    Rebuilt from disk (the source of truth), so it stays correct even if
    ``index.csv`` is stale or missing.

    Raises ``FileNotFoundError`` if ``study_dir`` is not a generation study. The
    check matters because a *trained* suite tree (``Studies/<study>/<combo>__<dataset>/
    <Model>/config.json``) puts a config.json at exactly the same glob depth as a
    generation tree (``<combo>/<dataset>/config.json``). Pointing this function at
    the trained folder therefore used to match those model-spec configs and die on
    a bare ``KeyError: 'combo'`` deep inside the loop; now it says what is wrong.
    """
    study_dir = Path(study_dir)
    if not study_dir.is_dir():
        raise FileNotFoundError(f"study_dir does not exist: {study_dir}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for cfg_path in sorted(study_dir.glob("*/*/config.json")):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        if "combo" not in cfg:
            skipped += 1          # not a dataset config (e.g. a trained model spec)
            continue
        rows.append({
            "combo": cfg["combo"], "dataset": cfg["dataset"],
            "mean_transaction_rate": cfg["mean_transaction_rate"],
            "churn_rate": cfg["churn_rate"],
            "alpha": cfg["params"]["alpha"], "beta": cfg["params"]["beta"],
            "n_customers": cfg["n_customers"], "n_weeks": cfg["n_weeks"],
            "seed": cfg["seed"],
            "panel_path": str((cfg_path.parent / cfg["files"]["panel"]).as_posix()),
        })

    if not rows:
        # Distinguish the two ways an empty result happens: a path that looks like a
        # study but is the trained-results tree, vs a path with nothing in it at all.
        hint = (
            f"found {skipped} config.json file(s) with no 'combo' key — this looks like a "
            "trained-results folder (Studies/...), not a generation study"
            if skipped
            else "no <combo>/<dataset>/config.json files found"
        )
        raise FileNotFoundError(f"no Pareto/NBD datasets under {study_dir}: {hint}")

    return pd.DataFrame(rows)


def load_pnbd_dataset(
    study_dir: str | Path, combo: str, dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load one dataset by ``combo`` (``Dataset_{rate%}_{churn%}``) and ``dataset``
    (``Dataset_k``) folder names.

    Returns ``(panel, ground_truth, config)`` — the training-ready panel, the
    per-customer latent ground truth, and the parsed ``config.json``.
    """
    ds_dir = Path(study_dir) / combo / dataset
    cfg_path = ds_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no dataset at {ds_dir} (looked for {cfg_path})")
    with open(cfg_path) as fh:
        config = json.load(fh)
    panel = pd.read_csv(ds_dir / config["files"]["panel"])
    ground_truth = pd.read_csv(ds_dir / config["files"]["ground_truth"])
    return panel, ground_truth, config
