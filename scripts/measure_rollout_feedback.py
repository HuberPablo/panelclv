"""What the autoregressive rollout costs a trained model, and whether a death latch helps.

Backs §4 of `docs/absorbing-death-state.md`. Two measurements, both on one set of
weights per replication so every comparison is within-model:

**The 2x2.** A forecast is a rollout: the model's own sampled count is fed back and the
AR target-features are recomputed from that sampled history. Training never does this --
it always shows the true history -- so the fitted conditional may be far better than the
forecast it produces. Reading the same weights with the true history substituted tells
you how much is lost, and substituting it in only one channel at a time tells you where.
The square is (target channel) x (AR columns), each either SAMPLED or TRUE:

    rollout   sampled / sampled   the package's own `forecast_recurrent`, the scored forecast
    ar_true   sampled / true      the recency-reset channel switched off
    tgt_true  true    / sampled   the counts handed over, recency still self-generated
    teacher   true    / true      expectation read off the softmax, no sampling at all

Every cell but `rollout` reads holdout truth: they are diagnostics, never forecasts.

**The latch.** The package rollout with one rule added -- a path silent for `L`
consecutive periods emits zero for the remainder -- as the crudest possible stand-in for
an absorbing death state: no hazard head, no new loss, nothing trained. The silence
counter is seeded from each customer's calibration-end recency, so a customer who
arrives already quiet latches at once; this is a death model, not a horizon truncation.
It is absorbing for free, because forced zeros keep the counter climbing.

**Why the AR columns are rebuilt rather than read.** `prepare_dataset` leaves
`data["holdout"]`'s AR columns as raw ZERO placeholders -- see `docs/feature_engineering.md`,
"Instead:" (3) -- because the rollout always overwrites them and true values sitting
there would be a standing leakage hazard. Reading them directly feeds a cohort that is
100% `has_transacted_before = 1` a value of 0, which is -4.11 standardized, and the LSTM
leaves the region it was fitted on entirely. The true trajectory is therefore rebuilt
the way the rollout rebuilds the sampled one: `ARFeatureState` seeded on the calibration
target history, advanced one step per period, re-standardized through `covariate_stats`.

**Read rho beside the scored metrics, never instead of them.** Spearman rank correlation
between predicted and actual per-customer holdout totals is a diagnostic added here; it
is not one of `compute_forecast_metrics`'s outputs. §4.5 is the cautionary case: the
latch lifts rho reliably in 8 replications out of 8 and is worse on every metric the
package actually scores.

Training is unseeded (CLAUDE.md priority 3), so replications are genuine replications
and the summary is the spread across them.

Usage (the package is not installed in the venv; `PYTHONPATH=src` is how every
measurement in the docs is run):
    # the 2x2, eight replications on the arm the design keeps
    PYTHONPATH=src python scripts/measure_rollout_feedback.py --mode square --arm ar_bounded_32

    # the latch sweep
    PYTHONPATH=src python scripts/measure_rollout_feedback.py --mode latch \
        --arm ar_bounded_32 --Ls 8,26,52

    # a fast wiring probe before committing a box to a full run. Note that a 3-trial
    # search and 20 paths are far too few to reproduce the numbers in the doc -- this
    # checks the wiring, not the result.
    PYTHONPATH=src python scripts/measure_rollout_feedback.py --mode square \
        --n-studies 1 --n-trials 3 --n-simulations 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from scipy.stats import spearmanr

from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.ar_features import ARFeatureState
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.models import compute_forecast_metrics
from panelclv.models.monte_carlo_forecasting import forecast_recurrent
from panelclv.trials import make_data_builder, refit_best_trial
from panelclv.tuning import run_optuna_study

from run_ar_encoding_ablation import (           # the arms and search space under test
    PANELS, arms_for, SHARED_SEARCH_SPACE, SHARED_TRAINING, check_arm_depth,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "Studies" / "rollout_feedback"


# ---------------------------------------------------------------------------
# One holdout walk, parameterised by what is fed at each step
# ---------------------------------------------------------------------------


def walk(model, db, device, *, true_ar: bool, true_target: bool, expectation: bool,
         truth: np.ndarray, latch_L: int | None = None,
         seed_silence: np.ndarray | None = None) -> np.ndarray:
    """One holdout path, mirroring `simulate_recurrent_path` step for step.

    `true_ar` / `true_target` substitute the true count for the drawn sample in the AR
    state and in the target channel respectively; they are independent, which is what
    makes the 2x2 a square rather than a line. `expectation` reads `E_q[y]` off the
    softmax instead of drawing, so the readout is deterministic. `latch_L`, when given,
    forces zero for the rest of the path once a customer has been silent that many
    consecutive periods, counting from `seed_silence` at the forecast start.

    Returns (N, T_HOLD) on CPU.
    """
    bb = model.backbone                                # logits, not samples (ADR-0007)
    K = int(model.num_target_classes)
    tgt = int(db["target_idx"])
    seq_cols = list(db["seq_cols"])
    ar_features = list(db.get("ar_features", []))
    cstats = db.get("covariate_stats") or {}

    calib = torch.as_tensor(np.asarray(db["calibration"]), dtype=torch.float32, device=device)
    hold = torch.as_tensor(np.asarray(db["holdout"]), dtype=torch.float32, device=device)
    N, T = hold.shape[0], hold.shape[1]
    levels = torch.arange(K, dtype=torch.float32, device=device)

    ar_idx = {n: seq_cols.index(n) for n in ar_features}
    ar_norm = {n: cstats.get(n, (0.0, 1.0)) for n in ar_features}
    ar_state = (ARFeatureState(calib[:, :, tgt].detach().cpu().numpy(), ar_features)
                if ar_features else None)

    latching = latch_L is not None
    silence = seed_silence.copy() if latching else None
    dead = (torch.as_tensor(silence >= latch_L, device=device) if latching
            else torch.zeros(N, dtype=torch.bool, device=device))

    out = torch.zeros((N, T), dtype=torch.float32, device=device)
    bb.to(device).eval()

    def draw(logits_1d):
        probs = torch.softmax(logits_1d, dim=-1)
        v = ((probs * levels).sum(-1) if expectation
             else torch.distributions.Categorical(probs=probs).sample().float())
        return torch.where(dead, torch.zeros_like(v), v) if latching else v

    with torch.inference_mode():
        # Warm-up: the last calibration position IS the holdout step-0 forecast, and the
        # state now summarises the whole calibration window.
        logits, state = bb(calib, None)
        drawn = draw(logits[:, -1, :])
        out[:, 0] = drawn

        for t in range(T - 1):
            drawn_np = drawn.detach().cpu().numpy()
            if latching:
                silence = np.where(drawn_np > 0, 0, silence + 1)
                dead = dead | torch.as_tensor(silence >= latch_L, device=device)
            # What the next step's input carries. The AR update and the target channel
            # are chosen independently so one channel can be fixed while the other is not.
            fed = (torch.as_tensor(truth[:, t], dtype=torch.float32, device=device)
                   if true_target else drawn)
            ar_fed = truth[:, t] if true_ar else drawn_np

            x = hold[:, t:t + 1, :].clone()
            x[:, 0, tgt] = fed.clamp(max=K - 1)        # the head only owns 0..K-1; the
            #                                            holdout target is left unclipped
            #                                            by `prepare_dataset` on purpose.
            if ar_state is not None:
                feats = ar_state.update(ar_fed)
                for name, col in ar_idx.items():
                    mean, std = ar_norm[name]
                    x[:, 0, col] = torch.as_tensor((feats[name] - mean) / std,
                                                   dtype=x.dtype, device=x.device)
            logits, state = bb(x, state)
            drawn = draw(logits[:, 0, :])
            out[:, t + 1] = drawn
    return out.cpu().numpy()


def calibration_end_silence(db) -> np.ndarray:
    """Periods since each customer's last calibration purchase, at the forecast start."""
    cal = np.asarray(db["calibration"])[:, :, int(db["target_idx"])]
    active = cal > 0
    T = cal.shape[1]
    last = np.where(active.any(1), T - 1 - active[:, ::-1].argmax(1), -1)
    return (T - 1 - last).astype(np.int64)


def score(actual: np.ndarray, pred: np.ndarray, tag: str) -> dict:
    """`compute_forecast_metrics` (the scoring authority) plus the rank diagnostic."""
    m = compute_forecast_metrics(actual, pred)
    m["rho"] = float(spearmanr(pred.sum(1), actual.sum(1)).statistic)
    m["readout"] = tag
    return m


# ---------------------------------------------------------------------------
# One replication: search, refit, then every readout off those weights
# ---------------------------------------------------------------------------

SQUARE = {                       # name: (true_ar, true_target, expectation)
    "ar_true":  (True, False, False),
    "tgt_true": (False, True, False),
    "teacher":  (True, True, True),
}


def run(panel: str, arm: str, mode: str, n_studies: int, n_trials: int,
        n_sims: int, base_seed: int, device: str, Ls: list[int]) -> pd.DataFrame:
    csv, config_for = PANELS[panel]
    data = panel_dataset.prepare_dataset(pd.read_csv(csv), config_for(arms_for(panel)[arm]))
    check_arm_depth(arm, data)

    rows: list[dict] = []
    for i in range(1, n_studies + 1):
        t0, seed = time.time(), base_seed + i
        sdir = OUT_DIR / f"{panel}_{arm}_{mode}" / f"study_{i:02d}"
        study = run_optuna_study(
            model_type="lstm",
            data_builder=make_data_builder(data),
            search_space=dict(SHARED_SEARCH_SPACE),
            training={**SHARED_TRAINING, "seed": seed,
                      "checkpoint_dir": str(sdir / "checkpoints")},
            n_trials=n_trials, device=device, study_name=f"study_{i:02d}",
            append_timestamp=False, summary_dir=sdir,
            sampler=optuna.samplers.TPESampler(seed=seed),
            keep_only_best_checkpoint=True,
        )
        # Every forecast comes from a refit on the full calibration window (ADR-0008).
        model, db = refit_best_trial(study, data, "lstm", device=device,
                                     checkpoint_dir=str(sdir / "refit"), verbose=False)
        truth = holdout_actuals(db)

        fc = forecast_recurrent(model, db, n_simulations=n_sims, seed=seed,
                                device=device, return_simulations=False)
        actual = fc["actual"]
        preds = {"rollout": fc["prediction_mean"]}

        if mode == "square":
            for tag, (t_ar, t_tg, exp) in SQUARE.items():
                torch.manual_seed(seed)
                preds[tag] = (walk(model, db, device, true_ar=t_ar, true_target=t_tg,
                                   expectation=True, truth=truth) if exp else
                              np.stack([walk(model, db, device, true_ar=t_ar,
                                             true_target=t_tg, expectation=False,
                                             truth=truth)
                                        for _ in range(n_sims)], 0).mean(0))
        else:
            seed_sil = calibration_end_silence(db)
            for L in Ls:
                torch.manual_seed(seed)
                preds[f"latch_{L}"] = np.stack(
                    [walk(model, db, device, true_ar=False, true_target=False,
                          expectation=False, truth=truth, latch_L=L, seed_silence=seed_sil)
                     for _ in range(n_sims)], 0).mean(0)
            preds["teacher"] = walk(model, db, device, true_ar=True, true_target=True,
                                    expectation=True, truth=truth)

        for tag, p in preds.items():
            r = score(actual, p, tag)
            r.update(panel=panel, arm=arm, mode=mode, study=i, seed=seed)
            rows.append(r)
        d = {r["readout"]: r for r in rows[-len(preds):]}
        print(f"  study {i:02d} ({time.time() - t0:4.0f}s)  " + " | ".join(
            f"{k} rho {d[k]['rho']:+.3f} bias {d[k]['bias_percent']:+7.1f}%"
            for k in preds), flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, mode: str) -> None:
    """Distribution across replications, plus the 2x2 main effects when there is a square."""
    cols = ["rho", "bias_percent", "mape_aggregate", "rmse", "rmse_customer_total"]
    print(f"\n{df.study.nunique()} replications\n")
    print(df.groupby("readout")[cols].agg(["mean", "std"]).round(4).to_string())

    rho = df.groupby("readout")["rho"].mean()
    if mode == "square":
        print("\n2x2 of mean Spearman rho (target channel x AR columns):")
        print(f"{'':>16}{'AR sampled':>13}{'AR true':>10}")
        print(f"{'target sampled':>16}{rho['rollout']:>13.3f}{rho['ar_true']:>10.3f}")
        print(f"{'target true':>16}{rho['tgt_true']:>13.3f}{rho['teacher']:>10.3f}")
        gap = rho["teacher"] - rho["rollout"]
        print(f"\nfixing AR    : {rho['ar_true'] - rho['rollout']:+.3f} (sampled target), "
              f"{rho['teacher'] - rho['tgt_true']:+.3f} (true target)")
        print(f"fixing target: {rho['tgt_true'] - rho['rollout']:+.3f} (sampled AR), "
              f"{rho['teacher'] - rho['ar_true']:+.3f} (true AR)")
        print(f"total gap {gap:+.3f}; AR share "
              f"{(rho['ar_true'] - rho['rollout']) / gap:.0%} .. "
              f"{(rho['teacher'] - rho['tgt_true']) / gap:.0%}")
    else:
        base = df.loc[df.readout == "rollout", "rho"].to_numpy()
        gap = rho["teacher"] - rho["rollout"]
        print(f"\nrollout {rho['rollout']:.3f} -> teacher {rho['teacher']:.3f}; gap {gap:.3f}")
        for k in sorted(r for r in rho.index if r.startswith("latch_")):
            d = df.loc[df.readout == k, "rho"].to_numpy() - base
            print(f"  {k:>9}: rho {rho[k]:.3f}  {(rho[k] - rho['rollout']) / gap:>4.0%} of gap  "
                  f"paired {d.mean():+.3f} +/- {d.std(ddof=1):.3f}, wins {int((d > 0).sum())}/{len(d)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default="electronics", choices=sorted(PANELS))
    p.add_argument("--arm", default="ar_bounded_32")
    p.add_argument("--mode", default="square", choices=("square", "latch"))
    p.add_argument("--n-studies", type=int, default=8)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--n-simulations", type=int, default=200)
    p.add_argument("--base-seed", type=int, default=4000)
    p.add_argument("--Ls", default="8,26,52", help="latch depths, --mode latch only")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ls = [int(x) for x in a.Ls.split(",")]
    df = run(a.panel, a.arm, a.mode, a.n_studies, a.n_trials, a.n_simulations,
             a.base_seed, device, Ls)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a.mode}_{a.panel}_{a.arm}.csv"
    df.to_csv(out, index=False)
    report(df, a.mode)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
