"""When do trials become distinguishable, and when does one run leave its flat start?

`docs/hyperparameter-search.md` §5.1. The Optuna pruner judges trials from the 4th epoch
and the `min_epochs` floor was set by precedent (0, 50 or 90). Both should come from the
curves. This trains ValendinLSTM trials for 150 epochs with no early stopping and no
pruning, records the validation loss every epoch, and rolls the weights out over the
VALIDATION window at fixed epochs. Nothing here reads the holdout, so the warm-up and
floor chosen from it have not seen the data they are later tested on.

**The trials.** 20 sets of training settings drawn at random from the archive search
space (learning rate and weight decay log-uniform, batch from {64, 128, 256}), the same
20 on every panel, seeded from `BASE_SEED`. Random rather than TPE, so the trials stand
for the space rather than for one search's preferences.

**The configuration.** Family U's `archive` · no cluster label: ValendinLSTM reading count
and week, both embedded (`run_factorial.build_data`), under the per-cell validation score
of ADR-0010.

**One work item** is one (panel, trial). It writes
`Studies/epoch_probe__ValendinLSTM__<panel>__tNN/`: `history.csv` (one row per epoch),
`rollouts.csv` (one row per checkpoint epoch) and, last, `results.csv` — the completion
marker `VastAI/supervise/pull_results.sh` and `reap_finished.sh` check.

Usage:
    python scripts/run_epoch_probe.py --preflight          # every panel, 3 epochs
    python scripts/run_epoch_probe.py --worker 3/20        # a rented box's slice
    python scripts/run_epoch_probe.py --check-complete
    python scripts/run_epoch_probe.py --report             # tables + plots
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panelclv.models import compute_forecast_metrics
from panelclv.models.losses import build_criterion
from panelclv.registry import build_model, rollout_for
from panelclv.training.loop import train_one_epoch, validate_one_epoch
from panelclv.trials import split_calibration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_factorial import build_data                                  # noqa: E402
from run_real_panel_benchmarks import WINDOWS, spearman               # noqa: E402
from run_selection_rescore import validation_view                     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"

EXPERIMENT = "epoch_probe"
MODEL, FAMILY = "ValendinLSTM", "valendin_lstm"
PANELS = sorted(WINDOWS)
BASE_SEED = 42
N_TRIALS = 20
N_EPOCHS = 150
# 1-based epochs whose weights are rolled out over the validation window.
CHECKPOINTS = (5, 10, 20, 30, 50, 75, 100, 150)
N_PATHS = 100                   # as family V: ranks trials, so stability is what counts
GRAD_CLIP = 1.0                 # fit_model's default, which every archived study used
# The archive search space (before d7e902d / 57b8c18), which family U searched.
LR_RANGE, WD_RANGE, BATCHES = (1e-4, 3e-3), (1e-6, 1e-2), (64, 128, 256)
# The stopping rule `fit_model` applies, replayed on each recorded curve by `--report`.
PATIENCE, MIN_DELTA = 7, 1e-4


def trial_settings() -> list[dict]:
    """The 20 settings, identical on every panel and on every machine."""
    rng = np.random.default_rng(BASE_SEED)
    out = []
    for t in range(N_TRIALS):
        out.append({
            "trial": t,
            "learning_rate": float(np.exp(rng.uniform(*np.log(LR_RANGE)))),
            "weight_decay": float(np.exp(rng.uniform(*np.log(WD_RANGE)))),
            "batch_size": int(rng.choice(BATCHES)),
        })
    return out


def suite_dir(panel: str, trial: int) -> Path:
    return STUDIES_BASE / f"{EXPERIMENT}__{MODEL}__{panel}__t{trial:02d}"


def work_list() -> list[tuple[str, int]]:
    """Trial-major, so a strided worker draws from every panel."""
    return [(p, t) for t in range(N_TRIALS) for p in PANELS]


def score_validation_rollout(model, view: dict, seed: int, device: str) -> dict:
    """Roll a copy of `model` out over the validation window and score it."""
    rollout = copy.deepcopy(model).to_rollout()
    forecast = rollout_for(FAMILY)(rollout, view, n_simulations=N_PATHS, seed=seed,
                                   device=device, return_simulations=False)
    pred, actual = forecast["prediction_mean"], forecast["actual"]
    return {**compute_forecast_metrics(actual, pred),
            "spearman": spearman(pred.sum(axis=1), actual.sum(axis=1)),
            "forecast_cv": float(pred.sum(axis=1).std() / max(pred.sum(axis=1).mean(), 1e-12))}


def run_item(panel: str, trial: int, device: str, data: dict | None = None,
             n_epochs: int = N_EPOCHS, checkpoints: tuple[int, ...] = CHECKPOINTS) -> Path:
    """Train one trial for `n_epochs` with no stopping, recording its curve."""
    settings = trial_settings()[trial]
    out = suite_dir(panel, trial)
    out.mkdir(parents=True, exist_ok=True)
    data = data if data is not None else build_data(panel, MODEL, "no_cluster")
    split = split_calibration(data, settings["batch_size"])
    view = validation_view(data)
    model = build_model(FAMILY, settings, split.recipe).to(device)
    criterion = build_criterion("cross_entropy")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    K = model.num_target_classes
    seed = BASE_SEED + trial

    history, rollouts = [], []
    started = time.time()
    for epoch in range(1, n_epochs + 1):
        tr = train_one_epoch(model, split.train_loader, optimizer, criterion, device, K,
                             grad_clip=GRAD_CLIP)
        va = validate_one_epoch(model, split.val_loader, criterion, device, K,
                                compute_f1=False,
                                val_score_start=split.recipe["val_score_start"])
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"]})
        if epoch in checkpoints:
            rollouts.append({"epoch": epoch, "val_loss": va["loss"],
                             **score_validation_rollout(model, view, seed, device)})
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            pd.DataFrame(rollouts).to_csv(out / "rollouts.csv", index=False)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(rollouts).to_csv(out / "rollouts.csv", index=False)
    # Written last: its presence means the item is complete.
    pd.DataFrame([{"panel": panel, **settings, "n_epochs": n_epochs,
                   "seconds": round(time.time() - started, 1)}]).to_csv(
        out / "results.csv", index=False)
    return out


def run_worker(index: int, total: int) -> int:
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} items", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache: dict[str, dict] = {}
    for n, (panel, trial) in enumerate(mine, start=1):
        if (suite_dir(panel, trial) / "results.csv").exists():
            print(f"[{n}/{len(mine)}] {panel} t{trial:02d}: done, skipping", flush=True)
            continue
        if panel not in cache:
            cache.clear()
            cache[panel] = build_data(panel, MODEL, "no_cluster")
        print(f"[{n}/{len(mine)}] {panel} t{trial:02d}: training", flush=True)
        run_item(panel, trial, device, data=cache[panel])
    return 0


def preflight() -> int:
    """Every panel, one trial, 3 epochs, rolled out at each, in a temp tree."""
    global STUDIES_BASE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real, failures = STUDIES_BASE, []
    with tempfile.TemporaryDirectory() as tmp:
        STUDIES_BASE = Path(tmp)
        for panel in PANELS:
            try:
                out = run_item(panel, 0, device, n_epochs=3, checkpoints=(1, 2, 3))
                r = pd.read_csv(out / "rollouts.csv")
                print(f"  {panel:13s} ok  val loss {r.val_loss.iloc[-1]:.4f}  "
                      f"val-rollout MAPE {r.mape_aggregate.iloc[-1]:.1f}")
            except Exception as exc:                    # noqa: BLE001 — report them all
                print(f"  {panel:13s} FAIL {type(exc).__name__}: {exc}")
                failures.append(panel)
    STUDIES_BASE = real
    if failures:
        print(f"\nfailed: {failures}")
        return 1
    print("\nEvery panel trains, rolls out and scores. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [(p, t) for p, t in work_list() if not (suite_dir(p, t) / "results.csv").exists()]
    for p in PANELS:
        have = sum((suite_dir(p, t) / "results.csv").exists() for t in range(N_TRIALS))
        print(f"{p:13s} {have:2d}/{N_TRIALS}")
    if missing:
        print(f"{len(missing)} missing")
        return 1
    print("complete")
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def replay_stopping(val: np.ndarray) -> tuple[int, int]:
    """(stop epoch, kept epoch), 1-based, under `fit_model`'s rule with no floor."""
    best, best_epoch, waited = np.inf, 0, 0
    for e, v in enumerate(val, start=1):
        if v + MIN_DELTA < best:
            best, best_epoch, waited = v, e, 0
        else:
            waited += 1
        if waited >= PATIENCE:
            return e, best_epoch
    return len(val), best_epoch


def first_stable(epochs: list[int], rhos: list[float], level: float = 0.8) -> float:
    """The first epoch from which the rank correlation stays at or above `level`."""
    for i, e in enumerate(epochs):
        if all(r >= level for r in rhos[i:]):
            return e
    return np.nan


def report(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, predict = [], []
    for panel in PANELS:
        hist, roll = {}, {}
        for t in range(N_TRIALS):
            d = suite_dir(panel, t)
            if (d / "results.csv").exists():
                hist[t] = pd.read_csv(d / "history.csv")
                roll[t] = pd.read_csv(d / "rollouts.csv")
        if len(hist) < 5:
            print(f"{panel}: {len(hist)} trials, skipped")
            continue
        V = pd.DataFrame({t: h.val_loss.values for t, h in hist.items()})   # epochs × trials
        best_so_far = V.cummin()
        final = V.min()
        epochs = list(range(1, len(V) + 1))
        rho_loss = [best_so_far.iloc[e - 1].corr(final, method="spearman") for e in epochs]
        M = pd.DataFrame({t: r.set_index("epoch").mape_aggregate for t, r in roll.items()})
        ck = list(M.index)
        rho_mape = [M.loc[c].corr(M.loc[ck[-1]], method="spearman") for c in ck]

        # Does the CE a pruner or early stopping sees at epoch t predict the forecast the
        # trial ends with? Signed so positive means "low CE at t goes with a good final
        # rollout"; the rollout MAPE at t is the comparison row.
        last = {t: r.set_index("epoch").iloc[-1] for t, r in roll.items()}
        fin = pd.DataFrame(last).T
        for t_ep in [e for e in (1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150) if e <= len(V)]:
            ce_t = best_so_far.iloc[t_ep - 1]
            row = {"panel": panel, "epoch": t_ep,
                   "CE→MAPE": ce_t.corr(fin.mape_aggregate, method="spearman"),
                   "CE→|bias|": ce_t.corr(fin.bias_percent.abs(), method="spearman"),
                   "CE→Spearman": ce_t.corr(-fin.spearman, method="spearman")}
            if t_ep in M.index:
                row["rollout MAPE→MAPE"] = M.loc[t_ep].corr(fin.mape_aggregate, method="spearman")
            predict.append(row)

        per_trial = []
        for t, h in hist.items():
            v = h.val_loss.values
            drop = v[0] - v.min()
            leave = next((e for e, x in enumerate(v, 1) if x < v[0] - MIN_DELTA), np.nan)
            half = next((e for e, x in enumerate(v, 1) if x <= v[0] - 0.5 * drop), np.nan)
            stop, kept = replay_stopping(v)
            per_trial.append(dict(leave=leave, half=half, best=int(v.argmin()) + 1,
                                  stop=stop, kept=kept,
                                  gap=float((v[kept - 1] - v.min()) / v.min() * 100)))
        pt = pd.DataFrame(per_trial)
        rows.append({
            "panel": panel, "trials": len(hist),
            "distinguishable (val loss)": first_stable(epochs, rho_loss),
            "distinguishable (rollout MAPE)": first_stable(ck, rho_mape),
            "leaves epoch-1 loss (median)": pt.leave.median(),
            "half of its improvement (median)": pt.half.median(),
            "own best epoch (median)": pt.best.median(),
            "patience 7 stops at (median)": pt.stop.median(),
            "loss lost by stopping, % (median)": round(pt.gap.median(), 2),
        })

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for t in V:
            ax[0].plot(epochs, V[t], lw=0.8)
            ax[1].plot(ck, M[t], lw=0.8, marker="o", ms=2)
        ax[0].set(title=f"{panel}: validation loss", xlabel="epoch", ylabel="CE")
        ax[1].set(title=f"{panel}: validation-rollout MAPE", xlabel="epoch", ylabel="MAPE")
        fig.tight_layout()
        fig.savefig(out_dir / f"{panel}.png", dpi=120)
        plt.close(fig)

    table = pd.DataFrame(rows).set_index("panel").T
    print(table.to_string())
    table.to_csv(out_dir / "summary.csv")

    pred = pd.DataFrame(predict)
    print("\nDoes CE at epoch t predict the final validation rollout (epoch 150)?")
    print("Rank correlation across trials; positive = low CE at t, good final forecast.\n")
    for panel, g in pred.groupby("panel"):
        print(f"-- {panel}")
        print(g.drop(columns="panel").set_index("epoch").round(2).to_string(), "\n")
    pred.to_csv(out_dir / "ce_predicts_rollout.csv", index=False)
    print(f"\nplots and summary in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / ".scratch" / "epoch-probe"),
                        help="where --report writes its plots and summary")
    args = parser.parse_args()
    if args.worker:
        i, n = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(i, n))
    if args.preflight:
        sys.exit(preflight())
    if args.check_complete:
        sys.exit(check_complete())
    report(Path(args.out))


if __name__ == "__main__":
    main()
