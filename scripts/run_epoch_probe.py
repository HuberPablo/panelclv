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

**Calibrations.** `--calibration` picks the windows: `2y` (the four panels, the default),
`3y` (electronics, gift and multichannel with three calibration years) or `5y` (the
paper's electronics cohort on its 260-week split), as `run_real_panel_benchmarks` defines
them; several may be given, comma-separated, for one fleet.

**One work item** is one (calibration, panel, trial). It writes
`Studies/epoch_probe[_cal3y|_cal5y]__ValendinLSTM__<panel>__tNN/`: `history.csv` (one row per epoch),
`rollouts.csv` (one row per checkpoint epoch) and, last, `results.csv` — the completion
marker `VastAI/supervise/pull_results.sh` and `reap_finished.sh` check.

Usage:
    python scripts/run_epoch_probe.py --preflight          # every panel, 3 epochs
    python scripts/run_epoch_probe.py --worker 3/20        # a rented box's slice
    python scripts/run_epoch_probe.py --check-complete
    python scripts/run_epoch_probe.py --report             # tables + plots
    python scripts/run_epoch_probe.py --worker 3/20 --calibration 3y,5y
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
from run_real_panel_benchmarks import CALIBRATIONS, calibration_tag, spearman  # noqa: E402
from run_real_panel_benchmarks import build_data as build_benchmark_data       # noqa: E402
from run_selection_rescore import validation_view                     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"

EXPERIMENT = "epoch_probe"
MODEL, FAMILY = "ValendinLSTM", "valendin_lstm"
PANELS = sorted(CALIBRATIONS["2y"])
# The calibrations this invocation covers; `--calibration` sets it.
CALS = ["2y"]
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


def panels_of(cal: str) -> list[str]:
    return sorted(CALIBRATIONS[cal])


def suite_dir(panel: str, trial: int, cal: str = "2y") -> Path:
    return STUDIES_BASE / f"{EXPERIMENT}{calibration_tag(cal)}__{MODEL}__{panel}__t{trial:02d}"


def load_panel(panel: str, cal: str) -> dict:
    """The panel as ValendinLSTM reads it: count and week, both embedded.

    `2y` goes through `run_factorial`, the configuration of family U's cell the probe
    started from; the others through `run_real_panel_benchmarks`, which declares the 3y and
    5y windows. The two build the same columns.
    """
    if cal == "2y":
        return build_data(panel, MODEL, "no_cluster")
    return build_benchmark_data(panel, cal)


def work_list() -> list[tuple[str, str, int]]:
    """Trial-major, so a strided worker draws from every panel of every calibration."""
    return [(c, p, t) for t in range(N_TRIALS) for c in CALS for p in panels_of(c)]


# Customers rolled out at once. The warm-up reads every customer's whole pre-validation
# history in one pass; on the 5y split (3,755 customers x 208 weeks) that needs about
# 8 GB and crashed an 8 GB card. A customer's rollout never reads another's, so chunking
# changes nothing but the memory peak (and which random draws each customer gets).
ROLLOUT_CHUNK = 1000


def score_validation_rollout(model, view: dict, seed: int, device: str) -> dict:
    """Roll a copy of `model` out over the validation window and score it."""
    rollout = copy.deepcopy(model).to_rollout()
    preds, actuals = [], []
    n = np.asarray(view["calibration"]).shape[0]
    for i, start in enumerate(range(0, n, ROLLOUT_CHUNK)):
        part = dict(view)
        part["calibration"] = view["calibration"][start:start + ROLLOUT_CHUNK]
        part["holdout"] = view["holdout"][start:start + ROLLOUT_CHUNK]
        forecast = rollout_for(FAMILY)(rollout, part, n_simulations=N_PATHS, seed=seed + i,
                                       device=device, return_simulations=False)
        preds.append(forecast["prediction_mean"])
        actuals.append(forecast["actual"])
    pred, actual = np.concatenate(preds), np.concatenate(actuals)
    return {**compute_forecast_metrics(actual, pred),
            "spearman": spearman(pred.sum(axis=1), actual.sum(axis=1)),
            "forecast_cv": float(pred.sum(axis=1).std() / max(pred.sum(axis=1).mean(), 1e-12))}


def run_item(panel: str, trial: int, device: str, data: dict | None = None,
             n_epochs: int = N_EPOCHS, checkpoints: tuple[int, ...] = CHECKPOINTS,
             cal: str = "2y") -> Path:
    """Train one trial for `n_epochs` with no stopping, recording its curve."""
    settings = trial_settings()[trial]
    out = suite_dir(panel, trial, cal)
    out.mkdir(parents=True, exist_ok=True)
    data = data if data is not None else load_panel(panel, cal)
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
    pd.DataFrame([{"calibration": cal, "panel": panel, **settings, "n_epochs": n_epochs,
                   "seconds": round(time.time() - started, 1)}]).to_csv(
        out / "results.csv", index=False)
    return out


def run_worker(index: int, total: int) -> int:
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} items", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache: dict[tuple[str, str], dict] = {}
    for n, (cal, panel, trial) in enumerate(mine, start=1):
        tag = f"{cal} {panel} t{trial:02d}"
        if (suite_dir(panel, trial, cal) / "results.csv").exists():
            print(f"[{n}/{len(mine)}] {tag}: done, skipping", flush=True)
            continue
        if (cal, panel) not in cache:
            cache.clear()               # one panel at a time; the 5y panel is large
            cache[(cal, panel)] = load_panel(panel, cal)
        print(f"[{n}/{len(mine)}] {tag}: training", flush=True)
        run_item(panel, trial, device, data=cache[(cal, panel)], cal=cal)
    return 0


def preflight() -> int:
    """Every panel, one trial, 3 epochs, rolled out at each, in a temp tree."""
    global STUDIES_BASE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real, failures = STUDIES_BASE, []
    with tempfile.TemporaryDirectory() as tmp:
        STUDIES_BASE = Path(tmp)
        for cal in CALS:
            for panel in panels_of(cal):
                try:
                    out = run_item(panel, 0, device, n_epochs=3, checkpoints=(1, 2, 3),
                                   cal=cal)
                    r = pd.read_csv(out / "rollouts.csv")
                    print(f"  {cal} {panel:13s} ok  val loss {r.val_loss.iloc[-1]:.4f}  "
                          f"val-rollout MAPE {r.mape_aggregate.iloc[-1]:.1f}")
                except Exception as exc:                # noqa: BLE001 — report them all
                    print(f"  {cal} {panel:13s} FAIL {type(exc).__name__}: {exc}")
                    failures.append(f"{cal}/{panel}")
    STUDIES_BASE = real
    if failures:
        print(f"\nfailed: {failures}")
        return 1
    print("\nEvery panel trains, rolls out and scores. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [w for w in work_list()
               if not (suite_dir(w[1], w[2], w[0]) / "results.csv").exists()]
    for c in CALS:
        for p in panels_of(c):
            have = sum((suite_dir(p, t, c) / "results.csv").exists() for t in range(N_TRIALS))
            print(f"{c} {p:13s} {have:2d}/{N_TRIALS}")
    if missing:
        print(f"{len(missing)} missing")
        return 1
    print("complete")
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def replay_stopping(val: np.ndarray) -> tuple[int, int]:
    """(stop epoch, kept epoch), 1-based, under `fit_model`'s rule: patience 7, no floor."""
    best, best_epoch, waited = np.inf, 0, 0
    for e, v in enumerate(val, start=1):
        if v + MIN_DELTA < best:
            best, best_epoch, waited = v, e, 0
        else:
            waited += 1
        if waited >= PATIENCE:
            return e, best_epoch
    return len(val), best_epoch


# Prechelt (1998), "Early Stopping -- But When?": three families of stopping criteria,
# each read off the validation and training loss alone. The weights kept are always those
# of the lowest validation loss so far.
#   GL_a  generalisation loss: stop once the validation loss is more than a% above its
#         best so far.
#   PQ_a  progress quotient: stop once that generalisation loss, divided by the training
#         progress over the last strip of K epochs, exceeds a -- it waits while training
#         is still improving.
#   UP_s  stop once the validation loss has risen at the end of s successive strips.
STRIP = 5                       # Prechelt's strip length K
PRECHELT = [("GL", 1), ("GL", 2), ("GL", 3), ("GL", 5),
            ("PQ", 0.5), ("PQ", 1), ("PQ", 2), ("PQ", 3),
            ("UP", 2), ("UP", 3), ("UP", 4)]
CHOSEN = ("PQ", 1)              # the criterion §5.2 recommends, marked in the figures


def replay_prechelt(val: np.ndarray, train: np.ndarray, criterion: str,
                    alpha: float) -> tuple[int, int]:
    """(stop epoch, kept epoch), 1-based, under one of Prechelt's criteria."""
    best, best_epoch, ups, last_strip = np.inf, 0, 0, None
    for t in range(1, len(val) + 1):
        if val[t - 1] < best:
            best, best_epoch = val[t - 1], t
        gl = 100 * (val[t - 1] / best - 1)
        if criterion == "GL" and gl > alpha:
            return t, best_epoch
        if t % STRIP == 0:
            strip = train[t - STRIP:t]
            progress = 1000 * (strip.sum() / (STRIP * strip.min()) - 1)
            if criterion == "PQ" and progress > 0 and gl / progress > alpha:
                return t, best_epoch
            if criterion == "UP":
                ups = ups + 1 if last_strip is not None and val[t - 1] > last_strip else 0
                last_strip = val[t - 1]
                if ups >= alpha:
                    return t, best_epoch
    return len(val), best_epoch


def score_at(rollouts: pd.DataFrame, epoch: int) -> pd.Series:
    """The validation rollout at the latest recorded checkpoint at or before `epoch`."""
    earlier = [c for c in rollouts.index if c <= epoch]
    return rollouts.loc[earlier[-1]] if earlier else rollouts.iloc[0]


def first_stable(epochs: list[int], rhos: list[float], level: float = 0.8) -> float:
    """The first epoch from which the rank correlation stays at or above `level`."""
    for i, e in enumerate(epochs):
        if all(r >= level for r in rhos[i:]):
            return e
    return np.nan


def report(out_dir: Path, cal: str = "2y") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, predict = [], []
    PANELS = panels_of(cal)
    for panel in PANELS:
        hist, roll = {}, {}
        for t in range(N_TRIALS):
            d = suite_dir(panel, t, cal)
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

        # Where each stopping rule would end, replayed on the recorded curves.
        kept_now = int(np.median([replay_stopping(h.val_loss.values)[1] for h in hist.values()]))
        kept_new = int(np.median([replay_prechelt(h.val_loss.values, h.train_loss.values,
                                                  *CHOSEN)[1] for h in hist.values()]))
        own_best = int(pt.best.median())
        B = pd.DataFrame({t: r.set_index("epoch").bias_percent.abs() for t, r in roll.items()})
        C = pd.DataFrame({t: r.set_index("epoch").forecast_cv for t, r in roll.items()})

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for t in V:
            ax[0].plot(epochs, V[t], lw=0.6, color="0.75")
            ax[1].plot(ck, M[t], lw=0.6, color="0.75")
        ax[0].plot(epochs, V.median(axis=1), lw=2, color="k", label="median over trials")
        ax[1].plot(ck, M.median(axis=1), lw=2, color="k", marker="o", label="median MAPE")
        ax[1].plot(ck, B.median(axis=1), lw=2, color="tab:purple", marker="s",
                   label="median |bias| %")
        ax[2].bar(ck, (C < 0.2).sum(axis=1), width=4, color="tab:red")
        for a in ax:
            a.axvline(kept_now, color="tab:orange", ls="--",
                      label=f"patience 7 keeps epoch {kept_now}")
            a.axvline(kept_new, color="tab:green", ls="--",
                      label=f"Prechelt {CHOSEN[0]}{CHOSEN[1]} keeps epoch {kept_new}")
            a.axvline(own_best, color="tab:blue", ls=":", label=f"loss minimum, epoch {own_best}")
        # The first epochs sit orders of magnitude above the rest; scale to what follows
        # them so the late differences between trials are visible.
        tail = V.iloc[4:].values.ravel()
        ax[0].set_ylim(np.nanmin(tail) * 0.98, np.nanquantile(tail, 0.98) * 1.02)
        ax[1].set_yscale("log")
        ax[0].set(title=f"{panel}: validation loss", xlabel="epoch", ylabel="CE")
        ax[1].set(title="validation-window forecast", xlabel="epoch", ylabel="% (log)")
        ax[2].set(title="collapsed trials (forecast CV < 0.2)", xlabel="epoch",
                  ylabel=f"of {len(hist)}")
        ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{panel}.png", dpi=110)
        plt.close(fig)

    table = pd.DataFrame(rows).set_index("panel").T
    print(table.to_string())
    table.to_csv(out_dir / "summary.csv")

    # Prechelt's criteria (§5.2), replayed on every recorded curve and scored afterwards
    # on the validation-window rollout at the epoch each keeps. The current rule and the
    # oracle (the single checkpoint with the best median validation MAPE) frame them.
    runs = {}
    for panel in PANELS:
        for t in range(N_TRIALS):
            d = suite_dir(panel, t, cal)
            if (d / "results.csv").exists():
                runs[(panel, t)] = (pd.read_csv(d / "history.csv"),
                                    pd.read_csv(d / "rollouts.csv").set_index("epoch"))
    rules = [("current: patience 7", lambda h: replay_stopping(h.val_loss.values))]
    rules += [(f"{c}{a}", lambda h, c=c, a=a: replay_prechelt(
        h.val_loss.values, h.train_loss.values, c, a)) for c, a in PRECHELT]
    table = []
    for name, rule in rules:
        row, cost = {"rule": name}, []
        for panel in PANELS:
            mine = [(h, r, rule(h)) for (p, _), (h, r) in runs.items() if p == panel]
            sc = [score_at(r, kept) for _, r, (_, kept) in mine]
            gap = [(h.val_loss.values[kept - 1] - h.val_loss.min()) / h.val_loss.min() * 100
                   for h, _, (_, kept) in mine]
            row[f"{panel} MAPE"] = float(np.median([x.mape_aggregate for x in sc]))
            row[f"{panel} |bias|"] = float(np.median([abs(x.bias_percent) for x in sc]))
            row[f"{panel} Spearman"] = float(np.median([x.spearman for x in sc]))
            row[f"{panel} collapsed"] = int(sum(x.forecast_cv < 0.2 for x in sc))
            row[f"{panel} epoch"] = float(np.median([kept for _, _, (_, kept) in mine]))
            row[f"{panel} gap %"] = float(np.median(gap))
            cost += [stop for _, _, (stop, _) in mine]
        row["epochs trained"] = float(np.mean(cost))
        table.append(row)
    oracle = {"rule": "oracle: best fixed checkpoint"}
    for panel in PANELS:
        R = {t: r for (p, t), (_, r) in runs.items() if p == panel}
        med = pd.DataFrame({t: r.mape_aggregate for t, r in R.items()}).median(axis=1)
        e = med.idxmin()
        oracle[f"{panel} MAPE"] = float(med.min())
        oracle[f"{panel} |bias|"] = float(np.median([abs(r.loc[e].bias_percent) for r in R.values()]))
        oracle[f"{panel} Spearman"] = float(np.median([r.loc[e].spearman for r in R.values()]))
        oracle[f"{panel} collapsed"] = int(sum(r.loc[e].forecast_cv < 0.2 for r in R.values()))
        oracle[f"{panel} epoch"] = float(e)
    table.append(oracle)
    table = pd.DataFrame(table)
    print("\nStopping criteria (Prechelt 1998) on the validation-window forecast")
    print(table[["rule", "epochs trained"] + [f"{p} MAPE" for p in PANELS]
                + [f"{p} epoch" for p in PANELS]].round(1).to_string(index=False))
    table.to_csv(out_dir / "stopping_criteria.csv", index=False)

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
    parser.add_argument("--calibration", default="2y",
                        help="comma-separated, from " + ", ".join(CALIBRATIONS)
                             + "; --report takes one")
    parser.add_argument("--out", default=str(REPO_ROOT / ".scratch" / "epoch-probe"),
                        help="where --report writes its plots and summary")
    args = parser.parse_args()
    CALS[:] = args.calibration.split(",")
    unknown = [c for c in CALS if c not in CALIBRATIONS]
    if unknown:
        parser.error(f"unknown calibration(s) {unknown}")
    if args.worker:
        i, n = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(i, n))
    if args.preflight:
        sys.exit(preflight())
    if args.check_complete:
        sys.exit(check_complete())
    if len(CALS) != 1:
        parser.error("--report takes one calibration")
    out = Path(args.out) if CALS[0] == "2y" else Path(args.out) / f"cal{CALS[0]}"
    report(out, CALS[0])


if __name__ == "__main__":
    main()
