"""Weekly holdout volume on each real panel: the actuals against each arm's best study.

The real-panel tables (`docs/benchmarks-real-panels.md`) say how far each arm's total
volume is from the truth over the whole holdout year; they do not say *when* the miss
happens. This script draws that: for every panel, the true weekly transaction volume over
the holdout and, on the same axes, the simulated volume of each arm.

**One curve per arm: its best-MAPE study.** An arm is a distribution — 100 replications
for an LSTM encoding, 20 for ValendinLSTM, 1 for the deterministic Pareto/NBD fit — and a
single curve has to come from one of them. The one drawn is the replication with the
lowest `mape_aggregate`, so every arm is shown at its best rather than at a draw of the
weight-init lottery. The legend carries that MAPE, and the printed table carries which
replication it was, so a curve is always traceable back to a suite on disk.

**Two campaigns, never on one axes.** The `2y` run uses the benchmark's windows, so
Pareto/NBD and ValendinLSTM are scored on the same holdout year and belong beside the
LSTM encodings. The `_cal3y` run moves electronics, gift and multichannel to a
three-year calibration and therefore forecasts a *different* holdout year; the frozen
benchmarks were never run on it, so those figures carry the four encodings alone.

Curves are rebuilt from what the suites stored, not re-simulated: each
``Predictions/Prediction_1.csv`` is a wide per-customer x holdout-week table of mean
simulated counts (`panelclv.predictions`), so summing it over customers is the arm's
weekly aggregate. Actuals come from the rebuilt cohort (`holdout_actuals`), clipped at the
same ``clip_target_upper`` the models were trained under, and every metric quoted comes
from `run_real_panel_benchmarks.score` — the same scoring path as the `--report` tables,
so a MAPE here matches a MAPE there to the last decimal.

Run from the repo root:

    PYTHONPATH=src python scripts/plot_real_panel_forecasts.py

Writes ``figures/real_panel_weekly__<cal>__<panel>.png`` and, beside each, the ``.csv``
of the plotted curves — the table view the palette's contrast relief requires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_real_panel_ar as ar                       # noqa: E402
import run_real_panel_benchmarks as benchmarks       # noqa: E402

from panelclv.data_preparation.target_channel import holdout_actuals   # noqa: E402
from panelclv.studies import load_model_predictions                    # noqa: E402

OUT = Path("figures")

# The encodings the AR runner stored on the real panels, in the order they are drawn.
# `none` is absent: it was never run here.
ENCODINGS = ("bounded32", "log", "ratio", "bounded32ratio")

# Categorical identity is fixed per arm and never cycled: the first six slots of the
# reference palette, assigned by arm rather than by rank, so an arm keeps its colour
# whichever panel it is drawn on and whether or not its neighbours are present.
# Validated for the adjacent-pair line case on the light surface (worst CVD dE 9.1,
# worst normal-vision dE 19.6); three slots fall below 3:1 contrast, which the sidecar
# CSV and the in-legend MAPE answer.
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SERIES = {
    "Pareto/NBD":                 "#2a78d6",   # slot 1, blue
    "ValendinLSTM":               "#eb6834",   # slot 2, orange
    "LSTM + ar_bounded32":        "#1baf7a",   # slot 3, aqua
    "LSTM + ar_log":              "#eda100",   # slot 4, yellow
    "LSTM + ar_ratio":            "#e87ba4",   # slot 5, magenta
    "LSTM + ar_bounded32ratio":   "#008300",   # slot 6, green
}

mpl.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160,
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9, "axes.titleweight": "medium",
    "axes.labelsize": 8.5, "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Which suites make up an arm
# ---------------------------------------------------------------------------


def arms(panel: str, cal: str) -> list[tuple[str, list[Path]]]:
    """``(label, model directories)`` for every arm on one panel, in drawing order.

    A model directory is ``Studies/<suite>/<ModelName>``, the folder holding
    ``Predictions/``. A replication that was never run is simply absent from the list, so
    a partial tree draws from fewer studies rather than failing.
    """
    out: list[tuple[str, list[Path]]] = []
    if cal == "2y":
        # The benchmarks are scored on their own windows only; on any other calibration
        # they forecast a different holdout year and must not be drawn beside these.
        pareto = benchmarks.STUDIES_BASE / benchmarks.pareto_suite_name(panel) / "ParetoNBD"
        out.append(("Pareto/NBD", [pareto] if (pareto / "Predictions").is_dir() else []))
        out.append(("ValendinLSTM",
                    [benchmarks.forecast_path(panel, r).parents[1]
                     for r in range(benchmarks.N_REPLICATIONS)
                     if benchmarks.forecast_path(panel, r).exists()]))
    for enc in ENCODINGS:
        out.append((f"LSTM + ar_{enc}",
                    [ar.forecast_path("lstm", enc, panel, r, cal).parents[1]
                     for r in range(ar.REPLICATIONS["lstm"])
                     if ar.forecast_path("lstm", enc, panel, r, cal).exists()]))
    return out


def best_by_mape(model_dirs: list[Path], actual: np.ndarray,
                 ref_ids: np.ndarray) -> tuple[Path, dict[str, float], int] | None:
    """The arm's lowest-MAPE study: its directory, its metrics, and how many were searched.

    Every study is scored through `run_real_panel_benchmarks.score`, which checks that the
    stored forecast describes the rebuilt cohort before scoring it, so a suite left over
    from a different cohort raises here rather than being silently plotted.
    """
    scored = [(benchmarks.score(d, actual, ref_ids), d) for d in model_dirs]
    if not scored:
        return None
    metrics, best = min(scored, key=lambda pair: pair[0]["mape_aggregate"])
    return best, metrics, len(scored)


def weekly_aggregate(model_dir: Path) -> np.ndarray:
    """One stored forecast summed over customers: mean simulated transactions per week."""
    values, _ = load_model_predictions(model_dir, study=1)      # (N, T_HOLD)
    return values.sum(axis=0)


# ---------------------------------------------------------------------------
# One panel
# ---------------------------------------------------------------------------


def panel_figure(panel: str, cal: str) -> None:
    """Read, score, draw and write one panel's figure and its sidecar table."""
    # The cohort and the holdout do not depend on the encoding, so any one rebuilds them.
    data = ar.build_data(ENCODINGS[0], panel, cal)
    actual = holdout_actuals(data)                              # (N, T_HOLD)
    ref_ids = np.asarray(data["ids"])
    weeks = pd.date_range(ar.CALIBRATIONS[cal][panel]["holdout_start"],
                          periods=actual.shape[1], freq="7D")

    curves: dict[str, np.ndarray] = {}
    table: list[dict] = []
    for label, model_dirs in arms(panel, cal):
        chosen = best_by_mape(model_dirs, actual, ref_ids)
        if chosen is None:
            print(f"  {panel} [{cal}]: no studies for {label}", flush=True)
            continue
        best, metrics, n = chosen
        curves[label] = weekly_aggregate(best)
        table.append({"arm": label, "studies": n, "best": best.parent.name,
                      **{m: metrics[m] for m in benchmarks.METRICS}})
        print(f"  {panel} [{cal}]: {label:<26} best of {n:>3} — "
              f"MAPE {metrics['mape_aggregate']:.1f}  ({best.parent.name})", flush=True)

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    # The actuals are not a series but the thing being matched, so they wear ink and a
    # fill rather than a categorical hue.
    ax.fill_between(weeks, actual.sum(axis=0), color=INK, alpha=0.10, linewidth=0)
    ax.plot(weeks, actual.sum(axis=0), color=INK, linewidth=1.8, label="actual", zorder=3)
    for row in table:
        label = row["arm"]
        ax.plot(weeks, curves[label], color=SERIES[label], linewidth=1.4, zorder=2,
                label=f"{label}  (MAPE {row['mape_aggregate']:.1f})")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    ax.set_ylabel("transactions / week")
    ax.set_xlabel(f"holdout week ({weeks[0]:%Y-%m-%d} to {weeks[-1]:%Y-%m-%d})")
    ax.grid(axis="y", color=INK_MUTED, alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{panel} — weekly holdout volume, actual vs each arm's best-MAPE study\n"
        f"{len(ref_ids)} customers · {int(actual.sum())} holdout transactions · "
        f"{'benchmark windows' if cal == '2y' else 'three-year calibration'}",
        loc="left")
    # The legend sits outside the axes under constrained_layout, so it is packed
    # against the figure edge rather than floating on a hand-set offset.
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3)

    OUT.mkdir(exist_ok=True)
    stem = OUT / f"real_panel_weekly__{cal}__{panel}"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    # The table view: every plotted curve, week by week, beside the selection it came from.
    pd.DataFrame({"week_start": weeks, "actual": actual.sum(axis=0),
                  **curves}).to_csv(stem.with_suffix(".csv"), index=False)
    pd.DataFrame(table).to_csv(f"{stem}__selection.csv", index=False)
    print(f"  wrote {stem}.png\n", flush=True)


def main() -> None:
    # Both campaigns: the benchmark windows, then the three-year calibration.
    for cal in ("2y", "3y"):
        for panel in ar.CALIBRATIONS[cal]:
            panel_figure(panel, cal)


if __name__ == "__main__":
    main()
