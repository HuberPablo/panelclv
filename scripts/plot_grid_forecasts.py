"""Weekly-aggregate holdout forecasts for a synthetic grid, per dataset and per model.

The grid's tables say how far each model's total volume is from the truth; they do not
say *what shape* the miss has. This script draws the shape: for every panel in the grid,
the actual weekly transaction volume over the holdout year and each model's simulated
volume on the same axes.

Two figures, both from the same curves:

    --overview    one 4x4 figure, an axes per (rate, churn) cell, curves averaged over
                  the cell's ten replicate panels. The grid at a glance.
    --per-cell    sixteen figures, one per cell, each a 2x5 of the cell's ten replicate
                  panels drawn individually. Every dataset in the grid appears exactly
                  once across the set.

Curves are rebuilt from what the suites stored, not re-simulated: each
``Predictions/Prediction_<k>.csv`` is a wide per-customer x holdout-week table of mean
simulated counts (`panelclv.predictions`), so summing it over customers is the model's
weekly aggregate. Actuals come from the generated panel, restricted to the customer ids
the suite actually forecast (``require_calibration_activity`` drops the rest) and clipped
at ``clip_target_upper`` so the comparison is against the same target the models were
trained on. Summing both and taking the relative difference reproduces the
``bias_percent`` in each suite's ``results.csv`` exactly, which is the check that this
reader agrees with the scoring authority.

Run from the repo root:

    PYTHONPATH=src python scripts/plot_grid_forecasts.py --grid seasonal_4x4x10 --overview
    PYTHONPATH=src python scripts/plot_grid_forecasts.py --grid seasonal_4x4x10 --per-cell

``--configs`` picks which trees to overlay, as ``Model[:arm]`` (repeatable). The default
is the benchmark plus the best arm of each architecture; more than about four curves per
axes stops being readable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grids import available_grids, load_grid            # noqa: E402
from panelclv.studies.pareto_nbd_grid import list_pnbd_datasets   # noqa: E402

OUT = Path("figures")

# Categorical identity is fixed per curve and never cycled: slots 1-3 of the reference
# palette, plus ink for the actuals, which are not a series but the thing being matched.
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

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
# Reading one dataset's curves
# ---------------------------------------------------------------------------


def _forecast_curve(suite: Path, model_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(ids, weekly aggregate) for one suite's stored forecast, or None if absent.

    A suite holds one prediction file per study; this grid runs one study per suite, so
    there is normally exactly one. Several are averaged, which is the same across-studies
    mean ``aggregate_suite_predictions`` writes.
    """
    files = sorted((suite / model_name / "Predictions").glob("Prediction_*.csv"))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    ids = frames[0]["Id"].to_numpy()
    stack = np.stack([f.drop(columns="Id").to_numpy() for f in frames])   # (S, N, T_HOLD)
    return ids, stack.mean(axis=0).sum(axis=0)                           # (T_HOLD,)


def _actual_curve(panel_path: Path, ids: np.ndarray, holdout_year: int, clip: int) -> np.ndarray:
    """Weekly aggregate of the true holdout counts, over exactly the forecast customers.

    Clipped at the same ``clip_target_upper`` the models were trained under: their softmax
    cannot emit a count above it, so scoring against unclipped actuals would charge them
    for a class that does not exist.
    """
    panel = pd.read_csv(panel_path, usecols=["Id", "year", "week", "Transactions"])
    hold = panel[panel.year == holdout_year]
    wide = (hold.pivot_table(index="Id", columns="week", values="Transactions")
                .reindex(ids))                                            # (N, T_HOLD)
    return np.clip(wide.to_numpy(), 0, clip).sum(axis=0)


def collect_curves(spec, configs: list[tuple[str, str | None]]) -> dict:
    """Every dataset's actual and per-config forecast curve, keyed by (combo, dataset).

    Returns ``{(combo, dataset): {"actual": arr, "coords": (rate, churn), <label>: arr}}``.
    A config with no suite for a dataset is simply absent from that entry, so a partial
    tree draws fewer curves rather than failing.
    """
    grid = list_pnbd_datasets(spec.dataset_dir)
    clip = spec.panel.clip_target_upper
    holdout_year = int(spec.panel.holdout_start[:4])

    out: dict = {}
    for n, g in enumerate(grid.itertuples(index=False), 1):
        entry = {"coords": (g.mean_transaction_rate, g.churn_rate)}
        ids = None
        for model_name, arm in configs:
            suite = spec.train_base(model_name, arm) / f"{g.combo}__{g.dataset}"
            got = _forecast_curve(suite, model_name)
            if got is None:
                continue
            sid, curve = got
            # Every model in a suite forecasts the same cohort, so the first set of ids
            # seen fixes the rows the actuals are summed over.
            if ids is None:
                ids = sid
            entry[_label(model_name, arm)] = curve
        if ids is None:
            continue
        entry["actual"] = _actual_curve(Path(g.panel_path), ids, holdout_year, clip)
        out[(g.combo, g.dataset)] = entry
        if n % 20 == 0 or n == len(grid):
            print(f"  {n}/{len(grid)} panels read", flush=True)
    print()
    return out


def _label(model_name: str, arm: str | None) -> str:
    return model_name if arm is None else f"{model_name} · {arm}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _draw(ax, entry: dict, labels: list[str], colors: dict, weeks: np.ndarray) -> None:
    """One axes: the actual curve in ink, each model's forecast over it."""
    ax.fill_between(weeks, entry["actual"], color=INK, alpha=0.10, linewidth=0)
    ax.plot(weeks, entry["actual"], color=INK, linewidth=1.4, label="actual", zorder=3)
    for lab in labels:
        if lab in entry:
            ax.plot(weeks, entry[lab], color=colors[lab], linewidth=1.2, label=lab, zorder=2)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)


def _mean_entry(entries: list[dict], labels: list[str]) -> dict:
    """Average a cell's replicate panels curve-wise, skipping curves a replicate lacks."""
    out = {"actual": np.mean([e["actual"] for e in entries], axis=0)}
    for lab in labels:
        have = [e[lab] for e in entries if lab in e]
        if have:
            out[lab] = np.mean(have, axis=0)
    return out


def figure_overview(curves: dict, labels: list[str], colors: dict, grid_name: str) -> Path:
    """One 4x4 figure: an axes per cell, curves averaged over the cell's replicates."""
    rates = sorted({c["coords"][0] for c in curves.values()})
    churns = sorted({c["coords"][1] for c in curves.values()})
    weeks = np.arange(len(next(iter(curves.values()))["actual"]))

    fig, axes = plt.subplots(len(rates), len(churns), figsize=(13, 10.5),
                             sharex=True, constrained_layout=True)
    for i, rate in enumerate(rates):
        for j, churn in enumerate(churns):
            ax = axes[i, j]
            cell = [e for e in curves.values() if e["coords"] == (rate, churn)]
            if not cell:
                ax.set_visible(False)
                continue
            _draw(ax, _mean_entry(cell, labels), labels, colors, weeks)
            ax.set_title(f"rate {rate:g} · churn {churn:g}   (n={len(cell)})", loc="left")
            if j == 0:
                ax.set_ylabel("transactions / week")
            if i == len(rates) - 1:
                ax.set_xlabel("holdout week")

    handles, hlabels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, hlabels, loc="outside lower center", ncol=len(hlabels))
    fig.suptitle(
        f"{grid_name} — weekly holdout volume, actual vs simulated\n"
        "mean over each cell's replicate panels; rows = mean transaction rate, "
        "columns = churn rate",
        ha="left", x=0.01, fontsize=11,
    )
    path = OUT / f"{grid_name}__forecast_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figures_per_cell(curves: dict, labels: list[str], colors: dict, grid_name: str) -> list[Path]:
    """One figure per cell, a small multiple per replicate panel — every dataset drawn."""
    cells: dict = {}
    for (combo, dataset), entry in curves.items():
        cells.setdefault(entry["coords"], []).append((combo, dataset, entry))
    weeks = np.arange(len(next(iter(curves.values()))["actual"]))

    written = []
    for (rate, churn), members in sorted(cells.items()):
        members.sort(key=lambda m: m[1])
        ncol = 5
        nrow = int(np.ceil(len(members) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow),
                                 sharex=True, constrained_layout=True)
        flat = np.atleast_1d(axes).ravel()
        for ax, (combo, dataset, entry) in zip(flat, members):
            _draw(ax, entry, labels, colors, weeks)
            ax.set_title(dataset, loc="left")
        for ax in flat[len(members):]:
            ax.set_visible(False)
        for ax in flat[:ncol]:
            ax.set_ylabel("transactions / week")
        for ax in flat[len(members) - ncol:len(members)]:
            ax.set_xlabel("holdout week")

        handles, hlabels = flat[0].get_legend_handles_labels()
        fig.legend(handles, hlabels, loc="outside lower center", ncol=len(hlabels))
        fig.suptitle(f"{grid_name} — {combo}: rate {rate:g}, churn {churn:g} — "
                     f"{len(members)} replicate panels", ha="left", x=0.01, fontsize=11)
        path = OUT / f"{grid_name}__cell_{combo}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------


def parse_config(spec, text: str) -> tuple[str, str | None]:
    """``"Model"`` or ``"Model:arm"`` -> the (model, arm) pair naming one trained tree."""
    model_name, _, arm = text.partition(":")
    known = {m.name for m in spec.models}
    if model_name not in known:
        raise SystemExit(f"unknown model {model_name!r}; grid declares {sorted(known)}")
    arm = arm or None
    if arm is not None and arm not in {a.name for a in spec.arms}:
        raise SystemExit(f"unknown arm {arm!r}; grid declares {[a.name for a in spec.arms]}")
    return model_name, arm


def default_configs(spec) -> list[str]:
    """The benchmark plus the best arm of each neural model, where those trees exist."""
    best = "ar_bounded-no_cluster-valendin"
    wanted = ["ParetoNBD", f"LSTM:{best}", f"Transformer:{best}"]
    return [w for w in wanted
            if spec.train_base(*parse_config(spec, w)).is_dir()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", required=True, help=f"one of: {', '.join(available_grids())}")
    ap.add_argument("--configs", action="append", metavar="MODEL[:ARM]",
                    help="a trained tree to overlay; repeatable (default: benchmark + "
                         "best arm per architecture)")
    ap.add_argument("--overview", action="store_true", help="the 4x4 cell figure")
    ap.add_argument("--per-cell", action="store_true",
                    help="one figure per cell, every replicate panel drawn")
    args = ap.parse_args()

    if not (args.overview or args.per_cell):
        raise SystemExit("nothing to draw: pass --overview, --per-cell, or both")

    spec = load_grid(args.grid)
    configs = [parse_config(spec, c) for c in (args.configs or default_configs(spec))]
    labels = [_label(m, a) for m, a in configs]
    colors = {lab: SLOTS[i % len(SLOTS)] for i, lab in enumerate(labels)}

    print(f"{spec.name}: overlaying {len(labels)} tree(s)")
    for lab in labels:
        print(f"  - {lab}")
    curves = collect_curves(spec, configs)
    if not curves:
        raise SystemExit("no forecasts found; has the grid been trained?")
    print(f"{len(curves)} panels with forecasts")

    OUT.mkdir(exist_ok=True)
    if args.overview:
        print(f"wrote {figure_overview(curves, labels, colors, spec.name)}")
    if args.per_cell:
        for p in figures_per_cell(curves, labels, colors, spec.name):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
