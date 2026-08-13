"""Thesis figures for the Pareto/NBD synthetic-grid study.

Figure 1 — the neural models have no death state: aggregate bias grows with the
           churn rate, and their simulated weekly volume fails to decay.
Figure 2 — the compensating strength: they track within-year seasonal shape,
           which Pareto/NBD structurally cannot.

Both measurements live in `panelclv.studies.synthetic_grid`, not here: the arithmetic
below the figures is what the thesis reports, so it is exposed for the notebooks rather
than kept as a second copy that could drift from them. This script only plots it.

Run from the repo root:
    PYTHONPATH=src python scripts/make_grid_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from panelclv.studies.pareto_nbd_grid import collect_grid_results
from panelclv.studies.synthetic_grid import dead_customer_mass, shape_correlation

GEN = Path("Datasets/Synthetic/pnbd_study_4x4x10_20260716-154143")
TRAIN = Path("Studies/pnbd_study_4x4x10_20260716-154143")
OUT = Path("figures")

# Categorical identity is fixed per model and never cycled or re-assigned when a
# panel shows a subset. Slots 1-3 of the reference palette (blue / green / magenta).
MODELS = ["LSTM", "Transformer", "ParetoNBD_MLE"]
# This grid was trained against the frequentist-MLE Pareto/NBD (now retired to
# archive/), so the label names the estimator rather than just the model family.
LABEL = {"LSTM": "LSTM", "Transformer": "Transformer",
         "ParetoNBD_MLE": "Pareto/NBD (MLE)"}
COLOR = {"LSTM": "#2a78d6", "Transformer": "#008300", "ParetoNBD_MLE": "#e87ba4"}
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8a85"

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "medium",
    "axes.labelsize": 9, "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def style(ax, ylabel=None, title=None):
    """Recessive grid + axis furniture, applied identically to every panel."""
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.7)
    ax.set_axisbelow(True)                      # grid behind the marks, never over
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    if title:
        ax.set_title(title, color=INK, loc="left", pad=8)


def end_label(ax, x, y, text, color):
    """Direct label at the line end. The text wears ink; the line end carries hue.

    Three of the reference palette's light-mode slots sit under 3:1 contrast, so
    the palette's relief rule requires visible direct labels — this is that relief.
    """
    ax.plot([x], [y], "o", color=color, ms=6, zorder=5)
    ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8.5, color=INK)


# ---------------------------------------------------------------------------
# Figure 1 — no death state
# ---------------------------------------------------------------------------

def figure_1(results, dead):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # (a) Aggregate bias against churn. One line per model; churn is ordinal with
    # four levels, so a line reads the trend better than four grouped bars.
    by_churn = (results.pivot_table(index="churn_rate", columns="model",
                                    values="bias_percent", aggfunc="mean"))
    ax1.axhline(0, color=INK_MUTED, lw=0.9, zorder=1)          # unbiased reference
    for m in MODELS:
        ax1.plot(by_churn.index, by_churn[m], color=COLOR[m], lw=2, marker="o",
                 ms=5.5, mec="white", mew=1.2, zorder=3, label=LABEL[m])
        end_label(ax1, by_churn.index[-1], by_churn[m].iloc[-1], LABEL[m], COLOR[m])
    ax1.set_xlim(0.15, 1.02)                                    # room for end labels
    ax1.set_xticks(by_churn.index)
    ax1.set_xlabel("Churn rate", color=INK_2)
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    style(ax1, "Aggregate bias", "a. Over-prediction grows with churn")

    # (b) The mechanism, over the same axis as (a) so the two panels read together:
    # the extra volume is spent on customers who never purchase again. Both panels
    # average the same long tables the package returns, so both read churn off the
    # generator's own coordinate rather than parsing it back out of a folder name.
    by_churn_dead = dead.pivot_table(index="churn_rate", columns="model",
                                     values="dead_customer_mass", aggfunc="mean")
    xs = by_churn_dead.index
    for m in MODELS:
        ax2.plot(xs, by_churn_dead[m], color=COLOR[m], lw=2, marker="o", ms=5.5,
                 mec="white", mew=1.2, zorder=3, label=LABEL[m])
        end_label(ax2, xs[-1], by_churn_dead[m].iloc[-1], LABEL[m], COLOR[m])
    ax2.set_xlim(0.15, 1.02)
    ax2.set_xticks(xs)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Churn rate", color=INK_2)
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    style(ax2, "Share of predicted volume",
          "b. The excess is spent on customers who never return")
    # Sits in the empty band below the Pareto/NBD line, clear of every mark.
    ax2.text(0.02, 0.04, "Customers with zero holdout purchases · mean of 160 datasets",
             transform=ax2.transAxes, fontsize=7.5, color=INK_MUTED)

    fig.tight_layout(pad=1.4)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — seasonal shape
# ---------------------------------------------------------------------------

def figure_2(corr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)

    for ax, key, xlabel, title in (
        (ax1, "churn_rate", "Churn rate", "a. Shape tracking degrades with churn"),
        (ax2, "mean_transaction_rate", "Mean transaction rate",
         "b. Shape tracking improves with volume"),
    ):
        agg = corr.pivot_table(index=key, columns="model",
                               values="shape_correlation", aggfunc="mean")
        xs = np.arange(len(agg.index))                 # evenly spaced ordinal levels
        ax.axhline(0, color=INK_MUTED, lw=0.9, zorder=1)
        for m in MODELS:
            ax.plot(xs, agg[m], color=COLOR[m], lw=2, marker="o", ms=5.5,
                    mec="white", mew=1.2, zorder=3, label=LABEL[m])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in agg.index])
        ax.set_xlabel(xlabel, color=INK_2)
        ax.set_ylim(-0.15, 1.0)
        style(ax, None, title)

    ax1.set_ylabel("Correlation with actual weekly totals", color=INK_2)
    ax1.legend(loc="upper right", handlelength=1.6)
    ax2.text(0.02, 0.04, "Pareto/NBD has no seasonal component by construction",
             transform=ax2.transAxes, fontsize=7.5, color=INK_MUTED)

    fig.tight_layout(pad=1.4)
    return fig


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    # Only `bias_percent` is plotted (figure 1a). Asking for just that keeps the
    # script readable off grids archived before `mape_aggregate_style` was renamed —
    # `collect_grid_results`' default metric set now names the current column.
    results = collect_grid_results(GEN, TRAIN, metrics=("bias_percent",))
    corr = shape_correlation(GEN, TRAIN)
    dead = dead_customer_mass(GEN, TRAIN)
    # Source data for both figures, so every plotted number has a table view. Both are
    # the package's long tables — one row per (dataset, model), which carries the
    # replicate labels the earlier wide layout dropped.
    corr.to_csv(OUT / "shape_correlation.csv", index=False)
    dead.to_csv(OUT / "dead_customer_mass.csv", index=False)

    for name, fig in (("fig1_no_death_state", figure_1(results, dead)),
                      ("fig2_seasonal_shape", figure_2(corr))):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
        print("wrote", OUT / f"{name}.png")
