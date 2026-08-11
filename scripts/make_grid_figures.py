"""Thesis figures for the Pareto/NBD synthetic-grid study.

Figure 1 — the neural models have no death state: aggregate bias grows with the
           churn rate, and their simulated weekly volume fails to decay.
Figure 2 — the compensating strength: they track within-year seasonal shape,
           which Pareto/NBD structurally cannot.

Run from the repo root:
    PYTHONPATH=src python scripts/make_grid_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelclv.data_preparation import pareto_simulation as ps
from panelclv.studies.pnbd_grid import collect_grid_results

GEN = Path("Datasets/Synthetic/pnbd_study_4x4x10_20260716-154143")
TRAIN = Path("Studies/pnbd_study_4x4x10_20260716-154143")
OUT = Path("figures")
HOLDOUT_YEAR = 2001
REPLICATES = [f"Dataset_{i}" for i in range(1, 11)]

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
# Data
# ---------------------------------------------------------------------------

def dead_customer_mass():
    """Share of each model's predicted holdout volume assigned to customers who
    made no holdout purchase — the direct signature of a missing death mechanism.

    Note this is a *relative* diagnostic, not an error rate: a customer who is
    still alive but has a low transaction rate can legitimately record zero
    purchases in 52 weeks, so the correct share is well above zero and is not
    observable. Pareto/NBD — near-unbiased in aggregate across this grid — is
    therefore the reference level the neural models are read against.
    """
    rows = []
    for combo in sorted(p.name for p in GEN.iterdir() if p.name.startswith("Dataset_")):
        rate, churn = (int(v) for v in combo.split("_")[1:3])
        for ds in REPLICATES:
            panel, _, _ = ps.load_pnbd_dataset(GEN, combo, ds)
            actual_per_customer = (panel[panel["year"] == HOLDOUT_YEAR]
                                   .groupby("Id")["Transactions"].sum())
            silent = set(actual_per_customer.index[actual_per_customer.values == 0])
            rec = {"rate": rate, "churn": churn}
            for m in MODELS:
                pred = pd.read_csv(TRAIN / f"{combo}__{ds}" / m / "Predictions" / "Prediction_1.csv")
                total = pred.set_index("Id").sum(axis=1)
                rec[m] = total[total.index.isin(silent)].sum() / total.sum()
            rows.append(rec)
    return pd.DataFrame(rows)


def shape_correlation():
    """Per-dataset correlation between predicted and actual weekly holdout totals.

    This isolates *shape* from *level*: correlation is invariant to a multiplicative
    over-prediction, so a model can score 1.0 here while being 300% biased. That is
    exactly the separation Figure 2 needs to make.
    """
    rows = []
    for combo in sorted({p.name for p in GEN.iterdir() if p.name.startswith("Dataset_")}):
        rate, churn = (int(x) for x in combo.split("_")[1:3])
        for ds in REPLICATES:
            panel, _, _ = ps.load_pnbd_dataset(GEN, combo, ds)
            act = (panel[panel["year"] == HOLDOUT_YEAR]
                   .groupby("week")["Transactions"].sum().sort_index().values)
            rec = {"rate": rate, "churn": churn}
            for m in MODELS:
                pred = pd.read_csv(TRAIN / f"{combo}__{ds}" / m / "Predictions" / "Prediction_1.csv")
                v = pred.drop(columns=["Id"]).sum(axis=0).values[: len(act)]
                a = act[: len(v)]
                rec[m] = np.corrcoef(a, v)[0, 1] if v.std() > 0 and a.std() > 0 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


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
    # the extra volume is spent on customers who never purchase again.
    by_churn_dead = dead.groupby("churn")[MODELS].mean()
    xs = by_churn_dead.index / 100
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
        (ax1, "churn", "Churn rate", "a. Shape tracking degrades with churn"),
        (ax2, "rate", "Mean transaction rate", "b. Shape tracking improves with volume"),
    ):
        agg = corr.groupby(key)[MODELS].mean()
        xs = np.arange(len(agg.index))                 # evenly spaced ordinal levels
        ax.axhline(0, color=INK_MUTED, lw=0.9, zorder=1)
        for m in MODELS:
            ax.plot(xs, agg[m], color=COLOR[m], lw=2, marker="o", ms=5.5,
                    mec="white", mew=1.2, zorder=3, label=LABEL[m])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v/100:g}" for v in agg.index])
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
    results = collect_grid_results(GEN)
    corr = shape_correlation()
    dead = dead_customer_mass()
    # Source data for both figures, so every plotted number has a table view.
    corr.to_csv(OUT / "shape_correlation.csv", index=False)
    dead.to_csv(OUT / "dead_customer_mass.csv", index=False)

    for name, fig in (("fig1_no_death_state", figure_1(results, dead)),
                      ("fig2_seasonal_shape", figure_2(corr))):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
        print("wrote", OUT / f"{name}.png")
