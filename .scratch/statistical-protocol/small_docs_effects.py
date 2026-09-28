"""Rescore four smaller docs under the statistical protocol (docs/statistical-protocol.md).

Every comparative claim in the docs below is re-derived here from the stored per-run
numbers and put through `panelclv.evaluation.effects.effect`: Δ = mean(B) − mean(A),
a 95% percentile-bootstrap interval from 10,000 resamples, supported iff 0 is outside
it. Nothing here re-implements a bootstrap. The docs quote exactly what this prints.

    docs/absorbing-death-state.md        §3.3, §4.2-§4.3, §4.5
    docs/loss-functions.md               §6 R1 and R2 (CDNOW loss ablation)
    docs/pareto-nbd-cdnow-replication.md §9(b) window shift
    docs/p-slstm.md                      §10-§11 (run 1 and run 2)

Pairing decisions, one per data source (protocol §2):

* rollout_feedback CSVs -- every readout of study i is the SAME trained weights read
  a different way, so the study is a shared unit: paired=True, n = 8.
* loss_ablation_cdnow -- arm j of every loss runs under seed base_seed + j, but that
  seed drives only the Optuna TPE sampler and the Monte Carlo forecast
  (studies/config.py, `base_seed`); training is unseeded, and the two arms optimise
  different objectives, so the sampler's trajectories part after its random start-up
  trials. Nothing the two arms share is a unit the result is measured on: paired=False.
* window_shift_results.csv -- the Pareto/NBD MCMC fit is seeded (benchmarks/pareto_nbd.py
  SeedSequence(seed)), and both arms run the same seed on the same cohort, differing only
  in the sufficient statistics handed to the sampler. Same chain randomness on both
  sides: paired=True, n = 3.
* p-slstm comparison-results.json -- seed j calls torch.manual_seed(j) for both the
  LSTM and P-sLSTM, but two different architectures consume that stream differently, so
  "seed j" is not a shared unit between them: paired=False. Pareto/NBD is its own
  independent condition (n = 3 seeded fits).
* A single statistic tested against a fixed reference (P-sLSTM run 1 against the
  deterministic class-prior CE; a ρ against 0) is the paired case against a constant.

Run from the repo root:
    PYTHONPATH=src python .scratch/statistical-protocol/small_docs_effects.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from panelclv.evaluation.effects import effect, table

ROOT = Path(__file__).resolve().parents[2]


def show(title: str, effects: list, label: str = "comparison") -> None:
    """Print one block of effects as the protocol's markdown table, with the metric."""
    print(f"\n### {title}\n")
    print(table(effects, label=label))


def named(e, name: str):
    """Relabel an Effect's panel column so the printed table names the comparison."""
    e.panel = f"{name} [{e.metric}]"
    return e


# ---------------------------------------------------------------------------------
# 1. absorbing-death-state.md
# ---------------------------------------------------------------------------------

def absorbing_death_state() -> None:
    print("\n## absorbing-death-state.md")

    # §3.3 -- recompute the archived replications' per-customer Spearman against the
    # true electronics holdout (2001, 52 weeks, unclipped counts summed per customer).
    panel = pd.read_csv(ROOT / "Datasets/Dataset_clean/electronics_customer_week_panel.csv")
    actual = panel[panel.year == 2001].groupby("Id").Transactions.sum()

    def rhos(prefix: str) -> list[float]:
        out = []
        for d in sorted((ROOT / "FOR_ANALYSIS").glob(f"{prefix}*")):
            pred = pd.read_csv(d / "predictions.csv").set_index("Id").sum(axis=1)
            out.append(spearmanr(pred, actual.reindex(pred.index)).statistic)
        return out

    lstm, trf, pnbd = rhos("lstm_"), rhos("transformer_"), rhos("pareto_hb_")
    print("\n§3.3 rho per replication: LSTM", np.round(lstm, 4), "Transformer",
          np.round(trf, 4), "Pareto/NBD", np.round(pnbd, 4))
    # The three Pareto/NBD archives are byte-identical: one fit, stored three times.
    # So Pareto/NBD is n = 1. A bootstrap cannot resample one value (scipy refuses), so
    # the fit enters as a fixed reference -- the paired case against a constant. The
    # interval then carries only the neural side's spread; Pareto/NBD's own refit
    # spread is unmeasured here, which the doc states beside every such row.
    def ref(values):
        return np.full(len(values), pnbd[0])
    show("§3.3 archived electronics, rho (Pareto/NBD n = 1)", [
        named(effect(lstm, np.zeros(3), paired=True, metric="spearman", panel=""),
              "LSTM vs 0"),
        named(effect(trf, np.zeros(3), paired=True, metric="spearman", panel=""),
              "Transformer vs 0"),
        named(effect(lstm, ref(lstm), paired=True, metric="spearman", panel=""),
              "LSTM vs Pareto/NBD fit (fixed ref.)"),
        named(effect(trf, ref(trf), paired=True, metric="spearman", panel=""),
              "Transformer vs Pareto/NBD fit (fixed ref.)"),
    ])

    # §4.2-§4.3 -- the 2x2, same weights per study, paired on study (n = 8).
    sq = pd.read_csv(ROOT / "Studies/rollout_feedback/square_electronics_ar_bounded_32.csv")
    w = {m: sq.pivot(index="study", columns="readout", values=m)
         for m in ("rho", "bias_percent", "mape_aggregate")}

    def pe(metric, b, a, name):
        return named(effect(w[metric][b], w[metric][a], paired=True, metric=metric,
                            panel=""), name)

    show("§4.2-§4.3 the 2x2 (paired on study, n = 8)", [
        pe("rho", "ar_true", "rollout", "AR true vs sampled, at sampled target"),
        pe("rho", "teacher", "tgt_true", "AR true vs sampled, at true target"),
        pe("rho", "tgt_true", "rollout", "target true vs sampled, at sampled AR"),
        pe("rho", "teacher", "ar_true", "target true vs sampled, at true AR"),
        pe("rho", "teacher", "rollout", "teacher vs rollout"),
        pe("bias_percent", "teacher", "rollout", "teacher vs rollout"),
        pe("mape_aggregate", "teacher", "rollout", "teacher vs rollout"),
        pe("bias_percent", "ar_true", "rollout", "AR true vs rollout"),
        pe("mape_aggregate", "ar_true", "rollout", "AR true vs rollout"),
    ])
    # The same readouts against the archived Pareto/NBD fit (n = 1, from §3.3), as a
    # fixed reference: one number whose refit spread this study never measured.
    show("§4.2 readouts vs archived Pareto/NBD rho (fixed reference)", [
        named(effect(w["rho"]["teacher"], ref(w["rho"]["teacher"]), paired=True, metric="spearman",
                     panel=""), "teacher vs Pareto/NBD fit (fixed ref.)"),
        named(effect(w["rho"]["rollout"], ref(w["rho"]["rollout"]), paired=True, metric="spearman",
                     panel=""), "rollout vs Pareto/NBD fit (fixed ref.)"),
    ], label="comparison (B vs A)")

    # §4.3 -- the unbounded arm: does teacher forcing rescue ranking, and is its bias
    # worse teacher-forced than rolled out? Same weights per study, paired (n = 8).
    ub = pd.read_csv(ROOT / "Studies/rollout_feedback/threeway_electronics_ar_unbounded.csv")
    show("§4.3 ar_unbounded arm (paired on study, n = 8)", [
        named(effect(*(ub.pivot(index="study", columns="readout", values=m)[c]
                       for c in ("teacher", "rollout")), paired=True, metric=m, panel=""),
              "teacher vs rollout")
        for m in ("rho", "bias_percent")
    ])

    # §4.5 -- the latch, same weights per study, paired on study (n = 8).
    la = pd.read_csv(ROOT / "Studies/rollout_feedback/latch_electronics_ar_bounded_32.csv")
    rows = []
    for metric in ("rho", "mape_aggregate", "bias_percent"):
        p = la.pivot(index="study", columns="readout", values=metric)
        for L in ("latch_52", "latch_26", "latch_8"):
            rows.append(named(effect(p[L], p["rollout"], paired=True, metric=metric,
                                     panel=""), f"{L} vs rollout"))
    p = la.pivot(index="study", columns="readout", values="rho")
    rows.append(named(effect(p["latch_26"], p["latch_52"], paired=True, metric="rho",
                             panel=""), "latch_26 vs latch_52"))
    rows.append(named(effect(p["latch_26"], p["latch_8"], paired=True, metric="rho",
                             panel=""), "latch_26 vs latch_8"))
    show("§4.5 latch (paired on study, n = 8)", rows)


# ---------------------------------------------------------------------------------
# 2. loss-functions.md
# ---------------------------------------------------------------------------------

def loss_functions() -> None:
    print("\n## loss-functions.md")
    r = pd.read_csv(ROOT / "Studies/loss_ablation_cdnow/results.csv")
    r["abs_bias"] = r.bias_percent.abs()
    rows = []
    for arm in ("LSTM_emd", "LSTM_ce_emd"):
        for metric in ("mape_aggregate", "abs_bias", "bias_percent"):
            b = r[r.model == arm][metric].to_numpy()
            a = r[r.model == "LSTM_ce"][metric].to_numpy()
            rows.append(named(effect(b, a, paired=False, metric=metric, panel="cdnow"),
                              f"{arm} vs LSTM_ce"))
    # Signed bias as a direction claim: each arm against 0.
    for arm in ("LSTM_ce", "LSTM_emd", "LSTM_ce_emd"):
        b = r[r.model == arm].bias_percent.to_numpy()
        rows.append(named(effect(b, np.zeros_like(b), paired=True,
                                 metric="bias_percent", panel="cdnow"), f"{arm} vs 0"))
    show("CDNOW loss ablation (independent, n = 10 / 10)", rows)
    # Description only (protocol §3): per-seed counts, not evidence.
    for arm in ("LSTM_emd", "LSTM_ce_emd"):
        m = r.pivot(index="seed", columns="model", values="mape_aggregate")
        print(f"description: {arm} MAPE above LSTM_ce in "
              f"{int((m[arm] > m['LSTM_ce']).sum())} of {len(m)} seeds")


# ---------------------------------------------------------------------------------
# 3. pareto-nbd-cdnow-replication.md
# ---------------------------------------------------------------------------------

def pnbd_window_shift() -> None:
    print("\n## pareto-nbd-cdnow-replication.md")
    ws = pd.read_csv(ROOT / ".scratch/pnbd-cdnow-replication/window_shift_results.csv")
    rows = []
    for panel, arms in (("cdnow", ("T+0.5", "T+1", "no_collapse")),
                        ("electronics", ("T+1",))):
        d = ws[ws.panel == panel]
        for metric in ("bias", "spearman", "mape"):
            p = d.pivot(index="seed", columns="mode", values=metric)
            for arm in arms:
                rows.append(named(effect(p[arm], p["baseline"], paired=True,
                                         metric=metric, panel=panel),
                                  f"{panel} {arm} vs baseline"))
    show("§9(b) window shift (paired on seed, n = 3)", rows)

    # §7 -- config A (package sampler, 5 seeded fits) against config D (MLE at the paper's
    # estimates, deterministic, so one number: a fixed reference), and A's bias against 0.
    rep = json.loads((ROOT / ".scratch/pnbd-cdnow-replication/results.json").read_text())
    a_bias = np.array([r["bias"] for r in rep["runs"]["A"]])
    d_bias = rep["D_paper"]["bias"]
    show("§7 config A (n = 5) vs D (fixed reference) and vs 0", [
        named(effect(a_bias, np.full(len(a_bias), d_bias), paired=True, metric="bias",
                     panel="cdnow"), "A vs D (fixed ref.)"),
        named(effect(a_bias, np.zeros(len(a_bias)), paired=True, metric="bias",
                     panel="cdnow"), "A vs 0"),
    ])


# ---------------------------------------------------------------------------------
# 4. p-slstm.md
# ---------------------------------------------------------------------------------

def p_slstm() -> None:
    print("\n## p-slstm.md")
    # Run 1: best val CE per seed (training-test-p-slstm.md's per-seed table) against
    # the training-window class prior, which is one deterministic number.
    prior = 0.12618
    ce = np.array([0.11485, 0.11543, 0.11544])
    gain = 100 * (prior - ce) / prior
    show("Run 1 (paired against the fixed prior, n = 3)", [
        named(effect(ce, np.full(3, prior), paired=True, metric="val_ce",
                     panel="electronics"), "P-sLSTM vs class prior"),
        named(effect(gain, np.zeros(3), paired=True, metric="ce_gain_pct",
                     panel="electronics"), "P-sLSTM % below prior vs 0"),
    ])

    # Run 2: per-seed metrics from comparison-results.json.
    res = json.loads((ROOT / ".scratch/p-slstm/comparison-results.json").read_text())
    runs = {m: {k: np.array(v[k]["runs"]) for k in v} for m, v in res.items()}
    for m in runs:
        runs[m]["abs_bias"] = np.abs(runs[m]["bias_percent"])
    rows = []
    for metric in ("mape_aggregate", "abs_bias", "bias_percent", "rmse"):
        rows.append(named(effect(runs["p_slstm"][metric], runs["lstm"][metric],
                                 paired=False, metric=metric, panel="electronics"),
                          "P-sLSTM vs LSTM"))
    for metric in ("mape_aggregate", "abs_bias"):
        rows.append(named(effect(runs["pareto_nbd"][metric], runs["lstm"][metric],
                                 paired=False, metric=metric, panel="electronics"),
                          "Pareto/NBD vs LSTM"))
        rows.append(named(effect(runs["pareto_nbd"][metric], runs["p_slstm"][metric],
                                 paired=False, metric=metric, panel="electronics"),
                          "Pareto/NBD vs P-sLSTM"))
    for m in ("lstm", "p_slstm", "pareto_nbd"):
        b = runs[m]["bias_percent"]
        rows.append(named(effect(b, np.zeros_like(b), paired=True,
                                 metric="bias_percent", panel="electronics"), f"{m} vs 0"))
    show("Run 2 (independent, n = 8 / 8, Pareto/NBD n = 3)", rows)


if __name__ == "__main__":
    absorbing_death_state()
    loss_functions()
    pnbd_window_shift()
    p_slstm()
