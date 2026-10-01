# Behavioural clusters: what was run, and what K bought

`PanelConfig.cluster_features` declares a **behavioural cluster** — k-means over the
Pareto/NBD sufficient statistics `(t_x, x, T)` at the last calibration period, the group
index handed to the model as a frozen embedded category (`CONTEXT.md`,
`docs/feature_engineering.md` §4). The number of clusters `K` is part of the feature's
*name*: `kmeans_8` means eight clusters and an embedding of cardinality eight.

This document answers two questions that were never written down: **how K was chosen**, and
**what the K sweep found**. The second half exists because family F of
`docs/studies-run.md` — 24 suites, 480 studies, all complete on disk since 3 September 2026
— had never been reported. Every level number below was recomputed from the archived
`Studies/cluster__*/results.csv` files, and every Spearman from the stored
`Studies/cluster__*/LSTM/Predictions/` via `run_cluster_ablation.py --report`.

**Revised under the statistical protocol (2026-09-28).** Every comparison now follows
`docs/statistical-protocol.md`: Δ = difference of means between two arms' 40 independent
replications, with its 95% percentile-bootstrap interval (`evaluation.effects.effect`,
`paired=False`), one panel at a time, printed by
`.scratch/statistical-protocol/grid_cluster_ablation_effects.py`. What moved:

- §4 and §5.2: every verdict on |bias| is unchanged. §4 now also tests MAPE, the primary
  level metric, and on CDNOW `cluster_16` is worse on MAPE (+6.1, where |bias| shows no
  clear difference).
- §5.1: the Spearman gain is now a tested effect (Δ +0.22 to +0.23 at every K, intervals
  excluding 0) rather than a comparison of spreads. Against 20 seeded Pareto/NBD refits on the
  same windows, the labelled arms go from "level with Pareto/NBD" to clearly behind it
  (by 0.04–0.06).
- §6 now quotes the per-cell version of `docs/insights-arm-sweep.md` §4.

## Contents

1. [How K has been chosen so far](#1-how-k-has-been-chosen-so-far)
2. [What ran](#2-what-ran)
3. [The level, per K](#3-the-level-per-k)
4. [Is any K better than no cluster at all?](#4-is-any-k-better-than-no-cluster-at-all)
5. [The two things clusters do](#5-the-two-things-clusters-do)
   - [5.1 They rank customers](#51-they-rank-customers-which-nothing-else-in-this-family-does)
   - [5.2 They rescue the unbounded counters](#52-they-rescue-the-unbounded-counters-on-one-panel)
6. [The synthetic corroboration](#6-the-synthetic-corroboration)
7. [Calibration does not rank K](#7-calibration-does-not-rank-k)
8. [What this does not establish](#8-what-this-does-not-establish)

---

## 1. How K has been chosen so far

Declared, never fitted, as the ladder 4 / 8 / 16, with 8 used everywhere else:
`docs/feature_engineering.md` §5. This document is the downstream test of that ladder.

## 2. What ran

Family F, `scripts/run_cluster_ablation.py`, vast.ai, 3 September 2026.

| | |
|---|---|
| Panels | electronics — 829 customers, calibration 1999-01-01 → 2000-12-31 (validation from 2000-01-01), holdout 2001, 104 / 52 weeks, 7 classes. CDNOW — 2,357 customers, calibration 1997-01-01 → 1997-09-30 (validation from 1997-08-06), holdout 1997-10-01 → 1998-06-30, 39 / **38** weeks, 5 classes. |
| Model | LSTM, **`valendin` embedder** (verified from `param_embedder` in every `results.csv`), `cross_entropy` loss |
| Arms | `no_cluster`, `cluster_4`, `cluster_8`, `cluster_16`, `ar_unbounded`, `ar_plus_cluster_8` — the cluster arms carry no AR channel, and every arm carries the embedded target count |
| Budget | **40 replications × 50 Optuna trials × 300 Monte Carlo paths** per arm per panel, as 20 studies in each of 2 seed shards (`a` = seeds 43–62, `b` = 63–82) |
| Suites | 24 (6 arms × 2 panels × 2 shards), **all 24 present, 20/20 studies each** |
| Metrics | bias, MAPE, RMSE from each suite's `results.csv`; **Spearman recomputed** from the stored `Predictions/` by `--report`, on electronics only (§3) |

CDNOW's windows are the **pre-ADR-0009** ones — 38 holdout weeks from 1997-10-01, not the
39 from 09-30 that `docs/benchmarks-real-panels.md` uses. A CDNOW row here cannot be put in
a table beside one from the four-panel runs without saying so.

The two shards carry disjoint seed ranges, so each arm has **40 distinct replications** per
panel, not twenty run twice.

Two properties make the comparison clean. Nothing can drop the cluster column — `ModelSpec`
has no `removable_features` field and no study suite passes one, so the covariate-subset
search cannot silently delete the feature under test. And k-means runs at a fixed
`random_state`, so cluster assignment is a deterministic property of the panel and not a
hidden second variance component inside a suite that reports across-study SD.

Replications are unpaired: training is unseeded by design (`CLAUDE.md` priority 3), so
study *i* of one arm and study *i* of another share a seed but nothing else. Every test
below therefore resamples the two arms independently (`docs/statistical-protocol.md` §2):
Δ = mean(arm) − mean(baseline) over 40 vs 40 replications, with its 95% percentile-bootstrap
interval, supported when the interval excludes 0.

## 3. The level, per K

Aggregate bias %, over 40 replications per arm. `|bias|` columns treat over- and
under-forecasting alike, which is what makes arms with opposite signs comparable.

**electronics** — 40 replications × 50 trials × 300 paths per arm, `valendin` embedder,
104 / 52 weeks, holdout 2001. Spearman is the rank correlation of per-customer holdout
totals, recomputed from the stored forecasts.

| arm | mean bias | sd | median bias | mean \|bias\| | median \|bias\| | RMSE | MAPE | **Spearman** | val. objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cluster` | +20.9 | 14.1 | +22.8 | 21.7 | 22.8 | 0.377 | 54.9 | **0.039 ± 0.078** | 0.0893 |
| `cluster_4` | +13.3 | 24.8 | +9.1 | 19.8 | 14.9 | 0.377 | 51.8 | **0.264 ± 0.015** | **0.0840** |
| `cluster_8` | **+11.4** | 22.9 | +9.8 | **19.0** | **13.0** | 0.377 | 53.1 | **0.270 ± 0.055** | 0.0849 |
| `cluster_16` | +18.7 | 19.0 | +15.2 | 21.0 | 16.7 | 0.377 | 55.2 | **0.257 ± 0.043** | 0.0863 |
| `ar_unbounded` | +208.6 | 174.7 | +154.1 | 208.6 | 154.1 | 0.433 | 213.4 | 0.229 ± 0.069 | 0.0892 |
| `ar_plus_cluster_8` | +91.8 | 126.5 | +47.0 | 94.5 | 47.0 | 0.402 | 115.5 | **0.279 ± 0.051** | 0.0850 |

**CDNOW** — same budget and embedder, 39 / 38 weeks, holdout 1997-10-01 → 1998-06-30.

| arm | mean bias | sd | median bias | mean \|bias\| | median \|bias\| | RMSE | MAPE | Spearman | val. objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cluster` | **−1.0** | **16.2** | −5.4 | **12.9** | 9.7 | **0.148** | 23.1 | n/a | 0.0933 |
| `cluster_4` | +30.0 | 28.5 | +20.9 | 30.1 | 20.9 | 0.151 | 39.6 | n/a | 0.0705 |
| `cluster_8` | +2.9 | 20.7 | +1.9 | 14.3 | 9.8 | 0.151 | 26.9 | n/a | 0.0690 |
| `cluster_16` | +1.0 | 24.3 | −3.0 | 15.7 | **9.6** | 0.151 | 29.1 | n/a | **0.0668** |
| `ar_unbounded` | +313.9 | 531.8 | +102.1 | 314.6 | 102.1 | 0.251 | 325.6 | n/a | 0.0929 |
| `ar_plus_cluster_8` | +249.3 | 364.7 | +119.8 | 273.2 | 119.8 | 0.286 | 296.7 | n/a | 0.0680 |

**CDNOW has no Spearman and cannot get one from this archive.**
`run_cluster_ablation.py --panel cdnow --report` rebuilds the panel under today's week rule
(ADR-0009), which yields 39 holdout weeks, and scores it against stored predictions of
width 38 — it raises `ValueError: operands could not be broadcast together with shapes
(2357,38) (2357,39)` rather than returning a number. Recovering it means rebuilding the
pre-ADR-0009 panel. `n/a` above means exactly that, not "small".

`val. objective` is the best Optuna trial's validation cross-entropy
(`tuning.optuna_tuning.objective` returns `best_val_loss`) — the best score obtainable
**from calibration alone**, averaged over the arm's 40 studies. It is included because §7
turns on it.

Three readings, before any test:

**The K axis has no common shape across panels.** On electronics median |bias| runs
14.9 → 13.0 → 16.7 for K = 4, 8, 16: a shallow interior optimum at 8. On CDNOW it runs
20.9 → 9.8 → 9.6: monotone improving in K, with K = 4 the outlier. There is no K that is
best on both panels, and the two curves do not even share a direction.

**RMSE is untouched.** Every electronics cluster arm reports 0.377, to three decimals, the
same as the baseline; CDNOW moves 0.148 → 0.151, the wrong way. Whatever the label does, it
does it to the aggregate level and not to per-customer accuracy.

**Clusters widen the distribution.** On electronics the across-study SD of bias rises from
14.1 (`no_cluster`) to 19.0–24.8 with a label present, and on CDNOW from 16.2 to 20.7–28.5.
A better central tendency bought with more dispersion is a worse instrument, not a better
one — the point of the study-suite design (`CLAUDE.md` priority 3) is that the spread is
part of the result.

**And the one thing they move decisively is the ranking.** On electronics Spearman goes
from **0.039** without a label to **0.257–0.270** with one — a sevenfold rise, supported at
every K (§5.1: Δ +0.22 to +0.23, each 95% interval excluding 0). It still falls short of
Pareto/NBD on that panel: 0.314 over 20 seeded refits, clearly ahead of every labelled arm
by 0.04–0.06 (§5.1).
This is the measurement §8 of the first draft of this document listed as missing, and it
does not point the same way as the level does: on the metric the rest of this document
scores, clusters are indistinguishable from nothing; on the metric it never computed, they
are the difference between ordering customers and not ordering them at all. §5.1 takes it
up, because it changes what "drop the cluster axis" can be read to mean.

## 4. Is any K better than no cluster at all?

Each arm against `no_cluster`, 40 vs 40 independent replications: Δ = mean(arm) −
mean(`no_cluster`) with its 95% bootstrap interval. Negative Δ is better.

| panel | arm | mean \|bias\| | Δ \|bias\| [95% CI] | mean MAPE | Δ MAPE [95% CI] |
|---|---|---:|---:|---:|---:|
| electronics | `no_cluster` | 21.7 | — | 54.9 | — |
| | `cluster_4` | 19.8 | −1.8 [−8.6, +5.8] | 51.8 | −3.1 [−7.6, +2.0] |
| | `cluster_8` | 19.0 | −2.7 [−9.0, +3.9] | 53.1 | −1.9 [−6.3, +3.2] |
| | `cluster_16` | 21.0 | −0.6 [−7.0, +5.9] | 55.2 | +0.3 [−3.2, +3.9] |
| | `ar_unbounded` | 208.6 | **+186.9 [+136.0, +242.9]** | 213.4 | **+158.5 [+108.8, +213.1]** |
| | `ar_plus_cluster_8` | 94.5 | **+72.9 [+37.2, +114.0]** | 115.5 | **+60.6 [+27.9, +99.4]** |
| CDNOW | `no_cluster` | 12.9 | — | 23.1 | — |
| | `cluster_4` | 30.1 | **+17.2 [+8.5, +27.4]** | 39.6 | **+16.5 [+8.9, +25.6]** |
| | `cluster_8` | 14.3 | +1.4 [−3.8, +7.1] | 26.9 | +3.9 [−0.2, +8.2] |
| | `cluster_16` | 15.7 | +2.8 [−3.1, +9.7] | 29.1 | **+6.1 [+1.2, +11.7]** |
| | `ar_unbounded` | 314.6 | **+301.8 [+156.5, +476.1]** | 325.6 | **+302.6 [+158.2, +475.8]** |
| | `ar_plus_cluster_8` | 273.2 | **+260.3 [+160.4, +373.2]** | 296.7 | **+273.7 [+174.1, +384.8]** |

Bold: supported. **No K, on either panel, beats the no-cluster baseline on either level
metric.** The best cluster result anywhere here is `cluster_4` on electronics, −3.1 MAPE
points with an interval from −7.6 to +2.0: no clear difference at n = 40, which is not the
same as no difference. Every cluster effect that *is* supported is on CDNOW and is
**worse**: `cluster_4` on both metrics (+17.2 |bias|, +16.5 MAPE) and `cluster_16` on MAPE
(+6.1).

So the ladder's verdict on its own question is: the encoding does not pay, and the choice
among 4, 8 and 16 is a choice among three arms none of which is clearly better than not
using the feature. K = 8 is not a mistake; it is simply not a decision the downstream
evidence was ever able to make.

## 5. The two things clusters do

### 5.1 They rank customers, which nothing else in this family does

On electronics the label is the difference between a model that orders customers and one
that does not. Spearman over the same 40 replications per arm (mean ± sd), and each arm
against `no_cluster` (independent, 40 vs 40, 95% bootstrap interval):

| arm | Spearman | Δ vs `no_cluster` [95% CI] | mean \|bias\| |
|---|---:|---:|---:|
| `no_cluster` | 0.039 ± 0.078 | — | 21.7 |
| `cluster_4` | 0.264 ± 0.015 | **+0.226 [+0.199, +0.248]** | 19.8 |
| `cluster_8` | 0.270 ± 0.055 | **+0.232 [+0.200, +0.259]** | 19.0 |
| `cluster_16` | 0.257 ± 0.043 | **+0.219 [+0.189, +0.244]** | 21.0 |
| Pareto/NBD (benchmark, 20 seeded refits) | 0.314 ± 0.010 | +0.275 [+0.249, +0.297] | 63.1 |

**Against Pareto/NBD.** Twenty seeded Pareto/NBD refits on the same panel and windows
(`Studies/real_panel_benchmarks__ParetoNBD__electronics__r00`–`r19`, scored against this
family's cohort with the customer ids checked) are the benchmark's replications. Each arm
against them, independent, 40 vs 20, Δ = mean(arm) − mean(Pareto/NBD):

| arm | Δ Spearman vs Pareto/NBD [95% CI] |
|---|---:|
| `no_cluster` | **−0.275 [−0.297, −0.249]** |
| `cluster_4` | **−0.049 [−0.056, −0.043]** |
| `cluster_8` | **−0.043 [−0.062, −0.027]** |
| `cluster_16` | **−0.056 [−0.070, −0.043]** |
| `ar_plus_cluster_8` | **−0.035 [−0.053, −0.021]** |

The label closes most of the gap but not all of it: Pareto/NBD ranks customers clearly
better than every arm, labelled or not. (An earlier version quoted a single Pareto/NBD fit
at 0.297 and called the labelled arms "level" with it.)

The gain is supported at every K, and every interval sits far from 0, so unlike the level
effect it is not a coin toss at n = 40. **It is bought for nothing on the level** — §4
finds no clear |bias| or MAPE difference at any K on this panel, and RMSE is unmoved at
0.377.

This is the same phenomenon `docs/benchmarks-real-panels.md` calls *the forecast collapse
on long sparse panels*: with only the count as input, an LSTM on electronics gives every
customer nearly the same forecast (Spearman 0.03–0.04), and **any** persistent per-customer
channel lifts it — bounded flags reach 0.20–0.26, clusters 0.26–0.27. The cluster label is
one instance of that fix, and on this panel it is the strongest one measured.

Two things this does **not** license. It is one panel: CDNOW's Spearman cannot be recovered
from this archive at all (§3), and CDNOW is the panel that does not collapse, so there is no
reason to expect the same effect there. And it says nothing about the level, where §4 is
unchanged and §6's synthetic evidence says the label actively hurts. **The two criteria
disagree, and the arm has to be chosen against one of them** — §6.

### 5.1.1 Confirmed on four panels, and it has a rival — 21 September 2026

`docs/training-budget.md` §15 (family U, 640 suites) ran the cluster axis on all four
panels, crossed with a second lever this document did not know about: how long the model
trains. Per-customer Spearman, 20 replications a cell, `ValendinLSTM`:

| panel | no label | **label** | no label, trained longer | label + trained longer | Pareto/NBD |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronics | 0.021 | **0.305** | 0.178 | 0.305 | 0.297 |
| multichannel | −0.004 | **0.178** | 0.119 | 0.195 | 0.189 |
| cdnow | 0.364 | **0.403** | 0.383 | 0.406 | 0.450 |
| gift | 0.349 | **0.359** | 0.280 | 0.363 | 0.383 |

Three things this adds to §5.1.

**The effect is supported on two panels, not four — and §5.1's guess about CDNOW was
right.** Δ Spearman from adding the label, 95% bootstrap CI, 20 replications a cell,
`ValendinLSTM`:

| panel | Δ | 95% CI | supported | refit noise |
| --- | ---: | :---: | :---: | ---: |
| electronics | **+0.283** | +0.265 to +0.302 | yes | 0.0105 |
| multichannel | **+0.182** | +0.162 to +0.199 | yes | 0.0152 |
| cdnow | +0.039 | −0.003 to +0.088 | no | 0.0159 |
| gift | +0.010 | −0.005 to +0.026 | no | 0.0116 |

On the two panels that collapse the label is decisive. On the two that do not, the
interval spans zero — so this document's "sevenfold rise" is an electronics and
multichannel result, and §5.1's expectation that CDNOW would not show the same effect is
borne out. (An earlier version of this subsection reported cdnow as a marginal gain; its
interval spans zero, so it is not supported.)

**What *is* supported on all eight (panel, model) cells is the label added to a model
that has already been trained past its early plateau** — Δ +0.023 to +0.127, every
interval excluding zero. That is the robust form of the claim.

**A training floor is a partial substitute for the label.** On the collapsing panels,
simply training the same no-label model to the reference paper's epoch count reaches
0.178 and 0.119 — over half of what the label buys, from no extra input at all. So "any
persistent per-customer channel lifts it" is not the whole story: *enough training* lifts
it too, and the archived 0.039 was measured on a model that stopped at epoch 8.

**Crossing them adds little, and the increment is bounded.** Adding the floor on top of
the label moves Spearman by +0.001 to +0.018 depending on the cell, with 95% bootstrap
intervals spanning zero in seven of eight and ruling out a gain larger than about +0.035
anywhere — the order of what an unseeded refit moves on its own
(`docs/model-selection.md` §2). One cell, multichannel/ValendinLSTM,
does show a supported increment of +0.018. So the label and the training budget overlap
heavily; whether anything beyond them is reachable is untested, and an apparent plateau is
not an established ceiling.

### 5.2 They rescue the unbounded counters, on one panel

`ar_plus_cluster_8` against `ar_unbounded` — the same three counters, with the bounded
category added. 40 vs 40 independent replications, Δ = mean(`ar_plus_cluster_8`) −
mean(`ar_unbounded`) with its 95% bootstrap interval:

| panel | `ar_unbounded` mean \|bias\| | `+kmeans_8` | Δ \|bias\| [95% CI] | Δ MAPE [95% CI] |
|---|---:|---:|---:|---:|
| electronics | 208.6 | **94.5** | **−114.0 [−179.0, −49.5]** | **−97.9 [−160.4, −36.1]** |
| CDNOW | 314.6 | 273.2 | −41.5 [−246.6, +144.1] | −28.9 [−232.3, +156.3] |

On electronics the rescue is large and supported on both level metrics: adding a bounded
categorical summary of the same history cuts the unbounded counters' mean |bias| by more
than half. **On CDNOW there is no clear rescue at n = 40**: the interval spans a gain of
250 points and a loss of 140.

This matters for how the effect is read. The synthetic grid explained the rescue as the
clusters *displacing* the broken counters rather than contributing: the more of the
prediction the bounded channel carries, the less rests on channels that drift out of their
fitted range during the rollout. That story predicts the rescue should be *strongest* where
the counters escape most — and CDNOW is that panel (recency leaves the calibration range on
56.9% of holdout cells, against 37.7% on electronics). It is the panel where the rescue
fails. Either the displacement account is incomplete, or 39 calibration periods are too few
for k-means to find groups worth displacing anything with.

Either way, the rescued arm remains clearly worse than `no_cluster` on both panels (§4:
Δ |bias| +72.9 on electronics and +260.3 on CDNOW, both supported), so it rescues an arm
that should not be used.

## 6. The synthetic corroboration

`docs/insights-arm-sweep.md` §4 tests `kmeans_8` on stronger evidence — the same generated
panels on both sides, where the data-generating process is known. Each rate × churn cell
is tested on its own (paired, 10 panels, 95% bootstrap interval of the mean |bias|
difference), and the table counts the cells:

| contrast | model | mean \|bias\| (pooled) | cells: falls | cells: rises | no clear difference |
|---|---|---:|---:|---:|---:|
| `kmeans_8` vs none, under `no_ar` | LSTM | 153.5 → 179.5 | 1 | 6 | 9 |
| `kmeans_8` vs none, under `no_ar` | Transformer | 92.0 → 96.4 | 1 | 1 | 14 |
| `kmeans_8` vs none, under `ar_bounded` | LSTM | 78.3 → 115.8 | 1 | 9 | 6 |
| `kmeans_8` vs none, under `ar_bounded` | Transformer | 60.8 → 80.1 | 0 | 4 | 12 |
| `kmeans_8` vs none, under `ar_unbounded` | LSTM | 384.9 → 431.9 | **5** | 2 | 9 |
| `kmeans_8` vs none, under `ar_unbounded` | Transformer | 110.6 → **69.8** | **8** | 1 | 7 |

Under a usable AR encoding the label raises |bias| in 20 cells and lowers it in 3. It
helps more often than it hurts only under `ar_unbounded` (for the LSTM there it still
hurts in two rate-0.01 cells, by more than 1,000 points) — the arm `docs/insights-arm-sweep.md` §9 recommends
dropping anyway. The mechanism given there also survives everything above: a label assigned
from calibration behaviour cannot update when the simulated customer goes quiet, so it keeps
asserting the customer is who they used to be. That is the `ar_unbounded` failure in a
different costume — bounded in *range*, but equally stale in *time*.

Family F neither confirms nor contradicts this on the real panels: it finds no clear
difference on electronics, and a clearly worse level for `cluster_4` (and for `cluster_16`
on MAPE) on CDNOW. The synthetic evidence is what settles the direction.

**On the level.** Every number in this section, and in §3 and §4, is aggregate bias. The
synthetic grid computes no per-customer Spearman at all — it reports `shape_correlation`,
the correlation of the *weekly aggregate* curves, which is a different quantity — so it has
nothing to say about §5.1. The two bodies of evidence are therefore not in conflict; they
measure different things, and they point opposite ways:

| criterion | evidence | verdict on `kmeans_8` |
|---|---|---|
| aggregate bias | 16 cells × 10 paired synthetic panels, both models (§6) | **hurts**: \|bias\| rises in 20 cells and falls in 3 across the 4 usable contrasts |
| aggregate bias, MAPE | 40 replications, 2 real panels (§4) | no clear difference on electronics; worse for K = 4 (and K = 16 on MAPE) on CDNOW |
| per-customer Spearman | 40 replications, electronics (§5.1) | **helps**, Δ +0.22 to +0.23, supported at every K |
| per-customer Spearman | synthetic grid | never computed |

**Consolidated verdict: drop the cluster axis.** It improves exactly one arm, and that arm
is the one to drop.

## 7. Calibration does not rank K

Moved to `docs/model-selection.md` §3.7.

## 8. What this does not establish

- **Discrimination on one panel only — CDNOW's is unrecoverable.** §5.1 answers "does the
  *category* carry the discrimination the *counters* carried?" on electronics, and yes: the
  label reaches 0.26–0.27 against the unbounded triple's 0.229 and no-cluster's 0.039.
  CDNOW has no such number and cannot get one from this archive, because
  `--panel cdnow --report` raises on the pre-ADR-0009 window (§3). So the discrimination
  finding rests on a single panel — and on the one panel where the count-only model
  collapses, which is where any persistent channel is expected to help.
- **LSTM only.** Family F ran no Transformer and no ValendinLSTM. The synthetic grid shows
  the two architectures respond differently to `kmeans_8` (§6: under `no_ar` the LSTM is
  hurt in 6 cells, the Transformer in 1), so the real-panel result should not
  be read as architecture-independent.
- **Three rungs, one basis.** K ∈ {4, 8, 16} on one feature triple with one algorithm, one
  standardisation, one `random_state`. Nothing here separates "K = 8 is right" from "k-means
  on three standardised sufficient statistics is the wrong basis at every K".
- **Unpaired, n = 40.** Independent bootstrap intervals on unseeded training runs. A paired
  design is impossible while training is unseeded, and that is a deliberate choice, but it
  costs power: the electronics `cluster_8` interval (−9.0 to +3.9 on |bias|) is too wide to
  resolve an improvement of the size its mean suggests.
- **The validation-window deviation is untested.** The label is fitted on the full
  calibration window, which contains the validation window early stopping scores on
  (`docs/feature_engineering.md` §4). The bias is argued to be small and bounded; it has
  never been measured, and §7's finding is exactly the place it would show up if it were not.
