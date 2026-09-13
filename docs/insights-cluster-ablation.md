# Behavioural clusters: what was run, and what K bought

`PanelConfig.cluster_features` declares a **behavioural cluster** — k-means over the
Pareto/NBD sufficient statistics `(t_x, x, T)` at the last calibration period, the group
index handed to the model as a frozen embedded category (`CONTEXT.md`,
`docs/feature_engineering.md` §4). The number of clusters `K` is part of the feature's
*name*: `kmeans_8` means eight clusters and an embedding of cardinality eight.

This document answers two questions that were never written down: **how K was chosen**, and
**what the K sweep found**. The second half exists because family F of
`docs/studies-run.md` — 24 suites, 480 studies, all complete on disk since 3 September 2026
— had never been reported. Every number below was recomputed from the archived
`Studies/cluster__*/LSTM/metrics.csv` files.

## Contents

1. [How K has been chosen so far](#1-how-k-has-been-chosen-so-far)
2. [What ran](#2-what-ran)
3. [The level, per K](#3-the-level-per-k)
4. [Is any K better than no cluster at all?](#4-is-any-k-better-than-no-cluster-at-all)
5. [The one thing clusters do](#5-the-one-thing-clusters-do)
6. [The synthetic corroboration](#6-the-synthetic-corroboration)
7. [Calibration does not rank K](#7-calibration-does-not-rank-k)
8. [What this does not establish](#8-what-this-does-not-establish)

---

## 1. How K has been chosen so far

**There is no elbow method in this package, and no other unsupervised selection rule.** No
inertia curve, no silhouette, no gap statistic — `compute_cluster_labels` reads K out of
the feature name and calls `KMeans(n_clusters=K, n_init=10, random_state=0)` on the
standardised triple. K is *declared*, never *fitted*.

Two constraints shaped that.

**K is an architecture dimension, not only a partition size.** The label is embedded, with
cardinality pinned to K (`docs/feature_engineering.md` §4). Choosing K chooses the width of
a learned embedding table at the same time as the coarseness of the partition, and those
two pull in opposite directions: a larger K describes a customer more finely and gives each
group fewer customers to learn from.

**K is an arm, never an Optuna knob** (`scripts/run_cluster_ablation.py`). Had the search
tuned K per trial, arms would stop being comparable and no difference could be attributed
to the encoding. So K was fixed per suite and swept *between* suites.

The sweep declared was a geometric ladder, `kmeans_4 / kmeans_8 / kmeans_16`, on the
grounds that K is the sensitive knob of the design — the same criticism `docs/p-slstm.md`
§11 makes of never sweeping P-sLSTM's patch size. Everywhere outside family F, K is frozen
at **8** as the single representative rung: the arm sweep (family B) and the real-panel arms
(family H) both use a binary cluster axis `{no_cluster, kmeans_8}`.

So the honest statement of provenance is: **8 is the mid-rung of a declared ladder, and the
ladder was tested downstream rather than in feature space.** The rest of this document is
that test.

## 2. What ran

Family F, `scripts/run_cluster_ablation.py`, vast.ai, 3 September 2026.

| | |
|---|---|
| Panels | electronics (104 calibration / 52 holdout weeks), CDNOW (39 / 39) |
| Model | LSTM, `valendin` embedder, `cross_entropy` loss |
| Arms | `no_cluster`, `cluster_4`, `cluster_8`, `cluster_16`, `ar_unbounded`, `ar_plus_cluster_8` |
| Budget | 50 Optuna trials × 20 studies × 2 shards, 300-path Monte Carlo rollout |
| Suites | 24 (6 arms × 2 panels × 2 shards), **all 24 present, 20/20 studies each** |

Shards `a` and `b` carry disjoint seed ranges (43–62, 63–82), so each arm has **40 distinct
replications** per panel, not twenty run twice.

Two properties make the comparison clean. Nothing can drop the cluster column — `ModelSpec`
has no `removable_features` field and no study suite passes one, so the covariate-subset
search cannot silently delete the feature under test. And k-means runs at a fixed
`random_state`, so cluster assignment is a deterministic property of the panel and not a
hidden second variance component inside a suite that reports across-study SD.

Replications are unpaired: training is unseeded by design (`CLAUDE.md` priority 3), so
study *i* of one arm and study *i* of another share a seed but nothing else. All tests
below are therefore two-sample rank tests, not paired ones.

## 3. The level, per K

Aggregate bias %, over 40 replications per arm. `|bias|` columns treat over- and
under-forecasting alike, which is what makes arms with opposite signs comparable.

**electronics**

| arm | mean bias | sd | median bias | mean \|bias\| | median \|bias\| | RMSE | MAPE | val. objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cluster` | +20.9 | 14.1 | +22.8 | 21.7 | 22.8 | 0.377 | 54.9 | 0.0893 |
| `cluster_4` | +13.3 | 24.8 | +9.1 | 19.8 | 14.9 | 0.377 | 51.8 | **0.0840** |
| `cluster_8` | **+11.4** | 22.9 | +9.8 | **19.0** | **13.0** | 0.377 | 53.1 | 0.0849 |
| `cluster_16` | +18.7 | 19.0 | +15.2 | 21.0 | 16.7 | 0.377 | 55.2 | 0.0863 |
| `ar_unbounded` | +208.6 | 174.7 | +154.1 | 208.6 | 154.1 | 0.433 | 213.4 | 0.0892 |
| `ar_plus_cluster_8` | +91.8 | 126.5 | +47.0 | 94.5 | 47.0 | 0.402 | 115.5 | 0.0850 |

**CDNOW**

| arm | mean bias | sd | median bias | mean \|bias\| | median \|bias\| | RMSE | MAPE | val. objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cluster` | **−1.0** | **16.2** | −5.4 | **12.9** | 9.7 | **0.148** | 23.1 | 0.0933 |
| `cluster_4` | +30.0 | 28.5 | +20.9 | 30.1 | 20.9 | 0.151 | 39.6 | 0.0705 |
| `cluster_8` | +2.9 | 20.7 | +1.9 | 14.3 | 9.8 | 0.151 | 26.9 | 0.0690 |
| `cluster_16` | +1.0 | 24.3 | −3.0 | 15.7 | **9.6** | 0.151 | 29.1 | **0.0668** |
| `ar_unbounded` | +313.9 | 531.8 | +102.1 | 314.6 | 102.1 | 0.251 | 325.6 | 0.0929 |
| `ar_plus_cluster_8` | +249.3 | 364.7 | +119.8 | 273.2 | 119.8 | 0.286 | 296.7 | 0.0680 |

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

## 4. Is any K better than no cluster at all?

Two-sample Mann-Whitney on |bias %|, each arm against `no_cluster`, 40 vs 40.

| panel | arm | median \|bias\| | vs baseline | z | p |
|---|---|---:|---:|---:|---:|
| electronics | `no_cluster` | 22.8 | — | — | — |
| | `cluster_4` | 14.9 | −7.9 | −1.41 | 0.16 |
| | `cluster_8` | 13.0 | −9.7 | −1.43 | 0.15 |
| | `cluster_16` | 16.7 | −6.1 | −0.44 | 0.66 |
| | `ar_unbounded` | 154.1 | +131.3 | 7.19 | 7×10⁻¹³ |
| | `ar_plus_cluster_8` | 47.0 | +24.2 | 3.09 | 0.002 |
| CDNOW | `no_cluster` | 9.7 | — | — | — |
| | `cluster_4` | 20.9 | +11.2 | 4.09 | 4×10⁻⁵ |
| | `cluster_8` | 9.8 | +0.1 | −0.47 | 0.64 |
| | `cluster_16` | 9.6 | −0.1 | 0.04 | 0.97 |
| | `ar_unbounded` | 102.1 | +92.4 | 4.84 | 1×10⁻⁶ |
| | `ar_plus_cluster_8` | 119.8 | +110.1 | 6.05 | 1×10⁻⁹ |

**No K, on either panel, beats the no-cluster baseline at any conventional level.** The
strongest cluster result anywhere here is `cluster_8` on electronics at p = 0.15 — a
9.7-point median improvement that 40 replications cannot separate from training noise. The
only *significant* effect of a cluster arm against the baseline is `cluster_4` on CDNOW,
which is significantly **worse** (p = 4×10⁻⁵).

So the ladder's verdict on its own question is: the encoding does not pay, and the choice
among 4, 8 and 16 is a choice among three arms that are individually indistinguishable from
not using the feature. K = 8 is not a mistake; it is simply not a decision the downstream
evidence was ever able to make.

## 5. The one thing clusters do

`ar_plus_cluster_8` against `ar_unbounded` — the same three counters, with the bounded
category added:

| panel | `ar_unbounded` median \|bias\| | `+kmeans_8` | z | p |
|---|---:|---:|---:|---:|
| electronics | 154.1 | **47.0** | −4.14 | 3×10⁻⁵ |
| CDNOW | 102.1 | 119.8 | 0.66 | 0.51 (n.s.) |

On electronics the rescue is large and significant: adding a bounded categorical summary of
the same history cuts the unbounded counters' median |bias| by two thirds. **On CDNOW it
does not happen at all** — the point estimate moves the wrong way.

This matters for how the effect is read. The synthetic grid explained the rescue as the
clusters *displacing* the broken counters rather than contributing: the more of the
prediction the bounded channel carries, the less rests on channels that drift out of their
fitted range during the rollout. That story predicts the rescue should be *strongest* where
the counters escape most — and CDNOW is that panel (recency leaves the calibration range on
56.9% of holdout cells, against 37.7% on electronics). It is the panel where the rescue
fails. Either the displacement account is incomplete, or 39 calibration periods are too few
for k-means to find groups worth displacing anything with.

Either way, the rescued arm remains far worse than `no_cluster` on both panels (47.0 and
119.8 against 22.8 and 9.7), so it rescues an arm that should not be used.

## 6. The synthetic corroboration

`docs/insights-arm-sweep.md` §4 tests `kmeans_8` on far stronger evidence — a *paired*
Wilcoxon over the same 160 synthetic panels, where the data-generating process is known:

| contrast | model | median \|bias\| | Δ | p |
|---|---|---:|---:|---:|
| `kmeans_8` vs none, under `no_ar` | LSTM | 56.5 → 100.4 | +17.3 | 2×10⁻⁷ |
| `kmeans_8` vs none, under `no_ar` | Transformer | 71.0 → 73.4 | +7.9 | 0.749 (n.s.) |
| `kmeans_8` vs none, under `ar_bounded` | LSTM | 32.2 → 70.2 | +34.1 | 2×10⁻¹³ |
| `kmeans_8` vs none, under `ar_bounded` | Transformer | 49.5 → 58.8 | +15.6 | 8×10⁻⁴ |
| `kmeans_8` vs none, under `ar_unbounded` | LSTM | 261.2 → **132.9** | −59.2 | 7×10⁻⁵ |
| `kmeans_8` vs none, under `ar_unbounded` | Transformer | 97.7 → **55.7** | −30.7 | 9×10⁻⁹ |

Five of six significant; four of those say the label **hurts**, and the two that say it
helps are both under `ar_unbounded` — the arm `docs/insights-arm-sweep.md` §9 recommends
dropping anyway. The mechanism given there also survives everything above: a label assigned
from calibration behaviour cannot update when the simulated customer goes quiet, so it keeps
asserting the customer is who they used to be. That is the `ar_unbounded` failure in a
different costume — bounded in *range*, but equally stale in *time*.

Family F neither confirms nor contradicts this on the real panels: it is consistent with a
null effect, and the synthetic evidence is what settles the direction.

**Consolidated verdict: drop the cluster axis.** It improves exactly one arm, and that arm
is the one to drop.

## 7. Calibration does not rank K

This is the finding with consequences beyond family F, and it is the reason the next
question — *can K be chosen from calibration alone?* — is harder than it looks.

The `val. objective` column in §3 is the best validation cross-entropy any of 50 Optuna
trials achieved, on the temporal validation window (ADR-0001). It is the natural
calibration-only criterion: it is what early stopping monitors, what the architecture search
minimises, and it never touches a holdout period. If a calibration-only rule for K exists,
this is the obvious candidate.

**It points the wrong way on both panels.** On CDNOW the objective falls monotonically as K
rises — 0.0933 (no cluster) → 0.0705 → 0.0690 → 0.0668 for K = 4, 8, 16 — while holdout
median |bias| goes 9.7 → 20.9 → 9.8 → 9.6. A rule that minimised validation loss would
choose the *largest* K on offer, and would choose a cluster arm over no clusters on every
panel here, at exactly the moment §4 shows the feature buys nothing.

Ranking the six arms by each criterion:

| panel | by calibration objective (best first) | by holdout median \|bias\| (best first) | Spearman |
|---|---|---|---:|
| electronics | `cluster_4`, `cluster_8`, `ar_plus_cluster_8`, `cluster_16`, `ar_unbounded`, `no_cluster` | `cluster_8`, `cluster_4`, `cluster_16`, `no_cluster`, `ar_plus_cluster_8`, `ar_unbounded` | 0.66 |
| CDNOW | `cluster_16`, `ar_plus_cluster_8`, `cluster_8`, `cluster_4`, `ar_unbounded`, `no_cluster` | `cluster_16`, `no_cluster`, `cluster_8`, `cluster_4`, `ar_unbounded`, `ar_plus_cluster_8` | 0.09 |

On both panels the calibration criterion ranks `no_cluster` **last of six**, and on CDNOW
the two rankings are essentially unrelated. Within a single arm it is no better: across an
arm's 40 replications, Spearman between the validation objective and holdout |bias| is
between −0.37 and +0.37, and is 0.00 for `cluster_8` on CDNOW.

The reason is structural rather than statistical. Validation cross-entropy scores
**one-step-ahead conditional** predictions with the true history fed in; the holdout metric
scores a **52-step free-running rollout** where the model consumes its own samples. A
feature that sharpens the next-period conditional while going stale over a long rollout —
which is exactly what a frozen label does — improves the first and degrades the second *by
construction*. More clusters give the model a finer conditional lookup and a staler frozen
assertion at once.

Any calibration-only rule for K therefore has to be scored against something that shares the
rollout's structure, not against the fitting loss. That is the thing to design next.

## 8. What this does not establish

- **No discrimination measurement.** Everything above is level (bias, RMSE, MAPE). The
  achieved-Spearman number that `run_cluster_ablation.py --report` computes from the
  archived predictions is not included here, because it requires the panel actuals and was
  not recomputed. The AR ablation's finding — that the continuous triple raises per-customer
  discrimination ~5× — has no cluster counterpart yet, so "does the *category* carry the
  discrimination the *counters* carried?" remains open on the real panels.
- **LSTM only.** Family F ran no Transformer and no ValendinLSTM. The synthetic grid shows
  the two architectures respond differently to `kmeans_8` (§6: the LSTM is hurt
  significantly under `no_ar`, the Transformer is not), so the real-panel result should not
  be read as architecture-independent.
- **Three rungs, one basis.** K ∈ {4, 8, 16} on one feature triple with one algorithm, one
  standardisation, one `random_state`. Nothing here separates "K = 8 is right" from "k-means
  on three standardised sufficient statistics is the wrong basis at every K".
- **Unpaired, n = 40.** Two-sample tests on unseeded training runs. A paired design is
  impossible while training is unseeded, and that is a deliberate choice, but it costs
  power: the electronics `cluster_8` improvement would need a substantially larger n to
  resolve if it is real.
- **The validation-window deviation is untested.** The label is fitted on the full
  calibration window, which contains the validation window early stopping scores on
  (`docs/feature_engineering.md` §4). The bias is argued to be small and bounded; it has
  never been measured, and §7's finding is exactly the place it would show up if it were not.
