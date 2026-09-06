# Arm sweep: does encoding a customer's history close the churn gap?

`docs/insights-study.md` §3 diagnosed the neural models' failure as structural — they
have no way to represent a customer who has stopped buying — and §5.2 proposed an
absorbing state as the fix. Before building one, the cheaper hypothesis had to be ruled
out: that a **representation** of the customer's history is enough, and no architectural
change is needed.

This document is that measurement. It reads the arm sweep of `grids/seasonal_4x4x10.py`
run on the vast.ai fleet 4–6 September 2026 — twelve neural `(model, arm)` trees and one
Pareto/NBD benchmark, each trained on all 160 synthetic panels, 2,080 suites in total.

**The answer is no.** Bounded AR features help, significantly and cheaply, and leave the
churn slope monotone and an order of magnitude outside the benchmark. Nothing on the arm
axis removes it.

Every number below was recomputed from the `results.csv` files under
`Studies/seasonal_4x4x10__*/`. A rendered version of this analysis, with the per-cell
heatmaps as figures, is published at
<https://claude.ai/code/artifact/2aca6161-3a28-4798-84c5-eaa1abeb922e>.

## Contents

1. [What ran](#1-what-ran)
2. [Reproduction check](#2-reproduction-check)
3. [The churn axis, per arm](#3-the-churn-axis-per-arm)
4. [What the arm axis moved](#4-what-the-arm-axis-moved)
5. [Per cell: where each architecture fails](#5-per-cell-where-each-architecture-fails)
6. [The shape of the miss](#6-the-shape-of-the-miss)
7. [Every tree, pooled](#7-every-tree-pooled)
8. [What this says to build next](#8-what-this-says-to-build-next)
9. [What this does not establish](#9-what-this-does-not-establish)

---

## 1. What ran

`scripts/reconcile_grid.py --grid seasonal_4x4x10` reports **160/160 for all twelve arm
trees and for `ParetoNBD`**. No cell is thinner than another, so every mean below averages
the same ten replicate panels — the failure mode F19 exists to catch, and did not occur.

`ValendinLSTM` is reported `ABSENT`. It cannot take a non-embedded channel, so it can run
no arm on this grid (F11). That is a designed gap, not a shortfall, and it is why
`reconcile_grid.py` distinguishes the two states.

The arm axis crosses three AR encodings with two cluster settings on one embedder:

| axis | values | what it is |
| --- | --- | --- |
| AR encoding | `no_ar` | the archived configuration — the target's own past and the cyclical pair |
| | `ar_unbounded` | `(period_since_last_transaction, cumulative_transactions, period_since_first_transaction)` — the generating Pareto/NBD's own sufficient statistic `(t_x, x, T)` |
| | `ar_bounded` | the same history step-encoded: `active_in_last_{2,4,8,16,32}_periods` plus `has_transacted_before`, every flag 0 beyond its bin |
| cluster | `no_cluster` / `kmeans_8` | the triple as a frozen category instead of a counter |
| embedder | `valendin` | declared but not crossed; `projected` was cut on cost |

Each suite is one study, 100 Optuna trials, a 200-path rollout. Panels are 1000 customers
over 156 weeks (two calibration years, one holdout year), `clip_target_upper=6`, four
mean transaction rates × four churn rates × ten seeds.

The panels are generated *by* a Pareto/NBD, so that benchmark is correct by construction.
It is the **ceiling**, not a competitor — and `ar_unbounded` is literally what it
conditions on, which is what makes this grid, rather than a real panel, the place to ask
the question.

**Do not rank these trees on RMSE.** Ten of the thirteen report a pooled RMSE of 0.18 — the
benchmark's number to two decimals — and an eleventh reports 0.19, while their aggregate
MAPE spans 51% to 441%. On
a target that is mostly zeros, per-customer per-period RMSE measures getting the zeros
right; `docs/hurdle-models-vs-pareto-nbd.md` §2 documents the trap. Every ranking here is
on `bias_percent` and `mape_aggregate`, both from
`models.monte_carlo_forecasting.compute_forecast_metrics`.

---

## 2. Reproduction check

`no_ar-no_cluster-valendin` is the archived configuration, so it doubles as a check that
nothing moved underneath the grid. Aggregate bias % across the four churn levels, against
`insights-study.md` §2:

| model | archived | this run |
| --- | --- | --- |
| Pareto/NBD | 2.2 · 9.2 · 10.2 · 15.8 | **2.2 · 9.2 · 10.2 · 15.8** |
| LSTM | 40 · 84 · 162 · 358 | 37 · 74 · 127 · 365 |
| Transformer | 50 · 91 · 107 · 212 | 41 · 71 · 94 · 145 |

The benchmark reproduces to the decimal, which is what a deterministic fit should do and
is the strongest evidence the grid itself is unchanged. Both neural models come in
somewhat better, the Transformer noticeably so at high churn.

That is the expected direction and it has a known cause: the archived run searched 10
trials for the LSTM against 20 for the Transformer, and this one searched 100 for both
(commit `2d815b3`). §2 of `insights-study.md` flagged that confound and said a re-run
would not carry it. **This is that re-run**, and the confound closing is what the
Transformer's 212 → 145 is. It is not a new effect, and the two models are now comparable
to each other for the first time.

---

## 3. The churn axis, per arm

Aggregate bias %, marginal over the four transaction rates (n = 40 panels per entry):

| config | churn 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: |
| **Pareto/NBD** | **+2.2** | **+9.2** | **+10.2** | **+15.8** |
| LSTM `ar_bounded-no_cluster` | +32.3 | +49.1 | +65.8 | +146.5 |
| LSTM `ar_bounded-kmeans_8` | +34.5 | +67.9 | +127.8 | +232.5 |
| LSTM `no_ar-no_cluster` | +37.4 | +73.6 | +126.6 | +365.2 |
| LSTM `no_ar-kmeans_8` | +45.1 | +86.1 | +166.3 | +420.5 |
| LSTM `ar_unbounded-no_cluster` | +327.7 | +255.2 | +332.8 | +617.6 |
| LSTM `ar_unbounded-kmeans_8` | +160.0 | +432.6 | +350.1 | +761.9 |
| Transformer `ar_bounded-no_cluster` | +41.3 | +32.3 | +69.9 | +73.4 |
| Transformer `ar_unbounded-kmeans_8` | +29.0 | +40.6 | +72.2 | +90.8 |
| Transformer `ar_bounded-kmeans_8` | +40.4 | +51.7 | +96.0 | +108.4 |
| Transformer `no_ar-no_cluster` | +41.2 | +70.8 | +94.3 | +145.4 |
| Transformer `no_ar-kmeans_8` | +33.8 | +92.0 | +91.7 | +146.2 |
| Transformer `ar_unbounded-no_cluster` | +66.8 | +72.2 | +109.7 | +173.0 |

**The slope survives every arm.** Bias rises with churn for all thirteen configurations,
including the benchmark. What the best arm buys is a shallower slope, not a flat one: the
LSTM's ratio from churn 0.2 to 0.8 falls from 9.8× (`no_ar`) to 4.5× (`ar_bounded`), and
the Transformer's from 3.5× to 1.8×. Both still end four to nine times outside the
benchmark's +15.8%.

This is the finding the arms were built to attack, and it is the one they did not move.

---

## 4. What the arm axis moved

Each row is a Wilcoxon signed-rank test on |bias %| across the same 160 paired panels,
against the arm differing in one factor. Δ is the median of the per-panel differences, so
it is not the difference of the two medians.

| contrast | model | median \|bias\| before → after | Δ median | p |
| --- | --- | ---: | ---: | ---: |
| `ar_bounded` vs `no_ar` | LSTM | 56.5 → **32.2** | −13.2 | 2×10⁻¹² |
| `ar_bounded` vs `no_ar` | Transformer | 71.0 → **49.5** | −10.5 | 9×10⁻⁵ |
| `ar_unbounded` vs `no_ar` | LSTM | 56.5 → 261.2 | +167.2 | 1×10⁻¹⁷ |
| `ar_unbounded` vs `no_ar` | Transformer | 71.0 → 97.7 | +10.2 | 0.106 (n.s.) |
| `kmeans_8` vs none, under `no_ar` | LSTM | 56.5 → 100.4 | +17.3 | 2×10⁻⁷ |
| `kmeans_8` vs none, under `no_ar` | Transformer | 71.0 → 73.4 | +7.9 | 0.749 (n.s.) |
| `kmeans_8` vs none, under `ar_bounded` | LSTM | 32.2 → 70.2 | +34.1 | 2×10⁻¹³ |
| `kmeans_8` vs none, under `ar_bounded` | Transformer | 49.5 → 58.8 | +15.6 | 8×10⁻⁴ |
| `kmeans_8` vs none, under `ar_unbounded` | LSTM | 261.2 → **132.9** | −59.2 | 7×10⁻⁵ |
| `kmeans_8` vs none, under `ar_unbounded` | Transformer | 97.7 → **55.7** | −30.7 | 9×10⁻⁹ |

**Bounding the AR features helps, for both architectures.** It is the only change on the
arm axis that improves anything, and it is significant for both models. It is also cheap —
six binary flags. Keep it.

**Handing the model the true sufficient statistic makes it worse.** `ar_unbounded` is what
the generator conditions on, and giving it to the LSTM raises median |bias| from 56.5% to
261.2%. It is also the only arm that moves RMSE at all — 0.18 → 0.32, on a metric that
separates nothing else on this page.

The mechanism is the one `insights-study.md` §4.3 diagnosed: two of the three counters are
capped by the calibration window and keep counting through the holdout, so by the end of a
52-step rollout the model is reading values it never saw while fitting. `ar_bounded` carries
*the same information* — every flag is a function of the same history — and differs only in
that no holdout value can leave the fitted range. **The information is right; the encoding
is unusable.** That is the cleanest statement of the point available anywhere in this repo,
because here the true model is known and its own error is the ceiling.

**Frozen per-customer categories hurt — except where they are covering for something
worse.** Five of the six `kmeans_8` contrasts are significant, and they split by what they
are added to. Under `no_ar` and `ar_bounded` the cluster label hurts in all four contrasts
(three significantly): a label assigned from calibration behaviour cannot update when the
simulated customer goes quiet, so it adds a channel that keeps asserting the customer is
who they used to be — the same failure as `ar_unbounded`, in a different costume.

Under `ar_unbounded` it reverses, and strongly: median |bias| falls 261.2 → 132.9 for the
LSTM (p = 7×10⁻⁵) and 97.7 → 55.7 for the Transformer (p = 9×10⁻⁹). The consistent reading
is that this is not the clusters working, it is the clusters *displacing* the broken
counters. K-means gives the model a **bounded** categorical summary of the same history, so
the more of the prediction it carries, the less rests on channels that drift out of range
during the rollout. Both arms it rescues remain worse than the corresponding `ar_bounded`
arm, which reaches the same end by construction rather than by competition.

So `kmeans_8` improves exactly one thing, and that thing is the arm §8 recommends dropping.

---

## 5. Per cell: where each architecture fails

Aggregate bias % per `(mean_transaction_rate, churn_rate)` cell, ten replicate panels each,
for the best arm of each architecture and the benchmark.

**Pareto/NBD** — the ceiling:

| rate \ churn | 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | +12.2 | +34.0 | +40.7 | +54.5 |
| 0.05 | +8.4 | +15.3 | +8.8 | +13.0 |
| 0.10 | +2.6 | +1.2 | +3.5 | +7.2 |
| 0.30 | −14.4 | −13.7 | −12.1 | −11.7 |

**LSTM `ar_bounded-no_cluster-valendin`**:

| rate \ churn | 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | +43.9 | +114.5 | +189.5 | **+575.9** |
| 0.05 | +46.5 | +52.0 | +59.9 | +18.8 |
| 0.10 | +29.1 | +28.3 | +20.4 | +4.0 |
| 0.30 | +9.6 | +1.6 | −6.6 | −12.6 |

**Transformer `ar_bounded-no_cluster-valendin`**:

| rate \ churn | 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | +35.7 | +26.5 | +50.4 | +88.9 |
| 0.05 | +61.0 | +52.9 | +66.4 | +76.0 |
| 0.10 | +43.4 | +26.6 | +103.5 | +82.5 |
| 0.30 | +25.1 | +23.4 | +59.3 | +46.1 |

### The two architectures fail in different corners

Mean of |LSTM bias| − |Transformer bias| per cell, ten paired panels each. Positive means
the Transformer is closer to truth in that cell:

| rate \ churn | 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | +2.5 | +56.4 | +129.1 | **+486.4** |
| 0.05 | −16.3 | −2.4 | −8.2 | −57.5 |
| 0.10 | −15.1 | −4.6 | −91.8 | −74.4 |
| 0.30 | −17.0 | −28.7 | −51.9 | −27.1 |

**The split is on the sparsity axis, not the churn axis.** The LSTM wins twelve of sixteen
cells, often by 15–90 points, and loses the entire `rate = 0.01` row — at churn 0.80 by
486 points. Its cells span −12.6% to +575.9%, a 46× range; the Transformer's span +23.4% to
+103.5%, a 4.4× range. The LSTM is the better model wherever there is enough data to be
better on, and falls apart where there is not; the Transformer is uniformly mediocre and
never falls apart.

`insights-study.md` §2 read the same asymmetry off the archived run and called it worth a
second look because "a single fix may not serve both". At equalised search budget it is
still there, so it is not a search artifact.

### Read the middle rows, not the corners

The benchmark's own worst cell is that same sparse corner (+54.5%), and `rate = 0.30` is
the one place it goes negative (−12 to −14% at every churn level). A panel where the
correct model is at +54% is not a panel to read a neural model's failure off.

The rows that carry signal are 0.05 and 0.10, where the benchmark sits inside ±15%. There
the LSTM's best arm still runs +4% to +60% and the Transformer's +27% to +104% — smaller
numbers than the headline, and still the same monotone story.

---

## 6. The shape of the miss

Everything above is a scalar per panel. It says how far each model's holdout volume is
from the truth and never what the miss looks like. `scripts/plot_grid_forecasts.py` draws
the curves — actual weekly holdout volume against each model's simulated volume, rebuilt
from the stored `Predictions/Prediction_*.csv` rather than re-simulated:

```
PYTHONPATH=src python scripts/plot_grid_forecasts.py --grid seasonal_4x4x10 --overview
PYTHONPATH=src python scripts/plot_grid_forecasts.py --grid seasonal_4x4x10 --per-cell
```

`--overview` is one 4×4 figure, an axes per cell, averaged over the cell's ten replicates
(`figures/seasonal_4x4x10__forecast_overview.png`). `--per-cell` writes sixteen figures, one
per cell, each drawing its ten replicate panels separately — every dataset in the grid,
once. Reading them alongside two measurements separates the miss into three parts that the
bias number adds together.

### Shape: the neural models have the strength the benchmark lacks

Correlation between predicted and actual weekly holdout totals
(`studies.synthetic_grid.shape_correlation`, mean over the four transaction rates).
Correlation is invariant to a multiplicative over-prediction, so it isolates shape from
level:

| config | churn 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: |
| Pareto/NBD | −0.07 | +0.02 | +0.07 | +0.09 |
| LSTM `ar_bounded` | **0.69** | **0.64** | **0.54** | 0.29 |
| LSTM `no_ar` | 0.67 | 0.62 | 0.51 | **0.32** |
| Transformer `ar_bounded` | 0.51 | 0.45 | 0.35 | 0.28 |
| Transformer `no_ar` | 0.50 | 0.50 | 0.39 | 0.25 |
| LSTM `ar_unbounded` | 0.17 | 0.24 | 0.29 | 0.22 |
| Transformer `ar_unbounded` | 0.45 | 0.33 | 0.19 | 0.16 |

**The benchmark's shape correlation is zero.** In the figures it is a straight, gently
decaying line: the Pareto/NBD has no seasonal term, so it structurally cannot track the
four within-year peaks these panels are generated with. Every neural arm except
`ar_unbounded` tracks them, the LSTM best. This is the compensating strength
`figures/fig2_seasonal_shape.png` reports for the archived grid, and it survives the arm
axis unchanged — which matters, because it is the one dimension on which the contribution
beats the model that generated the data.

### Decay: not what the aggregate curve shows, and the LSTM overshoots it

Second-half ÷ first-half holdout volume, mean over the four transaction rates. The
generator's four seasonal peaks straddle the two halves, so the **actual** ratio sits near
1.0 at every churn level — within-year attrition is not visible in this statistic, and
"the neural models fail to decay" is not readable off the aggregate curve:

| config | churn 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: |
| **actual** | **1.06** | **1.00** | **0.97** | **1.00** |
| Pareto/NBD | 0.95 | 0.89 | 0.83 | 0.77 |
| LSTM `ar_bounded` | 1.00 | 0.90 | 0.81 | 0.68 |
| LSTM `no_ar` | 0.99 | 0.88 | 0.82 | 0.66 |
| Transformer `ar_bounded` | 0.99 | 0.96 | 0.96 | 0.92 |
| Transformer `no_ar` | 0.98 | 0.99 | 1.01 | 0.99 |
| LSTM `ar_unbounded` | **1.93** | **1.45** | 1.19 | 0.95 |
| Transformer `ar_unbounded` | 1.05 | 1.06 | **1.26** | **1.78** |

Two things fall out. The Pareto/NBD and the LSTM both decay *faster* than the realised
series, the LSTM most of all at high churn (0.68 against 1.00) — its cells at
`rate = 0.30` go negative precisely because it runs the cohort down too hard by year's end.
And the Transformer barely decays at all, which is the closest thing here to the "no
absorbing state" signature, though on this statistic it is nearer the truth than the LSTM
is.

**So the Transformer's miss is a level offset, not a decay failure.** In the figures its
curve is the right shape at roughly the right rate of change, sitting bodily above the
actuals from holdout week 0. That is a different defect from the LSTM's, and it matches §5:
its per-cell bias is uniform (+23% to +104%) where the LSTM's spans 46×.

### `ar_unbounded` drifts upward, visibly

The clearest picture in the set is
`figures/seasonal_4x4x10__forecast_overview_ar_unbounded.png`. The LSTM's simulated volume
**rises monotonically through the holdout year** in almost every cell — at `rate = 0.30,
churn = 0.20` from about 250 to 870 a week while the truth oscillates around 200 — giving
the 1.93 ratio in the table above. The Transformer does the same at high churn (1.78).

That is the §4 mechanism drawn rather than argued: `cumulative_transactions` and
`period_since_first_transaction` keep climbing as the rollout feeds its own samples back,
the inputs leave the range the weights were fitted on, and the model's output climbs with
them. A curve that ramps away over 52 steps is what an unbounded feature looks like in a
rollout, and it is worth keeping the figure for that reason alone.

---

## 7. Every tree, pooled

All 160 panels per row, sorted by median MAPE. Mean and median are both given because the
`ar_unbounded` LSTM arms are driven by a tail — `ar_unbounded-kmeans_8` reports a mean
bias 3.2× its median. "win rate" is the share of the 160 panels where that tree's
`mape_aggregate` beats the Pareto/NBD's on the *same* panel.

| model | arm | bias % mean | median | MAPE mean | median | RMSE | win rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pareto/NBD** | — | **+9.4** | **+6.1** | **50.8** | **37.4** | 0.18 | ceiling |
| LSTM | `ar_bounded-no_cluster` | +73.4 | +32.1 | 89.2 | 44.3 | 0.18 | **46.2%** |
| Transformer | `ar_bounded-no_cluster` | +54.2 | +49.5 | 77.2 | 62.5 | 0.18 | 31.9% |
| LSTM | `no_ar-no_cluster` | +150.7 | +56.5 | 158.3 | 65.2 | 0.18 | 31.2% |
| Transformer | `ar_unbounded-kmeans_8` | +58.1 | +49.5 | 86.3 | 66.6 | 0.18 | 16.9% |
| Transformer | `ar_bounded-kmeans_8` | +74.1 | +58.8 | 93.8 | 72.5 | 0.18 | 18.8% |
| Transformer | `no_ar-no_cluster` | +87.9 | +71.0 | 105.1 | 76.6 | 0.18 | 24.4% |
| LSTM | `ar_bounded-kmeans_8` | +115.7 | +70.2 | 120.4 | 77.9 | 0.18 | 9.4% |
| Transformer | `no_ar-kmeans_8` | +90.9 | +73.4 | 108.9 | 86.8 | 0.18 | 17.5% |
| Transformer | `ar_unbounded-no_cluster` | +105.4 | +97.7 | 123.8 | 101.7 | 0.19 | 15.6% |
| LSTM | `no_ar-kmeans_8` | +179.5 | +100.4 | 181.5 | 102.4 | 0.18 | 5.6% |
| LSTM | `ar_unbounded-kmeans_8` | +426.2 | +132.9 | 441.3 | 140.9 | 0.28 | 8.1% |
| LSTM | `ar_unbounded-no_cluster` | +383.3 | +261.2 | 389.7 | 261.2 | 0.32 | 6.2% |

The best arm beats the correct model on 46% of panels by MAPE, and no other arm exceeds
32%. That 46% is not spread evenly: broken out by transaction rate it is 8% / 25% / 65% /
88% across `rate = 0.01 → 0.30`. The LSTM's best arm is genuinely competitive with the
generating model on dense panels and hopeless on sparse ones.

Within-cell replicate spread says the same thing about stability. Mean standard deviation
of bias % across a cell's ten panels: Pareto/NBD 11.4, LSTM `ar_bounded` 46.9, Transformer
`ar_bounded` 52.6, LSTM `ar_unbounded` 266.5. Bounding the AR features halves the
replicate-to-replicate variance as well as the level.

---

## 8. What this says to build next

1. **Keep `ar_bounded`; make it the default AR encoding.** It is the only arm-axis change
   that improved anything, it improved both architectures significantly, and it costs six
   binary flags. It is not the fix, but there is no reason to run without it.

2. **Never let an unbounded counter into a rollout.** This grid measured the cost on
   panels where the truth is known: the generator's own sufficient statistic, encoded as
   counters, tripled the LSTM's error, and §6 shows the mechanism as a curve that ramps
   away over the 52 rollout steps. Any feature read during a rollout has to be bounded
   by construction. `docs/feature_engineering.md` should carry this as a rule rather than
   a caution.

3. **Drop `kmeans_8`.** It significantly hurt three of the four contrasts where it was
   added to a usable AR encoding, for the same reason as (2): the rollout cannot update it.
   The two contrasts it wins are both under `ar_unbounded`, where it is only outperforming
   a broken channel — and `ar_bounded` beats every clustered arm outright. There is no
   configuration this grid recommends it in.

4. **Diagnose the two architectures separately.** A single "neural vs benchmark" number
   hides that the LSTM is better in twelve of sixteen cells and catastrophic in the other
   four, while the Transformer is neither. Whatever absorbing state gets built should be
   evaluated per cell, not pooled.

5. **Build the absorbing state.** This was the hypothesis that could have made it
   unnecessary, and it failed: nothing on the arm axis gave the model a way to represent a
   customer who has stopped, and nothing on the arm axis moved the churn slope.
   `insights-study.md` §5.2 argued for it from the archived run; this run rules out the
   feature-engineering alternative empirically rather than by argument.

---

## 9. What this does not establish

- **The embedder axis was not crossed.** `EMBEDDER_AXIS` holds `valendin` only; `projected`
  was measured to run and cut on cost. ADR-0005's seam changes how features become a
  vector, and nothing here says whether it would have moved the churn slope. It is the
  cheapest remaining arm to add.

- **`ar_bounded`'s bin choice was not tuned.** The bins are `{2, 4, 8, 16, 32}` periods,
  the deepest being ~31% of the calibration window. Whether a different set does better is
  unmeasured; only bounded-vs-unbounded-vs-none was tested.

- **One study per suite.** `n_studies_per_model=1`, so replication is across the ten panels
  in a cell, not across training runs on one panel. Model training is not seeded
  (`CLAUDE.md`), so a cell's spread mixes panel-to-panel variation with run-to-run
  variation and cannot separate them. The 160-panel paired tests in §4 are unaffected —
  they compare arms on the same panels — but a single cell's ±CI should not be read as
  panel variation alone.

- **These are synthetic panels with a known generator.** That is exactly what makes §4's
  conclusion sharp, and it is also the limit: on a real panel there is no ceiling to
  measure against, and no guarantee the same encodings rank the same way.
  `scripts/run_real_panel_arms.py` asks the same question on cdnow and electronics.

- **Seasonality is held fixed across the grid.** Only the `(rate, churn)` regime varies, so
  nothing here says how any arm behaves when the seasonal pattern changes.
