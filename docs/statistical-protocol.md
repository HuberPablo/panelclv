# Statistical protocol

How every comparative claim in this thesis is tested and worded. There is one estimand,
one interval and one decision rule. The only thing that changes from one experiment to
the next is how the bootstrap resamples, because that follows the way the experiment was
built.

The implementation is `panelclv.evaluation.effects.effect`. It is the only one, and
every doc's numbers come from it.

## 1. The evidence: what one replication is

A claim compares two conditions: **A**, the baseline, and **B**, the change. Each
condition has **n replications**, and each replication gives one value of the metric.
Every result states its n. 20 is the usual value but not a rule, and A and B may have
different n.

What counts as one replication depends on the experiment:

- **A real-panel study.** One replication is one complete Optuna search, followed by its
  refit and forecast. The trials inside a search are **not** replications. A search with
  100 trials still counts once. Training is unseeded, so the searches are independent
  draws from the training-and-selection process.
- **The synthetic grid.** One replication is one generated panel. A (rate × churn) cell
  holds 10 panels drawn independently from the same Pareto/NBD process.
- **Pareto/NBD on a real panel.** One replication is one seeded refit. Pareto/NBD is
  replicated in the same way as a neural model, so it is compared like any other
  condition.

What each interval measures follows from that:

- On a real panel, it measures how robust the effect is to the randomness of training
  and hyperparameter search on that panel.
- On the grid, it measures how robust the effect is to drawing new panels from the same
  process.

## 2. Independent or paired

Look for an experimental correspondence **A_i ↔ B_i**, a unit that both sides were
measured on. Sharing the same dataset is **not** enough.

| experiment | replication | A_i ↔ B_i? | bootstrap |
|---|---|---|---|
| Real panel: n searches for A, n new searches for B | search | no | independent |
| Real panel: a neural model against n seeded Pareto/NBD refits | search / refit | no | independent |
| Synthetic grid: A and B fitted on each of a cell's 10 panels | generated panel | yes | paired |
| The same searches or weights, scored in two ways | search | yes | paired |
| The same seed list driving both arms, where the doc establishes the coupling | seed | yes | paired |

- **Independent.** Resample A's n values and B's n values separately, each with
  replacement. Run *i* of A has nothing in common with run *i* of B.
- **Paired.** Compute the per-unit differences d_i = B_i − A_i and resample the units, so
  each pair travels together. Δ is the same number either way, because the mean of the
  differences equals the difference of the means. Pairing only removes the variation
  from unit to unit. On the grid that matters a great deal: one cell's MAPE can run from
  50 to 1,250 across its panels, while the effect being tested is a few points on every
  panel.
- **One statistic per replication.** Some claims concern a single number rather than a
  difference, such as each study's rank correlation between a selection criterion and
  the holdout. Those are the paired case against a reference value, which is 0 unless
  the claim states another.

## 3. The test

1. **Δ = M̄_B − M̄_A.**
2. Its **95% percentile-bootstrap interval**, from 10,000 resamples, drawn independently
   or paired as §2 says.
3. **A claim is supported if and only if 0 lies outside the interval.**

p-values are not the criterion. If one is shown at all, it is supplementary. Rank tests
(Wilcoxon, Mann–Whitney), Hodges–Lehmann shifts, false-discovery corrections, sign
counts and "outside one SD" heuristics are not used to make claims. A rank test beside a
mean-difference interval would be testing a different quantity from the one reported.
Counts such as "B wins in 14 of 20 studies" may be shown, but only as description.

## 4. Which metric may carry which claim

| metric | role | claim it may carry |
|---|---|---|
| Spearman (per customer) | primary: ranking customers | superiority, by §3 |
| Aggregate MAPE | primary: level accuracy | superiority, by §3 |
| Signed bias | secondary: direction of error | over- or under-forecasting, only if its own interval excludes 0; \|bias\| when comparing calibration accuracy |
| RMSE | descriptive | none; on these sparse panels every forecast sits next to the all-zero forecast |
| Forecast CV (std / mean of predicted totals) | diagnostic, reported beside Spearman | none; CV near 0 flags a collapse |

When metrics point in different directions, for example Spearman improves while RMSE
worsens slightly, that is not a contradiction. Each metric answers only its own question.

## 5. Refit noise gives magnitude, not significance

Refitting the same model with nothing else changed still moves the forecast. The size of
that movement on each panel is its refit noise. It is printed beside Δ so a reader can
tell whether an effect is large or small. It is **not** a second threshold. An effect
whose interval excludes 0 but whose Δ is smaller than the refit noise is described as
"supported and small".

## 6. "No clear difference" is not "equivalent"

- An interval that contains 0 means **no clear difference was found at this n**. It
  never means the conditions are the same.
- An equivalence claim needs a margin *m* that is **named before the result is looked
  at**. The refit noise is the natural choice. The claim is supported only when the
  whole 95% interval lies inside [−m, +m], and it is worded "no difference larger than
  ±m is detectable at n = …".

## 7. One panel, one cell at a time

The test is run separately on each real panel and on each grid cell. Panels, or cells,
are never pooled into one bootstrap, and no cross-panel rule is used (such as
"significant in at least two of four"). Across panels, or across rates and churn levels,
the pattern is described in words. Where panels disagree, that is heterogeneity to
explain, not noise to average away.

## 8. How a result is reported

The table that `effects.table()` prints:

| comparison | n (A / B) | mean A | mean B | Δ | 95% CI | supported | refit noise |
| --- | :---: | ---: | ---: | ---: | :---: | :---: | ---: |

Wording templates:

- **Supported:** "B lowers MAPE on electronics (Δ −8.4, 95% CI −12.1 to −4.7, n = 20 / 20)."
- **Not supported:** "No clear difference in MAPE on gift at n = 20 (Δ −3.2, 95% CI −7.5
  to +0.8)."
- **Equivalence:** "No difference larger than ±5 MAPE points is detectable at n = 20."
- **Across panels:** "The improvement appears on electronics and CDNOW but is not clearly
  reproduced on gift or multichannel."

## 9. The protocol in one paragraph

This is the version for the thesis's methods section:

> Every comparison between two conditions A and B rests on replications of the whole
> pipeline: independent training and hyperparameter searches on a real panel, or
> independently generated panels on the synthetic grid. The number of replications, n,
> is stated with each result. Effects are reported as the difference in mean
> performance, Δ = M̄_B − M̄_A, with a 95% percentile bootstrap confidence interval from
> 10,000 resamples. When the conditions are independent replications, each is resampled
> separately. When they share units (the same generated panels, or the same searches
> scored two ways), the units are resampled with their pairs intact. A comparative claim
> is considered statistically supported when the interval excludes zero; p-values, where
> shown, are supplementary. Spearman correlation and aggregate MAPE are the primary
> metrics for customer discrimination and level accuracy; bias is secondary and RMSE is
> descriptive. Refit variability is reported to put effect sizes in context and is not
> an additional significance threshold. An interval that contains zero is read as no
> clear difference at that n, never as equivalence, which requires a margin fixed in
> advance. Results are assessed separately for each panel and each grid cell, and
> disagreement between them is interpreted as heterogeneity rather than combined into a
> pooled test.
