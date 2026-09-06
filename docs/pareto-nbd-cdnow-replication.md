# Pareto/NBD on CDNOW: replicating Jerath, Fader & Hardie

Whether the Pareto/NBD's published near-perfect CDNOW result and this package's
archived Pareto/NBD bias figures describe the same model behaving differently on
different panels, or the same behaviour described by two different metrics.

**It is the second.** The published 1.35% and an aggregate bias of −11% are both
correct, are both about the Pareto/NBD on CDNOW, and are not the same quantity.

Read `CONTEXT.md` first for the vocabulary (*calibration*, *holdout*, *aggregate
bias*, *benchmark*). Unlike `docs/hurdle-models-vs-pareto-nbd.md`, which is a reading
exercise, everything here was **run**: one script, `.scratch/pnbd-cdnow-replication/replicate.py`,
produces every number below. Claims read from the paper rather than measured are
attributed to it inline; the one construction that was inferred rather than read is
marked **INFERRED** and restated in §10.

## Contents

1. [Why this was run](#1-why-this-was-run)
2. [What the paper actually reports](#2-what-the-paper-actually-reports)
3. [The cohort, rebuilt to their window](#3-the-cohort-rebuilt-to-their-window)
4. [Five configurations](#4-five-configurations)
5. [Check 1 — does the likelihood agree?](#5-check-1--does-the-likelihood-agree)
6. [Check 2 — their headline number reproduces](#6-check-2--their-headline-number-reproduces)
7. [Check 3 — the same model under this package's metric](#7-check-3--the-same-model-under-this-packages-metric)
8. [Why both numbers are true](#8-why-both-numbers-are-true)
9. [Two things this turned up about the repo](#9-two-things-this-turned-up-about-the-repo)
10. [What this does not establish](#10-what-this-does-not-establish)
11. [Reproducing](#11-reproducing)
12. [Sources](#12-sources)
13. [Open, in priority order](#13-open-in-priority-order)

---

## 1. Why this was run

Two numbers about the same benchmark sat in the docs without a bridge between them.

`docs/hurdle-models-vs-pareto-nbd.md` §2 quotes Jerath, Fader & Hardie's cumulative
aggregate MAPE of **1.35%** for the Pareto/NBD over a 39-week CDNOW holdout, and draws
the conclusion that "there is very little room above that" — the benchmark is close to
a ceiling.

The archived runs say something different. On electronics the Pareto/NBD sits at
**−53.4%** aggregate bias (`docs/loss-functions.md` §4.1) and **−63.7%**
(`docs/p-slstm.md` §10); on the synthetic seasonal grid, where it is the generating
process, at **+2.2% to +15.8%** (`docs/insights-study.md` §2). None of those is 1.35%,
and **no Pareto/NBD run on CDNOW existed in this repo at all** — the CDNOW work
(`Studies/loss_ablation_cdnow`, the AR-encoding shards) is neural-only.

So the comparison being drawn — a literature number for CDNOW against this package's
numbers for other panels — crossed both a panel boundary and, it turns out, a metric
boundary. This document closes the second one.

---

## 2. What the paper actually reports

Jerath, Fader & Hardie, *New Perspectives on Customer "Death" Using a Generalization of
the Pareto/NBD Model*, Marketing Science 30(5):866–880 (2011). The paper's subject is
their PDO model; the Pareto/NBD is its `τ → 0` limit and appears as the reference row.

Their §3.1, read from the PDF:

> This data set tracks 2,357 individuals who made their first-ever purchases at the
> CDNOW website in the first 12 weeks of 1997 and records their repeat purchasing
> through June 1998. The first 39 weeks of data are used for model calibration; the
> remaining 39 weeks of data are used as longitudinal holdout for model validation.

The three published quantities this replication targets:

| | value | where |
|---|---|---|
| Pareto/NBD MLE | `r = 0.55, α = 10.58, s = 0.61, β = 11.67` | Table 1 |
| log-likelihood | `−9,595.0` | Table 1, and stated in §3.1 |
| MAPE, weeks 40–78 | cumulative **1.35%**, weekly **20.89%** | Table 3 |

Two things the paper does *not* say, and both matter: it does not print the formula
behind the tracking curve those MAPEs are computed on, and it does not report an
aggregate bias or a per-customer error of any kind. Table 3's caption is "Measures of
Model Forecasting Performance"; the measures are those two MAPEs and nothing else.

---

## 3. The cohort, rebuilt to their window

The repo's committed CDNOW panel is already bucketed by the package's own calendar
(`docs/feature_engineering.md` §4 records the convention and its one-day divergence
from a plain seven-day grid), and it carries counts, not dates. A replication needs
the dates, so this starts from the canonical 1/10th sample rather than from
`Datasets/Dataset_clean/`.

CDNOW's window opens 1997-01-01 and closes 1998-06-30: 546 days, exactly 78 seven-day
weeks. Calibration is the first 273 days, so the holdout opens 1997-10-01 and runs 39
weeks to the end of the data.

Measured on that grid:

| quantity | value | matches the canonical figure |
|---|---|---|
| customers | 2,357 | yes — the 1/10th sample |
| raw records | 6,919 | yes — the file's own read-me |
| purchase **occasions** (distinct customer-days) | 6,696 | yes — `scripts/build_cdnow_panel.py` docstring |
| repeat transactions, weeks 1–39 | **2,457** | yes |
| repeat transactions, weeks 40–78 | **1,882** | yes |

An occasion, not a record, is Fader & Hardie's transaction unit: 223 of the 6,919
records share a customer-day with another. §5 shows what the likelihood does if they
are not collapsed.

---

## 4. Five configurations

All five run on that identical cohort and that identical 39/39 split. They differ on
two axes only — which estimator, and which metric — and the point is the pair of
contrasts that isolates each axis.

| | estimator | forecast construction | metric |
|---|---|---|---|
| **A** | this package's HB Gibbs sampler | posterior-averaged `λ·overlap(alive)` | holdout only |
| **B** | same, via `compute_pareto_predictions` | same | holdout only |
| **C** | same, on the committed repo panel | same | holdout only |
| **D** | textbook MLE | `Σᵢ E[Y(t) \| xᵢ, t_xᵢ, Tᵢ]` | holdout only |
| **E** | textbook MLE | `Σᵢ E[X(t)]`, population-level | JFH tracking |

**A against D** is the same metric by two estimators — it asks whether this package's
port of BTYDplus behaves like the maximum-likelihood Pareto/NBD the paper fits.

**D against E** is the same estimator under two metrics — it asks how much of the gap
between 1.35% and this package's numbers is metric definition rather than model
behaviour.

A exists separately from B because `compute_pareto_predictions` builds its own
sufficient statistics from a weekly panel; A feeds the same sampler a
BTYDplus-faithful CBS built from the event log at daily resolution, which is what the
paper's estimator sees. The gap between them is §9.

---

## 5. Check 1 — does the likelihood agree?

The Pareto/NBD likelihood is a sharp fingerprint of `(x, t_x, T)`, so before comparing
any forecast it is worth asking whether the sufficient statistics here are the paper's.
Evaluated at their published estimates, and then maximised:

| | r | α | s | β | LL |
|---|---|---|---|---|---|
| paper (Table 1) | 0.55 | 10.58 | 0.61 | 11.67 | **−9,595.0** |
| here, at their estimates | — | — | — | — | **−9,602.5** |
| here, maximised | **0.552** | **10.587** | 0.646 | 12.791 | −9,602.4 |

**The purchase process matches to three decimals. The death process does not, and the
gap is 7.5 log-points — 0.08%.**

That the mismatch is confined to `(s, β)` is not a coincidence. The paper's own Table 2
sweeps the PDO period-length parameter and prints `β` ranging between 2.02 and 115.29
while the log-likelihood moves under 14 points: the death parameters sit on a nearly
flat ridge. The same flatness shows up in §7 as sampler seed spread.

The residual gap was probed against the obvious data-prep choices and none of them
closes it — the paper's convention is the best of the four:

| same-day records | calibration ends | `Σx` | LL at their estimates |
|---|---|---|---|
| **collapsed** | **day 273 (Oct 1)** | **2,457** | **−9,602.5** |
| collapsed | day 274 | 2,462 | −9,622.7 |
| kept separate | day 273 | 2,603 | −9,795.3 |
| kept separate | day 274 | 2,608 | −9,815.7 |

Since the maximised likelihood here (−9,602.4) is *below* the value the paper reports
at a specific parameter vector, some difference in their `(x, t_x, T)` remains
unaccounted for. It is small, and it does not move any conclusion below, but it is
unresolved.

---

## 6. Check 2 — their headline number reproduces

A "cumulative MAPE over weeks 40–78" is not a holdout quantity. It is the mean relative
error of a curve that runs from week 1, evaluated on its last 39 points — so at week 40
the denominator already carries the 2,457 calibration repeat transactions. The curve
itself is the population-level `E[X(t)]` summed over the cohort with each customer's
clock starting at their own trial purchase: an aggregate cohort projection, not a
per-customer forecast. **INFERRED** — see §10.

Built that way (config E):

| parameters | cum. @ wk 39 | cum. @ wk 78 | cumulative MAPE | weekly MAPE |
|---|---|---|---|---|
| actual | 2,457 | 4,339 | — | — |
| paper's estimates | 2,510.4 | 4,231.6 | **1.68%** | **19.35%** |
| MLE fitted here | 2,523.8 | 4,230.7 | 1.65% | 19.35% |
| **JFH Table 3** | — | — | **1.35%** | **20.89%** |

**That is a replication.** 1.68 against 1.35 and 19.35 against 20.89, with the residual
attributable to the `(s, β)` gap of §5 and to whatever exact curve the paper drew. The
construction is right; the model is right; the periods are right.

---

## 7. Check 3 — the same model under this package's metric

Now score the identical fits the way `models.monte_carlo_forecasting.compute_forecast_metrics`
scores one — the holdout window alone, aggregate predicted against aggregate actual,
against the 1,882 repeat transactions of weeks 40–78.

Configs A, B and C are five seeds each, mean ± sd across them; D is deterministic.

| config | predicted total | aggregate bias % | cumulative MAPE % | weekly MAPE % |
|---|---|---|---|---|
| **A** package sampler, faithful CBS | 1,674.1 ± 36.1 | **−11.05 ± 1.92** | 11.38 ± 1.34 | 19.47 ± 0.05 |
| **B** `compute_pareto_predictions`, JFH-week panel | 1,637.6 ± 49.6 | −12.99 ± 2.64 | 13.28 ± 1.77 | 19.62 ± 0.19 |
| **C** `compute_pareto_predictions`, repo panel | 1,602.0 ± 47.8 | −13.78 ± 2.57 | 13.62 ± 1.74 | 17.80 ± 0.74 |
| **D** MLE at the paper's own estimates | 1,646.6 | **−12.51** | 12.50 | 19.51 |
| *(E, for reference: the tracking curve's own week-40–78 increment)* | *1,721.2* | *−8.55* | — | — |

Three readings.

**The port is fine.** A at −11.05% and D at −12.51% are two different estimators —
hierarchical-Bayes Gibbs against maximum likelihood — on the same data under the same
metric, 1.5 points apart, with A's seed spread alone covering 5.5 points. Nothing in
the gap between 1.35% and this table is attributable to `benchmarks/pareto_nbd.py`.

**Pareto/NBD under-predicts CDNOW by 11–14%.** Directionally the same as electronics,
where the archived runs put it at −53% and −64%, but roughly a fifth the size. The
"Pareto/NBD under-predicts" claim survives the move to CDNOW; the *magnitude* does not
travel, and quoting the electronics figure as though it characterised the model would
overstate it about fivefold.

**The sampler is not near-deterministic here.** Config A's per-seed biases are −8.50,
−9.84, −10.61, −13.99, −12.31 — and seeds 45 and 46 come out low in A, B *and* C alike.
That is the flat `(s, β)` ridge of §5 seen from the posterior: different chains settle
on different death processes that the likelihood barely distinguishes, and the
aggregate forecast moves 5.5 points as a result. `docs/p-slstm.md` §10 records
Pareto/NBD on electronics at ±0.35 over three seeds and calls it "near-deterministic";
on CDNOW, at these MCMC settings, it is not. **A single Pareto/NBD fit on CDNOW is not
a reportable number.**

---

## 8. Why both numbers are true

Three things separate the two numbers, and all three flatter the published one.

Take the paper's own parameters. Over weeks 40–78 the tracking curve accumulates
`4,231.6 − 2,510.4 = 1,721.2` expected repeat transactions against 1,882 actual — a
shortfall of **161**. That shortfall is real and it is present in both metrics. What
differs is what it is divided by, and what it is netted against:

- **This package's aggregate bias** divides it into the holdout total alone, which runs
  from zero to 1,882: `−161 / 1,882 = −8.5%`.
- **JFH's cumulative MAPE** divides it into a running total that starts at 2,457 and
  ends at 4,339 — and first nets it against the calibration window, where the same
  curve *over*-predicts by 53 (2,510.4 against 2,457). So the cumulative error at week
  78 is `161 − 53 = 107.4`, which is `2.5%` of 4,339, and averaging the 39 points of a
  curve whose error starts near zero and grows to that gives 1.35%.

That is two of the three: a denominator inflated by the calibration mass, and a partial
cancellation between an over-predicted calibration window and an under-predicted
holdout.

The third is conditioning. JFH's curve is unconditional: it asks what a
customer `w` weeks past their trial purchase is expected to have bought, ignoring what
they actually did in the calibration window. This package's forecast is conditional on
each customer's `(x, t_x, T)`, which is what a per-customer forecast has to be. On this
data the conditional version is the more pessimistic one: −12.5% against −8.5% at
identical parameters.

**Neither metric is wrong.** They answer different questions — "does this model
reproduce the cohort's sales trajectory" against "how many transactions will these
customers make next year". This package asks the second, because the thesis compares
per-customer forecasts. But the published 1.35% is not evidence about the second
question, and `docs/hurdle-models-vs-pareto-nbd.md` §2 read it as though it were.

---

## 9. Two things this turned up about the repo

**(a) The committed CDNOW panel is not JFH's holdout.** It carries 38 holdout periods
and 1,858 transactions where the published split has 39 and 1,882: the builder trims to
complete weeks under the package's calendar (`scripts/build_cdnow_panel.py`, "the grid
is trimmed to complete weeks") and 1998-06-25..30 falls outside the last complete one.
The trimming is correct and deliberate — a partial week would read as a demand drop —
but any comparison between a repo CDNOW number and a published CDNOW number is off by
that week and those 24 transactions.

**(b) Config B loses ~2 points of bias against config A on identical data.** Same
sampler, same seeds, same cohort, same 39-week holdout; the only difference is that B
goes through `compute_pareto_predictions`, which builds its own CBS from the weekly
panel. Two documented choices in `_build_cbs` explain it, and they were not separated
here:

- **Occasions are collapsed to active periods.** 201 of the 4,588 active calibration
  weeks hold more than one occasion, so `Σx` falls from 2,457 to 2,231 — 9% fewer
  repeat events, and therefore a lower fitted purchase rate. The docstring states this
  ("two transactions in the same week count as a single repeat event") and argues it
  from within-period timing being unknown, which is right for a weekly panel and wrong
  for reproducing a daily-resolution published fit.
- **`cal_end` is the last calibration period's `period_start`, not its end.** With
  `T = (cal_end − first)/P` and holdout period `t` spanning `[T+t−1, T+t]`, forecast
  column 0 covers the calendar week beginning at `cal_end` — the last *calibration*
  week. `scripts/validate_pareto_benchmark.py` does not catch this: it builds its
  calibration slice with `period_start <= cal_cut` and hands R the same `cal_cut` as
  `T.cal`, so both sides shift together and the gate passes.

The first is a documented modelling choice and defensible: within-period timing really
is unknown in a weekly panel. The second is not a choice anyone wrote down, and it
applies to the production path too — `pareto_from_data` passes `data["train_panel"]`,
whose last `period_start` is the last calibration period's start — so the benchmark's
forecast column 0 lines up against holdout period 0 while covering the calendar week
before it. That reads as an off-by-one, but it was not isolated here: the ~2 points
above are the two effects together, and the arithmetic could plausibly be defended as
"age measured to the start of the last complete period". **Isolating them is a two-fit
experiment and it has not been run.** Until it is, treat this as a flagged suspicion,
not a confirmed defect.

---

## 10. What this does not establish

- **INFERRED: the tracking construction.** JFH state that they "create total repeat
  sales forecasts" and report MAPEs of "cumulative total repeat sales and weekly total
  repeat sales over weeks 40–78". They do not print the expression. §6 uses the
  population-level `E[X(t)]` accumulated from week 1 because that is the standard Fader
  & Hardie tracking plot and because it reproduces their numbers; that is an argument
  from agreement, not from the text. A different construction that also lands near
  1.35% cannot be ruled out.
- **The 7.5 log-point likelihood gap is unresolved** (§5). `r` and `α` match to three
  decimals and the maximised likelihood here is below the value they report, so some
  difference in `(x, t_x, T)` remains. It is 0.08% and it moves nothing here.
- **Only the Pareto/NBD row of Table 3 was targeted.** The PDO rows, which are the
  paper's actual contribution, were not implemented or checked.
- **Five seeds is few**, on a posterior whose death process is weakly identified. The
  ±1.92 in §7 is a small-sample estimate of a spread that is genuinely wide.
- **One dataset, one split, one cohort.** Nothing here transfers to electronics or to
  the synthetic grid except by argument.
- **No neural model was run.** This constrains the Pareto/NBD side of an LSTM-versus-
  benchmark comparison on CDNOW and says nothing about the other side; the LSTM's own
  CDNOW bias figures live in `docs/loss-functions.md` §6 and `docs/insights-study.md` §4.3.
- **MCMC settings are the package's defaults** (`mcmc=2500, burnin=500, thin=50,
  chains=2`), which is what the studies use. Longer chains would narrow §7's seed
  spread and were not run.

---

## 11. Reproducing

The raw dataset is **not committed** — `Datasets/` is gitignored and holds only the
bucketed panel. Fetch the canonical 1/10th sample, which is the file
`scripts/build_cdnow_panel.py` is written against:

```bash
curl -sSO https://www.brucehardie.com/datasets/CDNOW_sample.zip
unzip -o CDNOW_sample.zip            # -> CDNOW_sample.txt, 6,919 records
```

Then, with the project venv's interpreter:

```bash
<venv>/bin/python .scratch/pnbd-cdnow-replication/replicate.py --raw CDNOW_sample.txt
```

Roughly 75 s per seed for the three sampler configurations, so ~7 minutes at the
default five seeds; the MLE and tracking checks are instant. Per-configuration
per-seed numbers land in `.scratch/pnbd-cdnow-replication/results.json`.

| Path | What |
|---|---|
| `.scratch/pnbd-cdnow-replication/replicate.py` | All five configurations and both metrics |
| `.scratch/pnbd-cdnow-replication/results.json` | Per-seed raw numbers behind §5–§7 |

---

## 12. Sources

| Claim | Source |
|---|---|
| CDNOW cohort of 2,357, first 39 weeks calibration / remaining 39 weeks holdout; Pareto/NBD `r, α, s, β` and LL = −9,595.0 (Table 1); `β` moving 2.61–115.29 across the τ sweep at near-constant LL (Table 2); cumulative 1.35% / weekly 20.89% MAPE over weeks 40–78 (Table 3) | Jerath, Fader & Hardie, *Marketing Science* 30(5):866–880 (2011) — https://business.columbia.edu/sites/default/files-efs/pubfiles/6057/customer_death.pdf (doi:10.1287/mksc.1110.0654) |
| CDNOW 1/10th systematic sample: 2,357 customers, 6,919 records, purchases through 1998-06-30 | Fader & Hardie — https://www.brucehardie.com/datasets/CDNOW_sample.zip and its read-me |
| Pareto/NBD likelihood, `P(alive)`, `E[Y(t)|x,t_x,T]` and the `A₀` integral term | Hardie, derivation note 009 — http://www.brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf |
| The estimator being reproduced: BTYDplus `pnbd.mcmc.DrawParameters`, `elog2cbs` at occasion granularity | `src/panelclv/benchmarks/pareto_nbd.py`, ported from https://github.com/mplatzer/BTYDplus |
| Pareto/NBD at −53.4% aggregate bias on electronics | `docs/loss-functions.md` §4.1, from the archived study suite |
| Pareto/NBD at −63.7% ± 0.35 on electronics, n=3 | `docs/p-slstm.md` §10 |
| Pareto/NBD at +2.2% to +15.8% on the synthetic seasonal grid | `docs/insights-study.md` §2 |
| CDNOW panel trimmed to complete weeks under the package calendar | `scripts/build_cdnow_panel.py`; convention in `docs/feature_engineering.md` §4 |

---

## 13. Open, in priority order

1. **Isolate §9(b).** Two fits: `compute_pareto_predictions` with the occasion collapse
   removed, and with `cal_end` advanced by one period. If the second carries most of the
   2 points, every archived Pareto/NBD number in the repo is shifted by one period and
   the electronics figures move too.
2. **Run the neural models on JFH's split.** The Pareto/NBD side of a CDNOW comparison
   now has a number; the LSTM side does not, on this cohort. `docs/insights-study.md`
   §4.3's CDNOW arms use the repo panel and the package's own metric, which is the right
   metric but the wrong 38-week window.
3. **Longer chains.** §7's ±1.92 over five seeds at the default `mcmc=2500` is wide
   enough that a single archived Pareto/NBD fit on CDNOW should not be quoted. Whether
   that is fixable by sampling more or is genuine posterior width is untested.
