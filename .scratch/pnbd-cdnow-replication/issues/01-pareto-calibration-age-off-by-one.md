# 01 — Pareto/NBD measures calibration age to the start of the last period, so every forecast is shifted one period early

Status: ready-for-human

Closes open item 1 of `docs/pareto-nbd-cdnow-replication.md` §13, which asked for this to
be isolated. It has been. The suspicion flagged in §9(b) of that document is **confirmed
as a real one-period misalignment**, on every panel and both calibrations, and its size is
measured below. What is left is a decision, not an investigation — hence
`ready-for-human`.

## The defect

`benchmarks/pareto_nbd._build_cbs` anchors each customer's age on

```python
cal_end = panel[time_col].max()          # time_col = "period_start"
cbs["T_cal"] = (cal_end - first_t).dt.days / period_in_days
```

`period_start` is the **first day of a period** (`data_preparation.panel_dataset.add_period_start`),
so `cal_end` is the *start* of the last calibration period, not its end. `T_cal` is
therefore one period short of the customer's true age at the calibration cut.

The forecaster puts holdout period `t` on customer time `[T_cal + t - 1, T_cal + t]`, so
**forecast column 0 covers the last calibration period** — a period the model was fitted
on — while it is scored against holdout period 0. Every column is one period early.

`.scratch/pnbd-cdnow-replication/check_window_alignment.py` prints the calendar span of
forecast column 0 against the holdout panel's own first period. Run 2026-09-17:

| calibration | panel | last cal period | forecast col 0 covers | holdout col 0 is | shift |
| --- | --- | --- | --- | --- | ---: |
| 2y | cdnow | 1997-09-23 | [1997-09-23, 1997-09-30) | [1997-09-30, 1997-10-07) | 1 period |
| 2y | electronics | 2000-12-22 | [2000-12-22, 2000-12-29) | [2001-01-01, 2001-01-08) | 1 period |
| 2y | gift | 2003-02-18 | [2003-02-18, 2003-02-25) | [2003-02-25, 2003-03-04) | 1 period |
| 2y | multichannel | 2006-12-23 | [2006-12-23, 2006-12-30) | [2007-01-01, 2007-01-08) | 1 period |
| 3y | electronics | 2001-12-23 | [2001-12-23, 2001-12-30) | [2002-01-01, 2002-01-08) | 1 period |
| 3y | gift | 2004-02-18 | [2004-02-18, 2004-02-25) | [2004-02-25, 2004-03-03) | 1 period |
| 3y | multichannel | 2007-12-23 | [2007-12-23, 2007-12-30) | [2008-01-01, 2008-01-08) | 1 period |

Not panel-specific and not calibration-specific: exactly one period, everywhere.

## Blast radius

`studies.runner` fits the benchmark through one call, with `time_col="period_start"`
hard-coded, and `benchmarks.pareto_from_data` passes `data["train_panel"]`, whose last
`period_start` is the last calibration period's start. **Every Pareto/NBD number in the
repository came through this path** — the four-panel benchmark tables, the three-year
control, the electronics figures in `docs/loss-functions.md` §4.1 and `docs/p-slstm.md`
§10, and the synthetic-grid rows in `docs/insights-study.md` §2.

## How much it is worth

`.scratch/pnbd-cdnow-replication/measure_window_shift.py`, 2026-09-17. Three arms on the
2-year benchmark windows and cohort, paired on seed, scored by
`compute_forecast_metrics`: `baseline` (the production path as archived), `shift`
(`T_cal + 1`, the corrected age), `no_collapse` (`x` counting transactions rather than
active periods — the *other* `_build_cbs` choice §9(b) could not separate from this one).
The `baseline` seed-42 rows reproduce the archived `results.csv` to every digit, so the
harness is the production fit.

**CDNOW** (39 calibration weeks), aggregate bias %:

| seed | baseline | shift | Δ | no_collapse | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | −15.97 | −22.03 | −6.06 | −10.52 | +5.45 |
| 43 | −15.08 | −22.43 | −7.35 | −9.63 | +5.44 |
| 44 | −11.50 | −19.24 | −7.74 | −5.16 | +6.34 |
| **mean** | **−14.18** | **−21.23** | **−7.05** | **−8.44** | **+5.74** |

MAPE follows: 20.44 → 24.06 under `shift`, → 18.84 under `no_collapse`. Spearman moves
0.449 → 0.441 → 0.436, i.e. **ranking is untouched**; this is a level effect only.

**Electronics** (104 calibration weeks), aggregate bias %:

| seed | baseline | shift | Δ |
| ---: | ---: | ---: | ---: |
| 42 | −63.02 | −64.20 | −1.19 |
| 43 | −63.21 | −64.01 | −0.80 |
| 44 | −62.99 | −64.86 | −1.88 |
| **mean** | **−63.07** | **−64.36** | **−1.29** |

MAPE 65.69 → 66.69; Spearman 0.304 → 0.312.

Three readings:

- **The shift makes the level worse, not better.** Correcting it ages every customer by
  one period, which lowers both the fitted purchase rate and `P(alive)`, and moves the
  forecast window one period further from the fit. Pareto/NBD already under-predicts; the
  corrected fit under-predicts *more*. The archived numbers are flattering.
- **Its size scales inversely with the calibration length**, as it must: one period is
  1/39 of CDNOW's window and 1/104 of electronics', and the effect is 7.1 against 1.3
  points. Gift and multichannel are 104-week panels, so ~1 point is the expectation there
  too; it has not been measured.
- **It is far larger than the ±2 points §9(b) attributed to both effects together, because
  the two effects fight.** The occasion collapse pushes the CDNOW level *up* by 5.7 points
  while the shift pushes it *down* by 7.1, so the baseline sits between them and their net
  is small. That is consistent with config C landing ~2 points from the faithful config A
  in the replication, though those two ran on the earlier 38-week panel and this does not
  prove the cancellation is exact. **Two offsetting errors, not one small one** — which
  means fixing either alone moves the reported level more than fixing both.
- **`no_collapse` on electronics does not run**, exactly as the `_build_cbs` docstring
  warns (169 of 829 customers get `x > 0` with `t_x = 0`, which the likelihood has no
  density for). The arm floods `invalid value encountered in multiply` from `p_alive` and
  `overflow encountered in divide` from the level-1 Gamma draw, and had produced nothing
  after 8 minutes against ~15 s for every other fit, so it was stopped. It is not a
  candidate fix; it is in the table only to bound the collapse's contribution on CDNOW,
  where the panel is nearly one item per active week and the sampler is well behaved.

## The fix, and what it costs

One line in `_build_cbs`: measure `T_cal` to the end of the last calibration period.

```python
cbs["T_cal"] = ((cal_end - first_t).dt.days + period_in_days) / period_in_days
```

`t_x` needs no change — it is a difference of two `period_start` values, so the anchor
cancels.

**ADR-0004 does not block this.** The ADR says frozen means the architecture, and
explicitly exempts "the plumbing that hands it data"; `_build_cbs` is that plumbing. The
gate is that `scripts/validate_pareto_benchmark.py` still lands in its band.

**But the gate has to move with it.** The gate hands R `T.cal = cal_cut` — the same date it
slices the panel by, i.e. the last `period_start` — so BTYDplus computes the same
one-period-short age and both sides agree. That is why the gate never caught this. Change
`_build_cbs` alone and the gate will start failing against an R side that is now wrong;
`cal_cut` must advance by one period there too, and that edit is part of the fix, not a
workaround for it.

## The decision this needs

Fixing the code invalidates every archived Pareto/NBD row. Three options, in increasing
cost:

1. **Document only.** Leave the code and carry the caveat wherever a Pareto number is
   quoted (now done in `docs/benchmarks-real-panels.md` § Setup and §9(b) of the
   replication doc). Cheapest, and the numbers stay comparable across the whole archive —
   but the thesis then reports a benchmark whose forecast window is knowingly misaligned.
2. **Fix and refit the reported panels.** Pareto/NBD is one MCMC fit per panel, ~15–17 s
   each, so the four 2-year and three 3-year benchmark rows are minutes of compute. The
   cost is editorial: every table quoting a Pareto row needs re-scoring, and the CDNOW
   level moves by 7 points in the direction that makes the benchmark look worse.
3. **Fix, refit, and re-examine the occasion collapse at the same time.** The two are
   offsetting, so changing one alone moves the level more than changing both. This is the
   honest version and the largest edit.

Whichever is chosen, note that a **single Pareto/NBD fit on CDNOW is not a reportable
number** regardless — the death process sits on a flat likelihood ridge and the baseline
seed spread here is 4.5 points over three seeds (−11.50 to −15.97), already comparable to
the shift itself. Any refit of CDNOW should report a seed distribution.

## Reproducing

With the project venv's interpreter, from the repo root (the package is not installed in
the venv, so `src` goes on the path):

```bash
PYTHONPATH=src "$VIRTUAL_ENV/bin/python" .scratch/pnbd-cdnow-replication/check_window_alignment.py
PYTHONPATH=src "$VIRTUAL_ENV/bin/python" .scratch/pnbd-cdnow-replication/measure_window_shift.py cdnow,electronics 42,43,44
```

The first prints the alignment table (seconds). The second runs 18 fits, ~15–17 s each,
and writes `.scratch/pnbd-cdnow-replication/window_shift_results.csv`.

## Comments

**2026-09-17 — opened.** Mechanism confirmed by execution on all seven panel/calibration
combinations; magnitude measured on CDNOW and electronics, paired on three seeds. Gift and
multichannel were not measured — both are 104-week panels, so ~1 point is the expectation,
but that is an argument, not a measurement.
