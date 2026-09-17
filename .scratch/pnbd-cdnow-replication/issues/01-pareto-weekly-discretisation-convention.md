# 01 — The Pareto/NBD's weekly clock: not an off-by-one, a within-week convention worth nothing measurable

Status: wontfix

Closes open item 1 of `docs/pareto-nbd-cdnow-replication.md` §13, and **overturns the
suspicion it was opened on**. §9(b) of that document flagged `cal_end` being the last
calibration period's `period_start` as a probable off-by-one. It is not one. The
arithmetic is self-consistent, and the residual convention effect is smaller than the
sampler's own seed noise.

An earlier version of this file (commit `5496d59`) called it a confirmed defect worth 7
points of bias and asked for a decision about refitting the archive. That was wrong; the
reasoning error is recorded in "Why the first reading was wrong" below so it is not
repeated.

## What the code does

The Pareto/NBD is a **continuous-time** model. It cannot read a weekly panel directly: it
reduces each customer to `x` (repeat purchases), `t_x` (when the last one was) and `T_cal`
(how long we watched them), the last two real-valued durations. Building those from weekly
buckets forces a choice about where inside a week a transaction sits, and that choice is
never stated in `benchmarks/pareto_nbd._build_cbs` — which is the entire substance of this
ticket.

`_build_cbs` measures every duration between week **labels** (`period_start`):

```
T_cal = (last calibration week's label − customer's first active week's label) / 7 days
t_x   = (customer's last active week's label − first active week's label) / 7 days
```

A 4-week calibration, 2-week holdout toy, one customer active in calibration weeks 1 and
3, run through the real function:

| | |
| --- | --- |
| calibration week labels | 2020-01-06, 01-13, 01-20, 01-27 |
| holdout week labels | 2020-02-03, 02-10 |
| `_build_cbs` output | `x = 1`, `t_x = 2.0`, `T_cal = 3.0` |

The forecaster then puts holdout period `t` on customer time `[T_cal + t − 1, T_cal + t]`,
so column 0 is `[3.0, 4.0]` and column 1 is `[4.0, 5.0]`. **Whether that is right depends
entirely on where customer time 0 sits**, and both readings are internally coherent:

| a week's transactions occur at... | time 0 is | `T_cal = 3` means observation ends | forecast column 0 covers |
| --- | --- | --- | --- |
| **the end of the week** | 2020-01-13 | 2020-02-03 — the end of calibration ✓ | 02-03 → 02-10, **holdout week 1** ✓ |
| the start of the week | 2020-01-06 | 2020-01-27 — the *start* of the last calibration week | 01-27 → 02-03, the **last calibration week** ✗ |

Because `t_x` and `T_cal` are both differences of week labels, the anchor cancels out of
both. **The code is exact under the end-of-week reading**, and that is the reading it
implements. There is no misalignment and nothing to fix.

Reproduce the toy with `.scratch/pnbd-cdnow-replication/check_window_alignment.py`, which
prints the same thing for all four real panels and both calibrations.

## What is actually there

End-of-week is the *shortest* admissible exposure. It places each customer's first
purchase at the latest instant its bucket allows, so `T_cal` is as small as the data
permits — and `T_cal` is the denominator of the purchase rate and the clock the death
process runs on. A shorter window means a higher fitted rate and a higher survival
probability, so among the admissible conventions this one **maximises** the forecast.

The reference implementation does not face the choice: BTYDplus works from an event log at
daily resolution, so a first purchase carries its real timestamp. Against that, bucketing
to week-ends shortens each customer's observed life by however far into its week that
first purchase actually fell — uniformly distributed, so **half a week on average**.

`scripts/validate_pareto_benchmark.py` cannot see this, and says so: it deliberately
buckets the R event log to whole weeks before handing it to `elog2cbs`, precisely so both
sides get identical `(x, t_x, T_cal)` and the gate tests the sampler rather than the
discretisation ("R's elog2cbs counts occasions at daily resolution otherwise"). The gate is
doing what it claims; this is simply outside its scope.

## What the convention is worth: nothing measurable

`.scratch/pnbd-cdnow-replication/measure_window_shift.py`, CDNOW, 2-year benchmark windows
and cohort, paired on seeds 42–44. `half` adds 0.5 periods to `T_cal` — the daily-resolution
equivalent. `full` adds 1.0, the start-of-week reading, kept only to bound the sensitivity.
The `baseline` seed-42 row reproduces the archived `results.csv` to every digit.

| seed | baseline | `half` (+0.5) | Δ | `full` (+1.0) | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | −15.97 | −18.62 | −2.65 | −22.03 | −6.06 |
| 43 | −15.08 | −11.93 | **+3.15** | −22.43 | −7.35 |
| 44 | −11.50 | −10.91 | +0.59 | −19.24 | −7.74 |
| **mean** | **−14.18** | **−13.82** | **+0.36** | **−21.23** | −7.05 |

**The realistic correction is worth +0.36 points of bias and does not even have a
consistent sign across seeds.** It is swamped by the sampler: the baseline alone spans 4.5
points over the same three seeds (−11.50 to −15.97), because CDNOW's death parameters sit
on a nearly flat likelihood ridge (§5 of the replication doc; the paper's own Table 2 moves
`β` from 2.02 to 115.29 for under 14 log-points). Spearman is untouched throughout
(0.449 → 0.446).

The full-week column is not a correction of anything — it is the wrong convention applied
deliberately — and it is listed only to show that even a whole week of clock error is
within 1.6× of the seed spread.

## The one durable finding

**A single Pareto/NBD fit on CDNOW is not a reportable number.** Every CDNOW Pareto/NBD row
in the docs is `n = 1` at seed 42, and seed 42 happens to be the most pessimistic of the
three tried (−15.97 against −11.50 at seed 44). The benchmark tables describe the fit as
"deterministic, so n = 1 and no spread", which holds on electronics (±0.35 over three
seeds, `docs/p-slstm.md` §10) and does **not** hold on CDNOW.

That is worth fixing and is cheap — a fit is ~16 s, so twenty seeds is five minutes — but
it is a separate ticket about how the benchmark is reported, not about `_build_cbs`.

## Why the first reading was wrong

Recorded so the next reader does not repeat it. `period_start` is a week's first day, so
`max(period_start)` reads naturally as "the start of the last calibration week", which
makes `T_cal` look a week short and the forecast columns look a week early. That reasoning
silently assumes transactions sit at the *start* of their bucket. Nothing in the code says
so, and the same arithmetic applied to `t_x` shows the anchor cancels — the convention is
end-of-week, under which every window lines up. **A duration built from label differences
carries no information about where in the bucket an event sits; only the pairing of the two
endpoints matters, and here they are paired consistently.**

## Reproducing

With the project venv's interpreter, from the repo root (the package is not installed in
the venv, so `src` goes on the path):

```bash
PYTHONPATH=src "$VIRTUAL_ENV/bin/python" .scratch/pnbd-cdnow-replication/check_window_alignment.py
PYTHONPATH=src "$VIRTUAL_ENV/bin/python" .scratch/pnbd-cdnow-replication/measure_window_shift.py cdnow 42,43,44
```

The first prints the window table per panel (seconds). The second runs the arms at ~16 s a
fit and writes `.scratch/pnbd-cdnow-replication/window_shift_results.csv`.

## Comments

**2026-09-17 — opened as a confirmed off-by-one, closed the same day as `wontfix`.** The
mechanism was reasoned from the wrong within-week convention; a four-week toy run through
the real function settles it. The measured sensitivity is recorded above so the question
does not get reopened from the same reading. What came out of it instead is the CDNOW seed
spread, which is a real reporting problem and is not this ticket.
