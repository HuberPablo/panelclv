# 11 — One week-numbering convention, one period-length table

**What to build:** the package converts between calendar time and period indices exactly one
way. A monthly panel produces the same answer no matter which code path computed its period
length.

**Blocked by:** 02

**Status:** done

Source: `.scratch/package-simplification/issues/05-reachability-ledger.md` (D7),
`06-target-architecture.md` (decision 10, decision 9)

## Why this is a correctness issue and not tidying

**Four week-numbering conventions and three period-length tables exist**, and **two of the
period tables disagree on `monthly` — 30.0 against 30.4368 — with both feeding the
Pareto/NBD fit.** Inert today because the live panels are weekly. Wrong by construction the
day a monthly panel runs, and wrong *quietly*: the fit still converges, it just fits
something else.

This is one of the three duplications ticket 06 promoted to its own issue on exactly that
basis — it can make a number wrong, not merely annoy.

## Hard constraint — do not dedupe the validation scripts

The two benchmark validation scripts carry **their own** weeks-per-year constant, their own
cohort filter and their own week index. **These are deliberate insulation, not duplication,
and are frozen.** A gate that imports the code it gates stops being a gate: a future bug in a
shared cohort filter would move the benchmark and its own check in lockstep and still pass.

This was settled explicitly against one audit's recommendation. **No issue in this set may
dedupe them**, and that includes the week-arithmetic copies living inside them.

- [x] One week-numbering convention in the package
- [x] One period-length table; the `monthly` disagreement resolved and the choice recorded
- [x] The Pareto/NBD fit reads that single table
- [x] Both validation scripts' internal copies untouched, with a comment saying why
- [x] Both validation scripts still land in their bands
- [x] Golden test green at rel=1e-6

## Comments

Landed 2026-08-13. Full suite green (234 passed, up from 225: nine new tests), golden
end-to-end included and unmoved at rel=1e-6. Both gates re-run and in band:
`validate_pareto_benchmark.py` PASS (aggregate diff -2.15 %, per-customer corr 0.9950),
`validate_valendin_lstm.py` PASS (val loss 0.4760 vs 0.44 ± 0.06, bias 0.51 %).

### The one place

`src/panelclv/data_preparation/period_calendar.py` — a leaf module importing only numpy
and pandas, holding the week convention, the days-per-period table, and the four
conversions that used to be spelled out at their call sites. Every caller now reads it:
`dynamic_panel_dataset` (both directions of the week convention), `pareto_simulation` and
`studies/synthetic_grid` (the flat-counter pack/unpack), `benchmarks/pareto_benchmark` and
`studies/runner` (the period length). `WEEKS_PER_YEAR` moved out of `pareto_simulation`,
where a package-wide constant had no business living; `pareto_simulation` re-exports it by
importing it, so `Pareto_Datasets.ipynb`'s `ps.WEEKS_PER_YEAR` still resolves.

`benchmarks/` importing `data_preparation/` is new, and deliberate: ADR-0004 freezes a
benchmark's *arithmetic*, not where a plumbing constant is read from, and the ADR names
data preparation as shared infrastructure. `tests/test_import_graph.py` still holds — the
graph gains no cycle, since `data_preparation` does not import `benchmarks`.

### The week convention, and the bug it was hiding

One statement, in the module docstring: a year is 52 weeks of seven days numbered 0..51,
week `w` of year `Y` starting on `Y-01-01 + 7w` days, with the year's trailing day or two
folded back into week 51.

Of the four conventions, two already agreed on it — `pareto_simulation`'s `% 52` and
`add_period_start`'s Jan-1 anchor — and the fourth is the Valendin gate's, frozen. The one
that had to move was `add_time_features`' daily branch, which used `isocalendar().week - 1`.
Both of ISO's departures cost something real:

- **ISO week 53 aliased onto week 0.** 2020-12-28 through 2020-12-31 landed on week index
  52, and `sin(2π·52/52), cos(2π·52/52)` is exactly `(0.0, 1.0)` — bit-identical to
  January 1st. Four days of late December were fed to the model as New Year's Day.
- **ISO weeks do not start on Jan 1.** 2019-12-30 and 2019-12-31 came out as week 0, i.e.
  December encoded as January, while the panel's own `period_start` anchor put them in
  week 51. The feature and the split anchor disagreed about which week a row was in.

Both are gone; the daily branch now reads `week_of_year`, which is the exact inverse of the
`week_start` that `add_period_start` cuts the split on. `tests/test_period_calendar.py`
asserts that round trip in both directions, and asserts that no date in a common or a leap
year ever reaches week 52.

Weekly panels are unaffected — hence the golden numbers not moving — because that branch
already read the panel's own 0-based `week` column.

### The monthly disagreement, resolved

**The mean Gregorian calendar month wins, 30.0 goes.** The surviving table writes it as
`365.2425 / 12` rather than as the rounded `30.4368` the benchmark's table carried, so the
value states its own derivation; the difference between the two spellings is 2.5e-6
relative and reaches nothing, since no monthly panel runs. The flat 30.0 is not a month by
any definition and shortens the period by 1.4 %, which on a
three-year calibration window inflates every customer's observation age by about half a
month. The Pareto/NBD fit absorbs that into its dropout rate and converges anyway, which is
exactly why this was worth an issue. The reasoning is recorded at the constant in
`period_calendar.py` and pinned by a test, so a later reader cannot re-open it by guess.

The loser was `studies/runner._PERIOD_DAYS`, which also had `.get(freq, 7.0)` — an unknown
frequency silently fitting as weekly. `days_per_period` raises instead: a panel whose
frequency does not map to a period length is one whose sufficient statistics would be on
the wrong time scale, and that is not a thing to guess at.

### The validation scripts

Untouched apart from a comment on each constant saying why it is a copy —
`PERIOD_DAYS = 7.0` in the Pareto gate, `WEEKS_PER_YEAR = 52` and the `dayofyear // 7`
grid in the Valendin gate. The Valendin comment also records that its grid is *not* the
package's convention (`dayofyear // 7` gives week 0 six days; the package uses
`(dayofyear - 1) // 7`), which is the second reason it must not be deduped: it reproduces
the notebook's grid, not ours. Deduping either would make a gate that moves in lockstep
with the thing it gates.

### Deliberately not done

- **`periods_per_year` still only reaches the weekly branch.** D7 lists this with the
  divisors, but the two branches need different numbers *because* the field counts periods,
  not weeks: on a daily panel it is 365, and dividing a 0..51 week index by it would
  compress the year into a seventh of the sine's period. The daily branch reads
  `WEEKS_PER_YEAR`, with a comment at the call site and a paragraph in
  `docs/feature_engineering.md`, whose formula column was wrong about the daily case and is
  now right. Removing the field is a `PanelConfig` change, and archived runs record it.
- **The month (12) and day-of-year (365) divisors.** Neither is a duplicated table, and the
  365 leap-day case is not the week-52 bug: day 366 maps onto day 1, which is where it
  belongs on the cycle.
- **`compute_pareto_predictions(period_in_days=7.0)`'s default.** A default argument, not a
  table, and it agrees with the table's `weekly`. No caller in the package, the tests or the
  gates relies on it — all four pass the value explicitly.
- **The golden fixture's ISO week numbers.** `tests/test_golden_end_to_end.py` builds its
  weekly panel with `d.isocalendar()[1]`, so its `week` column is 1..52 rather than 0..51 —
  off-convention, but it is fixture data, the pipeline treats it consistently, and changing
  it would move the pinned numbers for no gain in what the test pins.
