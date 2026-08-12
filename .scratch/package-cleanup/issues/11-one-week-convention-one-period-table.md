# 11 — One week-numbering convention, one period-length table

**What to build:** the package converts between calendar time and period indices exactly one
way. A monthly panel produces the same answer no matter which code path computed its period
length.

**Blocked by:** 02

**Status:** ready-for-agent

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

- [ ] One week-numbering convention in the package
- [ ] One period-length table; the `monthly` disagreement resolved and the choice recorded
- [ ] The Pareto/NBD fit reads that single table
- [ ] Both validation scripts' internal copies untouched, with a comment saying why
- [ ] Both validation scripts still land in their bands
- [ ] Golden test green at rel=1e-6
