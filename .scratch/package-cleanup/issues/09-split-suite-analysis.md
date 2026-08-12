# 09 — Split the suite analysis module, and collapse the customer-group set

**What to build:** the suite analysis surface is three modules named for what they do, and
the customer-group set is defined once by the predicates that define the groups. A plot's
band and a table's confidence interval are the same number because they are the same code.

**Blocked by:** 03

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/12-thesis-figure-code-home.md` (decisions 4,
7, 8), `09-module-naming.md` (decision 12), `08-reconcile-adrs-and-vocabulary.md`

## The split

One 1195-line module becomes three: a **suite reader** (prediction loading, suite
aggregation, discovery and config helpers), **suite plots** (the suite forecast plot and its
helpers), and **suite metrics** (study metrics, cross-study comparison, group metrics table).

**The split is free at the call sites** — every one of these has live notebook callers and
all of them import the subpackage, not the module, so **zero notebook edits.**

**The justification is not size.** It is that a single 1195-line file is what let a hardcoded
group vocabulary and a *second copy of the Student-t interval* hide inside it, both found
only by audit.

## The customer-group set

The hardcoded group tuple is not "thesis segmentation vocabulary baked into package code",
as it was first described. It is **the third of five encodings of one set** — the predicates,
the assignment default, this tuple, and the defaults of both group-metrics functions, with
the catch-all re-derived at three call sites.

**Survivor: the predicates that define the groups.** Their keys *are* the set by
construction and cannot drift from it. The catch-all is derived once, in the same place.

## One Student-t interval

Implemented three times: the across-study band, the study-metrics computation — **both
inside this one file** — and the Pareto grid's own copy, whose confidence arithmetic is
algebraically identical under a different spelling. **One implementation, living in suite
metrics**; the band and the grid's helper both call it. That two of the three were in one
file is the strongest single argument for the split.

## Folded in

The three deferred imports in this module commented as keeping it torch-free were **already
inert** — the studies package pulls torch at package import, so they never saved anything,
and one of them defers pandas, which cannot affect torch at all. Remove them and the
module-docstring claim with them.

## Sizing note

If the whole set needs headroom, **this split is the designated thing to defer** — it is the
one decision justified by future legibility rather than by a wrong number or an unreachable
path. The group collapse and the single Student-t interval must land regardless.

- [ ] Three modules, named for what they do; zero notebook edits required
- [ ] Group set has one encoding — the predicates' keys — with the catch-all derived once
- [ ] One Student-t implementation, called by the band and the grid helper
- [ ] Torch-free deferrals and the docstring claim removed
- [ ] Golden test green; notebook API test green
