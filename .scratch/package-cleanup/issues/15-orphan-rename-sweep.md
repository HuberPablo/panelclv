# 15 — The orphan rename sweep

**What to build:** the last names that could not fold into a structural issue. One concept,
one word, everywhere.

Lands **after all structural work**, so nothing rebases against a moving file.

**Blocked by:** 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/09-module-naming.md` (decisions 3, 9, 13)

## File renames

The remaining ones — the training loop module drops its `_utils` suffix, the panel dataset
module drops "dynamic" (which does no work), the Pareto simulation module and the Pareto
benchmark module join the unified spelling. **The Pareto grid module's rename already landed
with issue 10**, which split it.

Two of these are visible to notebooks — the panel dataset module at 6 sites and the Pareto
simulation module at 5, where an import alias absorbs most of the cost.

**Notebooks import subpackages, not modules**, with four exceptions — which is why only two
of these renames cost anything at all.

## The three collisions

Words meaning several things, not bad words:

- **"study" ×3.** The suite config's study name becomes **suite name** — it collides with a
  term the glossary defines as something else. The Pareto grid's "study", which is a folder
  of generated datasets, becomes a **dataset directory**. The layout helper's study directory
  **keeps its name**: it really is an Optuna study.
- **"group" ×2.** The Pareto grid's rate-and-churn grid point becomes a **cell**, leaving
  *group* to mean customer segment only. *Cell* stays out of `CONTEXT.md` — it is local to
  one module and needs no shared definition.
- **head size ×3.** The transaction-cap name is **killed** — it is the name that means
  neither of the other two. The class count keeps the head size, and the clip cap keeps the
  config knob that sets it. These are genuinely two concepts and rightly keep two names.

## Cost warning

**The transaction-cap name occurs 13 times across the notebooks and 22 times in the
package** — the single most expensive rename in the whole set, and the reason this issue is
worth its own session. Every occurrence must go in this commit or the notebook API test goes
red.

- [ ] Four remaining file renames applied; notebook import sites updated
- [ ] Suite name, dataset directory and cell renames applied; the layout helper's name untouched
- [ ] The transaction-cap name gone from the package and all 13 notebook occurrences
- [ ] Retired names appended to the import test's pattern
- [ ] Golden test green at rel=1e-6; notebook API test green
- [ ] The reachability tracer runs clean — nothing was orphaned by the whole set
