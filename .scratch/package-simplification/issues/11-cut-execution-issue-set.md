# Cut the execution issue set

Type: task
Status: resolved
Blocked by: 06, 07, 08, 09, 10, 12, 13, 14

## Question

The handoff, and the map's destination. Turn every decision into executable issues at
`.scratch/package-cleanup/issues/NN-<slug>.md` — a separate feature directory from this
map, following the repo's tracker conventions (`Status:` line per `docs/agents/triage-labels.md`,
`Blocked by:` where ordering matters).

Requirements on the set:

- **Ordered so the tree is never broken** — the golden test from ticket 01 stays green
  at every step, and any commit renaming a public name carries its notebook updates.
- **Sized for one agent session each.**
- **Carrying its evidence** — every deletion issue cites the ledger row that justifies it,
  so whoever executes it does not have to re-derive whether the symbol is dead.
- **Under the ~15-issue tripwire.** If it isn't, that is the signal to cut scope, and it
  should have been caught at ticket 06.

Resolve by recording the issue count and the execution order.

## Answer

Cut 2026-08-12 via `/to-spec` → `/to-tickets`. **15 issues**, published to
`.scratch/package-cleanup/issues/`, with the spec they collapse from at
`.scratch/package-cleanup/spec.md`. All `Status: ready-for-agent`.

### The set, in dependency order

| # | issue | blocked by |
|---|---|---|
| 01 | Parametrise the golden test over four model families | — |
| 02 | Delete the broken plot scripts and sweep the root | 01 |
| 03 | Execute the ledger's kill list | 02 |
| 04 | Delete rollout-composite selection | 02 |
| 05 | Refit-only, and `experiments/` becomes `trials/` | 02 |
| 06 | One registry entry per model | 02, 04 |
| 07 | A rollout model comes from its trained model | 05, 06 |
| 08 | Prediction I/O gets its own module; `evaluation/` reshaped | 02, 04 |
| 09 | Split the suite analysis module; collapse the group set | 03 |
| 10 | Split the Pareto grid at the real-panel boundary | 02 |
| 11 | One week convention, one period table | 02 |
| 12 | Derive the target column and time flags once | 02 |
| 13 | Output paths derivable from config and seed | 02 |
| 14 | Retarget the archive tests as read-path coverage | 09 |
| 15 | The orphan rename sweep | 03-13 |

Both ordering constraints hold: **01 is the net, first overall**; **02 is the sweep, first
among structural issues**. 15 lands last by construction.

### Two findings this ticket produced

- **The map's budget had a one-issue gap.** Ticket 09 folded the `trials/` creation into
  "the issue that creates `trials/`", but ticket 06's ten issues never contained one —
  `experiments/` was ruled to survive, so nothing in that list creates it. Counted honestly
  the carve was **16**, not ~15.
- **The designated deferral would not have freed a slot.** Ticket 13 nominated ticket 12's
  `analysis.py` three-way split as the thing to drop at 16. But that issue also carries the
  customer-group collapse and the single Student-t interval — both wrong-number fixes that
  must land — and no other issue touches that module. Deferring the split shrinks the issue;
  it does not remove it.

**Resolved by merging refit-only with the `trials/` creation** (issue 05, Pablo's call).
They rewrite the same module — refit-only deletes `prediction_source` and the notebook
toggles, `trials/` splits that module into loaders and refit halves — and neither moves a
number, so the golden test's signal stays unambiguous. **15, at the tripwire and not over.**

### Ownership gaps filled

- **Every `CLAUDE.md` edit has a named owner**, as ticket 07 required before code work
  starts: Edit 1 → issue 06, Edits 2-3 → issue 06, Edit 4 **plus the heading removal** →
  issue 07, the `scripts/` line → issue 02. Issue 07 is blocked by 06, so the heading goes
  with the last bullet as specified.
- **Ticket 08 left the torch-free removal unowned.** It inventoried six sites but assigned
  them to no issue. Placed: the lazy benchmark loader and the two `CLAUDE.md` lines on issue
  06 (the registry's direct references are the consequence), the three inert deferrals in the
  suite analysis module on issue 09. The two unrelated lazy imports are named in both issues
  as not-to-be-swept.
- **The Pareto grid file rename** moved from the orphan sweep into issue 10, which splits
  that module — ticket 09 assigned it to the sweep before ticket 12 decided it would split.
