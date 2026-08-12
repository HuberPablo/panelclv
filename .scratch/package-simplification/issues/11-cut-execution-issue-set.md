# Cut the execution issue set

Type: task
Status: open
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
