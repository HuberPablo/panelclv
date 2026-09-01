# 19 — Triage labels are unedited template, and `Status:` has three vocabularies

**Status:** ready-for-agent

An agent told to "apply the AFK-ready triage label" gets a mapping that does not describe this
tracker.

## Doc claim

`docs/agents/triage-labels.md:3` presents itself as a mapping onto local practice:

> The skills speak in terms of five canonical triage roles. This file **maps those roles to
> the actual label strings used in this repo's issue tracker.**

## Code reality

The two columns are **identical in all five rows** (`:5-11`), and the template instruction is
still in place at `:15`:

> Edit the right-hand column to match whatever vocabulary you actually use.

So nothing was mapped. And the vocabulary actually in use does not match either column. Across
`.scratch/`:

| value | occurrences |
|---|---|
| `done` | 25 |
| `resolved` | 14 |
| `ready-for-agent` | 5 |
| `wontfix` | 1 |

`done` — the most common value in the tracker — appears in neither the canonical five nor
anywhere in `triage-labels.md`. `needs-triage`, `needs-info` and `ready-for-human` are never
used.

## Second problem: `Status:` means two different things

`docs/agents/issue-tracker.md` defines the same field twice, with disjoint vocabularies:

- `:10` — "Triage state is recorded as a `Status:` line near the top of each issue file (**see
  `triage-labels.md` for the role strings**)" — i.e. the five triage roles.
- `:26` — "A `Status:` line records **`claimed`/`resolved`**" — i.e. wayfinding lifecycle.

On disk, all three vocabularies coexist in one field:
`.scratch/benchmark-refactor/issues/01-fix-benchmarks-import.md:3` is `Status: done`;
`.scratch/package-simplification/issues/05-reachability-ledger.md:4` is `Status: resolved`;
`.scratch/worker-scheduling/issues/01-work-queue.md:3` is `Status: ready-for-agent`.

## Also drifting

`docs/agents/issue-tracker.md:8` — "The spec is `.scratch/<feature-slug>/spec.md`". Present in
`worker-scheduling/` and `package-cleanup/`; absent in `benchmark-refactor/` (issues only),
`package-simplification/` (has `map.md`, which is legitimate under the wayfinding section) and
`p-slstm/` (a research directory with neither).

`docs/agents/issue-tracker.md:27` — blocking is specified as `Blocked by: NN, NN`, but
`.scratch/worker-scheduling/issues/01-work-queue.md:4` reads
`Blocked by: the running grid must finish first` — free prose, which the "unblocked when every
file it lists is `resolved`" rule on the same line cannot evaluate.

## Fix

1. Decide what `Status:` actually means here. The evidence says it is a **lifecycle** field
   (`done` / `resolved` dominate), with `ready-for-agent` used for handoff. Write that down
   once, in `issue-tracker.md`, and make `triage-labels.md` map onto it — including `done`,
   or standardise on `resolved` and sweep the 25 files.
2. Remove the "Edit the right-hand column" template line, or actually edit the column.
3. Reconcile `issue-tracker.md:10` and `:26` so one field has one vocabulary.
4. Either relax `:8` and `:27` to match practice (spec optional; `Blocked by:` may be prose) or bring
   the directories into line.

**Note:** the tickets in this directory use `ready-for-agent` / `needs-triage` — the
documented vocabulary — so that this issue is filed against the convention rather than
quietly departing from it.
