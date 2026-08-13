# 06 — One registry entry per model

**What to build:** adding a model means adding one entry to one table. Every model-type
list in the package derives from that table's keys, and whether a type is neural is read
off its entry rather than restated as a second list. The failure mode `CLAUDE.md` warns
about — register in two places out of three, find out only after training completes —
stops being expressible.

**Blocked by:** 02, 04

**Status:** done

Source: `.scratch/package-simplification/issues/06-target-architecture.md` (decisions 4, 5,
Q13, Q16), `07-collapse-invariants.md` (decisions 1, 5), `08-reconcile-adrs-and-vocabulary.md`,
`09-module-naming.md` (decision 7)

## Why one table

**Seven enumerations of the model set exist in the package** — the valid-types list, the
neural-types list, the search defaults, the suggesters, the builders, the inference-builder
if-chain, and the forecasters. Six list the same three neural types; only one carries
Pareto/NBD, which is absent from the forecasters. That asymmetry is what forces the
optional-field shape.

An **eighth** copy in the suite analysis module had already drifted and silently collapsed
the Valendin benchmark's across-study spread to a single study. **Counting copies was the
problem; no single copy was.**

## Shape

- **A new, tenth subpackage.** Both `models/` and `studies/` are blocked as homes by real
  import cycles — the registry must name benchmark classes while the benchmark file already
  imports the model layer, and the tuner is what needs the registry while the study runner
  already imports the tuner. A repository-root module was rejected as clutter. The folder
  holds one module, re-exported from its `__init__`, matching the seven of nine existing
  subpackages that re-export.
- **One table, optional fields.** Remaining fields after ticket 07's amendment: **search
  space, builder, forecaster / rollout function.** There is no inference-builder field and
  no rollout-class field — issue 07 puts that pairing on the training class.
- **The valid-types list becomes the table's keys.** The neural list becomes the derived
  predicate *"this entry has a training builder"*. That is the copy that already drifted; a
  derived predicate makes the bug unwritable, a second list does not.
- **Pareto/NBD's entry is declarative only.** It exists so the enumerations derive from one
  place. The suite runner keeps its neural and deterministic paths **separate** — they
  differ in more than the forecaster (no study, no refit, one prediction rather than
  several), so merging them is a different refactor with its own risk.
- **Entries hold direct references, not lazy ones.** The laziness existed so the suite
  config could validate a model type without importing torch; that goal is dropped, and it
  was protecting a property the studies package does not have — it already pulls torch at
  package import. No cycle either way, since the model layer does not import studies.
- The parameter-suggestion functions move into the registry subpackage. That is what "one
  entry per model" requires, and it leaves the tuner importing the registry with no cycle.
- **The model-to-rollout-function pairing is declared through the registry, not enforced by
  sealing.** The forecast entry point reads the required rollout from the registry rather
  than trusting its caller. Both rollout functions stay importable; no stepper abstraction
  is built for two implementations.

## Folded in: the search-space / training-controls split

The model spec's `data_info` dict carries Optuna search-space overrides **and** training
controls, and its validator already polices it against two separate allowlists — the code
knows they are two sets, so make that the interface. It becomes two fields: **search space**
and **training**. A typo then lands in the wrong field rather than relying on a
hand-maintained allowlist, and the validator shrinks. Cost: four notebooks, the suite config,
the suite runner, and the tuner. The archived config records the old key; the archive floor
is rescinded, so that is free.

## Folded in: removing the torch-free idea

Ruled by Pablo — the idea goes, not just the guarantee. It never bought the ability to run
without torch, which is a **hard** dependency. Remove: the lazy loader and its type-checking
block in the benchmarks `__init__` (~30 lines existing only so importing benchmarks skips
torch), and the two `CLAUDE.md` lines. The three deferred imports in the suite analysis
module go with issue 09.

**Two lazy imports stay and must not be swept up by pattern-match** — they have nothing to
do with torch: the training module defers two genuinely optional dependencies, and the
Pareto MCMC fitter is deferred because it is heavy and pure numpy.

## Docs

- **ADR-0006** — full text in ticket 08 of the map. Copy it.
- **`CLAUDE.md` Edit 1** — the "Adding a model touches three places" paragraph becomes one
  place. Verbatim replacement text in ticket 07 of the map.
- **`CLAUDE.md` Edits 2 and 3** — the target-column and clip-cap bullets leave the
  invariants section; the head-size fact relocates into "What the models are". Both are pure
  doc moves and this issue owns them because it lands first of the two that empty that
  section. **Leave the heading in place** — issue 07 removes it with the last bullet.
- `CLAUDE.md`'s "Where things live" gains the registry subpackage.

## Test

The registration test is rewritten from three-list membership to one-table checks. It does
**not** simply get deleted — its per-type class assertions guard a dispatch-fallthrough bug
it was built for, and that risk survives the registry.

- [x] One registry table in its own subpackage; suggestion functions moved into it
- [x] Valid types derive from its keys; neural is a derived predicate, not a list
- [x] Pareto/NBD has a declarative entry; the runner's two paths stay separate
- [x] Entries hold direct references; no lazy indirection
- [x] Model spec carries search space and training as separate fields; validator shrunk
- [x] Torch-free machinery and its two `CLAUDE.md` lines removed; the two unrelated lazy imports untouched
- [x] ADR-0006 present, copied verbatim from ticket 08
- [x] `CLAUDE.md` Edits 1, 2 and 3 applied verbatim from ticket 07; invariants heading left standing
- [x] Registration test rewritten, per-type class assertions kept
- [x] Golden test green at rel=1e-6; notebook API test green
