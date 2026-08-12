# 14 — Retarget the archive tests as read-path coverage

**What to build:** the test file that pinned the on-disk archive format becomes what it
actually earns its place as — the only coverage the suite layout and the suite read path
have — and stops asserting constraints that no longer exist.

**Blocked by:** 09

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/06-target-architecture.md` (decision 7),
`14-archive-format-gate.md`

## Why it changes

The file was built to give the on-disk-format floor an executable definition. **Pablo
rescinded that floor item** — archived studies no longer have to stay re-readable, and clean
code outranks archive compatibility. So the file's *mandate* is gone.

What survives is most of the file: **19 of its 24 tests run from a synthetic fixture and
exercise live code** — the layout path helpers, the config round-trip, prediction saving, and
the read path through dataset preparation. That code serves *current* studies, not only
archived ones, and this file is its only coverage.

## What goes

- The 4-5 skip-if-absent tests that read the real archive. Both the archive and the datasets
  directory are gitignored, so those can never run in CI — and there is no CI.
- The two tests pinning known warts **as-is**: aggregate files keyed by one id spelling
  beside predictions headed with another, and the metrics function raising on the legacy
  suites. Under the rescinded floor those warts are **fixable rather than preservable**, so
  tests asserting them would block the fix.

**Do not delete the file wholesale.** That trades one unjustified constraint for a coverage
hole.

## Relabelling

The file's name and docstring should say read-path coverage, not frozen format. Blocked by
issue 09 because the suite read path it exercises is split three ways there.

- [ ] The 19 fixture-driven tests kept and passing
- [ ] Real-archive tests removed
- [ ] The two wart-pinning tests removed
- [ ] File and docstring name what it now does
- [ ] Suite passes on a fresh clone with no archive present
