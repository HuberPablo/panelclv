# 17 — The doc-currency gate is smaller than it says, and skips the doc carrying issue `05`

**Status:** ready-for-agent

## Doc claim

`tests/test_docs_are_current.py:29-30`:

> ```python
> # The seven prose surfaces. Subpackage docstrings are excluded: they live in `src/`
> # and are already covered by `test_imports.py` resolving what they advertise.
> ```

## Code reality

`DOC_FILES` (`tests/test_docs_are_current.py:31-38`) is **thirteen** files, not seven:

```python
DOC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "CONTEXT.md",
    REPO_ROOT / "docs" / "running-a-model.md",
    REPO_ROOT / "docs" / "feature_engineering.md",
    *sorted((REPO_ROOT / "docs" / "adr").glob("*.md")),   # 8 ADRs
]
```

Three docs under `docs/` are silently outside it:

- `docs/loss-functions.md` (1015 lines)
- `docs/backpropagation.md`
- `docs/p-slstm.md`

The ADR glob means the count moves whenever an ADR is added, so "seven" was never going to
stay true — but the omission of three whole chapters is the part that matters. **Issue `05`
is in `docs/backpropagation.md`**, and it is exactly the kind of thing this gate exists to
catch: it names a symbol (`search space`) in a claim that a test could check.

## Second gap, already acknowledged

The test checks backticked **paths** and **symbols** and constructs fenced `PanelConfig`
literals. It cannot see a false prose claim that names nothing — its own docstring
(`:16-18`) says so, and issue `06` is a live instance in a file the gate *does* cover.

## Fix

1. Add the three missing docs to `DOC_FILES`, then run the suite — expect new failures, and
   triage them rather than adding to `ALLOWED_ABSENT`. (`ALLOWED_ABSENT` at `:47-73` is
   already documented as "the escape hatch, so adding to it should be a deliberate act
   rather than the reflex that silences a real failure"; that discipline is the point.)
2. Replace the "seven prose surfaces" comment with something that cannot go stale — describe
   the rule (every `.md` at the repo root and under `docs/`, ADRs included) rather than a
   count.
3. Consider making the list *derived* rather than enumerated: glob `docs/**/*.md` plus the
   three root files, so a new chapter is gated the day it is written. `docs/agents/*.md` would
   come in with that, which is where issue `19` lives.
