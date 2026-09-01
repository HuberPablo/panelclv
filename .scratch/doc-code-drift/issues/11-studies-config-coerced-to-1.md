# 11 — `studies/config.py`: "Coerced to 1 for the deterministic Pareto/NBD baseline" — nothing coerces

**Status:** done

## Doc claim

`src/panelclv/studies/config.py:111-113`, in `StudySuiteConfig`'s field documentation:

> `n_studies_per_model` — How many independent Optuna studies to run per neural model (each
> gets its own seed). **Coerced to 1 for the deterministic Pareto/NBD baseline.**

## Code reality

Nothing coerces the field, anywhere:

- `src/panelclv/studies/config.py:170` — `validate()` only checks `n_studies_per_model >= 1`
- `src/panelclv/studies/runner.py:197-252` — `_run_pareto_model` never reads the field; it
  runs one fit unconditionally, because that is what a single MCMC fit is
- `src/panelclv/studies/runner.py:269` — `_model_record` stores the **uncoerced** value, so a
  suite configured with `n_studies_per_model=10` archives `10` on the Pareto/NBD record while
  one study exists on disk

The *behaviour* is what the docstring describes — Pareto/NBD does get one study. The
mechanism is not: it is ignored, not coerced, and the archived record keeps the number that
was ignored.

## Why it matters beyond wording

The archived value is read back. A reader of `Studies/*/…/config.json` or of anything built
on `_model_record` sees a count that does not match the number of studies actually present
for that model, which is the same class of problem ADR-0006 records: a stale count in the
archive reader "silently collapsed the Valendin benchmark's across-study spread to a single
study".

## Fix options

**(a) Fix the docstring** (smaller). Say the field applies to neural models only and is
ignored for Pareto/NBD, which runs exactly one fit because a single MCMC fit is what the
model is.

**(b) Actually coerce it**, so the archived record matches reality:
`_model_record` (`runner.py:269`) writes `1` when `not is_neural(spec.model_type)`. Worth
checking whether anything under `Studies/` or in `studies/suite_reader.py` currently relies
on the uncoerced value before changing what is written.

(b) is the one that makes the archive self-describing; (a) alone leaves the misleading number
in every future `config.json`.

## Related

Issue `01` — the other false claim in this same docstring, four lines below
(`studies/config.py:116-118`, "for its sampler and training").

## Comments

Closed with fix option **(a)**, alongside issue `01` — the two false lines sit four lines
apart in the same `StudySuiteConfig` docstring and were corrected in one pass.

`n_studies_per_model` now reads:

> How many independent Optuna studies to run per neural model (each gets its own seed).
> Ignored for Pareto/NBD, which runs exactly one fit because a single MCMC fit is what the
> model is; the archived record still stores the value that was passed in.

Option (b) — having `_model_record` (`studies/runner.py:269`) write `1` for a non-neural
model — was **not** taken. The trade-off was put explicitly: (b) makes the archive
self-describing, (a) leaves a misleading number in every future `config.json`. (a) was
chosen because the session's standing decision was to correct prose and move no behaviour,
and because changing what is written into `config.json` needs `studies/suite_reader.py` and
the 19 existing archives checked for readers of the uncoerced value first.

The docstring now says the number survives into the record, so a reader of a Pareto/NBD
`config.json` is warned rather than misled. Reopening (b) is still worthwhile if the archive
reader ever grows to trust that field.
