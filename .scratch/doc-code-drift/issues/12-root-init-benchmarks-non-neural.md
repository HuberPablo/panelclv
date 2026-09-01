# 12 — The root `__init__` calls `benchmarks` "the non-neural Pareto/NBD comparator"

**Status:** ready-for-agent

This is the altitude map `CLAUDE.md` sends readers to, and it hides half of a subpackage.

## Doc claim

`src/panelclv/__init__.py:19`:

> - ``panelclv.benchmarks`` — the **non-neural** Pareto/NBD comparator (hierarchical Bayes).

## Code reality

`benchmarks/` holds two models, one of which is a trainable `nn.Module`:

- `src/panelclv/benchmarks/__init__.py:27` exports `ValendinLSTMModel` and
  `RolloutValendinLSTMModel`
- `src/panelclv/benchmarks/valendin_lstm.py:128` — `class ValendinLSTMModel(nn.Module)`
- `src/panelclv/registry/model_registry.py:369-378` — `valendin_lstm` is a registry entry
  **with a builder**, which is precisely what makes `is_neural("valendin_lstm")` return
  `True` (`model_registry.py:405-411`)

`CONTEXT.md:76-77` states the intended reading:

> **Benchmark**: A reference implementation used as a comparator in a study. **The two
> benchmarks** are the Valendin et al. LSTM and Pareto/NBD.

And `CLAUDE.md:52-53` describes the subpackage correctly ("frozen reference implementations
we reproduce, not develop"). The root `__init__` is the odd one out.

## Fix

`src/panelclv/__init__.py:19` — name both:

> ``panelclv.benchmarks`` — the two frozen reference implementations we reproduce rather
> than develop (ADR-0004): the Valendin et al. LSTM and the hierarchical-Bayes Pareto/NBD.

While in the file, check the same list's other entries against reality — this is the only one
I found wrong, but the file is prose with no gate on it.

## Related

Issue `20` — `README.md:3-4` has the same omission, announcing one benchmark where there are
two.
