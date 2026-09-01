# 03 — The frozen benchmark's widths are constructor-overridable

**Status:** needs-triage

## Doc claim

`docs/adr/0004-frozen-reference-implementations.md:10-11`:

> **Frozen means the numbers, not the surrounding code.** A benchmark's *architecture* — its
> layers, **widths**, activations and the arithmetic that produces a forecast — **may not
> change.**

`src/panelclv/registry/model_registry.py:363-367` restates it as the reason the benchmark
has no architecture in its search space:

> The Valendin benchmark's architecture is FROZEN (ADR-0004): its widths are the published
> `memory_units = 128` / `dense_units = 128` … searching a width would quietly unfreeze the
> reference implementation.

## Code reality

`src/panelclv/benchmarks/valendin_lstm.py:135-142`:

```python
def __init__(
    self,
    seq_cols: Sequence[str],
    embedded_cols: dict[str, int],
    target_col: str = "Transactions",
    memory_units: int = _MEMORY_UNITS,
    dense_units: int = _DENSE_UNITS,
) -> None:
```

Both widths are ordinary constructor arguments, forwarded straight into
`_ValendinLSTMBackbone` (`:143-149`). Any direct caller can construct a "Valendin benchmark"
with different widths and it will train, roll out and be scored like the published one.

## What is actually safe

Every path *inside* the package is fine. `_build_valendin`
(`src/panelclv/registry/model_registry.py:280-284`) passes only `seq_cols`,
`embedded_cols` and `target_col`, so the defaults always win; and the search space
(`:369-374`) carries only `learning_rate` / `weight_decay` / `batch_size`, with
`tests/test_model_registration.py::test_valendin_architecture_is_not_searched` pinning that.
So this is a hole in the *class*, not in any run that has happened.

## Fix options

**(a) Close the hole.** Drop the two parameters and read `_MEMORY_UNITS` / `_DENSE_UNITS`
directly in `__init__`. Nothing in `src/`, `tests/` or `scripts/` passes them, so this is a
removal with no call-site churn — but check `scripts/validate_valendin_lstm.py` first, since
that gate is the executable definition of ADR-0004 and is allowed to reach into the file.

**(b) Say why they stay.** If they exist for the faithfulness check (constructing the
paper's *other* configurations to compare parameter counts, say), record that in the
docstring and in ADR-0004, so "may not change" is qualified rather than contradicted.

Option (a) is the smaller change and matches how the frozen widths are described everywhere
else; (b) is only right if there is a reason the constructor arguments are load-bearing that
I did not find.
