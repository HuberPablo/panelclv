# 03 — The frozen benchmark's widths are constructor-overridable

**Status:** done

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

## Comments

Closed with fix option **(a)**, extended to the private backbone as well. `memory_units` and
`dense_units` are gone from both `ValendinLSTMModel.__init__` and
`_ValendinLSTMBackbone.__init__`; the backbone reads `_MEMORY_UNITS` / `_DENSE_UNITS`
directly.

A pure deletion, as the ticket predicted — every construction site already used the
three-argument form: `registry/model_registry.py:280`, `tests/test_valendin_lstm.py:43` and
`:152`, and `scripts/validate_valendin_lstm.py:197`. The ADR-0004 gate was checked first, as
the ticket asks: it passes only `seq_cols`, `embedded_cols`, `target_col`, so the deletion
does not touch it and the parameter count it prints is unchanged. Nothing in `src/`,
`tests/`, `scripts/`, `notebooks/` or `grids/` passed either width; the `dense_units=` hits
in the test suite are all `MultinomialLSTMModel`, the developed model, which searches its
widths by design.

Option (b) was considered and dropped: it would have meant documenting the arguments as
existing for a use that does not exist anywhere in the repo.

The two constants now carry the reason they are read rather than passed —
"a default is only a default: an argument that can be passed is an architecture that can be
changed, which is the same unfreezing the registry refuses to allow the search space to do."

No doc changed. ADR-0004's "widths ... may not change" and the registry comment at
`model_registry.py:363-367` were already right; this is the code moving to meet them.

`tests/test_valendin_lstm.py::test_the_widths_are_not_constructor_arguments` pins it from
both sides — neither class accepts a width, and a constructed model still measures 128/128.
It is the constructor half of what
`test_model_registration.py::test_valendin_architecture_is_not_searched` already pins for the
search space; together they mean 128/128 is the only shape the class can take, not merely
the shape every path happens to ask for.

`pytest -q`: **367 passed**.
