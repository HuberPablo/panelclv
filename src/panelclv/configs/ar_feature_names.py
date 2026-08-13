"""The vocabulary of `PanelConfig.ar_features` — the names, and how to read one.

`ar_features` is a declaration: the caller writes down which autoregressive,
target-derived signals the panel should carry, and `data_preparation.ar_features`
computes them. This module holds the *names* half of that split, so the grammar sits
in the same subpackage as the field that is written in it.

It lives here rather than beside the computation for a structural reason.
`PanelConfig` validates every field at construction, so it needs to know which names
are legal; `configs` is the bottom of the import stack, so it cannot reach up into
`data_preparation` to ask. Keeping the grammar down here lets both sides read one
statement of it instead of two — and it is the *one* upward import `configs` had.

The computation each name selects is documented in
`data_preparation.ar_features`, which imports `parse_ar_feature` from here and
dispatches on the `kind` it returns.

Standard library only — importing this must stay cheap enough that a leaf module can.
"""

from __future__ import annotations

import re
from typing import Sequence

# The fixed names. `active_in_last_<K>_periods` is not among them: it is a family
# parameterised by a window K, matched by the pattern below.
RECENCY = "period_since_last_transaction"
HAS = "has_transacted_before"
CUM_TXN = "cumulative_transactions"
CUM_CNT = "cumulative_count"
TENURE = "period_since_first_transaction"
RATE = "transaction_rate"

_ACTIVE_RE = re.compile(r"^active_in_last_(\d+)_periods$")

# name -> the `kind` token the computation dispatches on. Written once: the keys are
# the supported fixed names, so "what may be declared" and "what can be computed"
# cannot drift apart.
_FIXED_KINDS: dict[str, str] = {
    RECENCY: "recency",
    HAS: "has",
    CUM_TXN: "cum_txn",
    CUM_CNT: "cum_cnt",
    TENURE: "tenure",
    RATE: "rate",
}


def parse_ar_feature(name: str) -> tuple[str, int | None]:
    """Validate one AR-feature name → (kind, K or None).

    kind is one of: 'recency', 'has', 'active', 'cum_txn', 'cum_cnt',
    'tenure', 'rate'. Only 'active' carries a window K; the rest return None.
    """
    kind = _FIXED_KINDS.get(name)
    if kind is not None:
        return (kind, None)
    m = _ACTIVE_RE.match(name)
    if m:
        k = int(m.group(1))
        if k < 1:
            raise ValueError(f"active window must be >= 1 in {name!r}")
        return ("active", k)
    supported = ", ".join(repr(n) for n in _FIXED_KINDS)
    raise ValueError(
        f"unknown ar feature {name!r}; supported: {supported}, "
        "'active_in_last_<K>_periods'"
    )


def validate_ar_features(names: Sequence[str]) -> None:
    """Raise ValueError if any name is not a supported AR feature."""
    for n in names:
        parse_ar_feature(n)
