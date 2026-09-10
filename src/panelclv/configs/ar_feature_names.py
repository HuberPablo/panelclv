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

# The fixed names. The three windowed families are not among them -- each is
# parameterised by an integer and matched by a pattern below.
RECENCY = "period_since_last_transaction"
HAS = "has_transacted_before"
CUM_TXN = "cumulative_transactions"
CUM_CNT = "cumulative_count"
TENURE = "period_since_first_transaction"
RATE = "transaction_rate"
# Support-safe re-encodings of the two window-capped clocks. Same information,
# compressed so the holdout lands nearer the range the weights were fitted on
# (`.scratch/ar-encoding-support/spec.md`).
LOG_RECENCY = "log_period_since_last_transaction"
LOG_TENURE = "log_period_since_first_transaction"
RECENCY_RATIO = "recency_over_tenure"

_ACTIVE_RE = re.compile(r"^active_in_last_(\d+)_periods$")
# Two more windowed families, parameterised by a half-saturation constant C rather
# than by a bin edge: the feature reaches 1/2 at a gap of C periods.
_SAT_RECENCY_RE = re.compile(r"^saturating_recency_(\d+)_periods$")
_SAT_TENURE_RE = re.compile(r"^saturating_tenure_(\d+)_periods$")

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
    LOG_RECENCY: "log_recency",
    LOG_TENURE: "log_tenure",
    RECENCY_RATIO: "recency_ratio",
}

# pattern -> the `kind` token, for the families that carry a window. Written once, for
# the same reason `_FIXED_KINDS` is: the grammar and the dispatch cannot drift apart.
_WINDOWED_KINDS: list[tuple[re.Pattern[str], str]] = [
    (_ACTIVE_RE, "active"),
    (_SAT_RECENCY_RE, "sat_recency"),
    (_SAT_TENURE_RE, "sat_tenure"),
]


def parse_ar_feature(name: str) -> tuple[str, int | None]:
    """Validate one AR-feature name → (kind, K or None).

    kind is one of: 'recency', 'has', 'active', 'cum_txn', 'cum_cnt', 'tenure',
    'rate', 'log_recency', 'log_tenure', 'recency_ratio', 'sat_recency',
    'sat_tenure'. Only the three windowed families carry a K; the rest return None.
    """
    kind = _FIXED_KINDS.get(name)
    if kind is not None:
        return (kind, None)
    for pattern, windowed_kind in _WINDOWED_KINDS:
        m = pattern.match(name)
        if m:
            k = int(m.group(1))
            if k < 1:
                raise ValueError(f"window must be >= 1 in {name!r}")
            return (windowed_kind, k)
    supported = ", ".join(repr(n) for n in _FIXED_KINDS)
    raise ValueError(
        f"unknown ar feature {name!r}; supported: {supported}, "
        "'active_in_last_<K>_periods', 'saturating_recency_<C>_periods', "
        "'saturating_tenure_<C>_periods'"
    )


def validate_ar_features(names: Sequence[str]) -> None:
    """Raise ValueError if any name is not a supported AR feature."""
    for n in names:
        parse_ar_feature(n)
