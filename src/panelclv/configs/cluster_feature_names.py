"""The vocabulary of `PanelConfig.cluster_features` — the names, and how to read one.

A **behavioural cluster** is a per-customer categorical label: customers are grouped
by how they behaved across the calibration window, and the group index becomes a
learned embedding the model reads at every period. It is *static* — computed once
from calibration and constant for that customer through the whole holdout — which is
what makes it leak-free without any rollout machinery (`docs/feature_engineering.md`).

This module holds the *names* half of the split, exactly as `ar_feature_names` does
for `ar_features`, and for the same structural reason: `PanelConfig` validates every
field at construction, and `configs` sits at the bottom of the import stack, so it
cannot reach up into `data_preparation` to ask which names are legal. Keeping the
grammar down here lets both sides read one statement of it.

A name is also the panel column name, so a declaration like::

    cluster_features=("kmeans_8",)

puts a column `kmeans_8` in `seq_cols`, and an archived study `config.json` records
both the algorithm and K without a lookup table.

The computation each name selects is documented in
`data_preparation.cluster_features`, which imports `parse_cluster_feature` from here
and dispatches on the `kind` it returns.

Standard library only — importing this must stay cheap enough that a leaf module can.
"""

from __future__ import annotations

import re
from typing import Sequence

# One algorithm is supported. The pattern is a family parameterised by K, the number
# of clusters, mirroring `active_in_last_<K>_periods` in `ar_feature_names`.
_KMEANS_RE = re.compile(r"^kmeans_(\d+)$")

# K < 2 is rejected here rather than downstream: the label becomes an embedded column,
# and `prepare_dataset` refuses an embedding of cardinality < 2 because a constant
# column carries no information. Catching it at config construction gives a message
# that names the declaration rather than the resolved cardinality.
_MIN_CLUSTERS = 2


def parse_cluster_feature(name: str) -> tuple[str, int]:
    """Validate one cluster-feature name → (kind, K).

    kind is 'kmeans' — the only algorithm registered. K is the number of clusters,
    which is also the cardinality of the embedding the label drives.
    """
    m = _KMEANS_RE.match(name)
    if m:
        k = int(m.group(1))
        if k < _MIN_CLUSTERS:
            raise ValueError(
                f"cluster count must be >= {_MIN_CLUSTERS} in {name!r}; a single "
                "cluster is a constant column and carries no information"
            )
        return ("kmeans", k)
    raise ValueError(
        f"unknown cluster feature {name!r}; supported: 'kmeans_<K>' "
        f"(K >= {_MIN_CLUSTERS}), e.g. 'kmeans_8'"
    )


def validate_cluster_features(names: Sequence[str]) -> None:
    """Raise ValueError if any name is not a supported cluster feature."""
    for n in names:
        parse_cluster_feature(n)
