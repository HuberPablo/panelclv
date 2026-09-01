"""A single, validated config object for the panel → model pipeline.

`PanelConfig` replaces the four loose dicts that `prepare_dataset` used to
take (DATA_CONFIG, FEATURE_SCHEMA, TIME_FEATURES, INPUT_SPEC). It validates
its fields at construction (so typos and cross-field mistakes fail early) and
supplies per-frequency defaults. The target is declared once, as `target_col`.

    cfg = PanelConfig(
        # === REQUIRED ============================================================
        # --- identity / target ---
        id_col="Id",                       # customer identifier column
        target_col="Transactions",         # count column to forecast (and embed)
        frequency="weekly",                # "weekly" | "monthly" | "daily"

        # --- window dates ---
        training_start="2017-01-01",       # first date of the calibration window
        training_end="2018-12-31",         # last date of calibration (must precede holdout)
        validation_start="2018-07-01",     # first date of the temporal validation window;
                                           # carves the calibration TAIL [validation_start,
                                           # training_end] off as validation (all customers),
                                           # so weights train only on [training_start,
                                           # validation_start). Must satisfy
                                           # training_start < validation_start <= training_end.
        holdout_start="2019-01-01",        # first date of the holdout window
        holdout_end="2019-12-31",          # last date of holdout

        # --- time indexing: REQUIRED, but which one depends on frequency ---
        time_cols=("year", "week"),        # REQUIRED for weekly/monthly: (year_col, period_col)
        # date_col="Date",                 # REQUIRED for daily instead: single date column
        
        # === OPTIONAL (defaults shown) ===========================================
        periods_per_year=52,               # seasonal period; default per frequency (52/12/365)
        clip_target_upper=6,               # cap counts (sets softmax head size); default None = no cap
        require_calibration_activity=True, # keep only customers active in calibration (Valendin filter)
        # --- feature roles (target excluded; all default to ()) ---
        time=(),                           # cyclical time cols already precomputed in the panel
        known_future=("year_idx", "high.season"),  # covariates known at forecast time (future-dated)
        observed_past=("promo_flag",),     # covariates observed only up to the forecast origin
        static=("Gender", "Income"),       # per-customer constants (don't vary over time)
        # --- engineered calendar features (opt-in; default None = none) ---
        time_features={"add_year_idx": True, "add_week_sin_cos": True},  # derive year_idx + week_sin/cos
        # --- autoregressive target-derived features (leak-free; default ()) ---
        ar_features=("period_since_last_transaction",),  # recency/activity, recomputed during rollout
        # --- behavioural clusters (per-customer label from calibration; default ()) ---
        cluster_features=("kmeans_8",),    # k-means on (t_x, x, T); embedded automatically
        # --- embeddings (which cols to embed; int | "auto"; default ()) ---
        embedded_cols={"Transactions": "auto", "Gender": "auto"},  # categorical cols → learned embeddings
    )
    data = prepare_dataset(panel, cfg)

Feature roles follow the TFT-style grouping — `time`, `known_future`,
`observed_past`, `static` — with the target kept out of every role (it is
`target_col`). `ar_features` names autoregressive, target-derived signals
(a "transaction" = target > 0) that are recomputed from the sampled count
during the holdout rollout, so they stay leak-free. Supported names:

    period_since_last_transaction   recency: periods since the last transaction
                                    (0 this period; counts up if never seen).
    has_transacted_before           1 once any transaction has occurred, else 0.
    active_in_last_<K>_periods      1 if a transaction fell within the last K
                                    periods (inclusive); K >= 1, e.g. ..._3_... .
    cumulative_transactions         running count of active periods — the RFM /
                                    Pareto-NBD "frequency" (x).
    cumulative_count                running sum of the target counts themselves
                                    (>= cumulative_transactions).
    period_since_first_transaction  tenure: periods since the first transaction
                                    — the BTYD observation age "T".
    transaction_rate                cumulative_transactions / max(tenure, 1), an
                                    empirical per-period purchase rate.

`cluster_features` names **behavioural clusters**: customers are partitioned by
their calibration behaviour — the (t_x, x, T) Pareto/NBD triple at the last
calibration period — and each customer's group index becomes a static, embedded
column. Supported names:

    kmeans_<K>                      k-means into K clusters, K >= 2, e.g.
                                    "kmeans_8". The name is also the column name.

Unlike an AR feature, a cluster label is **frozen**: it is computed once from
calibration and stays constant through every holdout period, so the rollout
carries it untouched. Each declared name is added to `embedded_cols`
automatically with cardinality K — a cluster index is categorical by definition,
and standardising it would impose an ordering the labels do not have. See
`data_preparation.cluster_features`.

`require_calibration_activity` (the Valendin cohort filter, on by default)
restricts the panel to customers active in the calibration window, governing
both the LSTM and the Pareto/NBD benchmark.

`time_features` vs the `time` role are two different jobs and should not be
double-specified. A `time_features` flag (e.g. `add_week_sin_cos`) *engineers*
calendar columns that don't exist in the raw panel — and `.schema` then auto-
assigns its outputs (`week_sin`/`week_cos`) to the `time` role. So when a flag
produces a column you do NOT also list it under `time`. Use `time=(...)` only
for cyclical columns that are ALREADY precomputed in the panel (no flag engineers
them). In the example above `add_week_sin_cos` supplies `week_sin`/`week_cos`, so
`time` is left empty.

The `.data_config` and `.schema` properties expose the dict forms the existing
`prepare_dataset` internals consume; the embedding map is read straight off the
`.embedded_cols` field (normalized to `{col: int | "auto"}` by `prepare_dataset`).

Depends only on pandas (to parse the window dates) plus the sibling
`ar_feature_names` and `cluster_feature_names` grammars — no other third-party
libraries, and nothing from a subpackage above this one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import pandas as pd

from panelclv.configs.ar_feature_names import validate_ar_features
from panelclv.configs.cluster_feature_names import validate_cluster_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_FREQUENCIES: tuple[str, ...] = ("weekly", "monthly", "daily")
_DEFAULT_PERIODS_PER_YEAR: dict[str, int] = {"weekly": 52, "monthly": 12, "daily": 365}


@dataclass(frozen=True)
class TimeFeatureSpec:
    """What one engineered-calendar flag is: where it can run, and what it writes.

    `frequencies`  the panel frequencies that carry enough calendar information to
                   produce it (a weekly panel has no day-of-year).
    `columns`      every column the flag adds to the panel — the complete list, so a
                   column that gets created and then never given a role is a
                   contradiction the table itself shows rather than a silent orphan.
    `auto_time`    whether those columns are added to the `time` role automatically.
                   True for the cyclical sin/cos pairs, which are meaningless anywhere
                   else. False for `year_idx`: it is a trend feature whose role
                   (usually known_future) is the caller's choice, so it is placed
                   explicitly.
    """

    frequencies: frozenset[str]
    columns: tuple[str, ...]
    auto_time: bool


# The engineered calendar features, written once. This table is the whole statement of
# "the set of time flags": the legal keys are its keys, the per-frequency compatibility
# is its `frequencies`, the columns auto-assigned to the `time` role are its `columns`,
# and `data_preparation.add_time_features` validates against it rather than restating
# the frequency rules in its own if/elif chain.
#
# It used to be three tables plus that chain, and they had drifted: the daily branch
# also wrote a raw `dayofyear` column that no role table listed, so it rode into the
# model-ready tensor as a channel nothing read.
TIME_FEATURE_FLAGS: dict[str, TimeFeatureSpec] = {
    "add_year_idx": TimeFeatureSpec(
        frozenset({"weekly", "monthly", "daily"}), ("year_idx",), auto_time=False
    ),
    "add_week_sin_cos": TimeFeatureSpec(
        frozenset({"weekly", "daily"}), ("week_sin", "week_cos"), auto_time=True
    ),
    "add_month_sin_cos": TimeFeatureSpec(
        frozenset({"monthly", "daily"}), ("month_sin", "month_cos"), auto_time=True
    ),
    "add_dayofyear_sin_cos": TimeFeatureSpec(
        frozenset({"daily"}), ("day_sin", "day_cos"), auto_time=True
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_col_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a feature-role field to a tuple of column names.

    Accepts:
      - None         -> ()
      - a bare str   -> (value,)         e.g. "year_idx"
      - any iterable -> tuple(value)     e.g. ("a", "b") or ["a"]

    The bare-string case is the important one: in Python, ``("year_idx")``
    is *not* a 1-tuple — the parentheses are just grouping, so it evaluates
    to the string ``"year_idx"``. Without this normalization, downstream
    code that iterates the field would walk the string character by
    character and look for columns named ``'y'``, ``'e'``, ``'a'``, ...
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def normalize_embedded_cols(raw: Any) -> dict[str, Any]:
    """Normalize an `embedded_cols` spec to a plain ``{col: value}`` dict.

    The spec may be given two ways: a mapping ``{col: int | "auto"}`` (explicit
    cardinalities) or a bare iterable of names ``["Gender", "Income"]`` (all
    ``"auto"``). Both collapse to a dict here so every downstream consumer reads
    one shape. Shared by `PanelConfig` and `prepare_dataset` so the two cannot
    drift on what a valid spec looks like.
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (list, tuple, set)):
        return {c: "auto" for c in raw}
    raise TypeError(f"embedded_cols must be a dict or list, got {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelConfig:
    """Validated configuration for one panel → model run.

    Required: identity / target (`id_col`, `target_col`, `frequency`) and the
    five window dates (`training_start`, `training_end`, `validation_start`,
    `holdout_start`, `holdout_end`). Everything else has a sensible default. Feature roles
    follow the TFT-style grouping (`time`, `known_future`, `observed_past`,
    `static`); the target is NOT listed in a role — it is `target_col`.
    `embedded_cols` is the user's choice of which columns to embed; each value
    may be a pinned int or ``"auto"`` (inferred later by `prepare_dataset`),
    and the whole thing may be a plain list of names (all ``"auto"``).

    `time_features` is opt-in: when omitted, NO calendar features are engineered.
    List the flags you want (e.g. ``{"add_year_idx": True, "add_week_sin_cos": True}``);
    any flag the frequency cannot produce is dropped with a warning (not an error),
    and the cyclical columns of enabled flags are auto-added to the `time` role
    (see `.schema`).
    """

    # --- identity / target ---
    id_col: str
    target_col: str
    frequency: str
    training_start: str
    training_end: str
    # Temporal validation split: the calibration window [training_start, training_end]
    # is cut at `validation_start`. Periods [training_start, validation_start) train the
    # weights; periods [validation_start, training_end] are the validation window (used
    # for early stopping / model selection, never trained on). This replaces the old
    # customer-wise split — validation is a TIME window over ALL customers. Required.
    validation_start: str
    holdout_start: str
    holdout_end: str

    # --- time indexing ---
    time_cols: tuple[str, str] | None = None   # weekly / monthly
    date_col: str | None = None                # daily
    periods_per_year: int | None = None        # None → default from frequency

    # --- target handling ---
    clip_target_upper: int | None = None

    # --- cohort selection ---
    # Reproduce Valendin et al.: keep only customers with >=1 transaction during the
    # calibration window (equivalently, first purchase <= training_end). Customers
    # first seen only in the holdout are dropped — at forecast time they are unknown,
    # and they would otherwise feed the model an all-zero calibration history. The
    # filter is applied in prepare_dataset, so it governs BOTH the LSTM and the
    # Pareto/NBD benchmark (same cohort -> fair comparison). True = faithful to the paper.
    require_calibration_activity: bool = True

    # --- feature roles (target excluded; it is target_col) ---
    time: Sequence[str] = ()
    known_future: Sequence[str] = ()
    observed_past: Sequence[str] = ()
    static: Sequence[str] = ()

    # --- engineered calendar features ---
    time_features: Mapping[str, bool] | None = None

    # --- autoregressive target-derived features (recency / activity) ---
    # e.g. ("period_since_last_transaction", "has_transacted_before",
    #       "active_in_last_3_periods"); recomputed from the sampled target
    # during the holdout rollout, so no leakage.
    ar_features: Sequence[str] = ()

    # --- behavioural clusters (per-customer categorical label) ---
    # e.g. ("kmeans_8",); k-means over the (t_x, x, T) triple at the last calibration
    # period. The label is STATIC — frozen at calibration and constant across the
    # holdout — so the rollout carries it untouched, unlike an ar_feature.
    cluster_features: Sequence[str] = ()

    # --- embeddings (which columns to embed; values int | "auto") ---
    embedded_cols: Mapping[str, int | str] | Sequence[str] = ()

    # ------------------------------------------------------------------ #
    # Validation + normalization
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if self.frequency not in _VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {_VALID_FREQUENCIES}, got {self.frequency!r}"
            )

        # Normalize role sequences to tuples (frozen → use object.__setattr__).
        # `_as_col_tuple` is forgiving: a bare string like
        # ``known_future="year_idx"`` is treated as a 1-element tuple, which
        # avoids the classic ``("year_idx")``-is-a-string Python gotcha.
        for attr in (
            "time", "known_future", "observed_past", "static",
            "ar_features", "cluster_features",
        ):
            object.__setattr__(self, attr, _as_col_tuple(getattr(self, attr)))
        validate_ar_features(self.ar_features)  # fail early on a bad feature name
        validate_cluster_features(self.cluster_features)

        # Time-index layout depends on frequency.
        if self.time_cols is not None:
            tc = tuple(self.time_cols)
            if len(tc) != 2:
                raise ValueError(
                    f"time_cols must be (year_col, period_col), got {self.time_cols!r}"
                )
            object.__setattr__(self, "time_cols", tc)
        if self.frequency in ("weekly", "monthly"):
            if not self.time_cols:
                raise ValueError(
                    f"{self.frequency} frequency requires time_cols=(year_col, period_col)"
                )
        else:  # daily
            if not self.date_col:
                raise ValueError("daily frequency requires date_col")

        # Windows must parse and be ordered (training strictly before holdout).
        for name in (
            "training_start", "training_end", "validation_start",
            "holdout_start", "holdout_end",
        ):
            value = getattr(self, name)
            try:
                pd.Timestamp(value)
            except Exception as exc:  # noqa: BLE001 — re-raised with context
                raise ValueError(f"{name} is not a parseable date: {value!r}") from exc
        if pd.Timestamp(self.training_end) >= pd.Timestamp(self.holdout_start):
            raise ValueError(
                f"training_end ({self.training_end}) must be before "
                f"holdout_start ({self.holdout_start})"
            )
        # The temporal validation window sits at the TAIL of the calibration window:
        # training_start < validation_start <= training_end. The lower bound is strict
        # so at least one training period exists before validation begins; the upper
        # bound is inclusive so validation_start == training_end is allowed (a single
        # validation period). The exact period counts are re-checked in prepare_dataset
        # against the real calendar (0 < val_start_idx < T_CAL).
        if not (
            pd.Timestamp(self.training_start)
            < pd.Timestamp(self.validation_start)
            <= pd.Timestamp(self.training_end)
        ):
            raise ValueError(
                f"validation_start ({self.validation_start}) must satisfy "
                f"training_start ({self.training_start}) < validation_start "
                f"<= training_end ({self.training_end})"
            )

        if self.clip_target_upper is not None and int(self.clip_target_upper) < 0:
            raise ValueError(
                f"clip_target_upper must be >= 0, got {self.clip_target_upper}"
            )

        if not isinstance(self.require_calibration_activity, bool):
            raise TypeError(
                "require_calibration_activity must be a bool, got "
                f"{type(self.require_calibration_activity).__name__}"
            )

        # Reject unknown time-feature flags (catches typos early).
        if self.time_features is not None:
            unknown = [k for k in self.time_features if k not in TIME_FEATURE_FLAGS]
            if unknown:
                raise ValueError(
                    f"unknown time_features keys: {unknown}; "
                    f"valid keys are {list(TIME_FEATURE_FLAGS)}"
                )

        # Resolve time features: default per frequency when omitted, and drop
        # frequency-incompatible flags with a warning rather than erroring.
        object.__setattr__(self, "time_features", self._resolve_time_features())

        self._validate_embedded_cols()

        # Per-frequency default for periods_per_year.
        if self.periods_per_year is None:
            object.__setattr__(
                self, "periods_per_year", _DEFAULT_PERIODS_PER_YEAR[self.frequency]
            )

    def _resolve_time_features(self) -> dict[str, bool]:
        """Time features are opt-in: omitting `time_features` adds none.

        Only the explicitly requested flags are kept; any the frequency cannot
        produce are dropped with a warning (rather than a hard error).
        """
        if self.time_features is None:
            return {}
        cleaned: dict[str, bool] = {}
        for flag, on in dict(self.time_features).items():
            if self.frequency in TIME_FEATURE_FLAGS[flag].frequencies:
                cleaned[flag] = bool(on)
            elif on:
                warnings.warn(
                    f"time feature {flag!r} is not compatible with frequency "
                    f"{self.frequency!r}; ignoring it.",
                    stacklevel=3,
                )
            # incompatible and off → silently drop
        return cleaned

    def _validate_embedded_cols(self) -> None:
        items = list(normalize_embedded_cols(self.embedded_cols).items())
        for col, value in items:
            if value in (None, "auto"):
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"embedded_cols[{col!r}] must be an int or 'auto', got {value!r}"
                )
            if value <= 1:
                raise ValueError(
                    f"embedded_cols[{col!r}] cardinality must be > 1, got {value}"
                )

    # ------------------------------------------------------------------ #
    # Dict views consumed by prepare_dataset internals
    # ------------------------------------------------------------------ #

    @property
    def data_config(self) -> dict[str, Any]:
        """The DATA_CONFIG-equivalent dict."""
        d: dict[str, Any] = {
            "id_col": self.id_col,
            "target_col": self.target_col,
            "frequency": self.frequency,
            "periods_per_year": self.periods_per_year,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "validation_start": self.validation_start,
            "holdout_start": self.holdout_start,
            "holdout_end": self.holdout_end,
            "require_calibration_activity": self.require_calibration_activity,
        }
        if self.time_cols is not None:
            d["time_cols"] = list(self.time_cols)
        if self.date_col is not None:
            d["date_col"] = self.date_col
        if self.clip_target_upper is not None:
            d["clip_target_upper"] = self.clip_target_upper
        return d

    def to_dict(self) -> dict[str, Any]:
        """All fields as a JSON-serializable dict (post-validation/normalization).

        Captures the *resolved* config actually used by `prepare_dataset`: the
        feature-role sequences normalized to lists, `embedded_cols` collapsed to
        a plain ``{col: int | "auto"}`` map, and the per-frequency defaults filled
        in (`periods_per_year`, and `time_features` resolved to the kept flags).
        So a study `config.json` that stores this can fully reconstruct the run's
        PanelConfig — every constructor argument is present, with the same values
        the pipeline saw. Mirrors (and supersedes) the partial `.data_config` view.
        """
        return {
            "id_col": self.id_col,
            "target_col": self.target_col,
            "frequency": self.frequency,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "validation_start": self.validation_start,
            "holdout_start": self.holdout_start,
            "holdout_end": self.holdout_end,
            "time_cols": list(self.time_cols) if self.time_cols is not None else None,
            "date_col": self.date_col,
            "periods_per_year": self.periods_per_year,
            "clip_target_upper": self.clip_target_upper,
            "require_calibration_activity": self.require_calibration_activity,
            "time": list(self.time),
            "known_future": list(self.known_future),
            "observed_past": list(self.observed_past),
            "static": list(self.static),
            "time_features": dict(self.time_features),
            "ar_features": list(self.ar_features),
            "cluster_features": list(self.cluster_features),
            "embedded_cols": dict(normalize_embedded_cols(self.embedded_cols)),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PanelConfig":
        """Rebuild a `PanelConfig` from a `to_dict()` payload.

        The inverse of `to_dict`: every key it emits is a constructor argument, so
        reconstruction is just `cls(**d)`. `__post_init__` re-runs the same
        validation/normalization the original went through (e.g. a `time_cols`
        stored as a JSON list is re-tupled), so a config round-tripped through a
        study's `config.json` is the same config `prepare_dataset` originally saw.
        Keys that are not constructor fields are dropped, so a reader on an older
        schema never crashes on a newer file.
        """
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    @property
    def schema(self) -> dict[str, list[str]]:
        """The FEATURE_SCHEMA-equivalent dict.

        The target is derived from `target_col`. The `time` role is auto-extended
        with the cyclical columns the enabled time features produce (e.g.
        `week_sin`/`week_cos`), unless a column is already assigned to some role —
        so the flags and the schema can't drift apart.
        """
        time = list(self.time)
        assigned = (
            {self.target_col}
            | set(self.time)
            | set(self.known_future)
            | set(self.observed_past)
            | set(self.static)
        )
        for flag, on in self.time_features.items():
            spec = TIME_FEATURE_FLAGS[flag]
            if not on or not spec.auto_time:
                continue
            for col in spec.columns:
                if col not in assigned:
                    time.append(col)
                    assigned.add(col)
        return {
            "target": [self.target_col],
            "time": time,
            "known_future_time_varying_inputs": list(self.known_future),
            "observed_past_time_varying_inputs": list(self.observed_past),
            "static_covariates": list(self.static),
        }

