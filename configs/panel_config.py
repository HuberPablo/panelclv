"""A single, validated config object for the panel → model pipeline.

`PanelConfig` replaces the four loose dicts that `prepare_dataset` used to
take (DATA_CONFIG, FEATURE_SCHEMA, TIME_FEATURES, INPUT_SPEC). It validates
its fields at construction (so typos and cross-field mistakes fail early) and
supplies per-frequency defaults. The target is declared once, as `target_col`.

    cfg = PanelConfig(id_col="Id", target_col="Transactions", frequency="weekly",
                      training_start=..., training_end=...,
                      holdout_start=..., holdout_end=...,
                      time_cols=("year", "week"), clip_target_upper=6,
                      time=("week_sin", "week_cos"),
                      known_future=("year_idx", "high.season"),
                      static=("Gender", "Income"),
                      time_features={"add_year_idx": True, "add_week_sin_cos": True},
                      embedded_cols={"Transactions": "auto", "Gender": "auto"})
    data = prepare_dataset(panel, cfg)

The `.data_config`, `.schema` and `.input_spec` properties expose the
dict forms the existing `prepare_dataset` internals consume, so no downstream
helper had to change.

Standard library only — no third-party dependencies beyond pandas (used only
to parse the window dates).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from Data_preparation.ar_features import validate_ar_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_FREQUENCIES: tuple[str, ...] = ("weekly", "monthly", "daily")
_DEFAULT_PERIODS_PER_YEAR: dict[str, int] = {"weekly": 52, "monthly": 12, "daily": 365}
_KNOWN_TIME_FLAGS: tuple[str, ...] = (
    "add_year_idx",
    "add_week_sin_cos",
    "add_month_sin_cos",
    "add_dayofyear_sin_cos",
)

# Which time-feature flags each frequency can actually produce (mirrors
# Data_preparation.dynamic_panel_dataset.add_time_features).
_COMPATIBLE_TIME_FLAGS: dict[str, frozenset[str]] = {
    "weekly": frozenset({"add_year_idx", "add_week_sin_cos"}),
    "monthly": frozenset({"add_year_idx", "add_month_sin_cos"}),
    "daily": frozenset(
        {"add_year_idx", "add_week_sin_cos", "add_month_sin_cos", "add_dayofyear_sin_cos"}
    ),
}

# The cyclical (sin/cos) columns each flag creates — auto-added to the `time`
# role. `add_year_idx` is intentionally absent: `year_idx` is a trend feature
# whose role (usually known_future) is the caller's choice, so it is placed
# explicitly, not auto-assigned.
_FLAG_TIME_COLUMNS: dict[str, tuple[str, ...]] = {
    "add_week_sin_cos": ("week_sin", "week_cos"),
    "add_month_sin_cos": ("month_sin", "month_cos"),
    "add_dayofyear_sin_cos": ("day_sin", "day_cos"),
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
    four window dates. Everything else has a sensible default. Feature roles
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
        for attr in ("time", "known_future", "observed_past", "static", "ar_features"):
            object.__setattr__(self, attr, _as_col_tuple(getattr(self, attr)))
        validate_ar_features(self.ar_features)  # fail early on a bad feature name

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
        for name in ("training_start", "training_end", "holdout_start", "holdout_end"):
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
            unknown = [k for k in self.time_features if k not in _KNOWN_TIME_FLAGS]
            if unknown:
                raise ValueError(
                    f"unknown time_features keys: {unknown}; "
                    f"valid keys are {list(_KNOWN_TIME_FLAGS)}"
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
        compatible = _COMPATIBLE_TIME_FLAGS[self.frequency]
        cleaned: dict[str, bool] = {}
        for flag, on in dict(self.time_features).items():
            if flag in compatible:
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
            if not on:
                continue
            for col in _FLAG_TIME_COLUMNS.get(flag, ()):
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

    @property
    def input_spec(self) -> dict[str, Any] | None:
        """The INPUT_SPEC-equivalent dict, or None if no columns are embedded."""
        if not self.embedded_cols:
            return None
        return {"embedded_cols": normalize_embedded_cols(self.embedded_cols)}
