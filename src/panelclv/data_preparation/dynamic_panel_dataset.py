"""Turn a customer-period panel into the model-ready tensors `prepare_dataset` returns.

`prepare_dataset(panel, config)` is the single entry point. It takes a DataFrame
with ONE ROW PER (customer, period) and returns the `data` dict consumed by the
training loop (`samples`/`targets`) and the Monte-Carlo simulator
(`calibration`/`holdout`). All model-facing arrays are (N, T, F) float32; the
channel order is fixed: target → time → known_future → observed_past → static →
ar_features.

`config` is a `PanelConfig` — the single validated object that bundles the
layout, calendar toggles, feature roles, embeddings and AR features.
`prepare_dataset` reads everything from it (`.data_config`, `.schema`,
`.time_features`, `.input_spec`, `.ar_features`); see "what a PanelConfig
declares" below for the pieces.

What it does, in order: engineer the requested calendar features and the causal
AR target-features → add a `period_start` anchor → slice the calibration and
holdout date windows → apply the Valendin "active during calibration" cohort
filter to both windows → (optionally) clip the training target → resolve
embedding cardinalities → reshape each window and build the next-step AR
`(samples, targets)` pair. It validates aggressively and fails *early* (before
any tensor is built): missing id/target/schema columns, non-numeric columns,
ragged per-customer period counts, train/holdout cohorts that differ or are
mis-ordered, NaNs in a selected column, and empty date windows all raise.

What a PanelConfig declares (and how prepare_dataset consumes it)
----------------------------------------------------------------
Physical layout (→ `.data_config`): `id_col`, `target_col`, `frequency`
    ∈ {weekly, monthly, daily}, `time_cols=[year, period]` (weekly/monthly) or
    `date_col` (daily), `periods_per_year` (week sin/cos divisor, default 52),
    `training_start/end`, `holdout_start/end`, optional `clip_target_upper`
    (train-only clip) and `require_calibration_activity` (cohort filter, default on).
Calendar toggles (→ `.time_features`): `add_year_idx`, `add_week_sin_cos`,
    `add_month_sin_cos`, `add_dayofyear_sin_cos`. A column is created only when
    its flag is on AND the frequency supports it, else a clear error is raised.
Feature roles (→ `.schema`) — TFT-style; ONLY listed columns enter the tensor:
    `target` (exactly one), `time` (cyclical calendar), `known_future_…`
    (values known per step in advance — embeddings may be sized over both
    windows without leakage), `observed_past_…` (history-only — NOT YET
    SUPPORTED: dropped with a warning, since the AR simulator has no future for
    them and feeding it would leak), `static_covariates` (one value per
    customer, already broadcast across that customer's rows).
AR features (→ `.ar_features`, recency / frequency / tenure / rate) are appended
    last; being causal functions of the target's own past, their holdout values
    are recomputed from the SAMPLED target during the rollout, so they leak nothing.
Embeddings (→ `.input_spec`): which columns to embed and their cardinalities
    (pinned int or `"auto"`, resolved here against the data).

Output dict (full key list at the end of `prepare_dataset`)
-----------------------------------------------------------
    calibration   (N, T_CAL,     F) float32   training-window tensor
    holdout       (N, T_HOLD,    F) float32   holdout-window tensor
    samples       (N, T_CAL - 1, F) float32   AR inputs  (steps 0..T-2)
    targets       (N, T_CAL - 1, 1) float32   AR labels  (next-step target)
    seq_cols, target_col, target_idx          channel names / target position
    input_spec                                resolved {col: cardinality} embeddings
    ar_features                               the AR feature names that were added
    N, T_CAL, T_HOLD, F, ids                  shapes + customer order of the tensors
    panel, train_panel, holdout_panel         engineered panel + the two slices
                                              (train_panel feeds the Pareto/NBD benchmark)
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd

from panelclv.configs.panel_config import PanelConfig, normalize_embedded_cols
from panelclv.data_preparation.ar_features import (
    compute_ar_feature_columns,
    validate_ar_features,
)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
# Channel order of the final (N, T, F) tensor — fixed so two schemas with
# overlapping features land in comparable positions.

_SCHEMA_GROUP_ORDER: list[str] = [
    "target",
    "time",
    "known_future_time_varying_inputs",
    "observed_past_time_varying_inputs",
    "static_covariates",
    "ar_features",
]


def get_seq_cols(schema: dict[str, Sequence[str]]) -> list[str]:
    """Flatten FEATURE_SCHEMA into an ordered, de-duplicated column list."""
    cols: list[str] = []
    for group in _SCHEMA_GROUP_ORDER:
        cols.extend(schema.get(group, []))
    return list(dict.fromkeys(cols))  # dict preserves insertion order and de-dupes


def _known_future_cols(schema: dict[str, Sequence[str]]) -> set[str]:
    """Columns whose holdout values are known in advance (known_future + time).

    Their embeddings may be sized over BOTH windows without lookahead, since the
    future values are given, not predicted. Centralized so the embedding-sizing
    rule and the drift warning read the same definition.
    """
    return set(schema.get("known_future_time_varying_inputs", [])) | set(
        schema.get("time", [])
    )


def resolve_embedded_cols(
    input_spec: dict[str, Any] | None,
    *,
    target_col: str,
    seq_cols: Sequence[str],
    schema: dict[str, Sequence[str]],
    train_panel: pd.DataFrame,
    holdout_panel: pd.DataFrame,
    clip_upper: int | None,
) -> dict[str, int] | None:
    """Resolve `input_spec['embedded_cols']` to a plain ``{col: cardinality}``.

    The caller chooses WHICH columns are embedded (the dict keys, or a plain
    list of names) — including whether to embed the target. Each cardinality
    may be a pinned ``int`` or ``"auto"``/``None`` to be inferred here. The
    inference window is **role-aware**, read from `schema` (the TFT-style
    FEATURE_SCHEMA groups), 0-indexed integer categories assumed:

        target_col                       : clip_target_upper + 1
                                           (or calibration max + 1 if no clip)
        known_future / time columns      : max(calibration, holdout) + 1
                                           -- legitimate: their future values
                                           are known in advance, not leakage
        static / observed-past / other   : calibration max + 1
                                           -- never peek at the holdout

    Embedding the target is the caller's decision (it is what gives the model
    a multinomial head); this function does NOT add it automatically. Pinned
    ints are kept but validated to cover the values present in the relevant
    window. Returns ``None`` if `input_spec` is ``None``.
    """
    if input_spec is None:
        return None

    spec = normalize_embedded_cols(input_spec.get("embedded_cols", {}))

    unknown = [c for c in spec if c not in seq_cols]
    if unknown:
        raise ValueError(
            f"input_spec['embedded_cols'] references columns not in seq_cols: {unknown}"
        )

    # Columns whose future is known in advance — safe to size from the holdout.
    known_future = _known_future_cols(schema)

    def _observed_max(col: str) -> int:
        if col == target_col:
            return int(clip_upper) if clip_upper is not None else int(train_panel[col].max())
        train_max = int(train_panel[col].max())
        if col in known_future and col in holdout_panel.columns:
            # Known in advance for the holdout, so covering those values is
            # legitimate, not lookahead.
            return max(train_max, int(holdout_panel[col].max()))
        # static (constant per customer) / observed-past / unknown → calibration only.
        return train_max

    resolved: dict[str, int] = {}
    for col, value in spec.items():
        observed_max = _observed_max(col)
        if value in (None, "auto"):
            cardinality = observed_max + 1
        elif isinstance(value, bool):
            raise TypeError(f"embedded_cols[{col!r}] must be an int or 'auto', got bool")
        elif isinstance(value, int):
            if value <= observed_max:
                raise ValueError(
                    f"input_spec cardinality for {col!r} = {value} is too small; "
                    f"values up to {observed_max} are present in the relevant "
                    f"window (need cardinality >= {observed_max + 1})."
                )
            cardinality = value
        else:
            raise TypeError(
                f"embedded_cols[{col!r}] must be an int or 'auto', got {value!r}"
            )
        if cardinality < 2:
            raise ValueError(
                f"Inferred cardinality for {col!r} is {cardinality} (< 2); the "
                f"column looks constant — drop it from embedded_cols."
            )
        resolved[col] = cardinality
    return resolved


# ---------------------------------------------------------------------------
# Engineered time features
# ---------------------------------------------------------------------------


def _resolve_time_index(
    panel: pd.DataFrame,
    frequency: str,
    time_cols: Sequence[str] | None,
    date_col: str | None,
) -> tuple[pd.Series | None, str | None, str | None]:
    """Validate the frequency's time columns and return a normalized handle.

    The single place the frequency → time-column rules live (used by both
    `add_time_features` and `add_period_start`). Returns ``(date, year_col,
    period_col)``:
        daily            -> (datetime Series, None, None)
        weekly / monthly -> (None, year_col, period_col)
    Raises KeyError/ValueError on missing or malformed time columns.
    """
    if frequency == "daily":
        if not date_col or date_col not in panel.columns:
            raise KeyError(
                f"daily frequency requires date_col in panel; got {date_col!r}"
            )
        return pd.to_datetime(panel[date_col]), None, None
    if frequency in ("weekly", "monthly"):
        if not time_cols or len(time_cols) != 2:
            raise ValueError(
                f"{frequency} frequency requires time_cols=[year_col, period_col]"
            )
        year_col, period_col = time_cols
        if year_col not in panel.columns or period_col not in panel.columns:
            raise KeyError(f"time_cols {list(time_cols)} not found in panel")
        return None, year_col, period_col
    raise ValueError(f"Unsupported frequency {frequency!r}")


def add_time_features(
    panel: pd.DataFrame,
    *,
    time_features: dict[str, bool],
    frequency: str,
    base_year: int,
    periods_per_year: int = 52,
    time_cols: Sequence[str] | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Add only the engineered calendar columns whose flag is True.

    Raises a clear error if a requested feature is incompatible with the
    declared frequency (e.g. asking for `add_dayofyear_sin_cos` on monthly
    data).
    """
    date, year_col, period_col = _resolve_time_index(
        panel, frequency, time_cols, date_col
    )
    # year_series feeds add_year_idx; it comes from the date (daily) or the
    # year column (weekly/monthly).
    year_series = date.dt.year if date is not None else panel[year_col].astype(np.int64)

    if time_features.get("add_year_idx"):
        panel["year_idx"] = (year_series - base_year).astype(np.int64)

    if time_features.get("add_week_sin_cos"):
        if frequency == "weekly":
            wk = panel[period_col].astype(np.int64)
            panel["week_sin"] = np.sin(2 * np.pi * wk / periods_per_year)
            panel["week_cos"] = np.cos(2 * np.pi * wk / periods_per_year)
        elif frequency == "daily":
            wk = date.dt.isocalendar().week.astype(np.int64) - 1
            panel["week_sin"] = np.sin(2 * np.pi * wk / 52)
            panel["week_cos"] = np.cos(2 * np.pi * wk / 52)
        else:
            raise ValueError(
                f"add_week_sin_cos requires daily or weekly frequency, got {frequency!r}"
            )

    if time_features.get("add_month_sin_cos"):
        if frequency == "monthly":
            m0 = panel[period_col].astype(np.int64) - 1
        elif frequency == "daily":
            m0 = date.dt.month.astype(np.int64) - 1
        else:
            raise ValueError(
                f"add_month_sin_cos requires daily or monthly frequency, got {frequency!r}"
            )
        panel["month_sin"] = np.sin(2 * np.pi * m0 / 12)
        panel["month_cos"] = np.cos(2 * np.pi * m0 / 12)

    if time_features.get("add_dayofyear_sin_cos"):
        if frequency != "daily":
            raise ValueError(
                f"add_dayofyear_sin_cos requires daily frequency, got {frequency!r}"
            )
        doy = date.dt.dayofyear.astype(np.int64)
        panel["dayofyear"] = doy
        panel["day_sin"] = np.sin(2 * np.pi * (doy - 1) / 365)
        panel["day_cos"] = np.cos(2 * np.pi * (doy - 1) / 365)

    return panel


# ---------------------------------------------------------------------------
# period_start anchor for splitting train/holdout
# ---------------------------------------------------------------------------


def add_period_start(
    panel: pd.DataFrame,
    *,
    frequency: str,
    time_cols: Sequence[str] | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Add a single Timestamp column used to slice train/holdout uniformly."""
    date, year_col, period_col = _resolve_time_index(
        panel, frequency, time_cols, date_col
    )
    if date is not None:  # daily
        panel["period_start"] = date
    elif frequency == "weekly":
        panel["period_start"] = (
            pd.to_datetime(panel[year_col].astype(str) + "-01-01")
            + pd.to_timedelta(panel[period_col].astype(np.int64) * 7, unit="D")
        )
    else:  # monthly
        panel["period_start"] = pd.to_datetime(
            dict(year=panel[year_col], month=panel[period_col], day=1)
        )
    return panel


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_columns(
    panel: pd.DataFrame,
    target_col: str,
    seq_cols: Sequence[str],
    id_col: str,
) -> None:
    """Check column existence and that all selected columns are numeric."""
    if id_col not in panel.columns:
        raise KeyError(f"id_col {id_col!r} not in panel")
    if target_col not in panel.columns:
        raise KeyError(f"target_col {target_col!r} not in panel")

    missing = [c for c in seq_cols if c not in panel.columns]
    if missing:
        raise KeyError(f"schema columns missing from panel: {missing}")

    non_numeric = [
        c for c in seq_cols
        if not pd.api.types.is_numeric_dtype(panel[c])
    ]
    if non_numeric:
        raise TypeError(
            f"Non-numeric columns selected by the schema: {non_numeric}. "
            f"Encode them (label / one-hot / etc.) before calling "
            f"prepare_dataset — the model tensors are float32."
        )


def warn_known_future_drift(
    train_panel: pd.DataFrame,
    holdout_panel: pd.DataFrame,
    *,
    schema: dict[str, Sequence[str]],
    input_spec: dict[str, Any] | None,
    target_col: str,
) -> None:
    """Warn when an *embedded* known-future column takes holdout values that
    never appeared during calibration.

    A column in the `known_future` (or `time`) role is, by definition, allowed
    to use its holdout values when sizing the embedding: `resolve_embedded_cols`
    deliberately sizes those tables over BOTH windows (see `_observed_max`), so
    this case never crashes. The risk is subtler — the embedding rows for values
    that occur only in the holdout were never updated by training, so at forecast
    time the model falls back on their random initialization. That is almost
    always a data-prep slip (e.g. a seasonal flag whose level only ever shows up
    in the holdout window), so we surface it up front as a warning rather than
    let it pass silently into the forecast.

    Scoped to embedded columns on purpose: continuous known-future channels
    (`week_sin`/`week_cos`, `year_idx`, …) are *expected* to take new values
    every holdout period — that is their job — and have no embedding table to
    leave untrained, so flagging them would be pure noise. The target is also
    excluded: it is sampled during the rollout, never read from the holdout.
    """
    if input_spec is None:
        return

    embedded_names = set(normalize_embedded_cols(input_spec.get("embedded_cols", {})))
    known_future = _known_future_cols(schema)
    cols = sorted((embedded_names & known_future) - {target_col})

    for col in cols:
        if col not in train_panel.columns or col not in holdout_panel.columns:
            continue
        cal_values = set(train_panel[col].dropna().unique())
        unseen = sorted(set(holdout_panel[col].dropna().unique()) - cal_values)
        if not unseen:
            continue
        # Keep the message readable when a column drifts wildly.
        shown = unseen[:10]
        more = "" if len(unseen) <= 10 else f" (+{len(unseen) - 10} more)"
        warnings.warn(
            f"known_future embedded column {col!r} takes {len(unseen)} "
            f"value(s) in the holdout that never appear in calibration: "
            f"{shown}{more}. The embedding is sized to cover them (no crash), "
            f"but their rows were never trained — the forecast will use their "
            f"random initialization. Check that this column is encoded "
            f"consistently across the calibration and holdout windows.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


def make_block(
    df: pd.DataFrame,
    ids: Sequence,
    sort_cols: Sequence[str],
    seq_cols: Sequence[str],
) -> np.ndarray:
    """Reshape a sorted panel slice into (N, T, F) float32."""
    N = len(ids)
    arr = df.sort_values(list(sort_cols))[list(seq_cols)].to_numpy("float32")
    total = arr.shape[0]
    if total % N != 0:
        raise ValueError(
            f"DataFrame has {total} rows, not divisible by N={N} "
            f"(customers must have an identical number of periods)"
        )
    T = total // N
    return arr.reshape(N, T, len(seq_cols))


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------


def select_active_cohort(train_panel, id_col: str, target_col: str):
    """Customers with >=1 transaction during the calibration window.

    Reproduces the Valendin et al. cohort rule. `train_panel` is already the
    calibration slice (`period_start <= training_end`), so a positive total over it
    is exactly "first purchase <= training_end". Customers first seen only in the
    holdout sum to 0 here and are excluded — at forecast time they are unknown and
    would otherwise feed the model an all-zero calibration history.

    The test is upper-clip invariant (clipping a count never turns a positive into a
    zero), so it is unaffected by `clip_target_upper` regardless of call order.

    Returns the surviving ids (a pandas Index). Raises if the cohort is empty.
    """
    totals = train_panel.groupby(id_col)[target_col].sum()
    active = totals[totals > 0].index
    if len(active) == 0:
        raise ValueError(
            "require_calibration_activity is on, but no customer has any transaction "
            "during the calibration window — the cohort is empty. Check the "
            "training window or disable the filter."
        )
    return active


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def prepare_dataset(
    panel: pd.DataFrame,
    config: PanelConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    """Top-level pipeline.

    `config` is a `PanelConfig`: it carries the physical layout, calendar
    toggles, feature roles, embeddings and AR features in one validated object,
    and everything below is read from it (`.data_config`, `.schema`,
    `.time_features`, `.input_spec`, `.ar_features`).

    Steps:
        1. (optional) Engineer time features per `time_features` toggles.
        2. Add `period_start` for slicing.
        3. Resolve seq_cols from the schema.
        4. Validate column existence and numeric dtype.
        5. Slice train / holdout windows.
       5b. (optional) Clip the target on the TRAINING window only, if
           config["clip_target_upper"] is set. Holdout is left as-is.
        6. Check uniform period counts and matching customer order.
        7. Check no NaN in selected columns.
        8. Reshape to (N, T, F) and build AR (samples, targets).

    When the config declares `embedded_cols`, embedding cardinalities are
    resolved here (see `resolve_embedded_cols`): each entry may be a pinned int
    or `"auto"`/`None` (and the mapping may be given as a plain list of column
    names, all treated as `"auto"`). `"auto"` cardinalities are inferred from
    the data per column role — target -> clip_target_upper + 1, known-future ->
    max over both windows + 1, everything else -> calibration max + 1 — and
    pinned ints are validated to cover the observed values. The fully resolved
    spec is returned under the "input_spec" key so the model is built from it.
    Embedding the target is the caller's choice; it is not added automatically.

    Returns a dict described at the module level.
    """

    # PanelConfig is the only accepted config; expand it to the plain dicts the
    # body below runs on.
    if not isinstance(config, PanelConfig):
        raise TypeError(
            f"config must be a PanelConfig, got {type(config).__name__}. "
            "Build one with PanelConfig(...) — see panelclv.configs.panel_config."
        )
    ar_features = list(config.ar_features)
    schema = config.schema
    time_features = config.time_features
    input_spec = config.input_spec
    config = config.data_config

    panel = panel.copy()

    id_col      = config["id_col"]
    target_col  = config["target_col"]
    frequency   = config["frequency"]
    time_cols   = config.get("time_cols")
    date_col    = config.get("date_col")
    periods_per_year = int(config.get("periods_per_year", 52))
    base_year   = pd.Timestamp(config["training_start"]).year

    # 0) Cross-validate clip_target_upper against the target embedding,
    # if both are available. Catches the common mistake of clipping at 10
    # while declaring an embedding of cardinality 6 (or smaller).
    clip_upper = config.get("clip_target_upper")
    if clip_upper is not None and input_spec is not None:
        embedded = input_spec.get("embedded_cols", {})
        # Only a pinned int can be checked this early; 'auto' is resolved later
        # against the clipped training window (see resolve_embedded_cols).
        pinned = embedded.get(target_col) if isinstance(embedded, dict) else None
        if isinstance(pinned, int) and not isinstance(pinned, bool):
            if pinned <= int(clip_upper):
                raise ValueError(
                    f"input_spec['embedded_cols'][{target_col!r}] = {pinned} "
                    f"is too small for clip_target_upper={clip_upper}. "
                    f"Need cardinality >= {int(clip_upper) + 1} so all clipped "
                    f"target values in [0, {int(clip_upper)}] fit in the embedding."
                )

    # 1) Engineered calendar features (only the ones requested).
    if time_features:
        panel = add_time_features(
            panel,
            time_features=time_features,
            frequency=frequency,
            base_year=base_year,
            periods_per_year=periods_per_year,
            time_cols=time_cols,
            date_col=date_col,
        )

    # 2) period_start anchor.
    panel = add_period_start(
        panel, frequency=frequency, time_cols=time_cols, date_col=date_col,
    )

    # 2b) Autoregressive target-derived features (recency / activity). Here we only
    # CREATE the columns as zero placeholders; their values are filled in step 8 from
    # the CALIBRATION target alone. They must NOT be computed over the full per-customer
    # series: that series also spans the holdout (and any pre-window history), so a
    # customer's recency / frequency / tenure would absorb holdout activity — e.g.
    # "frequency" inferred from purchases in the forecast window. That is leakage, and
    # it would also disagree with inference, where `ARFeatureState` seeds purely from
    # the calibration target (models/monte_carlo_forecasting.py). Creating them now
    # keeps the column-existence check (step 4) and the window slices happy; the holdout
    # AR columns stay zero and are recomputed per step from the SAMPLED target during
    # the MC rollout, so they are never read as inputs.
    if ar_features:
        validate_ar_features(ar_features)
        collide = [n for n in ar_features if n in panel.columns]
        if collide:
            raise ValueError(
                f"ar_features names collide with existing panel columns: {collide}"
            )
        for n in ar_features:
            panel[n] = np.float32(0)

    # 3) Schema → seq_cols + target index.
    # target_col is declared once in DATA_CONFIG. If schema omits the 'target'
    # group, fill it from target_col; if present, it must match.
    schema = dict(schema)
    schema_targets = list(schema.get("target", []))
    if not schema_targets:
        schema_targets = [target_col]
        schema["target"] = schema_targets
    elif len(schema_targets) != 1:
        raise ValueError(
            f"schema['target'] must have exactly one element, got {schema_targets}"
        )
    elif target_col not in schema_targets:
        raise ValueError(
            f"target_col {target_col!r} must appear in schema['target'] "
            f"(got {schema_targets})"
        )

    # observed_past (unknown-future) covariates are NOT YET SUPPORTED: the
    # autoregressive holdout simulator would have to feed their true future
    # values, which is leakage. Drop them for now (so they never enter the
    # tensors / model / simulator). Planned later: encoder-only conditioning
    # during calibration warm-up, or lagging them into known_future.
    observed_past = list(schema.get("observed_past_time_varying_inputs", []))
    if observed_past:
        warnings.warn(
            "observed_past_time_varying_inputs are not yet supported by the "
            f"autoregressive simulator and are being dropped: {observed_past}. "
            "Planned later: encoder-only conditioning, or lag them into "
            "known_future_time_varying_inputs.",
            stacklevel=2,
        )
        schema["observed_past_time_varying_inputs"] = []

    # Register the AR-derived columns so they enter seq_cols (as continuous
    # covariates unless the caller also embeds them).
    if ar_features:
        schema["ar_features"] = list(ar_features)

    seq_cols = get_seq_cols(schema)
    target_idx = seq_cols.index(target_col)

    # 4) Validate columns exist and are numeric.
    validate_columns(panel, target_col, seq_cols, id_col)

    # 5) Train / holdout slices.
    training_start = pd.Timestamp(config["training_start"])
    training_end   = pd.Timestamp(config["training_end"])
    validation_start = pd.Timestamp(config["validation_start"])
    holdout_start  = pd.Timestamp(config["holdout_start"])
    holdout_end    = pd.Timestamp(config["holdout_end"])

    # Calibration slice is bounded on BOTH sides (symmetric with the holdout
    # slice below). Without the lower bound, panels whose source data starts
    # earlier than `training_start` would silently leak those pre-window rows
    # into the calibration tensor — passing the uniform-period-count check
    # only by coincidence and producing a wrong T_CAL. Keeping both bounds is
    # also what the coverage error message at the next step assumes.
    train_panel = panel[
        (panel["period_start"] >= training_start)
        & (panel["period_start"] <= training_end)
    ].copy()
    holdout_panel = panel[
        (panel["period_start"] >= holdout_start)
        & (panel["period_start"] <= holdout_end)
    ].copy()

    # 5a-pre) Coverage check. If the user's training or holdout window falls
    # outside the panel's date range, the slice is empty and every downstream
    # check (cohort selection, uniform period counts, tensor reshape) fails
    # with a misleading error (e.g. value_counts() on an empty Series → "0
    # unique counts"). Raise something explicit instead, citing both the
    # requested window and the panel's actual coverage so the fix is obvious.
    panel_min = panel["period_start"].min()
    panel_max = panel["period_start"].max()
    panel_range = (
        f"panel covers {panel_min.date()}..{panel_max.date()}"
        if pd.notna(panel_min) and pd.notna(panel_max)
        else "panel is empty"
    )
    if train_panel.empty:
        raise ValueError(
            f"training window {training_start.date()}..{training_end.date()} "
            f"contains 0 rows ({panel_range}). "
            f"Check the training_start / training_end dates in your config."
        )
    if holdout_panel.empty:
        raise ValueError(
            f"holdout window {holdout_start.date()}..{holdout_end.date()} "
            f"contains 0 rows ({panel_range}). "
            f"Check the holdout_start / holdout_end dates in your config."
        )

    # 5a) Cohort filter (Valendin et al.): keep only customers active during the
    # calibration window. Applied to BOTH slices so the train/holdout customer sets
    # stay identical (enforced by the ids_train == ids_hold check below) and so the
    # Pareto/NBD benchmark, which reads the returned train_panel, fits the same
    # cohort as the LSTM. On by default; see PanelConfig.require_calibration_activity.
    if config.get("require_calibration_activity", True):
        active_ids = select_active_cohort(train_panel, id_col, target_col)
        train_panel = train_panel[train_panel[id_col].isin(active_ids)].copy()
        holdout_panel = holdout_panel[holdout_panel[id_col].isin(active_ids)].copy()

    # 5a-bis) Early heads-up on embedding drift: an embedded known_future column
    # whose holdout introduces categories unseen in calibration will index
    # untrained embedding rows at forecast time. It is sized correctly (no
    # crash), but it is a likely data-prep slip, so warn now — on the final,
    # cohort-filtered slices — before any training happens.
    warn_known_future_drift(
        train_panel, holdout_panel,
        schema=schema, input_spec=input_spec, target_col=target_col,
    )

    # 5b) Clip the target on the training window only — holdout stays
    # untouched so evaluation runs against the real actuals.
    if clip_upper is not None:
        train_panel[target_col] = train_panel[target_col].clip(upper=int(clip_upper))

    # 5c) Fill the AR feature columns on the CALIBRATION window only (created as zero
    # placeholders in step 2b). Computed per customer, in period order, from the CLIPPED
    # calibration target — the exact array `ARFeatureState` re-seeds from at forecast time
    # — so the training AR features and the holdout rollout come from the identical
    # primitive on identical inputs, with NO holdout leakage. `holdout_panel`'s AR columns
    # stay zero: the rollout overwrites them each step from the sampled target
    # (models/monte_carlo_forecasting.py), so a zero placeholder is never read. Done here,
    # before resolve_embedded_cols (step 7b), so an embedded AR feature is sized from its
    # real calibration values rather than the zeros.
    if ar_features:
        train_panel = train_panel.sort_values(
            [id_col, "period_start"], kind="stable"
        ).reset_index(drop=True)
        for positions in train_panel.groupby(id_col, sort=False).indices.values():
            y = train_panel[target_col].to_numpy()[positions][None, :]   # (1, T_CAL) clipped
            cols = compute_ar_feature_columns(y, ar_features)
            for n in ar_features:
                train_panel.iloc[positions, train_panel.columns.get_loc(n)] = cols[n][0]

    # 6) Uniform period counts.
    train_counts = train_panel.groupby(id_col).size()
    hold_counts  = holdout_panel.groupby(id_col).size()
    if train_counts.nunique() != 1:
        raise ValueError(
            "Customers have inconsistent training-period counts:\n"
            f"{train_counts.value_counts()}"
        )
    if hold_counts.nunique() != 1:
        raise ValueError(
            "Customers have inconsistent holdout-period counts:\n"
            f"{hold_counts.value_counts()}"
        )
    T_CAL  = int(train_counts.iloc[0])
    T_HOLD = int(hold_counts.iloc[0])

    # 6b) Temporal validation boundary. Map `validation_start` (a date) onto a
    # calibration PERIOD index: val_start_idx = how many calibration periods fall
    # strictly before validation_start. The distinct, sorted calibration
    # period_start values ARE the T_CAL periods (uniform across customers, just
    # checked above), so counting those below the cutoff gives the split point.
    #   periods 0 .. val_start_idx-1   -> train the weights
    #   periods val_start_idx .. T_CAL-1 -> validation window (never trained on)
    # The PanelConfig date check already guarantees the date ordering; here we
    # re-check against the real calendar so a date that lands on the very first
    # or past the last calibration period (leaving no train or no val period)
    # fails loudly rather than producing an empty slice downstream.
    cal_periods = np.sort(train_panel["period_start"].unique())   # (T_CAL,)
    val_start_idx = int((cal_periods < validation_start).sum())
    if not 0 < val_start_idx < T_CAL:
        raise ValueError(
            f"validation_start={validation_start.date()} maps to calibration "
            f"period index {val_start_idx}, but it must be in (0, T_CAL={T_CAL}) "
            f"so that at least one period trains the weights and at least one is "
            f"held out for validation. Calibration covers "
            f"{pd.Timestamp(cal_periods[0]).date()}..{pd.Timestamp(cal_periods[-1]).date()}."
        )
    n_val_periods = T_CAL - val_start_idx

    sort_cols = [id_col, "period_start"]
    ids_train = train_panel.sort_values(sort_cols)[id_col].drop_duplicates().tolist()
    ids_hold  = holdout_panel.sort_values(sort_cols)[id_col].drop_duplicates().tolist()
    if ids_train != ids_hold:
        raise ValueError(
            "Train and holdout customers differ (or appear in a different order)"
        )
    ids = ids_train
    N = len(ids)

    # 7) NaN check on the columns we're about to reshape.
    for label, frame in (("train_panel", train_panel), ("holdout_panel", holdout_panel)):
        nan_counts = frame[seq_cols].isna().sum()
        if int(nan_counts.sum()) > 0:
            bad = nan_counts[nan_counts > 0]
            raise ValueError(f"NaN values in {label}: {dict(bad)}")

    # 7b) Resolve embedding cardinalities ('auto'/missing inferred from the
    # data per column role; pinned ints validated). Returned to the caller so
    # the model is built from the SAME resolved spec.
    resolved_embedded = resolve_embedded_cols(
        input_spec,
        target_col=target_col,
        seq_cols=seq_cols,
        schema=schema,
        train_panel=train_panel,
        holdout_panel=holdout_panel,
        clip_upper=clip_upper,
    )

    # 8) Reshape + AR (samples, targets). The AR feature columns were filled on the
    # calibration window in step 5c (and left zero on the holdout), so make_block carries
    # them straight into the tensors.
    calibration = make_block(train_panel,   ids, sort_cols, seq_cols)
    holdout     = make_block(holdout_panel, ids, sort_cols, seq_cols)
    samples     = calibration[:, :-1, :]
    targets     = calibration[:, 1:, target_idx:target_idx + 1]

    if verbose:
        print(f"N={N} T_CAL={T_CAL} T_HOLD={T_HOLD} F={len(seq_cols)}")
        print(
            f"validation_start={validation_start.date()} -> val_start_idx="
            f"{val_start_idx} (train periods 0..{val_start_idx - 1}, "
            f"validation periods {val_start_idx}..{T_CAL - 1}, V={n_val_periods})"
        )
        print(f"seq_cols   = {seq_cols}")
        print(f"target_col = {target_col!r} at index {target_idx}")
        print(
            f"calibration {calibration.shape} | samples {samples.shape} "
            f"| targets {targets.shape} | holdout {holdout.shape}"
        )
        if resolved_embedded is not None:
            print(f"embedded_cols = {resolved_embedded}")

    return {
        "calibration":   calibration,
        "holdout":       holdout,
        "samples":       samples,
        "targets":       targets,
        "seq_cols":      seq_cols,
        "target_col":    target_col,
        "target_idx":    target_idx,
        # Temporal validation split (see step 6b): the calibration PERIOD index
        # where the validation window begins, its length, and the source date.
        # Consumed by make_loaders (train = prefix transitions, val = full
        # sequence scored on the suffix) and the Optuna rollout horizon.
        "val_start_idx":   val_start_idx,
        "n_val_periods":   n_val_periods,
        "validation_start": config["validation_start"],
        # id_col + frequency are carried so the data dict is self-describing:
        # downstream consumers (e.g. the Pareto/NBD benchmark) can recover the
        # customer key and the period length without re-passing the config.
        "id_col":        id_col,
        "frequency":     frequency,
        "input_spec":    ({"embedded_cols": resolved_embedded}
                          if resolved_embedded is not None else None),
        "ar_features":   ar_features,
        "N  ":             N,
        "T_CAL":         T_CAL,
        "T_HOLD":        T_HOLD,
        "F":             len(seq_cols),
        "ids":           ids,
        "panel":         panel,
        "train_panel":   train_panel,
        "holdout_panel": holdout_panel,
    }


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Everything the pipeline needs lives in one validated PanelConfig. The
    # panel here matches the columns produced by the data-integration notebook:
    #   Id, year, week, Transactions, Gender, Income, high.season

    # ---- Minimal config: just the target + the raw weekly index ---------
    # No engineered calendar features, no covariates, no embeddings — the
    # `week` column is fed directly as a (precomputed) time feature.
    cfg_minimal = PanelConfig(
        id_col="Id", target_col="Transactions", frequency="weekly",
        time_cols=("year", "week"), periods_per_year=52,
        training_start="1999-01-01", training_end="2000-12-31",
        validation_start="2000-07-01", holdout_start="2001-01-01",
        holdout_end="2002-12-31",
        time=("week",),
    )

    # ---- Full config: engineered calendar features + covariates ---------
    # `add_week_sin_cos` engineers + auto-assigns week_sin/week_cos to the time
    # role; `add_year_idx` creates year_idx, placed here in known_future
    # alongside the externally-known high.season flag. Gender/Income are static.
    cfg_full = PanelConfig(
        id_col="Id", target_col="Transactions", frequency="weekly",
        time_cols=("year", "week"), periods_per_year=52,
        training_start="1999-01-01", training_end="2000-12-31",
        validation_start="2000-07-01", holdout_start="2001-01-01",
        holdout_end="2002-12-31",
        known_future=("year_idx", "high.season"),
        static=("Gender", "Income"),
        time_features={"add_year_idx": True, "add_week_sin_cos": True},
    )

    # ---- Run the pipeline -----------------------------------------------
    # Point `csv_path` at any panel CSV with the columns the configs reference.
    # How the panel was built is out of scope for this file — produce it from
    # raw transactions in your data-integration notebook, or by hand.
    csv_path = "Datasets/electronic_panel.csv"
    panel = pd.read_csv(csv_path)

    print("=== minimal config ===")
    data_min = prepare_dataset(panel, cfg_minimal)

    print("\n=== full config ===")
    data_full = prepare_dataset(panel, cfg_full)
