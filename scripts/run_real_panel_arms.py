"""The seasonal grid's arm axis, on the two REAL panels.

`grids/seasonal_4x4x10.py` crossed three AR encodings with two cluster settings and
found that none of them closes the gap to the Pareto/NBD: the best of twelve arms sits
at 54.2% mean |bias| against the benchmark's 15.8%, and no arm flattens the churn
gradient (`docs/insights-study.md` §2, §5). That answers the question on SYNTHETIC
panels, where the Pareto/NBD is the true model by construction and therefore the ceiling.

This script asks it on CDNOW and electronics, on the same axes, so the two runs can be
read together. It also puts the frozen benchmark beside the contribution, which is the
thing that has failed twice.

**Why ValendinLSTM produced nothing in two grids.** Not a fleet problem.
`benchmarks/valendin_lstm.py:99` refuses ANY non-embedded `seq_col` — the published
model has no covariate path (ADR-0004). Engineered time features and AR features are
both plain numerics that `prepare_dataset` does not embed, so both trip it; cluster
features are auto-embedded and are fine (VastAI/known_failures.md F11). `refuses()` below is the one place that rule lives, and it
reads the BUILT dataset rather than the arm declaration, so it stays true if a feature
later becomes embedded. Three callers use it — the work list, the preflight and
`--check-complete` — so an ineligible pair is never schedulable and never silently empty.

**Why Pareto/NBD is not crossed with the arms.** It is fit from the target column alone,
so an arm would train identical copies. It gets one suite per panel, un-suffixed, run on
the orchestrator: it needs no GPU and `studies/runner.py` gives it one MCMC fit
regardless of `n_studies_per_model`.

**Why one suite per arm.** `ar_features` and `cluster_features` are `PanelConfig`
properties and `StudySuiteConfig.data` is a single shared dataset, so each arm needs its
own `prepare_dataset` and therefore its own suite — the same reasoning as
`run_ar_encoding_ablation.py`. Every root is disjoint, so collecting from rented workers
is a plain copy (VastAI/Rules.md §4, §6).

**Read the distribution, not paired differences.** Training is unseeded (CLAUDE.md
priority 3): `base_seed + i` drives the Optuna sampler and the Monte Carlo forecast, not
weight init, `DataLoader` shuffling or dropout. Replications are genuine replications, so
the mean / SD / min / max across them is the honest summary.

Usage:
    # gate the launch — builds every arm, trains every eligible cell tiny, costs nothing
    python scripts/run_real_panel_arms.py --preflight

    # the benchmark row, locally, before any rental
    python scripts/run_real_panel_arms.py --model pareto_nbd --panel cdnow
    python scripts/run_real_panel_arms.py --model pareto_nbd --panel electronics

    # one worker's slice (this is what a rented box runs)
    python scripts/run_real_panel_arms.py --model transformer --shard 3/7

    # timing probe on a fresh box, before committing a slice to it
    python scripts/run_real_panel_arms.py --model transformer --panel cdnow \
        --arm no_ar-no_cluster-valendin --n-studies 1 --n-trials 5 --suite-suffix probe

    # after the run
    python scripts/run_real_panel_arms.py --check-complete
    python scripts/run_real_panel_arms.py --report --panel cdnow
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.models import compute_forecast_metrics
from panelclv.studies import (
    ModelSpec,
    StudySuiteConfig,
    load_model_predictions,
    run_study_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"
CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"

EXPERIMENT = "real_panel_arms"

# --- study size ------------------------------------------------------------------
# One arm-suite is N_STUDIES studies x N_TRIALS trainings, plus one refit and one
# N_SIMULATIONS-path rollout per study.
#
# N_SIMULATIONS was 50 for the first generation of this run and is now 200, matching
# `grids/seasonal_4x4x10.py`. The 50-path suites are preserved under
# `Studies/_archive_50sim/` and are NOT comparable with these: a path count is part of
# what produced a number, so the two generations must not be pooled or read against
# each other. Everything under `Studies/real_panel_arms__*` is 200-path.
#
# What that costs, MEASURED on one box on 2026-09-06 (one CDNOW study at two path
# counts, hardware held fixed): 4.38 s per path + 189 s fixed. `simulate_attention_path`
# is stateless and re-reads a growing context at every step, for every path, so the
# rollout dominates a Transformer suite -- 54% of it at 50 paths, ~82% at 200. Going to
# 200 therefore costs roughly 2.6x on the Transformer and less on the LSTM.
#
# What it buys is small and worth stating honestly: one path's aggregate has sd ~2.6% of
# the holdout total, so 50 paths already left ~0.37% Monte Carlo noise against the ~23pp
# across-study sd that actually limits a result. 200 paths halves an error term that was
# two orders of magnitude below the binding one. The reason to run it is consistency
# with the synthetic grid, not precision.
N_STUDIES = 20
N_SIMULATIONS = 200

# Trials per study, per model. The LSTM and the Transformer MUST stay equal -- an
# unequal budget makes a difference attributable to search effort rather than to the arm
# (the confound commit 2d815b3 fixed on the synthetic side). ValendinLSTM is lower
# because its registry entry searches only learning_rate / weight_decay / batch_size:
# ADR-0004 freezes the architecture, so there is no width to find and 50 TPE trials over
# three dimensions is spend rather than search.
N_TRIALS = {"lstm": 50, "transformer": 50, "valendin_lstm": 25}

# Shard -> base seed. Study i of a shard draws `base_seed + i`, so at N_STUDIES = 20
# shard "a" covers seeds 43-62 and shard "b" covers 63-82: disjoint. The gap between the
# base seeds must stay >= N_STUDIES or half the replications become duplicates in
# silence. Only "a" is run by default; "b" exists to add 20 more replications to an arm
# whose confidence interval turns out to overlap a neighbour's.
SHARDS: dict[str, int] = {"a": 42, "b": 62}


# ---------------------------------------------------------------------------
# The arm axis — copied from grids/seasonal_4x4x10.py so the two runs share axes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One feature/encoding configuration every model is trained under.

    An arm is not a model and not a panel: it is a third axis crossed with both.
    `ar_features` and `cluster_features` are `PanelConfig` fields, so each arm needs its
    own `prepare_dataset` and therefore its own suite.

    `calendar` overrides how the panel encodes calendar time (see the CAL_* constants).
    It is a flag rather than a fourth axis because its meaningful values differ by panel
    -- electronics engineers sin/cos by default and only ever turns it off, CDNOW
    engineers nothing and turns one of two encodings on -- so crossing it everywhere
    would schedule cells that do not exist. `None` means "whatever this panel does by
    default", which is what every arm inherited from the archived configs uses.
    """

    name: str
    ar_features: tuple[str, ...] = ()
    cluster_features: tuple[str, ...] = ()
    calendar: str | None = None


# The Pareto/NBD sufficient statistics (t_x, x, T). Two of the three are capped by the
# calibration window and keep counting through the holdout, which is the diagnosed
# failure (docs/insights-study.md §4.3) -- this arm reproduces it on real panels.
AR_UNBOUNDED = (
    "period_since_last_transaction",
    "cumulative_transactions",
    "period_since_first_transaction",
)


def bounded_flags(deepest: int) -> tuple[str, ...]:
    """Nested activity flags up to `deepest`, plus `has_transacted_before`.

    A bounded step encoding of the same silence `period_since_last_transaction`
    measures: every flag is 0 beyond its bin, so no holdout value can leave the range
    the weights were fitted on. `active_in_last_1_periods` is omitted because the target
    channel already carries the previous count.
    """
    ks = [k for k in (2, 4, 8, 16, 32, 52) if k <= deepest]
    return tuple(f"active_in_last_{k}_periods" for k in ks) + ("has_transacted_before",)


# The deepest bin is a WINDOW LENGTH, so it differs by panel: at or above the
# calibration length a flag can never be 0 while fitting and degenerates into a copy of
# `has_transacted_before` (`check_arm_depth` enforces this). Electronics at 32 of 104
# periods is bin-for-bin identical to the grid's AR_BOUNDED; CDNOW at 16 of 39 is one
# bin shallower and cannot be otherwise. That is a real difference in the axis and must
# be stated wherever the real and synthetic runs are compared.
BOUNDED_DEPTH = {"electronics": 32, "cdnow": 16}

CLUSTER_AXIS = {"no_cluster": (), "kmeans_8": ("kmeans_8",)}

# How an arm encodes calendar time. Three encodings, and which of them a panel can use
# is a fact about that panel's windows, not a preference:
#
#   CAL_NONE      no calendar column at all. Every seq_col stays embedded, so this is
#                 the one setting ValendinLSTM can always read (F11).
#   CAL_SIN_COS   week_sin/week_cos. Continuous and periodic, so it is the only encoding
#                 that carries a usable value for a calendar week the calibration window
#                 never contained -- which is CDNOW's situation exactly.
#   CAL_WEEK_EMB  week as an embedded categorical. The published model's own encoding
#                 (Valendin et al. read week as a category), and therefore the only way
#                 to give the frozen benchmark calendar information. Unlike sin/cos it
#                 has one weight row per week, so a week absent from calibration is
#                 forecast from that row's random initialization.
CAL_NONE = "none"
CAL_SIN_COS = "sin_cos"
CAL_WEEK_EMB = "week_emb"

# The embedder axis has one value, so it is not crossed -- but it stays in the arm name
# because the archived seasonal trees are named `<ar>-<cluster>-<embedder>` and a reader
# comparing the two runs should not have to translate.
EMBEDDER = "valendin"


def arms_for(panel: str) -> dict[str, Arm]:
    """`{arm name: Arm}` for one panel. The single declaration of the axis.

    Four arms everywhere: 2 AR encodings (no_ar, ar_bounded) x 2 cluster settings. Each panel then gets extra
    arms that vary only its calendar encoding, named as their twin plus a suffix so each
    sorts beside the cell it differs from by one setting:

    - electronics `-no_tf`: its `no_ar` arms with the engineered time features removed.
      The only shape ValendinLSTM can run in on that panel (F11).
    - CDNOW `-tf`: all six cells with `week_sin`/`week_cos` engineered. CDNOW's archived
      configs carry no calendar column at all, which was inherited rather than decided;
      these arms are what decides it, by measuring both.
    - CDNOW `-week_emb`: its `no_ar` cells with week as an embedded categorical. This is
      the published model's own encoding and the only one the benchmark can read, so it
      is what puts ValendinLSTM into the calendar comparison at all. Read its result
      knowing that 13 of CDNOW's 52 weeks (39-51) never occur in calibration, so those
      embedding rows go to the holdout untrained -- the contrast with `-tf` on the same
      cells is a measurement of what that costs.
    """
    # `ar_unbounded` (AR_UNBOUNDED above) is deliberately NOT crossed here any more. It
    # is the diagnosed-broken encoding -- +198%/+461% bias on these two panels in the
    # 50-path archive, and the same monotone failure across the synthetic grid -- so it
    # has already answered its question and re-measuring it at 200 paths would only
    # re-establish a known result at four times the rollout cost. The constant stays
    # defined because the archived suites are named after it and `--report`'s
    # reproduction check still knows its numbers.
    ar_axis = {
        "no_ar": (),
        "ar_bounded": bounded_flags(BOUNDED_DEPTH[panel]),
    }
    arms = {
        f"{ar}-{cl}-{EMBEDDER}": Arm(
            name=f"{ar}-{cl}-{EMBEDDER}", ar_features=f, cluster_features=c
        )
        for ar, f in ar_axis.items()
        for cl, c in CLUSTER_AXIS.items()
    }
    if panel == "electronics":
        for cl, c in CLUSTER_AXIS.items():
            name = f"no_ar-{cl}-{EMBEDDER}-no_tf"
            arms[name] = Arm(name=name, cluster_features=c, calendar=CAL_NONE)
    if panel == "cdnow":
        for ar, f in ar_axis.items():
            for cl, c in CLUSTER_AXIS.items():
                name = f"{ar}-{cl}-{EMBEDDER}-tf"
                arms[name] = Arm(
                    name=name, ar_features=f, cluster_features=c, calendar=CAL_SIN_COS
                )
        for cl, c in CLUSTER_AXIS.items():
            name = f"no_ar-{cl}-{EMBEDDER}-week_emb"
            arms[name] = Arm(name=name, cluster_features=c, calendar=CAL_WEEK_EMB)
    return arms


def calendar_kwargs(panel: str, arm: Arm) -> dict:
    """The `PanelConfig` fields that encode calendar time, for one arm on one panel.

    One function rather than a branch inside each panel config, because this is the one
    place that decides what a calendar setting MEANS -- and the mapping is not symmetric
    between the panels, so having it in two places is having it disagree in one.

    Unknown combinations raise rather than falling back to a default: an arm asking for
    an encoding this panel cannot express is a declaration bug, and silently training the
    default instead is exactly the class of failure `refuses()` exists to prevent.
    """
    if panel == "electronics":
        if arm.calendar is None:
            return {"time_features": {"add_year_idx": True, "add_week_sin_cos": True}}
        if arm.calendar == CAL_NONE:
            return {"time_features": None}
    if panel == "cdnow":
        if arm.calendar is None:
            return {}                      # the archived configs: no calendar column
        if arm.calendar == CAL_SIN_COS:
            # `add_year_idx` is deliberately NOT set here, though electronics sets it.
            # CDNOW calibrates entirely within 1997 and forecasts across into 1998, so a
            # year index is constant while fitting and out of its fitted range for the
            # back half of the holdout -- the unbounded-counter failure this whole study
            # measures, reintroduced through the calendar. sin/cos is bounded and
            # periodic and has no such range to leave.
            return {"time_features": {"add_week_sin_cos": True}}
        if arm.calendar == CAL_WEEK_EMB:
            # `week` is already a column of the panel (it is half of `time_cols`); giving
            # it the `time` role puts it in seq_cols, and embedding it keeps every
            # channel embedded, which is what the benchmark's guard requires.
            return {
                "time": ("week",),
                "embedded_cols": {"Transactions": "auto", "week": "auto"},
            }
    raise ValueError(
        f"panel {panel!r} has no calendar encoding {arm.calendar!r} (arm {arm.name!r}). "
        f"Add it to `calendar_kwargs` if it is meaningful on this panel's windows."
    )


# ---------------------------------------------------------------------------
# The two panels
# ---------------------------------------------------------------------------


def electronics_config(arm: Arm) -> PanelConfig:
    """The electronics panel as the ARCHIVED suites read it, with the arm varying.

    Windows, clipping, cohort rule and time features are copied from
    `scripts/run_ar_encoding_ablation.py`, which copied them from
    `Studies/cross_entropy_config_pareto_Comparaison_pareto/config.json`. That lineage is
    what lets `--report` check the `no_ar` and `ar_unbounded` arms against suites that
    already exist; if they do not reproduce, nothing else in the run is trustworthy.
    """
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        training_start="1999-01-01",
        validation_start="2000-01-01",   # the last calibration year
        training_end="2000-12-31",
        holdout_start="2001-01-01",
        holdout_end="2001-12-31",
        clip_target_upper=6,             # 7-class head
        require_calibration_activity=True,
        # The one thing the `-no_tf` arms change; `calendar_kwargs` owns the mapping.
        **calendar_kwargs("electronics", arm),
        ar_features=arm.ar_features,
        cluster_features=arm.cluster_features,
        known_future=(),
        static=(),
        observed_past=(),
        embedded_cols={"Transactions": "auto"},
    )


def cdnow_config(arm: Arm) -> PanelConfig:
    """The CDNOW panel exactly as the archived ablations read it, with the arm varying.

    A harsher test of the same hazard than electronics: the holdout is nearly as long as
    the calibration window (38 vs 39 periods, against 52 vs 104), so capped counters
    drift proportionally further.

    Its default carries no calendar column at all -- inherited from the archived configs
    rather than argued for, which is why the `-tf` and `-week_emb` arms exist to test it.
    Two facts constrain what those arms can mean. Calibration is weeks 0-38 of 1997, so
    it spans 0.75 of an annual cycle and the holdout runs through weeks 39-51 that
    calibration never contained: an encoding with a per-week weight has nothing fitted
    for a third of the forecast, and even sin/cos has no training signal about that
    quarter of the phase circle. Whatever these arms show, seasonal AMPLITUDE at those
    weeks is not learnable from this window -- only whether carrying the column helps or
    hurts the rest.
    """
    kwargs = dict(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        training_start="1997-01-01",
        validation_start="1997-08-06",   # 1997 week 31 - last 8 calibration weeks
        training_end="1997-09-30",       # inclusive of 1997 week 38
        holdout_start="1997-10-01",      # 1997 week 39
        holdout_end="1998-06-30",        # inclusive of the last complete week, 1998 w24
        clip_target_upper=4,             # 5-class head; 3 cells in 181,489 exceed it
        ar_features=arm.ar_features,
        cluster_features=arm.cluster_features,
        embedded_cols={"Transactions": "auto"},
    )
    # Applied last so a calendar encoding that needs its own `embedded_cols` (week_emb)
    # replaces the default rather than being silently dropped beside it.
    kwargs.update(calendar_kwargs("cdnow", arm))
    return PanelConfig(**kwargs)


PANELS = {
    "electronics": (CLEAN / "electronics_customer_week_panel.csv", electronics_config),
    "cdnow": (CLEAN / "cdnow_customer_week_panel.csv", cdnow_config),
}


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------

# Everything that is NOT the ablation, so the only thing moving is the feature set.
SHARED_TRAINING: dict[str, object] = {
    "n_epochs": 100,
    "patience": 7,
    "verbose": False,
    "loss_type": "cross_entropy",
}

# `embedding_dim` is deliberately absent from the LSTM's space: the `valendin` embedder
# has no common width, so naming one would advertise a search that never happens
# (docs/running-a-model.md §14). Neither space names `embedder` either -- the registry
# default IS "valendin", which is the one value of EMBEDDER above.
SEARCH_SPACES: dict[str, dict[str, object]] = {
    "lstm": {
        "lstm_hidden_size": {32, 64, 128},
        "dense_units":      {32, 64, 128},
        "dropout":          {0.0, 0.2},
        "learning_rate":    (1e-4, 1e-2, "log"),
        "weight_decay":     (1e-6, 1e-2, "log"),
        "batch_size":       {64, 128, 256},
    },
    "transformer": {
        "d_model":            {32, 64, 128},
        "nhead":              {2, 4, 8},
        "num_encoder_layers": (1, 3, "int"),
        "dropout":            {0.0, 0.1, 0.2, 0.3},
        "learning_rate":      (1e-4, 3e-3, "log"),
        "weight_decay":       (1e-6, 1e-2, "log"),
        "batch_size":         {64, 128, 256},
    },
    # ValendinLSTM declares NO search space: its architecture is frozen (ADR-0004), so
    # it inherits the registry entry's three training hyperparameters. Naming a width
    # here would quietly unfreeze the benchmark.
    "valendin_lstm": {},
}

MODEL_NAMES = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "valendin_lstm": "ValendinLSTM",
    "pareto_nbd": "ParetoNBD",
}

# The models crossed with the arm axis. Pareto/NBD is absent by design -- see
# `pareto_suite_name`.
NEURAL = ("lstm", "transformer", "valendin_lstm")


def model_spec(model_type: str, n_trials: int) -> ModelSpec:
    """One `ModelSpec`, built from the tables above rather than written out per model."""
    return ModelSpec(
        name=MODEL_NAMES[model_type],
        model_type=model_type,
        n_trials=n_trials,
        search_space=dict(SEARCH_SPACES[model_type]),
        training=dict(SHARED_TRAINING),
    )


# ---------------------------------------------------------------------------
# Eligibility — the one authority on which models can run which arms
# ---------------------------------------------------------------------------


def refuses(model_type: str, data: dict) -> str | None:
    """Why `model_type` cannot be trained on this arm's dataset, or None if it can.

    Keyed off the BUILT dataset, never off `arm.ar_features`: the guard in
    `benchmarks/valendin_lstm.py:99` compares `seq_cols` against `embedded_cols`, so the
    only correct way to predict it is to ask the same question of the same two lists.
    That also keeps this true if a feature later becomes embedded and the benchmark
    becomes able to read it.

    This is what turns "ValendinLSTM produced no results" from a discovery made three
    days and one fleet later into a property of the declaration.
    """
    if model_type == "valendin_lstm":
        embedded = set(data.get("embedded_cols") or {})
        covariates = [c for c in data["seq_cols"] if c not in embedded]
        if covariates:
            return (
                f"the Valendin benchmark reads embedded features only (ADR-0004), and "
                f"this arm carries {len(covariates)} non-embedded channel(s): "
                f"{covariates}"
            )
    return None


def check_arm_depth(arm_name: str, data: dict) -> None:
    """Refuse an `active_in_last_K` channel that cannot vary on this panel.

    Silence cannot exceed `T_CAL - 1` while fitting, so a flag with K at or above the
    calibration length is 1 for every customer who has ever transacted -- an exact copy
    of `has_transacted_before` -- and only becomes a distinct signal out in the holdout,
    where nothing constrained it. Cheap to check, silent and expensive to miss.
    """
    t_cal = int(data["T_CAL"])
    bad = [
        c for c in data["seq_cols"]
        if c.startswith("active_in_last_") and int(c.split("_")[3]) >= t_cal
    ]
    if bad:
        raise ValueError(
            f"arm {arm_name!r}: {bad} cannot vary on a panel with T_CAL={t_cal}. Such a "
            f"flag duplicates 'has_transacted_before' in calibration and diverges from "
            f"it only in the holdout. Lower BOUNDED_DEPTH for this panel."
        )


# ---------------------------------------------------------------------------
# Datasets, names and the work list
# ---------------------------------------------------------------------------


def build_data(panel: str, arm: Arm, verbose: bool = False) -> dict:
    """`prepare_dataset` for one (panel, arm), with the depth check applied."""
    panel_path, build_config = PANELS[panel]
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Datasets/ is gitignored, so a rented worker needs "
            f"the panel pushed to it (VastAI/Rules.md §3)."
        )
    data = panel_dataset.prepare_dataset(
        pd.read_csv(panel_path), build_config(arm), verbose=verbose
    )
    data["panel_name"] = panel
    check_arm_depth(arm.name, data)
    return data


def suite_name(
    model_type: str, panel: str, arm_name: str, shard: str, suffix: str | None = None
) -> str:
    """One disjoint root per (model, panel, arm, shard).

    The model is in the path because four models means two workers would otherwise
    target the same (panel, arm) root, and `create_suite_root` refuses a folder that
    already exists (`studies/layout.py`). Disjointness is what makes collecting from the
    fleet a plain rsync with no merge step.
    """
    stem = f"{EXPERIMENT}__{MODEL_NAMES[model_type]}__{panel}__{arm_name}__{shard}"
    return f"{stem}__{suffix}" if suffix else stem


def pareto_suite_name(panel: str, suffix: str | None = None) -> str:
    """Pareto/NBD carries no arm and no shard in its path.

    It is fit from the target column alone, so it is invariant to the arm axis and
    crossing it would train eight identical copies per panel. It also runs exactly one
    MCMC fit regardless of `n_studies_per_model`, so a shard would mean nothing.
    """
    stem = f"{EXPERIMENT}__ParetoNBD__{panel}"
    return f"{stem}__{suffix}" if suffix else stem


@dataclass(frozen=True)
class Item:
    """One unit of work: train `model_type` on (panel, arm) for one shard's studies."""

    model_type: str
    panel: str
    arm: Arm
    shard: str
    skipped: str | None = field(default=None, compare=False)


def work_list(
    model_types: tuple[str, ...] = NEURAL,
    panels: tuple[str, ...] = tuple(PANELS),
    shards: tuple[str, ...] = ("a",),
    data_cache: dict | None = None,
) -> tuple[list[Item], list[Item]]:
    """Every schedulable item, ARM-MAJOR, plus the items dropped and why.

    Arm-major ordering matters for the shard stride below. Striding `i::N` over an
    arm-major list gives every worker the same number of suites from every arm;
    dataset-major instead walks the arms in steps of `N mod A`, so a worker can end up
    covering only half of them and a lost worker guts those. An arm missing entirely is
    not a noisier result, it is no result.

    Building the datasets is the only way to answer `refuses()` honestly, so this needs
    them; pass `data_cache` to reuse them across callers.
    """
    cache = data_cache if data_cache is not None else {}
    scheduled: list[Item] = []
    dropped: list[Item] = []

    arm_names = sorted({a for p in panels for a in arms_for(p)})
    for arm_name in arm_names:                      # arm SLOWEST
        for panel in panels:
            arms = arms_for(panel)
            if arm_name not in arms:                # `-no_tf` is electronics-only
                continue
            arm = arms[arm_name]
            key = (panel, arm_name)
            if key not in cache:
                cache[key] = build_data(panel, arm)
            for shard in shards:
                for model_type in model_types:
                    reason = refuses(model_type, cache[key])
                    item = Item(model_type, panel, arm, shard, reason)
                    (dropped if reason else scheduled).append(item)
    return scheduled, dropped


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_item(
    item: Item,
    data: dict,
    n_studies: int,
    n_trials: int | None,
    n_simulations: int,
    suffix: str | None,
    device: str,
) -> Path:
    """Train one item's suite. Returns the suite root."""
    trials = n_trials if n_trials is not None else N_TRIALS[item.model_type]
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    config = StudySuiteConfig(
        studies_base_path=str(STUDIES_BASE),
        suite_name=suite_name(
            item.model_type, item.panel, item.arm.name, item.shard, suffix
        ),
        n_studies_per_model=n_studies,
        n_simulations=n_simulations,
        device=device,
        data=data,
        models=[model_spec(item.model_type, trials)],
        base_seed=SHARDS[item.shard],
        keep_only_best_checkpoint=True,
    )
    return run_study_suite(config)


def run_pareto(panel: str, n_simulations: int, suffix: str | None) -> Path:
    """One MCMC fit for one panel, arm-invariant.

    `n_studies_per_model=1` rather than N_STUDIES so the archived `config.json` does not
    record a 20 that never happened: `studies/runner.py` has no study loop for this
    model. The dataset is built from the plainest arm on the panel, since the fit reads
    only the target column and every arm shares it.
    """
    arm = arms_for(panel)[f"no_ar-no_cluster-{EMBEDDER}"]
    data = build_data(panel, arm, verbose=True)
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    config = StudySuiteConfig(
        studies_base_path=str(STUDIES_BASE),
        suite_name=pareto_suite_name(panel, suffix),
        n_studies_per_model=1,
        n_simulations=n_simulations,
        data=data,
        models=[ModelSpec(name="ParetoNBD", model_type="pareto_nbd")],
    )
    return run_study_suite(config)


# ---------------------------------------------------------------------------
# Preflight — everything that can fail before a box is rented
# ---------------------------------------------------------------------------


def preflight(train: bool = True) -> int:
    """Build every arm, print the eligibility matrix, and train every eligible cell tiny.

    Uses the REAL panels rather than a generated probe panel: they are 8 MB on disk and
    `prepare_dataset` takes seconds, so this tests the actual datasets the fleet will
    train on. Catches F11-class refusals, `check_arm_depth` failures, a degenerate
    k-means on the smaller cohort, and a rollout that reads a channel the warm-up never
    built -- all of which otherwise burn a worker to discover.

    Returns a process exit code so it can gate a launch.
    """
    cache: dict = {}
    scheduled, dropped = work_list(data_cache=cache)

    print("=" * 78)
    print("ARMS")
    print("=" * 78)
    for (panel, arm_name), data in sorted(cache.items()):
        embedded = set(data.get("embedded_cols") or {})
        non_embedded = [c for c in data["seq_cols"] if c not in embedded]
        print(
            f"{panel:12s} {arm_name:34s} F={len(data['seq_cols']):2d} "
            f"T_CAL={int(data['T_CAL']):3d} non-embedded={len(non_embedded)}"
        )

    print()
    print("=" * 78)
    print("ELIGIBILITY")
    print("=" * 78)
    for item in dropped:
        print(f"  SKIP  {MODEL_NAMES[item.model_type]:13s} {item.panel:12s} "
              f"{item.arm.name:34s} {item.skipped}")
    counts: dict[str, int] = {}
    for item in scheduled:
        counts[item.model_type] = counts.get(item.model_type, 0) + 1
    print(f"\n  scheduled: " + ", ".join(
        f"{MODEL_NAMES[m]} {n}" for m, n in sorted(counts.items())
    ))
    print(f"  refused:   {len(dropped)}")

    # Assert the negative as well as the positive. A cell that is absent because it was
    # never scheduled and a cell that is absent because it crashed look identical on
    # disk; this is what makes the difference a declared one.
    for item in dropped:
        assert refuses(item.model_type, cache[(item.panel, item.arm.name)]), (
            f"{item.model_type} on {item.panel}/{item.arm.name} was dropped but "
            f"refuses() now permits it — the work list and the predicate disagree"
        )

    if not train:
        return 0

    print()
    print("=" * 78)
    print(f"TINY TRAIN — {len(scheduled)} eligible cells")
    print("=" * 78)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for item in scheduled:
            label = (f"{MODEL_NAMES[item.model_type]:13s} {item.panel:12s} "
                     f"{item.arm.name:34s}")
            try:
                config = StudySuiteConfig(
                    studies_base_path=tmp,
                    suite_name=suite_name(
                        item.model_type, item.panel, item.arm.name, item.shard, "preflight"
                    ),
                    n_studies_per_model=1,
                    n_simulations=2,
                    device=device,
                    data=cache[(item.panel, item.arm.name)],
                    models=[
                        ModelSpec(
                            name=MODEL_NAMES[item.model_type],
                            model_type=item.model_type,
                            n_trials=1,
                            search_space=dict(SEARCH_SPACES[item.model_type]),
                            training={**SHARED_TRAINING, "n_epochs": 3, "patience": 2},
                        )
                    ],
                    base_seed=SHARDS[item.shard],
                    keep_only_best_checkpoint=True,
                )
                run_study_suite(config)
                print(f"  ok    {label}")
            except Exception as exc:                    # noqa: BLE001 — report them all
                print(f"  FAIL  {label} {type(exc).__name__}: {exc}")
                failures.append(label)

    if failures:
        print(f"\n{len(failures)} cell(s) failed. Fix before renting anything.")
        return 1
    print(f"\nAll {len(scheduled)} cells train. Safe to launch.")
    return 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, as Pearson on average-tied ranks.

    Written out rather than imported from `scipy.stats`: scipy is an undeclared
    transitive dependency of this project (it arrives via scikit-learn), so a script that
    names it directly would break on an environment that trimmed it.
    """
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def score_suite(root: Path, model_name: str, data: dict, base_seed: int) -> list[dict]:
    """Score every stored forecast in one suite root, one row per replication.

    Deliberately reads `Predictions/Prediction_*.csv` rather than the suite's
    `results.csv`. `run_study_suite` saves each study's forecast inside its loop but
    writes `results.csv`, `metrics.csv` and `config.json` only after the last one, so a
    shard killed by its watchdog leaves completed replications that none of those three
    files describe. Scoring the predictions directly makes a partial shard count for
    exactly the replications it finished, which is what lets a watchdog be a budget
    bound rather than an all-or-nothing gamble (F19).

    Alignment follows `studies.suite_metrics`: predictions are stored in the cohort's own
    order, and the ids are asserted against the rebuilt cohort so row i of the forecast
    and row i of the actuals are the same customer.
    """
    actual = holdout_actuals(data)                          # (N, T_HOLD)
    actual_totals = actual.sum(axis=1)
    ref_ids = np.asarray(data["ids"])

    model_dir = root / model_name
    if not (model_dir / "Predictions").is_dir():
        return []

    rows: list[dict] = []
    for path in sorted((model_dir / "Predictions").glob("Prediction_*.csv")):
        study = int(path.stem.split("_")[-1])
        values, ids = load_model_predictions(model_dir, study=study)
        if ids is not None and not np.array_equal(np.asarray(ids), ref_ids):
            raise ValueError(
                f"{root.name} study {study}: prediction ids do not match the rebuilt "
                f"cohort — is this the panel the suite was built from?"
            )
        rows.append(
            {
                "study": study,
                "seed": base_seed + study,
                **compute_forecast_metrics(actual, values),
                "spearman": spearman(values.sum(axis=1), actual_totals),
            }
        )
    return rows


def report(panel: str, suffix: str | None = None) -> None:
    """Pool every model and arm on one panel and print the comparison."""
    cache: dict = {}
    rows: list[dict] = []

    for arm_name, arm in arms_for(panel).items():
        for shard, base_seed in SHARDS.items():
            for model_type in NEURAL:
                root = STUDIES_BASE / suite_name(
                    model_type, panel, arm_name, shard, suffix
                )
                if not (root / MODEL_NAMES[model_type] / "Predictions").is_dir():
                    continue
                # Rebuild lazily: only needed to score, and only for arms with forecasts.
                if (panel, arm_name) not in cache:
                    cache[(panel, arm_name)] = build_data(panel, arm)
                for row in score_suite(
                    root, MODEL_NAMES[model_type], cache[(panel, arm_name)], base_seed
                ):
                    rows.append({
                        "model": MODEL_NAMES[model_type], "arm": arm_name,
                        "shard": shard, **row,
                    })

    # The benchmark row, which every arm is read against. Arm-invariant, so it is one row.
    pareto_root = STUDIES_BASE / pareto_suite_name(panel, suffix)
    if (pareto_root / "ParetoNBD" / "Predictions").is_dir():
        plain = arms_for(panel)[f"no_ar-no_cluster-{EMBEDDER}"]
        if (panel, plain.name) not in cache:
            cache[(panel, plain.name)] = build_data(panel, plain)
        for row in score_suite(pareto_root, "ParetoNBD", cache[(panel, plain.name)], 42):
            rows.append({"model": "ParetoNBD", "arm": "(none)", "shard": "-", **row})

    if not rows:
        print(f"No finished suites found for panel {panel!r}. Run them first.")
        return

    per_study = pd.DataFrame(rows)

    # Coverage FIRST: a shard the watchdog cut short must be visible as a short count
    # rather than silently shrinking an arm's sample.
    print(f"\n{len(per_study)} replications on {panel}")
    print(f"\nreplications per (model, arm, shard) — {N_STUDIES} expected each:")
    print(per_study.groupby(["model", "arm", "shard"]).size().to_string())

    # Distribution across replications, not a pooled point estimate: a single mean hides
    # exactly the across-study spread this run exists to measure.
    summary = per_study.groupby(["arm", "model"], sort=False).agg(
        n=("bias_percent", "size"),
        bias_mean=("bias_percent", "mean"),
        bias_sd=("bias_percent", lambda s: s.std(ddof=1)),
        bias_min=("bias_percent", "min"),
        bias_max=("bias_percent", "max"),
        abs_bias_mean=("bias_percent", lambda s: s.abs().mean()),
        spearman_mean=("spearman", "mean"),
        mape_mean=("mape_aggregate", "mean"),
        mape_sd=("mape_aggregate", lambda s: s.std(ddof=1)),
        rmse_mean=("rmse", "mean"),
    )
    print()
    print(summary.round(3).to_string())

    print(
        "\nRead the distribution, not the mean alone: a gap smaller than bias_sd is not"
        "\na result. RMSE is shown for completeness only — it is dominated by the zeros"
        "\nand separates nothing here."
    )

    # Reproduction check against the archive. If the arms this run shares with the
    # existing ablations do not land where they landed before, nothing else is
    # trustworthy — the same role `ar_unbounded` plays in run_ar_encoding_ablation.py.
    expected = {"electronics": {"no_ar": 22.0, "ar_unbounded": 235.0},
                "cdnow": {"no_ar": 2.0, "ar_unbounded": 334.0}}[panel]
    print("\nReproduction check against the archived LSTM ablations:")
    for stem, archived in expected.items():
        key = (f"{stem}-no_cluster-{EMBEDDER}", "LSTM")
        if key not in summary.index:
            continue
        got = summary.loc[key, "bias_mean"]
        ok = abs(got - archived) <= max(20.0, 0.5 * abs(archived))
        print(f"  {stem:14s} got {got:+8.1f}%   archived {archived:+8.1f}%   "
              f"{'OK' if ok else 'OFF — investigate before trusting the other arms'}")


def plot_tracking(panel: str, suffix: str | None = None) -> Path:
    """Weekly-aggregate tracking curve: each model's ENSEMBLE against the actual.

    The ensemble -- the mean of the 20 fits -- rather than one fit, because that is what
    a practitioner deploys and the only thing comparable to Pareto/NBD's single
    deterministic fit (see docs/insights-study.md §9).

    Each model is drawn at its best arm on this panel, so the picture compares models
    rather than one model against another's handicap. This is the plot that shows what
    `bias_percent` cannot: a forecast can hit the right total while tracking the wrong
    shape, by under-predicting early and over-predicting late.
    """
    import matplotlib
    matplotlib.use("Agg")
    from panelclv.evaluation.plots import plot_weekly_aggregated

    arms = arms_for(panel)
    # The cohort is arm-invariant, so one actuals array serves every arm.
    plain = arms[f"no_ar-no_cluster-{EMBEDDER}"]
    actual = holdout_actuals(build_data(panel, plain))

    # Pick each model's best arm by ensemble MAPE, not by |bias|. Selecting on bias
    # picks whichever arm hits the total by cancellation -- under-predicting early and
    # over-predicting late -- which is exactly the failure this figure exists to show,
    # so choosing arms that way would hide it behind its own symptom.
    series: dict[str, np.ndarray] = {}
    for model_type in NEURAL:
        best, best_score = None, float("inf")
        for arm_name in arms:
            md = STUDIES_BASE / suite_name(
                model_type, panel, arm_name, "a", suffix
            ) / MODEL_NAMES[model_type]
            paths = sorted((md / "Predictions").glob("Prediction_*.csv")) if (
                md / "Predictions").is_dir() else []
            if not paths:
                continue
            runs = [load_model_predictions(md, study=int(q.stem.split("_")[-1]))[0]
                    for q in paths]
            ens = np.mean(runs, axis=0)
            score = compute_forecast_metrics(actual, ens)["mape_aggregate"]
            if score < best_score:
                best, best_score = (arm_name, ens), score
        if best:
            series[f"{MODEL_NAMES[model_type]} · {best[0].replace('-' + EMBEDDER, '')}"] = best[1]

    pareto = STUDIES_BASE / pareto_suite_name(panel, suffix) / "ParetoNBD"
    if (pareto / "Predictions").is_dir():
        series["Pareto/NBD"] = load_model_predictions(pareto, study=1)[0]

    out = REPO_ROOT / "figures"
    out.mkdir(exist_ok=True)
    path = out / f"{EXPERIMENT}__{panel}__weekly_aggregate.png"
    plot_weekly_aggregated(
        actual.sum(axis=0), series,
        title=f"{panel} holdout — weekly aggregate transactions "
              f"(ensemble of {N_STUDIES} fits per model)",
        show_ci=False, save_path=path,
    )
    return path


def check_complete(suffix: str | None = None) -> int:
    """Compare what the declaration owes against what is on disk. Exit code gates a run.

    Expands the SAME work list the runner uses, including the same `refuses()` gate, so
    a model that cannot run an arm owes nothing there and is reported `n/a` rather than
    `missing`. That fourth state is the point of this function: two consecutive grids
    finished with zero ValendinLSTM results and nothing said so, because "declared and
    empty" and "not owed" looked identical.

    `SHORT` is the dangerous state — some replications present, some not — because the
    suite looks trained and is not (F19).
    """
    scheduled, dropped = work_list()
    print(f"{'model':14s} {'panel':12s} {'arm':34s} {'studies':>9s}  state")
    print("-" * 84)

    states: list[str] = []
    for item in scheduled:
        root = STUDIES_BASE / suite_name(
            item.model_type, item.panel, item.arm.name, item.shard, suffix
        )
        preds = root / MODEL_NAMES[item.model_type] / "Predictions"
        n = len(list(preds.glob("Prediction_*.csv"))) if preds.is_dir() else 0
        state = "ok" if n >= N_STUDIES else ("ABSENT" if n == 0 else "SHORT")
        states.append(state)
        print(f"{MODEL_NAMES[item.model_type]:14s} {item.panel:12s} "
              f"{item.arm.name:34s} {n:4d}/{N_STUDIES:<4d} {state}")

    for item in dropped:
        print(f"{MODEL_NAMES[item.model_type]:14s} {item.panel:12s} "
              f"{item.arm.name:34s} {'—':>9s}  n/a — {item.skipped}")

    for panel in PANELS:
        root = STUDIES_BASE / pareto_suite_name(panel, suffix)
        ok = (root / "ParetoNBD" / "Predictions").is_dir()
        states.append("ok" if ok else "ABSENT")
        print(f"{'ParetoNBD':14s} {panel:12s} {'(no arm)':34s} {'1/1' if ok else '0/1':>9s}"
              f"  {'ok' if ok else 'ABSENT'}")

    short = states.count("SHORT")
    absent = states.count("ABSENT")
    print(f"\nowed {len(scheduled)} neural suites + {len(PANELS)} Pareto; "
          f"{short} SHORT, {absent} ABSENT, {len(dropped)} n/a")
    return 1 if (short or absent) else 0


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models",
                        choices=sorted(MODEL_NAMES),
                        help="repeatable; default is every neural model")
    parser.add_argument("--panel", choices=sorted(PANELS),
                        help="restrict to one panel; default is both")
    parser.add_argument("--arm", help="restrict to one arm (validated per panel)")
    parser.add_argument("--seed-shard", choices=sorted(SHARDS), default="a",
                        help="which 20-seed block to run (default a: seeds 43-62)")
    parser.add_argument("--shard", default="1/1", metavar="I/N",
                        help="this worker's slice of the work list, arm-major strided")
    parser.add_argument("--n-studies", type=int, default=N_STUDIES)
    parser.add_argument("--n-trials", type=int,
                        help="override the per-model default (probe runs only)")
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--suite-suffix",
                        help="appended to every suite name — required when overriding "
                             "the trial or simulation budget, so a probe cannot be "
                             "collected as if it were the real run")
    parser.add_argument("--preflight", action="store_true",
                        help="build every arm and train every eligible cell tiny")
    parser.add_argument("--preflight-config-only", action="store_true",
                        help="preflight without the tiny training pass")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--check-complete", action="store_true")
    parser.add_argument("--plot", action="store_true",
                        help="write the weekly-aggregate tracking figure")
    args = parser.parse_args()

    if args.preflight or args.preflight_config_only:
        sys.exit(preflight(train=not args.preflight_config_only))
    if args.check_complete:
        sys.exit(check_complete(args.suite_suffix))
    if args.plot:
        for panel in ([args.panel] if args.panel else sorted(PANELS)):
            print(f"wrote {plot_tracking(panel, args.suite_suffix)}")
        return
    if args.report:
        for panel in ([args.panel] if args.panel else sorted(PANELS)):
            report(panel, args.suite_suffix)
        return

    # A probe trains at a budget the declaration does not carry, so it must not be
    # collectable: force it into its own suite root.
    if (args.n_trials is not None or args.n_simulations != N_SIMULATIONS
            or args.n_studies != N_STUDIES) and not args.suite_suffix:
        parser.error(
            "--n-trials / --n-simulations / --n-studies override the declared budget; "
            "pass --suite-suffix so the result cannot be mistaken for the real run"
        )

    models = tuple(args.models) if args.models else NEURAL
    panels = (args.panel,) if args.panel else tuple(PANELS)

    # Pareto/NBD is not on the arm axis and runs on the orchestrator, so it takes its
    # own path out of here rather than an entry in the work list.
    if "pareto_nbd" in models:
        for panel in panels:
            root = run_pareto(panel, args.n_simulations, args.suite_suffix)
            print(f"\nPareto/NBD written to: {root}")
        models = tuple(m for m in models if m != "pareto_nbd")
        if not models:
            return

    # One cache for both passes: `work_list` has to build every dataset to answer
    # `refuses()` honestly, and the run loop needs the same objects. Building them twice
    # would repeat `prepare_dataset` for every arm on the worker's own clock.
    cache: dict = {}
    scheduled, dropped = work_list(models, panels, (args.seed_shard,), cache)
    if args.arm:
        known = {a for p in panels for a in arms_for(p)}
        if args.arm not in known:
            parser.error(f"--arm {args.arm!r} is not defined for {panels}; "
                         f"choose from {sorted(known)}")
        scheduled = [i for i in scheduled if i.arm.name == args.arm]

    index, total = (int(x) for x in args.shard.split("/"))
    mine = scheduled[index - 1::total]

    print(f"work list: {len(scheduled)} suites ({len(dropped)} refused), "
          f"shard {index}/{total} -> {len(mine)} suites")
    for item in dropped:
        print(f"  refused: {MODEL_NAMES[item.model_type]} on {item.panel}/"
              f"{item.arm.name} — {item.skipped}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for n, item in enumerate(mine, start=1):
        name = suite_name(
            item.model_type, item.panel, item.arm.name, item.shard, args.suite_suffix
        )
        # results.csv is written last, so its presence means the suite finished. This is
        # what makes a killed shard resumable rather than restartable.
        if (STUDIES_BASE / name / "results.csv").exists():
            print(f"[{n}/{len(mine)}] skip (done): {name}")
            continue
        print(f"[{n}/{len(mine)}] {name}")
        key = (item.panel, item.arm.name)
        if key not in cache:
            cache[key] = build_data(item.panel, item.arm)
        run_item(item, cache[key], args.n_studies, args.n_trials,
                 args.n_simulations, args.suite_suffix, device)

    print(f"\nshard {index}/{total} complete: {len(mine)} suites")


if __name__ == "__main__":
    main()
