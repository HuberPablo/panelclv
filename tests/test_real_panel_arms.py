"""The real-panel arm axis is declared once, and its work list matches its own rules.

`scripts/run_real_panel_arms.py` schedules four models over two panels and fourteen
arms. Two of those cells are load-bearing and silent when wrong:

- **A model that cannot run an arm must be absent by declaration, not by crashing.**
  ValendinLSTM refuses any non-embedded `seq_col` (ADR-0004), and that has already cost
  two grids: both declared it, both produced zero results, and nothing said so because
  "declared and empty" looks exactly like "not owed". `refuses()` is the one predicate
  that answers it, and these tests assert the work list agrees with it exactly.
- **A bounded activity flag at or above the calibration length is a copy of
  `has_transacted_before`** that only diverges out in the holdout, where nothing
  constrained it. `check_arm_depth` refuses it; this asserts it actually fires.

Everything here is static or dataset-only — nothing trains — so it runs in seconds and
belongs in CI. The tests that need a built dataset skip when the panels are absent,
because `Datasets/` is gitignored.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_real_panel_arms.py"

pytest.importorskip("torch")


def _load():
    """Import the script by path — `scripts/` is deliberately outside the wheel.

    Registered in `sys.modules` before executing: `@dataclass` resolves annotations by
    looking its own class's module up there, so a module that defines one cannot be
    executed detached.
    """
    spec = importlib.util.spec_from_file_location("run_real_panel_arms", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rpa = _load()

# The panels are gitignored, so every dataset-building test is conditional on them.
PANELS_PRESENT = all(path.exists() for path, _ in rpa.PANELS.values())
needs_panels = pytest.mark.skipif(
    not PANELS_PRESENT, reason="Datasets/ is gitignored; panels not on this machine"
)


# ---------------------------------------------------------------------------
# Declaration — no dataset needed
# ---------------------------------------------------------------------------


def test_arm_counts_per_panel():
    """Four shared feature cells, plus each panel's own calendar-encoding arms.

    Electronics engineers sin/cos by default, so its extra pair REMOVES the calendar
    (`-no_tf`) -- the only shape ValendinLSTM can read there (F11). CDNOW engineers
    nothing by default, so its extras ADD one: `-tf` on all four cells, and `-week_emb`
    on the two `no_ar` cells, the latter being the only encoding the benchmark can read.
    The suffixes are panel-specific because the defaults are opposite; a `-no_tf` arm on
    CDNOW would duplicate its plain arm exactly.
    """
    cdnow, electronics = rpa.arms_for("cdnow"), rpa.arms_for("electronics")
    assert len(cdnow) == 10
    assert len(electronics) == 6
    assert not [a for a in cdnow if a.endswith("-no_tf")]
    assert len([a for a in electronics if a.endswith("-no_tf")]) == 2
    assert len([a for a in cdnow if a.endswith("-tf")]) == 4
    assert len([a for a in cdnow if a.endswith("-week_emb")]) == 2
    assert not [a for a in electronics if a.endswith("-tf") or a.endswith("-week_emb")]


def test_calendar_encodings_are_what_they_claim():
    """Each CDNOW calendar arm builds the config its name promises -- and only that.

    Asserted on the built `PanelConfig` rather than on the arm, because the arm carries
    a label and the config carries the consequence. The `-tf` arms must NOT set
    `add_year_idx`: CDNOW calibrates inside 1997 and forecasts into 1998, so a year
    index is constant while fitting and out of range in the holdout.
    """
    arms = rpa.arms_for("cdnow")
    plain = rpa.cdnow_config(arms["no_ar-no_cluster-valendin"])
    assert not plain.time_features

    tf = rpa.cdnow_config(arms["no_ar-no_cluster-valendin-tf"])
    assert tf.time_features == {"add_week_sin_cos": True}
    assert "add_year_idx" not in tf.time_features

    emb = rpa.cdnow_config(arms["no_ar-no_cluster-valendin-week_emb"])
    assert not emb.time_features
    assert "week" in dict(emb.embedded_cols)
    assert "week" in tuple(emb.time)


def test_unknown_calendar_encoding_raises():
    """An arm asking for an encoding its panel cannot express is a declaration bug.

    Silently falling back to the panel default would train the wrong thing under the
    right name, which is the failure the whole eligibility apparatus exists to stop.
    """
    bogus = rpa.Arm(name="bogus", calendar="week_emb")
    with pytest.raises(ValueError, match="no calendar encoding"):
        rpa.electronics_config(bogus)


def test_bounded_depth_is_below_the_calibration_window():
    """A flag at or above T_CAL cannot vary while fitting. 104 and 39 periods."""
    assert rpa.BOUNDED_DEPTH["electronics"] < 104
    assert rpa.BOUNDED_DEPTH["cdnow"] < 39
    assert max(int(f.split("_")[3]) for f in rpa.bounded_flags(16)
               if f.startswith("active_in_last_")) == 16


def test_shard_seed_gap_covers_a_full_shard():
    """Study i draws base_seed + i, so the gap must be >= N_STUDIES.

    At a smaller gap the two shards silently overlap and half the replications become
    duplicates — the same numbers reported as independent evidence.
    """
    seeds = sorted(rpa.SHARDS.values())
    assert min(b - a for a, b in zip(seeds, seeds[1:])) >= rpa.N_STUDIES


def test_the_two_searched_models_get_equal_trial_budgets():
    """An unequal budget makes a difference attributable to search effort, not the arm.

    ValendinLSTM is deliberately lower: ADR-0004 freezes its architecture, so its
    registry entry searches three training hyperparameters and nothing else.
    """
    assert rpa.N_TRIALS["lstm"] == rpa.N_TRIALS["transformer"]
    assert rpa.N_TRIALS["valendin_lstm"] < rpa.N_TRIALS["lstm"]


def test_the_frozen_benchmark_declares_no_search_space():
    """Naming a width for ValendinLSTM would quietly unfreeze the published model."""
    assert rpa.SEARCH_SPACES["valendin_lstm"] == {}


def test_no_search_space_names_the_embedder():
    """`run_pnbd_grid.py` pins `embedder` for every neural model; that would raise here.

    `valendin_lstm`'s registry entry declares no `embedder` key, so `suggest_param`
    rejects it. The bug has never fired only because ValendinLSTM never ran — this
    asserts the new runner does not inherit it.
    """
    assert not [m for m, s in rpa.SEARCH_SPACES.items() if "embedder" in s]


# ---------------------------------------------------------------------------
# The work list — needs the built datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built():
    """`(scheduled, dropped, cache)` for the full declaration. Built once."""
    cache = {}
    scheduled, dropped = work = rpa.work_list(data_cache=cache)
    return (*work, cache)


@needs_panels
def test_work_list_matches_the_eligibility_predicate(built):
    """The list and the predicate cannot disagree — that disagreement is the bug.

    Asserted in both directions: nothing scheduled is refused, and everything dropped
    still refuses. A one-directional check would pass a work list that silently omitted
    a cell nothing refuses.
    """
    scheduled, dropped, cache = built
    for item in scheduled:
        assert rpa.refuses(item.model_type, cache[(item.panel, item.arm.name)]) is None
    for item in dropped:
        assert rpa.refuses(item.model_type, cache[(item.panel, item.arm.name)])


@needs_panels
def test_the_benchmark_is_scheduled_on_both_panels(built):
    """The condition that failed in two consecutive grids.

    ValendinLSTM is eligible only where the built dataset has no non-embedded channel —
    which is `no_ar` and no engineered time features. That is two arms on CDNOW and the
    two `-no_tf` arms on electronics.
    """
    scheduled, _, _ = built
    panels = {i.panel for i in scheduled if i.model_type == "valendin_lstm"}
    assert panels == {"cdnow", "electronics"}, (
        f"the frozen benchmark must appear on both panels; got {panels}"
    )


@needs_panels
def test_scheduled_counts(built):
    """16 LSTM, 16 Transformer, 6 ValendinLSTM — the run's declared size."""
    scheduled, _, _ = built
    counts = {}
    for item in scheduled:
        counts[item.model_type] = counts.get(item.model_type, 0) + 1
    assert counts == {"lstm": 16, "transformer": 16, "valendin_lstm": 6}


@needs_panels
def test_suite_roots_are_unique(built):
    """Two workers sharing a root would collide: `create_suite_root` refuses one that
    exists, and disjointness is what makes collection a plain rsync."""
    scheduled, _, _ = built
    names = [
        rpa.suite_name(i.model_type, i.panel, i.arm.name, i.shard) for i in scheduled
    ]
    assert len(names) == len(set(names))


@needs_panels
def test_arm_depth_check_fires(built):
    """`check_arm_depth` must reject a flag that cannot vary, not merely document it."""
    _, _, cache = built
    data = dict(cache[("cdnow", f"no_ar-no_cluster-{rpa.EMBEDDER}")])
    data["seq_cols"] = list(data["seq_cols"]) + ["active_in_last_52_periods"]
    with pytest.raises(ValueError, match="cannot vary"):
        rpa.check_arm_depth("synthetic", data)
