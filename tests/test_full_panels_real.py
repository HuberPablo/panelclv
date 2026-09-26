"""Acceptance checks on the real full-dataset panels.

These run against the raw sources in `Datasets/Datasets_full/` and against the panels
`scripts/build_full_panels.py` wrote to `Datasets/Dataset_full_clean/`. They pin:

- **Paper reproduction.** The electronics raw file yields the 3,782-household cohort of
  Ni, Neslin & Sun (2012) at trip level: 19,441 household-days.
- **Continuity with the old panels.** The new cohorts are close to the old ones, and
  where the old rule was looser they are a subset of it.
- **Every written panel is consumable.** Each CSV and its sidecar `PanelConfig` pass
  `prepare_dataset` with a 52-week holdout.

`Datasets/` is gitignored, so every test skips when its inputs are absent.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation.panel_dataset import prepare_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_full_panels.py"
OLD_CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"


def _load():
    spec = importlib.util.spec_from_file_location("build_full_panels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bfp = _load()

needs_raw = pytest.mark.skipif(
    not bfp.RAW.exists(), reason="Datasets/ is gitignored; raw sources not on this machine"
)


def _old_ids(name):
    path = OLD_CLEAN / f"{name}_customer_week_panel.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not on this machine")
    return set(pd.read_csv(path, usecols=["Id"])["Id"].astype(str).unique())


@needs_raw
def test_electronics_reproduces_the_paper_cohort():
    tx, _ = bfp.electronics_transactions(bfp.load_electronics())
    first = tx.groupby("Id")["Date"].min()
    cohort = first[(first >= "1998-12-01") & (first <= "1999-11-30")].index
    assert len(cohort) == 3782
    assert tx[tx["Id"].isin(cohort)].drop_duplicates(["Id", "Date"]).shape[0] == 19441


@needs_raw
def test_gift_cohort_is_a_subset_of_the_old_panel():
    """The old gift cohort is 'first retained order in Mar-May 2001'. The new one adds
    the acquisition-date check, so it can only shrink."""
    tx, statics = bfp.read("gift")
    cohort = {str(i) for i in bfp.select_cohort(tx, bfp.SPECS["gift"])}
    old = _old_ids("gift")
    assert cohort and cohort <= old
    assert len(cohort) < len(old)


@needs_raw
@pytest.mark.parametrize("name, old_n, tolerance", [
    ("multichannel", 1402, 0.05),
    ("books", 1218, 0.01),
])
def test_cohort_sizes_are_close_to_the_old_ones(name, old_n, tolerance):
    tx, _ = bfp.read(name)
    n = len(bfp.select_cohort(tx, bfp.SPECS[name]))
    assert abs(n - old_n) / old_n <= tolerance, n


def _built():
    """Every (dataset, calibration) the builder declares, with its output paths."""
    return [
        (name, cal, *bfp.output_paths(name, cal, bfp.OUT))
        for name, spec in bfp.SPECS.items()
        for cal in spec.calibrations
    ]


@pytest.mark.parametrize("name, cal, csv_path, json_path", _built())
def test_built_panel_feeds_prepare_dataset(name, cal, csv_path, json_path):
    if not csv_path.exists():
        pytest.skip(f"{csv_path.name} not built on this machine")
    panel = pd.read_csv(csv_path)
    config = PanelConfig.from_dict(json.loads(json_path.read_text()))
    assert not panel.isna().any().any()
    out = prepare_dataset(panel, config, verbose=False)
    assert out["T_HOLD"] == 52
    assert out["T_CAL"] == 52 * bfp.CAL_YEARS[cal]
    # The cohort is chosen at build time, so the calibration-activity filter drops
    # nobody.
    assert out["N"] == panel["Id"].nunique()
