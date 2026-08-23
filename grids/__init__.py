"""One module per grid: what data it generates, what models train on it, how it splits.

A **grid** (CONTEXT.md's word — there is deliberately no term for "experiment") is a
set of study suites, one per synthetic dataset. Running several of them means several
declarations, so each lives in its own module here and is selected by name:

    python scripts/generate_pnbd_grid.py --grid seasonal_4x4x10
    python scripts/run_pnbd_grid.py      --grid seasonal_4x4x10 --model lstm --shard 1/4

Why Python modules and not YAML/JSON
------------------------------------
``ModelSpec.search_space`` distinguishes a categorical ``{64, 128, 256}`` from a range
``(1e-4, 1e-2, "log")`` by Python type. No data format carries that distinction without
a bespoke schema and a parser to match, and the registry already declares search spaces
as Python literals. A grid module is a declaration, not a program: constants and one
``GRID = GridSpec(...)``, nothing executable.

Why the name matters
--------------------
``GridSpec.name`` is the one string every path is derived from — the dataset directory
and each model's trained tree. It is passed to the generator explicitly rather than
letting it derive a name, because the derived name keys only on the grid's *shape* and
seed: two 4x4x10 grids at seed 42 that differ in seasonality or panel size would
otherwise land in the same folder and overwrite each other. Naming the grid is what
keeps several of them apart, while staying identical on every machine — which is what
the workers need.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from panelclv.configs.panel_config import PanelConfig
from panelclv.studies import ModelSpec

# Repo root = the parent of grids/. Every path below hangs off it, so the only
# machine-specific thing is where the repo was cloned.
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GridSpec:
    """One grid: the data to generate, the models to train on it, and how to split.

    The three halves are deliberately in one object. The generation parameters decide
    what the panels are, the ``panel`` config decides how they are read, and they have
    to agree — ``n_weeks`` and ``start_year`` fix the calendar that the window dates in
    ``panel`` must land inside. Keeping them apart is how a grid ends up trained on
    windows that do not match the data it generated.
    """

    # The grid's identity. Every path is derived from it, so it must be unique across
    # grids and stable across machines. Keep it filesystem-safe.
    name: str

    # --- generation: the axes ---------------------------------------------------
    mean_transaction_rates: Sequence[float]     # mean weekly purchases while active
    churn_rates: Sequence[float]                # fraction dropped out by the horizon
    n_weeks_for_churn_rate: float               # the horizon those churn rates refer to

    # --- generation: the panels -------------------------------------------------
    n_customers: int
    n_weeks: int
    n_datasets: int                             # replicate panels per (rate, churn) cell
    base_seed: int = 42
    start_year: int = 1999
    r: float = 2.0                              # Gamma shape, purchase prior
    s: float = 2.0                              # Gamma shape, dropout prior

    # --- generation: seasonality (fixed for the whole grid, never an axis) ------
    seasonal_peaks: Sequence[int] = ()
    seasonal_amplitude: float = 0.0
    seasonal_width: float = 1.0

    # --- training ---------------------------------------------------------------
    # How the generated panels are read: column roles, window dates, clipping.
    panel: PanelConfig | None = None
    # The models to train on every dataset, with their search spaces and trial counts.
    models: tuple[ModelSpec, ...] = ()
    # Workers to rent per model *type*: how many vast.ai boxes that model's datasets
    # are split across. 0 means "not on vast" — run it on the orchestrator instead
    # (the right setting for pareto_nbd, which is one MCMC fit and needs no GPU).
    # See VastAI/Rules.md §5.
    workers: dict[str, int] = field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        """Where this grid's generated panels live."""
        return REPO_ROOT / "Datasets" / "Synthetic" / self.name

    def train_base(self, model_name: str) -> Path:
        """Where one model's trained suites live.

        Each model gets its own root because two workers training *different* models on
        the *same* dataset would otherwise both target
        ``<train_base>/<combo>__<dataset>/``, and ``create_suite_root`` refuses a folder
        that already exists. Splitting on model here is what lets the shards be merged
        by plain copy (VastAI/Rules.md §4).
        """
        return REPO_ROOT / "Studies" / f"{self.name}__{model_name}"

    @property
    def n_cells(self) -> int:
        return len(self.mean_transaction_rates) * len(self.churn_rates)

    @property
    def n_panels(self) -> int:
        return self.n_cells * self.n_datasets


def load_grid(name: str) -> GridSpec:
    """Return the ``GRID`` declared by ``grids/<name>.py``.

    Raises a message naming the available grids rather than a bare ImportError, because
    the usual cause is a typo in ``--grid`` and the fix is to see the list.
    """
    try:
        module = importlib.import_module(f"grids.{name}")
    except ModuleNotFoundError as exc:
        if exc.name not in (f"grids.{name}", name):
            raise                                  # a real missing import inside the grid
        raise SystemExit(
            f"no grid named {name!r}. Available: {', '.join(available_grids()) or '(none)'}"
        ) from exc
    grid = getattr(module, "GRID", None)
    if not isinstance(grid, GridSpec):
        raise SystemExit(f"grids/{name}.py must define GRID = GridSpec(...)")
    return grid


def available_grids() -> list[str]:
    """Every grid module in this directory, by name."""
    return sorted(
        p.stem for p in Path(__file__).parent.glob("*.py") if not p.stem.startswith("_")
    )
