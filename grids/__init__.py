"""One module per grid: what data it generates, what models train on it, how it splits.

A **grid** (CONTEXT.md's word — there is deliberately no term for "experiment") is a
set of study suites, one per synthetic dataset. Running several of them means several
declarations, so each lives in its own module here and is selected by name:

    python scripts/generate_pnbd_grid.py --grid seasonal_4x4x10
    python scripts/run_pnbd_grid.py      --grid seasonal_4x4x10 --model lstm --shard 1/4

A grid may also declare **arms** (``Arm``): feature and embedding configurations that
every model is trained under, crossed with the panels. A shard then spans the whole
(arm x dataset) product, and each pair gets its own output tree.

Why Python modules and not YAML/JSON
------------------------------------
``ModelSpec.search_space`` distinguishes a categorical ``{64, 128, 256}`` from a range
``(1e-4, 1e-2, "log")`` by Python type. No data format carries that distinction without
a bespoke schema and a parser to match, and the registry already declares search spaces
as Python literals. A grid module is a declaration, not a program: constants, one
``GRID = GridSpec(...)``, and no control flow that a reader has to execute in their head
to know what the grid is.

A **cross product is still a declaration**, and ``seasonal_4x4x10`` builds its twelve
arms as one. The three axes are named tables directly above it, so the comprehension
adds no information a reader has to derive — it only spares them checking twelve
hand-copied rows for the cell that went missing. The line to hold is that a grid module
never *computes* what it declares: axes yes, a loop that reads a file or picks a value
no.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from panelclv.configs.panel_config import PanelConfig
from panelclv.studies import ModelSpec

# Repo root = the parent of grids/. Every path below hangs off it, so the only
# machine-specific thing is where the repo was cloned.
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Arm:
    """One configuration of the feature and embedding axes, held fixed across the grid.

    An arm is *not* a model and *not* a dataset: it is a third axis crossed with both.
    The same 160 panels are trained by the same model types under each arm, so a
    difference between two arms is attributable to the axes below and nothing else.

    Two of the three fields change what the panel carries and one changes how the model
    reads it, which is why they are declared together here rather than split across
    ``PanelConfig`` and ``ModelSpec``: an arm is the unit a result is reported against,
    so it has to be one object.

    ``ar_features`` and ``cluster_features`` are ``PanelConfig`` fields, so each arm
    needs its own ``prepare_dataset`` (``GridSpec.panel_for``). ``embedder`` is a
    search-space key both neural entries declare, so it is a *pin* rather than new code
    — ``run_pnbd_grid.py`` merges it over the model's declared space.
    """

    name: str
    ar_features: tuple[str, ...] = ()
    cluster_features: tuple[str, ...] = ()
    # Which strategy turns features into a vector (ADR-0005). The registry default is
    # "valendin", which is what the archived seasonal_4x4x10 run used, since that grid
    # never overrode it.
    embedder: str = "valendin"


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
    # Simulated paths per forecast. A rollout samples a count and feeds it back, so a
    # forecast is the average over this many paths; more paths means less Monte Carlo
    # noise in the metrics and proportionally more time, paid once per dataset.
    n_simulations: int = 200
    # The models to train on every dataset, with their search spaces and trial counts.
    models: tuple[ModelSpec, ...] = ()
    # The feature/embedding configurations every model is trained under. Empty means
    # the grid has no arm axis: models train on `panel` as declared, and every path
    # keeps the un-suffixed shape the archived suites are stored in.
    arms: tuple[Arm, ...] = ()
    # Workers to rent per model *type*: how many vast.ai boxes that model's datasets
    # are split across. 0 means "not on vast" — run it on the orchestrator instead
    # (the right setting for pareto_nbd, which is one MCMC fit and needs no GPU).
    # See VastAI/Rules.md §5.
    workers: dict[str, int] = field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        """Where this grid's generated panels live."""
        return REPO_ROOT / "Datasets" / "Synthetic" / self.name

    def train_base(self, model_name: str, arm_name: str | None = None) -> Path:
        """Where one model's trained suites live, under one arm.

        Each (model, arm) pair gets its own root because two workers training a
        *different* model or a *different* arm on the *same* dataset would otherwise
        both target ``<train_base>/<combo>__<dataset>/``, and ``create_suite_root``
        refuses a folder that already exists. Splitting the tree here is what lets the
        shards be merged by plain copy (VastAI/Rules.md §4).

        ``arm_name=None`` keeps the un-suffixed ``<grid>__<Model>/`` path the archived
        seasonal_4x4x10 suites are stored under. A grid with no arms must not silently
        move its own history, and the reader in `evaluation` joins on that path.
        """
        leaf = f"{self.name}__{model_name}"
        if arm_name is not None:
            leaf = f"{leaf}__{arm_name}"
        return REPO_ROOT / "Studies" / leaf

    def arm(self, name: str) -> Arm:
        """The arm declared under this name, or a message listing the declared ones."""
        for a in self.arms:
            if a.name == name:
                return a
        raise SystemExit(
            f"grid {self.name!r} declares no arm {name!r}. "
            f"Available: {', '.join(a.name for a in self.arms) or '(none)'}"
        )

    def panel_for(self, arm: Arm | None) -> PanelConfig:
        """This grid's ``PanelConfig`` as one arm reads it.

        Only the two target-derived axes move. Everything else — window dates, clipping,
        cohort rule, time features — is the grid's, so two arms differ in what the panel
        carries and in nothing else. That is what makes a difference between them
        attributable.
        """
        if arm is None:
            return self.panel
        return replace(
            self.panel,
            ar_features=arm.ar_features,
            cluster_features=arm.cluster_features,
        )

    @property
    def n_cells(self) -> int:
        return len(self.mean_transaction_rates) * len(self.churn_rates)

    @property
    def n_panels(self) -> int:
        return self.n_cells * self.n_datasets

    @property
    def n_suites_per_model(self) -> int:
        """Suites one model owes across the whole grid: every arm on every panel.

        Named for suites, not studies, because ``StudySuiteConfig.n_studies_per_model``
        already means something else — how many independent Optuna studies one model
        runs *inside* one suite, which a grid pins to 1. Two different quantities under
        one name is how a budget gets miscounted by the arm factor.
        """
        return self.n_panels * max(len(self.arms), 1)


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
