"""Generate one grid's synthetic panels — ``--grid <name>`` selects which.

Writes ``N_DATASETS`` replicate panels per (rate, churn) cell under
``Datasets/Synthetic/<grid name>/``, together with the manifest (``index.csv``) and the
generation record (``study_config.json``) the grid readers join against.

Usage:
    python scripts/generate_pnbd_grid.py --grid seasonal_4x4x10
    python scripts/generate_pnbd_grid.py --list

The grid's parameters live in ``grids/<name>.py``, not here, so that several grids can
coexist and each stays a committed declaration of what it is. The directory name is the
grid's name — passed explicitly rather than derived, because the derived name keys only
on grid shape and seed and two differently-seasoned 4x4x10 grids at seed 42 would
otherwise overwrite each other.

Regeneration is deterministic: seeds are handed out ``base_seed, base_seed+1, ...`` in
generation order, so this reproduces the same panels on any machine (same NumPy). That
is what lets a rented worker rebuild the data instead of waiting on an upload — though
VastAI/Rules.md §3 makes rsync the default and this the fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `python scripts/foo.py` puts scripts/ on sys.path, not the repo root, so the
# top-level `grids` package would not be importable. Add the root explicitly rather
# than requiring the caller to set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panelclv.data_preparation import pareto_nbd_simulation as ps

from grids import available_grids, load_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", help="name of a module in grids/")
    parser.add_argument(
        "--list", action="store_true", help="list the declared grids and exit"
    )
    args = parser.parse_args()

    if args.list or not args.grid:
        print("declared grids:", ", ".join(available_grids()) or "(none)")
        if not args.list:
            parser.error("--grid is required")
        return

    spec = load_grid(args.grid)

    dataset_dir, manifest = ps.generate_pnbd_study(
        spec.mean_transaction_rates,
        spec.churn_rates,
        n_customers=spec.n_customers,
        n_weeks=spec.n_weeks,
        n_datasets=spec.n_datasets,
        out_path=spec.dataset_dir.parent,
        r=spec.r,
        s=spec.s,
        n_weeks_for_churn_rate=spec.n_weeks_for_churn_rate,
        seasonal_peaks=spec.seasonal_peaks,
        seasonal_amplitude=spec.seasonal_amplitude,
        seasonal_width=spec.seasonal_width,
        base_seed=spec.base_seed,
        start_year=spec.start_year,
        dataset_dir_name=spec.name,
    )

    # The counts worth seeing before committing hours of training to them.
    print(f"grid               : {spec.name}")
    print(f"study saved to     : {dataset_dir}")
    print(f"datasets generated : {len(manifest)}  ({spec.n_cells} cells x {spec.n_datasets} replicates)")
    print(f"panel shape        : {spec.n_customers} customers x {spec.n_weeks} weeks")


if __name__ == "__main__":
    main()
