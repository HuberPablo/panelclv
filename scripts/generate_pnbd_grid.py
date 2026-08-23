"""Entry point for the Pareto/NBD grid study's data — edit the constants, run.

Sweeps ``mean_transaction_rate`` x ``churn_rate`` and writes ``N_DATASETS``
replicate panels per cell under ``Datasets/Synthetic/<derived name>/``, together
with the manifest (``index.csv``) and the generation record (``study_config.json``)
that the grid readers join against.

Usage:
    python scripts/generate_pnbd_grid.py

Why this is a committed script and not a notebook cell
------------------------------------------------------
The grid is a pure function of the constants below plus ``BASE_SEED``: seeds are
handed out ``BASE_SEED, BASE_SEED+1, ...`` in generation order, and the directory
name is derived from the grid shape and the seed alone (no wall clock). So running
this file on a rented box reproduces the *same* grid, byte for byte, as the copy on
the workstation — which is what lets the training shards regenerate their own data
instead of waiting on a 340 MB upload. Editing a constant here therefore changes
what every machine generates; that is the point, and it is why the parameters live
in version control rather than in a notebook's execution history.

Output lands under the repo root, resolved from this file's location rather than
from the working directory, so the script runs unchanged wherever it is cloned.
"""

from __future__ import annotations

from pathlib import Path

from panelclv.data_preparation import pareto_nbd_simulation as ps

# --- EDIT ME: grid axes ---------------------------------------------------------
# The two human-facing axes. The generator converts each to a Pareto/NBD parameter
# internally (alpha = R / rate; beta from the inverse Lomax survival at
# N_WEEKS_FOR_CHURN_RATE), so the folder labels always match the values below.
MEAN_TRANSACTION_RATES = [0.01, 0.05, 0.10, 0.30]   # mean weekly purchases while active
CHURN_RATES = [0.20, 0.40, 0.60, 0.80]              # fraction dropped out by the horizon
N_WEEKS_FOR_CHURN_RATE = 52                         # horizon (weeks) the churn rates refer to

# Shared Gamma shapes for the purchase / dropout priors.
R = 2.0
S = 2.0

# --- EDIT ME: seasonality (fixed for the whole study, not a grid axis) -----------
# High-season weeks-of-year that recur every year, in both calibration and holdout:
# week 12 (spring), 25 and 30 (summer), 47 (year-end).
SEASONAL_PEAKS = [12, 25, 30, 47]
SEASONAL_AMPLITUDE = 1.5   # relative peak strength (0 = seasonality off)
SEASONAL_WIDTH = 3         # weeks each peak spans (0 = single-week spike)

# --- EDIT ME: panel size and replication ----------------------------------------
N_CUSTOMERS = 1000
N_WEEKS = 156              # 3 years at weekly frequency
N_DATASETS = 10            # replicate panels per (rate, churn) cell
BASE_SEED = 42

# Repo root = the parent of scripts/, matching run_studies.py's convention.
OUT_PATH = Path(__file__).resolve().parents[1] / "Datasets" / "Synthetic"


def main() -> None:
    dataset_dir, manifest = ps.generate_pnbd_study(
        MEAN_TRANSACTION_RATES,
        CHURN_RATES,
        n_customers=N_CUSTOMERS,
        n_weeks=N_WEEKS,
        n_datasets=N_DATASETS,
        out_path=OUT_PATH,
        r=R,
        s=S,
        n_weeks_for_churn_rate=N_WEEKS_FOR_CHURN_RATE,
        seasonal_peaks=SEASONAL_PEAKS,
        seasonal_amplitude=SEASONAL_AMPLITUDE,
        seasonal_width=SEASONAL_WIDTH,
        base_seed=BASE_SEED,
        # dataset_dir_name is deliberately left to the default: it is derived from
        # the grid shape and BASE_SEED, so every machine lands in the same folder.
    )

    # The counts a caller wants to see before committing hours of training to them:
    # how many panels exist, and over how many cells they are spread.
    n_cells = len(MEAN_TRANSACTION_RATES) * len(CHURN_RATES)
    print(f"study saved to     : {dataset_dir}")
    print(f"datasets generated : {len(manifest)}  ({n_cells} cells x {N_DATASETS} replicates)")
    print(f"panel shape        : {N_CUSTOMERS} customers x {N_WEEKS} weeks")


if __name__ == "__main__":
    main()
