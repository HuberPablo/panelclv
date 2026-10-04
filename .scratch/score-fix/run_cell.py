"""Rerun electronics · archive · no cluster label with the per-cell validation score.

The archived cell (`factorial__<model>__electronics__archive-no_cluster__rNN`, run
2026-09-21) scored validation as a mean of batch means, which read 4.1% low at batch 256
on electronics. Since then `loop.py` weights each batch by its cells (af2b14b). This
reruns the same cell with that as the only change:

- same models (LSTM, ValendinLSTM), 20 replications, seeds BASE_SEED + rep, 100 trials,
  patience 7, 100 epochs, the same pruner, 200 paths, the same refit;
- the archive's search space restored: weight decay searched over (1e-6, 1e-2) log and
  batch over {64, 128, 256}, which the registry has since changed (d7e902d, 57b8c18).

Writes `Studies/scorefix__<model>__electronics__archive-no_cluster__rNN`, never the
archived suites. Resumable: a suite with a forecast is skipped.

    python .scratch/score-fix/run_cell.py            # both models, 20 replications
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_factorial as fa                                   # noqa: E402
from panelclv.studies import StudySuiteConfig, run_study_suite   # noqa: E402

fa.EXPERIMENT = "scorefix"           # suite_name / forecast_path read it at call time
ARCHIVE_SPACE = {"weight_decay": (1e-6, 1e-2, "log"), "batch_size": {64, 128, 256}}
PANEL, ARM, CLUSTER = "electronics", "archive", "no_cluster"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model in fa.MODELS:
        data = fa.build_data(PANEL, model, CLUSTER)
        for rep in range(fa.N_REPLICATIONS):
            name = fa.suite_name(PANEL, model, ARM, CLUSTER, rep)
            if fa.forecast_path(PANEL, model, ARM, CLUSTER, rep).exists():
                print(f"{name}: done, skipping", flush=True)
                continue
            spec = fa.model_spec(model, ARM)
            spec.search_space = {**spec.search_space, **ARCHIVE_SPACE}
            print(f"{name}: training", flush=True)
            run_study_suite(StudySuiteConfig(
                studies_base_path=str(fa.STUDIES_BASE), suite_name=name,
                n_studies_per_model=1, n_simulations=fa.N_SIMULATIONS, device=device,
                data=data, models=[spec], base_seed=fa.BASE_SEED + rep,
                overwrite=(fa.STUDIES_BASE / name).exists(),
                keep_only_best_checkpoint=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
