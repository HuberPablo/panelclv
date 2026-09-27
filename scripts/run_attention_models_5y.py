"""The 5y electronics experiments of the LSTM family, repeated for two attention models.

`docs/benchmarks-real-panels.md` ("Pareto/NBD on the paper's electronics split" and
after) ran ValendinLSTM and our LSTM on Valendin et al.'s electronics split: a searched
family, then the least-biased searched study's settings pinned under three rules for
the kept epoch, each with no added input, `ar_bounded_52` or `kmeans_8`. This runner
repeats that procedure for

`Transformer`      `models.MultinomialTransformerModel` — attention instead of
                   recurrence; forecast through `forecast_attention`.
`LSTMAttention`    `models.MultinomialLSTMAttentionModel` — an LSTM whose head also
                   reads causal attention over its own past outputs; forecast through
                   `forecast_recurrent`.

Both take the `valendin` embedder, as every model on this split did.

Phases
------
`searched`  20 replications, no added input, a 100-trial search over the model's
            registry space (embedder pinned), patience 7, n_epochs 100.
`pinned`    the least-biased `searched` study's parameters pinned (written by `--pin`
            to `PINNED_FILE`), one trial, arms `nofloor` / `from20` / `from30`
            (`select_from_epoch` 0 / 20 / 30) x inputs none / `ar_bounded_52` /
            `kmeans_8` x 20 replications.

Each phase has its own work list, so a phase is launched only over its own suites. A
suite counts as finished when its `results.csv` exists, so a fresh worker is kept from
retraining finished suites by seeding it with those files alone (kilobytes).

Replication r seeds the search and the forecast from `BASE_SEED + r`, as in the LSTM
runs. Training is unseeded (CLAUDE.md priority 3).

Usage:
    python scripts/run_attention_models_5y.py --preflight
    python scripts/run_attention_models_5y.py --phase searched --model LSTMAttention --worker 3/8
    python scripts/run_attention_models_5y.py --pin                # after `searched`
    python scripts/run_attention_models_5y.py --phase pinned --model LSTMAttention --worker 3/8
    python scripts/run_attention_models_5y.py --check-complete --phase pinned
    python scripts/run_attention_models_5y.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation.panel_dataset import prepare_dataset
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_epoch_floor_5y as floor  # noqa: E402
import run_real_panel_benchmarks as benchmarks  # noqa: E402
from run_real_panel_arms import bounded_flags  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"
EXPERIMENT = "attention_cal5y"
PANEL, CAL = "electronics", "5y"
PINNED_FILE = Path(__file__).resolve().parent / "run_attention_models_5y_pinned.json"

N_REPLICATIONS = 20
N_TRIALS = 100
N_SIMULATIONS = benchmarks.N_SIMULATIONS
BASE_SEED = benchmarks.BASE_SEED

MODELS: dict[str, str] = {"Transformer": "transformer", "LSTMAttention": "lstm_attention"}
FEATURES: dict[str, dict[str, tuple[str, ...]]] = {
    "none":          {},
    "ar_bounded_52": {"ar_features": bounded_flags(52)},
    "kmeans_8":      {"cluster_features": ("kmeans_8",)},
}
ARMS = floor.ARMS                               # nofloor / from20 / from30
TRAINING = {"n_epochs": 100, "patience": 7, "verbose": False, "loss_type": "cross_entropy"}
# Training controls live in `training`, not in the pinned search space.
CONTROLS = {"n_epochs", "patience", "min_epochs", "select_from_epoch"}


# ---------------------------------------------------------------------------
# Work items: (model, feature, arm, replication); arm "searched" is phase `searched`
# ---------------------------------------------------------------------------


def suite_name(model: str, feature: str, arm: str, rep: int) -> str:
    """`attention_cal5y__<Model>__electronics__<feature>-<arm>__r<NN>`."""
    return f"{EXPERIMENT}__{model}__{PANEL}__{feature}-{arm}__r{rep:02d}"


def work_list(phase: str, models: list[str]) -> list[tuple[str, str, str, int]]:
    """This phase's items, REPLICATION-major so every worker draws from every cell."""
    if phase == "searched":
        return [(m, "none", "searched", r) for r in range(N_REPLICATIONS) for m in models]
    return [(m, f, a, r) for r in range(N_REPLICATIONS) for m in models
            for f in FEATURES for a in ARMS]


def suite_dir(item: tuple[str, str, str, int]) -> Path:
    return STUDIES_BASE / suite_name(*item)


def finished(item: tuple[str, str, str, int]) -> bool:
    return (suite_dir(item) / "results.csv").exists()


# ---------------------------------------------------------------------------
# Data and specs
# ---------------------------------------------------------------------------


def build_data(feature: str) -> dict:
    """The benchmark's 5y panel config (count and embedded week) plus one input."""
    config = PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        time=("week",),
        embedded_cols={"Transactions": "auto", "week": "auto"},
        **FEATURES[feature],
        **benchmarks.WINDOWS_5Y[PANEL],
    )
    data = prepare_dataset(pd.read_csv(benchmarks.panel_path(PANEL, CAL)), config,
                           verbose=False)
    data["panel_name"] = PANEL
    return data


def pinned_params(model: str) -> dict:
    if not PINNED_FILE.exists():
        raise FileNotFoundError(f"{PINNED_FILE.name} not found: run --pin after `searched`")
    return json.loads(PINNED_FILE.read_text())[model]["params"]


def model_spec(model: str, arm: str, training_override: dict | None = None) -> ModelSpec:
    if arm == "searched":
        return ModelSpec(name=model, model_type=MODELS[model], n_trials=N_TRIALS,
                         search_space={"embedder": "valendin"},
                         training={**TRAINING, **(training_override or {})})
    training = {**TRAINING, "select_from_epoch": ARMS[arm], **(training_override or {})}
    return ModelSpec(name=model, model_type=MODELS[model], n_trials=1,
                     search_space=pinned_params(model), training=training)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_worker(phase: str, models: list[str], index: int, total: int) -> int:
    mine = work_list(phase, models)[index - 1::total]
    print(f"worker {index}/{total} ({phase}): {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    for n, item in enumerate(mine, start=1):
        model, feature, arm, rep = item
        name = suite_name(*item)
        if finished(item):
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if feature not in cache:
            cache[feature] = build_data(feature)
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=N_SIMULATIONS,
            device=device,
            data=cache[feature],
            models=[model_spec(model, arm)],
            base_seed=BASE_SEED + rep,
            overwrite=suite_dir(item).exists(),
            keep_only_best_checkpoint=True,
        ))
    return 0


def pin(models: list[str]) -> int:
    """Write each model's least-biased `searched` study's parameters to PINNED_FILE.

    A model is pinned only once all its `searched` suites are finished: pinning from
    a partial phase would pick from fewer candidates than the procedure declares.
    """
    out = {}
    for model in models:
        missing = [i for i in work_list("searched", [model]) if not finished(i)]
        if missing:
            print(f"{model}: {len(missing)} searched suites unfinished, not pinned")
            continue
        rows = []
        for item in work_list("searched", [model]):
            if finished(item):
                res = pd.read_csv(suite_dir(item) / "results.csv").iloc[0]
                best = json.loads(next((suite_dir(item) / model / "Optuna_Studies").glob(
                    "study_*/study_*_best.json")).read_text())
                rows.append((abs(res.bias_percent), res.bias_percent, item[3],
                             best["best_params"]))
        _, bias, rep, params = min(rows, key=lambda r: r[0])
        out[model] = {"from_replication": rep, "bias_percent": float(bias),
                      "n_searched": len(rows),
                      "params": {k: v for k, v in params.items() if k not in CONTROLS}}
        print(f"{model}: r{rep:02d} (bias {bias:+.2f}%) of {len(rows)} -> {out[model]['params']}")
    if PINNED_FILE.exists():
        out = {**json.loads(PINNED_FILE.read_text()), **out}
    PINNED_FILE.write_text(json.dumps(out, indent=2) + "\n")
    return 0


def preflight(models: list[str]) -> int:
    """Build every panel and train every model tiny in both phases. Costs nothing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiny = {"n_epochs": 2, "patience": 1}
    with tempfile.TemporaryDirectory() as tmp:
        for feature in FEATURES:
            data = build_data(feature)
            print(f"{feature}: seq_cols={data['seq_cols']}")
            for model in models:
                spec = ModelSpec(name=model, model_type=MODELS[model], n_trials=1,
                                 search_space={"embedder": "valendin"},
                                 training={**TRAINING, **tiny, "select_from_epoch": 2})
                run_study_suite(StudySuiteConfig(
                    studies_base_path=tmp, suite_name=f"pf__{model}__{feature}",
                    n_studies_per_model=1, n_simulations=2, device=device, data=data,
                    models=[spec], base_seed=BASE_SEED, keep_only_best_checkpoint=True,
                ))
                print(f"  {model} ok")
    print("\nEvery model trains and forecasts on every input. Safe to launch.")
    return 0


def check_complete(phase: str, models: list[str]) -> int:
    items = work_list(phase, models)
    missing = [suite_name(*i) for i in items if not finished(i)]
    print(f"{phase}: {len(items) - len(missing)}/{len(items)} finished")
    for name in missing:
        print(f"  MISSING {name}")
    return 1 if missing else 0


def report(models: list[str]) -> None:
    """One row per (model, input, arm): the distribution of each metric over studies."""
    base = floor.build_data()
    actual = holdout_actuals(base)                               # (N, T_HOLD)
    ref_ids = np.asarray(base["ids"])
    print("| model | input | arm | n | RMSE (customer total) | bias % | MAPE | Spearman |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model in models:
        cells = [("none", "searched")] + [(f, a) for a in ARMS for f in FEATURES]
        for feature, arm in cells:
            scores = [benchmarks.score(suite_dir((model, feature, arm, r)) / model,
                                       actual, ref_ids)
                      for r in range(N_REPLICATIONS)
                      if finished((model, feature, arm, r))]
            if not scores:
                continue
            df = pd.DataFrame(scores)
            fmt = lambda m, dp: f"{df[m].mean():.{dp}f} ± {df[m].std(ddof=1):.{dp}f}"  # noqa: E731
            print(f"| {model} | {feature} | {arm} | {len(df)} | {fmt('rmse_customer_total', 3)} "
                  f"| {fmt('bias_percent', 1)} | {fmt('mape_aggregate', 1)} "
                  f"| {df['spearman'].mean():.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["searched", "pinned"], default="searched")
    parser.add_argument("--model", choices=sorted(MODELS), action="append",
                        help="restrict to this model (repeatable); default both")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N")
    mode.add_argument("--pin", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    models = args.model or list(MODELS)
    if args.worker:
        index, total = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(args.phase, models, index, total))
    if args.pin:
        sys.exit(pin(models))
    if args.preflight:
        sys.exit(preflight(models))
    if args.check_complete:
        sys.exit(check_complete(args.phase, models))
    report(models)


if __name__ == "__main__":
    main()
