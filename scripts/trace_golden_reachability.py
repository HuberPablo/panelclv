"""Which symbols in `src/panelclv` a real run actually executes.

Evidence for the package-simplification audit (`.scratch/package-simplification/`,
ticket 01). "Is this function still used?" is normally answered by grepping for callers,
which finds references but cannot tell a live call path from a dead one that still
compiles. This script answers the complementary question by *running* the package and
recording what the interpreter enters.

Four scenarios are traced, one per model family the package supports, because tracing
only one would mark the other three's code unreached and make a correct implementation
look like dead code:

    lstm           the golden pipeline from tests/test_golden_end_to_end.py, verbatim
    transformer    the same panel through the Transformer family
    valendin_lstm  the frozen benchmark (ADR-0004)
    pareto_nbd     the frozen Pareto/NBD benchmark, on short MCMC chains

**Reached is proof of life. Unreached is not proof of death.** These scenarios are a
single small synthetic panel with no covariates, no Optuna search, no study suite and no
plotting, so large parts of `tuning/`, `studies/` and `evaluation/` are unreached simply
because nothing here calls them. Read the output as "this symbol is definitely live",
and treat the unreached list as a list of *candidates to investigate*, never a kill list.

Run from the repo root:
    /home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/python scripts/trace_golden_reachability.py

Writes `reachability.md` (summary per module) and `reachability.csv` (one row per symbol)
into `.scratch/package-simplification/`.
"""

from __future__ import annotations

import ast
import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "panelclv"
OUT_DIR = REPO_ROOT / ".scratch" / "package-simplification"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))


# --------------------------------------------------------------------------------------
# 1. Static inventory — every function and method defined under src/panelclv.
# --------------------------------------------------------------------------------------

def _first_line(node: ast.AST) -> int:
    """The line the interpreter reports as a code object's first line.

    For a decorated function that is the first decorator, not the `def`, so decorators
    are folded in here — otherwise every decorated function would fail to match its
    trace record and be reported unreached.
    """
    lines = [node.lineno]
    lines += [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(lines)


def static_inventory() -> dict[tuple[str, int], dict]:
    """Map (file, first line) -> symbol record, for every def in the package."""
    inventory: dict[tuple[str, int], dict] = {}

    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()

        def walk(node: ast.AST, prefix: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{prefix}{child.name}"
                    inventory[(str(path), _first_line(child))] = {
                        "module": rel,
                        "qualname": qualname,
                        "line": child.lineno,
                        # A leading underscore anywhere in the path marks a private
                        # helper; only public symbols are package surface.
                        "public": not any(p.startswith("_") for p in qualname.split(".")),
                    }
                    walk(child, prefix=f"{qualname}.")
                elif isinstance(child, ast.ClassDef):
                    walk(child, prefix=f"{prefix}{child.name}.")

        walk(tree)
    return inventory


# --------------------------------------------------------------------------------------
# 2. Dynamic trace — record every call into the package.
# --------------------------------------------------------------------------------------

class Tracer:
    """Collect (file, first line) for each package function the interpreter enters.

    `sys.settrace` only fires for Python frames, so C-level torch internals are invisible
    — which is what we want: the question is which *panelclv* code runs.
    """

    def __init__(self, root: Path) -> None:
        self.root = str(root)
        self.seen: set[tuple[str, int]] = set()

    def __call__(self, frame, event, arg):
        if event == "call":
            code = frame.f_code
            if code.co_filename.startswith(self.root):
                self.seen.add((code.co_filename, code.co_firstlineno))
        # Returning None declines line-level tracing: call events are all we need, and
        # per-line tracing would make the run an order of magnitude slower.
        return None

    def trace(self, fn, *args, **kwargs):
        sys.settrace(self)
        try:
            return fn(*args, **kwargs)
        finally:
            sys.settrace(None)


# --------------------------------------------------------------------------------------
# 3. The scenarios.
# --------------------------------------------------------------------------------------

def scenario_lstm(tmp: Path):
    """The golden pipeline, imported from the test so the two can never drift apart."""
    from test_golden_end_to_end import run_golden_pipeline

    return run_golden_pipeline(tmp)


def _prepared_panel(config=None):
    """The golden test's panel, prepared under `config` (its own by default)."""
    from test_golden_end_to_end import _golden_config, _golden_panel
    from panelclv.data_preparation import dynamic_panel_dataset

    return dynamic_panel_dataset.prepare_dataset(
        _golden_panel(), config or _golden_config(), verbose=False
    )


def _valendin_config():
    """The golden config stripped to what the published architecture can read.

    `ValendinEmbedder` has no covariate path — the paper's model consumes embedded
    features only — so the derived time features and autoregressive columns the other
    scenarios carry have to go. Constructing this config is itself the ADR-0004
    constraint made concrete.
    """
    from test_golden_end_to_end import _golden_config
    from dataclasses import replace

    return replace(_golden_config(), time_features=None, ar_features=())


def _train_and_roll(model, inference_model, data, tmp: Path, forecaster, n_classes: int):
    """Shared tail: fit two epochs, load the checkpoint, roll out, score."""
    import torch

    from panelclv.experiments import make_loaders
    from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
    from panelclv.training import fit_model

    train_loader, val_loader, metadata = make_loaders(data, batch_size=8)
    fit = fit_model(
        model, train_loader, val_loader, max_trans=n_classes, n_epochs=2, patience=2,
        device="cpu", checkpoint_dir=str(tmp), model_name="trace", verbose=False,
        val_score_start=metadata["val_score_start"],
    )
    inference_model.load_state_dict(torch.load(fit.checkpoint_path, map_location="cpu"))
    forecast = forecaster(inference_model, data, n_simulations=4, seed=7, device="cpu",
                          return_simulations=False)
    return compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])


def scenario_transformer(tmp: Path):
    import torch

    from panelclv.experiments import make_loaders
    from panelclv.models.embedders import ProjectedEmbedder
    from panelclv.models.monte_carlo_forecasting import run_monte_carlo_forecast_transformer
    from panelclv.models.multinomial_transformer import (
        InferenceMultinomialTransformerModel, MultinomialTransformerModel,
    )

    data = _prepared_panel()
    _, _, metadata = make_loaders(data, batch_size=8)
    n_classes = int(data["embedded_cols"]["Transactions"])

    def build(cls):
        torch.manual_seed(1234)
        embedder = ProjectedEmbedder(
            seq_cols=metadata["seq_cols"], embedded_cols=metadata["embedded_cols"],
            target_col=metadata["target_col"], embedding_dim=8,
        )
        return cls(embedder=embedder, seq_len=metadata["seq_len"], d_model=8, nhead=2,
                   num_encoder_layers=1, dropout=0.0)

    return _train_and_roll(build(MultinomialTransformerModel),
                           build(InferenceMultinomialTransformerModel),
                           data, tmp, run_monte_carlo_forecast_transformer, n_classes)


def scenario_valendin_lstm(tmp: Path):
    import torch

    from panelclv.benchmarks import InferenceValendinLSTMModel, ValendinLSTMModel
    from panelclv.experiments import make_loaders
    from panelclv.models.monte_carlo_forecasting import run_monte_carlo_forecast

    data = _prepared_panel(_valendin_config())
    _, _, metadata = make_loaders(data, batch_size=8)
    n_classes = int(data["embedded_cols"]["Transactions"])

    def build(cls):
        torch.manual_seed(1234)
        return cls(seq_cols=metadata["seq_cols"], embedded_cols=metadata["embedded_cols"],
                   target_col=metadata["target_col"])

    return _train_and_roll(build(ValendinLSTMModel), build(InferenceValendinLSTMModel),
                           data, tmp, run_monte_carlo_forecast, n_classes)


def scenario_pareto_nbd(tmp: Path):
    """Short chains: the trace records which code runs, not whether it has converged."""
    from panelclv.benchmarks import compute_pareto_predictions
    from test_golden_end_to_end import _golden_config, _golden_panel
    import pandas as pd

    config = _golden_config()
    panel = _golden_panel()
    # `period_start` is the calendar column the benchmark slices on; rebuild it the way
    # prepare_dataset does, from the ISO year/week pair.
    panel["period_start"] = pd.to_datetime(
        panel["year"].astype(str) + "-" + panel["week"].astype(str) + "-1", format="%G-%V-%u"
    )
    train = panel[panel["period_start"] <= pd.Timestamp(config.training_end)]
    return compute_pareto_predictions(
        train, holdout_length=25, id_col="Id", target_col="Transactions",
        period_in_days=7.0, mcmc=200, burnin=50, thin=10, chains=1, seed=42,
    )


SCENARIOS = {
    "lstm": scenario_lstm,
    "transformer": scenario_transformer,
    "valendin_lstm": scenario_valendin_lstm,
    "pareto_nbd": scenario_pareto_nbd,
}


# --------------------------------------------------------------------------------------
# 4. Run, match, report.
# --------------------------------------------------------------------------------------

def main() -> None:
    inventory = static_inventory()
    reached_by: dict[tuple[str, int], set[str]] = defaultdict(set)

    for name, scenario in SCENARIOS.items():
        tracer = Tracer(SRC)
        with tempfile.TemporaryDirectory() as tmp:
            print(f"tracing {name} ...", flush=True)
            tracer.trace(scenario, Path(tmp))
        for key in tracer.seen:
            if key in inventory:
                reached_by[key].add(name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, record in sorted(inventory.items(), key=lambda kv: (kv[1]["module"], kv[1]["line"])):
        scenarios = sorted(reached_by.get(key, ()))
        rows.append({
            "module": record["module"],
            "symbol": record["qualname"],
            "line": record["line"],
            "public": record["public"],
            "reached": bool(scenarios),
            "scenarios": " ".join(scenarios),
        })

    csv_path = OUT_DIR / "reachability.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_module: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per_module[row["module"]].append(row)

    lines = [
        "# Reachability trace",
        "",
        "Generated by `scripts/trace_golden_reachability.py`. Per-symbol detail is in",
        "`reachability.csv`.",
        "",
        "**Reached is proof of life; unreached is not proof of death.** The four traced",
        "scenarios are one small synthetic panel with no covariates, no Optuna search, no",
        "study suite and no plotting — so anything those layers own is unreached here for",
        "want of a caller, not for want of a purpose. Unreached means *investigate*.",
        "",
        f"Scenarios traced: {', '.join(SCENARIOS)}.",
        "",
        f"Reached {sum(r['reached'] for r in rows)} of {len(rows)} defined symbols "
        f"({sum(r['reached'] and r['public'] for r in rows)} of "
        f"{sum(r['public'] for r in rows)} public).",
        "",
        "| module | symbols | reached | public unreached |",
        "| --- | --- | --- | --- |",
    ]
    for module in sorted(per_module):
        module_rows = per_module[module]
        unreached_public = [r["symbol"] for r in module_rows if r["public"] and not r["reached"]]
        lines.append(
            f"| `{module}` | {len(module_rows)} | {sum(r['reached'] for r in module_rows)} | "
            f"{len(unreached_public)} |"
        )

    lines += ["", "## Public symbols no scenario reached", ""]
    for module in sorted(per_module):
        unreached_public = [r["symbol"] for r in per_module[module] if r["public"] and not r["reached"]]
        if unreached_public:
            lines.append(f"- `{module}` — {', '.join(sorted(unreached_public))}")

    md_path = OUT_DIR / "reachability.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {md_path.relative_to(REPO_ROOT)} and {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
