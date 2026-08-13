"""Which symbols in `src/panelclv` a real run actually executes.

Evidence for the package-simplification audit (`.scratch/package-simplification/`,
ticket 01). "Is this function still used?" is normally answered by grepping for callers,
which finds references but cannot tell a live call path from a dead one that still
compiles. This script answers the complementary question by *running* the package and
recording what the interpreter enters.

Four scenarios are traced, one per model family the package supports, because tracing
only one would mark the other three's code unreached and make a correct implementation
look like dead code:

    lstm           the golden recurrent pipeline
    transformer    the same panel through the Transformer family
    valendin_lstm  the frozen benchmark (ADR-0004)
    pareto_nbd     the frozen Pareto/NBD benchmark, on short MCMC chains

All four are imported verbatim from `tests/test_golden_end_to_end.py`, which pins their
outcomes — so what this script traces is exactly what the suite asserts. That import also
pulls in the package before tracing starts, so anything that runs at *import* time is
outside the trace.

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
# 3. The scenarios — owned by the test, not by this script.
# --------------------------------------------------------------------------------------

# `tests/test_golden_end_to_end.py` defines all four and asserts on all four. Importing
# them here rather than re-implementing them is what makes the evidence worth anything:
# the trace records the same code path the test pins, and the two cannot drift apart.
from test_golden_end_to_end import SCENARIOS  # noqa: E402


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
