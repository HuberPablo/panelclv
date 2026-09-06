#!/usr/bin/env python3
"""Compare what a grid *declared* against what is actually on disk, and fail if short.

A distributed run has no single step that asks this question. A worker's shard exits 0
when its own loop ends; ``VastAI/supervise/reap_finished.sh`` checks that every
``results.csv`` the worker holds is also local, which is a real gate but whose
denominator is the worker's own disk; and ``watch_fleet.sh`` prints a fleet-wide count
against a denominator that omits the models running on the orchestrator. None of them
knows how many suites the grid *owes*. So a run can empty its fleet, report success
everywhere, and still be missing suites — which is exactly what happened to
``seasonal_4x4x10`` (F19), and the shortfall was found days later while reading results.

This is that missing step. It expands the grid's declaration into the set of suites it
owes, counts the ``results.csv`` files that exist, and exits non-zero on any shortfall.

    python scripts/reconcile_grid.py --grid seasonal_4x4x10
    python scripts/reconcile_grid.py --grid seasonal_4x4x10 --missing   # list them

Run it before calling a run finished, and again before reading its results: "the fleet
is empty" and "the grid is complete" are different statements, and only this one checks
the second. ``supervise/reap_finished.sh`` calls it as its final gate.

Trees are DISCOVERED rather than assumed. A grid's arms are crossed with its neural
models, but a model that ignores the arm axis is legitimately trained once under the
un-suffixed path (Pareto/NBD is fit per dataset from the target column alone, so an arm
would train twelve identical copies). Asserting the full cross would report that correct
tree as twelve missing ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from grids import available_grids, load_grid           # noqa: E402
from panelclv.studies import pareto_nbd_grid as ps     # noqa: E402


def suites_for(spec) -> list[str]:
    """Every ``<combo>__<dataset>`` name the grid generates, in manifest order."""
    rows = ps.list_pnbd_datasets(spec.dataset_dir).itertuples(index=False)
    return [f"{r.combo}__{r.dataset}" for r in rows]


def trees_for(spec, model) -> list[tuple[str | None, Path]]:
    """The (arm, path) trees this model actually has, falling back to what it declares.

    Returns existing trees where any exist, so a model trained under one arm — or under
    none — reconciles against what it really wrote. When a model has no tree at all,
    the declared arms are returned instead, so the report says "absent" rather than
    silently skipping the model.
    """
    arms: list[str | None] = [a.name for a in spec.arms] or [None]
    found = [(a, spec.train_base(model.name, a)) for a in arms
             if spec.train_base(model.name, a).is_dir()]
    if found:
        return found
    plain = spec.train_base(model.name)
    if plain.is_dir():
        return [(None, plain)]
    return [(a, spec.train_base(model.name, a)) for a in arms]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", required=True, help=f"one of: {', '.join(available_grids())}")
    ap.add_argument("--missing", action="store_true", help="list every missing suite")
    ap.add_argument("--strict", action="store_true",
                    help="also fail when a declared model was never run at all")
    args = ap.parse_args()

    spec = load_grid(args.grid)
    expected = suites_for(spec)
    print(f"{spec.name}: {len(expected)} datasets declared\n")

    width = max(len(f"{m.name} / {a}") for m in spec.models
                for a, _ in trees_for(spec, m)) + 2
    shortfall: dict[str, list[str]] = {}       # started and incomplete — the real danger
    absent: list[str] = []                      # never started at all — visible anyway

    for model in spec.models:
        for arm, tree in trees_for(spec, model):
            label = f"{model.name} / {arm or '(no arm)'}"
            missing = [s for s in expected
                       if not (tree / s / "results.csv").exists()]
            have = len(expected) - len(missing)
            if not missing:
                print(f"  ok    {label:<{width}} {have:>3}/{len(expected)}")
            elif have == 0:
                print(f"  ABSENT {label:<{width}}   not run")
                absent.append(label)
            else:
                print(f"  SHORT {label:<{width}} {have:>3}/{len(expected)}"
                      f"   missing {len(missing)}")
                shortfall[label] = missing

    if absent:
        # An empty tree is obvious to anyone looking; a grid may also declare a model it
        # deliberately does not run (ValendinLSTM cannot take engineered time features,
        # F11, and Rules.md still lists its scope as open). So this reports but does not
        # fail unless asked -- otherwise the gate is permanently red and stops being read.
        print(f"\nnot run: {len(absent)} declared tree(s) have no suites at all")
        for label in absent:
            print(f"  - {label}")

    if not shortfall:
        print("\ncomplete: every started tree has all "
              f"{len(expected)} results.csv" + (" (see 'not run' above)" if absent else ""))
        return 1 if (absent and args.strict) else 0

    total = sum(len(v) for v in shortfall.values())
    print(f"\nINCOMPLETE: {total} suite(s) missing from {len(shortfall)} started tree(s)"
          "\nThis is the failure that hides: the tree looks trained and is not.")
    for label, missing in shortfall.items():
        shown = missing if args.missing else missing[:5]
        print(f"  {label}: " + ", ".join(shown)
              + ("" if args.missing or len(missing) <= 5
                 else f", ... (+{len(missing) - 5} more; pass --missing)"))
    print("\nRecover with a targeted resume — it skips suites that already have a\n"
          "results.csv and overwrites half-written ones (F19):\n"
          "  scripts/run_pnbd_grid.py --grid <grid> --model <type> --arm <arm> --shard 1/1")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
