"""Does a stored electronics forecast vary between customers, and by what did it read?

`docs/benchmarks-real-panels.md` §"The forecast collapse on long sparse panels" claims
that a neural model whose only *per-customer* input is the transaction count forecasts
nearly the same number for everyone on electronics, and that adding one persistent
customer-level channel restores the spread. This script is where that table comes from.

**The measure.** For one stored forecast, the coefficient of variation of per-customer
holdout totals, `std / mean` over customers. A model that hands every customer the same
number scores 0, whatever that number is; Pareto/NBD, which reads each customer's
recency and frequency, scores about 1.1 here. Beside it, Spearman of those totals against
the true holdout totals — CV says the forecast *varies*, Spearman says it varies in the
right order.

**Grouping is read off disk, not off the suite name.** A row is one
`(model, per-customer channels, calendar channels)` signature, taken from each suite's own
`data_summary.seq_cols`, with the target and the calendar columns split out: `week`,
`week_sin`, `week_cos` and `year_idx` are the same for every customer in a period, so a
model reading only those plus its own count has no per-customer input but the count. This
is deliberate. An experiment folder name (`no_ar`, `NoCov`, `real_panel_arms`) says what a
run was *called*; `seq_cols` says what it *read*, and the two have already diverged once —
`ar_encoding`'s `no_ar` arm carries `week_sin` / `week_cos`, so it is not a count-only run.

**A forecast is counted once.** Pareto/NBD is deterministic and its single fit was copied
into five old suite folders, so the same prediction file appears repeatedly; rows are
de-duplicated on the forecast's own bytes, which leaves the neural rows untouched (a
500-path Monte Carlo mean of an unseeded training run does not repeat) and counts that fit
once. The suite list under each row still names every folder the row was assembled from.

**Population: every stored electronics forecast on the benchmark's two-year windows.**
Membership is executable rather than nominal — a suite qualifies when its config names the
benchmark's electronics holdout (2001, 829 customers) and its stored predictions carry
exactly the rebuilt cohort in the rebuilt order over 52 weeks. Four early suites predate
`panel_config` in the config file and are matched on their `data_summary` instead
(`T_CAL` 104, `T_HOLD` 52, `validation_start` 2000-01-01), which is the same window. The
three-year suites are excluded by the same rule: they forecast 2002.

Run from the repo root:

    PYTHONPATH=src python scripts/measure_forecast_collapse.py

It prints the markdown table and, beneath it, the suites behind every row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_real_panel_benchmarks as benchmarks      # noqa: E402

from panelclv.data_preparation.target_channel import holdout_actuals   # noqa: E402

PANEL = "electronics"

# Channels that take the same value for every customer in a period. A model reading only
# these, plus its own past count, carries no per-customer signal into the rollout.
CALENDAR = ("week", "week_sin", "week_cos", "year_idx")

# Models that do not read a sequence at all: their column set says nothing about what they
# know about a customer, so it is replaced by what they actually fit.
NON_SEQUENTIAL = {"ParetoNBD": "recency, frequency, tenure (HB MCMC)",
                  "ParetoNBD_MLE": "recency, frequency, tenure (MLE)"}


def eligible(config: dict) -> bool:
    """Is this suite on the benchmark's electronics windows?

    `panel_config` carries the windows on every suite written since the runners were
    unified; the four earliest suites have only `data_summary`, whose T_CAL / T_HOLD /
    validation_start pin the same window.
    """
    summary = config.get("data_summary") or {}
    if summary.get("n_customers") != 829:
        return False
    panel = config.get("panel_config") or {}
    if panel:
        return panel.get("holdout_start") == "2001-01-01"
    return (summary.get("T_CAL") == 104 and summary.get("T_HOLD") == 52
            and summary.get("validation_start") == "2000-01-01")


def compress(names: list[str]) -> str:
    """`active_in_last_{2,4,8,16,32}_periods` for the five flags, other names verbatim.

    The nested activity flags are the only channel set long enough to make a table cell
    unreadable, and they always travel together, so only they are folded.
    """
    depths = [n for n in names if n.startswith("active_in_last_")]
    rest = [n for n in names if not n.startswith("active_in_last_")]
    folded = ([f"active_in_last_{{{','.join(n.split('_')[3] for n in depths)}}}_periods"]
              if depths else [])
    return ", ".join(folded + rest)


def fold(suites: set[str]) -> list[str]:
    """Suite names with their replication suffix folded, and nothing else hidden.

    A numbered family becomes `__r*`; a lettered one lists the letters that are actually
    on disk, so a single-suite arm reads `__a` rather than claiming an `a`-`e` sweep.
    """
    letters: dict[str, set[str]] = {}
    folded: set[str] = set()
    for suite in suites:
        head, _, tail = suite.rpartition("__r")
        if head and tail.isdigit():
            folded.add(f"{head}__r*")
        elif len(suite) > 3 and suite[-3:-1] == "__" and suite[-1] in "abcde":
            letters.setdefault(suite[:-1], set()).add(suite[-1])
        else:
            folded.add(suite)
    for prefix, found in letters.items():
        folded.add(prefix + (f"{{{','.join(sorted(found))}}}" if len(found) > 1
                             else found.pop()))
    return sorted(folded)


def signature(config: dict, model: str) -> tuple[str, str]:
    """`(per-customer channels, calendar channels)` this suite's model read."""
    if model in NON_SEQUENTIAL:
        return NON_SEQUENTIAL[model], "—"
    seq = list((config.get("data_summary") or {}).get("seq_cols") or [])
    target = (config.get("panel_config") or {}).get("target_col", "Transactions")
    calendar = [c for c in seq if c in CALENDAR]
    per_customer = [c for c in seq if c != target and c not in CALENDAR]
    return (compress(per_customer) or "count only", ", ".join(calendar) or "none")


def collect(actual: np.ndarray, ref_ids: np.ndarray) -> pd.DataFrame:
    """One row per stored forecast: its signature, its CV and its Spearman."""
    actual_totals = actual.sum(axis=1)
    rows: list[dict] = []
    for config_path in sorted(Path("Studies").glob("*/config.json")):
        suite = config_path.parent
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        if not eligible(config):
            continue
        for model_dir in sorted(p for p in suite.iterdir() if p.is_dir()):
            files = sorted((model_dir / "Predictions").glob("Prediction_*.csv"))
            per_customer, calendar = signature(config, model_dir.name)
            for file in files:
                frame = pd.read_csv(file)
                ids = frame.iloc[:, 0].to_numpy()
                values = frame.iloc[:, 1:].to_numpy(dtype=float)
                # The cohort check is the other half of the membership rule: a forecast
                # over different customers, or a different horizon, is not comparable.
                if values.shape != actual.shape or not np.array_equal(ids, ref_ids):
                    continue
                totals = values.sum(axis=1)
                rows.append({
                    # Identifies a forecast by its contents, so the deterministic
                    # benchmark's one fit is not counted once per folder holding it.
                    "digest": hash(values.tobytes()),
                    "model": model_dir.name,
                    "per_customer": per_customer,
                    "calendar": calendar,
                    "suite": suite.name,
                    "cv": float(totals.std() / totals.mean()),
                    "spearman": benchmarks.spearman(totals, actual_totals),
                })
    return pd.DataFrame(rows)


def main() -> None:
    data = benchmarks.build_data(PANEL, "2y")
    actual = holdout_actuals(data)                      # (N, T_HOLD)
    ref_ids = np.asarray(data["ids"])
    forecasts = collect(actual, ref_ids)

    # Statistics over distinct forecasts; the suite list below still reads from every
    # folder, so a folder holding a copy is named rather than silently dropped.
    unique = forecasts.drop_duplicates(subset=["model", "per_customer", "calendar", "digest"])
    grouped = unique.groupby(["per_customer", "model", "calendar"], sort=False)
    table = grouped.agg(forecasts=("cv", "size"), median_cv=("cv", "median"),
                        mean_spearman=("spearman", "mean")).reset_index()
    # Ordered by what the model knows about a customer: the collapsed rows first.
    table = table.sort_values(["median_cv", "model"]).reset_index(drop=True)

    print(f"{PANEL}: N = {len(ref_ids)}, holdout = 52 weeks from 2001-01-01, "
          f"{len(unique)} distinct forecasts ({len(forecasts)} stored) over "
          f"{forecasts.suite.nunique()} suites.\n")
    print("| per-customer input beside the count | model | calendar | forecasts | "
          "median CV | mean Spearman |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    for row in table.itertuples(index=False):
        print(f"| {row.per_customer} | {row.model} | {row.calendar} | {row.forecasts} | "
              f"{row.median_cv:.2f} | {row.mean_spearman:.2f} |")

    print("\nSuites behind each row:\n")
    for row in table.itertuples(index=False):
        part = forecasts[(forecasts.per_customer == row.per_customer)
                         & (forecasts.model == row.model)
                         & (forecasts.calendar == row.calendar)]
        print(f"- **{row.model}, {row.per_customer}** ({row.calendar}): "
              f"{', '.join(fold(set(part['suite'])))}")


if __name__ == "__main__":
    main()
