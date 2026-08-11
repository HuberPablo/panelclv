"""Relabel archived study results produced by the MLE Pareto/NBD.

Every `ParetoNBD` result under `Studies/` was produced by the frequentist-MLE
estimator (now in `archive/pareto_nbd.py`). Since "Pareto/NBD" now means the
hierarchical-Bayes model, those stored results are mislabeled — same name,
different model. Rename them `ParetoNBD_MLE` so the estimator is recorded on disk.

Text edits are line-oriented so the CSVs stay byte-identical apart from the label.
Idempotent: re-running finds nothing to do.

    python relabel_mle.py            # dry run
    python relabel_mle.py --apply
"""
import json
import sys
from pathlib import Path

OLD, NEW = "ParetoNBD", "ParetoNBD_MLE"
ROOT = Path("Studies")
apply = "--apply" in sys.argv


def relabel_model_column(path: Path) -> bool:
    """Rewrite `OLD` -> `NEW` in the leading `model` column of a metrics CSV."""
    lines = path.read_text().splitlines(keepends=True)
    if not lines or not lines[0].startswith("model,"):
        raise SystemExit(f"{path}: expected 'model' as the first column, got {lines[0]!r}")
    out, changed = [], False
    for line in lines[1:]:
        if line.startswith(f"{OLD},"):
            line = f"{NEW}," + line[len(OLD) + 1:]
            changed = True
        out.append(line)
    if changed and apply:
        path.write_text(lines[0] + "".join(out))
    return changed


def relabel_json_name(path: Path) -> bool:
    """Rewrite the model-spec `name` field, leaving the rest of the file alone."""
    text = path.read_text()
    old, new = f'"name": "{OLD}"', f'"name": "{NEW}"'
    if old not in text:
        return False
    if apply:
        path.write_text(text.replace(old, new))
    return True


actions = {"dirs": 0, "metrics": 0, "results": 0, "model_cfg": 0, "suite_cfg": 0}
suites = set()

for model_dir in sorted(ROOT.rglob(OLD)):
    if not model_dir.is_dir():
        continue
    suite = model_dir.parent
    suites.add(suite)

    # config.json / metrics.csv are edited in place first, while the paths are
    # still valid, then the folder itself is renamed.
    cfg = model_dir / "config.json"
    if cfg.exists():
        record = json.loads(cfg.read_text())
        if record.get("model_type") != "pareto_nbd":
            raise SystemExit(f"{cfg}: unexpected model_type {record.get('model_type')!r}")
        actions["model_cfg"] += relabel_json_name(cfg)
    metrics = model_dir / "metrics.csv"
    if metrics.exists():
        actions["metrics"] += relabel_model_column(metrics)

    target = suite / NEW
    if target.exists():
        raise SystemExit(f"{target} already exists — refusing to overwrite")
    actions["dirs"] += 1
    if apply:
        model_dir.rename(target)

for suite in sorted(suites):
    results = suite / "results.csv"
    if results.exists():
        actions["results"] += relabel_model_column(results)
    cfg = suite / "config.json"
    if cfg.exists():
        actions["suite_cfg"] += relabel_json_name(cfg)

print(f"{'APPLIED' if apply else 'DRY RUN'}: {len(suites)} suites")
for k, v in actions.items():
    print(f"  {k:10s} {v}")
