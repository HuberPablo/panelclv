"""Numbered forecast runs: deterministic save/load of per-model predictions.

One experiment "run" is a numbered folder under a config name:

    <root>/<config_name>/<number>/
        config.json                 # the PanelConfig used for the run
        manifest.json               # {model display name: relative predictions path}
        <model_slug>/predictions.csv
        ...

`ForecastRun.new` creates the next run as `max(existing numbers) + 1` (monotonic,
so a number always refers to the same run even after deletions). `ForecastRun.open`
reads an existing run by number. Saving and loading use the SAME model names — the
manifest maps each display name (e.g. "Pareto/NBD (HB)") to its slugified folder —
so retrieval never needs a hand-typed path.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .plot_utils import save_predictions_to_csv

_MANIFEST = "manifest.json"
_CONFIG = "config.json"


def _slug(name: str) -> str:
    """Filesystem-safe folder name from a model display name."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return s or "model"


class ForecastRun:
    """A single numbered run folder for one config; see module docstring."""

    def __init__(self, dir_: Path, number: int) -> None:
        self.dir = Path(dir_)
        self.number = number

    # -- construction ------------------------------------------------------

    @classmethod
    def new(cls, root: str | Path, config_name: str) -> "ForecastRun":
        """Create the next run, numbered `max(existing) + 1` (1 if none)."""
        parent = Path(root) / config_name
        parent.mkdir(parents=True, exist_ok=True)
        nums = [int(p.name) for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
        number = max(nums, default=0) + 1
        run_dir = parent / str(number)
        run_dir.mkdir()
        return cls(run_dir, number)

    @classmethod
    def open(cls, root: str | Path, config_name: str, run: int) -> "ForecastRun":
        """Open an existing run by number."""
        run_dir = Path(root) / config_name / str(run)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"no run {run} at {run_dir}")
        return cls(run_dir, int(run))

    # -- paths -------------------------------------------------------------

    def path(self, filename: str) -> Path:
        """A path inside this run's folder (e.g. for a metrics table)."""
        return self.dir / filename

    # -- write -------------------------------------------------------------

    def save_config(self, config: Any) -> Path:
        """Write the run's config to config.json (dataclass or plain dict)."""
        payload = asdict(config) if is_dataclass(config) else dict(config)
        payload["_run"] = self.number
        payload["_created"] = datetime.now().isoformat(timespec="seconds")
        out = self.dir / _CONFIG
        out.write_text(json.dumps(payload, indent=2, default=str))
        return out

    def save_predictions(self, name: str, forecast: dict[str, Any], data: dict[str, Any]) -> Path:
        """Write one model's per-customer mean predictions and record it.

        `forecast` is a forecaster dict (`mc_forecast` / `pareto_forecast`);
        its `prediction_mean` (N, T_HOLD) is saved as a wide per-customer CSV.
        The display `name` is stored in the manifest, mapped to its slug folder.
        """
        csv_path = self.dir / _slug(name) / "predictions.csv"
        save_predictions_to_csv(
            forecast["prediction_mean"],
            csv_path,
            customer_ids=data.get("ids"),
            id_col=data.get("id_col", "customer_id"),
        )
        manifest = self._read_manifest()
        manifest[name] = str(csv_path.relative_to(self.dir))
        (self.dir / _MANIFEST).write_text(json.dumps(manifest, indent=2))
        return csv_path

    # -- read --------------------------------------------------------------

    def predictions(self) -> dict[str, Path]:
        """Return {model name: predictions path} from the manifest."""
        return {name: self.dir / rel for name, rel in self._read_manifest().items()}

    def _read_manifest(self) -> dict[str, str]:
        f = self.dir / _MANIFEST
        return json.loads(f.read_text()) if f.exists() else {}
