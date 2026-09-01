"""The auto-named folder a forecast run is written into, and its provenance sidecar.

A prediction dump lands in ``base_dir/<run name>``, where the run name is derived by
the caller from its config and its seed — never from the wall clock. That is what
makes an earlier run findable: the same config and the same seed name the same
folder, so a reader holding the config can *construct* the path instead of having to
remember what time the run was started (``CLAUDE.md``'s reproducibility priority — a
path is part of a result).

The wall-clock time is still worth having, so it moves out of the folder name and
into ``run_metadata.json`` beside the predictions: provenance you can read, never a
key you have to know in order to look something up. Re-running the same config and
seed writes over that run's folder, and the timestamp then describes the files that
are actually sitting there.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# The provenance sidecar written beside every prediction dump. One spelling, because
# the two writers (the Monte Carlo simulator and the Pareto/NBD benchmark) produce one
# on-disk layout between them.
RUN_METADATA_FILE = "run_metadata.json"


def create_run_directory(base_dir: str | Path, run_name: str) -> Path:
    """Create ``base_dir/run_name`` and drop the run's timestamp in it.

    `run_name` is the caller's config-and-seed-derived name; nothing here adds to it.
    Existing folders are reused rather than refused, which is what lets the same
    config and seed be re-run into the same place. Returns the folder.
    """
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / RUN_METADATA_FILE, "w") as fh:
        json.dump(
            {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            fh,
            indent=2,
        )
    return run_dir
