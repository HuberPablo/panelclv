"""INPUT_SPEC validation + JSON save/load helpers.

An INPUT_SPEC describes which columns the model should embed and how large
each embedding table should be. Minimal format:

    INPUT_SPEC = {
        "embedded_cols": {
            "Transactions": 10,   # values must be encoded as 0..9
            "Gender":        2,   # values must be 0 or 1
        }
    }

Specs live as JSON on disk so notebooks and training scripts can reference them
by name. There is no machine-agnostic default location, so the save/load/list
helpers require an explicit ``directory=`` (e.g. the repo's ``inputs_configs/``).

Standard library only — no third-party dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# No machine-agnostic default exists, so this is intentionally None: callers must
# pass `directory=` explicitly (e.g. the repo's `inputs_configs/`). This replaces a
# previously hardcoded absolute path that only resolved on one machine.
DEFAULT_INPUT_SPEC_DIR: Path | None = None


def _require_directory(directory: Path | None) -> Path:
    """Resolve a caller-supplied INPUT_SPEC directory, failing loudly if omitted.

    There is no sensible default (see DEFAULT_INPUT_SPEC_DIR), so a missing
    directory is a usage error rather than something to guess at.
    """
    if directory is None:
        raise ValueError(
            "No INPUT_SPEC directory given. Pass directory= explicitly "
            "(e.g. the repo's 'inputs_configs/') — there is no default location."
        )
    return Path(directory)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_input_spec(input_spec: Any) -> None:
    """Sanity-check an INPUT_SPEC dict.

    Raises TypeError on wrong types, ValueError on wrong values. Returns
    None when the spec is valid.
    """
    if not isinstance(input_spec, dict):
        raise TypeError(
            f"input_spec must be a dict, got {type(input_spec).__name__}"
        )
    if "embedded_cols" not in input_spec:
        raise ValueError("input_spec must contain the key 'embedded_cols'")

    embedded = input_spec["embedded_cols"]
    if not isinstance(embedded, dict):
        raise TypeError(
            f"input_spec['embedded_cols'] must be a dict, "
            f"got {type(embedded).__name__}"
        )

    for col, num_categories in embedded.items():
        if not isinstance(col, str) or not col:
            raise ValueError(
                f"embedded_cols keys must be non-empty strings; got {col!r}"
            )
        # `bool` is a subclass of `int`, exclude it explicitly so True/False
        # can't sneak in as a cardinality.
        if isinstance(num_categories, bool) or not isinstance(num_categories, int):
            raise TypeError(
                f"embedded_cols[{col!r}] must be an int, "
                f"got {type(num_categories).__name__}"
            )
        if num_categories <= 1:
            raise ValueError(
                f"embedded_cols[{col!r}] must be > 1, got {num_categories}"
            )

    return None


# ---------------------------------------------------------------------------
# Save / load / list
# ---------------------------------------------------------------------------


def _resolve_path(name: str, directory: Path) -> Path:
    """Append '.json' if missing and join with `directory`."""
    fname = name if name.endswith(".json") else f"{name}.json"
    return Path(directory) / fname


def save_input_spec(
    input_spec: dict,
    name: str,
    directory: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate and persist an INPUT_SPEC as JSON.

    `directory` is required (no default location). The target directory is created
    if it doesn't exist. Refuses to clobber an existing file unless `overwrite=True`.
    Returns the path written to.
    """
    validate_input_spec(input_spec)

    directory = _require_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = _resolve_path(name, directory)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass overwrite=True to replace it."
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(input_spec, f, indent=2, sort_keys=True)

    return path


def load_input_spec(
    name: str,
    directory: Path | None = None,
    validate: bool = True,
) -> dict:
    """Load an INPUT_SPEC JSON file by name.

    `directory` is required (no default location). If `validate` is True, the loaded
    dict is passed through `validate_input_spec` before being returned.
    """
    path = _resolve_path(name, _require_directory(directory))
    if not path.exists():
        raise FileNotFoundError(f"No INPUT_SPEC at {path}")

    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    if validate:
        validate_input_spec(spec)
    return spec


def list_input_specs(
    directory: Path | None = None,
) -> list[str]:
    """Return sorted INPUT_SPEC names (without the .json extension).

    `directory` is required (no default location). Returns an empty list if the
    directory does not exist.
    """
    directory = _require_directory(directory)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


# ---------------------------------------------------------------------------
# Example usage (do not execute — kept here as a quick reference)
# ---------------------------------------------------------------------------
# from panelclv.configs.transformations_spec import (
#     save_input_spec, load_input_spec, list_input_specs,
# )
#
# INPUT_SPEC = {
#     "embedded_cols": {
#         "Transactions": 10,
#         "Gender":        2,
#     }
# }
#
# save_input_spec(INPUT_SPEC, "full_transactions_gender", overwrite=True)
# loaded_spec = load_input_spec("full_transactions_gender")
# print(loaded_spec)
# print(list_input_specs())
#
# model = MultinomialLSTMModel(
#     seq_cols=data["seq_cols"],
#     input_spec=loaded_spec,
# )
