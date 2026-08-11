"""Migrate checkpoints written before the embedder seam (ADR-0005) landed.

Extracting the embedder moved the embedding modules out of each backbone and into
`backbone.embedder`, which renames their `state_dict` keys. A checkpoint written
before that commit fails `load_state_dict(..., strict=True)` until its keys are
renamed. Weights themselves are untouched — only the names change, so a migrated
checkpoint reproduces its original forecast bit for bit.

Three renames, and the third is easy to miss: the LSTM called its covariate module
`covariate_proj` and the Transformer called the same thing `covariate_projection`.
`ProjectedEmbedder` unified them on `covariate_proj`, so Transformer checkpoints with
covariates need both a move and a rename.

    backbone._emb_modules.*        -> backbone.embedder._emb_modules.*
    backbone.covariate_proj.*      -> backbone.embedder.covariate_proj.*
    backbone.covariate_projection.* -> backbone.embedder.covariate_proj.*   (Transformer)

Run from the repo root:
    .../thesis_rocm/bin/python scripts/migrations/rename_embedder_checkpoint_keys.py checkpoints
    .../thesis_rocm/bin/python scripts/migrations/rename_embedder_checkpoint_keys.py checkpoints --apply

The first form is a dry run; only --apply writes. Idempotent: a checkpoint already
carrying the new keys is left alone, so re-running finds nothing to do.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

_NEW_PREFIX = "backbone.embedder."

# Old key prefix -> new key prefix. Order matters: `covariate_projection` must be
# tested before `covariate_proj`, since the latter is a prefix of the former.
_RENAMES = (
    ("backbone.covariate_projection.", _NEW_PREFIX + "covariate_proj."),
    ("backbone.covariate_proj.", _NEW_PREFIX + "covariate_proj."),
    ("backbone._emb_modules.", _NEW_PREFIX + "_emb_modules."),
)


def migrate_state_dict(state: dict) -> tuple[dict, int]:
    """Return (renamed state_dict, number of keys renamed)."""
    out, renamed = {}, 0
    for key, value in state.items():
        for old, new in _RENAMES:
            if key.startswith(old):
                key = new + key[len(old):]
                renamed += 1
                break
        out[key] = value
    return out, renamed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory to scan for checkpoints")
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"not a directory: {args.root}")

    paths = sorted(p for p in args.root.rglob("*") if p.suffix in {".pt", ".pth", ".ckpt"})
    migrated = skipped = failed = 0

    for path in paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:                    # not a bare state_dict, or corrupt
            print(f"  [skip] {path}: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        if not isinstance(state, dict):
            print(f"  [skip] {path}: not a state_dict")
            failed += 1
            continue

        new_state, renamed = migrate_state_dict(state)
        if renamed == 0:
            skipped += 1
            continue

        migrated += 1
        print(f"  {'[write]' if args.apply else '[would]'} {path}  ({renamed} keys)")
        if args.apply:
            torch.save(new_state, path)

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: {len(paths)} checkpoints scanned")
    print(f"  migrated {migrated}   already current {skipped}   unreadable {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
