"""The documentation names things that exist, and its configs still construct.

`docs/running-a-model.md` opens by claiming "every shape, value and file name below
was produced by executing the pipeline, not by reading the source". Nothing checked
that, and it decayed in exactly the two ways a prose claim decays:

- a **path** or **symbol** is renamed and one reference is missed. The
  `forecast_* -> rollout_*` sweep renamed `test_forecast_never_reads_the_holdout` in
  the same commit that swept the docs, and still left one doc reference behind.
- a **config literal** in a fenced block is copied by a reader and does not run. The
  README quickstart omitted `embedded_cols`, so the target went unembedded and
  `Embedder` raised on the first Optuna trial — long after `prepare_dataset` had
  succeeded, which reads as a package bug rather than a config typo.

Both are mechanical, so both are checked here. What is NOT checked, and cannot be, is
a claim that is false without naming anything that moved: this chapter once asserted
the package applies "no per-feature standardisation" while `standardize_covariates`
had been running for a day. Only a reader catches that one.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The seven prose surfaces. Subpackage docstrings are excluded: they live in `src/`
# and are already covered by `test_imports.py` resolving what they advertise.
DOC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "CONTEXT.md",
    REPO_ROOT / "docs" / "running-a-model.md",
    REPO_ROOT / "docs" / "feature_engineering.md",
    *sorted((REPO_ROOT / "docs" / "adr").glob("*.md")),
]

# Directories a documented path may resolve against, so `models/embedders.py` and
# `src/panelclv/models/embedders.py` are both legal ways to write one.
PATH_ROOTS = [REPO_ROOT, REPO_ROOT / "src" / "panelclv", REPO_ROOT / "src"]

# Where a documented symbol may live.
SYMBOL_ROOTS = [REPO_ROOT / "src" / "panelclv", REPO_ROOT / "tests", REPO_ROOT / "scripts"]

# Names that legitimately appear in the docs and legitimately do not exist in this
# repo. Each needs a reason: this set is the escape hatch, so adding to it should be
# a deliberate act rather than the reflex that silences a real failure.
ALLOWED_ABSENT = {
    # --- deliberately historical: an ADR records the world at decision time, and the
    # old names ARE its content. ADR-0006 says adding a model "used to touch"
    # VALID_MODEL_TYPES; renaming that to MODEL_REGISTRY would blame the fix for the
    # problem. These must NOT be swept when the code moves.
    "VALID_MODEL_TYPES",       # ADR-0006: one of the three pre-registry enumerations
    "_FORECASTERS",            # ADR-0006: ditto
    "plot_utils",              # ADR-0002: the module whose deletion the ADR records
    "rollout_composite",       # ADR-0003: retired selection metric
    "selection_metric",        # ADR-0003 + the gotcha recording its removal
    "prediction_source",       # ADR-0008: retired knob
    "REFIT_ON_FULL_CALIBRATION",  # ADR-0008: the notebook toggle it replaced
    "studies/analysis.py",     # ADR-0006: the module whose drift the ADR records
    "lstm_cross_entropy_rollout_composite_20260601_1651",  # ADR-0003: a stored study
                               # name, i.e. data that was written, not a symbol
    # --- external to this repo ---
    "rfm2lstm",                # Valendin et al.'s published GitHub package
    "script_on_start",         # VastAI/: a shell script, not a Python symbol
    # --- prose shorthand, not an identifier ---
    "active_in_last_K",        # informal for the `active_in_last_<K>_periods` family
    "NoCov",                   # a fragment of an archived suite directory name
    "config",                  # `config.json`, discussed as a filename
}


def _iter_fenced_blocks(text: str, lang: str):
    """Yield the body of every ```<lang> fenced block."""
    return re.findall(rf"```{lang}\n(.*?)```", text, re.DOTALL)


def _strip_fenced_blocks(text: str) -> str:
    """Prose only. Code blocks are checked separately and by stricter means."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _path_exists(path: str) -> bool:
    """Does a documented path resolve? A bare filename may live anywhere in the tree."""
    if any((root / path).exists() for root in PATH_ROOTS):
        return True
    # `Study.ipynb` and `map.md` are written without a directory; accept them wherever
    # they actually sit, since the doc is naming the file rather than its location.
    return "/" not in path and any(REPO_ROOT.rglob(path))


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_documented_paths_exist(doc):
    """Every backticked file path in the docs resolves to a real file."""
    referenced = set(
        re.findall(
            r"`([A-Za-z_][A-Za-z0-9_./-]*\.(?:py|ipynb|md|csv|json|sh|toml))`",
            doc.read_text(),
        )
    )
    missing = sorted(
        path
        for path in referenced
        if path not in ALLOWED_ABSENT
        and Path(path).stem not in ALLOWED_ABSENT
        and not _path_exists(path)
    )
    assert not missing, (
        f"{doc.name} references paths that do not exist: {missing}\n"
        "Either the file moved and the doc was not swept, or the reference is "
        "historical and belongs in ALLOWED_ABSENT with a reason."
    )


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_documented_symbols_exist(doc):
    """Every backticked identifier in the prose is findable somewhere in the code.

    Deliberately loose: a plain-text search, not an import. The point is to catch a
    rename that left a doc reference behind, not to verify that the symbol means what
    the sentence claims.
    """
    prose = _strip_fenced_blocks(doc.read_text())
    # snake_case or CamelCase of at least two segments — enough to be an identifier
    # rather than an English word in backticks ("`auto`", "`week`").
    candidates = set(re.findall(r"`(_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*)`", prose))

    # This file is excluded from its own haystack. Its docstring and ALLOWED_ABSENT
    # quote stale names as examples, which would silently vouch for every one of them —
    # a gate that counts its own prose as evidence cannot fail.
    sources = [
        path.read_text()
        for root in SYMBOL_ROOTS
        for path in root.rglob("*.py")
        if path.resolve() != Path(__file__).resolve()
    ]
    haystack = "\n".join(sources)

    # Searched exactly as written, underscore included. `\b` does not match between
    # "_" and a letter (both are word characters), so stripping the prefix would make
    # every private symbol look absent — and searching the bare stem would hide the
    # drift worth catching, a doc that says `suggest_x` for a private `_suggest_x`.
    missing = sorted(
        name for name in candidates
        if name not in ALLOWED_ABSENT and name not in haystack
    )
    assert not missing, (
        f"{doc.name} names symbols that appear nowhere in src/, tests/ or scripts/: "
        f"{missing}\nEither a rename missed this reference, or the name is historical "
        "and belongs in ALLOWED_ABSENT with a reason."
    )


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_documented_panel_configs_construct(doc):
    """Every `PanelConfig(...)` literal in a fenced block builds a valid config.

    This is the check the README quickstart needed. `PanelConfig` validates at
    construction, so building one is enough to catch a missing or misspelled field —
    and `embedded_cols` naming the target is asserted separately below, because a
    config can be perfectly valid and still build no neural model.
    """
    panel_config = pytest.importorskip("panelclv.configs.panel_config")
    PanelConfig = panel_config.PanelConfig
    normalize_embedded_cols = panel_config.normalize_embedded_cols

    literals = []
    for block in _iter_fenced_blocks(doc.read_text(), "python"):
        # Parse rather than regex: a config literal spans many lines and nests dicts.
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # illustrative fragments are not required to parse
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PanelConfig":
                literals.append(node)

    for node in literals:
        kwargs = {}
        for kw in node.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                pytest.fail(
                    f"{doc.name}: PanelConfig literal on line {node.lineno} of a code "
                    f"block passes a non-literal to {kw.arg!r}; the doc's example "
                    "cannot be checked, so simplify it or make it a literal."
                )

        config = PanelConfig(**kwargs)  # raises on an invalid combination

        embedded = normalize_embedded_cols(config.embedded_cols)
        assert config.target_col in embedded, (
            f"{doc.name}: the PanelConfig on line {node.lineno} of a code block does "
            f"not embed its target {config.target_col!r}. The target's cardinality IS "
            "the softmax head size, so this config builds no neural model — `Embedder` "
            "raises inside the first Optuna trial, long after prepare_dataset has "
            "succeeded. Add embedded_cols={<target>: 'auto'}."
        )
