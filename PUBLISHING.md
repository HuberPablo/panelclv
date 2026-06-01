# Publishing this package

The package is build-ready: `pyproject.toml` is valid (PEP 621 metadata, MIT license,
classifiers), the project uses a **src-layout** (`src/panelclv/`), and `LICENSE` /
`README.md` are in place. `from panelclv... import ...` works once the package is
installed. What remains needs your accounts / tooling and is listed below as copy-paste
steps.

Paths below assume the repo root is `/home/virthian/Desktop/Thesis/panelclv/` (the
folder may be named anything — git and PyPI don't care). The `origin` remote already
points at `https://github.com/HuberPablo/panelclv.git`, branch `main`, so there is no
repo-creation step.

## Layout that ships

The wheel contains **only** the importable package — `src/panelclv/` and its
subpackages (`models`, `training`, `tuning`, `evaluation`, `benchmarks`, `experiments`,
`data_preparation`, `configs`). `notebooks/`, `scripts/`, `tests/`, and all data
(`Datasets/`, etc.) sit outside `src/` and are intentionally **not** packaged.
`[tool.setuptools.packages.find]` uses `where = ["src"]` + `include = ["panelclv*"]`,
so new subpackages are auto-discovered.

## A. Local install (works today)

Into any environment you control (NOT the protected `thesis_rocm` venv):

```bash
cd /home/virthian/Desktop/Thesis/panelclv
pip install -e .              # editable install for development
pip install -e ".[dev]"       # + pytest, then: pytest -q
# or  pip install .           # regular install
```

## B. Build the distribution artifacts

Use a clean throwaway environment — do NOT modify `thesis_rocm`. `pipx` is simplest:

```bash
cd /home/virthian/Desktop/Thesis/panelclv
rm -rf dist/ build/ src/*.egg-info        # start clean
pipx run build                            # -> dist/panelclv-0.1.0-py3-none-any.whl + .tar.gz
pipx run twine check dist/*               # validates metadata + that the README renders
```

If you don't have `pipx`:

```bash
python -m venv /tmp/buildenv
/tmp/buildenv/bin/pip install build twine
/tmp/buildenv/bin/python -m build
/tmp/buildenv/bin/twine check dist/*
```

**Inspect the wheel before uploading** — confirm only `panelclv/` ships:

```bash
python -m zipfile -l dist/panelclv-0.1.0-py3-none-any.whl
# expect only panelclv/... entries (+ *.dist-info). NO notebooks/, tests/, Datasets/.
```

## C. Dry run on TestPyPI

```bash
pipx run twine upload --repository testpypi dist/*
# verify it installs + imports in a fresh env (real deps like torch come from real PyPI):
python -m venv /tmp/tpypi
/tmp/tpypi/bin/pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ panelclv
/tmp/tpypi/bin/python -c "from panelclv.models import MultinomialLSTMModel; print('ok')"
```

## D. Publish to PyPI — pick ONE auth method

**Option 1 — API token (quickest one-off):**
1. Account at https://pypi.org → Account settings → **API tokens** → create a token.
2. Upload (username `__token__`, password = the token):

   ```bash
   pipx run twine upload dist/*
   ```

**Option 2 — Trusted Publishing via GitHub Actions (recommended; no stored secrets):**
PyPI verifies a GitHub OIDC identity, so there's no token to leak.
1. On PyPI: project → **Publishing** → add a *trusted publisher*
   (repo `HuberPablo/panelclv`, workflow `release.yml`, environment `pypi`).
2. Add `.github/workflows/release.yml` that builds and runs
   `pypa/gh-action-pypi-publish` on a tagged release. Then publishing = pushing a tag.

## E. Tag the release

```bash
git tag v0.1.0 && git push origin v0.1.0
# (optional) create a GitHub Release from the tag for a changelog
pip install panelclv          # final sanity check from real PyPI, in a clean env
```

## Before the first real upload — check these

- **Name availability:** verify `https://pypi.org/project/panelclv/` returns 404
  (free). If taken, change `project.name` in `pyproject.toml`.
- **Version bumps are mandatory:** PyPI rejects re-uploading an existing version. Bump
  `version` in `pyproject.toml` (or push a new tag with Option 2) for every release.
- **Heavy deps:** `dependencies` includes `torch` by design. `pip install panelclv`
  leaves an already-installed torch (ROCm / CUDA / CPU) untouched and only pulls the
  default build into an env that has none — existing environments are unaffected.
- **`lifetimes`** (the MLE Pareto/NBD dep) is an older package; confirm it resolves on
  your target Python (esp. 3.12) before advertising that version.
- **README is the PyPI long-description** — make sure its top heading and quickstart are
  current before uploading, since that's the project's public page.
