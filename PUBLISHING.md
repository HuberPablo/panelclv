# Publishing this package

The package is build-ready: `pyproject.toml` is valid, the three importable
packages (`Models`, `Data_preparation`, `configs`) build into a correct wheel, and
`LICENSE` / `README.md` are in place. What remains needs your accounts / tooling and
is listed below as copy-paste steps.

This repo currently has a placeholder `origin` pointing at a local bundle
(`packagingrefactor.bundle`) — that is how the remote build session handed results
back. Part A repoints it at a real GitHub repo.

## A. Put it on GitHub (fresh repo)

`gh` (GitHub CLI) is **not installed** on this machine, so create the empty repo in
the browser first:

1. Go to https://github.com/new → create an **empty** repo (no README/license/
   .gitignore — this repo already has them). Note its URL, e.g.
   `https://github.com/<you>/customer-base-forecasting.git`.

2. Repoint this repo's remote and push the branch:

   ```bash
   cd /home/virthian/Desktop/Thesis/Package_Notebook_refactored
   git remote rename origin bundle          # keep the bundle remote, out of the way
   git remote add origin https://github.com/<you>/customer-base-forecasting.git
   git push -u origin packaging-refactor
   ```

3. (Optional) make `packaging-refactor` the default branch on GitHub, or open a PR
   and merge it into `main`/`master`.

   If you prefer the published history to live on `main`:
   ```bash
   git branch -m packaging-refactor main
   git push -u origin main
   ```

> Tip: to install `gh` later and automate step 1:
> `sudo apt install gh && gh auth login && gh repo create customer-base-forecasting --private --source=. --push`

## B. Make it pip-installable / publish to PyPI

### Already works today — local install
From the repo root, into any environment you control (NOT the protected
`thesis_rocm` venv):

```bash
pip install -e .            # editable install for development
# or
pip install .              # regular install
```

Then `from Models import ...` works from anywhere.

### Build the distribution artifacts
`build` and `twine` are not in the `thesis_rocm` venv (and it must not be modified).
Use any other environment:

```bash
python -m pip install --upgrade build twine     # in a throwaway/other env
cd /home/virthian/Desktop/Thesis/Package_Notebook_refactored
python -m build                                  # writes dist/*.whl and dist/*.tar.gz
twine check dist/*                               # validate metadata
```

### Publish to PyPI
1. Make an account at https://pypi.org (and https://test.pypi.org for a dry run).
2. Create an API token (Account settings → API tokens).
3. Upload:

   ```bash
   twine upload --repository testpypi dist/*     # dry run on TestPyPI first
   twine upload dist/*                            # the real thing
   ```

### Before the first real PyPI upload — check these
- **Name availability:** `panelclv` must be free on PyPI. If taken,
  change `project.name` in `pyproject.toml`.
- **Heavy deps:** `dependencies` includes `torch`. PyPI installs the default CPU/CUDA
  build; this repo's dev env is ROCm. Consider documenting that users pick their own
  torch build, or move `torch` to an optional extra so `pip install` doesn't pull an
  unwanted wheel.
- **Notebooks/datasets are not packaged** (only the three source packages ship), which
  is correct — keep large data out of the distribution.
- Bump `version` in `pyproject.toml` for every release (PyPI rejects re-uploads of an
  existing version).
