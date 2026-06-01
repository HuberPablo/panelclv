# Publishing this package

The package is build-ready: `pyproject.toml` is valid, the single top-level
`panelclv` package (with `panelclv.models`, `panelclv.data_preparation`,
`panelclv.configs` subpackages) builds into a correct wheel that ships nothing into
`site-packages` except `panelclv/`, and `LICENSE` / `README.md` are in place. What
remains needs your accounts / tooling and is listed below as copy-paste steps.

The repo's `origin` already points at the real GitHub repo
(`https://github.com/HuberPablo/panelclv.git`), so there is no repo-creation step —
just push the branch.

## A. Push to GitHub

The packaging work lives on the `pypi-restructure` branch. Push it and open a PR (or
push straight to `main` if you prefer):

```bash
cd /home/virthian/Desktop/Thesis/Package_Notebook_refactored
git push -u origin pypi-restructure       # then open a PR into main on GitHub
# or, to publish the history directly on main:
# git checkout main && git merge pypi-restructure && git push origin main
```

## B. Make it pip-installable / publish to PyPI

### Already works today — local install
From the repo root, into any environment you control (NOT the protected
`thesis_rocm` venv):

```bash
pip install -e .            # editable install for development
# or
pip install .              # regular install
```

Then `from panelclv.models import ...` works from anywhere.

### Build the distribution artifacts
The `thesis_rocm` venv already has `build` + setuptools 82, so you can build without
installing anything new (do NOT modify that venv):

```bash
cd /home/virthian/Desktop/Thesis/Package_Notebook_refactored
/home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/python -m build --no-isolation
# writes dist/panelclv-0.1.0-py3-none-any.whl and dist/panelclv-0.1.0.tar.gz
```

To validate metadata you need `twine` (not in `thesis_rocm`). Install it into a
throwaway/other env and run:

```bash
python -m pip install --upgrade twine      # in a throwaway/other env
twine check dist/*
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
- **Name availability:** `panelclv` is currently **free** on PyPI (verified — the
  project page returns 404). If someone registers it before you upload, change
  `project.name` in `pyproject.toml`.
- **Heavy deps:** `dependencies` includes `torch` (by design). `pip install panelclv`
  leaves an already-installed torch (ROCm / CUDA / CPU) untouched and only pulls the
  default build into an env that has none — so existing environments are unaffected.
- **Notebooks/datasets are not packaged** (only the `panelclv` package ships), which is
  correct — keep large data out of the distribution.
- Bump `version` in `pyproject.toml` for every release (PyPI rejects re-uploads of an
  existing version).
