#!/usr/bin/env bash
#
# vast_onstart.sh — provision a vast.ai instance to train panelclv models.
#
# Paste this file into the UI's "on-start script" box, or pass it to
# `vastai create instance ... --onstart vast_onstart.sh`. It runs as root on
# first boot, before you can connect.
#
# Progress goes to /root/onstart.log (`vastai logs <ID>`, or tail it once in).
# Wait for /root/.onstart_done before starting a run — it is only touched if
# every step below succeeded, so its absence means you would be racing a pip
# install still in flight.
#
# Datasets/ is gitignored and is NOT cloned; upload or regenerate it separately.

set -euo pipefail

BRANCH=covariate-standardization
REPO_DIR=/root/panelclv

exec > >(tee -a /root/onstart.log) 2>&1
echo "=== panelclv onstart: $(date -u '+%F %T UTC') ==="

# onstart runs in a non-interactive shell: ~/.bashrc is never sourced, so the
# image's conda/venv activation has not happened and `python` may be off PATH
# entirely. Probe the layouts vast's PyTorch images actually use.
for candidate in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
    [ -x "$candidate" ] && PY="$candidate" && break
done
: "${PY:?no python interpreter found}"
export PATH="$(dirname "$PY"):$PATH"
echo "python: $PY ($("$PY" --version 2>&1))"

# Blocking on purpose: a six-hour sweep that silently ran on CPU costs the whole
# rental. Failing here means .onstart_done is never touched.
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "FATAL: no CUDA device visible — wrong image or CPU-only host"; exit 1; }
echo "gpu: $("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))')"

# git to clone, rsync for the data push/pull (needed on BOTH ends), tmux to keep
# runs alive across SSH drops.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git rsync tmux >/dev/null

# vast replays onstart on every instance start, so the clone must be idempotent.
# Deliberately a fast-forward pull, not `reset --hard`: edits made on the box are
# kept, and a diverged checkout warns instead of being silently destroyed.
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" \
        || echo "WARNING: could not fast-forward — keeping the existing checkout"
else
    git clone --quiet --branch "$BRANCH" https://github.com/HuberPablo/panelclv.git "$REPO_DIR"
fi
echo "HEAD: $(git -C "$REPO_DIR" log -1 --oneline)"

# Editable, and deliberately NOT --no-deps / --force-reinstall: pyproject lists
# torch unpinned, so pip sees the image's CUDA build as already satisfying it and
# leaves it alone. Forcing a reinstall pulls a generic wheel and breaks the
# numpy/scipy ABI the image was built against.
"$PY" -m pip install -q -e "$REPO_DIR"

touch /root/.onstart_done
echo "=== provisioning complete: $REPO_DIR ($BRANCH) ==="
