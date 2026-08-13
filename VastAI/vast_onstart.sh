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

BRANCH=main
REPO_DIR=/root/panelclv
# 8080 is what vast's own "connect" button suggests forwarding, so keeping it
# here means the command the UI hands you already works.
JUPYTER_PORT=8080

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

# --- JupyterLab ---------------------------------------------------------------
# Not every image ships a notebook server (the plain pytorch/pytorch images do
# not), so install one into the SAME interpreter the package went into —
# otherwise `import panelclv` fails inside the notebook.
#
# Bound to 127.0.0.1, NOT 0.0.0.0: `ssh -L 8080:localhost:8080` connects from the
# instance's own loopback, so the tunnel reaches it either way, and a loopback
# bind keeps the server off the public internet.
#
# onstart replays on every instance start, so skip if a server is already up.
if ! pgrep -f "jupyter.*--port=?$JUPYTER_PORT" >/dev/null 2>&1; then
    "$PY" -m pip install -q jupyterlab

    # Random per-instance token, stored on disk rather than baked into this file
    # (which lives in a public repo). jupyter-server reads JUPYTER_TOKEN from the
    # environment, which is stable across the ServerApp/IdentityProvider flag
    # rename in jupyter-server 2.
    TOKEN="$("$PY" -c 'import secrets; print(secrets.token_hex(16))')"
    echo "$TOKEN" > /root/jupyter_token

    # setsid + nohup so the server outlives this script (onstart must exit).
    # --notebook-dir roots the file browser at the repo, so no symlink is needed.
    JUPYTER_TOKEN="$TOKEN" setsid nohup "$PY" -m jupyter lab \
        --ip 127.0.0.1 --port "$JUPYTER_PORT" --allow-root --no-browser \
        --notebook-dir "$REPO_DIR" > /root/jupyter.log 2>&1 &
    sleep 3
fi

touch /root/.onstart_done
cat <<EOF

=== provisioning complete: $REPO_DIR ($BRANCH) ===

JupyterLab is running on the instance's loopback. From your laptop:

  ssh -p <PORT> root@<HOST> -L ${JUPYTER_PORT}:localhost:${JUPYTER_PORT}

then open  http://localhost:${JUPYTER_PORT}/lab?token=$(cat /root/jupyter_token 2>/dev/null)

The token is also in /root/jupyter_token, and the server log in /root/jupyter.log.
EOF
