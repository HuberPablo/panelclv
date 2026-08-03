#!/usr/bin/env bash
#
# vast_launch.sh <OFFER_ID> — rent a vast.ai machine and drive it to `running`.
#
# Exists because `vastai create instance` alone is not enough:
#   1. Offers go stale fast. Creating against a taken offer returns
#      {'success': False, ...} with a contract id that vast then silently reaps —
#      which looks like "it worked" followed by a 404 on the very next command.
#   2. `create` only ALLOCATES. The container does not boot, the image is not
#      pulled, and --onstart never runs until you separately `start` it.
#   3. An instance created before your SSH key existed has no key injected.
#
# This script does the pre-flight check, creates, attaches the key, starts, and
# polls until the container is actually up — then prints the ssh command.
#
# Usage:
#   ./vast_launch.sh 39787605
#
# Override any of these by exporting them first:
#   VAST_IMAGE    docker image                (default: vast-native pytorch)
#   VAST_DISK     disk GB                     (default: 40)
#   VAST_KEY      private key path            (default: ~/.ssh/id_ed25519)
#   VAST_ONSTART  provisioning script         (default: VastAI/vast_onstart.sh)

set -euo pipefail

OFFER="${1:?usage: vast_launch.sh <OFFER_ID>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VAST_IMAGE:-vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312}"
DISK="${VAST_DISK:-40}"
KEY="${VAST_KEY:-$HOME/.ssh/id_ed25519}"
ONSTART="${VAST_ONSTART:-$SCRIPT_DIR/vast_onstart.sh}"

VASTAI="$(command -v vastai || echo /home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/vastai)"
PY="$(command -v python || command -v python3)"

# vast's CLI prints Python-dict repr in some paths and JSON in others, so parse
# leniently: try JSON, fall back to ast.literal_eval.
parse() {
    "$PY" -c "
import sys, json, ast
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    try:
        d = ast.literal_eval(raw.strip().splitlines()[-1])
    except Exception:
        d = {}
print(d.get('$1', '') if isinstance(d, dict) else '')
"
}

# --- 0. sanity: key present and registered -----------------------------------
if [ ! -f "${KEY}.pub" ]; then
    echo "FATAL: no public key at ${KEY}.pub"
    echo "  fix: ssh-keygen -t ed25519 -C vast -f $KEY -N ''"
    exit 1
fi
echo "[0/4] key: ${KEY}.pub"

# --- 1. (no pre-flight availability check) ------------------------------------
# There is deliberately no "is this offer still live?" gate here. `id=<N>` is not
# a filterable field in vast's search API — it returns 0 rows for offers that are
# demonstrably available — so any such check rejects every valid offer. Staleness
# is instead detected after the fact: if the offer was taken, vast reaps the
# instance within seconds and the poll loop in step 4 reports GONE.
echo "[1/4] (skipping availability pre-check — vast has no queryable offer-id filter)"

# --- 2. create ----------------------------------------------------------------
echo "[2/4] creating instance (image: ${IMAGE}, disk: ${DISK}G)"
OUT="$("$VASTAI" create instance "$OFFER" \
        --image "$IMAGE" --disk "$DISK" --ssh --direct \
        --onstart "$ONSTART" 2>&1)"
echo "  $OUT"
ID="$(printf '%s' "$OUT" | grep -oE "'new_contract':\s*[0-9]+" | grep -oE '[0-9]+' | tail -1)"
if [ -z "$ID" ]; then
    echo "FATAL: could not read an instance id from the create response"
    exit 1
fi
echo "  instance id: ${ID}"

# --- 3. attach key + start ----------------------------------------------------
# Order matters: attach before start so the key is present at first boot.
echo "[3/4] attaching ssh key and starting..."
"$VASTAI" attach ssh "$ID" "$(cat "${KEY}.pub")" 2>&1 | sed 's/^/  /' || true
"$VASTAI" start instance "$ID" 2>&1 | sed 's/^/  /'

# --- 4. poll until the container is genuinely running -------------------------
# `actual_status` can read 'loading' while cur_state is still 'stopped', so gate
# on cur_state — that is the field that means the container exists and is up.
echo "[4/4] waiting for cur_state=running (image pull can take several minutes)"
for attempt in $(seq 1 80); do
    STATE="$("$VASTAI" show instances --raw 2>/dev/null | "$PY" -c "
import json, sys
try:
    rows = [r for r in json.load(sys.stdin) if str(r.get('id')) == '${ID}']
except Exception:
    rows = []
if not rows:
    print('GONE|-|-')
else:
    r = rows[0]
    print(f\"{r.get('cur_state')}|{r.get('actual_status')}|{r.get('next_state')}\")
")"
    CUR="${STATE%%|*}"
    echo "  [${attempt}] cur_state=${STATE}"

    if [ "$CUR" = "running" ]; then
        echo
        echo "=== RUNNING ==="
        "$VASTAI" show instances --raw | "$PY" -c "
import json, sys
r = [r for r in json.load(sys.stdin) if str(r.get('id')) == '${ID}'][0]
print(f\"instance : {r['id']}\")
print(f\"gpu/cpu  : {r.get('gpu_name')} | {r.get('cpu_name')}\")
print(f\"cost     : \${r.get('dph_total'):.4f}/hr\")
print()
print(f\"ssh -i ${KEY} -p {r['ssh_port']} root@{r['ssh_host']}\")
print()
print(f\"logs     : $VASTAI logs {r['id']}\")
print(f\"destroy  : $VASTAI destroy instance {r['id']}\")
"
        exit 0
    fi

    if [ "$CUR" = "GONE" ]; then
        echo
        echo "INSTANCE ${ID} WAS REAPED — vast could not place it on that host."
        echo "  Almost always means the offer was taken between search and create."
        echo "  Re-run: python $SCRIPT_DIR/vast_search.py  and try the next row."
        exit 1
    fi
    sleep 15
done

echo "TIMED OUT after ~20 min still not running. Check: $VASTAI show instances"
exit 1
