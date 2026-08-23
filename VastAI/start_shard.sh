#!/usr/bin/env bash
#
# start_shard.sh <SSH_HOST> <SSH_PORT> <GRID> <MODEL> <SHARD> — put one worker to work.
#
# vast_launch.sh gets a box to `running`; this takes it from there to actually
# training: wait out provisioning, push the grid's data, start the shard detached.
#
# Usage:
#   ./VastAI/start_shard.sh 1.2.3.4 40719 seasonal_4x4x10 transformer 3/7
#
# The shard runs under nohup+setsid rather than in this SSH session, because the
# session ends when this script returns and a child of it would die with it. Progress
# goes to /root/shard.log on the worker; completion is /root/.shard_done.

set -euo pipefail

HOST="${1:?usage: start_shard.sh <HOST> <PORT> <GRID> <MODEL> <SHARD>}"
PORT="${2:?missing PORT}"
GRID="${3:?missing GRID}"
MODEL="${4:?missing MODEL}"
SHARD="${5:?missing SHARD}"

KEY="${VAST_KEY:-$HOME/.ssh/id_ed25519}"
REPO_DIR=/root/panelclv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DATA="$(dirname "$SCRIPT_DIR")/Datasets/Synthetic/$GRID"

# Rented boxes reuse IPs and get a fresh host key each time, so a changed fingerprint
# is expected rather than suspicious — keep them out of the real known_hosts.
SSH_OPTS=(-i "$KEY" -p "$PORT"
          -o StrictHostKeyChecking=accept-new
          -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast"
          -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

say() { echo "[$HOST:$PORT $MODEL $SHARD] $*"; }

# --- 1. wait for provisioning ------------------------------------------------
# vast_onstart.sh only touches .onstart_done if every step succeeded, so its absence
# means we would be racing a pip install still in flight.
say "waiting for /root/.onstart_done (image pull + pip install take several minutes)"
for attempt in $(seq 1 80); do
    if ssh "${SSH_OPTS[@]}" "root@$HOST" 'test -f /root/.onstart_done' 2>/dev/null; then
        say "provisioned"
        break
    fi
    [ "$attempt" = 80 ] && { say "FATAL: never finished provisioning"; exit 1; }
    sleep 15
done

# --- 2. push the data --------------------------------------------------------
# Datasets/ is gitignored, so the clone has no panels. --partial resumes a dropped
# transfer, and a re-run skips what is already there (VastAI/Rules.md §3).
[ -d "$LOCAL_DATA" ] || { say "FATAL: no local data at $LOCAL_DATA"; exit 1; }
say "pushing $(du -sh "$LOCAL_DATA" | cut -f1) of panels"
ssh "${SSH_OPTS[@]}" "root@$HOST" "mkdir -p $REPO_DIR/Datasets/Synthetic"
rsync -az --partial -e "ssh ${SSH_OPTS[*]}" \
      "$LOCAL_DATA/" "root@$HOST:$REPO_DIR/Datasets/Synthetic/$GRID/"

# --- 3. start the shard ------------------------------------------------------
say "starting shard"
ssh "${SSH_OPTS[@]}" "root@$HOST" bash -s <<REMOTE
set -euo pipefail
cd $REPO_DIR
# onstart installed the package into whichever interpreter the image ships; find the
# same one rather than assuming \`python\` is on a non-interactive shell's PATH.
for candidate in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
    [ -x "\$candidate" ] && PY="\$candidate" && break
done
rm -f /root/.shard_done
setsid nohup bash -c "
    \$PY scripts/run_pnbd_grid.py --grid $GRID --model $MODEL --shard $SHARD
    touch /root/.shard_done
" > /root/shard.log 2>&1 &
sleep 2
head -5 /root/shard.log || true
REMOTE

say "running — tail with: ssh ${SSH_OPTS[*]} root@$HOST 'tail -f /root/shard.log'"
