#!/usr/bin/env bash
# reap_finished.sh [INTERVAL_SEC] — destroy each worker once its shard is finished
# AND its results are provably local. Nothing else stops a box billing (§8).
#
# This is the job supervise.py used to do, rewritten because its version is what lost
# shard 7/8: it pulled from `Studies/<grid>__<Model>/`, the path a grid with no arm axis
# uses, while the workers write `Studies/<grid>__<Model>__<arm>/`. The rsync moved
# nothing, `len(list(local.glob("*__*")))` counted 0, and it destroyed the box anyway —
# the count was printed, never checked. 107 suites went with it.
#
# So the rule here is: the destroy is gated on a comparison, not on a pull returning.
# Every results.csv the WORKER holds must exist locally, by path, before the box dies.
# A mismatch leaves the box running and says so; a box that bills for another cycle is
# cheap next to a shard that has to be re-run.
#
# Read-only until the gate passes. Boxes that are still training are left alone.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="$HOME/venvs/panelclv/bin:$PATH"
INTERVAL="${1:-300}"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -n -i "$KEY" -o StrictHostKeyChecking=accept-new
     -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast" -o BatchMode=yes -o ConnectTimeout=15)
SSH_E="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vast -o BatchMode=yes -o ConnectTimeout=15"

log() { echo "[$(date +%H:%M:%S)] $*"; }

while true; do
  mapfile -t rows < <(vastai show instances --raw 2>/dev/null | python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
for i in d:
    ip=(i.get('public_ipaddr') or '').strip(); p=(i.get('ports') or {}).get('22/tcp') or []
    print(i['id'], ip or '-', (p[0].get('HostPort') if p else '-'), i.get('actual_status') or '?')
")
  if [ "${#rows[@]}" -eq 0 ]; then
    log "no instances left — nothing to reap"
    sleep "$INTERVAL"; continue
  fi

  for row in "${rows[@]}"; do
    set -- $row; id=$1; ip=$2; port=$3; status=$4

    # An exited box cannot be pulled from, and restarting one re-pulls the image and
    # bills bandwidth per GB (F14). Report rather than destroy: its disk may still hold
    # suites, and that is a decision for a human, not for a timer.
    if [ "$status" = "exited" ]; then
      log "$id EXITED — cannot pull, NOT destroying. Inspect or destroy by hand."
      continue
    fi
    [ "$ip" = "-" ] || [ "$port" = "-" ] && continue

    # `.shard_exit` is written by start_shard.sh when the trainer returns; its absence
    # means still running (or never started). Bracket the pgrep pattern or the remote
    # shell matches its own command line (F13).
    state=$("${SSH[@]}" -p "$port" "root@$ip" '
      if [ -f /root/.shard_exit ]; then echo "exit=$(cat /root/.shard_exit)";
      elif pgrep -f "[r]un_pnbd_grid" >/dev/null 2>&1; then echo running;
      else echo none; fi' 2>/dev/null | tail -1)
    case "$state" in
      running|none|"") continue ;;
    esac

    code="${state#exit=}"
    if [ "$code" != "0" ]; then
      # A non-zero exit is a code bug, not hardware — re-renting reproduces it. Keep the
      # log locally so the box does not have to stay alive to be read.
      log "$id shard CRASHED exit=$code — copying shard.log, leaving box up for inspection"
      rsync -az --timeout=60 -e "$SSH_E -p $port" \
        "root@$ip:/root/shard.log" "VastAI/state/crashed_${id}_shard.log" 2>/dev/null \
        && log "  saved VastAI/state/crashed_${id}_shard.log"
      continue
    fi

    log "$id shard finished (exit=0) — pulling before anything else"
    rsync -az --partial --timeout=180 -e "$SSH_E -p $port" \
      "root@$ip:/root/panelclv/Studies/" "Studies/" 2>/dev/null || {
        log "  pull FAILED — box left running, will retry next cycle"; continue; }

    # The gate. Every results.csv the worker holds must be present locally, by path.
    remote_paths=$("${SSH[@]}" -p "$port" "root@$ip" \
      'cd /root/panelclv/Studies 2>/dev/null && find . -name results.csv | sed "s|^\./||" | sort' 2>/dev/null)
    if [ -z "$remote_paths" ]; then
      log "  worker reports NO results at all — not destroying, this needs a human"
      continue
    fi
    missing=0; total=0
    while IFS= read -r p; do
      [ -z "$p" ] && continue
      total=$((total+1))
      [ -f "Studies/$p" ] || { missing=$((missing+1)); [ "$missing" -le 3 ] && log "  MISSING locally: $p"; }
    done <<< "$remote_paths"

    if [ "$missing" -ne 0 ]; then
      log "  $missing of $total suites NOT local — box KEPT ALIVE (this is the shard-7 failure)"
      continue
    fi

    log "  verified all $total suites present locally — destroying $id"
    if vastai destroy instance "$id" -y >/dev/null 2>&1; then
      log "  destroyed $id"
    else
      log "  destroy FAILED for $id — retry by hand: vastai destroy instance $id"
    fi
  done

  live=$(vastai show instances --raw 2>/dev/null | python -c \
    "import json,sys; d=json.load(sys.stdin); print(len(d), round(sum(i.get('dph_total') or 0 for i in d),4))" 2>/dev/null)
  log "fleet: ${live:-?} (instances, \$/hr) | transformer $(find Studies/seasonal_4x4x10__Transformer__* -name results.csv 2>/dev/null | wc -l)/960"
  sleep "$INTERVAL"
done
