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
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
# Both orchestrators this has run from keep their venv in a different place.
export PATH="$HOME/venvs/panelclv/bin:$HOME/thesis-agent/venv/bin:$PATH"
INTERVAL="${1:-300}"
GRID="${2:-seasonal_4x4x10}"   # which grid to reconcile against once the fleet empties
# The completeness gate. reconcile_grid.py expands a GridSpec, so a real-panel
# ablation overrides this with its own equivalent:
#   RECONCILE_CMD="scripts/run_real_panel_arms.py --check-complete"
RECONCILE_CMD="${RECONCILE_CMD:-scripts/reconcile_grid.py --grid $GRID}"
# A box older than this with no trainer and no exit status, while no launcher is running,
# is retired and queued for replacement (F32). Longer than a slow image pull plus the
# launcher's own start-up verdict, so it only ever catches what the launch gate missed.
IDLE_MINUTES="${IDLE_MINUTES:-35}"
mkdir -p VastAI/state/worker_logs
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -n -i "$KEY" -o StrictHostKeyChecking=accept-new
     -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast" -o BatchMode=yes -o ConnectTimeout=15)
SSH_E="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vast -o BatchMode=yes -o ConnectTimeout=15"

log() { echo "[$(date +%H:%M:%S)] $*"; }

while true; do
  # F22: an instance gets EITHER a direct ip=/port= or an sshN.vast.ai proxy endpoint.
  # Emitting the direct form alone made every proxy-only box invisible here, so it was
  # never pulled from and never destroyed — it just billed. Prefer direct, fall back to
  # the proxy.
  mapfile -t rows < <(vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys,time
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
for i in d:
    ip=(i.get('public_ipaddr') or '').strip(); p=(i.get('ports') or {}).get('22/tcp') or []
    host, port = (ip, p[0].get('HostPort')) if (ip and p) else (i.get('ssh_host'), i.get('ssh_port'))
    age = int((time.time() - float(i.get('start_date') or time.time())) / 60)
    print(i['id'], host or '-', port or '-', i.get('actual_status') or '?', age)
")
  if [ "${#rows[@]}" -eq 0 ]; then
    log "no instances left — nothing to reap"
    sleep "$INTERVAL"; continue
  fi

  for row in "${rows[@]}"; do
    set -- $row; id=$1; ip=$2; port=$3; status=$4; age=${5:-0}

    # An exited box cannot be pulled from, and restarting one re-pulls the image and
    # bills bandwidth per GB (F14). Report rather than destroy: its disk may still hold
    # suites, and that is a decision for a human, not for a timer.
    if [ "$status" = "exited" ]; then
      log "$id EXITED — cannot pull, NOT destroying. Inspect or destroy by hand."
      continue
    fi
    [ "$ip" = "-" ] || [ "$port" = "-" ] && continue

    # `.shard_exit` is written by start_shard.sh when the trainer returns; its absence
    # means still running (or never started). The trainer is counted by its python command line, anchored at the start so the
    # probe's own ssh command cannot match (F13) and any `scripts/run_*` runner counts
    # without this list having to name it.
    state=$("${SSH[@]}" -p "$port" "root@$ip" '
      if [ -f /root/.shard_exit ]; then echo "exit=$(cat /root/.shard_exit)";
      elif [ "$(ps -eo cmd | grep -c "^/[^ ]*python.* scripts/run_")" -gt 0 ]; then echo running;
      else echo none; fi' 2>/dev/null | tail -1)
    # A box whose launcher is still in flight and has not yet confirmed it training
    # belongs to that launcher: acting on it here raced the launcher's own verdict,
    # destroyed the box under it and queued the worker for replacement twice.
    if [ ! -f "VastAI/state/started/$id" ] && ! flock -n VastAI/state/add_workers.lock true; then
      continue
    fi
    case "$state" in
      running|"") continue ;;
      none)
        # Rented, not training, and no exit status: the state nothing else owns (F32).
        # While a launcher holds its lock it is still rendering this box's verdict, so
        # the start-up window is left to it; with no launcher running, a box this old
        # and idle is a start-up failure the launch gate missed.
        if [ "$age" -lt "$IDLE_MINUTES" ] || ! flock -n VastAI/state/add_workers.lock true; then
          continue
        fi
        idx=$(awk -v id="$id" '$1==id {print $2}' VastAI/state/assignments.txt 2>/dev/null | tail -1)
        log "$id IDLE for ${age} min with no trainer and no exit status — retiring (F32)"
        rsync -az --partial --timeout=180 -e "$SSH_E -p $port" \
          "root@$ip:/root/panelclv/Studies/" "Studies/" 2>/dev/null
        for f in onstart.log shard.log; do
          rsync -az --timeout=60 -e "$SSH_E -p $port" "root@$ip:/root/$f" \
            "VastAI/state/worker_logs/idle_${id}_$f" 2>/dev/null
        done
        vastai destroy instance "$id" -y >/dev/null 2>&1 && log "  destroyed $id"
        echo "$id ${idx:-?} idle-${age}min" >> VastAI/state/needs_replacement.txt
        continue ;;
    esac

    code="${state#exit=}"
    if [ "$code" != "0" ]; then
      # A non-zero exit is a code bug, not hardware — re-renting reproduces it, so the
      # replacement entry says so. Everything the box can tell us is copied first (its
      # finished suites and its log), after which keeping it up only bills: an idle box
      # waiting for a human is F32 again.
      log "$id shard CRASHED exit=$code — pulling suites and shard.log, then destroying"
      rsync -az --partial --timeout=180 -e "$SSH_E -p $port" \
        "root@$ip:/root/panelclv/Studies/" "Studies/" 2>/dev/null
      if rsync -az --timeout=60 -e "$SSH_E -p $port" \
           "root@$ip:/root/shard.log" "VastAI/state/crashed_${id}_shard.log" 2>/dev/null; then
        log "  saved VastAI/state/crashed_${id}_shard.log"
        idx=$(awk -v id="$id" '$1==id {print $2}' VastAI/state/assignments.txt 2>/dev/null | tail -1)
        vastai destroy instance "$id" -y >/dev/null 2>&1 && log "  destroyed $id"
        echo "$id ${idx:-?} crashed-exit=$code(fix-the-code-first)" >> VastAI/state/needs_replacement.txt
      else
        log "  could not copy shard.log — box left up for one more cycle"
      fi
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
  n_live=${live%% *}

  # The gate above only ever compares a worker against ITSELF: every results.csv that
  # box holds must be local. Nothing in it knows how many suites the GRID owes, which is
  # how a run ends with an empty fleet, every shard exit=0, and nine suites missing
  # (F19). Once the fleet is empty there is no worker left to ask, so ask the grid.
  if [ "${n_live:-1}" = "0" ]; then
    if python $RECONCILE_CMD > VastAI/state/reconcile.txt 2>&1; then
      log "fleet empty and $GRID reconciles — the run is complete"
    else
      log "fleet empty but $GRID is INCOMPLETE — see VastAI/state/reconcile.txt"
      grep -E "^  SHORT|^INCOMPLETE" VastAI/state/reconcile.txt | while IFS= read -r l; do
        log "  $l"; done
    fi
  fi

  log "fleet: ${live:-?} (instances, \$/hr)"
  sleep "$INTERVAL"
done
