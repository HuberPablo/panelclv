#!/usr/bin/env bash
# budget_watchdog.sh <MAX_SPEND_USD> — destroy the whole fleet once the run has cost
# that much, whatever state it is in.
#
# Rules.md section 8: a worker bills from start until destroy, and that is the main way
# money is lost here. The reaper destroys a box when its slice FINISHES; nothing else
# bounds a box that hangs, and a hung fleet at $0.73/hr empties a 17 CHF budget in a
# day. This is the unconditional bound.
#
# It integrates spend the same way the bill does: fleet $/hr sampled each cycle, times
# elapsed hours. It does NOT read invoices — those lag, and a watchdog that waits for
# billing data is not a watchdog. Bandwidth is a separate meter it cannot see (F14), so
# the real bill lands somewhat above this figure; the ceiling is set with that in mind.
#
# Pulling comes first, always. A box destroyed with unpulled suites is a shard to re-run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
# Both orchestrators this has run from keep their venv in a different place.
export PATH="$HOME/venvs/panelclv/bin:$HOME/thesis-agent/venv/bin:$PATH"
MAX="${1:?usage: budget_watchdog.sh <max_spend_usd>}"
INTERVAL=300
SPENT=0
LAST=$(date +%s)
KEY="$HOME/.ssh/id_ed25519"
SSH_E="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vast -o BatchMode=yes -o ConnectTimeout=15"

log() { echo "[$(date +%H:%M:%S)] $*"; }

while true; do
  DPH=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
print(sum(i.get('dph_total') or 0 for i in d))
" 2>/dev/null || echo 0)
  NOW=$(date +%s)
  SPENT=$(python3 -c "print($SPENT + $DPH * ($NOW - $LAST) / 3600.0)")
  LAST=$NOW
  N=$(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
  log "fleet=$N \$$DPH/hr, spent so far \$$(printf '%.3f' "$SPENT") of \$$MAX"

  if [ "$N" = "0" ]; then log "fleet empty — watchdog exiting"; exit 0; fi

  if python3 -c "import sys; sys.exit(0 if $SPENT >= $MAX else 1)"; then
    log "BUDGET REACHED — pulling everything, then destroying the fleet"
    mapfile -t rows < <(vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys
for i in json.load(sys.stdin):
    ip=(i.get('public_ipaddr') or '').strip(); p=(i.get('ports') or {}).get('22/tcp') or []
    host, port = (ip, p[0].get('HostPort')) if (ip and p) else (i.get('ssh_host'), i.get('ssh_port'))
    print(i['id'], host or '-', port or '-')
")
    for row in "${rows[@]}"; do
      set -- $row; id=$1; host=$2; port=$3
      [ "$host" = "-" ] && continue
      log "  final pull from $id"
      rsync -az --partial --timeout=180 -e "$SSH_E -p $port" \
        "root@$host:/root/panelclv/Studies/" "Studies/" 2>/dev/null \
        || log "  pull from $id FAILED"
    done
    for row in "${rows[@]}"; do
      set -- $row; id=$1
      # F24: -y, or it prompts, prints Aborted. and destroys nothing while it bills.
      vastai destroy instance -y "$id" >/dev/null 2>&1
      log "  destroy $id issued"
    done
    sleep 15
    LEFT=$(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
    # F24 again: verify by re-reading the fleet, never by trusting the destroy output.
    log "instances left after destroy: $LEFT"
    exit 0
  fi
  sleep "$INTERVAL"
done
