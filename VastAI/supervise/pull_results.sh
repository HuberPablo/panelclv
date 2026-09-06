#!/usr/bin/env bash
# pull_results.sh [INTERVAL_SEC] — copy finished suites off every worker, continuously.
#
# supervise.py pulls a shard's results when the shard FINISHES. Anything that ends a
# worker early — the --max-hours watchdog, a budget stop, a host reclaiming its GPU —
# therefore discards every suite that worker had already completed. On a job whose
# shards run tens of hours that is most of the run.
#
# This makes any ending non-destructive. Each cycle rsyncs every worker's Studies/ tree
# into the matching local one. That is a plain copy and needs no merge step, because
# each (model, arm) has its own root and every worker writes disjoint
# <combo>__<dataset>/ directories inside it (VastAI/Rules.md §4) — the property the
# whole split was designed around.
#
# Safe to run beside supervise.py: it only reads from workers, and rsync into a tree
# supervise.py may also write is idempotent (identical bytes, --ignore-existing on the
# suite directories that are already complete).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export PATH="$HOME/venvs/panelclv/bin:$PATH"
INTERVAL="${1:-1200}"
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vast -o BatchMode=yes -o ConnectTimeout=15"

while true; do
  mapfile -t rows < <(vastai show instances --raw 2>/dev/null | python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
for i in d:
    ip=(i.get('public_ipaddr') or '').strip(); p=(i.get('ports') or {}).get('22/tcp') or []
    if ip and p: print(i['id'], ip, p[0].get('HostPort'))
")
  before=$(find Studies -name results.csv 2>/dev/null | wc -l)
  for row in "${rows[@]:-}"; do
    [ -z "$row" ] && continue
    set -- $row; id=$1; ip=$2; port=$3
    # -n on the probe: ssh in a loop otherwise eats the worklist from stdin (F6).
    ssh -n -i "$KEY" -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast" -o BatchMode=yes \
        -o ConnectTimeout=15 -p "$port" "root@$ip" \
        'test -d /root/panelclv/Studies' 2>/dev/null || continue
    rsync -az --partial --timeout=120 -e "$SSH_OPTS -p $port" \
        "root@$ip:/root/panelclv/Studies/" "Studies/" 2>/dev/null \
        || echo "$(date +%H:%M) pull from $id failed — will retry next cycle"
  done
  after=$(find Studies -name results.csv 2>/dev/null | wc -l)
  [ "$after" != "$before" ] && echo "$(date +%H:%M) pulled: $before -> $after suites held locally"
  sleep "$INTERVAL"
done
