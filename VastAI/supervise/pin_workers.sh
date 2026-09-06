#!/usr/bin/env bash
# pin_workers.sh <COMMIT> — hold every rented worker at one commit until it sticks.
#
# Workers clone the repo themselves in vast_onstart.sh, so a push that lands AFTER a
# box has cloned leaves that box running stale code — and the shard it starts is a
# silently wrong experiment rather than an error. This reconciles that: it polls every
# instance, and any box whose HEAD is not <COMMIT> is fetched and hard-reset onto it.
#
# Idempotent and safe to leave running: a box already at the commit is untouched, an
# unreachable box is retried (F16 — a refused connection means "ask again", not
# "broken"), and a box that has already started its shard is REPORTED rather than
# reset, because resetting under a running trainer would mix two commits in one result.
set -u
WANT="${1:?usage: pin_workers.sh <commit-sha>}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export PATH="$HOME/venvs/panelclv/bin:$PATH"
SSH=(ssh -n -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new
     -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast" -o BatchMode=yes -o ConnectTimeout=15)

for cycle in $(seq 1 200); do
  mapfile -t rows < <(vastai show instances --raw 2>/dev/null | python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
for i in d:
    ip=(i.get('public_ipaddr') or '').strip(); p=(i.get('ports') or {}).get('22/tcp') or []
    if ip and p: print(i['id'], ip, p[0].get('HostPort'))
")
  pending=0
  for row in "${rows[@]}"; do
    set -- $row; id=$1; ip=$2; port=$3
    out=$("${SSH[@]}" -p "$port" "root@$ip" "
      if [ ! -d /root/panelclv/.git ]; then echo NOREPO; exit 0; fi
      if [ -f /root/.shard_exit ] || pgrep -f \"[r]un_(pnbd_grid|real_panel_arms)\" >/dev/null 2>&1; then
        echo \"RUNNING \$(git -C /root/panelclv rev-parse --short HEAD)\"; exit 0; fi
      cur=\$(git -C /root/panelclv rev-parse HEAD)
      if [ \"\$cur\" != \"$WANT\" ]; then
        git -C /root/panelclv fetch -q origin main && git -C /root/panelclv reset -q --hard $WANT
      fi
      echo \"AT \$(git -C /root/panelclv rev-parse --short HEAD)\"
    " 2>&1 | tail -1)
    case "$out" in
      AT\ *)   short=${out#AT }; [ "${WANT:0:7}" = "${short:0:7}" ] || { echo "$id: mismatch ($out)"; pending=1; } ;;
      RUNNING\ *) echo "$id: shard already started at ${out#RUNNING } — NOT resetting" ;;
      *)       pending=1 ;;
    esac
  done
  if [ ${#rows[@]} -gt 0 ] && [ $pending -eq 0 ]; then
    echo "all ${#rows[@]} reachable workers pinned at ${WANT:0:7} (cycle $cycle)"
  fi
  sleep 45
done
