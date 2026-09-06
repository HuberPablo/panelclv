#!/usr/bin/env bash
# watch_fleet.sh [INTERVAL_SEC] — periodic "are the workers alive and moving?" report.
#
# An instance bills whether or not it is doing anything, so "silently broken" and
# "working" cost the same (healthcheck.sh's opening line). This wraps that check on a
# timer and adds the two things a single-shot check cannot see:
#
#   STALE      a worker whose shard.log has not grown since the previous cycle —
#              healthcheck's own stall test uses an absolute age, which cannot tell
#              "slow suite" from "hung" on a job whose suites take many minutes.
#   NO-PROGRESS  the fleet as a whole completing no new suites between cycles.
#
# It also flags a box still unprovisioned well past the 5-10 minutes §8 budgets for it,
# because that is the shape of F14: a crash-looping instance re-pulls the image on every
# restart and bills bandwidth per GB while producing nothing.
#
# Read-only. It destroys nothing and starts nothing — supervise.py owns those decisions.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export PATH="$HOME/venvs/panelclv/bin:$PATH"
INTERVAL="${1:-900}"
GRID="${GRID:-seasonal_4x4x10}"
# The day this run's first box was rented; spend is summed from here.
# Epoch of 2026-09-04 00:00 UTC — the day this run rented its first box.
RUN_START_EPOCH="${RUN_START_EPOCH:-1788480000}"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -n -i "$KEY" -o StrictHostKeyChecking=accept-new
     -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast" -o BatchMode=yes -o ConnectTimeout=12)
STATE=/tmp/watch_fleet_prev.txt
: > "$STATE"

while true; do
  now=$(date +%H:%M)
  raw=$(vastai show instances --raw 2>/dev/null)
  if [ -z "$raw" ]; then echo "[$now] API_ERROR — will retry (F16: not evidence a box is gone)"; sleep "$INTERVAL"; continue; fi

  # Billed-to-date comes from the INVOICE, never from the live instance list. Summing
  # duration x rate over `show instances` silently drops every destroyed box, so the
  # running total goes DOWN each time a worker is retired — backwards for a budget
  # guard, and it understated this run by ~30%.
  # `show invoices` with no arguments returns only the LATEST BILLING BATCH, so the
  # running total silently resets to ~0 at each period boundary — it read $9.33 at
  # 23:49 and $0.05 at 00:05. Charges are posted in batches stamped at the boundary,
  # so the honest figure is "all charge rows since the run began". -c excludes credits,
  # which otherwise make the sum negative.
  INVOICED=$(vastai show invoices --raw -c -s 2026-08-01 2>/dev/null | python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('?'); raise SystemExit
rows = d if isinstance(d,list) else (d.get('invoices') or d.get('charges') or [])
# A narrow -s is ignored by the API (it returns only the latest batch), so the query
# is deliberately wide and the run window is applied here instead.
SINCE = $RUN_START_EPOCH
t=0.0
for r in rows:
    if isinstance(r,dict) and (r.get('timestamp') or 0) >= SINCE:
        try: t += float(r.get('amount') or r.get('total') or 0)
        except Exception: pass
print(f'{t:.2f}')" 2>/dev/null)
  summary=$(echo "$raw" | python -c "
import json,sys
d=json.load(sys.stdin)
tot=sum(i.get('dph_total') or 0 for i in d)
st={}
for i in d: st[i.get('actual_status') or '?']=st.get(i.get('actual_status') or '?',0)+1
print(f\"{len(d)} box | {' '.join(f'{k}:{v}' for k,v in sorted(st.items()))} | \${tot:.3f}/hr\")
")
  mapfile -t rows < <(echo "$raw" | python -c "
import json,sys
for i in json.load(sys.stdin):
    ip=(i.get('public_ipaddr') or '').strip(); p=(i.get('ports') or {}).get('22/tcp') or []
    print(i['id'], ip or '-', (p[0].get('HostPort') if p else '-'), i.get('actual_status'), int((i.get('duration') or 0)/60))
")

  alerts=(); ok=0; cur=""
  for row in "${rows[@]}"; do
    set -- $row; id=$1; ip=$2; port=$3; status=$4; agemin=$5
    [ "$status" = "exited" ] && alerts+=("$id EXITED — terminal, re-pulls image on restart (F14): destroy")
    if [ "$ip" = "-" ] || [ "$port" = "-" ]; then continue; fi
    probe=$("${SSH[@]}" -p "$port" "root@$ip" '
      done=no; test -f /root/.onstart_done && done=yes
      # Bracketed pattern: unbracketed, pgrep matches the very shell running it (F13).
      run=no; pgrep -f "[r]un_pnbd_grid" >/dev/null 2>&1 && run=yes
      ex=none; test -f /root/.shard_exit && ex=$(cat /root/.shard_exit)
      sz=0; test -f /root/shard.log && sz=$(stat -c %s /root/shard.log)
      n=$(find /root/panelclv/Studies -name results.csv 2>/dev/null | wc -l)
      head=$(git -C /root/panelclv rev-parse --short HEAD 2>/dev/null || echo "-")
      echo "$done $run $ex $sz $n $head"' 2>/dev/null | tail -1)
    if [ -z "$probe" ]; then
      [ "$agemin" -gt 25 ] && alerts+=("$id unreachable for ${agemin}m — past the 5-10m provisioning budget (§8/F14)")
      continue
    fi
    set -- $probe; d_done=$1; d_run=$2; d_ex=$3; d_sz=$4; d_n=$5; d_head=$6
    prev=$(grep "^$id:" "$STATE" 2>/dev/null | head -1)
    [ "$d_done" = "no" ] && [ "$agemin" -gt 25 ] && alerts+=("$id still unprovisioned after ${agemin}m (F14 shape)")
    [ "$d_ex" != "none" ] && [ "$d_ex" != "0" ] && alerts+=("$id shard CRASHED exit=$d_ex (F9) — code bug, not hardware")
    if [ -n "$prev" ] && [ "$d_run" = "yes" ]; then
      psz=$(echo "$prev" | cut -d: -f2); pn=$(echo "$prev" | cut -d: -f3)
      pstale=$(echo "$prev" | cut -d: -f4); pstale=${pstale:-0}
      # A quiet log is NOT a stall here. Between the refit and the next suite the worker
      # runs the Monte Carlo rollout — 200 paths x 52 steps — which prints nothing, and
      # for the Transformer that silent phase routinely outlasts one cycle. Requiring two
      # consecutive quiet cycles AND no new completed suite is what distinguishes a hung
      # box from one that is simply mid-rollout. Destroying on the weaker signal is how
      # healthy workers get killed (F16).
      if [ "$d_sz" = "$psz" ] && [ "$d_n" = "$pn" ]; then
        stale=$((pstale+1))
        [ "$stale" -ge 2 ] && alerts+=("$id STALE — no log growth and no new suite for ${stale} cycles (${d_sz}B, suites=$d_n)")
      else stale=0; fi
    else stale=0; fi
    cur+="$id:$d_sz:$d_n:${stale:-0}"$'\n'
    [ "$d_run" = "yes" ] && ok=$((ok+1))
  done
  echo "$cur" > "$STATE"

  local_done=$(find Studies -name results.csv 2>/dev/null | wc -l)
  echo "[$now] $summary | training:$ok | pulled locally:$local_done/1920 | billed to date \$${INVOICED:-?}"
  for a in "${alerts[@]}"; do echo "[$now]   ALERT $a"; done
  sleep "$INTERVAL"
done
