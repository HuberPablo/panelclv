#!/usr/bin/env bash
#
# healthcheck.sh <INSTANCE_ID>... | --all — verify rented workers are actually working.
#
# Every check here exists because its absence once cost billed hours; the failures
# they detect are catalogued in VastAI/known_failures.md. Run it right after launch,
# and periodically during a run — an instance bills whether or not it is doing
# anything, so "silently broken" and "working" have identical cost.
#
# Usage:
#   ./VastAI/healthcheck.sh --all
#   ./VastAI/healthcheck.sh 48500149
#   GRID=seasonal_4x4x10 ./VastAI/healthcheck.sh --all     # also checks data + shard
#
# Exit status is 0 only if every instance passed every check, so it can gate a
# script: `healthcheck.sh "$ID" && start_shard.sh ...`.
#
# The checks, in the order a worker fails them:
#   state   cur_state=running                       (still loading / stopped / gone)
#   ssh     reachable, and by which endpoint        (F1 proxy-vs-direct, F3 no key)
#   onstart /root/.onstart_done exists              (provisioning aborted)
#   cuda    a real allocation on the device         (F2 driver too old — error 804)
#   pkg     `import panelclv` works                 (pip install failed or partial)
#   data    the grid's panels are present           (rsync never ran / half ran)
#   shard   training alive, done, or stalled        (crashed, or hung with no output)

set -uo pipefail          # NOT -e: a failing check must be reported, not abort the sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KEY="${VAST_KEY:-$HOME/.ssh/id_ed25519}"
GRID="${GRID:-}"
STALL_MINUTES="${STALL_MINUTES:-30}"   # no new log output for this long = stalled
VASTAI="$(command -v vastai || echo "$REPO_ROOT/../venvs/thesis_rocm/bin/vastai")"
PY="$(command -v python3 || command -v python)"

# Rented boxes reuse IPs and get a fresh host key each time, so a changed fingerprint
# is expected. -n because ssh in a loop otherwise eats the worklist from stdin (F6).
SSH_BASE=(-n -i "$KEY"
          -o StrictHostKeyChecking=accept-new
          -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast"
          -o BatchMode=yes -o ConnectTimeout=12)

red()   { printf '\033[31m%s\033[0m' "$1"; }
green() { printf '\033[32m%s\033[0m' "$1"; }
yellow(){ printf '\033[33m%s\033[0m' "$1"; }

# --- resolve the fleet from the API, never from a hand-kept list ---------------
# The API is the only source that knows the current endpoint; ports change when an
# instance is rebooted or recycled.
FLEET_JSON="$("$VASTAI" show instances --raw 2>/dev/null)"
if [ -z "$FLEET_JSON" ] || [ "$FLEET_JSON" = "[]" ]; then
    echo "no instances (or the API key is not authenticating)"
    exit 1
fi

if [ "${1:-}" = "--all" ]; then
    IDS=$("$PY" -c "
import json,sys
print(' '.join(str(r['id']) for r in json.loads(sys.stdin.read())))" <<< "$FLEET_JSON")
else
    IDS="$*"
fi
[ -n "${IDS// /}" ] || { echo "usage: healthcheck.sh <INSTANCE_ID>... | --all"; exit 2; }

FAILED=0

for ID in $IDS; do
    # Pull this instance's row: state, both endpoints, gpu, price.
    read -r STATE DIRECT_HOST DIRECT_PORT PROXY_HOST PROXY_PORT GPU DPH <<< "$(
        "$PY" -c "
import json,sys
rows=[r for r in json.loads(sys.stdin.read()) if str(r['id'])=='$ID']
if not rows:
    print('GONE - - - - - 0'); raise SystemExit
r=rows[0]
p=(r.get('ports') or {}).get('22/tcp') or []
hp=p[0].get('HostPort') if p else '-'
print(r.get('cur_state'), r.get('public_ipaddr') or '-', hp or '-',
      r.get('ssh_host') or '-', r.get('ssh_port') or '-',
      (r.get('gpu_name') or '-').replace(' ','_'), round(r.get('dph_total') or 0, 4))
" <<< "$FLEET_JSON")"

    printf '%-10s %-9s %-13s $%-7s ' "$ID" "$STATE" "$GPU" "$DPH"

    # --- state ---------------------------------------------------------------
    if [ "$STATE" != "running" ]; then
        printf '%s (state=%s)\n' "$(yellow 'SKIP')" "$STATE"
        [ "$STATE" = "GONE" ] && FAILED=1
        continue
    fi

    # --- ssh: direct first, proxy as fallback (F1) ---------------------------
    # Auth failure is reported separately from an unreachable host, because the two
    # mean different things: a rejected key is F3 (destroy it), a closed connection
    # is usually the wrong endpoint (F1) or a box still coming up.
    ENDPOINT=""; SSH_NOTE=""
    for candidate in "$DIRECT_HOST:$DIRECT_PORT:direct" "$PROXY_HOST:$PROXY_PORT:proxy"; do
        IFS=: read -r h p label <<< "$candidate"
        [ "$h" = "-" ] || [ "$p" = "-" ] && continue
        out=$(timeout 20 ssh "${SSH_BASE[@]}" -p "$p" "root@$h" 'echo ok' 2>&1)
        if grep -q '^ok$' <<< "$out"; then ENDPOINT="$h $p"; SSH_NOTE="$label"; break; fi
        grep -q 'Permission denied' <<< "$out" && SSH_NOTE="key-rejected"
    done
    if [ -z "$ENDPOINT" ]; then
        printf '%s ssh (%s) — see known_failures.md %s\n' "$(red 'FAIL')" \
               "${SSH_NOTE:-unreachable}" \
               "$([ "$SSH_NOTE" = key-rejected ] && echo F3 || echo F1)"
        FAILED=1; continue
    fi
    read -r H P <<< "$ENDPOINT"

    # --- everything else, in one round trip ----------------------------------
    # One ssh call rather than seven: each round trip to a rented box is ~1s, and a
    # sweep over a dozen workers should not take a minute.
    REMOTE=$(timeout 60 ssh "${SSH_BASE[@]}" -p "$P" "root@$H" "
        for c in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
            [ -x \"\$c\" ] && PY=\"\$c\" && break
        done
        test -f /root/.onstart_done && echo 'onstart=ok' || {
            echo \"onstart=missing:\$(grep -oE 'FATAL.*' /root/onstart.log 2>/dev/null | tail -1)\"
        }
        # A real allocation, not nvidia-smi: nvidia-smi talks to the driver and
        # succeeds even when the CUDA runtime cannot initialise (F2).
        \"\$PY\" - <<'PYCHK' 2>/dev/null || echo 'cuda=error'
import torch
try:
    assert torch.cuda.is_available()
    torch.zeros(8, device='cuda').sum().item()
    print('cuda=ok')
except Exception as exc:
    print('cuda=fail:' + str(exc).replace(chr(10), ' ')[:90])
PYCHK
        \"\$PY\" -c 'import panelclv; print(\"pkg=ok\")' 2>/dev/null || echo 'pkg=fail'
        if [ -n '$GRID' ]; then
            d=/root/panelclv/Datasets/Synthetic/$GRID
            n=\$(find \"\$d\" -name panel.csv 2>/dev/null | wc -l)
            echo \"data=\$n\"
        fi
        if [ -f /root/.shard_exit ]; then
            code=\$(cat /root/.shard_exit)
            [ \"\$code\" = 0 ] && echo 'shard=done' || echo \"shard=CRASHED:\$code\"
        elif [ -f /root/.shard_done ]; then echo 'shard=done'
        elif [ -f /root/shard.log ]; then
            age=\$(( (\$(date +%s) - \$(stat -c %Y /root/shard.log)) / 60 ))
            trained=\$(grep -c 'min/dataset' /root/shard.log 2>/dev/null; true)
            trained=\${trained:-0}
            echo \"shard=running:\$trained:\${age}m\"
        else echo 'shard=none'; fi
    " 2>/dev/null)

    # --- verdict --------------------------------------------------------------
    get() { grep -oE "^$1=[^ ]*" <<< "$REMOTE" | head -1 | cut -d= -f2-; }
    ONSTART=$(get onstart); CUDA=$(get cuda); PKG=$(get pkg)
    DATA=$(get data);       SHARD=$(get shard)

    PROBLEMS=()
    [ "$ONSTART" = ok ]  || PROBLEMS+=("onstart:${ONSTART}")
    [ "$CUDA" = ok ]     || PROBLEMS+=("cuda:${CUDA} [F2]")
    [ "$PKG" = ok ]      || PROBLEMS+=("pkg-import-failed")
    [ -n "$GRID" ] && [ "${DATA:-0}" -eq 0 ] 2>/dev/null && PROBLEMS+=("no-data")

    # A stalled shard is the failure that looks most like success: the process is
    # alive, the box bills, and nothing has been written for half an hour.
    case "$SHARD" in
        CRASHED:*) PROBLEMS+=("shard $SHARD — read /root/shard.log") ;;&
        running:*) IFS=: read -r _ trained age <<< "$SHARD"
                   [ "${age%m}" -gt "$STALL_MINUTES" ] 2>/dev/null \
                       && PROBLEMS+=("STALLED ${age} no output") ;;
    esac

    if [ ${#PROBLEMS[@]} -eq 0 ]; then
        printf '%s via %-6s %s\n' "$(green 'OK  ')" "$SSH_NOTE" "${SHARD}${DATA:+ data=$DATA}"
    else
        printf '%s via %-6s %s\n' "$(red 'FAIL')" "$SSH_NOTE" "${PROBLEMS[*]}"
        FAILED=1
    fi
done

exit $FAILED
