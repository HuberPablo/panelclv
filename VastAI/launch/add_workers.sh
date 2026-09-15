#!/usr/bin/env bash
# Rent boxes for specific worker indices and start each on its slice.
#
# Unlike drive_fleet.py this takes an EXPLICIT offer->index mapping. Assigning by
# sorted instance id was fine for one pass and wrong the moment the fleet changed:
# destroying a keyless box renumbers every worker after it, so a box would be handed
# a slice already being run elsewhere. The index is what a box IS; it must be pinned
# to the box, not derived from its neighbours.
#
# Every box leaves here with a VERDICT, and this script acts on it (known_failures.md,
# "Start-up failures: the strategy"): either it is confirmed training, or its logs are
# saved, it is destroyed, and its worker index is appended to
# VastAI/state/needs_replacement.txt as `<instance> <worker> <reason>`. A box is never
# left rented after the launcher has given up on it (F32). Replacing is then a second
# invocation with fresh offers for the listed workers.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export PATH="$HOME/venvs/panelclv/bin:$HOME/thesis-agent/venv/bin:$PATH"
# Which run the boxes join. Defaults are the AR-encoding ablation this was written for;
# any runner taking `--worker I/N` fits:
#   RUNNER=scripts/run_real_panel_benchmarks.py TOTAL=20 ./VastAI/launch/add_workers.sh ...
RUNNER="${RUNNER:-scripts/run_ar_encoding_ablation.py}"
TOTAL="${TOTAL:-10}"
# What start_shard pushes before the trainer starts, relative to the repo root. `{IDX}`
# is replaced by each box's worker index, so a resumed slice can be sent only its own
# finished forecasts alongside the panels and skip them instead of retraining:
#   DATA_SRC='VastAI/state/seed_w{IDX}' DATA_DST=. ./VastAI/launch/add_workers.sh ...
DATA_SRC="${DATA_SRC:-Datasets/Dataset_clean}"
DATA_DST="${DATA_DST:-Datasets/Dataset_clean}"
LOG=VastAI/state/add_workers.log
REPLACE=VastAI/state/needs_replacement.txt
mkdir -p VastAI/state/worker_logs VastAI/state/started

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# One launcher at a time. Two launchers in flight reached the same box and both started a
# trainer on the same slice (F26); a superseded launcher also kept retrying a destroyed
# box's recycled proxy endpoint (F32). The lock is held until this script exits.
exec 9> VastAI/state/add_workers.lock
if ! flock -n 9; then
  echo "another add_workers.sh is still running — wait for it to exit (F26)" >&2
  exit 1
fi

# `<host> <port> <status>` for one instance. Direct endpoint first, proxy as fallback:
# the proxy does not reliably accept the instance key (F1), and this launcher was the one
# tool still reading it, which gave up on healthy boxes (F29). `gone` means the API no
# longer lists the instance at all (F31).
endpoint() {
  python3 - "$1" <<'PY'
import json, subprocess, sys
raw = subprocess.run(["vastai", "show", "instances", "--raw"],
                     capture_output=True, text=True).stdout
try:
    rows = json.loads(raw or "[]")
except json.JSONDecodeError:
    print("- - api-error"); raise SystemExit      # F16: not evidence the box is gone
for i in rows:
    if str(i["id"]) == sys.argv[1]:
        ip = (i.get("public_ipaddr") or "").strip()
        p = (i.get("ports") or {}).get("22/tcp") or []
        if ip and p and p[0].get("HostPort"):
            print(ip, p[0]["HostPort"], i.get("actual_status"))
        else:
            print(i.get("ssh_host") or "-", i.get("ssh_port") or "-", i.get("actual_status"))
        break
else:
    print("- - gone")
PY
}

# Terminal verdict: keep what the box can still tell us, destroy it, queue its index.
retire() {
  local id=$1 idx=$2 reason=$3 host=$4 port=$5
  if [ "$host" != "-" ]; then
    for f in onstart.log shard.log; do
      scp -q -i "$HOME/.ssh/id_ed25519" -o BatchMode=yes -o ConnectTimeout=15 \
          -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast" \
          -P "$port" "root@$host:/root/$f" "VastAI/state/worker_logs/dead_${id}_w${idx}_$f" 2>/dev/null
    done
  fi
  # F24: -y, and verify against the API rather than trusting the command's output.
  vastai destroy instance -y "$id" >/dev/null 2>&1
  sleep 5
  if [ "$(endpoint "$id" | awk '{print $3}')" = "gone" ]; then
    say "  $id w$idx RETIRED ($reason) — destroyed, logs in VastAI/state/worker_logs/dead_${id}_w${idx}_*"
  else
    say "  $id w$idx RETIRED ($reason) — DESTROY NOT CONFIRMED, check: vastai show instances"
  fi
  echo "$id $idx $reason" >> "$REPLACE"
}

for pair in "$@"; do
  OFFER="${pair%%:*}"; IDX="${pair##*:}"
  say "renting offer $OFFER for worker $IDX/$TOTAL"
  OUT=$(./VastAI/launch/vast_launch.sh "$OFFER" 2>&1)
  echo "$OUT" >> "$LOG"
  # The instance id is the one thing the launcher reports that the API cannot yet be
  # asked for (the box may not be listed at the instant it is created). Everything
  # else -- host, port, status -- is read back from `show instances --raw` (F22).
  ID=$(echo "$OUT" | grep -oP 'instance id: \K[0-9]+' | head -1)
  if [ -z "$ID" ]; then
    say "  offer $OFFER did not produce an instance — worker $IDX needs another offer"
    echo "- $IDX no-instance(offer $OFFER)" >> "$REPLACE"
    continue
  fi
  say "  instance $ID -> worker $IDX/$TOTAL"
  echo "$ID $IDX" >> VastAI/state/assignments.txt
  (
    # Wait for `running` before calling start_shard, whose probe would otherwise spend
    # its patience on an image pull (F27). `gone` is terminal: there is no box to start,
    # keyed or not, so nothing is attempted on it (F31).
    #
    # 15 minutes (30 checks x 30 s). Over the 2026-09-13 runs, every box that went on to
    # train reached `running` within ~10 minutes of rental, most in 1-6; one still loading
    # past 15 is stuck, and the previous 30-minute limit only paid for more of it.
    HOST=- PORT=- STATUS=unknown
    for _ in $(seq 1 30); do
      read -r HOST PORT STATUS < <(endpoint "$ID")
      [ "$STATUS" = "running" ] || [ "$STATUS" = "gone" ] && break
      sleep 30
    done
    if [ "$STATUS" = "gone" ]; then
      say "  $ID w$IDX vanished from the API before it was reachable (F31)"
      echo "$ID $IDX vanished" >> "$REPLACE"
      exit 0
    fi
    if [ "$STATUS" != "running" ]; then
      retire "$ID" "$IDX" "never-running(status=$STATUS)" "$HOST" "$PORT"; exit 0
    fi

    # start_shard.sh returns a verdict code (see its header). 1 is the only one worth
    # retrying; the endpoint is re-read first, since vast can remap ports on a restart.
    for attempt in 1 2; do
      read -r HOST PORT STATUS < <(endpoint "$ID")
      INSTANCE_ID="$ID" DATA_SRC="${DATA_SRC//\{IDX\}/$IDX}" DATA_DST="$DATA_DST" \
      RUNNER_CMD="$RUNNER --worker $IDX/$TOTAL" \
        ./VastAI/launch/start_shard.sh "$HOST" "$PORT" "$(basename "$RUNNER" .py)" worker "$IDX/$TOTAL" \
        >> VastAI/state/worker_logs/start_${ID}_w${IDX}.log 2>&1
      rc=$?
      case $rc in
        0) say "  $ID w$IDX TRAINING (confirmed)"; touch VastAI/state/started/"$ID"; exit 0 ;;
        5) say "  $ID w$IDX already has a trainer — left alone (F26)"; exit 0 ;;
        2) retire "$ID" "$IDX" "unreachable-or-keyless" "$HOST" "$PORT"; exit 0 ;;
        3) retire "$ID" "$IDX" "onstart-failed" "$HOST" "$PORT"; exit 0 ;;
        4) retire "$ID" "$IDX" "trainer-did-not-start" "$HOST" "$PORT"; exit 0 ;;
        *) say "  $ID w$IDX start attempt $attempt rc=$rc — see worker_logs/start_${ID}_w${IDX}.log" ;;
      esac
      sleep 30
    done
    retire "$ID" "$IDX" "setup-failed(rc=$rc)" "$HOST" "$PORT"
  ) &
done
wait
say "add_workers finished"
if [ -s "$REPLACE" ]; then
  say "workers queued for replacement (VastAI/state/needs_replacement.txt):"
  while read -r line; do say "  $line"; done < "$REPLACE"
fi
