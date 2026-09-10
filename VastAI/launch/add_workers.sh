#!/usr/bin/env bash
# Rent boxes for specific worker indices and start each on its slice.
#
# Unlike drive_fleet.py this takes an EXPLICIT offer->index mapping. Assigning by
# sorted instance id was fine for one pass and wrong the moment the fleet changed:
# destroying a keyless box renumbers every worker after it, so a box would be handed
# a slice already being run elsewhere. The index is what a box IS; it must be pinned
# to the box, not derived from its neighbours.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export PATH="$HOME/venvs/panelclv/bin:$HOME/thesis-agent/venv/bin:$PATH"
LOG=VastAI/state/add_workers.log
mkdir -p VastAI/state/worker_logs VastAI/state/started

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for pair in "$@"; do
  OFFER="${pair%%:*}"; IDX="${pair##*:}"
  say "renting offer $OFFER for worker $IDX/10"
  OUT=$(./VastAI/launch/vast_launch.sh "$OFFER" 2>&1)
  echo "$OUT" >> "$LOG"
  # The instance id is the one thing the launcher reports that the API cannot yet be
  # asked for (the box may not be listed at the instant it is created). Everything
  # else -- host, port, status -- is read back from `show instances --raw` (F22).
  ID=$(echo "$OUT" | grep -oP 'instance id: \K[0-9]+' | head -1)
  if [ -z "$ID" ]; then say "  offer $OFFER did not produce an instance — skipping"; continue; fi
  say "  instance $ID -> worker $IDX/10"
  echo "$ID $IDX" >> VastAI/state/assignments.txt
  (
    # Wait for `running` before calling start_shard: its reachability probe gives up
    # after ~90 s, which is right for a keyless box (F21) and wrong for one still
    # pulling its image, and returns rc=2 on a perfectly good box.
    for _ in $(seq 1 60); do
      read -r HOST PORT STATUS < <(python3 - "$ID" <<'PY'
import json, subprocess, sys
raw = subprocess.run(["vastai","show","instances","--raw"], capture_output=True, text=True).stdout
for i in json.loads(raw or "[]"):
    if str(i["id"]) == sys.argv[1]:
        print(i.get("ssh_host"), i.get("ssh_port"), i.get("actual_status")); break
else:
    print("- - gone")
PY
)
      [ "$STATUS" = "running" ] && break
      [ "$STATUS" = "gone" ] && break
      sleep 30
    done
    for attempt in 1 2 3 4; do
      INSTANCE_ID="$ID" DATA_SRC=Datasets/Dataset_clean DATA_DST=Datasets/Dataset_clean \
      RUNNER_CMD="scripts/run_ar_encoding_ablation.py --worker $IDX/10" \
        ./VastAI/launch/start_shard.sh "$HOST" "$PORT" ar_encoding lstm "$IDX/10" \
        >> VastAI/state/worker_logs/start_${ID}_w${IDX}.log 2>&1
      rc=$?
      if [ $rc = 0 ]; then
        say "  $ID w$IDX STARTED"; touch VastAI/state/started/"$ID"; break
      fi
      say "  $ID w$IDX start attempt $attempt rc=$rc"
      sleep 60
    done
  ) &
done
wait
say "add_workers finished"
