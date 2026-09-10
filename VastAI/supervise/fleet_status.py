#!/usr/bin/env python3
"""One line per rented box: is it training, how far in, and what it is costing.

Reads the fleet from `vastai show instances --raw` (F22 -- never from a launcher's
stdout) and then asks each box directly. `.shard_exit` is the authority on whether a
worker finished and with what status: an unconditional done-marker once made a shard
that crashed in seconds indistinguishable from one that trained for hours (F9).
"""
import json, subprocess, pathlib, sys, time

KEY = pathlib.Path.home() / ".ssh/id_ed25519"
SSH = ["ssh", "-n", "-i", str(KEY), "-o", "StrictHostKeyChecking=accept-new",
       "-o", f"UserKnownHostsFile={pathlib.Path.home()}/.ssh/known_hosts_vast",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]

REMOTE = (
    # Count the PYTHON process, anchored at the start of the command line. `pgrep -f
    # run_ar_encoding_ablation` also matches the ssh command carrying this probe, so it
    # reported 4 where there was 1 -- F13, in the diagnostic rather than in a killer.
    "R=$(ps -eo cmd | grep -c '^/[^ ]*python.* scripts/run_ar_encoding_ablation'); "
    "E=$(cat /root/.shard_exit 2>/dev/null || echo -); "
    "D=$(ls -d /root/panelclv/Studies/*/ 2>/dev/null | wc -l); "
    "P=$(ls /root/panelclv/Studies/*/LSTM/Predictions/Prediction_*.csv 2>/dev/null | wc -l); "
    # The box is the authority on which slice it was given: start_shard.sh writes it to
    # .shard_spec. Deriving it from position in the fleet listing was wrong the moment a
    # box was destroyed, because every later box renumbered.
    "W=$(awk '{print $3}' /root/.shard_spec 2>/dev/null || echo ?); "
    "echo \"$R $E $D $P $W\""
)

raw = subprocess.run(["vastai", "show", "instances", "--raw"],
                     capture_output=True, text=True).stdout
instances = sorted(json.loads(raw), key=lambda i: i["id"])
print(f"{'instance':10s} {'slice':>6s} {'status':9s} {'$/hr':>7s} {'run':>3s} {'exit':>4s} "
      f"{'suites':>6s} {'preds':>5s}  gpu | cpu")
total_dph = 0.0
total_preds = 0
for inst in instances:
    dph = inst.get("dph_total") or 0.0
    total_dph += dph
    status = str(inst.get("actual_status"))
    fields = ["?", "?", "?", "?", "?"]
    if status == "running":
        out = subprocess.run(
            SSH + ["-p", str(inst.get("ssh_port")), f"root@{inst.get('ssh_host')}", REMOTE],
            capture_output=True, text=True, timeout=40)
        parts = [l for l in out.stdout.strip().splitlines() if l and l[0].isdigit()]
        if parts:
            fields = (parts[-1].split() + ["?"] * 5)[:5]
    try:
        total_preds += int(fields[3])
    except ValueError:
        pass
    print(f"{inst['id']:<10d} {fields[4]:>6s} {status:9s} {dph:7.4f} {fields[0]:>3s} {fields[1]:>4s} "
          f"{fields[2]:>6s} {fields[3]:>5s}  {inst.get('gpu_name')} | {str(inst.get('cpu_name'))[:26]}")
print(f"\nfleet ${total_dph:.4f}/hr, {total_preds} forecasts on the workers "
      f"(900 needed across the run)")
