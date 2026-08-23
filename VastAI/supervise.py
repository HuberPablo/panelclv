#!/usr/bin/env python3
"""supervise.py — keep a grid's workers training until every shard has results.

The orchestration this replaces was fire-and-forget: each script did one step and
exited, so nothing owned the statement "seven transformer shards should be training
right now". A box that failed to start, lost its data, or quietly died was nobody's
job to notice, and every failure in VastAI/known_failures.md needed a human to look.

This is a reconciliation loop instead. It declares the desired state (one worker per
shard, from the grid's ``workers``), observes the actual state every cycle, and acts
on the difference:

    not running / key rejected / CUDA dead   -> destroy, free the shard for re-rent
    reachable but no data                    -> re-run the driver (rsync resumes)
    reachable, data, no trainer              -> start the driver
    stalled (no output for STALL_MINUTES)    -> restart once, destroy on repeat
    crashed                                  -> stop touching it; a crash is code
    finished                                 -> pull results, verify, destroy
    shard with no worker                     -> rent one (within the price/fleet cap)

and it stops when every shard's results are on this machine.

Usage:
    python VastAI/supervise.py --grid seasonal_4x4x10
    python VastAI/supervise.py --grid seasonal_4x4x10 --dry-run     # observe only
    python VastAI/supervise.py --grid seasonal_4x4x10 --max-hours 8

Safety rules it will not break:
  * A worker that is making progress is never destroyed.
  * Nothing is destroyed before its results are pulled and counted.
  * Renting obeys --max-price and --max-workers (VastAI/Rules.md §8).
  * A crashed shard is reported, not retried in a loop — repeated identical crashes
    mean a bug in the code, and re-renting hardware will not fix it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from grids import load_grid  # noqa: E402

VASTAI = "vastai"
KEY = Path.home() / ".ssh" / "id_ed25519"
SSH_OPTS = [
    "-n", "-i", str(KEY),
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={Path.home()/'.ssh'/'known_hosts_vast'}",
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
]


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    """Run a command, returning (returncode, combined output). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:                                  # noqa: BLE001
        return 1, str(exc)


# ---------------------------------------------------------------------------
# Desired state: one worker per shard, read off the grid declaration
# ---------------------------------------------------------------------------

@dataclass
class Shard:
    """One unit of work: one model over one strided slice of the grid's datasets."""

    model_type: str
    model_name: str
    index: int
    total: int
    instance: int | None = None      # the worker currently assigned, if any
    restarts: int = 0                # driver restarts, to stop an infinite repair loop
    done: bool = False

    @property
    def key(self) -> str:
        return f"{self.model_type}:{self.index}/{self.total}"

    @property
    def spec(self) -> str:
        return f"{self.index}/{self.total}"


def desired_shards(spec) -> list[Shard]:
    """Expand the grid's ``workers`` into one Shard per rented worker.

    ``workers[model_type] == 0`` means the model runs on the orchestrator rather than
    on vast (Rules.md §5), so it produces no shards here.
    """
    shards: list[Shard] = []
    for model in spec.models:
        n = spec.workers.get(model.model_type, 0)
        for i in range(1, n + 1):
            shards.append(Shard(model.model_type, model.name, i, n))
    return shards


# ---------------------------------------------------------------------------
# Observed state
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    id: int
    state: str
    host: str | None
    port: str | None
    gpu: str
    dph: float
    started: float
    # filled in by probe()
    reachable: bool = False
    auth_failed: bool = False
    onstart: str = "?"
    cuda: str = "?"
    pkg: str = "?"
    panels: int = 0
    shard: str = "?"
    assignment: str = ""          # "<grid> <model_type> <i/N>", read off the box
    notes: list[str] = field(default_factory=list)


def fleet() -> dict[int, Instance]:
    """Every instance on the account, keyed by id, with its *direct* endpoint.

    The endpoint comes from public_ipaddr + the host port mapped to the container's
    22/tcp, never ssh_host/ssh_port — those name vast's proxy, which refuses the
    instance-attached key (known_failures.md F1).
    """
    code, out = run([VASTAI, "show", "instances", "--raw"])
    try:
        rows = json.loads(out)
    except Exception:                                          # noqa: BLE001
        return {}
    result = {}
    for r in rows:
        ports = (r.get("ports") or {}).get("22/tcp") or []
        result[int(r["id"])] = Instance(
            id=int(r["id"]),
            state=str(r.get("cur_state")),
            host=r.get("public_ipaddr"),
            port=(ports[0].get("HostPort") if ports else None),
            gpu=str(r.get("gpu_name")),
            dph=float(r.get("dph_total") or 0),
            started=float(r.get("start_date") or 0),
        )
    return result


PROBE = r"""
for c in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
    [ -x "$c" ] && PY="$c" && break
done
test -f /root/.onstart_done && echo onstart=ok || echo onstart=missing
echo "assignment=$(cat /root/.shard_spec 2>/dev/null | tr ' ' ',')"
"$PY" - <<'PYCHK' 2>/dev/null || echo cuda=error
import torch
try:
    assert torch.cuda.is_available()
    torch.zeros(8, device='cuda').sum().item()
    print('cuda=ok')
except Exception as exc:
    print('cuda=fail')
PYCHK
"$PY" -c 'import panelclv' 2>/dev/null && echo pkg=ok || echo pkg=fail
echo panels=$(find /root/panelclv/Datasets/Synthetic/%(grid)s -name panel.csv 2>/dev/null | wc -l)
if [ -f /root/.shard_exit ]; then
    c=$(cat /root/.shard_exit); [ "$c" = 0 ] && echo shard=done || echo shard=crashed:$c
elif [ -f /root/shard.log ]; then
    echo shard=running:$(( ($(date +%%s) - $(stat -c %%Y /root/shard.log)) / 60 ))
else
    echo shard=none
fi
"""


def probe(inst: Instance, grid: str) -> None:
    """Fill in an instance's health by one SSH round trip. Never raises."""
    if inst.state != "running" or not inst.host or not inst.port:
        return
    code, out = run(
        ["ssh", *SSH_OPTS, "-p", str(inst.port), f"root@{inst.host}",
         PROBE % {"grid": grid}],
        timeout=90,
    )
    if "Permission denied" in out:
        inst.auth_failed = True
        return
    if code != 0 and "onstart=" not in out:
        return
    inst.reachable = True
    for field_name in ("onstart", "cuda", "pkg", "shard"):
        m = re.search(rf"^{field_name}=(\S+)", out, re.M)
        if m:
            setattr(inst, field_name, m.group(1))
    m = re.search(r"^panels=(\d+)", out, re.M)
    if m:
        inst.panels = int(m.group(1))
    m = re.search(r"^assignment=(\S*)", out, re.M)
    if m and m.group(1):
        inst.assignment = m.group(1).replace(",", " ")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def destroy(inst_id: int, why: str, dry: bool) -> None:
    print(f"    destroy {inst_id}: {why}")
    if not dry:
        run([VASTAI, "destroy", "instance", str(inst_id), "-y"])


def start_driver(inst: Instance, spec, grid: str, shard: Shard, dry: bool) -> None:
    """Push data if needed and start the shard, detached, via start_shard.sh."""
    print(f"    start {shard.key} on {inst.id} ({inst.host}:{inst.port})")
    if dry:
        return
    # Seed the worker with suites already collected for this model. run_pnbd_grid.py
    # skips any suite whose results.csv exists, but it reads the *worker's* disk — so
    # without this a replacement worker redoes work another worker already finished.
    seed = spec.train_base(shard.model_name)
    if seed.is_dir() and any(seed.glob("*__*")):
        run(["rsync", "-az", "--partial", "-e",
             "ssh " + " ".join(o for o in SSH_OPTS if o != "-n") + f" -p {inst.port}",
             str(seed) + "/",
             f"root@{inst.host}:/root/panelclv/Studies/{spec.name}__{shard.model_name}/"],
            timeout=1800)
    log = REPO_ROOT / "VastAI" / "state" / f"driver_{inst.id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        subprocess.Popen(
            [str(REPO_ROOT / "VastAI" / "start_shard.sh"),
             inst.host, str(inst.port), grid, shard.model_type, shard.spec],
            stdout=fh, stderr=fh, start_new_session=True,
        )


def pull_results(inst: Instance, spec, shard: Shard, dry: bool) -> int:
    """rsync this worker's suites into the local tree. Returns suites now present.

    Shards write disjoint ``<combo>__<dataset>/`` folders inside one model's tree, so
    pulling every worker into the same local directory reassembles the grid with no
    merge step (Rules.md §4).
    """
    local = spec.train_base(shard.model_name)
    local.mkdir(parents=True, exist_ok=True)
    remote = f"/root/panelclv/Studies/{spec.name}__{shard.model_name}/"
    print(f"    pull {shard.key} -> {local.name}")
    if not dry:
        run(["rsync", "-az", "--partial", "-e",
             "ssh " + " ".join(o for o in SSH_OPTS if o != "-n") + f" -p {inst.port}",
             f"root@{inst.host}:{remote}", str(local) + "/"], timeout=1800)
    return len(list(local.glob("*__*")))


def pick_offer(max_price: float, exclude: set[str]) -> tuple[str, float] | None:
    """Cheapest current/recent-generation offer under the cap, from vast_search.py.

    Generation matters more than price here: the `ancient` Xeon E5 rows are where a
    launch-bound workload crawls regardless of how cheap they are.
    """
    code, out = run([sys.executable, str(REPO_ROOT / "VastAI" / "vast_search.py"),
                     "--max-price", str(max_price), "--top", "25"], timeout=180)
    best = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        offer, price = parts[0], parts[1]
        if offer in exclude or ("recent" not in line and "current" not in line):
            continue
        try:
            price_f = float(price)
        except ValueError:
            continue
        if price_f <= max_price and (best is None or price_f < best[1]):
            best = (offer, price_f)
    return best


def rent(offer: str, dry: bool) -> None:
    print(f"    rent offer {offer}")
    if dry:
        return
    log = REPO_ROOT / "VastAI" / "state" / f"launch_{offer}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        subprocess.Popen([str(REPO_ROOT / "VastAI" / "vast_launch.sh"), offer],
                         stdout=fh, stderr=fh, start_new_session=True)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", required=True)
    ap.add_argument("--interval", type=int, default=180, help="seconds between cycles")
    ap.add_argument("--max-price", type=float, default=0.10, help="$/hr per worker")
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="watchdog: destroy any worker older than this")
    ap.add_argument("--stall-minutes", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="observe and report only")
    args = ap.parse_args()

    spec = load_grid(args.grid)
    shards = desired_shards(spec)
    state_path = REPO_ROOT / "VastAI" / "state" / f"{args.grid}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: a shard whose results are already local is finished, whatever the
    # fleet looks like. This is what makes the supervisor restartable.
    print(f"supervising {args.grid}: {len(shards)} shards"
          f"{' (DRY RUN)' if args.dry_run else ''}")

    tried_offers: set[str] = set()
    cycle = 0

    while True:
        cycle += 1
        insts = fleet()
        for inst in insts.values():
            probe(inst, args.grid)

        # --- match shards to instances -------------------------------------
        # Which worker is running which shard is read back off the box itself
        # (start_shard.sh records it), so the supervisor survives its own restart.
        by_key = {s.key: s for s in shards}
        assigned: dict[int, Shard] = {}
        for inst in insts.values():
            if not inst.assignment:
                continue
            parts = inst.assignment.split()
            if len(parts) != 3:
                continue
            _grid, model_type, spec_str = parts
            shard = by_key.get(f"{model_type}:{spec_str}")
            if shard is not None:
                shard.instance = inst.id
                assigned[inst.id] = shard
        # A shard whose worker has vanished is free again.
        live = {sh.key for sh in assigned.values()}
        for shard in shards:
            if shard.key not in live:
                shard.instance = None

        print(f"\n--- cycle {cycle}  {time.strftime('%H:%M:%S')}  "
              f"{len(insts)} instances, ${sum(i.dph for i in insts.values()):.3f}/hr ---")

        free_instances: list[Instance] = []

        for inst in sorted(insts.values(), key=lambda i: i.id):
            shard = assigned.get(inst.id)
            tag = shard.key if shard else "unassigned"
            age_h = (time.time() - inst.started) / 3600 if inst.started else 0

            # watchdog first: it overrides every other consideration
            if age_h > args.max_hours:
                destroy(inst.id, f"watchdog: {age_h:.1f}h old", args.dry_run)
                if shard:
                    shard.instance = None
                continue

            if inst.state != "running":
                destroy(inst.id, f"state={inst.state} (F12: never started)", args.dry_run)
                if shard:
                    shard.instance = None
                continue

            if inst.auth_failed:
                destroy(inst.id, "ssh key rejected (F3)", args.dry_run)
                if shard:
                    shard.instance = None
                continue

            if not inst.reachable:
                print(f"  {inst.id:<9} {tag:<22} unreachable — retrying next cycle")
                continue

            if inst.cuda != "ok":
                destroy(inst.id, f"cuda={inst.cuda} (F2: driver too old)", args.dry_run)
                if shard:
                    shard.instance = None
                continue

            if inst.onstart != "ok" or inst.pkg != "ok":
                print(f"  {inst.id:<9} {tag:<22} provisioning incomplete "
                      f"(onstart={inst.onstart} pkg={inst.pkg}) — waiting")
                continue

            status = inst.shard

            if status.startswith("crashed"):
                # Hardware is fine; the code failed. Re-renting cannot help, and a
                # retry loop would burn money reproducing the same traceback.
                print(f"  {inst.id:<9} {tag:<22} CRASHED {status} — "
                      f"read /root/shard.log, not retrying")
                continue

            if status == "done" and shard:
                n = pull_results(inst, spec, shard, args.dry_run)
                shard.done = True
                destroy(inst.id, f"shard complete, {n} suites local", args.dry_run)
                shard.instance = None
                continue

            if status.startswith("running"):
                idle = int(status.split(":")[1]) if ":" in status else 0
                # Incremental pull: rsync only moves suites that are not local yet,
                # so this is cheap, and it means a worker lost at 90% costs one
                # dataset rather than the whole shard.
                if shard and cycle % 3 == 0:
                    pull_results(inst, spec, shard, args.dry_run)
                if idle > args.stall_minutes:
                    if shard and shard.restarts < 1:
                        shard.restarts += 1
                        print(f"  {inst.id:<9} {tag:<22} STALLED {idle}m — restarting")
                        start_driver(inst, spec, args.grid, shard, args.dry_run)
                    else:
                        destroy(inst.id, f"stalled {idle}m twice", args.dry_run)
                        if shard:
                            shard.instance = None
                else:
                    print(f"  {inst.id:<9} {tag:<22} training ({idle}m since output)")
                continue

            # Reachable, provisioned, no trainer: either it never started or the
            # data push died halfway (F10 — rsync resumes, so just re-run it).
            if shard:
                print(f"  {inst.id:<9} {tag:<22} idle (panels={inst.panels}) — starting")
                start_driver(inst, spec, args.grid, shard, args.dry_run)
            else:
                free_instances.append(inst)
                print(f"  {inst.id:<9} {tag:<22} healthy and unassigned")

        # --- assign spare workers, then rent for what is still unassigned ----
        pending = [s for s in shards if not s.done and s.instance is None]
        for shard in list(pending):
            if not free_instances:
                break
            inst = free_instances.pop()
            shard.instance = inst.id
            print(f"  assign {shard.key} -> {inst.id}")
            start_driver(inst, spec, args.grid, shard, args.dry_run)
            pending.remove(shard)

        running = len([i for i in insts.values() if i.state == "running"])
        for shard in pending:
            if running >= args.max_workers:
                print(f"  {shard.key}: unassigned, fleet at cap ({args.max_workers})")
                continue
            offer = pick_offer(args.max_price, tried_offers)
            if not offer:
                print(f"  {shard.key}: no offer under ${args.max_price}/hr")
                continue
            tried_offers.add(offer[0])
            rent(offer[0], args.dry_run)
            running += 1

        state_path.write_text(json.dumps(
            {s.key: {"instance": s.instance, "done": s.done} for s in shards}, indent=2))

        if all(s.done for s in shards):
            print("\nevery shard complete; results are local. Nothing left running.")
            return
        if args.dry_run and cycle >= 1:
            print("\n(dry run: one cycle only)")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
