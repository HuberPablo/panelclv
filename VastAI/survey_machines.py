#!/usr/bin/env python3
"""survey_machines.py — rank rentable machines by MEASURED cost per replication.

Why this exists
---------------
`vast_search.py` ranks offers by a hand-built CPU-generation table, on the theory
that this workload is launch-latency bound and therefore follows single-thread CPU
speed. Measured on identical transformer shards, that theory picks the *worst*
machines: the two highest-clock CPUs in the first distributed fleet were 49% and
95% slower than a 3.0 GHz EPYC (`.scratch/worker-scheduling/spec.md`). Published
specs do not predict this workload's throughput, so the only defensible ranking is
one built from a stopwatch.

This script is that stopwatch. For each candidate offer it rents the machine, runs
a fixed, real slice of the actual workload, and records

    $/study = dph_total * seconds_per_study / 3600

which is the quantity total cost is governed by. Hourly price is not: across the
first fleet, $/hr varied 20% while $/suite varied 2.5x.

What it measures
----------------
Three replications of one CDNOW AR-encoding arm — the same command a rented worker
runs for real, only shorter:

    run_ar_encoding_ablation.py --panel cdnow --arm ar_bounded_16 --shard a \
        --n-studies 3 --suite-suffix probe

Deliberately NOT a reduced-trial smoke test. `--n-trials` and `--n-simulations`
keep their real values, so a second of probe time is a second of production time
and the number transfers directly. Three studies rather than one because per-study
wall-clock has a 16-22% standard deviation (measured across 20 studies on two live
workers), which would otherwise swamp a comparison.

`--shard a` fixes base_seed=42 on every machine, so all of them drive the Optuna
sampler and the Monte Carlo forecast down the same path. It does NOT make the work
identical — training is unseeded by design (CLAUDE.md priority 3: weight init,
DataLoader shuffling and dropout draw on the global torch RNG), so early stopping
still varies. It removes the largest source of between-machine variance, not all
of it.

Feasibility filters, and why these and no others
------------------------------------------------
The candidate pool is deliberately unfiltered on anything that expresses a *taste*
in machines — CPU generation, core count, RAM, reliability, bandwidth price. Those
are exactly the priors this survey exists to test, and filtering on them would
sample only the machines the old rule already likes.

What is filtered is what makes a machine unable to produce a measurement at all:

  * GPU architecture. PyTorch dropped Maxwell and Pascal from its CUDA 12.8+
    builds at 2.8, and the pinned image is `pytorch:2.10.0-cu128`. A GTX 10-series
    card provisions fine, passes `vast_onstart.sh`'s `torch.cuda.is_available()`
    check, and then dies at the first kernel launch with "no kernel image is
    available". Renting one buys a bill and no number.
  * Driver CUDA below the image's runtime — F2, error 804.
  * Disk below what `vast_launch.sh` asks for; the create simply fails.

Pass `--include <id>` to override all of that for a named offer, which is how you
keep a negative control (one Pascal box) in the sample.

Cost and safety
---------------
Every rented instance is destroyed: on success, on failure, on watchdog expiry,
and on Ctrl-C. `--budget` caps projected spend and refuses to launch past it.
A box seen `exited` is destroyed immediately rather than waited out — F14, where a
crash-looping instance replayed its image pull twenty times and billed $5.37 of
bandwidth on a $0.10/hr rental.

Usage
-----
    python VastAI/survey_machines.py --max-price 0.05 --budget 6.00
    python VastAI/survey_machines.py --max-price 0.05 --include 47635953  # + control
    python VastAI/survey_machines.py --dry-run          # show the pool, rent nothing
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BENCH_CSV = HERE / "machine_benchmarks.csv"

# The panel the probe trains on. Small enough (2.6 MB) that pushing it to every
# worker costs nothing measurable, which is why the probe uses a real panel rather
# than a synthetic stand-in.
PANEL = REPO / "Datasets" / "Dataset_clean" / "cdnow_customer_week_panel.csv"
REMOTE_REPO = "/root/panelclv"
REMOTE_PANEL_DIR = f"{REMOTE_REPO}/Datasets/Dataset_clean"

PROBE_CMD = (
    "scripts/run_ar_encoding_ablation.py "
    "--panel cdnow --arm ar_bounded_16 --shard a "
    "--n-studies {n_studies} --suite-suffix probe"
)

# torch 2.8+ cu128 wheels ship sm_75 and up. Anything older has no kernels, however
# healthy the box looks. Matched on the GPU name because vast reports no capability
# field; `--include` exists for when this table is wrong.
UNSUPPORTED_GPU = re.compile(
    r"(GTX\s*(9|10)\d{2}|TITAN\s*X|TITAN\s*V|\bP100\b|\bP40\b|\bP4\b|\bM40\b|\bM60\b|"
    r"Quadro\s*[PM]\d|\bV100\b|\bK80\b)",
    re.I,
)

# The pinned image's CUDA runtime. F2: minor-version compatibility still requires a
# driver at least as new as the runtime, so a host advertising less than this hits
# `CUDA error 804: forward compatibility was attempted on non supported HW`.
IMAGE_CUDA = 12.9

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={Path.home()}/.ssh/known_hosts_vast",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=20",
    "-o", "BatchMode=yes",
]

print_lock = threading.Lock()


def log(tag: str, msg: str) -> None:
    with print_lock:
        print(f"[{dt.datetime.now():%H:%M:%S}] {tag:<24} {msg}", flush=True)


# --- vast plumbing ------------------------------------------------------------

def vastai_bin() -> str:
    return shutil.which("vastai") or f"{Path.home()}/thesis-agent/venv/bin/vastai"


def search_offers(max_price: float) -> list[dict]:
    """Verified, rentable offers under the price ceiling — and nothing else.

    Every other clause `vast_search.py` applies encodes a belief about which
    machines are fast, which is the belief under test here.
    """
    query = f"verified=true rentable=true dph_total<{max_price}"
    out = subprocess.run(
        [vastai_bin(), "search", "offers", query, "-o", "dph+", "--raw"],
        capture_output=True, text=True, timeout=180, check=True,
    )
    return json.loads(out.stdout)


# Returned when the vast API could not be reached or did not parse, as distinct
# from a successful call that listed no such instance. Conflating the two destroys
# healthy machines: three concurrent probes once lost their boxes in the same second
# because one API hiccup read as "all three were reaped".
API_ERROR = object()


def instance_row(instance_id: int):
    """The instance's live record, None if vast says it is gone, API_ERROR if it did
    not answer.

    Callers must treat API_ERROR as "unknown, ask again" — never as absence.
    """
    try:
        out = subprocess.run(
            [vastai_bin(), "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=90, check=True,
        )
        rows = json.loads(out.stdout)
    except Exception:
        return API_ERROR
    for r in rows:
        if str(r.get("id")) == str(instance_id):
            return r
    return None


def destroy(instance_id: int, tag: str) -> None:
    """Destroy unconditionally. Billing runs until this succeeds, so it is retried."""
    for attempt in range(3):
        try:
            subprocess.run(
                [vastai_bin(), "destroy", "instance", str(instance_id), "-y"],
                capture_output=True, text=True, timeout=90, check=True,
            )
            log(tag, f"destroyed {instance_id}")
            return
        except Exception as exc:
            log(tag, f"destroy attempt {attempt + 1} failed: {exc}")
            time.sleep(5)
    log(tag, f"!! COULD NOT DESTROY {instance_id} — destroy it by hand")


def launch(offer_id: int, tag: str, key: Path, disk: int) -> tuple[int | None, str]:
    """Rent one offer via vast_launch.sh, which does create -> attach key -> start.

    Returns (instance_id, launcher output). The id is parsed even on failure so a
    partially-created instance can still be destroyed.
    """
    env = dict(os.environ, VAST_KEY=str(key), VAST_DISK=str(disk))
    proc = subprocess.run(
        [str(HERE / "vast_launch.sh"), str(offer_id)],
        capture_output=True, text=True, timeout=1800, env=env, cwd=str(REPO),
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"instance id:\s*(\d+)", out)
    iid = int(m.group(1)) if m else None
    if proc.returncode != 0:
        log(tag, f"launch failed (rc={proc.returncode}): {out.strip().splitlines()[-1][:120]}")
    return iid, out


def ssh_endpoint(row: dict) -> tuple[str, str]:
    """Prefer the container's mapped port on the public IP; fall back to vast's proxy.

    Same reasoning as `vast_launch.sh`: despite --direct, the sshN.vast.ai proxy
    often refuses the instance-attached key while the direct address accepts it.
    """
    ports = (row.get("ports") or {}).get("22/tcp") or []
    if ports and ports[0].get("HostPort"):
        return row["public_ipaddr"], str(ports[0]["HostPort"])
    return row["ssh_host"], str(row["ssh_port"])


def ssh(host: str, port: str, key: Path, cmd: str, timeout: int,
        attempts: int = 3) -> subprocess.CompletedProcess:
    """Run one command on the box, retrying transient failures.

    Rented boxes drop connections and stall (F10), and a first `import torch` on a
    cold host can take minutes. Letting `subprocess.run`'s TimeoutExpired escape
    turns either into a destroyed machine and a lost measurement — which is how an
    otherwise healthy RTX 3070 was thrown away two minutes after provisioning. A
    timeout here means "ask again", not "this machine is broken".
    """
    argv = ["ssh", "-i", str(key), "-p", port, *SSH_OPTS, f"root@{host}", cmd]
    last = subprocess.CompletedProcess(argv, 255, "", "no attempt made")
    for attempt in range(attempts):
        try:
            last = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            if last.returncode == 0:
                return last
        except subprocess.TimeoutExpired:
            last = subprocess.CompletedProcess(argv, 124, "", f"timed out after {timeout}s")
        if attempt < attempts - 1:
            time.sleep(10)
    return last


# --- the probe ----------------------------------------------------------------

def parse_study_times(log_text: str) -> list[float]:
    """Seconds between consecutive `A new study created` lines in Optuna's log.

    Intervals, not a total: the first study starts only after the panel is read and
    the dataset built, so `total / n` would charge every machine a fixed setup cost
    it does not pay per replication. n studies yield n-1 intervals.
    """
    stamps = [
        dt.datetime.strptime(m, "%Y-%m-%d %H:%M:%S")
        for m in re.findall(
            r"\[I (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\] A new study created", log_text
        )
    ]
    return [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]


def wait_provisioned(iid: int, host: str, port: str, key: Path, tag: str, window: int) -> bool:
    """Poll for /root/.onstart_done, but treat `exited` as terminal.

    F14: a crash-looping box and a slow box both simply lack the marker file. The
    looper replays the image pull on every restart, so waiting one out is how a
    $0.10/hr rental ran up $5.37 of bandwidth. `exited` distinguishes them.
    """
    deadline = time.time() + window
    unreachable = 0
    while time.time() < deadline:
        row = instance_row(iid)
        if row is API_ERROR:
            # The API, not the box, is what failed. Keep waiting; only give up if it
            # stays unreachable, and say so rather than blaming the machine.
            unreachable += 1
            if unreachable >= 10:
                log(tag, "vast API unreachable for ~3 min — abandoning this probe")
                return False
            time.sleep(20)
            continue
        unreachable = 0
        if row is None:
            log(tag, "instance reaped by vast")
            return False
        status = row.get("actual_status")
        if status == "exited":
            log(tag, "actual_status=exited — terminal (F14), not waiting it out")
            return False
        r = ssh(host, port, key, "test -f /root/.onstart_done && echo DONE", 40)
        if "DONE" in r.stdout:
            return True
        time.sleep(20)
    log(tag, f"not provisioned within {window // 60} min")
    return False


def check_gpu_usable(host: str, port: str, key: Path, tag: str) -> tuple[bool, str]:
    """Launch a real kernel, because `torch.cuda.is_available()` does not.

    On an architecture the wheels have no cubin for, `is_available()` still returns
    True and `vast_onstart.sh` therefore passes — the failure only appears when a
    kernel is launched. This runs one.
    """
    probe = (
        "import torch;"
        "cap=torch.cuda.get_device_capability();"
        "x=torch.randn(64,64,device='cuda');"
        "y=(x@x).sum().item();"
        "print('CAPOK', torch.cuda.get_device_name(0), 'sm_%d%d' % cap)"
    )
    # 300s, not 180: this is the first CUDA context on a cold host, and on a slow box
    # it is legitimately slow rather than broken.
    r = ssh(host, port, key, f"cd {REMOTE_REPO} && /venv/main/bin/python -c \"{probe}\"", 300)
    blob = r.stdout + r.stderr
    if "CAPOK" in r.stdout:
        return True, r.stdout.strip().split("CAPOK")[-1].strip()
    reason = "no kernel image" if "no kernel image" in blob else blob.strip()[-160:]
    log(tag, f"GPU unusable: {reason}")
    return False, reason


def probe_machine(offer: dict, key: Path, n_studies: int, watchdog: int,
                  provision_window: int, disk: int) -> dict:
    """Rent one offer, time n_studies replications on it, destroy it, return a row.

    Always destroys. The result dict always has a `status`, so a failed probe is
    recorded as evidence about that machine rather than silently dropped.
    """
    oid = offer["id"]
    tag = f"{oid} {str(offer.get('gpu_name'))[:12]}"
    row = {
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "offer_id": oid, "instance_id": "", "gpu_name": offer.get("gpu_name"),
        "cpu_name": offer.get("cpu_name"), "cpu_ghz": offer.get("cpu_ghz"),
        "cpu_cores": offer.get("cpu_cores_effective"),
        "offer_dph": offer.get("dph_total"), "dph_total": offer.get("dph_total"),
        "disk_gb": "", "storage_cost": "",
        "inet_down_cost": offer.get("inet_down_cost"),
        "cuda_max_good": offer.get("cuda_max_good"), "reliability": offer.get("reliability"),
        "geolocation": offer.get("geolocation"), "workload": "cdnow/ar_bounded_16",
        "n_studies": n_studies, "n_intervals": "", "s_per_study": "", "sd_s": "",
        "usd_per_study": "", "provision_s": "", "status": "", "note": "",
    }
    started = time.time()
    iid = None
    try:
        log(tag, f"launching  ${offer['dph_total']:.4f}/hr  {offer.get('cpu_name')}")
        iid, _ = launch(oid, tag, key, disk)
        if iid is None:
            row["status"] = "launch_failed"
            return row
        row["instance_id"] = iid

        live = instance_row(iid)
        for _ in range(5):                       # tolerate a transient API failure
            if live is not API_ERROR:
                break
            time.sleep(15)
            live = instance_row(iid)
        if live is None or live is API_ERROR:
            row["status"] = "reaped" if live is None else "api_unreachable"
            return row
        host, port = ssh_endpoint(live)

        # Bill from the INSTANCE, not the offer. An offer's dph_total excludes disk;
        # the instance's includes it at disk_gb * storage_cost / 730 per hour, which
        # for the launcher's default 40 GB is ~$0.011/hr — a 23% markup on a $0.041
        # offer. Ranking on the advertised price understates every cheap machine's
        # cost by roughly a fixed amount, which is exactly the term that decides a
        # close comparison.
        row["dph_total"] = live.get("dph_total", offer.get("dph_total"))
        row["disk_gb"] = live.get("disk_space")
        row["storage_cost"] = live.get("storage_cost")
        log(tag, f"billing ${row['dph_total']:.4f}/hr "
                 f"(offer ${offer['dph_total']:.4f} + {live.get('disk_space')}G disk)")

        if not wait_provisioned(iid, host, port, key, tag, provision_window):
            row["status"] = "provision_failed"
            return row
        row["provision_s"] = round(time.time() - started)
        log(tag, f"provisioned in {row['provision_s']}s")

        ok, note = check_gpu_usable(host, port, key, tag)
        row["note"] = note[:120]
        if not ok:
            row["status"] = "gpu_unusable"
            return row

        # The panel is gitignored, so the clone has no data. 2.6 MB, seconds.
        ssh(host, port, key, f"mkdir -p {REMOTE_PANEL_DIR}", 60)
        rs = subprocess.run(
            ["rsync", "-az", "--partial", "-e", f"ssh -i {key} -p {port} {' '.join(SSH_OPTS)}",
             str(PANEL), f"root@{host}:{REMOTE_PANEL_DIR}/"],
            capture_output=True, text=True, timeout=600,
        )
        if rs.returncode != 0:
            row["status"] = "data_push_failed"
            row["note"] = rs.stderr.strip()[-120:]
            return row

        # Start the probe DETACHED and poll for it, rather than holding one long-lived
        # ssh open for the duration. A dropped connection (F10) otherwise loses the
        # measurement on a box that was running fine — which is exactly what happened
        # to the first anchor probe: "Connection to ssh5.vast.ai closed by remote host"
        # after the box had already provisioned in 81s.
        log(tag, f"probing ({n_studies} studies)")
        start = (
            f"cd {REMOTE_REPO} && rm -f /root/probe.log /root/probe.done && "
            f"setsid nohup sh -c 'timeout {watchdog} /venv/main/bin/python "
            + PROBE_CMD.format(n_studies=n_studies)
            + " > /root/probe.log 2>&1; touch /root/probe.done' < /dev/null "
            "> /dev/null 2>&1 & echo STARTED"
        )
        if "STARTED" not in ssh(host, port, key, start, 120).stdout:
            row["status"] = "probe_start_failed"
            return row

        # Poll for the marker file. Each call is short, so a dropped connection costs
        # one poll rather than the whole measurement.
        probe_deadline = time.time() + watchdog + 300
        log_text = ""
        while time.time() < probe_deadline:
            time.sleep(30)
            r = ssh(host, port, key,
                    "test -f /root/probe.done && echo DONE; "
                    "grep 'A new study created' /root/probe.log 2>/dev/null | tail -40", 90)
            if r.stdout.strip():
                log_text = r.stdout
            if "DONE" in r.stdout:
                break
        intervals = parse_study_times(log_text)
        if len(intervals) < 1:
            row["status"] = "no_timing"
            row["note"] = (pr.stdout + pr.stderr).strip()[-120:]
            return row

        mean = sum(intervals) / len(intervals)
        var = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        row.update(
            n_intervals=len(intervals),
            s_per_study=round(mean, 1),
            sd_s=round(var ** 0.5, 1),
            usd_per_study=round(row["dph_total"] * mean / 3600, 6),
            status="ok",
        )
        log(tag, f"{mean:.0f} s/study  ->  ${row['usd_per_study']:.5f}/study")
        return row
    except Exception as exc:
        row["status"] = "error"
        row["note"] = f"{type(exc).__name__}: {exc}"[:120]
        return row
    finally:
        if iid is not None:
            destroy(iid, tag)


# --- selection and reporting --------------------------------------------------

def feasible(offer: dict, min_disk: int) -> str | None:
    """None if the machine can produce a measurement, else why it cannot."""
    gpu = str(offer.get("gpu_name") or "")
    if UNSUPPORTED_GPU.search(gpu):
        return f"{gpu}: no kernels in torch 2.8+ cu128"
    try:
        if float(offer.get("cuda_max_good") or 0) < IMAGE_CUDA:
            return f"driver CUDA {offer.get('cuda_max_good')} < image {IMAGE_CUDA} (F2)"
    except (TypeError, ValueError):
        return "cuda_max_good unreadable"
    if float(offer.get("disk_space") or 0) < min_disk:
        return f"disk {offer.get('disk_space')}G < {min_disk}G"
    return None


def write_rows(rows: list[dict]) -> None:
    """Append to the benchmark table, so evidence accumulates across runs.

    Reconciles the schema rather than appending blindly. The table is meant to be
    read by later runs, and a column added between runs used to be written in the
    new field order under the old header — every value after the insertion point
    shifted by one, silently, with no error at write time and no error at read time
    either. Reading the header first and rewriting on a mismatch is the difference
    between a table that accumulates evidence and one that accumulates corruption.
    """
    old_rows: list[dict] = []
    old_fields: list[str] = []
    if BENCH_CSV.exists():
        with BENCH_CSV.open(newline="") as fh:
            reader = csv.DictReader(fh)
            old_fields = list(reader.fieldnames or [])
            old_rows = [dict(r) for r in reader]

    new_fields = list(rows[0].keys())
    if old_fields and old_fields != new_fields:
        # Union, preserving the old order and appending what is new, then rewrite
        # every row under it. Missing cells are blank, which is honest: that run did
        # not record the column.
        merged = old_fields + [f for f in new_fields if f not in old_fields]
        for r in old_rows:
            r.pop(None, None)
        with BENCH_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=merged, extrasaction="ignore")
            w.writeheader()
            w.writerows({f: r.get(f, "") for f in merged} for r in old_rows + rows)
        return

    exists = BENCH_CSV.exists()
    with BENCH_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=new_fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def report(rows: list[dict], baseline_s: float) -> None:
    ok = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: r["usd_per_study"])
    bad = [r for r in rows if r["status"] != "ok"]
    print("\n" + "=" * 104)
    print(f"MEASURED — baseline to beat: {baseline_s:.0f} s/study, 2x bar = {baseline_s * 2:.0f} s")
    print("=" * 104)
    hdr = (f"{'offer':>10} {'$/hr':>7} {'s/study':>8} {'sd':>6} {'vs base':>8} "
           f"{'$/study':>9} {'save':>6}  {'GPU':<14} {'CPU':<28}")
    print(hdr); print("-" * len(hdr))
    for r in ok:
        ratio = r["s_per_study"] / baseline_s
        save = 1 - r["usd_per_study"] / (0.0911 * baseline_s / 3600)
        flag = "" if ratio <= 2 else "  OVER 2x"
        print(f"{r['offer_id']:>10} {r['dph_total']:>7.4f} {r['s_per_study']:>8.0f} "
              f"{r['sd_s']:>6.0f} {ratio:>7.2f}x {r['usd_per_study']:>9.5f} "
              f"{save * 100:>5.0f}%  {str(r['gpu_name'])[:14]:<14} {str(r['cpu_name'])[:28]:<28}{flag}")
    if bad:
        print(f"\nno measurement ({len(bad)}):")
        for r in bad:
            print(f"{r['offer_id']:>10}  {r['status']:<18} {str(r['gpu_name'])[:16]:<16} {r['note'][:60]}")
    print(f"\nappended {len(rows)} rows to {BENCH_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.05, help="$/hr ceiling (default 0.05)")
    ap.add_argument("--budget", type=float, default=6.0, help="max projected spend, $ (default 6)")
    ap.add_argument("--n-probe", type=int, default=8, help="machines to probe (default 8)")
    ap.add_argument("--n-studies", type=int, default=3, help="replications timed per box (default 3)")
    ap.add_argument("--parallel", type=int, default=8, help="probes in flight at once")
    ap.add_argument("--watchdog", type=int, default=2700,
                    help="seconds before the probe is killed (default 2700)")
    ap.add_argument("--provision-window", type=int, default=1500,
                    help="seconds to wait for /root/.onstart_done (default 1500)")
    ap.add_argument("--baseline", type=float, default=132.0,
                    help="s/study on the machine being compared against (default 132, "
                         "the RTX 3070 / EPYC 7302P at $0.0911/hr)")
    # Disk is billed at disk_gb * storage_cost / 730 per hour on top of the offer
    # price. The launcher asks for 40 GB; the probe needs the image (~10 GB), the
    # repo and a 2.6 MB panel, so the default here is deliberately lower.
    ap.add_argument("--disk", type=int, default=20, help="disk GB to rent (default 20)")
    ap.add_argument("--min-disk", type=int, default=40,
                    help="reject offers whose host has less free disk than this")
    ap.add_argument("--include", type=int, action="append", default=[],
                    help="probe this offer id even if judged infeasible (negative controls)")
    # Selecting a machine by offer id is structurally racy: vast re-issues ids
    # continuously, and two consecutive searches a minute apart share almost none of
    # them. So a target is named by DESCRIPTION and resolved inside the same search
    # that will be rented from. `--match` is a regex over "<gpu> <cpu> cuda<version>".
    ap.add_argument("--match",
                    help="regex over '<gpu_name> <cpu_name> cuda<cuda_max_good>'; "
                         "probes the cheapest --n-probe offers that match")
    ap.add_argument("--allow-infeasible", action="store_true",
                    help="let --match reach offers the feasibility veto excluded — "
                         "this is how a negative control (a Pascal box, a driver below "
                         "the image's CUDA) gets measured instead of assumed")
    ap.add_argument("--key", default=f"{Path.home()}/.ssh/id_ed25519")
    ap.add_argument("--dry-run", action="store_true", help="print the pool and exit")
    args = ap.parse_args()

    if not PANEL.exists():
        sys.exit(f"FATAL: no panel at {PANEL} — the probe has nothing to train on.")

    offers = search_offers(args.max_price)
    pool, rejected = [], []
    for o in offers:
        why = feasible(o, args.min_disk)
        (rejected if (why and o["id"] not in args.include) else pool).append((o, why))

    print(f"\n{len(offers)} verified offers under ${args.max_price:.4f}/hr — "
          f"{len(pool)} can run the pinned image, {len(rejected)} cannot\n")
    hdr = (f"{'offer':>10} {'$/hr':>7} {'$/GB':>7} {'GPU':<15} {'CPU':<30} "
           f"{'cores':>5} {'cuda':>5} {'rel':>5}  loc")
    print(hdr); print("-" * len(hdr))
    for o, _ in pool:
        print(f"{o['id']:>10} {o['dph_total']:>7.4f} {(o.get('inet_down_cost') or 0):>7.4f} "
              f"{str(o.get('gpu_name'))[:15]:<15} {str(o.get('cpu_name'))[:30]:<30} "
              f"{int(o.get('cpu_cores_effective') or 0):>5} {str(o.get('cuda_max_good')):>5} "
              f"{float(o.get('reliability') or 0) * 100:>4.1f}  {o.get('geolocation')}")
    if rejected:
        print(f"\nexcluded ({len(rejected)}):")
        for o, why in rejected:
            print(f"{o['id']:>10} {o['dph_total']:>7.4f}  {str(o.get('gpu_name'))[:15]:<15} {why}")

    candidates = [o for o, _ in (pool + rejected)] if args.allow_infeasible \
        else [o for o, _ in pool]
    if args.match:
        pat = re.compile(args.match, re.I)
        candidates = [o for o in candidates
                      if pat.search(f"{o.get('gpu_name')} {o.get('cpu_name')} "
                                    f"cuda{o.get('cuda_max_good')}")]
        if not candidates:
            sys.exit(f"\nNothing in this search matches {args.match!r}.")
    chosen = sorted(candidates, key=lambda o: o["dph_total"])[: args.n_probe]
    if not chosen:
        sys.exit("\nNothing feasible to probe.")

    # Projected worst case: every box bills for the full provision window plus the
    # full watchdog, and pulls the image once. Bandwidth is the term F14 says the
    # $/hr ceiling does not cover, so it is priced in explicitly.
    hours = (args.provision_window + args.watchdog) / 3600
    image_gb = 8.0
    projected = sum(o["dph_total"] * hours + image_gb * (o.get("inet_down_cost") or 0)
                    for o in chosen)
    print(f"\nprobing {len(chosen)} machines, {args.n_studies} studies each")
    print(f"projected worst case: ${projected:.2f}  (budget ${args.budget:.2f}) — "
          f"typical will be far less, since a healthy box finishes long before the watchdog")
    if projected > args.budget:
        sys.exit(f"\nREFUSING: projected ${projected:.2f} exceeds --budget ${args.budget:.2f}.")
    if args.dry_run:
        print("\n--dry-run: nothing rented.")
        return

    rows: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool_exec:
            futures = [pool_exec.submit(probe_machine, o, Path(args.key), args.n_studies,
                                        args.watchdog, args.provision_window, args.disk)
                       for o in chosen]
            for f in futures:
                rows.append(f.result())
    except KeyboardInterrupt:
        print("\ninterrupted — check `vastai show instances` and destroy anything left.")
        raise
    finally:
        if rows:
            write_rows(rows)
            report(rows, args.baseline)
        print("\nremaining instances (should be none from this survey):")
        subprocess.run([vastai_bin(), "show", "instances"], timeout=90)


if __name__ == "__main__":
    main()
