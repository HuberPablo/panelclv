#!/usr/bin/env python3
"""vast_search.py — find vast.ai machines suited to panelclv training.

Why this exists instead of a one-line `vastai search offers`:

    The panelclv workload is CPU-bound, not GPU-bound. The electronics panel is
    829 customers x 208 periods, so at batch 256 an epoch is ~4 batches — far too
    small to fill any modern GPU. And `run_monte_carlo_forecast` loops simulations
    sequentially in Python (600 sims x 52 autoregressive steps = ~31k tiny forward
    passes), which is dominated by kernel-launch latency, i.e. by CPU single-thread
    speed. VRAM demand is trivial: one simulation path is (829, 52) float32 = 172 KB.

    So the right filter is "any GPU that works, on the fastest CPU I can get" —
    the opposite of a normal GPU search. vast's API can filter the numeric CPU
    fields, but CPU *generation* only shows up as a model-name string in the
    results, so this script filters server-side and then ranks client-side.

Usage:
    python vast_search.py                      # default search, top 15
    python vast_search.py --max-price 0.25     # cap $/hr
    python vast_search.py --max-bandwidth-cost 0.005   # cap $/GB egress
    python vast_search.py --min-cores 16 --top 30
    python vast_search.py --show-fields        # dump one raw offer to see the schema

Nothing here rents anything — it is read-only. Copy an offer ID from the output
into `./VastAI/vast_launch.sh <ID>` when you have picked one.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

# Minimum CUDA the host driver must advertise. panelclv itself pins nothing —
# pyproject declares a bare `torch` — so this floor exists only to exclude hosts
# too old for the CUDA family the PyTorch wheels are built against.
#
# Deliberately NOT raised to match the launcher's image tag (currently a cu128 /
# CUDA 12.9 build). CUDA 12 is minor-version compatible: a 12.x runtime runs on
# any 12.x driver, so requiring the host to advertise 12.9 would drop machines
# that run the image fine. The cheap check is downstream anyway — the CUDA probe
# in vast_onstart.sh aborts provisioning if the GPU is not visible, which costs
# minutes rather than a lost run.
IMAGE_CUDA = "12.4"


# --- CPU ranking table --------------------------------------------------------
# vast exposes `cpu_ghz` on most (not all) offers. Clock alone is a poor proxy
# across generations — a 3.0 GHz Broadwell is far slower per-clock than a 3.0 GHz
# Zen 4 — so we combine a generation tier with the clock. Tiers are a hand-built
# heuristic over the CPU model strings vast actually reports; adjust freely.
#
# tier 3 = current-gen, best single-thread (what you want)
# tier 2 = one or two generations back, still good
# tier 1 = older server silicon, usable but slow per core
# tier 0 = pre-2018 server silicon — this is the Xeon E5 v3/v4 class that makes a
#          launch-bound workload crawl regardless of how good the GPU is
CPU_TIERS: list[tuple[str, int]] = [
    # --- tier 3: modern high-clock desktop / current-gen server ---------------
    (r"ryzen\s*9\s*9\d{3}", 3),        # Zen 5  (9950X, ...)
    (r"ryzen\s*9\s*7\d{3}", 3),        # Zen 4  (7950X, ...)
    (r"epyc\s*9\d{3}", 3),             # Genoa / Bergamo
    (r"core.*i9-1[34]\d{3}", 3),       # Raptor Lake
    (r"xeon.*w[579]-\d{4}", 3),        # Sapphire Rapids workstation
    (r"ultra\s*[579]", 3),             # Core Ultra
    # Threadripper PRO is generation-ambiguous by name, so match the model number
    # explicitly and BEFORE the generic `threadripper` catch-all in tier 1 — the
    # lookup returns on first match, so a 9975WX (Zen 5, 5.5 GHz) would otherwise
    # be scored as tier 1 alongside a 2020-era 3955WX.
    (r"threadripper.*\s9\d{3}wx", 3),  # Zen 5  (9975WX, 9995WX, ...)
    (r"threadripper.*\s7\d{3}wx", 3),  # Zen 4  (7975WX, ...)
    # --- tier 2: previous gen, still strong ----------------------------------
    (r"ryzen\s*9\s*5\d{3}", 2),        # Zen 3  (5950X, ...)
    (r"ryzen\s*[79]\s*3\d{3}", 2),     # Zen 2
    (r"epyc\s*7[0-9]{3}", 2),          # Rome / Milan
    (r"core.*i9-1[12]\d{3}", 2),
    (r"xeon.*gold\s*6[34]\d{2}", 2),   # Ice Lake SP
    (r"xeon.*platinum\s*8[34]\d{2}", 2),
    # --- tier 1: older but not ancient ---------------------------------------
    (r"xeon.*gold\s*6[12]\d{2}", 1),   # Skylake / Cascade Lake SP
    (r"xeon.*silver", 1),
    (r"epyc\s*7[0-9]{2}\b", 1),        # Naples
    (r"threadripper", 1),
    # --- tier 0: avoid for this workload -------------------------------------
    (r"xeon.*e5-", 0),                 # Haswell / Broadwell EP (2014-2016)
    (r"xeon.*e3-", 0),
    (r"opteron", 0),
]

TIER_LABEL = {3: "current", 2: "recent", 1: "older", 0: "ancient", -1: "unknown"}


def cpu_tier(cpu_name: str) -> int:
    """Map a vast CPU model string to a generation tier. -1 when unrecognised."""
    name = (cpu_name or "").lower()
    for pattern, tier in CPU_TIERS:
        if re.search(pattern, name):
            return tier
    return -1


def build_query(args: argparse.Namespace) -> str:
    """Assemble the server-side filter.

    Deliberately permissive on the GPU and strict on everything that actually
    gates this workload. Each clause is `field op value`; they are ANDed.
    """
    clauses = [
        # -- GPU: just needs to exist and run the CUDA image ------------------
        # `>=`, not `=`: a 2- or 4-GPU offer runs this job perfectly well on one
        # of its GPUs. Pinning it to exactly 1 dropped viable machines for no
        # reason — cost is already bounded by dph_total below.
        "num_gpus>=1",
        # A simulation path is 172 KB, so any card qualifies. The floor only
        # exists to drop broken / near-zero-VRAM listings.
        "gpu_ram>=4",
        # The host driver must support the CUDA runtime of the image that
        # vast_launch.sh actually launches. Renting below this gives you a box
        # where torch.cuda.is_available() is False — and vast_onstart.sh then
        # aborts provisioning, so you pay for a machine you cannot use.
        f"cuda_max_good>={IMAGE_CUDA}",

        # -- CPU: the part that decides how long your sweep takes -------------
        f"cpu_cores_effective>={args.min_cores}",
        # UNITS ARE ASYMMETRIC, verified against the live API: the query threshold
        # is in GB, the cpu_ram field in the JSON response is in MB. `cpu_ram>=32`
        # returns machines whose reported cpu_ram is >= 32009, i.e. 32 GB. Do NOT
        # "fix" this by multiplying by 1024 — `cpu_ram>=8192` asks for 8 TB and
        # matches nothing, which is silent because the script just prints
        # "No offers matched".
        f"cpu_ram>={args.min_ram}",

        # -- host quality ------------------------------------------------------
        # Below ~98% reliability, multi-hour runs get interrupted.
        f"reliability>{args.min_reliability}",
        "rentable=true",
        "verified=true",
        # Enough headroom for the image, the repo, the regenerated synthetic grid
        # (~340 MB) and the Studies/ output (~435 MB for a 160-dataset sweep).
        f"disk_space>{args.min_disk}",
        # Sized by the IMAGE PULL (several GB), not by the data: the synthetic
        # grid is regenerated on the box from base_seed rather than transferred.
        "inet_down>=100",
        # Bandwidth is billed PER GB, separately from $/hr, and the image pull is the
        # biggest transfer a worker makes. A host may price egress far above the
        # ~$0.004/GB median; one at $0.039/GB turned a single crash-looping instance's
        # repeated image pulls into a $5.37 line item (F14 in known_failures.md). The
        # $/hr ceiling does not bound this, so it needs its own.
        f"inet_down_cost<{args.max_bandwidth_cost}",

        f"dph_total<{args.max_price}",
    ]
    return " ".join(clauses)


def fetch_offers(vastai: str, query: str, verbose: bool) -> list[dict]:
    """Shell out to the vast CLI and parse its JSON output."""
    cmd = [vastai, "search", "offers", query, "-o", "dph+", "--raw"]
    if verbose:
        print(f"$ {' '.join(cmd[:3])} '{query}' -o dph+ --raw\n", file=sys.stderr)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"vastai search failed (exit {exc.returncode}):\n{exc.stderr}\n\n"
            "If it complains about an unknown field, run `vastai search offers --help` "
            "to see the field list for your CLI version and adjust build_query()."
        )
    except FileNotFoundError:
        sys.exit(f"vastai not found at {vastai!r} — pass --vastai /path/to/vastai")

    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        sys.exit(f"could not parse vastai output as JSON:\n{out.stdout[:500]}")


def score(offer: dict) -> tuple:
    """Sort key: best machine for panelclv first.

    Ranks on CPU generation, then clock, then cores, and only then on price —
    because at these prices ($0.05-0.40/hr) a full sweep costs a few dollars
    either way, so wall-clock is worth far more than cents per hour.
    Returned negated where higher-is-better, so plain ascending sort works.
    """
    tier = cpu_tier(offer.get("cpu_name", ""))
    ghz = float(offer.get("cpu_ghz") or 0)
    cores = float(offer.get("cpu_cores_effective") or 0)
    price = float(offer.get("dph_total") or 99)
    return (-tier, -ghz, -cores, price)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.40, help="max $/hr (default 0.40)")
    ap.add_argument("--min-cores", type=int, default=8, help="min effective CPU cores (default 8)")
    ap.add_argument(
        "--max-bandwidth-cost", type=float, default=0.01,
        help="max $/GB for downloads (default 0.01). Billed separately from $/hr; "
             "the median host is ~$0.004/GB and the image pull is several GB per worker",
    )
    # The panels are tiny (1000 customers x 156 weeks of float32 is ~1 MB), so
    # the real ceiling is the Optuna study plus the MC buffers — hundreds of MB.
    # 8 GB is already generous; 32 was cargo-culted from GPU-training defaults.
    ap.add_argument("--min-ram", type=int, default=8, help="min system RAM in GB (default 8)")
    ap.add_argument("--min-disk", type=int, default=40, help="min disk GB (default 40)")
    ap.add_argument("--min-reliability", type=float, default=0.98)
    ap.add_argument("--top", type=int, default=15, help="rows to print (default 15)")
    ap.add_argument("--vastai", default=shutil.which("vastai") or
                    "/home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/vastai",
                    help="path to the vastai CLI")
    ap.add_argument("--show-fields", action="store_true",
                    help="print one raw offer as JSON and exit (to inspect the schema)")
    ap.add_argument("--verbose", action="store_true", help="echo the query sent to vast")
    args = ap.parse_args()

    offers = fetch_offers(args.vastai, build_query(args), args.verbose)

    if not offers:
        sys.exit("No offers matched. Loosen --max-price / --min-cores, or drop "
                 "verified=true in build_query().")

    if args.show_fields:
        print(json.dumps(offers[0], indent=2, sort_keys=True))
        return

    offers.sort(key=score)

    hdr = f"{'ID':>10}  {'$/hr':>6}  {'CPU':<34} {'gen':<8} {'GHz':>4} {'cores':>5} {'RAM':>5}  {'GPU':<16} {'rel':>6}  loc"
    print(f"\n{len(offers)} matching offers — ranked by CPU first, price last\n")
    print(hdr)
    print("-" * len(hdr))

    for o in offers[: args.top]:
        tier = cpu_tier(o.get("cpu_name", ""))
        print(
            f"{o.get('id', 0):>10}  "
            f"{float(o.get('dph_total') or 0):>6.3f}  "
            f"{str(o.get('cpu_name', '?'))[:34]:<34} "
            f"{TIER_LABEL[tier]:<8} "
            f"{float(o.get('cpu_ghz') or 0):>4.1f} "
            f"{int(o.get('cpu_cores_effective') or 0):>5} "
            f"{int(float(o.get('cpu_ram') or 0) / 1024):>4}G  "
            f"{str(o.get('gpu_name', '?'))[:16]:<16} "
            f"{float(o.get('reliability') or 0) * 100:>5.1f}%  "
            f"{o.get('geolocation', '?')}"
        )

    print(
        "\nPick from the top rows, not the cheapest row: a 'current'/'recent' CPU is\n"
        "worth several times its price premium here, because a full sweep costs only\n"
        "a few dollars either way but can differ by days in wall-clock.\n\n"
        "Then, from the repo root:  ./VastAI/vast_launch.sh <ID>\n\n"
        "Use the launcher rather than a bare `vastai create instance`: create only\n"
        "ALLOCATES — the container does not boot, the image is not pulled and\n"
        "--onstart never runs until you separately `start` it, and a key attached\n"
        "after creation is not injected. vast_launch.sh does create -> attach ssh ->\n"
        "start -> poll until cur_state=running, then prints the ssh command.\n"
    )


if __name__ == "__main__":
    main()

# Run from the repo root:
#
# python VastAI/vast_search.py --verbose
# Echoes the query actually sent to vast before the results.
#
# python VastAI/vast_search.py --show-fields
# Dumps one raw offer as JSON. Run this first if anything errors — it tells you what fields your CLI version actually returns.
#
# python VastAI/vast_search.py --max-price 0.15 --min-cores 16 --top 30
# Tightens price, widens the core requirement, shows more rows.
