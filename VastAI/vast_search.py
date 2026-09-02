#!/usr/bin/env python3
"""vast_search.py — find vast.ai machines suited to panelclv training.

Why this exists instead of a one-line `vastai search offers`:

    The panelclv workload is CPU-bound, not GPU-bound. An epoch is a handful of
    batches, and `run_monte_carlo_forecast` loops simulations sequentially in Python
    — thousands of tiny autoregressive forward passes dominated by kernel-launch
    latency. VRAM demand is trivial.

    MEASURED, 2026-09-02, seven machines timed on identical work (see
    VastAI/machine_benchmarks.csv):

      * CPU *generation* decides throughput. AMD Zen 2/3 (EPYC Rome, Ryzen 5000)
        averaged 169 s/study against 249 s for Xeon E5 v3/v4 — a 1.48x ratio,
        r = +0.94 with the family.
      * GPU tier barely matters. An RTX 3080 on a Xeon E5-2620 v4 ran 262 s/study;
        an RTX 3060 on an EPYC 7452 ran 175 s. Two tiers of GPU lost to the CPU.
      * `cpu_ghz` carries no usable signal once generation is known: r = +0.21. It
        was the secondary sort key here, spending rank on a field that does not
        predict anything. (A smaller sample of the same data gave r = +0.61, which
        reads as "clock is actively harmful" — that was an artifact of every
        high-clock machine in it happening to be a Xeon. Generation is the effect;
        clock only correlates with it.)
      * `cpu_cores_effective` carries no signal either: r = +0.30, and the slow group
        alone spans 4 to 32 cores.

    So: filter on feasibility, rank on CPU generation, and ignore clock and cores.
    Generation only shows up as a model-name string, so that part is client-side.

    This script cannot tell you which individual offer is cheapest per unit work —
    only a stopwatch can. `VastAI/survey_machines.py` is that stopwatch.

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

# Minimum CUDA the host driver must advertise: the runtime of the image
# vast_launch.sh pins (vastai/pytorch:2.10.0-cu128-cuda-12.9). Keep the two in step.
#
# This was 12.4, on the argument that CUDA 12 minor-version compatibility lets a 12.x
# runtime run on any 12.x driver. That argument is wrong, and renting on it costs a
# machine every time: minor-version compatibility still requires the driver to be at
# least as new as the runtime. MEASURED 2026-09-02 — an RTX 3060 whose host advertised
# CUDA 12.8 failed against this image with
#
#     Error 804: forward compatibility was attempted on non supported HW
#
# and never provisioned (F2). 12.8 < 12.9 by one minor version and it still failed, so
# the floor is exact, not approximate.
IMAGE_CUDA = "12.9"


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


# GPU architectures the pinned image has no kernels for. PyTorch dropped Maxwell and
# Pascal from its CUDA 12.8+ builds at 2.8, and the wheels in this image ship sm_75 and
# up. A card below that provisions cleanly, passes vast_onstart.sh's
# `torch.cuda.is_available()` check — which only initialises CUDA and says nothing about
# whether a kernel exists for the device — and then dies at the first kernel launch with
# "no kernel image is available for execution on the device".
#
# Matched on the model name because vast publishes no compute-capability field. This
# excludes roughly a third of the verified sub-$0.06 market, so it is the single biggest
# filter here. It is also the least directly tested: the failure is documented by
# PyTorch, but every Pascal offer available when this was written sat on a host that
# refused the ssh key (F3), so the "no kernel image" error has not been reproduced
# in-market. It costs nothing to keep — every Pascal offer in that market was also on a
# Xeon E5 v3/v4, the CPU family measured 1.48x slower, so none of them was a machine
# worth renting anyway.
UNSUPPORTED_GPU = re.compile(
    r"(GTX\s*(9|10)\d{2}|TITAN\s*[XV]|\bP100\b|\bP40\b|\bP4\b|\bM40\b|\bM60\b|"
    r"Quadro\s*[PM]\d|\bV100\b|\bK80\b)",
    re.I,
)


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
        # Cores are NOT that part. Measured r = +0.30 against seconds-per-study, and
        # the slowest group spans 4 to 32 cores; the fastest EPYC measured had 8.
        # The floor is kept only to drop listings reporting absurdly little CPU, and
        # defaults low enough not to exclude anything real.
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
    """Sort key: CPU generation first, then price. Nothing else.

    Clock and core count were in this key until they were measured. Both had to go:
    against seconds-per-study, clock scores r = +0.21 and cores r = +0.30, while the
    generation family scores r = +0.94. Sorting on a field with no signal is not
    neutral — it reorders machines that generation had correctly grouped.
    """
    tier = cpu_tier(offer.get("cpu_name", ""))
    price = float(offer.get("dph_total") or 99)
    return (-tier, price)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.40, help="max $/hr (default 0.40)")
    ap.add_argument("--min-cores", type=int, default=2,
                    help="min effective CPU cores (default 2 — cores do not predict "
                         "throughput here, so this only drops broken listings)")
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

    # The search API cannot express "sm_75 or newer", so this is a client-side pass.
    unsupported = [o for o in offers if UNSUPPORTED_GPU.search(str(o.get("gpu_name") or ""))]
    offers = [o for o in offers if o not in unsupported]
    if unsupported:
        print(f"\ndropped {len(unsupported)} offers whose GPU has no kernels in this image "
              f"(torch 2.8+ cu128 ships sm_75+): "
              f"{', '.join(sorted({str(o.get('gpu_name')) for o in unsupported}))}")
    if not offers:
        sys.exit("No offers left after dropping unsupported GPUs.")

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
        "\nRows are ranked by CPU GENERATION, then price. Prefer AMD Zen 2/3 (EPYC\n"
        "Rome/Milan, Ryzen 5000) over Xeon E5 v3/v4: measured, that is a 1.48x\n"
        "difference in throughput at the same GPU, and it dwarfs the GPU tier. The\n"
        "clock column is shown but does not predict speed. The listed $/hr is the\n"
        "GPU only; the instance is billed\n"
        "$/hr + disk_gb * storage_cost / 730, about +$0.006/hr at the 20 GB the\n"
        "launcher should rent.\n\n"
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
