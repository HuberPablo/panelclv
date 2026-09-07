# Known failure modes on rented workers

Every entry here cost real debugging time and real billed hours. The point of the
catalogue is that none of them should ever be diagnosed from scratch again: each
gives the **symptom** as it actually appears, the **cause**, the **check** that
detects it, and the **fix**.

**Where the checks live.** Only five are in `healthcheck.sh` (F1, F2, F3, F9, F10) —
it probes one box at a time for `state / ssh / onstart / cuda / pkg / data / shard`.
F4 and F12 are caught at launch by `launch/vast_launch.sh`, before the image pull is
paid for. F17 and F18 pass every one of those checks and are caught by
`choose/vast_search.py` and `supervise/pin_workers.sh` instead. F19 and F20 are about
the *grid*, not a box, so `scripts/reconcile_grid.py` is their check. The rest are
scripting-hygiene rules with no automated check at all. An earlier version of this
paragraph claimed all of them lived in `healthcheck.sh`, which was never true and
made the four it cannot see look covered.

A worker that fails a check is either repaired or destroyed — never left running.
An idle rented box costs exactly as much as a working one.

Add an entry whenever a new failure costs more than a minute to understand.

---

## F1 — SSH goes to vast's proxy instead of the machine

**Symptom.** `Permission denied (publickey)`, or a bare `Connection closed by
<ip> port <port>`, against the ssh line that `vast_launch.sh` printed. The box is
`running` and perfectly healthy; `vastai attach ssh` reports the key is *already
associated*.

**Cause.** `show instances` reports `ssh_host`/`ssh_port` as vast's proxy
(`sshN.vast.ai`), and the launcher printed those even though it creates instances
with `--direct`. The proxy does not reliably accept an instance-attached key. The
machine's own `public_ipaddr`, on the host port mapped to the container's
`22/tcp`, accepts it immediately.

**Check.** `ssh` — tries the direct endpoint first and reports which endpoint
answered.

**Fix.** Use `public_ipaddr` + `ports["22/tcp"][0].HostPort`. Fixed in
`vast_launch.sh`, which now prints the direct endpoint and labels the proxy as a
fallback. *Nothing is wrong with the box — do not destroy it for this.*

---

## F2 — GPU driver too old for the pinned image (CUDA error 804)

**Symptom.** `vast_onstart.sh` aborts with `FATAL: no CUDA device visible`, so
`/root/.onstart_done` never appears and nothing is installed. Confusingly,
`nvidia-smi` on the same box lists the GPU. The real error only shows in the log:

    CUDA error 804: forward compatibility was attempted on non supported HW

**Cause.** The host's NVIDIA driver is older than the CUDA runtime in our image
(`cu128` / CUDA 12.9). `nvidia-smi` talks to the driver and works; the CUDA
runtime refuses to initialise.

Note this contradicts the reasoning in `vast_search.py`, which sets
`IMAGE_CUDA = "12.4"` and argues that CUDA 12 minor-version compatibility makes a
higher floor unnecessary. Minor-version compatibility still requires a driver at
least as new as the runtime; error 804 *is* that requirement being violated. The
search floor should track the image tag.

**Check.** `cuda` — runs a real allocation on the device rather than trusting
`nvidia-smi`, and surfaces the CUDA error verbatim.

**Fix.** Not repairable from our side. Destroy the instance and rent another. To
prevent it, raise the CUDA floor in `vast_search.py` to match the image tag.

---

## F3 — SSH key never injected into the container

**Symptom.** `Permission denied (publickey)` on **both** the direct endpoint and
the proxy, while `vastai attach ssh` insists the key is already associated.

**Cause.** Unknown at vast's end; the key is registered against the instance but
absent from the container's `authorized_keys`.

**Not fixable by:** rebooting the instance (tried — the key is still missing
afterwards), or `vastai execute` (it accepts only a whitelisted command set, so
you cannot append to `authorized_keys` out of band).

**Check.** `ssh` — distinguishes an auth failure from a network failure, so this
reads as "key rejected" rather than "box not up yet".

**Fix.** Destroy and re-rent. Debugging costs more than the replacement.

---

## F4 — A swallowed key-attach failure

**Symptom.** A box that is unreachable after a full, billed image pull, with no
error anywhere in the launch log.

**Cause.** `vast_launch.sh` ran `vastai attach ssh ... || true`, discarding the
result. A failed attach was indistinguishable from a successful one until someone
tried to connect, several minutes and several cents later.

**Check.** Caught at launch now, not by the health check.

**Fix.** Fixed in `vast_launch.sh`: the attach response is inspected, and only
`'success': True` or `already associated` is allowed through. Anything else is
fatal before the image pull is paid for.

---

## F5 — The onstart CUDA probe loses a race with its own driver

**Symptom.** `FATAL: no CUDA device visible` on a box that is fine seconds later.

**Cause.** `vast_onstart.sh` probed `torch.cuda.is_available()` exactly once, and
the container can start before the GPU is visible to it.

**Fix.** Fixed in `vast_onstart.sh`: the probe now polls for 100 s before giving
up. Still fatal afterwards, because a sweep that silently ran on CPU wastes the
whole rental. Note this is a *different* failure from F2, which no amount of
waiting resolves — the distinguishing evidence is CUDA error 804 in the log.

---

## F6 — `ssh` inside a `while read` loop eats the worklist

**Symptom.** A loop over N workers checks only the first one or two, then exits
silently with no error.

**Cause.** `ssh` reads stdin, which inside `while read ... done < list.txt` is the
worklist itself. It consumes the remaining lines.

**Fix.** Always `ssh -n` (or `< /dev/null`) in a loop. Applies to any script here
that iterates over workers.

---

## F7 — `pkill -f <pattern>` kills the shell running it

**Symptom.** A command that ends in `pkill -f foo.sh` dies mid-way with an odd
exit code, and later commands in the same shell never run.

**Cause.** `pkill -f` matches against full command lines, including the command
line of the shell that is executing the `pkill` itself.

**Fix.** Bracket a character so the pattern cannot match itself:
`pkill -f "[f]oo.sh"`.

---

## F8 — Confirmation prompts with no TTY

**Symptom.** `vastai destroy instance <id>` prints `Aborted.` and the instance
keeps billing.

**Cause.** The command prompts for confirmation; a non-interactive shell answers
nothing, so it aborts. It is easy to read `Aborted.` as "done".

**Fix.** Always pass `-y` to `vastai destroy instance`. Verify with
`vastai show instances` rather than trusting the command's output.

---

## F9 — A crashed shard reports itself as finished

**Symptom.** The health check reads `shard=done` on a box that has been up for
seconds. The results are not there.

**Cause.** `start_shard.sh` ran the trainer and then touched a done-marker
unconditionally (`python … ; touch /root/.shard_done`), so the marker said only
"the command returned", not "the command succeeded".

**Check.** `shard` — reads `/root/.shard_exit`, which now records the exit status,
and reports `CRASHED:<code>` for anything non-zero.

**Fix.** Fixed in `start_shard.sh`: `echo $? > /root/.shard_exit`. A marker that
cannot distinguish success from failure is worse than no marker, because it is
trusted.

---

## F10 — rsync dies with a broken pipe mid-transfer

**Symptom.** The driver log ends with
`rsync: [sender] write error: Broken pipe (32)`, and the box has partial or no
data. The health check reports `no-data`.

**Cause.** The SSH connection dropped during the 340 MB push. Rented boxes have
variable network quality.

**Fix.** Re-run the driver. The push uses `rsync --partial`, so it resumes rather
than restarting, and files already transferred are skipped. No cleanup needed.

---

## F11 — The Valendin benchmark cannot take ANY non-embedded channel

**Widened 2026-09-06.** This entry said "engineered time features". That is one
instance of the rule, not the rule, and reading it narrowly is what let the defect
survive a second grid. AR features trip the identical guard.

**Symptom.** The shard crashes immediately with

    ValueError: The Valendin benchmark reads embedded features only, but seq_cols
    carries 2 non-embedded column(s): ['week_sin', 'week_cos']. The published model
    has no covariate path (ADR-0004).

**Cause.** Not a bug. `benchmarks/valendin_lstm.py:99` computes
`[c for c in seq_cols if c not in embedded_cols]` and refuses a non-empty result —
ADR-0004 freezes the published architecture, which reads embedded features only.
Three kinds of column land in `seq_cols` unembedded:

| source | embedded? | Valendin can read it? |
|---|---|---|
| the target, and `embedded_cols` entries | yes | yes |
| `cluster_features` (`kmeans_K`) | yes, automatically | yes |
| `time_features` (`week_sin`, `week_cos`, `year_idx`) | no | **no** |
| `ar_features` (counters and `active_in_last_K` flags) | no | **no** |

So the eligibility rule is: **ValendinLSTM runs iff the built dataset has zero
non-embedded `seq_cols`** — which means no AR features *and* no engineered time
features. Cluster features are free.

**How it cost two grids.** `seasonal_4x4x10` declared `valendin_lstm` and set
`add_week_sin_cos`, so every one of its arms refused it, and the arm axis added AR
features that would have refused it anyway. Both runs finished reporting success with
zero ValendinLSTM results, because "declared and empty" and "not owed" look identical
on disk. `scripts/reconcile_grid.py` reports the first as `ABSENT`; nothing reported
that it was *expected*.

**Fix.** A design decision, not a repair. Either drop `valendin_lstm` from the arms it
cannot read, or add an arm it can — one with no AR features and `time_features=None`.
Using `ProjectedEmbedder` instead would unfreeze the benchmark and is not an option.

`scripts/run_real_panel_arms.py` does this properly and is the pattern to copy:
`refuses(model_type, data)` is the one authority, it keys off the **built** dataset
rather than the arm declaration, and three callers use it — the work list (so an
ineligible pair is not schedulable), the preflight (so it is caught locally), and
`--check-complete` (which reports such a cell `n/a` with the reason, not `missing`).
`tests/test_real_panel_arms.py` asserts the work list and the predicate agree in both
directions.

**Check this before renting.** A grid that pairs `valendin_lstm` with any unembedded
channel will burn a worker to discover it — or worse, will not, and will simply
produce nothing.

---

## F12 — `start instance` is queued and never runs

**Symptom.** An instance sits at `cur_state=stopped`, `actual_status=loading`
indefinitely. `vastai show instances` reveals the giveaway: `intended_status` and
`next_state` are both `stopped` — vast does not believe it should be running.
`vast_launch.sh` polls the full 20 minutes and times out.

**Cause.** The host had no free GPU when we asked. `start instance` answers

    Required resources are currently unavailable, state change queued.

and leaves the instance allocated but stopped. It is oversubscribed — someone else
holds the GPU — and the queued start may never fire.

**Check.** Caught at launch: `vast_launch.sh` now reads the `start` response
instead of piping it to `sed`, and destroys the instance immediately rather than
waiting out a boot that will not happen.

**Fix.** Destroy and rent a different offer. Nothing about the box will improve by
waiting. Related to F4: the pattern is the same — a command whose output was
discarded because it "obviously" succeeded.

---

## F13 — `pkill -f` matches the shell even with the bracket trick

**Symptom.** A compound command dies partway through with exit code 144; later
steps never run. This is F7 again, in a form the usual fix does not cover.

**Cause.** `pkill -f "[s]upervise.py"` is safe only if the *whole* command line
contains no plain occurrence of the target. A command that kills a process and then
restarts it mentions the real name in the restart step, so the shell's own command
line matches and `pkill` kills its own shell.

**Fix.** Kill by PID (`kill "$PID"`), or split the kill and the restart into two
separate commands. The bracket trick protects the pattern, not the rest of the line.

---

## F14 — Bandwidth is billed per GB, and a crash-looping box re-pulls the image

**Symptom.** A fleet that produced nothing still emptied the account. The invoice
shows one instance far above all the others:

```
Instance 49577975 download charge: quantity 137.400 GB  rate $0.039/GB  = $5.367
```

Ten rented boxes, none of which finished provisioning, cost $6.67 — of which 80% was
that single line. GPU charges across all ten came to $0.36.

**Cause.** Two things compounding.

*Bandwidth is a separate meter.* vast bills `inet_down_cost` per GB on top of `$/hr`,
and the image pull (several GB) is the largest transfer a worker ever makes. Hosts
price egress independently: the median is ~$0.004/GB and the worst offer on a typical
search is ~$0.026/GB, but the host above charged $0.039/GB. `Rules.md` §8's "$0.10/hr
ceiling" bounds the hourly rate and says nothing about this, and `vast_search.py`
filtered `inet_down>=100` — download *speed*, not price.

*A box that never provisions keeps re-pulling.* 137 GB is roughly twenty pulls of the
pinned image. The instance was restart-looping, and every restart replays the image
pull and the `onstart` `apt`/`pip` downloads. Nothing in the launcher noticed: the
health check only asks whether `/root/.onstart_done` exists yet, which is false for a
slow box and for a looping one alike.

*The window made it worse.* The health wait had been lengthened from 15 to 35 minutes
precisely so that slow-but-progressing boxes were not destroyed (an earlier run killed
nine of them). That fix handed the crash-looper 35 minutes to keep downloading.

**Check.** `vastai show instances` reporting `actual_status: exited` after creation, or
a box still not healthy while others rented at the same moment are running. Bill it back
with `vastai show invoices --raw` and look for a `download charge` line an order of
magnitude above the others.

**Fix.** Three, all needed:

- `vast_search.py` now filters `inet_down_cost<--max-bandwidth-cost` (default $0.01/GB).
- Treat `exited` as terminal. A box seen in that state has failed and is re-pulling on
  every restart; destroy it rather than waiting out the health window.
- **Rent fewer boxes.** The image pull is per worker and dominates a short job. The
  electronics ablation used ~3.5 box-hours of compute spread over seven machines — seven
  image pulls to save wall-clock on a job one box finishes in an afternoon. Worker count
  should be chosen against the *provisioning* cost, not just the compute, which sharpens
  §8's "worker count is chosen from measured per-dataset wall-clock".

---

## F15 — The offer price is not the rental price

**Symptom.** A fleet costs consistently more per hour than the offers said it would, by
roughly the same amount on every machine regardless of GPU.

**Cause.** An offer's `dph_total` prices the GPU rental only. The instance is billed

    dph_total(instance) = dph_total(offer) + disk_gb * storage_cost / 730

`storage_cost` is $/GB/month. At the 40 GB `vast_launch.sh` requests by default and a
typical `storage_cost` of $0.20, that is **$0.011/hr** — which on a $0.041/hr offer is a
**23% markup**, and is enough to reverse a comparison between two machines whose measured
throughput differs by less than that.

Verified against the invoice: instance 49620391 billed a `GPU charge` at rate $0.0507
and a separate `storage charge` at rate $0.0111, summing to the $0.0618 `dph_total` the
API reported, against an offer that advertised $0.0521.

**Check.** Compare `dph_total` from `vastai search offers` with `dph_total` from
`vastai show instances` for the same machine, or read the two charge lines in
`vastai show invoices --raw`.

**Fix.** Two, both applied:

- **Rent less disk.** The image layers live on the *host*, outside the instance's
  writable overlay: a provisioned box rented with 20 GB reports `23M used, 20G avail`.
  Nothing in this workload needs 40 GB — the panels are megabytes and a full grid shard
  writes ~435 MB. `survey_machines.py` defaults to `--disk 20`, saving ~$0.006/hr, which
  is about 11% of a cheap machine's total cost and larger than the difference between the
  best and worst machine choice.
- **Rank on the billed price.** `survey_machines.py` records `offer_dph` and `dph_total`
  separately and computes `$/study` from the latter.

---

## F16 — A vast API hiccup read as "the instance is gone", and destroyed three healthy boxes

**Symptom.** Three independent probes, on three unrelated hosts, all logged
`instance reaped by vast` and destroyed their machines within four seconds of each other.
All three boxes had been `running` minutes earlier and were provisioning normally.

**Cause.** A helper that looked an instance up in `vastai show instances --raw` wrapped
the whole call in `try/except` and returned `None` on any failure. `None` was also what it
returned when the API answered correctly and simply did not list that instance — i.e.
when it had genuinely been reaped. One transient API error therefore looked exactly like
simultaneous reaping of every machine, and the caller's response to reaping is to destroy
and give up.

Three machines can't vanish in the same second. Simultaneity across independent hosts is
the signature: suspect the thing they share (the API client) rather than the things they
don't (the hosts).

**Fix.** Distinguish "the API did not answer" from "the API says it is gone". The lookup
now returns a distinct `API_ERROR` sentinel, and callers retry it rather than treating it
as absence. The same rule applies to SSH: a timeout or dropped connection to a rented box
means "ask again", not "this machine is broken" — an otherwise healthy RTX 3070 was
destroyed two minutes after provisioning because a 180 s timeout on its first
`import torch` escaped as a fatal error.

---

## F17 — A supported-looking GPU with no kernels: `torch.cuda.is_available()` lies

**Symptom.** A box provisions cleanly, `nvidia-smi` lists the GPU, `vast_onstart.sh`
completes and touches `/root/.onstart_done` — and then training dies at the first CUDA
op with

    torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device

**Cause.** PyTorch dropped Maxwell and Pascal from its CUDA 12.8+ builds at 2.8. The
pinned image (`vastai/pytorch:2.10.0-cu128-cuda-12.9`) ships cubins for sm_75 and newer
only, so a GTX 9xx/10xx, Titan X/Xp or P100 has no kernel to run.

This is **not** F2. F2 is an old *driver* (error 804) and fails during provisioning.
Here the driver is current and CUDA initialises fine; only the compiled architecture is
missing. Verified on a rented GTX 1070 with driver 580.173.02 advertising CUDA 13.0:

    capability (6, 1)
    torch.AcceleratorError: CUDA error: no kernel image is available ...

**Why the health check misses it.** `vast_onstart.sh` gates on
`torch.cuda.is_available()`, which only asks whether a device and driver exist. On that
GTX 1070 it returned **True** — `get_device_capability()` printed before the crash. A
check that catches this has to *launch a kernel*, not query for a device.

**Fix.** `vast_search.py` drops these cards client-side (`UNSUPPORTED_GPU`), because
vast's query language has no compute-capability predicate. `survey_machines.py` also
runs a real matmul before committing a shard. Roughly a third of the verified sub-$0.06
market is excluded by this, all of it correctly.

---

## F18 — Workers clone from GitHub, so uncommitted local work produces a wrong run

**Symptom.** None. That is the whole problem. The fleet provisions cleanly, every
health check passes, shards train to completion and write `results.csv` — and the
results answer a different question than the one the orchestrator was configured for.

**Cause.** `vast_onstart.sh` builds the worker's repo with

    git clone --quiet --branch "$BRANCH" https://github.com/HuberPablo/panelclv.git

so a worker runs whatever is on **`origin/main`**, never the orchestrator's working
tree. A grid declaration, a runner change or a new search space that is committed
locally but not pushed — or not committed at all — simply is not there. The worker
falls back to the previous declaration and trains it without complaint.

This nearly cost a full fleet: the arm axis (`grids.Arm`, 6 arms x 160 panels, 100
trials) existed only as uncommitted local edits when ten boxes were launched. Every
one of them would have trained the *previous* single-configuration grid at the
*previous* trial budget. `Rules.md` §9's "provided every worker runs the same commit"
is usually read as a statement about workers agreeing with each other; it is also a
statement about them agreeing with the orchestrator, and only the second half fails
silently.

**Why nothing catches it.** `healthcheck.sh` checks `state / ssh / onstart / cuda /
pkg / data / shard` — every one of which passes on a worker running the wrong commit.
`start_shard.sh` waits for provisioning, pushes the panels and starts the trainer; it
never asks what the checkout contains. There is no check anywhere that compares the
worker's HEAD with the orchestrator's.

**Check.** Before starting any shard:

    git rev-parse HEAD                                    # orchestrator
    ssh -n root@HOST -p PORT 'git -C /root/panelclv rev-parse HEAD'

and refuse to start on a mismatch. `VastAI/supervise/pin_workers.sh <sha>` does the reconciling
form of this — it polls every instance and hard-resets any box off the target commit,
skipping (and reporting) any box whose shard has already started, because resetting
under a running trainer mixes two commits into one result.

**Fix.** Three, in order of how much they buy:

- **Push before launching.** `git status` must be clean and `git ls-remote origin main`
  must equal local HEAD before a single box is rented. This is the whole fix; the rest
  is defence.
- **Do not push to `main` while a fleet is running.** `vast_onstart.sh` replays on every
  instance *restart* and does `git pull --ff-only origin main`, so a mid-run push means a
  restarted box silently runs a different commit than its peers — the exact condition
  §9 requires against.
- **Verify in `start_shard.sh`**, which is the one place that knows both ends.

**Related.** F13 bit while fixing this: the natural probe for "has this box started its
shard?" is `ssh root@HOST 'pgrep -f run_pnbd_grid'`, and the remote shell's own command
line contains that string, so it always matches itself. Every box reports "already
training" and the reconciler resets nothing. Bracket it: `pgrep -f "[r]un_pnbd_grid"`.

---

## F19 — A shard reports success with suites untrained

**Symptom.** Every shard exits 0, the state file says `done`, the fleet empties, and
the grid is short. On `seasonal_4x4x10` one arm finished at **151/160**: four suites
had no directory at all, five had a directory holding `config.json` and an empty model
folder but no `results.csv`.

**Cause.** Nothing compares what a shard *owed* against what it *produced*.
`supervise/reap_finished.sh` verifies that every `results.csv` the worker holds is also
local — a real gate, but its denominator is the worker's own disk, so a box destroyed
mid-suite passes it trivially. `supervise/watch_fleet.sh` prints a fleet-wide count
whose denominator omits the models that run on the orchestrator, so it read
`2071/1920` — apparently over-complete — while nine suites were missing.

The five half-written suites are the signature: `scripts/run_pnbd_grid.py` calls
`run_study_suite` with no `try`/`except`, so a suite that *raised* would have exited
non-zero. A suite that leaves `config.json` and nothing else was interrupted from
outside — the box went away underneath it.

**Why the losses cluster.** They are not spread evenly. `pareto_nbd_simulation.py`
builds the manifest with a lexicographic `sorted(glob(...))`, so `Dataset_5_60` and
`Dataset_5_80` sort **last**. Any pass cut off before the end loses the same tail every
time, which is why all nine sat in the same transaction-rate column. Combined with F20's
un-seeded replacements — which restart a stride from the top — a shard that is replaced
twice will truncate that identical tail twice.

**Check.**

    python scripts/reconcile_grid.py --grid seasonal_4x4x10

It expands the grid's declared (model, arm, dataset) product, compares it against the
`results.csv` files on disk, and exits non-zero on any shortfall. Run it before calling
a run finished and before reading its results — "the fleet is empty" is not the same
statement as "the grid is complete".

**Fix.** Recover with a targeted resume rather than a re-run: `run_pnbd_grid.py` skips
any suite whose `results.csv` exists and passes `overwrite=True` for the rest, so

    scripts/run_pnbd_grid.py --grid <grid> --model <type> --arm <arm> --shard 1/1

trains only what is missing, half-written directories included. Seed the worker first
(see F20) or it will retrain the whole arm.

**Related.** `collect_grid_results` skips any *dataset* with no `results.csv` and said
nothing, so an arm missing nine panels reads as a complete arm with fewer rows. Since
a grid *cell* averages the ten replicate panels sharing a `(rate, churn)` coordinate,
one model's cell mean then rests on the panels that finished while another's rests on
all ten. It now warns.
Until it did, a grid analysis compared one model's *finished* panels against another's
full set and drew the wrong winner in several cells.

---

## F20 — The fleet state file drifts, in both directions

**Symptom.** `VastAI/state/<grid>.json` disagrees with reality both ways at once. After
one run it marked `transformer:4/8` and `transformer:6/8` as `"done": false` against
instance IDs that no longer existed — while five *other* shards it marked `done` were
each missing suites.

**Cause.** `supervise.py` is the only writer of that file, and it is no longer the only
process that retires a box. `reap_finished.sh`, `pin_workers.sh` and a human running
`vastai destroy` all act outside its bookkeeping, so once `supervise.py` stops polling —
or is stopped — the file freezes at whatever it last believed. Both those shards had in
fact finished cleanly, hours after the supervisor's last cycle:

    [19:16:31] 49936444 shard finished (exit=0) — pulling before anything else
    [21:49:58] 49957817 shard finished (exit=0) — pulling before anything else

**Check.** Do not read the state file to answer "is this shard done". Read the disk:
`supervise.py`'s `shard_is_complete()` reproduces the arm-major stride and checks for
each owed `results.csv`, and `scripts/reconcile_grid.py` does it for the whole grid.

**Fix.** Treat the file as a cache, never as the authority. `supervise.py` now
recomputes every shard's `done` from disk at startup, which is what its docstring always
promised and what makes a restart cheap — before this, restarting it re-rented a box for
every shard that had already finished.

**Related.** The same divergence made replacement workers expensive. `start_driver`
seeds a new box with the suites already collected so `run_pnbd_grid.py` can skip them,
but it read them from the un-suffixed path (the bug behind the 107-suite loss in
`supervise/reap_finished.sh`'s header), so replacements arrived empty and restarted their
stride from the top. Seeding needs only the `results.csv` files — 61 KB against a 2.6 GB
tree, and bandwidth is billed per GB (F14).

## F21 — Half a fleet rented with no ssh key injected, and the launcher took 20 minutes each to notice

**Symptom.** 14 boxes launched; 8 provisioned and trained, 6 answered every ssh attempt
with `root@sshN.vast.ai: Permission denied (publickey)`. `start_shard.sh` did not report
that. It reported

    [ssh8.vast.ai:18030 transformer 10/12] FATAL: never finished provisioning

after polling `test -f /root/.onstart_done` 80 times at 15 s. Because the starter walks
its host list serially, six such boxes cost **two hours of wall-clock before the first
one was even declared dead**, and they then sat rented until a human looked. ~7 idle
hours across 6 machines, roughly half the run's total spend, for nothing.

**Cause.** Two separate things, and the second is what made the first expensive.

`vast_launch.sh` creates, attaches the key, starts and polls to `running` — but
`running` is the *container's* state, and it is reached whether or not the key landed.
When many instances are created back to back some do not get the key, and nothing in
the launch path asks.

`start_shard.sh`'s first act is the provisioning wait, whose failure mode ("no
`.onstart_done` after 20 minutes") is indistinguishable from a slow image pull. A box
that will never accept a connection and a box still pulling a 4 GB image look identical
to it, so it gives both the full 20 minutes.

**Check.** One ssh round-trip, immediately after launch, before any work is assigned:

    ssh -n -i ~/.ssh/id_ed25519 -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 \
        "root@$HOST" true

A box that cannot answer that in the first few minutes will not answer in twenty. Probe
the whole fleet in a loop with a couple of retries, keep what answers, and destroy the
rest **immediately** — a keyless box is worth nothing and bills like any other. On the
recovery run this separated 6 good boxes from 2 dead ones within seconds of launch.

**Fix.** Probe-then-assign, never assign-then-discover. The provisioning wait is for
boxes that have already proved they are reachable; it must not be the thing that finds
out they are not. Also worth remembering that vast's own `actual_status: running` is not
evidence the box is usable — F17 makes the same point about `torch.cuda.is_available()`.

## F22 — vast returns two endpoint formats, and parsing the launcher's stdout loses the fleet

**Symptom.** A launcher loop that recorded `<instance> <ip> <port>` per box wrote an
empty `fleet.txt` while 14 instances were live and billing. Every line of its log read
`no endpoint; skipping` directly under a successful launch.

**Cause.** `vast_launch.sh` prints whatever endpoint vast gives it, and vast gives two
different shapes depending on the machine:

    ready ip=84.249.79.112 port=30221          # direct
    ssh -i ~/.ssh/id_ed25519 -p 17940 root@ssh4.vast.ai   # proxy

A parser written against one silently matches nothing on the other. Nothing failed — the
boxes were fine — but the orchestrator had no record of what it had rented, which is the
state in which machines get forgotten.

**Check.** Never parse the launcher's stdout for the fleet list. Ask the API, which
always reports both fields in the same place:

    vastai show instances --raw | python3 -c 'import json,sys; [print(i["id"], i["ssh_host"], i["ssh_port"]) for i in json.load(sys.stdin)]'

**Fix.** Build the fleet file from `show instances`, filtered to `actual_status ==
"running"`. It is also the only list that survives the launcher dying halfway.

## F23 — A changed study constant makes suite names lie, and the orchestrator keeps both generations

**Symptom.** After re-declaring `run_real_panel_arms.py` from 50 to 200 Monte Carlo
paths, the orchestrator's `Studies/` held 29 suites at 50 paths and 12 at 200 —
**under the same names**. `real_panel_arms__Transformer__cdnow__no_ar-no_cluster-valendin__a`
existed twice over, meaning two different things.

**Cause.** A suite root is named from (experiment, model, panel, arm, shard). None of
those is the path count, the trial count or the study count, so changing one produces
results that collide with the previous generation's by name while being incomparable to
them. The workstation copy was moved to `Studies/_archive_50sim/` when the constant
changed; the orchestrator's copy was not, and `pull_results.sh` then merged new suites
into a tree that still held old ones.

**Check.** Read the generation off `config.json`, never off the path:

    python3 -c 'import json,glob,collections; print(collections.Counter(json.load(open(c))["n_simulations"] for c in glob.glob("Studies/<exp>__*/config.json")))'

More than one value in that counter means the tree is mixed and no aggregate over it is
meaningful.

**Fix.** When a study constant changes, archive the previous generation **on every
machine that holds a copy**, in the same commit that changes the constant — the
orchestrator and the workstation both. And never rsync a mixed tree back wholesale;
filter to the current generation first. (`_archive_50sim` deliberately does not match
the `real_panel_arms__*` glob, so the runner, `--report` and `--check-complete` all stop
seeing the old generation the moment it is moved.)

## F24 — `vastai destroy instance` aborts without `-y`

**Symptom.** A destroy loop printed `Aborted.` for every instance and killed none, while
reporting nothing that looked like an error. The boxes kept billing.

**Cause.** The CLI prompts `Are you sure you want to destroy instance N? [y/N]` and a
non-interactive stdin answers no.

**Check / Fix.** `vastai destroy instance -y <id>`. Pass `-y` in every script; verify by
re-reading `show instances` rather than by trusting the command's output.

## F25 — Diagnosing a supervisor from the instance count instead of its log

**Symptom.** Seven of eight working boxes had disappeared while six broken ones
remained. Read as "a stale reaper from the previous run is destroying my fleet", and the
reaper and puller were killed on that basis.

**Cause.** The inference was wrong, and the evidence against it was one file away.
`VastAI/state/reap_finished.log` said plainly:

    [04:35:36] 50097941 shard finished (exit=0) — pulling before anything else
    [04:35:37]   verified all 2 suites present locally — destroying 50097941

The boxes were gone *because they had finished*, results pulled and verified first. The
reaper was the only thing keeping the run from billing idle machines all night, and
killing it was the actual damage.

**Check.** Before stopping any long-running supervisor, read its log. These processes
narrate what they do precisely so that "why did the fleet shrink" is answerable without
guessing. A shrinking fleet is the reaper's *success* condition, not a symptom.

**Fix.** Restart what was killed, and confirm exactly one of each is running — `pkill -f`
is unreliable here (F13), so kill by PID from `pgrep -af` and re-check. Duplicated
reapers race on the same instances.
