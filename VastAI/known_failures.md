# Known failure modes on rented workers

Every entry here cost real debugging time and real billed hours. The point of the
catalogue is that none of them should ever be diagnosed from scratch again: each
gives the **symptom** as it actually appears, the **cause**, the **check** that
detects it (all of them are implemented in `healthcheck.sh`), and the **fix**.

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

## F11 — The Valendin benchmark cannot take engineered time features

**Symptom.** The shard crashes immediately with

    ValueError: ValendinEmbedder has no covariate path, but seq_cols carries
    2 non-embedded column(s): ['week_sin', 'week_cos']

**Cause.** Not a bug — ADR-0004 freezes the published architecture, which reads
embedded features only. A grid whose `PanelConfig` sets
`time_features={"add_week_sin_cos": True}` puts plain numeric columns into
`seq_cols`, and the frozen embedder correctly refuses them.

**Fix.** A design decision, not a repair: either drop `valendin_lstm` from that
grid, or drop the engineered time features from the grid's `PanelConfig` — which
changes what *every* model in the grid sees, so it is not a per-model workaround.
Using `ProjectedEmbedder` instead would unfreeze the benchmark and is not an
option. **Check this before renting:** a grid that pairs `valendin_lstm` with
`add_week_sin_cos` will burn a worker to discover it.

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
