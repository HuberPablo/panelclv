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
