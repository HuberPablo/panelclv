# 22 — The loss ablation's "paired on seed" pairs the sampler, not the weights

**Status:** needs-triage

Found as a consequence of issue `01` (training is unseeded), not by re-auditing
`docs/loss-functions.md`. The audit spec marks that file verified clean and warns against
re-chasing the eleven false findings a previous pass reported against it — **this is not one
of them.** Every number it quotes still reproduces from `Studies/loss_ablation_cdnow`; what
is wrong is the sentence explaining what the pairing removes.

## Doc claim

`scripts/run_loss_ablation.py:170-171`, justifying the headline statistic:

> ```python
> # Paired per-seed differences against the baseline arm. Every arm ran study i under
> # the same seed, so differencing on `seed` removes the study-to-study variance that
> ```

`docs/loss-functions.md:784` and `:792` repeat it:

> paired seeds (`base_seed + j` for study `j`), read back with …
> report the paired per-seed difference between arms rather than the difference of means.

And `scripts/run_cdnow_embedding_ablation.py:27` makes the same claim for the embedding
ablation:

> the same seeds — study `j` of every arm uses `base_seed + j` — so the arms are paired

## Code reality

`base_seed + j` reaches two things (issue `01`): the Optuna TPE sampler
(`tuning/optuna_tuning.py:493`, `studies/runner.py:150`) and the Monte Carlo forecast
(`studies/runner.py:178` → `models/monte_carlo_forecasting.py:374`).

Neither ablation script calls `torch.manual_seed`. Confirmed:

```console
$ grep -n "manual_seed" scripts/run_loss_ablation.py scripts/run_cdnow_embedding_ablation.py
$
```

So across arms, study `j` shares its dataset, its sampler stream and its MC draws — but its
**weight initialisation, `DataLoader` shuffle order and dropout masks are independent**.
"Removes the study-to-study variance" overstates what the differencing removes.

## What this does and does not invalidate

**Does not.** The paired test is still a legitimate paired test — the arms are genuinely
matched on dataset and sampler, which is the largest shared component. And the effect sizes
dwarf initialisation noise: `emd` is worse than cross-entropy on `mape_aggregate` by **82.3
points** on average (`docs/loss-functions.md:852-853`) and `ce_emd` by **17.7 points**
(`:892-893`), against a within-arm across-studies SD of 25.6–48.0. The R1 and R2
falsification verdicts stand.

**Does.** The stated *mechanism* is wrong, and it is load-bearing twice over:

1. `docs/loss-functions.md:787-788` uses the across-studies SD as a live power argument for
   how many studies a future ablation needs. If part of that spread is unpaired init noise
   rather than sampler variance, the sizing argument is measuring something other than what
   it names.
2. The "9 of 10 seeds" / "8 of 10 seeds" counts (`:852-853`, `:892-893`) read as sign tests
   over matched pairs. They are matched on less than the sentence claims.

## Fix options

**(a) Correct the mechanism sentence** in both scripts and in `docs/loss-functions.md`
§7 — say the arms are paired on dataset, sampler stream and MC draws, and that weight init is
independent across arms. Leave every number and every verdict. Smallest honest change.

**(b) Also caveat the power argument** at `:787-788`, since it is the one place the SD is
used to decide something rather than to report it.

**(c) Re-run the ablation under genuinely paired training** — a `torch.manual_seed(base_seed + j)`
before each arm's study `j` — and recheck the verdicts. 3 arms × 10 studies × 20 Optuna
trials × 300 simulations. Only worth it if (1) above matters for a future ablation's design.

## Related

Issue `01` — the seeded/unseeded boundary this follows from. Its resolution deliberately
left the training path unseeded, so (c) would be a behaviour change against that decision
and needs it reopened first.
