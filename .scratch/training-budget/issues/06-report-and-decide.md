# 06 — Read the result and decide what changes

Status: `ready-for-human`
Blocked by: 03

## Read

`python scripts/run_training_budget.py --report`, against the family N electronics rows in
`docs/benchmarks-real-panels.md`: ValendinLSTM bias +46.0 ± 14.8, MAPE 70.8, RMSE 0.3770,
Spearman 0.032; Pareto/NBD bias −63.0, MAPE 65.7, Spearman 0.297.

Judge on **MAPE and per-customer Spearman**. Bias is not a deciding metric: the refit
noise floor on electronics is 8.9 points of sd (`docs/benchmarks-real-panels.md`, "The
refit noise floor, and the stopping epoch").

## Then decide, in order

1. **If `paper` beats `archive` clearly** — the training recipe is the problem. Decide
   whether to add `32` to the registry's `batch_size` set
   (`registry/model_registry.py:373`) and whether family N is re-run. Both change
   published numbers, so both go through an ADR entry.
2. **If `floor50` matches `paper`** — the floor is the general fix, and it is the one the
   developed models can use, since they have no published recipe. Decide whether
   `min_epochs` becomes a default in the study runners rather than an opt-in.
3. **If neither beats `archive`** — record that in `docs/training-budget.md`, close this
   effort, and leave the inputs (`docs/insights-cluster-ablation.md` §5.1) as the standing
   explanation for the collapse.

Whatever the outcome, update `docs/training-budget.md` §8 with what was run and what it
showed, and register the family in `docs/studies-run.md` (ticket 04).

## The result (20 September 2026)

160 suites on vast.ai, $0.44, complete. Full tables in `docs/training-budget.md` §9;
per-suite scores in `results/family_t_scores.csv`.

| model | arm | MAPE | vs archive | Spearman | vs archive |
| --- | --- | ---: | ---: | ---: | ---: |
| ValendinLSTM | `archive` | 69.1 | — | 0.027 | — |
| | `paper` | 68.5 | p = 0.76 | 0.016 | p = 0.70 |
| | `paper90` | **46.8** | p = 2×10⁻⁶ | **0.177** | p = 5×10⁻⁶ |
| | `floor50` | 47.9 | p = 1×10⁻⁶ | 0.168 | p = 1×10⁻⁵ |
| LSTM | `archive` | 59.3 | — | 0.030 | — |
| | `paper` | 63.5 | p = 0.15 | 0.020 | p = 0.20 |
| | `paper90` | 54.7 | p = 0.10 | **0.174** | p = 2×10⁻⁶ |
| | `floor50` | **50.1** | p = 1×10⁻⁴ | 0.074 | p = 2×10⁻³ |

**The recipe's settings do nothing; its epoch count does everything.** `paper` is
indistinguishable from `archive`. `paper90` — the same pinned recipe with a 90-epoch floor
— cuts MAPE by 22 points and multiplies Spearman by seven on the benchmark.

## Decisions this forces, in order

1. **Make `min_epochs` a default rather than an opt-in.** `floor50` improves both models
   with the existing search untouched, and it is the only form of the fix available to
   models with no published recipe. Open question: what floor, and whether it scales with
   the panel (50 epochs at batch 256 on electronics is 200 updates).
2. **Decide what happens to the published electronics rows.** They are reproduced exactly
   by `archive`, and they are 22 MAPE points worse than the same architecture trained
   properly. Re-running family N under a floor changes `docs/benchmarks-real-panels.md`,
   so it needs an ADR entry recording that the schedule changed and why.
3. **Test the other three panels before generalising.** CDNOW leaves the most validation
   loss on the table (10.7-13.0%) and is the panel whose ranking never collapsed — it is
   the one that says whether this is a property of sparse panels or of electronics.
4. **Do not add batch 32 to the registry's search space.** `paper` settles it: at
   patience 7 the search stops a small-batch trial early anyway, and under a floor the
   batch size is not what does the work.

## What it does not overturn

Inputs still dominate ranking: a cluster label reaches 0.27 and Pareto/NBD 0.297, against
0.177 here. `docs/insights-cluster-ablation.md` §5.1 stands.

## The cause, measured after the run

`paper` stopping at epoch 1 is **our split, not the panel**. Running the notebook's recipe
under the notebook's own customer-wise 10% split on electronics trains 34-62 epochs (best
epoch 28-56), in the same range as the ~90 it reports on its banking data
(`paper_split_check.py`, `results/paper_split_check.csv`).

So the finding is not that Valendin et al.'s protocol is wrong. It is that **this
package's temporal validation split (ADR-0001) and the paper's patience rule do not work
together**: a temporal window over a 98.6%-zero panel is flat from epoch 1, and
a patience rule with an absolute 10⁻⁴ improvement threshold reads a curve gaining
5.4×10⁻⁵ an epoch as converged (`docs/training-budget.md` §13.2; `min_delta=0` is the
notebook's Keras setting, not ours). Every neural result in the repository was
trained under that combination.

That makes a fifth decision, and it outranks decision 1:

5. **A floor is a patch for a stopping criterion that no longer fits.** The principled fix
   is a criterion suited to a flat temporal curve — a relative `min_delta`, a longer
   patience scaled to the curve, or rollout-based selection
   (`docs/insights-study.md` §5.4). Untested. Whoever takes decision 1 should decide
   whether the floor is the answer or the stopgap.

## The selection question, answered (21 September 2026)

80 studies, 2,813 trials scored twice, $1.67. `docs/training-budget.md` §14.

Validation cross-entropy is **wrong-signed** against the holdout: rho −0.141 on MAPE
(p = 6e-7) and −0.264 on |bias| (p = 2e-13). A study's lowest-CE trial is a worse choice
than one of its trials picked at random. The mechanism is in §14.2 — the best in-window
fit carries the calibration era's purchase rate forward into a holdout year whose rate is
lower, so it over-predicts most (bias +35.1 in the best-CE quartile against +22.0 in the
worst).

A validation-window rollout beats it at p < 1e-7, and lands on zero. The largest
correlation anywhere in the test is +0.128. So:

6. **Swap the selection criterion for a validation-window rollout scored on the metric
   being reported** — rollout MAPE for level, rollout Spearman for ranking. Keep
   cross-entropy as the training loss.
7. **Do not build the composite** (§13.3 A2). Averaging three ranks scores worse than
   whichever one matches the target.
8. **Keep the ordering.** Selection is worth a couple of MAPE points; the training budget
   was worth twenty. Decisions 1-3 above come first.
