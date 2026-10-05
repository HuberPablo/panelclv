# The epoch loss is a per-cell mean, whatever the batch size

Until 4 October 2026, `validate_one_epoch` in `training/loop.py` returned the mean of its
per-batch mean losses. The validation loader runs at the trial's own batch size, in fixed
customer order, so the last batch of an epoch is usually short, and averaging batch means
gave each customer in it more weight than the rest. The same weights therefore scored a
different validation loss at each batch size. That loss is what early stopping reads and
what Optuna selects on (ADR-0001), so the search compared trials on a number that moved
with one of the hyperparameters it was searching.

Each batch's mean is now weighted by what it divided by, the number of cells scored
(`_batch_weight`), so the epoch loss is the exact per-cell mean for every batch size.
Class-weighted cross-entropy divides by the summed weights of its targets instead, and is
weighted by that. The training loss is reported the same way, for the same reason.

Shuffling the validation set was considered and rejected. It removes the bias's fixed
direction, but each epoch's score then depends on which customers fall in the short batch:
at batch 256 on the four real panels, a 95% range of about ±3 to ±9% around the true mean,
larger than the differences early stopping and selection act on.

## How large it was

The stored winners of family U, each scored at every batch size with its weights fixed,
against the true per-cell mean (`docs/model-selection.md` §3.9):

| panel | customers in the short batch at 256 | batch 256 | batch 64 |
| --- | ---: | ---: | ---: |
| electronics | 61 of 829 | −4.1% | −0.1% |
| multichannel | 122 of 1,402 | −6.2% | −0.3% |
| cdnow | 53 of 2,357 | +2.8% | +0.2% |
| gift | 14 of 2,062 | +13.4% | +3.0% |

The sign is set by whether the highest-id customers happen to buy less (electronics,
multichannel) or more (CDNOW, gift) than average. Either way it is larger than the 0.08–3.1%
the winning loss varies by across replications.

## Consequences

Every searched result produced before the fix (commit af2b14b) was selected under the
biased score: the `archive` and `floor50` arms of every family, and every benchmark search.
On electronics and multichannel the bias favoured batch 256. Rerunning electronics ·
`archive` · no cluster label with the fix moved the winners off it (18–19 of 20 to 1–2 of
20) and lowered ValendinLSTM's holdout MAPE by 9.4 points; the LSTM still collapses
(`docs/model-selection.md` §3.9). Arms that run one pinned trial select nothing between
trials, so they are affected only through which epoch early stopping keeps.

A validation loss written before the fix is not comparable with one written after it, nor
with another archived one at a different batch size. The archived `val_loss` and
`best_objective_value` fields keep their old meaning.
