# 16 — The refit's learning rate and batch size cite an ADR that does not say it

**Status:** needs-triage

## Doc claim

`src/panelclv/studies/runner.py:72-74`:

> ```python
> # Loss controls `refit_best_trial` accepts. `n_epochs` / `batch_size` / learning rate
> # are deliberately NOT here: the refit owns those (ADR-0008 — a few big-batch epochs,
> # not the trial's schedule). Only the objective is shared.
> ```

## Code reality

ADR-0008 contains neither "big-batch" nor any statement about the optimiser. Read in full
(`docs/adr/0008-forecast-from-a-refit.md`), it decides exactly one thing: that the forecast
comes from a warm-start fine-tune over the full calibration window rather than from the
winning trial's checkpoint as it stands. Its only quantitative sentence is `:23` — "a fixed
few epochs with no validation set and therefore no early stopping".

The refit's optimiser settings are defaults in `refit_best_trial`
(`src/panelclv/trials/refit.py:42-44`):

```python
batch_size: int = 512,
learning_rate: float = 1e-3,
weight_decay: float = 1e-3,
```

and `studies/runner.py:75` forwards only `_REFIT_LOSS_KEYS`, so the study's winning
`learning_rate` — searched over `(1e-4, 3e-3, "log")`
(`src/panelclv/registry/model_registry.py:337`) — is discarded. A trial selected at the
bottom of that range is fine-tuned at **10× its own selected rate**.

The asymmetry is deliberate and visible: the *loss* is forwarded
(`runner.py:78-98`, added in "Give the refit the loss its study was tuned under"), the
optimiser is not.

## Why this is `needs-triage` rather than a doc fix

Two different things could be wrong, and I cannot tell which from the code:

**(a) The comment over-attributes.** The decision is fine, ADR-0008 just does not record it.
Fix: state the reason in the comment on its own terms (a warm-start fine-tune wants a fresh,
uniform schedule, not the schedule that was tuned for from-scratch training), or add it to
ADR-0008 as a consequence so the citation becomes true.

**(b) The default is wrong.** If the argument that carried the loss over — "give the refit
the objective its study was tuned under" — applies to the learning rate too, then discarding
a searched `1e-4` in favour of `1e-3` is a real modelling defect, and it is silent: nothing
warns, nothing records it. Worth a quick check on whether refit loss curves diverge for
low-`learning_rate` winners before deciding.

If (a): a two-line edit. If (b): the fix is to forward `learning_rate` (and possibly
`weight_decay`) from `best_trial.params` and to say so in ADR-0008.
