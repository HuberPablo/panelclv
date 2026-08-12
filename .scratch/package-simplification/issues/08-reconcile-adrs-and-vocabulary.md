# Reconcile the ADRs and CONTEXT.md with the redesign

Type: grilling
Status: resolved
Blocked by: 06

## Question

The ADRs document why the code is shaped as it is. After an open redesign, three of the
five may describe a shape that no longer exists — ADR-0002 (the simulator lives with the
model), ADR-0004 (frozen reference implementations) and ADR-0005 (the embedder seam).
Docs that lie are worse than no docs.

**Audit 04 adds a fourth, and it is a stronger case than the other three:** ADR-0003
(rollout-composite selection) describes a decision that is *unreachable from the production
path*. `studies/runner._run_neural_model` passes neither `selection_metric` nor
`removable_features`, and `StudySuiteConfig` has no field for either, so the rollout selection
the ADR records — and the entire covariate-subset search — can only be reached through three
`run_optuna_study` cells in the two `Data_integration` notebooks, never from `run_studies.py`
or any suite cell. Worse, audit 04 measured that the selection metric ADR-0003 aligns on does
not agree with the reported one: `tuning.weekly_aggregate_rollout_metrics` recomputes RMSE over
customer-summed totals where `compute_forecast_metrics` uses per-cell values (62× apart) and
MAPE under a different estimator (masked at 5.0, clipped at 300); only bias matches. So
ADR-0003's "aligning selection with the metric actually reported" holds for one of three
metrics. Decide whether the ADR is amended to say notebook-only, or the wiring is a bug to fix.

Decide, per ADR: still true / amended / superseded. Then decide what ADR-0006 records —
the target architecture is exactly the kind of decision the format exists for.

**`CLAUDE.md` needs a line corrected too**, and it is the highest-traffic doc in the repo —
every agent session reads it. It states: "`compute_forecast_metrics` is the single scoring
authority — Plots, tables and study results all delegate to it so they agree to the last
decimal." Audit 04 verified that is true *within* `evaluation/` and `studies/` (it reproduced
all nine stored `results.csv` values of an archived suite to full float precision) but false
package-wide: `tuning.weekly_aggregate_rollout_metrics` recomputes all three metrics, and
`evaluation/__init__.py:9-13` makes the same claim scoped narrowly enough to be true. Either
the `tuning` carve-out gets spelled out or that function is renamed off the shared vocabulary —
`rollout_mape` currently sits one word away from `mape_aggregate_style` while computing
something else.

`CONTEXT.md` needs the same pass. It defines *Trial*, *Study* and *Study suite* but
never *experiment*, while `experiments/` is a subpackage — a name outside the project's
own vocabulary is why nobody can say what separates it from `studies/`. Whatever 06 and
09 decide, the vocabulary follows.

Per the charting session: selective, not exhaustive. The synthesis outcome and any
invariant-collapsing decision become ADRs; routine kill/keep calls stay in tickets.

## Answer

Settled with Pablo, 2026-08-12, over three rounds of `/grilling` (Q1-Q9). Every ADR is
ruled on, `CONTEXT.md` has its vocabulary pass, and the frontier is empty.

**One item on this ticket's charter turned out not to need doing.** The Question asks
whether `CLAUDE.md`'s "single scoring authority" line gets a `tuning` carve-out or whether
`weekly_aggregate_rollout_metrics` is renamed off the shared vocabulary. Neither: ticket 06
decision 1 deletes that function outright. Verified by counting what actually computes
these numbers — `src/` holds exactly **two** RMSE computations, `models/monte_carlo_forecasting.py:561`
(the authority) and `optuna_tuning.py:684`, which sits inside the deleted function, as do
all four `rollout_*` names at `:714-717`. The only other metric function is
`evaluation/segment_analysis.aggregate_bias:39`, which `evaluation/__init__.py` already
documents as the deliberate exception. **So the claim simply becomes true, package-wide,
for the first time.** No carve-out, no rename.

### Delivery: two kinds of doc edit, landing at different times

Per Q8, these edits split by whether they are true *on decision* or only *on landing*:

- **Decision records — landed with this ticket's own commit.** ADR-0003's retirement,
  ADR-0004's amendments, and the three `CONTEXT.md` terms. These record decisions already
  made, so publishing them is not a lie, and the vocabulary has to reach ticket 09 while
  ticket 09 still needs it.
- **Structure descriptions — attached to their execution issues**, same rule ticket 07
  set for `CLAUDE.md`: ADR-0006/0007/0008, ADR-0002's new line, ADR-0001's fix, and
  `evaluation/__init__.py`'s rewording all describe code that does not exist yet.

Per Q9 this ticket carries the **full finished text** of everything in the second group, so
an execution issue copies a block rather than interpreting a summary.

### Ruling per ADR

| ADR | ruling |
|---|---|
| 0001 temporal validation split | **amended** — one stale line |
| 0002 simulator lives with the model | **still true, amended** — gains a line recording that it is finally enforced |
| 0003 rollout-composite selection | **retired** — kept in place with a `Retired 2026-08-12` header |
| 0004 frozen reference implementations | **amended** — three edits, all landed now |
| 0005 embedder seam | **still true, unchanged** |

**ADR-0005 needs saying explicitly rather than by omission.** Nothing in tickets 06 or 07
touches the seam. `to_rollout()` shares the backbone and therefore the embedder, which is
the seam working exactly as the ADR describes — the rollout model consumes the same output
width without knowing the strategy. It is the only one of the five that survives an open
redesign untouched.

### What landed with this ticket

**ADR-0003** keeps its file. Deleting it would lose why `rollout_composite` existed and
leave the next person free to re-propose it. The header reads `Retired 2026-08-12`, **not**
`Superseded by` — nothing replaces it; it goes because the feature is deleted, and
"superseded" would send a reader hunting for a successor that does not exist. The header
also states that the code still runs until the execution set lands, so the file is not
ahead of the repository. A "Why it goes" section at the foot records the three things worth
keeping: that it was never reachable from the production path, that all **1256** archived
`selection_metric` values read `val_loss`, and that its "aligns selection with the metric
actually reported" claim held for one metric of three (RMSE 62x apart, MAPE under a
different estimator, only bias agreeing).

**ADR-0004** gains two blocks and loses one.

- *Frozen means the numbers, not the surrounding code.* The ADR said "their architectures
  are frozen" and was silent on the file around them, so the map had to state the real rule
  in its Notes — which is where a rule goes to be forgotten. Ticket 07 needs this: it edits
  `benchmarks/valendin_lstm.py` to declare its own training-to-rollout pairing. The test is
  now named as the two `validate_*.py` scripts rather than as a judgement about whether a
  file was touched.
- *The validation scripts re-implement what they check, deliberately.* Ticket 06 decision 9,
  which had no home outside a ticket. A gate that imports the code it gates stops being a
  gate.
- Its torch-free consequence is **deleted** — see below.

**`CONTEXT.md`** gains three terms and one explicit absence:

- **Registry** (under "The models") — the single table declaring every model. Ticket 09
  flagged that this word had no definition.
- **Rollout model** (under "Forecasting and scoring") — the object that performs a rollout,
  obtained from a trained model. `_Avoid_: inference model, prediction model, sampler`.
  This is the most load-bearing of the three: `CONTEXT.md` already lists *inference* under
  `_Avoid_` while the codebase ships `InferenceMultinomialLSTMModel`, and ticket 09 was
  blocked on having a correct name to rename toward. It now has one.
- **Refit** (under "Experiments") — the warm-start fine-tune over the full calibration
  window that now produces every forecast.
- **No term for "experiment"**, stated as a line under *Study* rather than left implicit.
  *experiment* was already listed under `_Avoid_` there, so defining it would have meant
  reversing a vocabulary ruling to accommodate a folder name. Ticket 09 inherits the
  consequence: **the `experiments/` subpackage is named outside the project's own
  vocabulary and needs a different name.** Ticket 06 decided it survives as a subpackage,
  not what it is called, so this contradicts nothing.

### The torch-free idea is removed, not merely un-adopted

Ruled by Pablo: *"We don't care about the torch free logic, remove the idea of a torch free
logic."* Ticket 06 had routed the question here, having measured the cost at ~1.2 s /
~540 MB with torch a **hard** dependency in `pyproject.toml` — so it never bought the
ability to run without torch.

**Measured, per subpackage** (import it, then check `sys.modules`):

    configs: torch-free          evaluation: TORCH        studies: TORCH
    data_preparation: torch-free experiments: TORCH       training: TORCH
    benchmarks: torch-free       models: TORCH            tuning: TORCH

**The full removal inventory**, so no site is missed:

| site | what it is |
|---|---|
| `benchmarks/__init__.py:15-56` | a PEP 562 `__getattr__` lazy loader, a `_LAZY` map and a `TYPE_CHECKING` block — ~30 lines existing only so importing `benchmarks` skips torch |
| `docs/adr/0004` | the torch-free consequence — **already deleted by this ticket** |
| `CLAUDE.md:46` | "Torch is imported lazily here" |
| `CLAUDE.md:95` | "Data preparation needs only numpy and pandas, so it runs without loading torch" |
| `studies/analysis.py:27` | the same claim in the module docstring |
| `studies/analysis.py:149, 878, 1098` | three deferred imports, each commented "keeps ... torch-free" |

**Three of those were already defeated and nobody noticed.** `panelclv.studies` pulls torch
at package import, so `import panelclv.studies.analysis` loads torch regardless — measured,
it returns `True`. The three deferrals in `analysis.py` have never saved anything, and
`:1098` defers `pandas`, which cannot affect torch at all. That comment was simply wrong.

**Two lazy imports stay, and are not to be swept up by pattern-match.** They have nothing to
do with torch: `training/training_utils.py:11` defers `wandb` and `optuna`, which are
genuinely optional dependencies, and `evaluation/plot_utils.py:248` defers the Pareto MCMC
fitter, which is heavy and pure numpy.

**Consequence for ticket 06:** the registry held *lazy* references so `studies/config.py`
could validate a `model_type` without torch. That justification is gone — and it was
protecting a property `panelclv.studies` does not have. **The registry may hold direct
references**, which makes it simpler. There is no cycle either way, since `models/` does not
import `studies/`. Ticket 13 also loses a candidate: no import test to write.

### The three new ADRs, in full

Per Q5 these are three files rather than one "target architecture" ADR. The twelve decisions
in ticket 06 have twelve different reversal costs; a single bundled file would be a summary,
which is not what the format is for. "Consolidate in place, do not re-partition" folds into
0006's context rather than earning a file of its own.

Each lands with the execution issue that makes it true.

#### `docs/adr/0006-one-registry-entry-per-model.md` — owner: the registry issue

    # Adding a model means adding one registry entry

    Adding a model used to touch three places — `VALID_MODEL_TYPES`, `_FORECASTERS`, and a
    `suggest_*_params` branch — and missing one failed only after training completed. Seven
    separate enumerations of the model set had in fact accumulated across `src/`, and an
    eighth copy in `studies/analysis.py` drifted out of sync and silently collapsed the
    Valendin benchmark's across-study spread to a single study. Counting copies was the
    problem; no single copy was.

    A model is now **one entry in one registry table**, holding its search space, its builder
    and the rollout function it forecasts through. Every model-type list in the package
    derives from that table's keys.

    The table's fields are optional, because Pareto/NBD is a valid model type with no search
    space, no builder and no rollout — its entry exists so the enumerations still derive from
    one place. Whether a model is neural is *read off* the entry ("it has a training builder")
    rather than restated as a second list, because that restatement is exactly the copy that
    drifted.

    ## Consequences

    A model type is registered everywhere or nowhere; there is no state in which it is known
    to the tuner and unknown to the forecaster. The neural / non-neural distinction cannot
    drift, because there is nothing left to keep in sync.

    The registry is its own subpackage. Both `models/` and `studies/` were blocked by real
    import cycles, and a repository-root module was rejected as clutter.

    Entries hold **direct** references. An earlier design used lazy ones so a `model_type`
    could be validated without importing torch; that goal was dropped, so the indirection
    bought nothing.

    Pareto/NBD's entry is declarative. `studies/runner.py` keeps separate neural and
    deterministic paths, which differ in more than the forecaster — no Optuna study, no refit,
    one prediction rather than several.

#### `docs/adr/0007-rollout-model-from-trained-model.md` — owner: the `to_rollout()` issue

    # A rollout model is obtained from a trained model, never rebuilt beside it

    A trained model and the rollout model that forecasts with it are two classes over one
    backbone: the training class returns logits for cross-entropy, the rollout class draws a
    count from the softmax and threads the recurrent state (see "What the models are" in
    `CLAUDE.md`). Their constructor arguments used to be written out separately — three times,
    counting the two class bodies and the tuning builders — and a mismatch surfaced only when
    the state dict failed to load, which is after the training has finished.

    The trained model now hands over its own backbone. `trained.to_rollout()` returns the
    paired rollout model, sharing the same weights object. Nothing outside `models/` and
    `benchmarks/` names a rollout class, and a mismatch is not expressible.

    This deletes the path that rebuilt a rollout model from a stored study's parameters and
    loaded a checkpoint into it.

    For this to be correct the training loop must leave the **best** weights in the model it
    returns, not only on disk. Early stopping keeps training past the best epoch by design, so
    a loop that saves a snapshot and never loads it back returns a model that quietly differs
    from its own checkpoint.

    ## Consequences

    `to_rollout()` shares the backbone rather than copying it. Sharing is what makes a
    mismatch unconstructible, and a copy would double peak memory at the moment a large model
    has just finished training.

    A stored Optuna study plus a checkpoint, with no live model, can no longer be rebuilt and
    loaded — you run the refit's few epochs instead. Consistent with ADR-0008, which makes the
    refit the only path to a forecast anyway.

    The two-class shape survives in `benchmarks/valendin_lstm.py`, which declares its own
    pairing inside the frozen file. Editing a frozen benchmark's *surrounding code* is
    permitted (ADR-0004); `scripts/validate_valendin_lstm.py` is the gate that proves the
    numbers did not move.

    `fit_model` returns a model whose weights match its checkpoint. That property needs its
    own test: the golden end-to-end fixture cannot catch its absence, because its two epochs
    improve monotonically, so both channels agree there by luck rather than by construction.

#### `docs/adr/0008-forecast-from-a-refit.md` — owner: the refit-only issue

    # A forecast is made by a model refit on the full calibration window

    Optuna selects an architecture and a stopping epoch on the temporal validation window
    (ADR-0001). Two things could then produce the holdout forecast: the winning trial's
    checkpoint as it stands, or a warm-start fine-tune of that checkpoint over the whole
    calibration window — the validation tail included — so the weights also *learn* the most
    recent periods rather than only conditioning on them at forecast time.

    Valendin et al. describe the second: after selection they "perform several 'fine-tuning'
    training epochs using the entire calibration data set". This package does only that. Their
    published `rfm2lstm` code does the first; where the code and the paper disagree, we follow
    the paper, and say so rather than inheriting the difference silently.

    ## Consequences

    `prediction_source` and the notebooks' `REFIT_ON_FULL_CALIBRATION` toggle both go: one
    legal value is not a choice. The two knobs expressed the same decision at different
    altitudes, which is how they came to disagree with each other.

    The published-GitHub baseline is no longer expressible from this package. Restoring it
    would be a re-implementation, not a flag.

    The refit trains a fixed few epochs with no validation set and therefore no early
    stopping, so the weights it ends holding are the weights it saves. That is what makes
    ADR-0007's `to_rollout()` exact on the production path rather than merely close.

### The three deferred amendments, in full

**Edit A — `docs/adr/0001-temporal-validation-split.md`, the Consequences paragraph.**
Owner: the issue deleting `rollout_composite`. Its last sentence names two selection
metrics; decision 1 leaves one. No audit caught this.

Replace:

    Model selection scores the same window under both selection metrics, which is what
    makes them comparable.

with:

    Model selection scores that same window, so trials are compared on periods none of
    them trained on.

**Edit B — `docs/adr/0002-simulator-lives-with-the-model.md`, the Consequences section.**
Owner: the issue moving `save_predictions_to_csv` out of `plot_utils`. The rule
"`evaluation/` imports the simulator from `models/`, never the other way round" was
violated for as long as the ADR has existed — `models/` reached into
`evaluation/plot_utils` through a lazy mid-function import to dodge the resulting cycle.
Recording when it became true is what stops the next reader assuming it always was.

Append to Consequences:

    This held as a rule before it held as a fact. `models/` reached back into
    `evaluation/plot_utils` for prediction I/O through a deferred import that hid the
    cycle rather than removing it. Prediction I/O now has its own module and the
    dependency runs one way only.

**Edit C — `src/panelclv/evaluation/__init__.py`, the module docstring.** Owner: the issue
deleting `weekly_aggregate_rollout_metrics`. The claim is currently scoped to "everything
here" — deliberately narrow, because package-wide it was false. It is now true package-wide.

Replace:

    ``models.monte_carlo_forecasting.compute_forecast_metrics`` is the single authority for
    ``rmse`` / ``bias_percent`` / ``mape_aggregate_style``; everything here delegates to it
    rather than defining its own.

with:

    ``models.monte_carlo_forecasting.compute_forecast_metrics`` is the single authority for
    ``rmse`` / ``bias_percent`` / ``mape_aggregate_style`` — the only place in the package
    that computes them. Everything here delegates to it rather than defining its own.

Keep the following sentence about ``aggregate_bias`` unchanged: it is the documented
exception and remains one.

### Findings this ticket produced

- **The `CLAUDE.md` scoring-authority fix this ticket was chartered to make is unnecessary.**
  Decision 1 deletes the only other computation of those three metrics in `src/`. Verified by
  count, not by reading: two RMSE sites, one of them inside the deleted function.
- **ADR-0001 carries a stale line** — "under both selection metrics" — that none of the three
  lane audits or the ledger recorded, because it only became stale when decision 1 was taken.
- **Three of the torch-free deferrals were already inert**, defeated by `panelclv.studies`'
  own package import. Measured.
- **ADR-0005 is the only one of the five untouched** by an open redesign, which is a
  data point about the seam rather than an absence of work.
