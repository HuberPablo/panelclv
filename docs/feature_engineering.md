# Feature Engineering

This chapter documents how `panelclv` turns a raw customer-period panel into the
`(N, T, F)` float32 tensors the models consume, and — just as importantly — how it keeps
every engineered feature *reconstructible at forecast time without reading the future*.

The guiding constraint is the Valendin et al. design: the model is a **classifier over
transaction-count classes** whose forecast is produced by an **autoregressive Monte Carlo
rollout**. That has a hard consequence for feature engineering:

> Every feature the model sees at step *t* of the holdout must be either (a) genuinely
> known in advance, or (b) computable from the model's own *sampled* history.
> A feature that is neither cannot exist in this package — there is no way to supply it
> during the rollout without leaking the answer.

Everything below is a consequence of that rule.

**Where the code lives**

| Concern | Module |
| --- | --- |
| Declarative feature spec + validation | `src/panelclv/configs/panel_config.py` |
| Panel → tensor pipeline, calendar features, embeddings | `src/panelclv/data_preparation/panel_dataset.py` |
| Autoregressive target-derived features | `src/panelclv/data_preparation/ar_features.py` |
| Feature consumption (embeddings / covariate projection) | `src/panelclv/models/multinomial_lstm.py`, `multinomial_transformer.py` |
| Feature reconstruction during the rollout | `src/panelclv/models/monte_carlo_forecasting.py` |
| Feature *selection* as a tuned decision | `src/panelclv/tuning/optuna_tuning.py` |

---

## 1. The contract: what a feature *is* in this package

A feature is a **numeric column of the panel** that has been assigned a **role**. Roles
are declared once, in a `PanelConfig`, and `prepare_dataset` does the rest:

```
panel (one row per customer × period)
   │
   ├─ 1. engineer calendar columns          (time_features flags)
   ├─ 2. add `period_start` anchor          (uniform date handle for slicing)
   ├─ 3. create AR feature columns          (zero placeholders for now)
   ├─ 4. flatten roles → seq_cols           (fixed channel order)
   ├─ 5. slice calibration / holdout windows
   ├─ 6. cohort filter + target clipping
   ├─ 7. FILL AR features from calibration only
   ├─ 8. resolve embedding cardinalities
   ├─ 9. reshape → (N, T, F)
   ├─ 10. standardise the numeric channels  (calibration-fitted; §5)
   └─ 11. build (samples, targets)
```

The output is a plain dict; the feature-relevant keys are:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `calibration` | `(N, T_CAL, F)` | calibration-window tensor |
| `holdout` | `(N, T_HOLD, F)` | holdout-window tensor (covariates real, target/AR columns never fed) |
| `samples` | `(N, T_CAL-1, F)` | next-step inputs, `calibration[:, :-1, :]` |
| `targets` | `(N, T_CAL-1, 1)` | next-step class labels, `calibration[:, 1:, target_idx]` |
| `seq_cols` | `list[str]` | channel names, in tensor order — the single source of truth for `F` |
| `target_idx` | `int` | position of the target channel |
| `embedded_cols` | `{col: cardinality}` | resolved categorical embedding map |
| `ar_features` | `list[str]` | which channels must be recomputed during the rollout |
| `covariate_stats` | `{col: (mean, std)}` | calibration-fitted scaling of the numeric channels — the rollout must re-apply it |

`seq_cols` is the contract. The model addresses channels **by name**, never by a
hardcoded position, which is what makes the package dataset-agnostic: a new panel with
different covariates needs a different `PanelConfig`, not a code edit.

### Channel order

Channels are laid out in a fixed group order (`_SCHEMA_GROUP_ORDER`):

```
target → time → known_future → observed_past → static → ar_features
```

De-duplicated, first-occurrence-wins. Fixing the order means two configurations that
share features put them in comparable positions, which keeps checkpoints, diagnostics
and the feature-subset slicing legible.

---

## 2. Feature roles (the TFT-style grouping)

Roles are not decoration — each one carries a *different guarantee about the future*, and
the pipeline treats them differently.

| Role | `PanelConfig` field | Guarantee | Available in holdout? |
| --- | --- | --- | --- |
| target | `target_col` | the thing being predicted | **never fed** — replaced by the sample |
| time | `time` (+ auto-filled by `time_features`) | deterministic function of the calendar | yes, computable arbitrarily far ahead |
| known future | `known_future` | value is known per period in advance (promo calendar, season flag, `year_idx`) | yes, read from the holdout tensor |
| observed past | `observed_past` | observed only up to the forecast origin | **not supported — dropped with a warning** |
| static | `static` | one value per customer, broadcast over that customer's rows | yes, constant |
| AR features | `ar_features` | causal function of the target's own past | recomputed from the *sampled* target |

Two of these deserve their justification spelled out.

**`observed_past` is deliberately unimplemented.** An unknown-future covariate has no
value at holdout step *t* unless you read the true one — which is leakage — or model it
jointly, which is out of scope. Rather than silently mis-handling it, `prepare_dataset`
drops the group and emits a warning naming the columns. The two planned honest routes
are (i) encoder-only conditioning during warm-up, or (ii) lagging the covariate into
`known_future` so its value at *t* is its observed value at *t − k*.

**The target is a role of its own** and is *not* listed in any covariate group. It is
declared once as `target_col`, its cardinality drives the softmax head, and at rollout
time its channel is overwritten with the sampled class each step.

---

## 3. Calendar features

Calendar features are the cheapest genuinely-known-future signal available, and the only
family the package *engineers from scratch*. They are **opt-in**: omitting
`time_features` engineers nothing.

This table is not the source: `configs/panel_config.py`'s `TIME_FEATURE_FLAGS` is, and
`add_time_features` builds against it, so the columns and frequencies below are read off
one declaration rather than restated in three.

| Flag | Columns created | Formula | Valid frequencies |
| --- | --- | --- | --- |
| `add_year_idx` | `year_idx` | `year − year(training_start)` | weekly, monthly, daily |
| `add_week_sin_cos` | `week_sin`, `week_cos` | `sin/cos(2π·w / periods_per_year)` weekly, `sin/cos(2π·w / 52)` daily | weekly, daily |
| `add_month_sin_cos` | `month_sin`, `month_cos` | `sin/cos(2π·(m−1) / 12)` | monthly, daily |
| `add_dayofyear_sin_cos` | `day_sin`, `day_cos` | `sin/cos(2π·(doy−1) / 365)` | daily |

**Where `w` comes from, and why the daily divisor is 52 rather than
`periods_per_year`.** A weekly panel carries its own `week` column and the divisor is the
declared `periods_per_year` (52 by convention). A daily panel has no week column, so the
week-of-year is read off the date by the package's single week convention
(`data_preparation/period_calendar.py`): a year is **52 weeks of seven days, numbered
0..51**, week `w` covering days-of-year `7w+1 .. 7w+7`, with the trailing day or two of
the calendar year folded back into week 51. The divisor there is 52 because that is how
many weeks the cycle has — on a daily panel `periods_per_year` counts *days* (365), and
using it would compress the year into a seventh of the sine's period. This is deliberately
not ISO 8601: ISO gives some years a 53rd week, which aliases exactly onto week 0 under a
52-week sine and would encode New Year's Eve as New Year's Day.

**A stored panel may not agree with that convention, and nothing checks it.** The
convention above is what `period_calendar.week_of_year` implements and what
`week_start` inverts, but a panel arrives with its `week` column already computed by
whatever built it — the package never re-derives it from a date. The two builders in
this repo disagree by one day:

| builder | rule | week 0 | week 51 |
|---|---|---|---|
| `notebooks/archive/dataset_building.ipynb` (electronics, apparel, gift, multichannel) | `dayofyear // 7` | Jan 1–6 (6 days) | 9 days |
| `period_calendar.week_of_year` / `scripts/build_cdnow_panel.py` (CDNOW) | `(dayofyear − 1) // 7` | Jan 1–7 | 8 days |

Measured, not inferred — rebuilding the stored electronics panel from
`Datasets/Electronic.csv` under each rule:

```
dayofyear // 7        (archived notebook):      0 mismatched cells of 172432
(dayofyear - 1) // 7  (package week_of_year):  913 mismatched cells of 172432
```

So for the electronics family, the panel's week `w` sits one day earlier than
`week_start(year, w)` places it on the calendar. Two consequences, one live and one
latent:

- **Cross-dataset:** "week 39" is not the same seven days on an electronics panel as on
  CDNOW. Comparing a seasonal position between the two families is off by a day.
- **Window slicing:** `prepare_dataset` cuts the calibration/holdout windows on
  `week_start`, so a boundary falling mid-week could place up to one day of transactions
  in the wrong window. **Neither current config is affected** — every electronics window
  date is a year boundary, where both rules agree the year's last bucket ends on Dec 31,
  and CDNOW's mid-year boundaries were built with the package rule, so `week_start` is
  its exact inverse. The risk arrives with the first mid-year boundary on a panel from
  the archived builder.

A panel built outside this package is trusted as given. If you build one, use
`period_calendar.week_of_year` so `week_start` inverts it.

**Why sin/cos rather than the raw index.** A raw week number is a discontinuous
encoding of a circular quantity: weeks 52 and 1 are adjacent in the world but maximally
distant in the feature. Projecting onto the unit circle makes the encoding continuous and
periodic — the model can learn "late December ≈ early January" without spending capacity
undoing the wrap-around. It also extrapolates perfectly into the holdout, since it is a
pure function of the calendar.

**Auto-assignment to the `time` role.** A flag that produces cyclical columns registers
them into the `time` role automatically (`PanelConfig.schema`), *unless* the column is
already assigned to some role. So you never list `week_sin`/`week_cos` yourself. The
`time` field is reserved for cyclical columns **already present** in the panel. `year_idx`
is deliberately *not* auto-assigned: it is a trend feature, and where it belongs
(usually `known_future`) is a modeling choice.

**The `year_idx` caveat.** `year_idx` is monotone and, by construction, takes values in
the holdout that were never seen in calibration. A model that leans on it is extrapolating
a trend off the end of its training support — in practice the classic failure mode is a
rollout that over-predicts with no decay. It is included because reproducing the reference
workflow requires it, but it is a prime candidate for `removable_features` (§8), and the
rollout-based selection metric exists partly to catch exactly this.

**`period_start`.** Independently of the flags, the pipeline adds a single `period_start`
Timestamp so both windows are sliced by one uniform rule (weekly: `Jan-1 of year + 7·week`
days; monthly: first of month; daily: the date itself). It is a slicing anchor, not a
model feature. The weekly rule is the same convention `week_sin`/`week_cos` read, and both
come from `period_calendar` — the anchor and the feature cannot drift apart.

---

## 4. Autoregressive target-derived features

This is the family that makes the recency/frequency structure of the BTYD literature
available to a neural model without breaking the rollout. A "transaction" is defined as
`target > 0`.

All of them are read off a small per-customer running state, maintained in period order:

| State | Definition |
| --- | --- |
| `since` | periods since the last transaction (0 in a transacting period) |
| `ever` | whether any transaction has occurred yet |
| `cum_txn` | number of *active periods* so far |
| `cum_cnt` | sum of the target counts so far |
| `tenure` | periods since the first transaction (0 before and at the first) |

and the exposed features are pure functions of it:

| Feature name | Value | BTYD analogue |
| --- | --- | --- |
| `period_since_last_transaction` | `since` | recency (the gap since the last purchase) |
| `has_transacted_before` | `1[ever]` | — |
| `active_in_last_<K>_periods` | `1[ever and since < K]` | windowed activity flag, `K ≥ 1` |
| `cumulative_transactions` | `cum_txn` | frequency **x** |
| `cumulative_count` | `cum_cnt` | total count (≥ `cum_txn`; differs when a period holds several transactions) |
| `period_since_first_transaction` | `tenure` | observation age **T** |
| `transaction_rate` | `cum_txn / max(tenure, 1)` | empirical Poisson rate **λ** |

Notes on the conventions, which matter for reproducibility:

- Before a customer's first transaction, `since` counts up from the start of the series
  (index `t` ⇒ `since = t + 1`), and `tenure` is pinned to 0. `has_transacted_before` and
  every `active_*` flag are 0. So "never purchased yet" is representable and distinct
  from "purchased long ago".
- `tenure` is 0 *at* the first transaction and increments from the next period — matching
  the "age since first purchase" convention, not "number of observed periods".
- `transaction_rate` guards its denominator with `max(tenure, 1)`, so the first period
  never divides by zero and later periods are undistorted.
- All counters are integer-valued (the counts are multinomial class indices), so the two
  compute paths (§6) agree **exactly**, not approximately.

**Which ones to prefer.** `cumulative_transactions`, `cumulative_count` and
`period_since_first_transaction` grow without bound and, over a long holdout, drift past
the range the model ever saw in calibration — the same extrapolation hazard as `year_idx`.
`transaction_rate`, `has_transacted_before` and `active_in_last_<K>_periods` are bounded
and stationary, and carry much of the same information.

Standardisation (§5) does **not** rescue an unbounded counter. The mean and standard
deviation are fitted on the calibration window, so a counter still climbing through the
holdout still climbs after the transform — from a recentred origin, at a rescaled rate,
but out of the range the weights were trained on all the same. What standardisation does
neutralise is *scale*: an unbounded channel no longer dominates the shared covariate
projection merely for being measured in larger units. So the case for preferring a
bounded feature is extrapolation alone, which is a judgement about the length of your
holdout rather than about the architecture.

**Adding a new AR feature.** Extend the running state in `_base_states` (the vectorised
`(N, T)` precompute), mirror the increment in `ARFeatureState.update` (the per-step
rollout), add a branch in `_render` and a name in `parse_ar_feature`. The three must stay
consistent — `tests/test_ar_features.py` asserts the precompute and the incremental path
produce identical columns, which is the test to extend alongside.

---

## 5. Encoding: categorical embeddings and numeric standardisation

`embedded_cols` declares **which columns are categorical**; everything else in `seq_cols`
is treated as continuous. The spec is either a mapping `{col: int | "auto"}` or a plain
list of names (all `"auto"`).

Inside the model, the two paths are:

- **Embedded columns** → `nn.Embedding(cardinality, √cardinality + 1)` → `LayerNorm` →
  `Linear(→ embedding_dim)` → `LayerNorm`. The target's embedding is kept separate; all
  other embeddings are **summed** into a context representation.
- **Continuous columns** → concatenated → `Linear(n → embedding_dim)` → `LayerNorm`, then
  added into the same context representation.

The LSTM/Transformer input is `[context, target_embedding]` when any context exists, and
the target embedding alone otherwise (so the minimum legal model is target-only,
`F = 1`). The Transformer mirrors this encoder exactly; the two families differ in how
history is carried, not in what a feature means.

### Numeric channels are standardised, fitted on calibration

Every channel that is **not** embedded is put on a common scale — mean 0, std 1 — by
`standardize_covariates`, which runs after the reshape to `(N, T, F)` and before
`samples` / `targets` are sliced. Two things are excluded, for different reasons:

- everything in `embedded_cols`, because those are integer class indices cast with
  `.long()` and used as embedding-table lookups; rescaling them would corrupt the lookup
  outright;
- `target_col`, excluded explicitly. A valid config always embeds it, so the first rule
  would already cover it, but the explicit exclusion also protects the autoregressive
  contract: `targets` are sliced straight out of `calibration` and have to stay integer
  class indices for the cross-entropy head.

**Why it is needed.** The models push every non-embedded column through *one* shared
`Linear(n_covariates → embedding_dim)`, which computes `sum_k W_k · x_k` with all `W_k`
drawn from the same initial distribution. Each column's contribution to that sum — and
the gradient reaching its weights — therefore scales with its **raw magnitude**. Mixing
`week_sin` (std ≈ 0.7) with `period_since_last_transaction` (std ≈ 27) hands the recency
channel almost all of the pre-activation variance purely because it is measured in weeks
rather than in a sine wave. The `LayerNorm` that follows cannot repair this: it
normalises the *sum*, after the columns have already been mixed, so it fixes the output
scale while leaving the drowned-out columns drowned out. Before the sum is the only place
the imbalance can be corrected.

**Fitted on calibration, applied to both windows.** The `{col: (mean, std)}` map is
computed on the calibration window and returned as `covariate_stats`; the holdout is
transformed with those same statistics, never with its own. Fitting on the holdout would
leak its distribution into the forecast — the quiet kind of leakage this chapter exists
for.

**The rollout has to re-apply it.** The holdout tensor's declared covariates were
standardised once, by `prepare_dataset`, along with the calibration window. The *derived*
ones are the problem: `ARFeatureState` regenerates them in raw units at each step, so the
simulator puts every recomputed AR value back through its `(mean, std)` before writing it
into the step input. Skipping that would feed the model raw recency after warming it up on
standardised recency — a silent unit mismatch no shape check can catch. A column missing
from the map (an AR feature the caller chose to embed, or a dict from an older run) passes
through with the identity `(0.0, 1.0)` rather than raising — see §6.

### Cardinality resolution is role-aware

`"auto"` cardinalities are inferred from the data, but **which window is read depends on
the role** — this is where leakage would otherwise creep in:

| Column | Inference window | Rationale |
| --- | --- | --- |
| target | `clip_target_upper + 1`, else calibration max + 1 | the head size is a modeling decision, not a data peek |
| `time` / `known_future` | `max(calibration, holdout) + 1` | legitimate: those future values are *given*, not predicted |
| static / everything else | calibration max + 1 | never peek at the holdout |

Pinned integers are kept but validated to cover the values actually present in the
relevant window, and a column whose inferred cardinality is 1 (constant in-window) raises
rather than producing a degenerate embedding.

### The known-future drift warning

Sizing a known-future embedding over both windows is safe, but there is a subtler failure:
embedding **rows** for categories that appear *only* in the holdout are never touched by
training, so at forecast time the model reads their random initialisation.
`warn_known_future_drift` reports this up front, before any training, listing the offending
column and values. It is scoped to *embedded* known-future columns on purpose — continuous
known-future channels (`week_sin`, `year_idx`, …) are *expected* to take new values every
period and have no table to leave untrained.

---

## 6. Leakage discipline: one primitive, two call sites

The single most important implementation detail in this chapter.

AR features are **not** computed over each customer's full series. That series spans the
holdout, so recency/frequency/tenure would absorb activity from the forecast window — a
model told "this customer purchases often" using purchases it is supposed to predict.

Instead:

1. In the panel, AR columns are created as **zero placeholders**, so the column-existence
   and window-slicing checks pass.
2. After the cohort filter and after target clipping, they are filled **on the calibration
   window only**, per customer, in period order, from the *clipped* calibration target —
   via `compute_ar_feature_columns`.
3. The holdout's AR columns are **left at zero and never read**. During the rollout, an
   `ARFeatureState` is seeded from the calibration target history and advanced one step at
   a time with the **sampled** count, overwriting those channels in the input row before
   each model call.

Both paths — the vectorised training-time precompute and the incremental rollout state —
are the same recurrence over the same five state variables, expressed twice for
performance reasons and kept identical by construction (`_render` is shared) and by test.
This is what makes the training distribution and the inference distribution of these
features match.

The rollout more generally, per step *t*:

| Channel group | Source at rollout step *t* |
| --- | --- |
| target | previous **sampled** class |
| AR features | `ARFeatureState.update(sample)` |
| time / known future / static | true holdout values (legitimately known) |
| observed past | not present (dropped upstream) |

The true holdout target is never fed to either model family.

---

## 7. Target handling and cohort selection

Two panel-level operations sit alongside feature construction because they change what the
features *describe*.

**Upper clipping (`clip_target_upper`).** Counts are clipped on the **training window
only**; the holdout is left untouched so evaluation runs against real actuals. The clip
sets the softmax head size (`clip_target_upper + 1` classes) and thereby the target
embedding's cardinality — a cross-check that fires at config time if a pinned target
cardinality is too small for the clip. Because AR features are filled *after* clipping,
`cumulative_count` reflects the clipped counts, consistent with what the model can sample.
The clip is invariant for the "transaction occurred" test (clipping never turns a positive
into a zero), so recency/frequency/tenure are unaffected by it.

**Cohort filter (`require_calibration_activity`, on by default).** Keeps only customers
with at least one transaction during calibration — equivalently, first purchase ≤
`training_end`. This reproduces the Valendin et al. cohort rule. Customers first seen in
the holdout are unknown at forecast time and would otherwise present the model with an
all-zero history. Crucially the filter is applied inside `prepare_dataset`, so the
Pareto/NBD benchmark (which reads the returned `train_panel`) fits the **same cohort** as
the neural models — the comparison stays fair.

---

## 8. Feature selection as a tuned decision

Which covariates to keep is a hyperparameter, not a prior belief. `run_optuna_study`
accepts `removable_features`, a list where each entry is either a single column (its own
on/off toggle) or a **group toggled as a unit** — e.g. `("week_sin", "week_cos")`, since
half a cyclical pair is meaningless.

Per trial, the sampled drop-set is applied by `select_features`, which is pure column
slicing on the already-built tensors: it re-indexes the feature axis of
`calibration`/`holdout`, rebuilds `samples`/`targets`/`target_idx`, filters
`embedded_cols`, and — importantly — filters `ar_features` in lockstep, so a dropped AR
column cannot be looked up by the rollout and raise. No data re-prep happens per trial.
The target is never removable.

Each trial records `selected_features` / `dropped_features` as user attributes, so the
winning feature set is recoverable after the fact with `select_features_for_trial` — which
matters because a checkpoint trained on a sliced layout will not load into a full-feature
model.

Selection interacts with feature engineering directly, and not in your favour: trials are
scored on `val_loss` — teacher-forced next-step cross-entropy — which is blind to the
rollout, so it can happily keep an extrapolating trend feature and drop the
seasonal/recency signals. Nothing in tuning penalises drift over a long horizon
(ADR-0003, retired). If the feature set includes unbounded or out-of-range channels,
that is a judgement you have to make yourself.

---

## 9. Worked configurations

**Minimal — target plus a raw weekly index.** No engineered calendar features, no
covariates, no embeddings beyond the target:

```python
cfg = PanelConfig(
    id_col="Id", target_col="Transactions", frequency="weekly",
    time_cols=("year", "week"),
    training_start="1999-01-01", training_end="2000-12-31",
    validation_start="2000-07-01",
    holdout_start="2001-01-01", holdout_end="2001-12-31",
    time=("week",),                       # already in the panel; no flag engineers it
    clip_target_upper=6,
    embedded_cols={"Transactions": "auto"},
)
# seq_cols → ["Transactions", "week"]
```

**Full — calendar + covariates + leak-free RFM signals:**

```python
cfg = PanelConfig(
    id_col="Id", target_col="Transactions", frequency="weekly",
    time_cols=("year", "week"), periods_per_year=52,
    training_start="1999-01-01", training_end="2000-12-31",
    validation_start="2000-07-01",
    holdout_start="2001-01-01", holdout_end="2001-12-31",
    known_future=("year_idx", "high.season"),
    static=("Gender", "Income"),
    time_features={"add_year_idx": True, "add_week_sin_cos": True},
    ar_features=("period_since_last_transaction",
                 "active_in_last_4_periods",
                 "transaction_rate"),
    clip_target_upper=6,
    embedded_cols={"Transactions": "auto", "Gender": "auto", "high.season": "auto"},
)
# seq_cols → ["Transactions",                       # target
#             "week_sin", "week_cos",               # time (auto-assigned by the flag)
#             "year_idx", "high.season",            # known future
#             "Gender", "Income",                   # static
#             "period_since_last_transaction",      # ar features
#             "active_in_last_4_periods",
#             "transaction_rate"]
```

Paired with a tuning run that is allowed to question the risky channels:

```python
study = run_optuna_study(
    model_type="lstm",
    data_builder=make_data_builder(data_full),
    search_space={...},
    removable_features=[("week_sin", "week_cos"), "year_idx",
                        "Gender", "Income", "transaction_rate"],
    n_trials=40,
)
```

---

## 10. Validation and failure modes

Feature construction fails **early and loudly**, before any tensor is built or any epoch
is run. The checks, in the order they fire:

- unknown `ar_features` name, or an `active_in_last_K` with `K < 1` → at `PanelConfig`
  construction;
- unknown `time_features` flag (typo) → error; a flag the frequency cannot produce →
  dropped with a warning;
- date-window ordering (`training_start < validation_start ≤ training_end <
  holdout_start`) → at construction; re-checked against the real calendar so a
  `validation_start` leaving zero training or zero validation periods raises;
- AR feature name colliding with an existing panel column → error;
- missing `id_col` / `target_col` / schema columns; non-numeric selected columns (encode
  them first — the tensors are float32);
- empty training or holdout window, quoting the panel's actual date coverage;
- empty cohort after the activity filter;
- ragged per-customer period counts, or train/holdout customer sets that differ or are
  ordered differently;
- NaN in any selected column, named per window;
- pinned embedding cardinality too small for the observed values, or an `"auto"` column
  that is constant in-window;
- `clip_target_upper` ≥ the pinned target cardinality.

Warnings (not errors): `observed_past` columns being dropped, and known-future embedding
drift.

---

## 11. Limitations and open extensions

- **`observed_past` covariates are unsupported.** See §2 for the two honest routes.
- **No per-feature scaling.** Continuous channels are projected raw; prefer bounded AR
  features, or pre-scale in the panel, when magnitudes differ by orders of magnitude.
- **Unbounded counters extrapolate.** `cumulative_*` and tenure leave their calibration
  range during a long holdout; `transaction_rate` exists as the bounded alternative.
- **Uniform panels only.** Every customer must have an identical number of periods in
  each window; ragged panels must be padded upstream (`notebooks/archive/dataset_building.ipynb`).
- **Static covariates must already be broadcast** to every row of a customer and must be
  numerically encoded — the pipeline does not label-encode strings for you.
