# panelclv

Customer-base forecasting: predicting how many transactions each customer will make
in each future period, from their own transaction history.

## Language

### The data

**Panel**:
A rectangular table with one row per customer per period, covering every period in
the window whether or not the customer transacted.
_Avoid_: dataset, matrix, table

**Period**:
The time unit a panel is measured in — weekly throughout this project, though the
package supports daily and monthly.
_Avoid_: week, timestep, bucket

**Cohort**:
The set of customers a model is fit on. Customers first seen only in the holdout are
excluded, so every model sees the same cohort.
_Avoid_: sample, population

**Transaction count**:
The number of transactions a customer made in one period. This is the target, and it
is treated as a category, never as a quantity.
_Avoid_: frequency, volume, y

**Covariate**:
Any model input that is not the transaction count being predicted. A covariate is
either **declared** — a panel column named in one of a `PanelConfig`'s roles — or
**derived** — computed from the target's own past. The distinction is about where the
value comes from, not how the model reads it: both occupy a channel of the same tensor,
both are standardised the same way, and the covariate-subset search drops either.
_Avoid_: predictor, regressor, explanatory variable

**AR feature**:
A derived covariate: a causal function of the target's own past — recency, cumulative
counts, tenure, rate. The one kind of covariate that cannot be read from the panel
during a rollout, because the panel's future does not exist yet; it is recomputed at
each step from the counts the model itself sampled. That recomputation is what makes a
rollout leak-free.
_Avoid_: lagged feature, history feature, target encoding

**Behavioural cluster**:
A derived covariate that groups customers by how they behaved across the calibration
window, and hands the model the group index as a category. Like an AR feature it comes
from the target's own past; unlike one it is **frozen** — computed once and constant
for that customer through every holdout period, so a rollout carries it rather than
recomputing it. A customer never changes cluster mid-forecast.
Not to be confused with the **customer groups** of the segment analysis, which are
derived from calibration *and holdout* activity and exist only to break a results
table apart. Those are a way of reading a forecast; a behavioural cluster is an input
to one.
_Avoid_: segment, group, cohort, customer type

### The time windows

**Calibration window**:
The span of periods a model may learn from. Splits into the training window and the
validation window.
_Avoid_: training period, in-sample window, history

**Training window**:
The leading part of the calibration window whose periods update model weights.

**Validation window**:
The trailing part of the calibration window, used to choose between models and to
stop training. Never used to update weights. It is a span of time applying to every
customer, not a subset of customers.
_Avoid_: validation set, validation split, dev set

**Holdout window**:
The span of periods after the calibration window that a model is scored on and never
sees during fitting or selection.
_Avoid_: test set, out-of-sample period, forecast period

### The models

**Reference implementation**:
A model reproduced faithfully from published work so it can serve as an honest
comparator. It is not developed or improved — deviations from its source are
deliberate, enumerated, and recorded.
_Avoid_: baseline model, legacy model

**Benchmark**:
A reference implementation used as a comparator in a study. The two benchmarks are
the Valendin et al. LSTM and Pareto/NBD.
_Avoid_: baseline, comparator, competitor

**Contribution**:
A model under active development, free to depart from published architectures. The
Transformer is the current one.
_Avoid_: our model, the new model, experimental model

**Embedder**:
The component that turns a customer's per-period features into the vector a model
consumes. Which embedder a model uses is part of its identity.
_Avoid_: encoder, feature extractor, input layer

**Registry**:
The single table that declares every model the package knows: for each one, its search
space, how to build it and the rollout function it forecasts through. Adding a model means
adding an entry (ADR-0006). Every list of model types derives from its keys.
_Avoid_: factory, dispatch table, model map

### Forecasting and scoring

**Rollout**:
Stepping a trained model forward through future periods by feeding its own sampled
output back as the next period's input, without ever reading the true future.
_Avoid_: inference, prediction loop, autoregression

**Rollout model**:
The object that performs a rollout: it draws a count from the model's own softmax and
carries its recurrent state from one period to the next. It is obtained from a trained
model, which hands over its weights (ADR-0007) — never built alongside one.
_Avoid_: inference model, prediction model, sampler

**Simulated path**:
One rollout. A forecast is the average over many of them, because a single path is a
draw, not an expectation.
_Avoid_: run, sample, trajectory

**Warm-up**:
Replaying observed history through a model to build its recurrent state before a
rollout begins.
_Avoid_: priming, seeding, burn-in

**Aggregate bias**:
Total predicted transactions minus total actual, over all customers and periods.
Reported as a percentage of the actual total.
_Avoid_: error, drift, offset

**Tracking**:
How closely a model's per-period aggregate follows the actual per-period aggregate
across the holdout — the shape of the forecast rather than its total.
_Avoid_: fit, accuracy

### Experiments

**Trial**:
One trained model with one sampled set of hyperparameters and features.

**Study**:
One search over trials, yielding a single winning trial.
_Avoid_: run, experiment, sweep

There is deliberately **no term for "experiment"**. Every unit of work here is a trial, a
study, a study suite or a grid; a further word for the same thing would only blur which of
the four is meant. The four are levels of containment, not synonyms — that is why each
earns a name and "experiment" does not.

**Refit**:
Warm-start fine-tuning of a study's winning trial over the full calibration window — the
validation window included — for a few large-batch epochs, so the weights also learn the
most recent periods instead of only conditioning on them at forecast time. Every forecast
comes from a refit (ADR-0008).
_Avoid_: retrain, fine-tune, final training

**Study suite**:
Several studies per model over one shared dataset, so each model's result is reported
as a distribution across replications rather than a single number.
_Avoid_: batch, campaign

**Grid**:
A set of study suites, one per dataset, run to see how a model's error moves as some
property of the data is varied. The Pareto/NBD grid varies the generating parameters of
synthetic panels. A grid holds suites, which hold studies, which hold trials — it is a
level above the suite, not another word for one.
_Avoid_: sweep, matrix, experiment set
