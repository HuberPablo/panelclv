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

### Forecasting and scoring

**Rollout**:
Stepping a trained model forward through future periods by feeding its own sampled
output back as the next period's input, without ever reading the true future.
_Avoid_: inference, prediction loop, autoregression

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

**Study suite**:
Several studies per model over one shared dataset, so each model's result is reported
as a distribution across replications rather than a single number.
_Avoid_: batch, campaign
