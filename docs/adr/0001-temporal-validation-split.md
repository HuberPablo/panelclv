# Validation is a time window, not a subset of customers

Valendin et al. hold out a random 10% of customers for validation, which means the
validation periods are the same periods the model trained on — it measures fit to
observed history, not the ability to forecast forward. We instead cut the calibration
window at a date: every customer's periods before it train the weights, every
customer's periods after it validate. This is a deliberate departure from the paper,
applied uniformly to every model so comparisons stay fair.

## Consequences

Class weights are computed on the training window only, so the validation window never
leaks into the loss. Model selection scores that same window, so trials are compared
on periods none of them trained on.
