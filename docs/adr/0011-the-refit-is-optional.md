# The refit is optional: a forecast may come from the winning checkpoint

ADR-0008 made the refit the only route to a forecast, on the grounds that Valendin et al.'s
paper describes it ("several 'fine-tuning' training epochs using the entire calibration
data set") while their published code does not. Since 8 October 2026 the reading is
narrower: the paper reports the refit as something tried, and its published results come
from the selected model without it. Following the paper's *results* therefore means no
refit, and the refit becomes a variant to compare against, not the reference.

It also matters for model selection. The refit sits between the search and the holdout
score: two refits of one checkpoint differ by 5.9–15.0 points of |bias| on average
(`docs/model-selection.md` §2), and in a four-study pilot it erased the order the
checkpoints' validation scores gave on electronics (`docs/hyperparameter-search.md` §6.4).
A question about whether the search picks well cannot be answered through a step that
reshuffles what it picked.

`StudySuiteConfig.refit` chooses the route. `True` keeps ADR-0008's refit
(`trials.refit_best_trial`); `False` loads the winning trial's checkpoint into a model
built at the full calibration length and rolls it out as it stands
(`trials.load_best_trial`). `refit_kwargs` with `refit=False` is refused, since they would
configure a step that does not run.

## Consequences

The default stays `True`. Every archived result was produced through a refit, so a default
of `False` would make every runner written before today read differently from its own
archive.

The suite record, each model record and `results.csv` carry `refit`, so a forecast states
which route produced it.

The checkpoint holds the weights of the epoch early stopping kept, trained on the
calibration window minus the validation tail. The rollout still warms up over the whole
calibration window, validation tail included, so the model *conditions* on the most recent
periods without having *learned* from them; that is the difference the refit was meant to
remove, and the one a comparison of the two routes measures.
