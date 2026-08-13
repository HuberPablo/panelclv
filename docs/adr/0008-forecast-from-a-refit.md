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
