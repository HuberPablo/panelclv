# The Monte Carlo simulator lives in `models/`, not `evaluation/`

These models emit a distribution over transaction counts per period, not a number.
Turning that into a forecast requires sampling a count, feeding it back as the next
period's input, and averaging over many paths — so the simulator is the model's
forecast mechanism, not a scoring step applied afterwards. Putting it in `evaluation/`
would suggest you could swap it out and still have the same model.

## Consequences

`evaluation/` imports the simulator from `models/`, never the other way round.
Anything that changes how a forecast is produced is a model change.

This held as a rule before it held as a fact. `models/` reached back into
`evaluation/plot_utils` for prediction I/O through a deferred import that hid the
cycle rather than removing it. Prediction I/O now has its own module and the
dependency runs one way only.
