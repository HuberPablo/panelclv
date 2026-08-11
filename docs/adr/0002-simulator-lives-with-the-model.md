# The Monte Carlo simulator lives in `models/`, not `evaluation/`

These models emit a distribution over transaction counts per period, not a number.
Turning that into a forecast requires sampling a count, feeding it back as the next
period's input, and averaging over many paths — so the simulator is the model's
forecast mechanism, not a scoring step applied afterwards. Putting it in `evaluation/`
would suggest you could swap it out and still have the same model.

## Consequences

`evaluation/` imports the simulator from `models/`, never the other way round.
Anything that changes how a forecast is produced is a model change.
