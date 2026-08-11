# How features become a vector is a swappable component

The Valendin LSTM and our own models need genuinely different embedding strategies —
theirs concatenates raw square-root-sized embeddings, ours projects each feature to a
common width and sums the context. Welding either into a model would mean the frozen
reference and the models under development share code that one of them must never
change. Instead an embedder is an object a model is given: it exposes a forward pass
and the output width, and the model consumes that width without knowing the strategy.

## Consequences

A new embedding strategy is a new class, not a new branch inside a model. The seam is
testable on shapes alone, without training anything. LSTM and Transformer share it.
