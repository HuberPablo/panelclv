"""The single table declaring every model the package knows (ADR-0006).

One entry per model, holding its search space, how to sample it, how to build the
training model, and the rollout function it forecasts through. Every model-type
list in the package derives from that table's keys, and whether a type is neural is
read off the entry rather than restated — see ``model_registry`` for why each of
those is the shape it is.

It is its own subpackage because both plausible homes are blocked by import cycles:
the table names benchmark classes while ``benchmarks`` already imports ``models``,
and ``tuning`` is what needs the registry while ``studies`` already imports
``tuning``.
"""

from .model_registry import (
    MODEL_REGISTRY,
    MODEL_TYPES,
    ModelEntry,
    build_model,
    entry,
    is_neural,
    rollout_for,
    suggest_param,
    suggest_params,
    validate_model_knobs,
)

__all__ = [
    # The table and its keys
    "MODEL_REGISTRY",
    "MODEL_TYPES",
    "ModelEntry",
    # Reading it
    "entry",
    "is_neural",
    "validate_model_knobs",
    "suggest_params",
    "build_model",
    "rollout_for",
    # The search-space mini-language, shared with the tuner (which resolves the
    # training controls that may also carry a spec, e.g. patience over {5,7,9}).
    "suggest_param",
]
