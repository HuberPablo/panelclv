# Test that the three model-registration lists agree

Status: ready-for-agent

Adding a model touches `VALID_MODEL_TYPES` (`studies/config.py`), `_FORECASTERS`
(`studies/runner.py`) and a `suggest_*_params` branch (`tuning/optuna_tuning.py`).
Missing the second one fails only after training completes, which is an expensive way
to learn about a typo.

Add a test walking `VALID_MODEL_TYPES` and asserting every neural entry has both a
forecaster and a parameter suggester. Roughly fifteen lines, no refactor.

A real self-registering registry is the better end state, but that is a design job for
its own session.

Done when: removing any one of the three entries makes the test fail immediately.
