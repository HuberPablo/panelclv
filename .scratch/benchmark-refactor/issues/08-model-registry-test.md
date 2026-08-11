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

## Comments

Concrete hazard for this test to catch, found while building the Valendin benchmark
(ticket 06). Four sites dispatch on `model_type` in `tuning/optuna_tuning.py`:

- `:402` `_validate_data_info` — `else: raise`
- `:729` `suggest_*_params` — `else: raise`
- `:749` `_build_lstm if model_type == "lstm" else _build_transformer` — **unguarded**
- `:655` `_validation_rollout_score` — **unguarded**

Today the two unguarded ones are unreachable, because the raising pair runs first. But a
new model type added to `VALID_MODEL_TYPES` and to `:729` only would train a
**Transformer** under the new type's name, silently. That is the same class of failure
CLAUDE.md's "missing the second fails only after training completes" warns about, and a
registry test that only checks membership in the three documented lists would pass
vacuously. Worth asserting the built model's *class* per registered type, not just that
each type appears in each list.
