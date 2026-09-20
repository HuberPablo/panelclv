# 04 — Register family T in `docs/studies-run.md`

Status: `ready-for-agent`
Blocked by: 03

Once the suites exist, add them to the global inventory in §4 — one row per (model, arm),
family letter **T** (S is the last one used) — and a short §4.x note saying:

- what the family tests (the training recipe, not the architecture or the inputs),
- that its ValendinLSTM suites sit **beside** family N rather than replacing it, so no
  published row moves,
- that the `paper` arm runs a single pinned trial, so its `results.csv` carries no
  meaningful `param_*` spread.

Keep the row format of the existing table: experiment, where it ran, dates, panels,
models, arms, trials, studies, paths, suites, status.
