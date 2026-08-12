# Audit: the data lane

Type: task
Status: resolved

## Question

Read `configs/` (494 lines) and `data_preparation/` (1799 lines: `dynamic_panel_dataset`
1030, `pareto_simulation` 544, `ar_features` 225) and report against the four
dimensions every lane audit shares:

1. **Unreachable** — no caller in `src/` or a live entry point (see the map's Notes for
   the rule and the thesis carve-out).
2. **Duplicated** — the same concept implemented twice, or one concept named two ways.
3. **Over-parameterised** — options, flags and escape hatches nobody exercises.
4. **Hardcoded dataset assumptions** — column names, frequency, week-of-year
   arithmetic. Record only; acting on these is out of scope for this effort.

Specific things known to want a ruling here: `pareto_simulation.py` exposes
`_seasonal_weekly_multiplier` as private yet `studies/pnbd_grid.py` imports it, so the
seam is in the wrong place; and `dynamic_panel_dataset.py` is the single largest module
in the package with no obvious internal split.

Report is evidence, not decisions — ticket 06 decides. Do not change code.

## Answer

**Scope read in full:** `configs/panel_config.py` (494), `data_preparation/ar_features.py`
(225), `dynamic_panel_dataset.py` (1030), `pareto_simulation.py` (544) — 2293 lines plus
the two `__init__.py` files.

**Method.** For every public symbol and every `prepare_dataset` return key, grepped for
callers across `src/panelclv/`, the live entry points (`scripts/run_studies.py`, the two
`validate_*.py`, the two `main_plot*.py`, the four live notebooks) and `tests/`, keeping
those three populations separate. Claims about validation behaviour were confirmed by
running the code, not read off the docstrings.

**Headline.** Nothing in this lane is dead code. Every module and every public function
has a caller in `src/` or a live entry point, or an intra-module one. The rot here is a
different shape: an over-wide public surface hiding a module with no seams, a vestigial
dict-adapter layer between `PanelConfig` and `prepare_dataset`, three return keys nobody
reads, five concepts implemented twice, and 2068 of the lane's 2293 lines with no
dedicated test.

---

### 1. Unreachable

**No module or public function is unreachable by the map's rule.** The candidates that
look dead on a naive grep — `get_seq_cols`, `resolve_embedded_cols`, `add_time_features`,
`add_period_start`, `validate_columns`, `warn_known_future_drift`,
`standardize_covariates`, `make_block`, `select_active_cohort`,
`normalize_embedded_cols`, `simulate_pareto_nbd_panel`, `parse_ar_feature` — are all
called from inside their own module. They are *over-exposed*, not dead; see the table in
§3.

What *is* unreachable is smaller-grained:

**Three return keys of `prepare_dataset` that no reader anywhere consumes.** I checked
all 24 keys against `src/`, entry points, notebooks and tests:

| key | readers |
| --- | --- |
| `"N  "` | **none** |
| `"panel"` | **none** |
| `"holdout_panel"` | **none** |

- `"N  "` (`dynamic_panel_dataset.py:966`) carries **two trailing spaces** in the key
  literal, so `data["N"]` raises `KeyError`. `README.md:56`-ish region documents
  `data_full["N"]` — the only place in the repo that names it — so the README's example is
  broken. Nothing in live code depends on either spelling.
- `"panel"` is the whole engineered panel (both windows, all engineered columns) and
  `"holdout_panel"` the holdout slice. Both are DataFrames retained for the lifetime of
  the `data` dict. `"train_panel"` by contrast is genuinely live — `studies/runner.py:201`,
  `evaluation/plot_utils.py:272` and `scripts/main_plot.py:174` feed it to the Pareto/NBD
  benchmark.

**Unreachable branches inside reachable functions.**

- `add_time_features`' three frequency-incompatibility `raise ValueError` branches
  (`dynamic_panel_dataset.py:277-280, 287-290, 295-298`) cannot fire. By the time
  `prepare_dataset` calls it, `PanelConfig._resolve_time_features` has already *dropped*
  every frequency-incompatible flag with a warning. `add_time_features` has no other
  caller, so the two layers disagree on policy (warn-and-drop vs. raise) and only the
  outer one is ever observed.
- The **`observed_past` role is inert end to end**: declared as a `PanelConfig` field,
  plumbed into `.schema` as `observed_past_time_varying_inputs`, then emptied with a
  warning at `dynamic_panel_dataset.py:701-712`. Every live caller passes
  `observed_past=()` anyway (`scripts/verify_transformer_training.py:44`,
  `Pareto_Datasets.ipynb`, `Study.ipynb`, `Data_integration_LSTM_v2.ipynb`), so the
  warning has never fired in a live run. A role that is declarable, documented in three
  docstrings, and unconditionally discarded.
- **`daily` frequency has no caller and no test.** `date_col` is never set anywhere in the
  repo; `add_dayofyear_sin_cos` is never requested. That leaves the daily branch of
  `_resolve_time_index`, the daily branch of `add_week_sin_cos`, the whole
  `add_dayofyear_sin_cos` block and the daily branch of `add_period_start` exercised by
  nothing. **`monthly` is live** — `notebooks/Data_integration_LSTM_v2.ipynb` cell 6 runs
  `frequency="monthly"` with `time_cols=("year","month")` and `add_month_sin_cos` — so the
  frequency axis is not uniformly speculative, only its daily third.
- `add_time_features` creates a bare `dayofyear` column (`:300`) that
  `_FLAG_TIME_COLUMNS["add_dayofyear_sin_cos"]` does not list (`panel_config.py:131`), so
  no role ever claims it and it never enters `seq_cols`. Already-drifted duplication (§2)
  producing an orphan column.

**Public but with no caller outside this LANE** — over-exposure, listed here as evidence for
ticket 05's ledger rather than as kill candidates. **Corrected after the ledger merge:** the
heading originally read "outside its own module", which mislabels three rows — the callers of
`normalize_embedded_cols` (`dynamic_panel_dataset.py:76`), `PanelConfig.data_config` (`:617`) and
`PanelConfig.schema` (`:605`) sit in a *different module of the same lane*, as the caller column
below always said. The evidence is unchanged; the granularity claim was wrong. Read the table as
lane-scoped:

| symbol | intra-module caller | external callers |
| --- | --- | --- |
| `dynamic_panel_dataset.get_seq_cols` | `prepare_dataset` | none |
| `.resolve_embedded_cols` | `prepare_dataset` | none |
| `.add_time_features` | `prepare_dataset` | none |
| `.add_period_start` | `prepare_dataset` | none |
| `.validate_columns` | `prepare_dataset` | none |
| `.warn_known_future_drift` | `prepare_dataset` | none |
| `.standardize_covariates` | `prepare_dataset` | none |
| `.make_block` | `prepare_dataset` | none |
| `.select_active_cohort` | `prepare_dataset` | none |
| `panel_config.normalize_embedded_cols` | `PanelConfig`, `prepare_dataset` | none |
| `PanelConfig.data_config` | `prepare_dataset` | none |
| `PanelConfig.schema` | `prepare_dataset` | none |
| `pareto_simulation.simulate_pareto_nbd_panel` | `generate_pnbd_study` | none |
| `ar_features.parse_ar_feature` | `validate_ar_features`, `_render` | tests only |
| `ar_features.compute_ar_feature_columns` | — | `prepare_dataset` (lane-internal); tests |

Note `simulate_pareto_nbd_panel` is the *headline documented API* of its module — 60 lines
of docstring — reached only through `generate_pnbd_study`. And `PanelConfig.data_config` /
`.schema` are the vestigial layer: they exist to hand `prepare_dataset` "the dict forms
the existing internals consume" (`panel_config.py:82-84`), a seam left over from the
four-loose-dicts era that no consumer outside the lane has ever used.

**Thesis carve-out applies to `pareto_simulation.py` regardless of the above.** It
generated the synthetic grid panels behind the grid figures
(`scripts/make_grid_figures.py:18`, `scripts/recheck_season_churn.py:28`,
`studies/pnbd_grid.py:42`, `notebooks/Pareto_Datasets.ipynb`), so it is alive by the
carve-out even where individual entry points are one-shot.

---

### 2. Duplicated

**Three incompatible week-index conventions**, none of them shared:

| site | convention | range |
| --- | --- | --- |
| `pareto_simulation.py:246` | `week_idx % 52`, year rolls every 52 weeks | 0..51 |
| `dynamic_panel_dataset.add_period_start:325-328` | `Jan-01 + week*7 days` | assumes 0-based |
| `dynamic_panel_dataset.add_time_features:274` (daily) | `isocalendar().week - 1` | 0..52 |
| `scripts/validate_valendin_lstm.py:108,119` | `dayofyear // 7` clipped to 51 | 0..51 |

The daily branch can emit week 52 while the sin/cos divisor is a hardcoded `52`
(`:275-276`), so week 52 aliases exactly onto week 0. Nothing enforces that a panel's week
column matches the convention `add_period_start` assumes.

**Two Pareto/NBD forward simulators.** `pareto_simulation.simulate_pareto_nbd_panel` uses
the exact vectorised trick (never materialises timestamps);
`scripts/validate_pareto_benchmark.py:37 make_synthetic_elog` re-implements the same
generative model with an explicit per-customer event loop. The duplication is *partly
justified* — the validation script needs sub-week event timestamps to feed R's BTYDplus,
which the vectorised version structurally cannot produce — but the shared part (draw
`lambda ~ Gamma(r,alpha)`, `mu ~ Gamma(s,beta)`, `tau ~ Exp(mu)`) is written twice with
different parameters and different acquisition conventions. Flagging with care: this script
is one of the map's two floor gates.

**Two cohort filters.** `select_active_cohort` (`:530`) implements "positive target total
over the calibration slice"; `scripts/validate_valendin_lstm.py:98` implements
`first_seen <= TRAINING_END` directly. `panel_config.py:227` states these are equivalent —
so the equivalence is asserted in a comment while both implementations stand.

**The standardization transform is applied in two places.**
`standardize_covariates` (`:430`) fits and applies `(x - mean)/std`;
`models/monte_carlo_forecasting.py:172-174` and `:275-277` re-apply the same arithmetic
inline to the AR features it recomputes per rollout step. Correct today, but the transform
is now a convention shared across two subpackages rather than one function, and the
docstring at `:481-486` has to say so in prose.

**Four encodings of the time-feature flag set**, all in the lane:
`_KNOWN_TIME_FLAGS` (`panel_config.py:107`, typo-checking), `_COMPATIBLE_TIME_FLAGS`
(`:116`, frequency rules), `_FLAG_TIME_COLUMNS` (`:128`, output columns), and the
`if/elif` frequency logic inside `add_time_features` itself. The `dayofyear` orphan column
above is these four already having drifted apart.

**Two config→dict views.** `PanelConfig.data_config` (`:392`) and `PanelConfig.to_dict`
(`:414`). `to_dict`'s own docstring says it "Mirrors (and supersedes) the partial
`.data_config` view" (`:423`) — yet `.data_config` remains, and is the one
`prepare_dataset` actually consumes. `data_config` also emits *optional* keys (`time_cols`,
`date_col`, `clip_target_upper` appear only when non-`None`), which is why the whole body
of `prepare_dataset` reads its config through `config.get(...)`.

**`prepare_dataset` reads its config twice, two different ways.** Lines 597-616 read
`config.ar_features`, `.schema`, `.time_features`, `.embedded_cols` off the object; line
618 then rebinds `config = config.data_config` and everything after reads a dict. The
original object survives only as `panel_config` to be handed back out. One object, two
access idioms, in one function.

**Feature roles are named twice.** `PanelConfig` fields are `time`, `known_future`,
`observed_past`, `static`; `.schema` renames them `time`,
`known_future_time_varying_inputs`, `observed_past_time_varying_inputs`,
`static_covariates`. The translation exists solely to feed `get_seq_cols`.

**The `schema` dict has two shapes.** `_SCHEMA_GROUP_ORDER` (`:92`) lists six groups
including `ar_features`, but `PanelConfig.schema` emits only five and never `ar_features`
(verified at runtime). `prepare_dataset:718-719` injects the sixth. So "a schema" means one
thing inside `prepare_dataset` and another everywhere else.

**Two manifests per generated Pareto/NBD study.** `generate_pnbd_study` writes
`index.csv` *and* returns the same frame; `list_pnbd_datasets` then rebuilds it from the
per-dataset `config.json` files, its docstring explaining that it does so because
`index.csv` may be stale (`:483-484`). Separately, `study_config.json` re-states fields
(`r`, `s`, `birth_purchase`, `seasonal_*`, `n_customers`, `n_weeks`, `start_year`) that
every dataset's `config.json` already carries.

---

### 3. Over-parameterised

Options no live caller exercises:

- **`daily` frequency** — see §1. `date_col` set nowhere.
- **`observed_past`** — see §1. Always `()`, always discarded.
- **Pinned-int embedding cardinalities.** Every live `embedded_cols` spec in the repo is a
  dict whose values are all `"auto"` (`run_studies.py`, `verify_transformer_training.py:45`,
  all four live notebooks, `test_golden_end_to_end.py:126`). So: the pinned-int coverage
  check in `resolve_embedded_cols` (`:186-192`), the pinned-int shape check in
  `PanelConfig._validate_embedded_cols` (`:378-385`), and the clip-vs-pinned cross-check at
  `prepare_dataset:632-648` all guard a path nobody uses.
- **`embedded_cols` as a bare list.** `normalize_embedded_cols`' list/tuple/set branch
  (`:172-173`) — never used; every caller passes a dict.
- **`require_calibration_activity=False`** — set nowhere in the repo. The filter is always
  on, so `select_active_cohort`'s empty-cohort error path and the `if` at
  `prepare_dataset:773` are effectively unconditional.
- **AR features: 7 supported, 3 used.** Live: `period_since_last_transaction`,
  `active_in_last_3_periods` (`Data_integration_LSTM_v2.ipynb`), `cumulative_transactions`
  (`test_golden_end_to_end.py:125` only). Never requested anywhere:
  `has_transacted_before`, `cumulative_count`, `period_since_first_transaction`,
  `transaction_rate`. All four are covered by `tests/test_ar_features.py`, so they are
  tested-but-unused rather than untested.
- **`birth_purchase`** — never set `True`; `Pareto_Datasets.ipynb` explicitly relies on the
  `False` default.
- **`periods_per_year`** — only ever the frequency default (52) or passed explicitly *as*
  52. Note it is also the `week_sin/cos` divisor only in the weekly branch; the daily
  branch hardcodes 52 instead.

Not over-parameterised, checked and cleared: `verbose` (defaults `True` for the notebooks,
`False` from every programmatic caller), the `seasonal_*` knobs (live via
`Pareto_Datasets.ipynb`, `studies/pnbd_grid.py:167`, `scripts/recheck_season_churn.py:47`),
`clip_target_upper` (live everywhere, 6 in production, 4 in the golden test).

**A validation hole worth recording under this heading**, because it is what the
warn-don't-error policy costs: `PanelConfig` validates that a time-feature flag suits the
frequency, but never that `time_cols`' *period column* does. Verified at runtime —

    PanelConfig(frequency="weekly", time_cols=("year","month"), ...)   # accepted

is accepted silently, and `add_period_start` then computes `Jan-01 + month*7 days`,
collapsing a year into 12 periods. `notebooks/Study.ipynb` cell 8 contains exactly this
mistake against a *weekly* panel CSV, with the stored
`UserWarning: time feature 'add_month_sin_cos' is not compatible with frequency 'weekly'`
as its only trace. It corrupts nothing today — cell 10 runs a different config object
(`cfg_2yTrain_1yPred_NoCov`), so cell 8's `cfg` is unused — but it is a live demonstration
that the two halves of a frequency declaration are validated independently and a
contradiction between them surfaces as a warning about something else.

---

### 4. Hardcoded dataset assumptions

Recorded only; acting on these is out of scope per the map.

- **`pareto_simulation.py` is weekly, throughout.** `WEEKS_PER_YEAR = 52` at module scope,
  `n_weeks`, `_PANEL_SCHEMA = {"time_cols": ["year","week"], "frequency": "weekly"}`,
  `lambda` documented as purchases *per week*, `alpha`/`beta` in week units,
  `n_weeks_for_churn_rate=521`, and `_seasonal_weekly_multiplier(period=52)`. A weekly-only
  generator inside the subpackage that is supposed to be frequency-parameterised.
- **`_PANEL_SCHEMA` hardcodes `id_col="Id"`, `target_col="Transactions"`** — the exact
  column names `PanelConfig` exists to keep out of module code. `generate_pnbd_study`
  forces them (`:411`) with no way to override, while `simulate_pareto_nbd_panel` does
  accept `id_col`/`target_col`.
- **`add_period_start` weekly** assumes week 0 = Jan 1 and a uniform 7-day step
  (`:325-328`), so periods drift against the real calendar across years and week 52 (if a
  panel is 1-indexed or ISO) lands on Dec 31.
- **Hardcoded divisors** in `add_time_features`: `52` for the daily week branch (`:275-276`),
  `12` for months (`:291-292`), `365` for day-of-year (`:301-302`) — the last ignores leap
  years. Only the weekly branch honours `periods_per_year`.
- **`start_year=1999` default** in both `simulate_pareto_nbd_panel` and
  `generate_pnbd_study` — the thesis panel's epoch baked in as a default.
- **`base_year = pd.Timestamp(config["training_start"]).year`** (`:627`) ties `year_idx`'s
  origin to the training window, so the same panel trained on a different window yields a
  different `year_idx` for the same calendar year. Correct for a single run, and worth
  noting because `year_idx` is a `known_future` covariate carried into the holdout.
- **`_pct` folder labels round to integer percent** (`pareto_simulation.py:271`). Verified:
  `_pct(0.104) == _pct(0.096) == _pct(0.1) == "10"`. A grid with two rates inside the same
  percent silently writes both into one `Dataset_10_20/` folder, the second replicate set
  overwriting the first. Latent, not triggered by any grid actually run.
- **Non-deterministic study naming.** `_auto_study_name` uses `datetime.now()` (local, no
  tz) while `created_at` uses `datetime.now(timezone.utc)`. Study folder names are not
  reproducible from config + seed, which sits against priority 2.

---

### The two rulings the ticket asked for

**`_seasonal_weekly_multiplier`'s seam is in the wrong place — confirmed, and it is a
cross-subpackage private import.** `studies/pnbd_grid.py:42-46` imports
`_seasonal_weekly_multiplier` *and* `WEEKS_PER_YEAR` from `pareto_simulation`, and
`pnbd_grid.py:160` documents *why* it must be the same function: it reconstructs the
seasonal pattern a stored study was generated with, so a re-implementation would silently
disagree with the data on disk. That is a legitimate need reaching through a private name.
Note the sibling case resolved the other way: `scripts/validate_valendin_lstm.py:73`
defines its own `WEEKS_PER_YEAR = 52` rather than importing the public one.

**`dynamic_panel_dataset.py` has no internal split — but the audit found the seams.**
The 1030 lines are one public orchestrator plus nine module-private-in-practice helpers
(§1 table), and they group cleanly by what they touch:

| group | functions | touches |
| --- | --- | --- |
| schema/embedding resolution | `get_seq_cols`, `_known_future_cols`, `resolve_embedded_cols` | schema dict + panel maxima |
| calendar | `_resolve_time_index`, `add_time_features`, `add_period_start` | frequency rules |
| validation | `validate_columns`, `warn_known_future_drift` | panel, warns |
| cohort | `select_active_cohort` | panel |
| tensorisation | `make_block`, `standardize_covariates` | ndarray only |
| orchestration | `prepare_dataset` (422 lines, 8 numbered steps) | everything |

The calendar group is where the duplication of §2 concentrates and is the only group
coupled to `frequency`; the tensorisation group touches no pandas and no config at all.
`prepare_dataset` itself is 422 lines of numbered steps (1, 2, 2b, 3, 4, 5, 5a-pre, 5a,
5a-bis, 5b, 5c, 6, 6b, 7, 7b, 8, 8b) — the step-letter suffixes are the record of where
each later concern was inserted into a sequence that was never re-cut.

---

### Cross-lane observations (recorded so they aren't lost; not this lane's to decide)

- **`data_preparation/__init__.py` documents 2 of its 3 modules** — `pareto_simulation` is
  absent. It also states "Building the raw customer-period panel itself now lives outside
  the package — see `notebooks/archive/dataset_building.ipynb`", pointing at the
  deliberately frozen archive, while `pareto_simulation.py` in that very package builds
  panels.
- **`src/panelclv/__init__.py` lists 8 subpackages; there are 9** — `studies` is missing.
  `tests/test_imports.py:17-27` has the same 8-entry list, so no test notices.
- **`configs/__init__.py` is a single docstring line** and mentions "embedded_cols
  normalization" as if it were a second module.
- **`README.md`'s quickstart reads `data_full["N"]`**, which raises `KeyError` (§1).

### Test coverage of this lane

`tests/test_ar_features.py` (171 lines) is the lane's only dedicated test file, covering
`ar_features.py` (225 lines) well — including the four AR features nothing uses. The other
**2068 lines have no dedicated test**: `panel_config.py`, `dynamic_panel_dataset.py` and
`pareto_simulation.py` are exercised only indirectly, via
`tests/test_golden_end_to_end.py` (one weekly happy path: `clip_target_upper=4`, two AR
features, `add_year_idx` + `add_week_sin_cos`, all-`"auto"` embeddings) and
`tests/test_studies_analysis.py`. Nothing tests any `PanelConfig` validation error, the
monthly path, the daily path, `warn_known_future_drift`, `standardize_covariates` in
isolation, or `pareto_simulation.py` at all — the 544-line generator behind the thesis's
synthetic grid panels. The generated datasets do record their seeds and parameters in
`config.json`, so re-derivability is at least checkable.
