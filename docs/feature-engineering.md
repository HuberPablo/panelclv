# Feature Engineering

How `panelclv` turns a customer-period panel into the `(N, T, F)` tensors the models read,
which features exist, and why each was built. Results of using them are not here: they
are in `docs/insights-real-panels.md` (real panels) and `docs/insights-synthetic-grid.md`
(synthetic grid).

## 1. The rule every feature obeys

The model is a classifier over count classes whose forecast is an autoregressive Monte
Carlo rollout (`CLAUDE.md`). So:

> Every feature the model reads at holdout step *t* is either **known in advance** or
> **computable from the model's own sampled history**. Anything else would leak the
> answer, and cannot exist in this package.

| Concern | Module |
| --- | --- |
| Feature declaration and validation | `configs/panel_config.py` |
| Panel → tensors, calendar features, embeddings | `data_preparation/panel_dataset.py` |
| Autoregressive (AR) features | `data_preparation/ar_features.py` |
| Behavioural clusters | `data_preparation/cluster_features.py` |
| Re-computing features during the rollout | `models/monte_carlo_forecasting.py` |
| Covariate-subset search | `tuning/optuna_tuning.py` |

## 2. Roles

A feature is a numeric panel column with a **role**, declared once in `PanelConfig`.
`seq_cols` (the channel names in tensor order) is the contract: models address channels by
name, so a new panel needs a new config, not a code edit. Channel order is fixed:
`target → time → known_future → observed_past → static → ar_features → cluster_features`.

| Role | Field | In the holdout |
| --- | --- | --- |
| target | `target_col` | never fed; replaced by the sampled class each step |
| time | `time` (+ `time_features` flags) | computed from the calendar |
| known future | `known_future` | read from the holdout tensor (promo calendar, `year_idx`) |
| observed past | `observed_past` | **unsupported, dropped with a warning**: it has no honest holdout value |
| static | `static` | constant per customer |
| AR features | `ar_features` | recomputed from the sampled target (§4) |
| cluster | `cluster_features` | frozen at the end of calibration (§5) |

## 3. Calendar features

Opt-in through `time_features`; `TIME_FEATURE_FLAGS` in `panel_config.py` is the source.

| Flag | Columns | Formula |
| --- | --- | --- |
| `add_year_idx` | `year_idx` | `year − year(training_start)` |
| `add_week_sin_cos` | `week_sin`, `week_cos` | `sin/cos(2π·w / 52)` |
| `add_month_sin_cos` | `month_sin`, `month_cos` | `sin/cos(2π·(m−1) / 12)` |
| `add_dayofyear_sin_cos` | `day_sin`, `day_cos` | `sin/cos(2π·(doy−1) / 365)` |

- **Week convention (ADR-0009).** Valendin's `dayofyear // 7`, capped at 51: 52 weeks,
  0..51. Every stored panel is built on it (`period_calendar.week_of_year`,
  `complete_week_grid`); a panel on another rule is silently misaligned.
- **Why sin/cos.** Week 52 and week 1 are neighbours; the circle encodes that, and is
  defined arbitrarily far into the holdout.
- **`year_idx` caveat.** Monotone, so the holdout takes values never seen in calibration.
  A model leaning on it extrapolates a trend. It is not auto-assigned to a role.

## 4. Autoregressive features

Functions of the target's own past, recomputed at every rollout step from the count the
model just sampled. They bring the recency/frequency/age structure of the BTYD models
into the neural input. A transaction is `target > 0`.

### 4.1 The running state and the features read off it

| State | Definition |
| --- | --- |
| `since` | periods since the last transaction (0 in a transacting period; before the first, `t + 1`) |
| `ever` | any transaction yet |
| `cum_txn` | active periods so far |
| `cum_cnt` | sum of counts so far |
| `tenure` | periods since the first transaction (0 at and before it) |

| Feature | Value | Pareto/NBD analogue | Bounded? |
| --- | --- | --- | --- |
| `period_since_last_transaction` | `since` | recency gap | no, window-capped |
| `period_since_first_transaction` | `tenure` | age **T** | no, window-capped |
| `cumulative_transactions` | `cum_txn` | frequency **x** | no |
| `cumulative_count` | `cum_cnt` | — | no |
| `has_transacted_before` | `1[ever]` | — | yes |
| `active_in_last_<K>_periods` | `1[ever and since < K]` | windowed activity | yes |
| `transaction_rate` | `cum_txn / max(tenure, 1)` | rate **λ** | yes |
| `log_period_since_last_transaction` | `log(1 + since)` | Lomax log-survival coordinate | no, compressed |
| `log_period_since_first_transaction` | `log(1 + tenure)` | — | no, compressed |
| `saturating_recency_<C>_periods` | `since / (since + C)` | — | yes, in [0, 1) |
| `saturating_tenure_<C>_periods` | `tenure / (tenure + C)` | — | yes, in [0, 1) |
| `recency_over_tenure` | `(T − t_x) / T`, 0 before the first purchase | share of life spent silent | yes |

All states are integers, so the training-time precompute and the rollout's incremental
update agree exactly (§7).

### 4.2 Why bounding matters

`period_since_last_transaction` and `period_since_first_transaction` cannot exceed
`T_CAL` while the model is fitted, but keep counting through the holdout. So the holdout
lands outside the range the weights were trained on. Measured on electronics (104
calibration weeks) by `scripts/measure_ar_support.py`:

| Feature | Holdout cells outside the calibration range | Worst cell, calibration sd past the max |
| --- | ---: | ---: |
| `period_since_first_transaction` | 88.8% | 1.74 |
| `period_since_last_transaction` | 37.7% | 1.90 |
| `cumulative_transactions` | 0.04% | 0.64 (9.19 on CDNOW) |
| bounded features | 0% | — |

A network does not go quiet past its training range; it continues along its fitted
slope (Xu et al., ICLR 2021, [arXiv:2009.11848](https://arxiv.org/abs/2009.11848)).
Unlike ordinary covariate shift (Shimodaira 2000), the supports barely overlap, so
nothing can be reweighted. Standardisation (§6) rescales a counter but does not stop it
climbing. Pareto/NBD conditions on the same `(t_x, x, T)` without trouble because its
likelihood encodes the decay analytically. **The information is right; an unbounded
encoding of it is not usable in a rollout.**

There are two ways to keep a clock usable:
- **Collapse the tail** onto a value calibration contains (the nested flags): safe, but
  every gap past the deepest bin looks the same.
- **Shorten the distance** travelled outside (log, saturating, ratio): keeps resolution.

The escape fraction cannot rank two encodings of one clock, because it is invariant to
order-preserving transforms. That is why the second route was specified separately
(`.scratch/ar-encoding-support/spec.md`).

### 4.3 The encodings tested, and why

Each arm is a fixed `ar_features` tuple. Every neural arm also carries the embedded
target count.

| Arm | Columns | Why it was tested |
| --- | --- | --- |
| `no_ar` | — | Baseline: the model sees its own past counts only. |
| `ar_unbounded` | `period_since_last_transaction`, `cumulative_transactions`, `period_since_first_transaction` | Pareto/NBD's own sufficient statistic `(t_x, x, T)`: hand the network exactly what the benchmark conditions on. Kept afterwards as the known-broken reference. |
| `ar_bounded_K` | nested `active_in_last_{2,4,8,…,K}_periods` + `has_transacted_before` | The same silence information, collapsed past `K` so no holdout value leaves the fitted range. `active_in_last_1` is omitted because the target channel already carries it. |
| `ar_log` | `log_period_since_last_transaction`, `cumulative_transactions`, `log_period_since_first_transaction` | The unbounded set with only the coordinate changed. Pareto/NBD's Lomax survival is linear in `log(1 + gap)`, so a network's linear extrapolation is the right shape there. Isolates the effect of `log1p`. |
| `ar_saturating` | `saturating_recency_C_periods`, `transaction_rate`, `saturating_tenure_C_periods` | Both clocks bounded in [0, 1) rather than compressed, plus the bounded rate. |
| `ar_ratio` | `recency_over_tenure`, `transaction_rate`, `saturating_tenure_C_periods`, `has_transacted_before` | The Pareto/NBD triple made bounded: `t_x/T`, `x/T` and a saturating `T`. It keeps the resolution the flags discard. `has_transacted_before` disambiguates the ratio's 0. |
| `ar_bounded32ratio` | `ar_bounded_32` + `ar_ratio` | The flags and the ratio together, to test whether each set does its own job when combined. |

`C` is about a quarter of the calibration window: 10 on CDNOW, 26 elsewhere.

**Choosing the deepest bin `K`.** Choose it against the calibration window, not by
copying a number.
- `check_arm_depth` refuses `K ≥ T_CAL`, where the flag duplicates
  `has_transacted_before` in calibration.
- A `K` close to `T_CAL` is also unsafe. On CDNOW, `K = 32` against 39 calibration
  periods leaves 3.5% of calibration cells past the bin and 68.9% of holdout cells.
- Depths used: 16 or 32 on CDNOW, 32 or 52 on electronics, 32 on gift, multichannel and
  the synthetic grid, 52 on electronic_5y. A bare `ar_bounded` means `ar_bounded_32`,
  except in `real_panel_arms` on CDNOW, where it means 16.

**Adding a feature.** Extend `_base_states` (vectorised precompute) and
`ARFeatureState.update` (rollout), add a branch in `_render` and a name in
`parse_ar_feature`. `tests/test_ar_features.py` asserts the two paths agree.

## 5. Behavioural cluster (`kmeans_<K>`)

**What it is.** Each customer's `(t_x, x, T)` at the last calibration period is
standardised and partitioned by k-means (`KMeans(n_clusters=K, n_init=10,
random_state=0)`). The group index becomes one channel, named after the feature
(`kmeans_8` = K of 8). It is embedded automatically with cardinality `K`, since a group
index has no order to standardise.

**Why it was tested.** It carries the same information as `ar_unbounded` (the Pareto/NBD
triple), but as a single bounded, frozen category:
- **No rollout machinery.** The simulator overwrites only the target and AR channels, so
  the label rides through the holdout untouched.
- **It cannot escape its support.** The holdout takes the same `K` values as calibration.
- **It is deterministic.** It has a fixed `random_state`, so it adds no hidden variance
  across studies.

The trade-off is that it cannot update when a simulated customer goes quiet.

**How K was chosen.** It is declared, never fitted: there is no elbow, silhouette or gap
statistic. K is also the width of an embedding table, so a larger K describes customers
more finely but gives each group fewer customers to learn from. K is an arm, never an
Optuna knob, so arms stay comparable. The ladder 4 / 8 / 16 was swept once (family F);
everywhere else K = 8, the middle rung.

**One deliberate deviation.** The label is fitted on the full calibration window, which
includes the validation window that early stopping scores. Every other calibration-derived
quantity uses that same window: cardinalities, Pareto/NBD and the ADR-0008 refit. Holdout
scoring is clean. Whether selection is affected is experiment E2
(`docs/model-selection.md`).

## 6. Encoding: embeddings and standardisation

- **Embedded columns** (`embedded_cols`) → `nn.Embedding(card, √card + 1)` → LayerNorm →
  Linear → LayerNorm. The target embedding is kept apart; the other embeddings are
  summed into a context vector.
- **Continuous columns** → one shared `Linear(n → embedding_dim)` → LayerNorm, added into
  the context.
- **Standardisation.** Every non-embedded channel is set to mean 0, sd 1 by
  `standardize_covariates`, fitted on calibration and applied to both windows
  (`covariate_stats`). It is needed because the shared projection would otherwise let
  the largest-unit channel (recency in weeks) dominate. The rollout re-applies the same
  `(mean, std)` to every recomputed AR value.
- **Cardinality by role.** For the target: `clip_target_upper + 1`. For time and known
  future: the max over both windows (these values are given). For everything else: the
  calibration max only. A constant column raises.
- **Known-future drift.** `warn_known_future_drift` flags embedded known-future categories
  that occur only in the holdout, whose embedding rows would never be trained.

## 7. Leakage discipline

AR features are never computed over the full series, which spans the holdout.
1. In the panel, AR columns are created as zero placeholders.
2. After the cohort filter and target clipping, they are filled **on the calibration
   window only** (`compute_ar_feature_columns`).
3. The holdout's AR columns stay zero and are **never read**. The rollout seeds an
   `ARFeatureState` from the calibration history and advances it with each sampled count.

Both paths share `_render`, so the training and inference distributions match. At rollout
step *t*:

| Channel | Source |
| --- | --- |
| target | previous sampled class |
| AR features | `ARFeatureState.update(sample)` |
| time / known future / static / cluster | true holdout values, legitimately known |

## 8. Target handling and cohort

- **`clip_target_upper`** clips the training target only, and sets the head size. The
  holdout stays unclipped for scoring. AR features are filled after clipping.
- **`require_calibration_activity`** (on by default) keeps customers with a calibration
  purchase, which is Valendin et al.'s cohort rule. It runs inside `prepare_dataset`, so
  Pareto/NBD fits the same cohort.

## 9. Feature selection as a hyperparameter

`run_optuna_study(removable_features=[...])` toggles single columns or groups (e.g.
`("week_sin", "week_cos")`). It works by slicing the built tensors (`select_features`),
with `ar_features` filtered in lockstep. Each trial records `selected_features` /
`dropped_features`. The search scores teacher-forced validation CE, which cannot see
rollout drift, so it may keep an extrapolating channel. No study suite passes
`removable_features`, so no suite can drop the feature under test.

## 10. Configuration example

```python
cfg = PanelConfig(
    id_col="Id", target_col="Transactions", frequency="weekly",
    time_cols=("year", "week"), periods_per_year=52,
    training_start="1999-01-01", training_end="2000-12-31",
    validation_start="2000-01-01",
    holdout_start="2001-01-01", holdout_end="2001-12-31",
    known_future=("high.season",), static=("Gender", "Income"),
    time_features={"add_week_sin_cos": True},
    ar_features=("active_in_last_2_periods", "active_in_last_4_periods",
                 "active_in_last_8_periods", "active_in_last_16_periods",
                 "active_in_last_32_periods", "has_transacted_before"),
    cluster_features=("kmeans_8",),
    clip_target_upper=6,
    embedded_cols={"Transactions": "auto", "Gender": "auto", "high.season": "auto"},
)
```

## 11. Failure modes

These fail at config or `prepare_dataset` time, before any training:
- an unknown `ar_features` or `time_features` name, or `active_in_last_K` with `K < 1`
- date windows out of order, or a validation window left empty
- an AR name colliding with a panel column
- missing or non-numeric columns, or NaN in a selected column
- an empty window or an empty cohort
- ragged or misordered customers
- a pinned cardinality too small, or a constant `"auto"` column
- `clip_target_upper` at or above the pinned target cardinality

These only warn:
- dropped `observed_past` columns
- known-future embedding drift

**Not supported:**
- `observed_past` covariates
- ragged panels (pad them upstream)
- string-valued statics (encode them first)
