# Every study run, and what is still owed

A global inventory of every study suite on disk under `Studies/`, what it varied, at what
budget, on which windows — so the gaps are visible without re-deriving them from **6,533
archived suite directories holding 10,136 completed studies**.

Built by reading the archived `config.json` and `results.csv` of every suite (they record
the panel config, the model specs, the budget and the embedder Optuna sampled, verbatim)
and cross-checking the commands that own the "what is owed" question:
`scripts/run_real_panel_arms.py --check-complete`, `scripts/run_real_panel_ar.py
--check-complete`, `scripts/run_real_panel_benchmarks.py --check-complete` and
`scripts/reconcile_grid.py --grid <name>`. Those commands are the authority; this document
is a snapshot of them taken **18 September 2026**, plus the families they do not cover.

---

## 1. How to read a row

Every result in this repo is quoted with the same five things. A number without them is
not a result, because none of the five is recoverable from the number.

**Replications × trials × paths.** Written as *"20 studies × 100 trials × 500 paths"*.

| Term | Meaning |
|---|---|
| **studies** (replications) | Independent repetitions of the whole pipeline — its own Optuna study, its own ADR-0008 refit, its own forecast — seeded `base_seed + i`. Model training is unseeded by design (`CLAUDE.md` priority 3), so these differ in weight init, shuffling and dropout. **The spread across them is part of the result.** Carried either as `n_studies_per_model` inside one suite (the ablations) or as one suite per replication (the `real_panel_*` families). |
| **trials** | Optuna trials inside one study. The winner is the trial with the lowest teacher-forced validation cross-entropy. |
| **paths** (sims) | `n_simulations` — Monte Carlo rollout paths averaged into one forecast. |
| **shard `a`–`e`** | Disjoint 20-seed blocks of the same arm: `a` = 43–62, `b` = 63–82, `c` = 83–102, `d` = 103–122, `e` = 123–142 (`base_seed` 42, 62, 82, 102, 122). Running all five gives 100 replications of one arm. |

**Embedder.** Which strategy turned features into the model's input vector.

| Token | Meaning |
|---|---|
| `valendin` | Each feature keeps its own sqrt(cardinality)+1 vector. **The registry default, and the only embedder any archived neural study has ever used.** Recorded per study in the `param_embedder` column of `results.csv`, and `valendin` in all 4,870 neural rows that carry it. Families A, J, K, L and M predate that column — the embedder was not a searched parameter before ADR-0005's seam — so for those 922 rows it is inferred from the code of the day, not read from the archive. |
| `projected` | All features projected to one common width (`embedding_dim` for the LSTM, `d_model` for the Transformer). **Never run** (§6). |
| *frozen* | `ValendinLSTM` does not take an embedder parameter at all: ADR-0004 freezes the published raw sqrt(n)+1 embeddings, so `param_embedder` is empty in its rows. `ParetoNBD` is not a neural model and has none. |

**Arm.** The feature set, named `<ar>-<cluster>-<embedder>[-<calendar>]`; §3 decodes every
token. Where a run wrote a bare `ar_bounded`, §3's depth table resolves it.

**Metrics.** All three aggregate metrics come from
`models.monte_carlo_forecasting.compute_forecast_metrics`, the single scoring authority.

| Metric | What it is | Where it lives |
|---|---|---|
| `bias_percent` | signed total predicted minus total actual, over the actual total | `results.csv` |
| `mape_aggregate` | L1 on the per-period aggregate curve, over the actual total. `MAPE >= \|bias\|` always; the gap is shape error | `results.csv` |
| `rmse` | per-customer per-period. **On panels that are 98–99.7% zeros it separates nothing** — every archived row sits within 0.004 of the all-zero forecast — so it is reported, never ranked on | `results.csv` |
| **Spearman** | rank correlation of per-customer holdout totals: does the model order customers correctly. **Not in `results.csv`** — it needs the panel actuals, so it is recomputed from the stored `Predictions/Prediction_*.csv` by each runner's `--report` | recomputed |

Spearman is therefore available only where a `--report` path still runs against the
archive. §7 lists the two families where it does not.

**Windows.** Calibration (which splits into training and validation at
`validation_start`, ADR-0001) and holdout, given as dates and as period counts
`T_CAL / T_HOLD`. §2 holds them per panel; a family that deviates says so.

---

## 2. The panels

### Two-year calibration — the benchmark windows

These are the windows `scripts/run_real_panel_benchmarks.py` declares and every runner
since 13 September imports rather than restates, so the families cannot drift apart. Week
buckets follow `dayofyear // 7` capped at 51 (ADR-0009).

| panel | customers | calibration | validation from | holdout | T_CAL / T_HOLD | holdout transactions | zero cells | `clip_target_upper` → classes |
|---|---:|---|---|---|---:|---:|---:|---|
| cdnow | 2,357 | 1997-01-01 → 1997-09-29 | 1997-08-05 | 1997-09-30 → 1998-06-30 | 39 / 39 | 1,895 | 98.0% | 4 → 5 |
| electronics | 829 | 1999-01-01 → 2000-12-31 | 2000-01-01 | 2001-01-01 → 2001-12-31 | 104 / 52 | 1,467 | 98.6% | 6 → 7 |
| gift | 2,062 | 2001-02-25 → 2003-02-24 | 2002-02-25 | 2003-02-25 → 2004-02-24 | 104 / 52 | 1,146 | 99.0% | none → 5 (max 4/week) |
| multichannel | 1,402 | 2005-01-01 → 2006-12-31 | 2006-01-01 | 2007-01-01 → 2007-12-31 | 104 / 52 | 228 | 99.7% | none → 5 (max 4/week) |

CDNOW's panel spans 77 weeks, so it keeps the published 39 / 39 split with the last eight
calibration weeks as validation instead of a whole year. Gift's panel opens on 2001 week
8, so its years are counted from there.

### Three-year calibration

Two years fit the weights, the third is the validation window, and the year after is the
holdout. **The holdout year moves with the window**, so a 3-year row forecasts a different
year from its 2-year counterpart, on different actuals. CDNOW's 77-week panel has no room
for it.

| panel | customers | calibration | validation from | holdout | T_CAL / T_HOLD | holdout transactions | zero cells |
|---|---:|---|---|---|---:|---:|---:|
| electronics | 829 | 1999-01-01 → 2001-12-31 | 2001-01-01 | 2002-01-01 → 2002-12-31 | 156 / 52 | 1,541 | 98.7% |
| gift | 2,062 | 2001-02-25 → 2004-02-24 | 2003-02-25 | 2004-02-25 → 2005-02-24 | 156 / 52 | 1,040 | 99.1% |
| multichannel | 1,402 | 2005-01-01 → 2007-12-31 | 2007-01-01 | 2008-01-01 → 2008-12-31 | 156 / 52 | 173 | 99.8% |

### The archived CDNOW window — pre-ADR-0009

**Every CDNOW suite created before 13 September 2026 uses a different holdout** and cannot
be scored beside a current one:

| | calibration | validation from | holdout | T_CAL / T_HOLD |
|---|---|---|---|---:|
| archived (families E, F, G, H, I) | 1997-01-01 → 1997-09-30 | 1997-08-06 | 1997-10-01 → 1998-06-30 | 39 / **38** |
| current (families N, O) | 1997-01-01 → 1997-09-29 | 1997-08-05 | 1997-09-30 → 1998-06-30 | 39 / **39** |

One week, from the `(dayofyear − 1) // 7` week rule ADR-0009 replaced. It is enough to
break scoring outright: §7 records the two report paths it takes down.

### The synthetic panels

| | |
|---|---|
| grid | `seasonal_4x4x10` — 4 mean transaction rates {0.01, 0.05, 0.10, 0.30} × 4 churn rates {0.20, 0.40, 0.60, 0.80} × 10 replicate panels = **160 panels** |
| each panel | 1,000 customers × 156 weeks, four seasonal peaks, generated *by* a Pareto/NBD process |
| windows | calibration 1999-01-01 → 2000-12-31, validation from 2000-01-01, holdout 2001-01-01 → 2001-12-31; 104 / 52 |
| classes | `clip_target_upper=6` → 7 |
| calendar | `add_week_sin_cos`, `add_year_idx` |

A second grid, `seasonal_4x4x10_n3000`, is the same declaration with `n_customers=3000`
and nothing else changed — same axes, same `base_seed=42`, same seasonality, same windows,
same classes and calendar — so the two differ in panel size alone. 1.1 GB on disk against
340 MB; panel checksum `4d7928fff7736b7ba6f67b070990e844` (computed as §2 prescribes,
`index.csv` excluded).

**The generated count is not the trained count.** `require_calibration_activity=True`
drops customers with no calibration purchase, and drops most of them where the panel is
sparsest, so each suite reports its own surviving `n_customers` in `data_summary`:

| churn | surviving @1,000 generated | @3,000 generated |
| ---: | ---: | ---: |
| 0.20 | 798 [454, 971] | 2,395 [1,403, 2,909] |
| 0.40 | 726 [363, 942] | 2,165 [1,144, 2,802] |
| 0.60 | 633 [286, 893] | 1,902 [874, 2,635] |
| 0.80 | 511 [177, 796] | 1,535 [553, 2,351] |

That is why the churn axis of a single-size grid cannot separate regime from panel size,
and why family S exists (`docs/insights-arm-sweep.md` §11).

Because the generator *is* a Pareto/NBD, that benchmark is correct by construction here
and is the **ceiling**, not a competitor.

Electronics is **not** Valendin et al.'s 3,782-customer Electronics Retailer cohort: 829
customers, and its `Transactions` counts line items rather than purchase occasions
(2.43 per active holdout week, against 1.02–1.05 on the other three panels).

---

## 3. Vocabulary: what an arm token means

The same token always means the same feature set. Every neural arm additionally carries
the target's own count as an embedded channel — that is the model contract, not a
covariate choice.

### AR encoding — how a customer's transaction history enters the input

| Token | Columns | Note |
|---|---|---|
| `no_ar` | — | The level baseline. The target's own past is still an input channel. |
| `ar_unbounded` | `period_since_last_transaction`, `cumulative_transactions`, `period_since_first_transaction` | The Pareto/NBD sufficient statistics (t_x, x, T). Two of the three keep counting through the holdout, past the range the weights were fitted on — the diagnosed failure. |
| `ar_bounded_K` | `active_in_last_{2,4,8,16,…,K}_periods` + `has_transacted_before` | **Nested** flags up to and including the deepest bin `K` — *not* a single flag at K. `ar_bounded_32` = five flags (2, 4, 8, 16, 32) plus `has_transacted_before`. All are 0 beyond the deepest bin, so no holdout value can leave the fitted range. `active_in_last_1_periods` is omitted because the target channel already carries the previous count. Definition: `scripts/run_ar_encoding_ablation.py:bounded_flags`. |
| `ar_log` | `log_period_since_last_transaction`, `cumulative_transactions`, `log_period_since_first_transaction` | The coordinate in which Pareto/NBD's log-survival is linear. Keeps the resolution the flags throw away, and still drifts past the calibration ceiling. |
| `ar_ratio` | `recency_over_tenure`, `transaction_rate`, `saturating_tenure_C_periods`, `has_transacted_before` | The bounded Pareto/NBD triple: the first two cannot leave their calibration range by construction. `C` is a quarter of the calibration window — 10 on CDNOW, 26 elsewhere. |
| `ar_saturating` | `saturating_recency_C_periods`, `transaction_rate`, `saturating_tenure_C_periods` | Both clocks saturated at `C` rather than ratio-normalised. Same `C` rule. |
| `ar_bounded32ratio` | `ar_bounded_32` + `ar_ratio` | Both sets together — the flags, which protected the level, and the ratio triple, which protected the ranking. |

`K` must stay below the panel's calibration length or the column degenerates into a
duplicate of `has_transacted_before` in calibration and diverges from it only in the
holdout — the exact failure the encoding exists to remove. `check_arm_depth` refuses a
flag at or above `T_CAL`. Hence `K` differs per family:

| Panel | Calibration periods | `K` used |
|---|---|---|
| CDNOW | 39 | 16 (≈41%, families E and H), 32 (≈82%, family E ablation and **family O by decision** — see §7) |
| electronics | 104 | 32 (≈31%), 52 (50%, ablation only) |
| gift, multichannel | 104 | 32 |
| synthetic `seasonal_4x4x10` | 104 | 32 |

**Where a run writes plain `ar_bounded`, resolve it here.** The synthetic grid's
`ar_bounded` is `ar_bounded_32`. `real_panel_arms`' is `ar_bounded_16` on CDNOW and
`ar_bounded_32` on electronics (`BOUNDED_DEPTH`, `run_real_panel_arms.py:182`). Family O's
`bounded32` is depth 32 on every panel, CDNOW included.

### Cluster

| Token | Column |
|---|---|
| `no_cluster` | — |
| `kmeans_4` / `kmeans_8` / `kmeans_16` | one frozen categorical `kmeans_<n>`, embedded, from k-means over `(t_x, x, T)` at the last calibration period |
| `ar_plus_cluster_8` | `ar_unbounded` + `kmeans_8` (cluster ablation only) |

### Calendar encoding

Named suffixes exist only where a run varied it; the base arm's encoding is whatever the
panel's `time_features` gives.

| Token | `time_features` | Channels added to `seq_cols` |
|---|---|---|
| `-no_tf` | `{}` | none |
| *(base, CDNOW archived)* | `{}` | none |
| *(base, electronics)* | `add_week_sin_cos`, `add_year_idx` | `week_sin`, `week_cos` |
| `-tf` | `add_week_sin_cos` | `week_sin`, `week_cos` |
| `-week_emb` | `{}`, `week` as an embedded categorical (`known_future`) | `week` |

`week_sin`/`week_cos` are non-embedded float channels; `week` under `-week_emb` is
embedded, as is the benchmark's own week input. That distinction decides which arms
`ValendinLSTM` can run at all (§5).

---

## 4. Global inventory

`platform`, read from each config's `studies_base_path`: **vast** = rented vast.ai GPU box
(`/root/panelclv`), **VM** = orchestrator QEMU VM (`/home/agent`, CPU), **local** =
workstation, **colab** = `/content/drive`.

Every neural suite in every family used the **`valendin` embedder** and **`cross_entropy`
loss**, except family I, whose whole point is the loss axis, and the `ValendinLSTM` and
`ParetoNBD` rows, which have no embedder parameter (§1). Families A, J, K, L and M predate
the `param_embedder` column, so their embedder is inferred rather than recorded.

| # | Family | Platform | Date | Panel(s) | Models | Arms | Trials | Studies | Paths | Suites | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | `seasonal_4x4x10` baseline | vast | 23–24 Aug | synthetic ×160 | LSTM | `no_ar-no_cluster-valendin` | **10** | 1/panel | 200 | 160 | complete, **superseded by B** |
| A | " | vast | 23–24 Aug | " | Transformer | same | **20** | 1/panel | 200 | 160 | complete, **superseded by B** |
| B | `seasonal_4x4x10` arm sweep | vast | 4–6 Sep | synthetic ×160 | LSTM, Transformer | 6: {`no_ar`, `ar_unbounded`, `ar_bounded_32`} × {`no_cluster`, `kmeans_8`} | 100 | 1/panel | 200 | **1,920** | complete |
| C | `seasonal_4x4x10` ceiling | VM | 5 Sep | synthetic ×160 | ParetoNBD | (none) | — | 1/panel | 200 | 160 | complete |
| D | `seasonal_4x4x10` benchmark | — | — | " | ValendinLSTM | — | — | — | — | 0 | **not run, see §5** |
| E | `ar_encoding` ablation | vast | 1–3 Sep, **extended 10 Sep** | CDNOW, electronics | LSTM | 7 AR encodings, see §4.1 | 50 | 20/shard, **2–5 shards** | 300 | 61 (60 with `results.csv`) | **ragged**, §4.1 |
| F | `cluster` ablation | vast | 3 Sep | CDNOW, electronics | LSTM | `no_cluster`, `kmeans_4/8/16`, `ar_unbounded`, `ar_plus_cluster_8` | 50 | 20 × 2 shards | 300 | 24 | complete |
| G | `real_panel_arms` @ 50 paths | vast | 6 Sep | CDNOW, electronics | LSTM, Transformer, ValendinLSTM, ParetoNBD | as H | 50 / 25¹ | 20 (shard a) | **50** | 34 | complete, **superseded by H**; moved to `Studies/_archive_50sim/` |
| H | `real_panel_arms` @ 200 paths | vast | 6–8 Sep | CDNOW, electronics | LSTM, Transformer, ValendinLSTM, ParetoNBD | 16 arms, see §4.2 | 50 / 25¹ | 20 (shard a) | 200 | 40 (39 with `results.csv`) | **37 ok, 1 short (19/20)** |
| I | `loss_ablation_cdnow` | local | 31 Aug | CDNOW | LSTM ×3 losses (`ce`, `emd`, `ce_emd`) | `ar_unbounded`-lite² | 20 | 10 | 300 | 1 (3 arms) | complete |
| J | `pnbd_study_4x4x10` | colab | 16 Jul | synthetic ×160 | LSTM, Transformer, ParetoNBD_MLE | `no_ar-no_cluster` | 20 / 50 | 1/panel | 100 | 160 | complete, superseded by A/B |
| K | `pnbd_study_4x4x10__ar` | colab | 2 Aug | synthetic ×160 | LSTM, ParetoNBD_MLE | ad-hoc AR set³ | 20 / 50 | 1/panel | 100 | 160 | complete, superseded by B |
| L | `pnbd_study_3x4x10` | local | 15 Jul | synthetic ×120 | LSTM | `no_ar-no_cluster` | 20 | 1/panel | 200 | 120 | complete, superseded |
| S | `seasonal_4x4x10_n3000` panel size | vast | 17–18 Sep | synthetic ×160, **3,000 customers** | LSTM | 2: {`no_ar`, `ar_bounded_32`} × {`no_cluster`} | 100 | 1/panel | 200 | 320 | complete |
| S | " | local | 17–18 Sep | " | ParetoNBD | (ignores arms; run under both trees) | — | 1/panel | 200 | 320 | complete |
| S | " | — | — | " | Transformer, ValendinLSTM | — | — | — | — | 0 | **not run**, §4.5 |
| M | `cross_entropy_*` suites (9 dirs) | local | Jun–Aug | electronics | LSTM, Transformer, ParetoNBD / ParetoNBD_MLE | `no_ar-no_cluster` | 1–50⁴ | 1–10⁴ | 200 | 9 | historical, budgets not comparable |
| **N** | **`real_panel_benchmarks`** | vast + local | **13 Sep** | **all four** | ValendinLSTM, ParetoNBD | count + embedded week only | **100** | **20** | **500** | **84** | **complete** |
| **O** | **`real_panel_lstm_*` (2y)** | vast | **13–14 Sep** | **all four** | LSTM | 4: `bounded32`, `log`, `ratio`, `bounded32ratio` | **100** | **100** | **500** | **1,600** | **complete** |
| **P** | **`real_panel_lstm_*_cal3y`** | vast | **14–15 Sep** | electronics, gift, multichannel | LSTM | the same 4 | **100** | **100** | **500** | **1,200** | **complete** |
| **Q** | **`real_panel_benchmarks_cal3y`** | local | **16 Sep** | electronics, gift, multichannel | ParetoNBD | (none) | — | 1 | 500 | **3** | **complete; ValendinLSTM 0/20, §6** |
| **R** | **`run_rescore_trials`** | vast + local | **17 Sep** | all four | ValendinLSTM | as N | — | **698 refits** | 500 | 0 (rescores N) | **complete**, written to `Rescored/` |
| **T** | **`training_budget`** | vast | **20 Sep** | electronics | ValendinLSTM, LSTM | 4 recipes: `archive`, `paper`, `paper90`, `floor50` (§4.6) | 100 / **1**⁵ | **20** | **200** | **160** | **complete** |
| **U** | **`factorial`** | vast | **21 Sep** | **all four** | ValendinLSTM, LSTM | 2×2: {`archive`, `floored`} × {`no_cluster`, `kmeans_8`} (§4.7) | 100 / **1**⁵ | **20** | **200** | **640** | **complete** |
| **V** | **`selection_rescore`** | vast | **21 Sep** | electronics, cdnow | ValendinLSTM, LSTM | `archive` only; scores every trial twice (§4.8) | 100 | 40 / 10 | 100 | **90** | **complete** |

¹ `ValendinLSTM` gets 25 trials in families G and H: ADR-0004 freezes its architecture, so
its search space holds only `learning_rate` / `weight_decay` / `batch_size`. Family N gives
it the full 100.
² `period_since_last_transaction` + `has_transacted_before` — a two-column set matching no
token in §3; it predates the vocabulary.
³ `period_since_last_transaction`, `active_in_last_3_periods`,
`period_since_first_transaction`, `transaction_rate` — likewise pre-vocabulary, and
`transaction_rate` is not an encoding any current arm uses.
⁴ Several of the nine are evidently probes (1 trial, 1 study). Treat them as exploratory,
not as replications of each other.
⁵ Family T's `paper` arm pins every hyperparameter, so it runs a single trial by design —
its `results.csv` carries no `param_*` spread, and that is the point (§4.6).

Families B, E, F, H, N, O and P are the ones results are read off today. Where they are
reported: family B in `docs/insights-arm-sweep.md`, family F in
`docs/insights-cluster-ablation.md`, families N, O, P and R in
`docs/benchmarks-real-panels.md`, family H in `docs/insights-study.md` §8 (at family G's
50-path budget) and `docs/benchmarks-real-panels.md` (at H's 200).

### 4.1 Family E is ragged — three arms were never finished at the new depth

The ablation was re-run on 10 September with three new encodings and three extra shards.
The older arms were not brought up with them, so the arms are **not budget-matched to each
other**. Replications per arm, counted from disk:

| panel | arm | shards | replications | trials × paths |
|---|---|---|---:|---|
| electronics | `no_ar` | a b c d e | **100** | 50 × 300 |
| electronics | `ar_bounded_32` | a b c d e | **100** | 50 × 300 |
| electronics | `ar_bounded_52` | a b c d e | **100** | 50 × 300 |
| electronics | `ar_log` | a b c d e | **100** | 50 × 300 |
| electronics | `ar_ratio` | a b c d e | **100** | 50 × 300 |
| electronics | `ar_saturating` | a b c d e | **100** | 50 × 300 |
| electronics | `ar_unbounded` | a b | **40** | 50 × 300 |
| cdnow | `no_ar` | a b c d e | **100** | 50 × 300 |
| cdnow | `ar_bounded_16` | a b c d e | **100** | 50 × 300 |
| cdnow | `ar_log` | a b c d e | **100** | 50 × 300 |
| cdnow | `ar_saturating` | a b c d e | **100** | 50 × 300 |
| cdnow | `ar_ratio` | a c d e | **80** | 50 × 300 |
| cdnow | `ar_bounded_32` | a b | **40** | 50 × 300 |
| cdnow | `ar_unbounded` | a b | **40** | 50 × 300 |

Shard `b` of CDNOW `ar_ratio` is the one hole inside an otherwise complete arm, and it is
a **partial run rather than an absent one**: `Studies/ar_encoding__cdnow__ar_ratio__b/`
holds 8 of its 20 forecasts and no `results.csv`, so those 8 are excluded from every
number above and are not recoverable without re-running the shard. The two
40-replication arms are deliberate: `ar_unbounded` is the diagnosed-broken encoding and
re-measuring it more finely buys nothing.

**`ar_saturating` has never been reported anywhere**, and on electronics it is the best arm
in the family on both level and MAPE (§6).

### 4.2 Family H — `real_panel_arms` @ 200 paths

`scripts/run_real_panel_arms.py --check-complete`, re-run 17 September: **38 schedulable
neural suites + 2 Pareto; 37 ok, 1 SHORT, 0 ABSENT, 10 n/a.**

The declared axis is `{no_ar, ar_bounded} × {no_cluster, kmeans_8}` per panel —
`ar_unbounded` was **deliberately dropped** at this budget: it is the diagnosed-broken
encoding (+198% / +461% bias in family G) and re-measuring it at 200 paths would
re-establish a known result at four times the rollout cost. Plus per-panel calendar
variants: CDNOW `-tf` (all four cells) and `-week_emb` (its two `no_ar` cells);
electronics `-no_tf` (its two `no_ar` cells). 10 arms on CDNOW, 6 on electronics, all at
20 studies × 50 trials × 200 paths (25 trials for `ValendinLSTM`), `valendin` embedder,
archived CDNOW window (§2).

The one gap is a single missing replication of `Transformer / cdnow /
no_ar-no_cluster-valendin` — 19 of 20 studies. The suite already reports a 19-study
distribution, so finishing it is optional.

### 4.3 Families N, O, P, Q — the four-panel runs

All four share the benchmark's windows, cohort and scoring code, imported rather than
restated, so a row from one reads directly against a row from another.

| family | model | inputs | embedder | replications | trials | paths | seeds | panels | windows |
|---|---|---|---|---:|---:|---:|---|---|---|
| N | ValendinLSTM | count + week, both embedded; every other column discarded | frozen (ADR-0004) | 20 | 100 | 500 | 43–62 | all four | 2y (§2) |
| N | ParetoNBD | `(t_x, x, T)` from active weeks | — | 1 (deterministic) | — | — | 42 | all four | 2y |
| O | LSTM | count (embedded), `week_sin`/`week_cos`, one AR encoding | `valendin` | 100 | 100 | 500 | 43–142 | all four | 2y |
| P | LSTM | as O | `valendin` | 100 | 100 | 500 | 43–142 | 3 panels | 3y (§2) |
| Q | ParetoNBD | as N | — | 1 | — | — | 42 | 3 panels | 3y |

O and P cross four AR encodings — `bounded32`, `log`, `ratio`, `bounded32ratio` — with no
cluster column and no calendar variant. That is 4 encodings × 4 panels × 100 replications
= 1,600 suites in O, and 4 × 3 × 100 = 1,200 in P.

**O and N are budget-matched** (100 trials, 500 paths) — unlike family H against family G,
where the benchmark had 25 trials and the developed models 50. Family Q exists so the
3-year comparison has a benchmark; it was run on 16 September, after
`docs/benchmarks-real-panels.md` was written, and its numbers are in §6 below.

### 4.4 Family R — rescoring non-winning trials

Not a new suite family: `scripts/run_rescore_trials.py` took the non-winning trial
checkpoints that survived `keep_only_best_checkpoint` in 30 of family N's 80 studies, put
each through the ADR-0008 refit and the registry rollout at the study's own forecast seed,
and scored it with `compute_forecast_metrics`. **698 refits, 17 September**, written to
`Rescored/` and `Rescored_local/`. Only the refit's training RNG differs from the archived
run. Reported in `docs/benchmarks-real-panels.md`, "Does the search select the best
trial?".

---

### 4.5 Family S — the panel-size grid, and what it does not carry

`grids/seasonal_4x4x10_n3000.py` declares four models and two arms; this run trained two
of the four. `scripts/reconcile_grid.py --grid seasonal_4x4x10_n3000` reports **160/160
for both LSTM arm trees and both ParetoNBD trees**, and `ABSENT` — *not run*, as distinct
from short — for the Transformer and `ValendinLSTM` on both arms.

- **Transformer: deliberately deferred, and now priced.** Its rollout cost at 3,000
  customers was measured on 17 September at **~6.7 s per Monte Carlo path against ~1.0 s
  at 1,000**, the Optuna search merely doubling. At the declared 200 paths that is ~22 min
  per suite and ~150–215 GPU-hours for the same two arms, about $17–24. It is left
  declared with zero workers precisely so this table keeps reporting it as owed.
- **`ValendinLSTM`: cannot run at all**, F11, same as every other grid carrying
  `add_week_sin_cos` — see §5.
- **Pareto/NBD reads the panel, not the engineered features**, so its two arm trees are
  identical by construction; that they came out identical is the check, not a duplicate.

Read in `docs/insights-arm-sweep.md` §11.

### 4.6 Family T — the training recipe, not the architecture or the inputs

Declared 20 September, specified in `.scratch/training-budget/spec.md` and motivated by
`docs/training-budget.md`: every archived neural study stopped while its validation loss
was still falling, because our training recipe is not the reference notebook's. The
notebook trains at batch 32 with plain Adam for ~90 epochs — about 2,300 gradient updates
on electronics — while a family N winner receives about 32.

Three recipes, on electronics only, crossed with two models:

| arm | recipe | trials |
|---|---|---:|
| `archive` | lr / weight decay / batch searched, `patience=7`, `n_epochs=100` | 100 |
| `paper` | pinned: `lr=1e-3`, `weight_decay=0.0`, `batch_size=32`, `patience=5`, `n_epochs=150` | 1 |
| `paper90` | the same, plus `min_epochs=90` — the notebook's own epoch count | 1 |
| `floor50` | `archive`'s search plus `min_epochs=50`, `n_epochs=300` | 100 |

`archive` gets 100 trials because family N gave this panel 100: a control is only a
control if it searches what the archive searched. `paper90` exists because the recipe
alone does not reproduce the training — measured before launch, `paper` stops at epoch 1
on electronics and scores *worse* than `archive`
(`.scratch/training-budget/issues/01-paper-recipe-arm.md`).

Three things a reader has to know before comparing a family T row with anything:

- **It changes the training recipe only.** The architectures are the ones every other
  family uses, and the inputs are family N's for `ValendinLSTM` and family H's
  `no_ar-no_cluster-valendin` for `LSTM`.
- **Its `ValendinLSTM` suites sit beside family N, not in place of it.** No published row
  moves; whether any is regenerated is decided after this reports
  (`.scratch/training-budget/issues/06-report-and-decide.md`).
- **The `paper` arm pins everything and runs one trial**, so its `param_*` columns are
  constant by design. That is what makes it a control: it removes hyperparameter
  selection, which `docs/benchmarks-real-panels.md` shows does not predict the forecast.

Its two models read different panels — `ValendinLSTM` cannot take `week_sin`/`week_cos`
(F11, §5) — so the runner builds each model's dataset through the runner that already
declares it.

### 4.7 Family U — the 2×2 that crosses training with inputs

Families T and F each moved one lever and neither knew about the other: T the training
recipe, F the cluster label. Family U crosses them on all four panels, so the question
"do they add?" has an answer rather than two separate before-and-afters.
`scripts/run_factorial.py`, reported in `docs/training-budget.md` §15.

**Crossing them adds little.** In seven of eight (panel, model) cells the increment from
adding the floor on top of the label has a 95% interval spanning zero, and in all eight
that interval rules out a gain larger than about +0.035 Spearman — the order of what an
unseeded refit moves on its own. The label reaches most of what is reachable by itself and
the floor reaches a little over half of it; stacking them buys a few hundredths at most.

Two things a reader must carry when quoting a family U row:

- **Its `floored` arm runs ONE pinned trial**, not a search, so its `param_*` columns are
  constant by design. Family T §4.6 shows the search adds nothing once a model is floored.
- **Its LSTM carries no year index on any panel**, unlike families H, O and T, which
  inherited one on electronics. `run_real_panel_arms.py` argues against a year index for
  CDNOW — constant in calibration, out of range across the holdout — and family U applies
  that reasoning everywhere for internal consistency. **An electronics LSTM cell here is
  therefore not the same configuration as family T's**; read family U's own controls.

### 4.8 Family V — scoring every trial, not just the winner

`scripts/run_selection_rescore.py`. Trains its own studies with
`keep_only_best_checkpoint=False`, then scores 40 randomly chosen trials of each twice: a
leak-free Monte Carlo rollout over the **validation** window (the candidate selection
criterion) and the production path — ADR-0008 refit plus a holdout rollout — for the
target. The output is a per-trial table inside each suite, `selection_rescore.csv`, not a
`results.csv`; `--check-complete` reads those.

It exists because the archive cannot answer the question: every family T and U suite kept
only its winner's checkpoint. 40 studies per model on electronics, 5 per model on CDNOW.
Reported in `docs/training-budget.md` §14 and §15.3.

---

## 5. Where `ValendinLSTM` can and cannot run

`benchmarks/valendin_lstm.py:99` refuses any `seq_col` that is not embedded — the published
model reads week and transaction count, both categorical, and has no covariate path
(ADR-0004). This is not a bug to route around; it is what makes the benchmark a
reproduction.

Consequences, all **verified by running the code**:

- **On the four real panels at the benchmark windows it runs everywhere** (family N), because
  that runner hands it `week` as an *embedded* categorical rather than `week_sin`/`week_cos`.
  20 replications × 100 trials × 500 paths on each of the four.
- **On the real-panel arms it is eligible on 6 of 16** (family H) — the `no_ar` cells with no
  engineered calendar: CDNOW base (×2 cluster) and `-week_emb` (×2 cluster), electronics
  `-no_tf` (×2 cluster). All 6 are complete at 20/20.
- **On the synthetic grid it has not been run.** Every one of the six arms carries
  `week_sin`/`week_cos` as non-embedded floats, because `grids/seasonal_4x4x10.py`'s
  `PANEL.time_features` sets `add_week_sin_cos`; `preflight_grid_arms.py --stage train
  --model valendin_lstm` raises on the first arm. **This is a config gap, not a missing
  capability** — family N shows the embedded-week treatment works — so putting the benchmark
  on the grid means declaring a `week`-embedded (or no-calendar) grid arm, which nobody has
  declared yet. Until then the synthetic grid compares two developed models against
  Pareto/NBD with no published-model reference at all.
- **It carries no AR channel on any panel**, so every bounded-AR row in families E, H, O and
  P has no benchmark counterpart at the same arm. Comparisons against it hold the *panel*
  fixed, never the arm.

---

## 6. The gap list

Ordered by what a result depends on.

1. **The `projected` embedder — never run anywhere.** `scripts/run_cdnow_embedding_ablation.py`
   declares it fully (arms `LSTM_valendin`, `LSTM_projected_32`, `LSTM_projected_64`,
   `LSTM_projected_128`; 15 trials, 3 studies, 300 paths) and has never been executed —
   there is no `Studies/cdnow_embedding_ablation`. On the synthetic grid the embedder axis
   is declared but not crossed (`EMBEDDER_AXIS = ("valendin",)`), cut on cost: adding
   `"projected"` to that tuple is the only edit needed, and it doubles the bill.
   **Every number in this repo was produced by one embedder, and any claim about embedding
   strategy rests on zero measurements.** Still the largest unmeasured axis in the project.
2. **`ValendinLSTM` on the three-year windows — 0 of 20 replications.** Family Q fitted
   Pareto/NBD on them (16 Sep) but the frozen LSTM was never run, so
   `scripts/run_real_panel_benchmarks.py --calibration 3y --report` prints `nan` for every
   Valendin cell. Family P's 1,200 LSTM suites therefore have one benchmark, not two.
3. **`ValendinLSTM` on the synthetic grid** — §5. Needs a declared `week`-embedded grid arm,
   or an explicit statement in the thesis that the frozen benchmark is a real-panel-only
   comparator.
4. **`ar_saturating` is unreported.** Family E's best electronics arm by MAPE and among the
   best on level — bias −11.6 ± 18.0, MAPE 43.3 ± 3.6, Spearman 0.299 ± 0.019 over 100
   replications — appears in no document. It was also never carried onto gift and
   multichannel, where family O ran only four of the seven encodings.
5. **Family E is not budget-matched across its own arms** (§4.1): three arms at 40–80
   replications against eleven at 100. The missing shard `b` of CDNOW `ar_ratio` is cheap
   to finish; the two `ar_unbounded` arms are deliberate.
6. **One replication short** — `Transformer / cdnow / no_ar-no_cluster-valendin` at 19/20
   (family H). Cheap to finish, and defensible to leave.
7. **The Transformer on the four-panel runs.** `scripts/run_real_panel_ar.py --model
   transformer` exists and has never been run on any panel, so families O and P are
   LSTM-only. The synthetic grid found the two architectures fail in different corners
   (`docs/insights-arm-sweep.md` §5), so the four-panel picture is one architecture's.
8. **Trial-budget confound in family A.** The archived baseline gave the LSTM 10 trials and
   the Transformer 20. Family B re-ran the same arm for both at 100, so use B for any
   LSTM-vs-Transformer statement and treat A as superseded.
9. **Extra shards** anywhere a comparison turns out to be within noise. Nothing beyond
   shard `a` exists in families B, G or H.

---

## 7. Known inconsistencies

**The pre-ADR-0009 CDNOW window breaks two report paths.** `run_ar_encoding_ablation.py
--panel cdnow --report` and `run_cluster_ablation.py --panel cdnow --report` both raise
`ValueError: operands could not be broadcast together with shapes (2357,38) (2357,39)`:
they rebuild the panel with today's week rule, which gives 39 holdout weeks, and score it
against archived predictions that have 38. Consequences:

- The **bias, MAPE and RMSE** of CDNOW's families E and F are still readable — they are in
  each suite's `results.csv`, written when the run happened, on that run's own window.
- Their **Spearman is not recoverable** without rebuilding the 38-week panel, so no CDNOW
  discrimination number exists for either ablation. Electronics is unaffected and both
  reports run there.
- A CDNOW row from families E, F, G, H or I **must not be placed in the same table** as one
  from families N or O without saying which window it is on.

**`Studies/ar_encoding__electronics__ar_bounded_52__*` orders its columns**
`…active_in_last_32_periods, has_transacted_before, active_in_last_52_periods`, whereas
`bounded_flags(52)` today emits `…active_in_last_32_periods, active_in_last_52_periods,
has_transacted_before`. The **set** is identical and each flag is its own input channel, so
the models saw the same information; only the `seq_cols` order differs. Re-running that arm
will not reproduce the archived column order.

**`grids/seasonal_4x4x10_ar.py` declares a grid that was never generated** —
`Datasets/Synthetic/seasonal_4x4x10_ar` does not exist. It is **superseded**: its feature
set is exactly family B's `ar_unbounded` arm, measured at 100 trials on the same panels.
