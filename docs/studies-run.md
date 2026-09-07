# Every study run, and what is still owed

A global inventory of every study suite on disk under `Studies/`, what it varied, and
what budget it was run at — so the gaps are visible without re-deriving them from 6,389
`config.json` files.

Built by reading the archived `config.json` of every suite (they record the panel
config, the model specs and the budget verbatim) and cross-checking the two scripts that
own the "what is owed" question: `scripts/run_real_panel_arms.py --check-complete` and
`scripts/reconcile_grid.py --grid <name>`. Those two commands are the authority; this
document is a snapshot of them taken **7 September 2026, 21:10 CEST**, plus the families
they do not cover. The `real_panel_arms` fleet was still draining while this was written,
so re-run those two commands before trusting §4.

---

## 1. Vocabulary

Every arm name in this document decomposes as `<ar>-<cluster>-<embedder>[-<calendar>]`.
The same token always means the same feature set. Where a run used a different label for
an identical set, the equivalence is stated.

### AR encoding — how a customer's transaction history enters the input

| Token | Columns | Note |
|---|---|---|
| `no_ar` | — | The level baseline. The target's own past is still an input channel. |
| `ar_unbounded` | `period_since_last_transaction`, `cumulative_transactions`, `period_since_first_transaction` | The Pareto/NBD sufficient statistics (t_x, x, T). Two of the three keep counting through the holdout, past the range the weights were fitted on — the diagnosed failure. |
| `ar_bounded_K` | `active_in_last_{2,4,8,16,…,K}_periods` + `has_transacted_before` | **Nested** flags up to and including the deepest bin `K` — *not* a single flag at K. `ar_bounded_32` = five flags (2, 4, 8, 16, 32) plus `has_transacted_before`. All are 0 beyond the deepest bin, so no holdout value can leave the fitted range. `active_in_last_1_periods` is omitted because the target channel already carries the previous count. Definition: `scripts/run_ar_encoding_ablation.py:bounded_flags`. |

`K` must stay below the panel's calibration length or the column degenerates into a
duplicate of `has_transacted_before` in calibration and diverges from it only in the
holdout — the exact failure the encoding exists to remove. Hence `K` differs per panel:

| Panel | Calibration periods | `K` used |
|---|---|---|
| CDNOW | 39 | 16 (≈41%), 32 (≈82%, ablation only) |
| electronics | 104 | 32 (≈31%), 52 (50%) |
| synthetic `seasonal_4x4x10` | 104 | 32 |

**Where a run writes plain `ar_bounded`, resolve it with this table.** The synthetic grid's
`ar_bounded` is `ar_bounded_32`. `real_panel_arms`' `ar_bounded` is `ar_bounded_16` on
CDNOW and `ar_bounded_32` on electronics (`BOUNDED_DEPTH`, `run_real_panel_arms.py:182`).

### Cluster

| Token | Column |
|---|---|
| `no_cluster` | — |
| `kmeans_4` / `kmeans_8` / `kmeans_16` | one frozen categorical `kmeans_<n>`, embedded |
| `ar_plus_cluster_8` | `ar_unbounded` + `kmeans_8` (cluster ablation only) |

### Embedder

Two strategies exist in `registry/model_registry.py:_EMBEDDERS`.

| Token | Meaning |
|---|---|
| `valendin` | Each feature keeps its own sqrt(cardinality)+1 vector. **The registry default, and the only embedder any archived study has ever used.** |
| `projected` | All features projected to one common width (`embedding_dim` for the LSTM, `d_model` for the Transformer). **Never run.** |

The `-valendin` suffix on every arm name is therefore a constant, not a variable. It is
kept in the name so the synthetic and real trees sort against each other.

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
embedded. That distinction decides which arms `ValendinLSTM` can run at all (§5).

### Budget

| Term | Meaning |
|---|---|
| **trials** | Optuna trials per study. |
| **studies** | `n_studies_per_model` — independent replications, each its own Optuna study + full-calibration refit + forecast, seeded `base_seed + i`. The spread across these is the reported distribution. |
| **sims** | `n_simulations` — Monte Carlo rollout paths averaged into one forecast. |
| **shard `a`/`b`** | Disjoint 20-seed blocks (`a` = seeds 43–62, `b` = 63–82). Running both gives 40 replications of one arm. |

---

## 2. Global inventory

`platform`: **vast** = rented vast.ai GPU box (`/root/panelclv`), **VM** = orchestrator
QEMU VM (`/home/agent`, CPU), **local** = workstation, **colab** = `/content/drive`.

| # | Family | Platform | Date | Panel(s) | Models | Arms | Trials | Studies | Sims | Suites | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | `seasonal_4x4x10` baseline | vast | 23–24 Aug | synthetic, 160 panels | LSTM | `no_ar-no_cluster-valendin` | **10** | 1/cell | 200 | 160 | complete |
| A | " | vast | 23–24 Aug | " | Transformer | same | **20** | 1/cell | 200 | 160 | complete |
| B | `seasonal_4x4x10` arm sweep | vast | 4–6 Sep | synthetic, 160 panels | LSTM, Transformer | 6: {`no_ar`, `ar_unbounded`, `ar_bounded_32`} × {`no_cluster`, `kmeans_8`} | 100 | 1/cell | 200 | **1,920** | complete |
| C | `seasonal_4x4x10` ceiling | VM | 5 Sep | synthetic, 160 panels | ParetoNBD | (none) | — | 1/cell | 200 | 160 | complete |
| D | `seasonal_4x4x10` benchmark | — | — | " | ValendinLSTM | — | — | — | — | 0 | **blocked, see §5** |
| E | `ar_encoding` ablation | vast | 1–3 Sep | CDNOW, electronics | LSTM | CDNOW: `no_ar`, `ar_unbounded`, `ar_bounded_16`, `ar_bounded_32`; electronics: `no_ar`, `ar_unbounded`, `ar_bounded_32`, `ar_bounded_52` | 50 | 20 × 2 shards | 300 | 16 | complete |
| F | `cluster` ablation | vast | 3 Sep | CDNOW, electronics | LSTM | `no_cluster`, `kmeans_4`, `kmeans_8`, `kmeans_16`, `ar_unbounded`, `ar_plus_cluster_8` | 50 | 20 × 2 shards | 300 | 24 | complete |
| G | `real_panel_arms` @ 50 paths | vast | Aug | CDNOW, electronics | LSTM, Transformer, ValendinLSTM | 3 AR × 2 cluster (+ 2 `-no_tf` on electronics) | 50 / 25¹ | 20 (shard a) | **50** | 32 + 2 Pareto | complete, **superseded by H** |
| H | `real_panel_arms` @ 200 paths | vast | 6–7 Sep | CDNOW, electronics | LSTM, Transformer, ValendinLSTM | see §4 | 50 / 25¹ | 20 (shard a) | 200 | 38 + 2 Pareto | **37 ok, 1 short (19/20)** |
| I | `loss_ablation_cdnow` | local | 31 Aug | CDNOW | LSTM ×3 losses (`ce`, `emd`, `ce_emd`) | `ar_unbounded`-lite² | 20 | 10 | 300 | 1 (3 arms) | complete |
| J | `pnbd_study_4x4x10` | colab | 16 Jul | synthetic, 160 panels | LSTM, Transformer, ParetoNBD_MLE | `no_ar-no_cluster` | 20 | 1/cell | 100 | 160 | complete, superseded by A/B |
| K | `pnbd_study_4x4x10__ar` | colab | 2 Aug | synthetic, 160 panels | LSTM, ParetoNBD_MLE | ad-hoc AR set³ | 20 | 1/cell | 100 | 160 | complete, superseded by B |
| L | `pnbd_study_3x4x10` | local | 15 Jul | synthetic, 120 panels | LSTM | `no_ar-no_cluster` | 20 | 1/cell | 200 | 120 | complete, superseded |
| M | `cross_entropy_*` suites (9 dirs) | local | Jun–Aug | electronics | LSTM, Transformer, ParetoNBD / ParetoNBD_MLE | `no_ar-no_cluster` | 1–100⁴ | 1–10⁴ | 1–600⁴ | 9 | historical, budgets not comparable |

¹ `ValendinLSTM` gets 25 trials: ADR-0004 freezes its architecture, so its search space
holds only `learning_rate` / `weight_decay` / `batch_size`.
² `period_since_last_transaction` + `has_transacted_before` — a two-column set matching
no token above; it predates the vocabulary.
³ `period_since_last_transaction`, `active_in_last_3_periods`,
`period_since_first_transaction`, `transaction_rate` — likewise pre-vocabulary, and
`transaction_rate` is not an encoding any current arm uses.
⁴ Every one of the nine ran a different budget, several of them evidently probes (1 trial,
1 simulation). Treat them as exploratory, not as replications of each other.

**Every neural suite in every family above used `cross_entropy` loss and the `valendin`
embedder**, except family I, whose whole point is the loss axis.

### Panel facts

| Panel | Customers | Calibration | Holdout | `clip_target_upper` (softmax classes) |
|---|---|---|---|---|
| synthetic `seasonal_4x4x10` | 1,000/panel × 160 | 104 wk | 52 wk | 6 (7 classes) |
| CDNOW | 2,357 | 39 wk | 38 wk | 4 (5 classes) |
| electronics | 829 | 104 wk | 52 wk | 6 (7 classes) |

The electronics file is **not** Valendin et al.'s 3,782-customer Electronics Retailer
cohort.

---

## 3. Synthetic grid — reconciled

`scripts/reconcile_grid.py --grid seasonal_4x4x10`, 7 Sep:

```
  ok      LSTM        / all 6 arms                160/160 each
  ok      Transformer / all 6 arms                160/160 each
  ok      ParetoNBD   / (no arm)                  160/160
  ABSENT  ValendinLSTM / (no arm)                 not run
```

`grids/seasonal_4x4x10_ar.py` declares a second grid (the Pareto/NBD statistics as AR
features) whose datasets were never generated — `Datasets/Synthetic/seasonal_4x4x10_ar`
does not exist. It is **superseded**: its feature set is exactly family B's
`ar_unbounded` arm, measured at 100 trials on the same panels.

---

## 4. `real_panel_arms` @ 200 paths — essentially complete

`scripts/run_real_panel_arms.py --check-complete`, 7 Sep 21:10 CEST (three consecutive
runs agreeing): **38 schedulable neural suites + 2 Pareto; 37 ok, 1 SHORT, 0 ABSENT,
10 n/a.**

The declared axis is `{no_ar, ar_bounded} × {no_cluster, kmeans_8}` per panel —
`ar_unbounded` was **deliberately dropped** at this budget: it is the diagnosed-broken
encoding (+198%/+461% bias in the 50-path archive) and re-measuring it at 200 paths would
re-establish a known result at four times the rollout cost. Plus per-panel calendar
variants: CDNOW `-tf` (all four cells) and `-week_emb` (its two `no_ar` cells);
electronics `-no_tf` (its two `no_ar` cells). 10 arms on CDNOW, 6 on electronics.

Complete at 20/20 studies:

| Arm | Panel | LSTM | Transformer | ValendinLSTM |
|---|---|---|---|---|
| `ar_bounded-kmeans_8-valendin` | electronics | ok | ok | n/a |
| `ar_bounded-kmeans_8-valendin` | CDNOW | ok | ok | n/a |
| `ar_bounded-kmeans_8-valendin-tf` | CDNOW | ok | ok | n/a |
| `ar_bounded-no_cluster-valendin` | electronics | ok | ok | n/a |
| `ar_bounded-no_cluster-valendin` | CDNOW | ok | ok | n/a |
| `ar_bounded-no_cluster-valendin-tf` | CDNOW | ok | ok | n/a |
| `no_ar-kmeans_8-valendin` | electronics | ok | ok | n/a |
| `no_ar-kmeans_8-valendin` | CDNOW | ok | ok | ok |
| `no_ar-kmeans_8-valendin-no_tf` | electronics | ok | ok | ok |
| `no_ar-kmeans_8-valendin-tf` | CDNOW | ok | ok | n/a |
| `no_ar-kmeans_8-valendin-week_emb` | CDNOW | ok | ok | ok |
| `no_ar-no_cluster-valendin` | electronics | ok | ok | n/a |
| `no_ar-no_cluster-valendin` | CDNOW | ok | **19/20** | ok |
| `no_ar-no_cluster-valendin-no_tf` | electronics | ok | ok | ok |
| `no_ar-no_cluster-valendin-tf` | CDNOW | ok | ok | n/a |
| `no_ar-no_cluster-valendin-week_emb` | CDNOW | ok | ok | ok |
| *(no arm)* | both | — | — | ParetoNBD 1/1 ok, both panels |

The one gap is a single missing replication of `Transformer / cdnow /
no_ar-no_cluster-valendin` — 19 of 20 studies. Whether that is worth a box is a judgement
call: the suite already reports a 19-study distribution.

Shard `b` (20 more replications per arm) has not been run for any arm in any family
except E and F, and is only worth spending where a confidence interval overlaps a
neighbour's.

### Fleet postscript

The run drained cleanly: six workers finished `exit=0`, were verified suite-by-suite
against local disk, and were destroyed by `reap_finished.sh` between 09:32 and 13:52.
One box (`50097932`, RTX 3060, $0.0589/hr) outlived them **without doing work**: its
trainer died at 00:30 UTC leaving no `.shard_exit` and no traceback, and the reaper's
state probe returns `none` for that case — which is a silent `continue`, deliberately
left for a human. Nothing it held was unique (its one `results.csv` and all 20 electronics
predictions were already local), so it is safe to destroy. **A worker that dies without
writing `.shard_exit` bills indefinitely and logs nothing** — worth a heartbeat check on
`shard.log` mtime in a future reaper.

## 5. Where `ValendinLSTM` can and cannot run

`benchmarks/valendin_lstm.py:99` refuses any `seq_col` that is not embedded — the
published model reads week and transaction count, both categorical, and has no covariate
path (ADR-0004). This is not a bug to route around; it is what makes the benchmark a
reproduction.

Consequences, both **verified by running the code**:

- **On the synthetic grid it cannot run at all as declared.** Every one of the six arms
  carries `week_sin`/`week_cos` as non-embedded floats, because `PANEL.time_features`
  sets `add_week_sin_cos`. `preflight_grid_arms.py --stage train --model valendin_lstm`
  raises on the first arm. Putting the benchmark on this grid needs a no-calendar or
  `week`-embedded panel variant, which does not exist.
- **On the real panels it is eligible on 6 of 16 arms** — the `no_ar` cells with no
  engineered calendar: CDNOW base (×2 cluster) and `-week_emb` (×2 cluster), electronics
  `-no_tf` (×2 cluster). **All 6 are complete at 20/20**, so the frozen benchmark is in
  the real-panel comparison at full budget, including both `-week_emb` cells — the only
  arms that give it calendar information at all.

---

## 6. The gap list

Ordered by what a result depends on.

1. **The `projected` embedder — never run anywhere.** `scripts/run_cdnow_embedding_ablation.py`
   declares it fully (arms `LSTM_valendin`, `LSTM_projected_32`, `LSTM_projected_64`,
   `LSTM_projected_128`; 15 trials, 3 studies, 300 sims) and has never been executed —
   there is no `Studies/cdnow_embedding_ablation`. On the synthetic grid the embedder axis
   is declared but not crossed (`EMBEDDER_AXIS = ("valendin",)`), cut on cost: adding
   `"projected"` to that tuple is the only edit needed, and it doubles the bill.
   **Any claim about embedding strategy currently rests on zero measurements.** This is now
   the largest unmeasured axis in the project.
2. **`ValendinLSTM` on the synthetic grid** — blocked by design (§5). Needs a decision:
   declare a no-calendar (or `week`-embedded) arm, or state in the thesis that the frozen
   benchmark is a real-panel-only comparator. Right now the synthetic grid compares our two
   models against Pareto/NBD with no published-model reference at all.
3. **One replication short** — `Transformer / cdnow / no_ar-no_cluster-valendin` at 19/20.
   Cheap to finish, and defensible to leave.
4. **Trial-budget confound in family A.** The archived baseline gave the LSTM 10 trials
   and the Transformer 20. Family B re-ran the same arm for both at 100, so use B for any
   LSTM-vs-Transformer statement and treat A as superseded rather than as a baseline.
5. **Shard `b`** anywhere a comparison turns out to be within noise.

## 7. Known inconsistency

`Studies/ar_encoding__electronics__ar_bounded_52__*` orders its columns
`…active_in_last_32_periods, has_transacted_before, active_in_last_52_periods`, whereas
`bounded_flags(52)` today emits `…active_in_last_32_periods, active_in_last_52_periods,
has_transacted_before`. The **set** is identical and each flag is its own input channel,
so the models saw the same information; only the `seq_cols` order differs. Re-running that
arm will not reproduce the archived column order.
