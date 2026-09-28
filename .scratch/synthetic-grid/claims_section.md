## Claims and whether the evidence supports them

Every claim below is tested per rate × churn cell, per rate with churn pooled, over the
whole grid and, where the data allow, at both cohort sizes. Most verdicts depend on those
conditions, so each claim states where it holds and where it does not. All numbers are
recomputed from the stored forecasts by `.scratch/synthetic-grid/metrics_by_cell.py` and
`claims.py`.

**Setting.** Seasonal panels (4 peaks, amplitude 1.5), 1,000 customers unless a table says
3,000. Four purchase rates × four churn rates (20, 40, 60 and 80% of customers dropped out
by week 52), 10 panels per cell, 40 per rate, 160 over the grid.

**How to read the tests.**

- **Effect Δ.** Every comparison is written "A → B" and Δ is B minus A. For MAPE, |bias| and
  RMSE a negative Δ means B is better; for Spearman, R_A and shape correlation a positive Δ
  does. MAPE and |bias| are in percentage points.
- **Paired comparisons** (two trees on the same panels): Δ is the Hodges–Lehmann estimate,
  the median of the averages of all pairs of per-panel differences, with its exact 95%
  interval. The test is Wilcoxon signed-rank. **Unpaired comparisons** (1,000 vs 3,000
  customers, which are different panels): the two-sample Hodges–Lehmann shift with its
  exact interval, and the Mann–Whitney U test.
- Δ is a median effect, so it can differ from the difference of the means shown as
  "A → B": a few panels with bias in the hundreds move the means far more than the median.
- **Bold** means significant at a 5% false discovery rate (Benjamini–Hochberg) within the
  claim's table. With 10 panels per cell the smallest possible Wilcoxon p is 0.002, so a
  single cell can only reach significance when all 10 panels move the same way.
- **Grids.** One grid per model and context: rows are purchase rates, columns churn levels,
  and each cell reads "Δ MAPE / Δ |bias| / Δ Spearman". The intervals and p-values behind
  every cell are in the collapsed tables under each grid.

### Summary

| # | Claim | Verdict | Holds | Does not hold |
| --- | --- | --- | --- | --- |
| 1 | Pareto/NBD beats the neural models. | **Partly** | Ranking (Spearman): every tree, every cell, both cohort sizes. Level (MAPE, \|bias\|): against the Transformer at every rate; against the LSTM at rates 0.01–0.05. | LSTM `ar_bounded` on MAPE at rate 0.30 (churn 20–60%) at 1,000 customers, and at rates 0.05–0.30 at 3,000. |
| 2 | Neural error rises with churn. | **Partly** | \|bias\| of the flagless LSTM at rates 0.01–0.05, both cohort sizes; MAPE of nearly every tree. | Not neural-specific for MAPE: Pareto/NBD's also rises (ρ +0.70 to +0.97). The flags remove the \|bias\| trend at rates 0.05–0.10. |
| 3 | Bounded AR flags help. | **Supported, conditionally** | LSTM: rates 0.01–0.10, from churn 40–60% up, by 34–468 MAPE points per cell; at 3,000 customers at every churn level on the sparsest panels. Transformer: rates 0.05–0.10. | Rate 0.30 (level unchanged, only RMSE improves); churn 20%; the Transformer at rate 0.01. |
| 4 | Unbounded counters hurt. | **Supported for the LSTM, partly for the Transformer** | LSTM: every rate, +98 to +260 MAPE points. Transformer: rate 0.30. | Transformer at rates 0.01–0.10; ranking improves for both models at rate 0.05. |
| 5 | A k-means cluster label hurts. | **Partly** | LSTM level at rates ≥ 0.10 (≥ 0.05 with flags). | It raises Spearman on sparse panels, and it helps whenever the unbounded counters are present. |
| 6 | One architecture is better overall. | **Not supported** | — | The Transformer is better at rate 0.01 and with unbounded counters; the LSTM at rates 0.10–0.30 with `no_ar` or flags. |
| 7 | Neural models capture seasonality; Pareto/NBD cannot. | **Supported** | Shape correlation 0.11–0.96 (LSTM) against −0.09 to +0.13 (Pareto/NBD); the true season cuts Pareto/NBD's MAPE by 5–20 points in every cell. | — |
| 8 | Pareto/NBD's low bias means it is accurate per customer. | **Not supported** | — | In every cell it serves living customers only 51–87% of their volume and leaks 11–104% onto dead ones. |
| 9 | Neural models cannot detect a customer who has stopped. | **Partly** | Transformer everywhere; LSTM on rates 0.01–0.10. | LSTM `ar_bounded` at rate 0.30 with churn 60–80%, and at rate 0.10 with churn 80%: its leakage equals Pareto/NBD's. |
| 10 | A bigger hyperparameter search helps. | **Partly** | Rate 0.30 for both models; the Transformer's ranking at rates 0.05–0.30. | Level at rates 0.01–0.10. |
| 11 | Specific hyperparameters drive the error. | **Not supported** | — | No within-cell correlation survives the FDR correction. |
| 12 | More customers improve the neural forecast. | **Supported, conditionally** | LSTM `ar_bounded`: every cell at rates 0.01–0.10. LSTM `no_ar`: rates 0.05–0.30. | LSTM `no_ar` at rate 0.01; both at rate 0.30 above churn 20%. The Transformer was not run at 3,000. |
| 13 | RMSE can rank these models. | **Not supported** for per-week RMSE; **supported** within one rate for customer-total RMSE | Customer-total RMSE ranks trees like MAPE (ρ +0.88 to +0.97 within each rate). | Per-week RMSE puts 4–10 of 13 trees on the same value to two decimals. |

### 1. Pareto/NBD beats the neural models

**Verdict: partly.** Pareto/NBD ranks customers better than every neural tree in every
cell, at both cohort sizes. On the level of the forecast it beats the Transformer
everywhere and the LSTM on sparse panels, but the LSTM with bounded flags beats it on MAPE
on dense panels, and at 3,000 customers from rate 0.05 up.

- **Ranking.** Δ Spearman is negative in all 16 cells for all six neural contexts, and
  significant in 95 of 96. The gap is largest at rate 0.01 (−0.11 to −0.34) and at rate
  0.30 with churn 80% (−0.19 to −0.26).
- **Level, LSTM `ar_bounded`.** Pareto/NBD is better at rate 0.01 (MAPE +11 to +394 points
  for the LSTM) and at rate 0.05 with churn 20–60%. At rate 0.10 MAPE is level and only
  |bias| favours Pareto/NBD. At rate 0.30 with churn 20–60% the LSTM is better by 13–18
  MAPE points and 2–8 |bias| points.
- **Level, Transformer.** Pareto/NBD is better at every rate pooled (MAPE +9 to +32 with
  flags, +17 to +79 without).
- **At 3,000 customers** the LSTM `ar_bounded` beats Pareto/NBD on MAPE at rates 0.05–0.30
  (−4 to −24 points) but still has the larger |bias| at rates 0.01–0.10.

Δ = neural minus Pareto/NBD: positive MAPE/|bias| or negative Spearman means Pareto/NBD
is better.

@@1_pnbd_vs_neural@@

### 2. Neural error rises with churn

**Verdict: partly.** MAPE rises with churn for nearly every tree, Pareto/NBD included, so
that half of the claim is not specific to the neural models. What is specific is the size
of the |bias| rise without flags: the LSTM `no_ar` goes from 59 to 755 at rate 0.01 and
from 42 to 497 at rate 0.05, against 12 to 54 and 8 to 21 for Pareto/NBD. With bounded
flags the |bias| trend disappears at rates 0.05–0.10 at both cohort sizes. It stays at rate
0.01 and, at 3,000 customers, reappears at rate 0.30 (3 → 24).

Each row gives the mean at churn 20 / 40 / 60 / 80% and the Spearman correlation of the
metric with churn over the rate's 40 panels. RMSE is left out: it falls with churn for
every tree because there is less volume.

@@2_churn_trend@@

### 3. Bounded AR flags help

**Verdict: supported, conditionally.** The flags improve the LSTM's level and ranking on
panels with purchase rates 0.01–0.10. The gain grows with churn, is absent at churn 20% and
at rate 0.30, and is larger with more customers. For the Transformer they help less, only
at rates 0.05–0.10.

**How much, LSTM at 1,000 customers.**

- Over the grid, the median panel's MAPE falls by 28 points [17, 44] and its |bias| by 32
  [20, 48]; the mean MAPE falls from 158 to 89.
- By rate: −44 MAPE points at rate 0.01, −85 at 0.05, −26 at 0.10, and no change at 0.30
  (−1 [−3, +1], p = 0.34). RMSE on customer totals improves at every rate, by 0.05–0.45.
- By churn: at rate 0.05 the gain is +5 (not significant), −42, −91 and −439 MAPE points
  at churn 20, 40, 60 and 80%. The same rise with churn holds at rates 0.01 and 0.10.
  Where silence is common, the flags are what lets the model learn that silence means
  dropout.
- Ranking improves at rates 0.05–0.10 (+0.41 and +0.03 Spearman) but gets worse at rate
  0.01 (−0.07 [−0.09, −0.04]).

**How much, Transformer at 1,000 customers.**

- Over the grid: −20 MAPE points [9, 31], −24 |bias|; the mean MAPE falls from 105 to 77.
- By rate: significant at 0.05 (−35) and 0.10 (−48), nothing at 0.01 or 0.30 on the level.
  Per cell, the MAPE gain is significant only at rate 0.10, churn 40% (−56) and 80% (−166).
- Ranking improves from rate 0.05 up in most cells, by 0.06–0.14.

**How much, LSTM at 3,000 customers.**

- The flags help more: −32 MAPE points over the grid, and −125 at rate 0.01, where they are
  significant in every churn cell (−46 to −251).
- At 3,000 customers the flags also improve the ranking at rate 0.01 (+0.07), where they
  worsened it at 1,000.
- At rate 0.30 they cost 2 MAPE points [0, 4] (p = 0.03), and 6 at churn 60%.

@@3_bounded@@

### 4. Unbounded counters hurt

**Verdict: supported for the LSTM, partly for the Transformer.** The counters make the
LSTM's level worse at every rate (+98 to +260 MAPE points, rate pooled) because they keep
growing through the holdout, past any value seen in training. The Transformer is hurt only
at rate 0.30 (+37 MAPE, +43 |bias|). Ranking does not follow the level: the counters raise
both models' Spearman at rate 0.05 (+0.24 LSTM, +0.11 Transformer).

@@4_unbounded@@

### 5. A k-means cluster label hurts

**Verdict: partly.** The label's effect depends on the model, the AR encoding and the rate.

- **LSTM `no_ar`:** no change in level at rates 0.01–0.05; worse at 0.10 (+34 MAPE) and
  0.30 (+53). It raises Spearman at 0.01–0.05 (+0.13, +0.31) and lowers it at 0.10–0.30
  (−0.07).
- **LSTM `ar_bounded`:** worse level from rate 0.05 up (+33 to +37 MAPE), better ranking at
  rate 0.01 only (+0.16).
- **Transformer `no_ar` and `ar_bounded`:** level mostly unchanged per cell; worse at rate
  0.30 without flags (+11 MAPE) and over the grid with flags (+12). Spearman rises on sparse
  panels (+0.08 to +0.13 at rate 0.01).
- **With `ar_unbounded`** the label helps both models (−74 and −31 MAPE over the grid),
  most at rates 0.05–0.30. It helps only by displacing the broken counters.

@@5_kmeans@@

### 6. One architecture is better overall

**Verdict: not supported.** Over the grid the two architectures are level with `no_ar`
(−4 MAPE, p = 0.62), and the LSTM is slightly better with flags (+9 for the Transformer,
p = 0.02). Both hide opposite effects by rate:

- **Rate 0.01:** the Transformer is better with every arm (−69 to −493 MAPE points), with
  a better ranking.
- **Rates 0.10–0.30:** the LSTM is better with `no_ar` and `ar_bounded` (+27 to +34 MAPE
  for the Transformer) and ranks customers better at rate 0.30.
- **With unbounded counters** the Transformer is better at every rate but 0.30: it
  tolerates the counters and the LSTM does not.

Δ = Transformer minus LSTM on the same panels.

@@6_architecture@@

### 7. Neural models capture seasonality; Pareto/NBD cannot

**Verdict: supported** for this one seasonal pattern.

- **Shape.** The LSTM's predicted weekly totals correlate with the actual ones at 0.11–0.96
  and the Transformer's at 0.09–0.62, against −0.09 to +0.13 for Pareto/NBD. The
  correlation rises with the rate and falls with churn.
- **Counterfactual.** Multiplying Pareto/NBD's forecast by the true seasonal curve (rescaled
  to mean 1) lowers its MAPE by 5–20 points, significant in 15 of 16 cells. |bias| and
  Spearman do not move (under 1 point and 0.005): the season changes the weekly shape, not
  the yearly total or the order of customers.
- **Given the season, Pareto/NBD beats the LSTM** on MAPE in every cell at rates 0.01–0.10
  (+7 to +399 points for the LSTM). At rate 0.30 the two are level (+2, not significant).
  So the LSTM's dense-panel MAPE wins in claim 1 come from modelling the season.

@@7b_shape@@

@@7_seasonality@@

### 8. Pareto/NBD's low bias means it is accurate per customer

**Verdict: not supported.** In every cell Pareto/NBD under-serves customers who are still
alive (alive ratio R_A 0.51–0.87, below 1 in all 16 cells) and assigns volume to customers
who have already died (dead leakage L_D 0.11–1.04, above 0 in all 16). The two errors
cancel in the total, which is why its |bias| is small. Both worsen with churn: at rate 0.01
and churn 80% it puts more volume on dead customers (L_D 1.04) than the living ones should
get. R_A and L_D use the generator's hidden truth (each customer's λ and death week).

@@8_pnbd_volume_split@@

### 9. Neural models cannot detect a customer who has stopped

**Verdict: partly.** On dead leakage L_D, lower is better.

- **LSTM `ar_bounded`** leaks more than Pareto/NBD at rates 0.01–0.10 (+0.13 to +4.44), most
  at rate 0.01, churn 80%. At rate 0.30 with churn 60–80% and at rate 0.10 with churn 80%
  its leakage equals Pareto/NBD's: there the flags work.
- **Transformer `ar_bounded`** leaks more than Pareto/NBD in every cell (+0.06 to +0.78),
  and significantly in 12 of 16.

@@9_dead_leakage@@

### 10. A bigger hyperparameter search helps

**Verdict: partly.** From the archived 10-trial (LSTM) and 20-trial (Transformer) `no_ar`
runs on the same panels to the 100-trial runs, the level improves only at rate 0.30 (LSTM
−11 MAPE / −15 |bias|, Transformer −22 / −29). At rates 0.01–0.10 no rate-pooled MAPE or
|bias| change is significant. The Transformer's ranking improves at rates 0.05–0.30 (+0.03
to +0.18). The archived run did not record its embedder, so the two runs may also differ
in that.

@@10_search_budget@@

### 11. Specific hyperparameters drive the error

**Verdict: not supported.** Four pooled correlations are significant, but none survives
within cells. For example, the LSTM `no_ar` batch size correlates with |bias| at +0.48 over
the grid and +0.04 within cells: the search picks larger batches on sparse panels, where
|bias| is large anyway. The closest to a within-cell effect is the Transformer `no_ar`
layer count against MAPE (+0.21, p = 0.008), which does not survive the correction across
the table's 88 tests. "Within cells" ranks both variables inside each rate × churn cell
before correlating.

@@11_hyperparameters@@

### 12. More customers improve the neural forecast

**Verdict: supported, conditionally.** Tripling the cohort (1,000 → 3,000 customers,
different panels, so unpaired tests):

- **LSTM `ar_bounded`:** MAPE improves in every cell at rates 0.01–0.10 (−12 to −298
  points); at rate 0.30 only at churn 20% (−5). Ranking improves only at rate 0.01 (+0.12).
- **LSTM `no_ar`:** MAPE improves at rates 0.05–0.30 but not at 0.01 (−11, p = 0.74).
  Without flags, extra customers do not help the sparsest panels.
- **Pareto/NBD:** MAPE improves modestly (−1 to −22 points, most at high churn); RMSE does
  not move.
- The Transformer was not run at 3,000 customers.

@@12_cohort_size@@

### 13. RMSE can rank these models

**Verdict: not supported for per-week RMSE; supported within one rate for customer-total
RMSE.** Per customer-week, RMSE puts 10 of 13 trees on the same value to two decimals at
rates 0.01–0.05, because almost every cell is a zero. On customer totals RMSE separates
the trees and orders them almost as MAPE does (ρ over the 13 trees +0.88 to +0.97). It grows
about 8-fold from rate 0.01 to 0.30, though, so it cannot be pooled across rates.

@@13_rmse@@
