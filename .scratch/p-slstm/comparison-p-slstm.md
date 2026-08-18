# P-sLSTM vs the LSTM vs Pareto/NBD

Run 2026-08-18. Script: `compare.py`. Raw per-seed numbers: `comparison-results.json`.
Follows `training-test-p-slstm.md`, which left this comparison open.

**Verdict: no evidence P-sLSTM beats the LSTM on the forecast, and it costs ~14x more
to train.** It is the better one-step-ahead density model — lower validation
cross-entropy in every seed — but that advantage does not survive the rollout.

---

## Setup

Every model is fit and scored through the package's own machinery: the ADR-0001
temporal calibration split, the ADR-0008 warm-started full-calibration refit, the
Monte Carlo rollout, and `compute_forecast_metrics` as the single scoring authority.
The only thing hand-written is the P-sLSTM model and the windowed training loop its
architecture requires.

**One shared dataset** — the config `scripts/run_studies.py` already carries:
electronics weekly panel, 829 customers, calibration 1999-01-01..2000-12-31 (104
weeks), holdout 2001 (52 weeks), `clip_target_upper=6` → 7 classes, no covariates.
`F = 1`: the only channel is the target.

**Protocol, identical for both neural models:** temporal split at `val_start_idx=78`
→ early stopping on validation CE (patience 5, max 50 epochs) → warm-started refit on
the full calibration window for `best_epoch+1` epochs → 30-path Monte Carlo rollout
over the 52 holdout weeks, paths averaged. AdamW, lr 1e-3, weight decay 1e-4 for both.
Seeds 0–7 (n=8). Pareto/NBD is a single MCMC fit per seed, BTYDplus defaults, n=3
(it is near-deterministic: RMSE std 0.00002).

P-sLSTM rolls out through `simulate_attention_path` — it is stateless like the
Transformer, so it takes the growing-context rollout and reads the last 24 periods of
that context, which is the fixed lookback its patching is defined on.

---

## Results

| model | RMSE | bias % | aggregate MAPE % | mean \|bias\| % |
|---|---|---|---|---|
| LSTM (n=8)       | **0.3807** ± 0.0012 | **+2.41** ± 24.38 | **54.14** ± 5.90 | **19.8** |
| P-sLSTM (n=8)    | 0.3815 ± 0.0018 | +11.87 ± 27.81 | 56.93 ± 6.37 | 26.6 |
| Pareto/NBD (n=3) | 0.3758 ± 0.0000 | −63.70 ± 0.35 | 66.18 ± 0.28 | 63.7 |

Per-seed ranges: LSTM bias −26.1 to +33.5, P-sLSTM bias −35.8 to +37.0.

### Reading it

**RMSE separates nothing.** All three sit in 0.376–0.382 — a spread smaller than the
seed-to-seed noise of either neural model. On a target that is ~97% zeros, per-customer
per-period RMSE is dominated by predicting zero correctly, and Pareto/NBD "wins" it
while under-predicting the total by 64%. RMSE should not be the deciding metric here.

**Pareto/NBD is stable and badly biased.** −63.7% ± 0.35 across seeds: it predicts
barely a third of the transactions that occur. But it is *reproducibly* wrong, which
the neural models are not.

**Both neural models are unbiased on average and unstable per run.** The LSTM's mean
bias of +2.4% is an artefact of averaging −26% and +34% runs; std 24. P-sLSTM is
slightly worse on all three metrics and slightly noisier. Neither is reliable at the
level of a single fit — which is the case for the study-suite design that reports
distributions across many studies rather than one number.

**P-sLSTM wins one-step-ahead and loses the forecast.** Best validation
cross-entropy, in the six seeds captured in the run log (seeds 3 and 4 scrolled past):

| | LSTM | P-sLSTM |
|---|---|---|
| best val CE | 0.0997 – 0.1012 (mean 0.1004) | 0.0963 – 0.0968 (mean **0.0967**) |

P-sLSTM is lower in every one of them, by ~0.004 — consistent, not noise, and it
converges faster (best epoch 2–6 vs 3–10). That is a real if small advantage as a
*density model over the next period*. It does not become a better forecast: errors
compound over 52 autoregressive steps, and the aggregate metrics are no better.

**Cost.** P-sLSTM ~102–146s per seed end to end, LSTM ~7–8s: roughly 14x. Both on the
same GPU, P-sLSTM on the vanilla sLSTM backend (the CUDA kernels do not build on ROCm).

---

## What this does NOT establish

- **Neither neural model is tuned.** The repo's real studies run 100 Optuna trials per
  model; here the LSTM sits at a hand-picked mid-range point of its registry search
  space (`embedding_dim=128, hidden=64, dense=64, dropout=0.1`) and P-sLSTM at the one
  point from `training-test-p-slstm.md` (24-week lookback, patch 8, stride 4, 2 blocks,
  embedding_dim 32). Equal treatment, but both are under-fit relative to a real study,
  and the paper says patch size is the sensitive knob. Pareto/NBD needs no tuning, so
  its numbers are its real ones — the neural columns are the pessimistic ones.
- **The NoCov config gives P-sLSTM nothing to be channel-independent about.** `F = 1`.
  Half of the paper's contribution is untested here. The earlier training test, which
  did have 6 covariates, is the only place channel independence did anything.
- **The neural heads cap at class 6** while holdout actuals reach 26, which biases both
  downward on heavy buyers. Pareto/NBD is unconstrained. This affects the comparison
  between families, not between the two neural models.
- One dataset, one holdout window, 8 seeds, one architecture point each.

## Reproducing

`compare.py` needs the vendored xLSTM stack next to it — see the clone recipe in
`training-test-p-slstm.md` — and the cached `prepare_dataset` output. Build the latter
from `Datasets/Dataset_clean/electronics_customer_week_panel.csv` with the PanelConfig
in `scripts/run_studies.py`, pickled to `elec_data.pkl`. The package is not installed
in the venv; run with `PYTHONPATH=src`.
