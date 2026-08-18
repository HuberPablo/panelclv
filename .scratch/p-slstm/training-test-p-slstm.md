# P-sLSTM — training test

Run 2026-08-18, following `research-p-slstm.md`. Script: `train_test.py` (run it with
the project venv; it reads `Datasets/electronics_panel.npz` and needs the vendored
xLSTM stack, see *Reproducing* below).

**Question asked:** does P-sLSTM's actual contribution — patching + channel
independence + an sLSTM stack — learn anything about sparse, zero-inflated
transaction counts, once its linear regression head is replaced by a softmax over
count classes?

**Answer: yes, modestly and reproducibly.** It beats the class prior by ~8.7% in
cross-entropy across three seeds. Aggregate count bias, which is what the thesis
actually scores, is wildly unstable.

---

## Setup

One-step-ahead categorical prediction, the shape `CLAUDE.md` mandates. Each example
is a lookback window of 24 weeks ending at t; the label is the count class at t+1.
`samples_f` is already the AR-shifted input (verified: `samples_f ==
calibration_f[:, :-1]` and `targets == calibration tx[:, 1:]`), so a window ending at
t sees nothing past t. Leak-free, and steppable by the Monte Carlo rollout.

- Data: `Datasets/electronics_panel.npz` — 829 customers, 6 float channels + income
  (constant per customer, embedded separately), 3 count classes (clip at 2).
- Split: **temporal.** Train on weeks 23–94 (59,688 windows), validate on weeks
  95–102 (6,632 windows), never trained on.
- Standardization: float channels z-scored on **training-window statistics only**.
- Trunk: patch_size 8, stride 4, seq_len 24 → **5 patches** = 5 recurrent steps.
  embedding_dim 32, 2 blocks, 2 heads, conv1d_kernel 4, `powerlaw_blockdependent`
  bias init, gelu FFN at proj_factor 1.3. 24,499 parameters.
- Head: per-channel trunk outputs concatenated + income embedding → Linear → 3 logits.
- Adam lr 1e-3, wd 1e-4, batch 256, grad clip 1.0, 15 epochs.

### Two deliberate departures from upstream

1. **The head mixes channels.** Upstream is strictly channel-independent because every
   channel forecasts itself. Here only the target channel is forecast, from all
   channels, so the per-channel trunk outputs are concatenated before the head.
2. **The head emits K logits for one step**, not `pred_len` floats. This is what makes
   the model steppable by an AR rollout instead of direct multi-step.

The trunk itself is unmodified.

---

## Results

Baseline is the training-window class prior — accuracy is meaningless here (98.6%
zeros in the training windows), so cross-entropy against the prior is the real bar.

| Seed | best val CE | at epoch | final-epoch val CE | vs prior |
|---|---|---|---|---|
| 0 | 0.11485 | 6  | 0.11613 | +9.0% |
| 1 | 0.11543 | 10 | 0.11580 | +8.5% |
| 2 | 0.11544 | 11 | 0.11687 | +8.5% |

- **Baseline (class prior): val CE 0.12618.**
- Best val CE: mean **0.11524**, std 0.00028, min 0.11485, max 0.11544.
- Final-epoch val CE: mean 0.11627, std 0.00045.
- Improvement over prior: **8.7% ± 0.3**.

Training CE plateaus at ~0.079 by epoch 3 and barely moves after; val CE bottoms out
around epoch 6–11 and then wanders. There is no meaningful overfitting at this size —
and not much more to extract either.

### The caveat that matters

**Aggregate expected-vs-actual counts on the validation window swing from -60% to
+125% bias, epoch to epoch, at essentially unchanged cross-entropy.** Actual val
count is 243; predicted totals range roughly 96–547 across epochs and seeds. The
model orders customers better than the prior does, but the *level* of its predicted
distribution is uncalibrated and unstable. Since `compute_forecast_metrics` scores
`bias_percent` on exactly this quantity, that instability — not the CE gain — is the
thing that would sink it in a real study. Any serious attempt needs calibration or a
loss that respects the aggregate.

### Cost

~190s for 15 epochs on the ROCm GPU (vanilla backend, 60k windows, 24.5k params).
Confirms the research note's prediction: the sequential Python loop is cheap because
the recurrence is over 5 patches, not 24 raw weeks.

---

## ROCm

Both fixes from `research-p-slstm.md` were applied and both were necessary:

1. `blocks/slstm/src/cuda_init.py` line 30 — `include_paths(cuda=True)` →
   `include_paths("cuda")`. Runs at import, so nothing imports without it.
2. `sLSTMLayerConfig(backend="vanilla", dtype="float32")`.

The vendored `blocks/slstm/src/cuda/` tree was deleted outright; nothing missed it.
No packages were installed; the venv is untouched.

---

## Reproducing

The vendored xLSTM stack is **not** committed here — it is ~272KB of upstream Apache-2.0
code and no decision has been made to vendor it into the package. To re-run:

```
git clone --depth 1 https://github.com/Eleanorkong/P-sLSTM.git
cp -r P-sLSTM/models/xlstm <dir>/          # next to train_test.py
rm -rf <dir>/xlstm/blocks/slstm/src/cuda
sed -i 's/include_paths(cuda=True)/include_paths("cuda")/' \
    <dir>/xlstm/blocks/slstm/src/cuda_init.py
SEED=0 <venv>/bin/python <dir>/train_test.py
```

---

## What this does NOT establish

- **No comparison against panelclv's own models.** Beating the class prior is a floor,
  not a benchmark. The open question is whether it beats `lstm` / `transformer` /
  `pareto_nbd` on the same split — untested.
- **No rollout.** This is teacher-forced one-step-ahead only. Nothing was forecast
  through `monte_carlo_forecasting`, so there are no `rmse` / `bias_percent` /
  `mape_aggregate` numbers comparable to a real study.
- **No holdout.** The 156-week holdout window was never touched.
- **No tuning.** One hyper-parameter point, picked to be small and plausible; no
  Optuna search, no patch-size sweep (which the paper says is the sensitive knob).
- **One dataset.** Electronics only.
