# P-sLSTM

How P-sLSTM works, what had to change to make it fit this package's contract, and
what it scored when it was measured against the LSTM and Pareto/NBD.

**P-sLSTM is not part of `panelclv`.** It was evaluated as a candidate contribution
in August 2026 and not adopted: it does not beat the LSTM on the forecast and costs
~14× more to train. Nothing under `src/panelclv/` mentions it. This document exists
because the negative result is worth keeping — it is a measured answer about a
current architecture, not a dead end to forget — and because understanding *why* it
loses says something about what this forecasting problem actually rewards.

Read `CONTEXT.md` first for the vocabulary (*calibration*, *holdout*, *rollout*,
*trial*). This document assumes it, and assumes the categorical-head contract
`CLAUDE.md` states: logits `(B, T, K)`, a count as a class not a quantity,
cross-entropy on a class index, evaluation by sampling-and-averaging.

Everything below about the upstream model was read from the paper and the source.
Everything about its behaviour here was measured; the runs are dated and their
scripts named. Claims that were not verified are marked **UNVERIFIED**.

## Contents

1. [Identity — which paper](#1-identity--which-paper)
2. [The idea](#2-the-idea)
3. [The trunk, shape by shape](#3-the-trunk-shape-by-shape)
4. [Patching — and the thing it does to the recurrence](#4-patching--and-the-thing-it-does-to-the-recurrence)
5. [Channel independence](#5-channel-independence)
6. [Inside an sLSTM cell](#6-inside-an-slstm-cell)
7. [Upstream's head, loss and normalization](#7-upstreams-head-loss-and-normalization)
8. [What had to change to fit this package](#8-what-had-to-change-to-fit-this-package)
9. [The rollout contract](#9-the-rollout-contract)
10. [What was measured](#10-what-was-measured)
11. [The verdict, and what it rests on](#11-the-verdict-and-what-it-rests-on)
12. [Running it — ROCm, vendoring, reproducing](#12-running-it--rocm-vendoring-reproducing)

---

## 1. Identity — which paper

There are two unrelated papers whose names collide. Getting this wrong wastes a day,
so it goes first.

| arXiv | Title | Authors | Relation |
|---|---|---|---|
| `2408.10006` | *Unlocking the Power of LSTM for Long Term Time Series Forecasting* | Kong, Wang, Nie, Zhou, Zohren, Liang, Sun, Wen | **This is P-sLSTM.** |
| `2506.11997` | *pLSTM: parallelizable Linear Source Transition Mark networks* | Pöppel, Freinschlag, Schmied, Lin, Hochreiter | **Unrelated.** Linear RNNs on DAGs and grids, evaluated on molecular graphs and vision. Not forecasting. Name collision only. |

- **Venue:** AAAI-25, vol. 39 no. 11, pp. 11968–11976.
- **Code:** `https://github.com/Eleanorkong/P-sLSTM`, Apache-2.0. The repo owner
  matches first author Yaxuan Kong, and its README names `2408.10006` verbatim.

**The letters mean:** **P** = **Patching** — not "parallel", not "probabilistic".
**sLSTM** = the **scalar LSTM** cell from Beck et al.'s xLSTM (2024), used verbatim.
The repo vendors NX-AI's `xlstm` v1.0.4 source into `models/xlstm/`, original
copyright headers intact, and builds a stack with `slstm_at="all"` — every block is
an sLSTM block. The mLSTM path ships in the vendored code but is commented out.

---

## 2. The idea

The paper's argument runs in three steps.

1. sLSTM's **exponential gating** (from xLSTM, built for language) helps long-range
   sequential learning — but sLSTM still has a **short memory** problem that blocks
   direct use on long forecasting horizons.
2. The fix is two tricks borrowed from the Transformer forecasting literature
   (PatchTST in particular): **patching** and **channel independence**.
3. A memory / geometric-ergodicity argument in the appendix says why those two
   should help.

So the novelty is not a new cell. It is the claim that an off-the-shelf sLSTM stack,
fed patches instead of raw steps and run one channel at a time, becomes competitive
with Transformers on long-term multivariate forecasting — at lower training cost.

The whole model is **73 lines** (`models/P_sLSTM.py`). Everything else in the repo
is either the vendored xLSTM stack or Time-Series-Library scaffolding.

---

## 3. The trunk, shape by shape

Upstream input `x` is `[B, L, M]` — batch, lookback length, channels.

| # | Operation | Shape after | What it means |
|---|---|---|---|
| 0 | input | `[B, L, M]` | `L` raw time steps, `M` parallel series |
| 1 | `rearrange 'b l c -> (b c) l'` | `[B·M, L]` | **channel independence** — every channel is now its own row in the batch |
| 2 | `unfold(-1, patch_size, stride)` | `[B·M, N, P]` | **patching** — `N = (L − P) // S + 1` |
| 3 | `nn.Linear(P, E)` | `[B·M, N, E]` | each patch of raw floats becomes one embedding vector |
| 4 | `xLSTMBlockStack` | `[B·M, N, E]` | **the recurrence — over `N`, not `L`** |
| 5 | `flatten(1)` | `[B·M, N·E]` | every patch position kept, not just the last |
| 6 | `nn.Linear(N·E, pred_len)` | `[B·M, T]` | one shot: all `T` horizon steps at once |
| 7 | reshape | `[B, T, M]` | channels folded back out |

Two shapes in that table carry all the design.

**Step 4 is the one to stare at.** The RNN never sees `L` steps. In every shipped
config `stride == patch_size`, so the patches are non-overlapping and
`N = L / P`: with `L=336, P=56` the recurrence is **6 steps long**. Across all five
of the paper's datasets `N` lands between 6 and 21. An architecture sold as "unlocking
LSTM for long-term forecasting" runs its LSTM over a sequence shorter than most
sentences.

**Step 5 keeps every patch.** The head reads the full `N·E` flattened stack, not the
final hidden state — so information from early patches reaches the output directly,
not only through the recurrence. That weakens how much the recurrence has to carry,
which is a second reason the short `N` is survivable.

**Depth:** `--num_blocks` is **1 or 2** in every shipped script and in every appendix
hyper-parameter table. Never deeper.

---

## 4. Patching — and the thing it does to the recurrence

Patching does three things at once, and they are worth separating because only two
are the paper's stated motivation.

1. **Stated:** it shortens the sequence the recurrence must remember across, which is
   the direct attack on sLSTM's short memory.
2. **Stated:** each input token is now a *window* of the series rather than one value,
   so a single step carries local shape — trend and periodicity within the patch —
   into the cell.
3. **Unstated, and the one that matters here:** it makes the sequential Python
   fallback of the sLSTM cell affordable. See §12 — the vanilla backend loops over
   the sequence axis in Python, which would be ruinous over 336 steps and is cheap
   over 6.

The cost is that the patch boundary is arbitrary. The paper flags this itself: patch
size is the **sensitive knob** — performance improves with it up to "an experimental
optimal solution" and then degrades — and future work is "more complex patching
techniques to preserve as much of the original periodicity of time series as possible."

---

## 5. Channel independence

Step 1 flattens the channel axis into the batch. Every channel is then forecast by
the same backbone with the same weights, seeing only itself. No channel ever sees
another.

The paper claims this is the first application of channel independence to RNN-based
forecasting, and Table 4 makes the case empirically: on Weather, channel-mixing
overfits badly — train MSE 0.049 against test MSE 0.227 — while channel independence
generalises. Stated future work is to "capture multivariate correlations" without
giving that up.

**This is the half of the contribution that this package cannot use as written**, and
§8 explains why.

---

## 6. Inside an sLSTM cell

Unchanged from Beck et al. — P-sLSTM sets configuration, not code. One block, in
order:

- **Pre-LayerNorm residual wrapper.**
- **Optional causal `conv1d`** (`conv1d_kernel_size`, 2–32 in the shipped scripts)
  with Swish, feeding the input and forget gates only.
- **Block-diagonal multi-head recurrent layer** — the memory-mixing is per head, and
  heads do not mix with each other inside the recurrence.
- **Head-wise GroupNorm.**
- **Gated MLP**, which P-sLSTM configures with `proj_factor=1.3` and `act_fn="gelu"`.

What makes it *s*LSTM rather than LSTM is the gating. The input and forget gates are
**exponential** rather than sigmoid, which lets the cell revise a stored value sharply
instead of decaying towards it. Exponentials overflow, so the cell carries two extra
states alongside `c_t`:

- a **normalizer** `n_t`, which tracks the accumulated gate mass so the output can be
  divided back to a sane scale;
- a **stabilizer** `m_t`, a running max in log space subtracted before exponentiating
  — the same trick as a numerically-stable softmax.

P-sLSTM also sets `bias_init="powerlaw_blockdependent"`, which initialises the forget
gate biases so that different blocks start out with different memory time constants.

**A wiring quirk, verified in the source:** `--dropout` and `--group_norm_weight` are
parsed by `run_longExp.py` and passed by every shipped script, but `models/P_sLSTM.py`
**never forwards them into `sLSTMLayerConfig`**. The layer uses its own defaults
(`dropout=0.0`). The dropout values printed in the appendix tables are dead in this
code.

---

## 7. Upstream's head, loss and normalization

This section is the one that decides how much of P-sLSTM can be reused here, so it is
worth being blunt about how far upstream sits from this package.

- **Direct multi-step, not autoregressive.** One forward pass emits all `pred_len`
  steps from a single linear layer. Nothing is sampled, nothing is fed back, there is
  no step-by-step decoding.
- **The head is a plain linear regressor over continuous values.** Output is floats.
- **Nothing categorical, nothing distributional.** No softmax, no classes, no
  quantiles, no mixture density, no count likelihood anywhere in the repository.
- **Loss is MSE, hard-coded.** `_select_criterion` returns `nn.MSELoss()`
  unconditionally; the `--loss` argument is parsed and never read.
- **No RevIN, no instance normalization** — a genuine difference from PatchTST.
  Standardization is a single sklearn `StandardScaler` fit on the training split.
- **Metrics are never un-scaled.** `inverse_transform` exists on the dataset and is
  not called in the metric path, so the published MSE/MAE are in z-score units. This
  is the Time-Series-Library convention, and it is much criticised.
- **Every evaluation dataset is dense, continuous and high-frequency** — Weather,
  Electricity, Solar, ETTm1, PEMS03, sampled from 6 minutes to 1 hour, no structural
  zeros. Sparse, intermittent and count-valued data are **not discussed anywhere in
  the paper** — not even as a stated limitation, because the regime is never
  considered.

Against that: this package forecasts weekly per-customer transaction counts that are
sparse, integer, and ~97% zeros. **The trunk may transfer. The head, the loss, and
the normalization assumptions do not.**

---

## 8. What had to change to fit this package

The adaptation keeps the trunk verbatim and replaces everything around it. Four
changes, in `.scratch/p-slstm/compare.py`.

**1 — The head emits `K` logits for one period, not `pred_len` floats.**
`nn.Linear(F · N · E, K)`. This is the change that makes the model steppable by an
autoregressive rollout at all: a direct multi-step head produces the whole horizon in
one shot and has nowhere to feed a sampled count back into.

**2 — The head mixes channels.** Upstream is strictly channel-independent because
every channel forecasts *itself*. Here only the target channel is forecast, from all
channels, so the per-channel trunk outputs are concatenated before the head. This is a
deliberate departure — the trunk stays channel-independent, the head does not.

**3 — The target channel is standardized *inside* the model.** The trunk patches raw
values and feeds them to `nn.Linear(P, E)`, so it needs z-scores; but the rollout
writes **raw sampled class indices** back into the target channel as it steps.
Normalizing outside the model would train it on z-scores and then hand it raw counts
at forecast time. So calibration mean and std ride along as registered buffers and the
model normalizes the count channel in its own `forward`:

```python
w[:, :, TGT] = (w[:, :, TGT] - self.mu) / self.sd
```

This is a real trap and it is specific to the count channel: it is the only channel
whose values the rollout invents.

**4 — Loss is cross-entropy on a class index**, per the contract, replacing MSE.

The trunk itself is unmodified.

---

## 9. The rollout contract

P-sLSTM is **stateless** — there is no hidden state to thread from one period to the
next, because the trunk re-reads a whole lookback window every call. That makes it a
Transformer-shaped model as far as this package is concerned, so it forecasts through
`simulate_attention_path`, the growing-context rollout, rather than
`simulate_recurrent_path`.

The wrapper implements the contract that simulator expects —
`forward(context, only_last=True) -> ((N, 1, 1) sample, None)`:

```python
window = x[:, -SEQ_LEN:, :]                      # ignore all context beyond the lookback
probs  = torch.softmax(self.trunk.logits(window), dim=-1)
sample = dist.Categorical(probs=probs).sample()
```

**The asymmetry worth noticing:** `simulate_attention_path` hands over a context that
grows one period per step — calibration, then calibration plus every holdout period
sampled so far. The Transformer uses all of it. P-sLSTM throws most of it away and
reads the last `SEQ_LEN` periods, because a fixed lookback is what its patching is
defined on. It is given more history than it can use, by construction.

**And it pays a second cost the LSTM does not.** Training windows need a full
lookback, so every transition before period `SEQ_LEN − 1` is dropped. The LSTM carries
state from period 0 and trains on all of them.

---

## 10. What was measured

Two runs, both 2026-08-18, both on the electronics weekly panel. Full write-ups:
`.scratch/p-slstm/training-test-p-slstm.md` and `.scratch/p-slstm/comparison-p-slstm.md`.

### Run 1 — does the trunk learn anything at all?

One-step-ahead categorical prediction, teacher-forced, no rollout. 829 customers, 6
float channels plus a separately-embedded income covariate, counts clipped at 2 → 3
classes. Temporal split: train weeks 23–94 (59,688 windows), validate weeks 95–102
(6,632 windows). Trunk at patch 8 / stride 4 / lookback 24 → **5 patches**,
`embedding_dim` 32, 2 blocks, 2 heads. **24,499 parameters.**

Accuracy is meaningless at 98.6% zeros, so the bar is cross-entropy against the
training-window class prior.

| | val CE |
|---|---|
| Class prior (baseline) | 0.12618 |
| P-sLSTM, best, mean of 3 seeds | **0.11524** ± 0.00028 |

**Answer: yes — 8.7% ± 0.3 better than the prior, reproducibly.** Training CE
plateaus by epoch 3 and val CE bottoms out around epoch 6–11, so the model is neither
overfitting nor holding much more in reserve at this size.

**The caveat that turned out to matter:** aggregate expected-vs-actual counts on the
validation window swung from **−60% to +125% bias, epoch to epoch, at essentially
unchanged cross-entropy** — actual 243, predicted totals ranging ~96–547. The model
orders customers better than the prior; the *level* of its predicted distribution is
uncalibrated. Since `compute_forecast_metrics` scores `bias_percent` on exactly that
quantity, this was flagged at the time as the thing that would sink it. It did.

### Run 2 — against the LSTM and Pareto/NBD

Full pipeline, everything scored through the package's own machinery: the ADR-0001
temporal calibration split, the ADR-0008 warm-started full-calibration refit, the
Monte Carlo rollout, `compute_forecast_metrics` as the single scoring authority. The
config `scripts/run_studies.py` already carries: 829 customers, calibration
1999–2000 (104 weeks), holdout 2001 (52 weeks), `clip_target_upper=6` → 7 classes, no
covariates so `F = 1`. Both neural models: identical protocol, AdamW, lr 1e-3, wd
1e-4, early stopping patience 5, 30-path rollout, seeds 0–7.

| model | RMSE | bias % | aggregate MAPE % | mean \|bias\| % |
|---|---|---|---|---|
| LSTM (n=8) | **0.3807** ± 0.0012 | **+2.41** ± 24.38 | **54.14** ± 5.90 | **19.8** |
| P-sLSTM (n=8) | 0.3815 ± 0.0018 | +11.87 ± 27.81 | 56.93 ± 6.37 | 26.6 |
| Pareto/NBD (n=3) | 0.3758 ± 0.0000 | −63.70 ± 0.35 | 66.18 ± 0.28 | 63.7 |

And the one-step-ahead density quality, from the six seeds captured in the run log:

| | LSTM | P-sLSTM |
|---|---|---|
| best val CE | 0.0997 – 0.1012 (mean 0.1004) | 0.0963 – 0.0968 (mean **0.0967**) |

**Cost:** P-sLSTM 102–146 s per seed end to end; LSTM 7–8 s. Roughly **14×**, same
GPU, P-sLSTM on the vanilla backend.

---

## 11. The verdict, and what it rests on

**P-sLSTM wins one step ahead and loses the forecast.** Its validation cross-entropy
is lower than the LSTM's in every seed by ~0.004 — consistent, not noise — and it
converges in fewer epochs. It is genuinely the better density model over the next
period. That advantage does not survive 52 autoregressive steps: errors compound, and
every aggregate metric comes out slightly worse.

Three things in that table deserve to be read carefully, because two of them are traps.

**RMSE separates nothing, and should not decide this.** All three models sit in
0.376–0.382 — a spread narrower than the seed-to-seed noise of either neural model.
On a target that is ~97% zeros, per-customer per-period RMSE is dominated by getting
the zeros right. Pareto/NBD "wins" RMSE while under-predicting the total number of
transactions by 64%.

**Pareto/NBD is stable and badly wrong; the neural models are unbiased on average and
unstable per run.** The LSTM's mean bias of +2.4% is an artefact of averaging a −26%
run with a +34% one, std 24. That is exactly why the study-suite design reports
distributions across many studies rather than a single number.

**P-sLSTM is worse on all three metrics and noisier on all three.** Not dramatically —
but it is worse, not better, and it costs 14×.

### What this does not establish

- **Neither neural model is tuned.** Real studies run 100 Optuna trials per model.
  Here the LSTM sits at a hand-picked mid-range point of its registry search space and
  P-sLSTM at the single point from run 1. Equal treatment, but both are under-fit
  relative to a real study — and the paper says patch size is the sensitive knob, which
  was never swept. Pareto/NBD needs no tuning, so its column is its real one and the
  neural columns are pessimistic.
- **`F = 1` gives channel independence nothing to be independent about.** Half the
  paper's contribution is untested by run 2. Run 1, with 6 covariates, is the only
  place it did anything.
- **Both neural heads cap at class 6 while holdout actuals reach 26**, biasing both
  downward on heavy buyers. Pareto/NBD is unconstrained. This affects the comparison
  *between families*, not between the two neural models.
- One dataset, one holdout window, 8 seeds, one architecture point each.
- **UNVERIFIED:** whether the published P-sLSTM numbers reproduce. No upstream
  training run was attempted.

**What would change the answer:** a patch-size sweep, a covariate config that gives
channel independence real work to do, and a loss or calibration step that respects the
aggregate rather than only the per-period density. The measured failure is calibration
of the *level*, not the ordering — and the ordering is the part P-sLSTM is good at.

---

## 12. Running it — ROCm, vendoring, reproducing

### Vendorability: high

The novel contribution is 73 lines. What it needs underneath is the vendored xLSTM
sLSTM stack — realistically `blocks/slstm/{cell,layer,block}.py`,
`blocks/xlstm_block.py`, `xlstm_block_stack.py` and `components/*`, roughly 1000–1500
lines of the 2701 vendored — and none of the mLSTM or language-model code. There is
**no `xlstm` pip dependency**; the source ships in the repo, Apache-2.0, with NX-AI's
copyright headers intact. Nothing outside `models/` is needed if you bring your own
data pipeline and training loop, which this package has.

### The sLSTM cell has two backends, and the default one does not build on ROCm

`sLSTMCellConfig.backend` defaults to `"cuda"`, which JIT-compiles hand-written CUDA
kernels through `torch.utils.cpp_extension.load`. The alternative, `"vanilla"`, is 39
lines of plain PyTorch gate math driven by a sequential Python loop over the sequence
axis.

Two fixes were needed on the ROCm workstation, and **both were necessary**:

1. **`blocks/slstm/src/cuda_init.py` line 30** calls
   `torch.utils.cpp_extension.include_paths(cuda=True)`. On torch 2.9 the signature is
   `include_paths(device_type='cpu')`, so this raises `TypeError`. It runs at *module
   import*, guarded by `torch.cuda.is_available()` — which is **True on ROCm** — and
   `cell.py` imports it unconditionally, so **nothing imports without this fix, not
   even the vanilla path.** One token: `include_paths("cuda")`.
2. **`sLSTMLayerConfig(backend="vanilla", dtype="float32")`.** The dtype matters:
   the config defaults to `bfloat16`, which is tuned for the CUDA kernel.

The CUDA backend genuinely does not build on ROCm. Torch's hipify converts the
sources, but the xLSTM config hard-codes nvcc-only flags (`-Xptxas`, `-res-usage`,
`--extra-device-vectorization`), an NVIDIA arch (`arch=compute_80` — A100), and hits a
missing `hipsparse/hipsparse.h`. Porting the kernels is not a small job. The vendored
`blocks/slstm/src/cuda/` tree was deleted outright and nothing missed it.

**The performance penalty is smaller than it sounds**, and this is a structural point
in the design's favour: the vanilla loop runs over the *patch* axis `N`, not the raw
lookback. In run 1 that was 5 steps, not 24 — 15 epochs over 60k windows took ~190 s.
(Exact vanilla-vs-CUDA speed ratio: **UNVERIFIED**, never benchmarked.)

### Reproducing

The vendored xLSTM stack is **not committed to this repo** — it is ~272 KB of upstream
code and no decision was made to vendor it. To re-run either script:

```bash
git clone --depth 1 https://github.com/Eleanorkong/P-sLSTM.git
cp -r P-sLSTM/models/xlstm .scratch/p-slstm/
rm -rf .scratch/p-slstm/xlstm/blocks/slstm/src/cuda
sed -i 's/include_paths(cuda=True)/include_paths("cuda")/' \
    .scratch/p-slstm/xlstm/blocks/slstm/src/cuda_init.py
```

Then, with the project venv's interpreter:

- **Run 1** — `SEED=0 <venv>/bin/python .scratch/p-slstm/train_test.py`, which reads
  `Datasets/electronics_panel.npz`.
- **Run 2** — `.scratch/p-slstm/compare.py`, which additionally needs a cached
  `prepare_dataset` output pickled to `elec_data.pkl`, built from
  `Datasets/Dataset_clean/electronics_customer_week_panel.csv` with the `PanelConfig`
  in `scripts/run_studies.py`. The package is not installed in the venv, so run it
  with `PYTHONPATH=src`.

  Note that `compare.py:44` hardcodes `SCRATCH` to a session scratchpad directory
  that no longer exists. Repoint it at wherever `elec_data.pkl` actually is before
  running.

### Where everything lives

| Path | What |
|---|---|
| `.scratch/p-slstm/research-p-slstm.md` | Source-level notes on the upstream paper and repo |
| `.scratch/p-slstm/train_test.py` | Run 1 — one-step-ahead trunk test |
| `.scratch/p-slstm/training-test-p-slstm.md` | Run 1 write-up |
| `.scratch/p-slstm/compare.py` | Run 2 — full pipeline vs LSTM and Pareto/NBD |
| `.scratch/p-slstm/comparison-p-slstm.md` | Run 2 write-up |
| `.scratch/p-slstm/comparison-results.json` | Run 2 raw per-seed numbers |
