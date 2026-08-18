# P-sLSTM — research notes

Researched 2026-08-13. Sources: the arXiv listing pages, the paper PDF (text-extracted
locally with `pdftotext`), and a shallow clone of the GitHub repo read file-by-file.
Anything not directly verified is marked **UNVERIFIED**.

---

## 0. Identity: which paper is which

The task named arXiv `2506.11997` as a primary source. **It is a different paper by
different authors and is not related to P-sLSTM.**

| arXiv | Title | Authors | Relation to P-sLSTM |
|---|---|---|---|
| `2506.11997` | *pLSTM: parallelizable Linear Source Transition Mark networks* | Korbinian Pöppel, Richard Freinschlag, Thomas Schmied, Wei Lin, Sepp Hochreiter | **Unrelated.** Multi-dimensional linear RNNs on DAGs/grids; evaluated on molecular graphs and computer vision. Not time-series forecasting. Name collision only (`pLSTM` vs `P-sLSTM`). |
| `2408.10006` | *Unlocking the Power of LSTM for Long Term Time Series Forecasting* | Yaxuan Kong, Zepu Wang, Yuqi Nie, Tian Zhou, Stefan Zohren, Yuxuan Liang, Peng Sun, Qingsong Wen | **This is P-sLSTM.** |

**The GitHub repo `Eleanorkong/P-sLSTM` implements `2408.10006`.** Its README states this
verbatim, and the repo owner (`Eleanorkong`) matches first author Yaxuan Kong.

- **Venue:** AAAI-25 (39th AAAI Conference on Artificial Intelligence), vol. 39 no. 11,
  pp. 11968–11976. Publisher link in README: `https://ojs.aaai.org/index.php/AAAI/article/view/33303`.
- **License:** Apache-2.0.

### Problem it solves
Long-term multivariate time series forecasting (LTSF). The claim: sLSTM's exponential
gating (from Beck et al.'s xLSTM, built for NLP) helps long-range sequential learning,
but sLSTM still has a *short memory* problem that blocks direct use in forecasting. The
fix is two borrowed tricks — **patching** and **channel independence** — plus a
theoretical argument (geometric ergodicity / memory analysis) for why they help.

---

## 1. Architecture

### What "P" and "sLSTM" denote
- **P** = **Patching**. (Not "parallel", not "probabilistic".)
- **sLSTM** = the **scalar LSTM** cell from Beck et al. 2024's xLSTM — exponential input
  and forget gates, a normalizer state `n_t`, a stabilizer state `m_t`, and multi-head
  memory mixing via a block-diagonal recurrent matrix.

**Yes, it is the Beck et al. xLSTM sLSTM cell — verbatim.** The repo vendors NX-AI's
`xlstm` package source (v1.0.4) into `models/xlstm/`, including the original NXAI
copyright headers. P-sLSTM builds an `xLSTMBlockStack` with `slstm_at="all"` (i.e. every
block is an sLSTM block; the mLSTM path is present in the vendored code but commented out
in the model file).

### Data flow (read directly from `models/P_sLSTM.py`, 73 lines)

Input `x`: `[B, L, M]` (batch, lookback length, channels).

1. `rearrange 'b l c -> b c l'` then `'b c l -> (b c) l'` → `[(B·M), L]`.
   **This is channel independence**: every channel becomes an independent row sharing one
   backbone. Confirmed present and central (the paper claims it is the first application
   of CI to RNN-based TSF).
2. `x.unfold(dim=-1, size=patch_size, step=stride)` → `[(B·M), N, P]`, where
   `N = (L - patch_size) // stride + 1`.
   In every shipped script **`stride == patch_size`** → non-overlapping patches, no
   padding. E.g. `L=336, P=56, S=56 → N=6`.
3. `nn.Linear(patch_size, embedding_dim)` → `[(B·M), N, E]`.
4. `xLSTMBlockStack` (the sLSTM blocks) → `[(B·M), N, E]`.
   **Critical shape fact: the recurrence runs over the *patch* axis `N`, not over `L`.**
   Sequence length seen by the RNN is ~6–21 steps in the shipped configs.
5. `flatten(1)` → `[(B·M), N·E]`.
6. `nn.Linear(N·E, pred_len)` → `[(B·M), T]`.
7. Reshape → `[B, T, M]`.

This matches the paper's prose exactly ("After flattening, the data becomes
(B·M)×(N·Embedding) and is finally projected to (B·M)×T by a linear layer").

### Number of sLSTM blocks
`--num_blocks`, **1 or 2** in every shipped script. Appendix hyper-parameter tables
confirm 1–2 across Weather / Electricity / Solar / ETTm1 / PEMS03. Never deeper.

### Inside one sLSTM block (vendored xLSTM, unchanged)
Pre-LayerNorm residual block → optional causal conv1d (`conv1d_kernel_size`, 2–32 in the
scripts) + Swish on the i/f gates → block-diagonal multi-head recurrent layer →
head-wise GroupNorm → gated MLP with GeLU. P-sLSTM sets `proj_factor=1.3` and
`act_fn="gelu"` on the feedforward, and `bias_init="powerlaw_blockdependent"`.

### Config-wiring quirks worth knowing (verified by reading the code)
- `--dropout` and `--group_norm_weight` are parsed in `run_longExp.py` and passed by every
  script (e.g. `--dropout 0.1`), but **`models/P_sLSTM.py` never forwards them into
  `sLSTMLayerConfig`.** The layer therefore uses its own defaults (`dropout=0.0`,
  `group_norm_weight=True`). The dropout values in the appendix tables are effectively
  dead in this code.
- `context_length=configs.seq_len` is passed to `xLSTMBlockStackConfig` even though the
  stack actually sees `N` steps. It feeds the `powerlaw_blockdependent` bias init.
- `configs.channel` is stored on the module but unused in `forward` (channel count comes
  from `x.shape`).

---

## 2. Output shape and head

- **Direct multi-step.** One forward pass emits all `H = pred_len` steps at once, from a
  single `nn.Linear(N·E, pred_len)` over the flattened block output.
- **Not autoregressive.** There is no sampling, no feedback of a predicted value, no
  step-by-step decoding. The training loop calls `self.model(batch_x)` once and compares
  to the whole target window.
- **Head is a plain linear regressor over continuous values.** Output is
  `[B, T, M]` of floats.
- **Nothing categorical, nothing distributional.** No softmax, no classes, no quantiles,
  no mixture density, no count likelihood anywhere in the repo. Grep for
  `CrossEntropy`/`softmax`/`NegativeBinomial`/`Poisson` in the model or training path
  returns nothing.

> Relevance to panelclv: this is the opposite shape from what `CLAUDE.md` mandates
> (categorical head, class-index target, evaluation by sampling-and-averaging rollout).
> Adopting P-sLSTM would mean keeping its *trunk* (patching + CI + sLSTM blocks) and
> replacing the head with a `(B, T, K)` softmax, plus driving it through the existing
> Monte Carlo rollout instead of its one-shot linear projection.

---

## 3. Loss and normalization

- **Loss: MSE.** `exp/exp_main.py::_select_criterion` returns `nn.MSELoss()`,
  unconditionally. The `--loss` CLI arg (default `'mse'`) is parsed but **never read** —
  the criterion is hard-coded. The appendix confirms: "employed mean squared error as our
  loss function".
- **Evaluation metrics:** `utils/metrics.py` — MAE, MSE, RMSE, MAPE, MSPE, RSE, CORR.
  Reported headline numbers in the paper are MSE/MAE.

### Normalization — important detail
- **No RevIN. No instance normalization.** Grep across the repo for
  `revin|instance_norm|normaliz` finds only an unrelated `--group_norm_weight` arg and
  comments in `layers/AutoCorrelation.py` (dead code from the Time-Series-Library base).
  This is a genuine difference from PatchTST, which does use RevIN.
- **Global standardization only.** `data_provider/data_loader.py` uses sklearn
  `StandardScaler`, **fit on the training split** and applied to the whole series.
- **It is never inverted for scoring.** `exp/exp_main.py::test` computes metrics directly
  on the scaled `preds`/`trues`; `inverse_transform` exists on the dataset but is not
  called in the metric path. This is the standard (and much-criticised) Time-Series-Library
  convention — metrics are in z-score units.
- The only normalization inside the network is the vendored xLSTM's own LayerNorm /
  head-wise GroupNorm.

---

## 4. Training setup

From `run_longExp.py` defaults, `exp/exp_main.py`, `utils/tools.py`, the shipped scripts,
and the appendix.

| Item | Value |
|---|---|
| Optimizer | Adam (`optim.Adam`), no weight decay set |
| Learning rate | CLI default `1e-4`; scripts use `6e-5`, `1e-4`, `5e-4`, `1e-3`. Appendix says "initial learning rate of 1e-3" — **inconsistent with the scripts**, flagged |
| LR schedule | `--lradj type1` (default): halve the LR every epoch, `lr * 0.5**(epoch-1)` |
| Batch size | 16 (Weather, Electricity), 32 (ETTm1, Solar, PEMS03) |
| Epochs | CLI default `train_epochs=10`; some scripts set `20` |
| Early stopping | `utils/tools.py::EarlyStopping`, on **validation loss**, `delta=0`, `--patience` default 3 (scripts use 3 or 5); restores best checkpoint |
| Lookback `L` | **336** for Weather / Electricity / Solar / ETTm1; **96** for PEMS03 |
| Horizons `H` | **{96, 192, 336, 720}** for the four; **{12, 24, 48, 96}** for PEMS03 |
| AMP | `--use_amp` available (off by default), uses `torch.cuda.amp` |
| `--itr` | number of repeat runs; default 2, scripts use 1 |
| Seeds | **UNVERIFIED** — no seeding call found in `run_longExp.py` or `exp_main.py`. Reproducibility is not enforced by the code. |

### Per-dataset hyper-parameters (from scripts, cross-checked against appendix Tables 6–10)

| Dataset | `L` | patch / stride | blocks | embed dim | heads | conv1d k | channels |
|---|---|---|---|---|---|---|---|
| Weather | 336 | 56 / 56 | 2,2,1,2 | 100 | 2–4 | 8 | 21 |
| Electricity | 336 | 56 / 56 (script) | 1 | 600 | 3 | 8 (script) / 32 (appendix) | 321 |
| Solar | 336 | 16 / 16 | 1 | 100 | 2 | 4 | 137 |
| ETTm1 | 336 | 6 / 6 | 1 | 100 | 2–4 | 2–32 | 7 |
| PEMS03 | 96 | 16 / 16 | 2 | 300 | 6 | 8 | 358 |

(Minor script-vs-appendix disagreements on conv size and patch size for Electricity; the
appendix Table 7 lists patch `56/16/16/16` across horizons while the first script block
uses 56. Noted, not resolved.)

---

## 5. Data assumptions

**Datasets evaluated:** Weather (21 vars, 10-min), Electricity (321 consumers, hourly),
Solar (137 PV plants, 10-min), ETTm1 (7 factors, 15-min), PEMS03 (358 traffic sensors,
6-min). Baselines/splits follow iTransformer (Liu et al. 2024) and Time-Series-Library.

**The model assumes dense, continuous-valued, smoothly-varying real series.** Evidence:

- Patches of raw floats go straight into `nn.Linear(patch_size, embedding_dim)` — a real-
  valued affine map, meaningless for a categorical code.
- MSE loss on z-scored values presumes an approximately continuous, roughly symmetric
  error distribution.
- Every dataset is a densely-sampled physical/consumption sensor series. All are
  high-frequency (6 min to 1 hour) with no structural zeros.
- Sequence lengths: raw lookback 96–336 steps, but the RNN only ever traverses `N ≈ 6–21`
  patch steps.

**No handling of sparse / intermittent / count-valued data anywhere.** No zero-inflation,
no count likelihood, no discrete support, no masking for missing periods. The paper does
not mention sparse, intermittent, count, or discrete data at any point (verified by
grepping the extracted full text of both the paper and the appendix).

> Relevance to panelclv: customer-level transaction counts are sparse, integer, heavily
> zero-inflated, and low-frequency. That is squarely outside the regime this model and its
> evaluation were built for. The *trunk* may still transfer; the head, loss, and
> normalization assumptions do not.

---

## 6. Dependencies (and the ROCm question)

### `requirements.txt` verbatim

```
# PyTorch 2.1.0, Python 3.10, and CUDA 12.1 or PyTorch 2.3.0, Python 3.12, CUDA 12.1 should work
matplotlib
pandas
scikit-learn
# torch==1.9.0
torch
tqdm
einops
pre-commit==3.6.0
ipykernel==6.29.0
dacite==1.8.1
omegaconf==2.3.0
torchmetrics==1.3.0
tqdm==4.66.1
pytest==8.0.0
pytest-xdist==3.5.0
numpy==1.26.4
ninja
```

Key points:

- **There is NO `xlstm` pip dependency.** The xLSTM source is vendored at
  `models/xlstm/` (`__version__ = "1.0.4"`, NX-AI copyright). Good for vendorability.
- **No `mlstm_kernels`.** Not imported, not present.
- **No Triton.** Zero `triton` references in the repo.
- `dacite` and `omegaconf` are listed but **never imported anywhere** in the repo
  (verified by grep) — dead requirements. Same for `torchmetrics`, `pytest`,
  `pre-commit`, `ipykernel` on the model path.
- Actually needed to run the model: `torch`, `einops`, `numpy`, `pandas`,
  `scikit-learn`, `matplotlib` (plots only). `ninja` only for the CUDA backend.

### The sLSTM cell has TWO backends

`models/xlstm/blocks/slstm/cell.py`:

```python
backend: Literal["vanilla", "cuda"] = "cuda"     # line 68 — default is CUDA
...
class sLSTMCell(sLSTMCellBase):
    def __new__(cls, config, skip_backend_init=False):
        if config.backend == "cuda":   return sLSTMCell_cuda(...)
        elif config.backend == "vanilla": return sLSTMCell_vanilla(config)
```

- **`cuda`** — JIT-compiles hand-written CUDA (`src/cuda/*.cu`, ~7 files) via
  `torch.utils.cpp_extension.load` at first use. This is why `ninja` is required.
- **`vanilla`** — **plain PyTorch**, no custom kernels. The pointwise gate math is 39
  lines (`src/vanilla/slstm.py`, uses only `torch.exp`, `logsigmoid`, `tanh`, `sigmoid`,
  `max`), driven by a sequential Python loop over the sequence axis in
  `src/vanilla/__init__.py`.

**`models/P_sLSTM.py` does not set `backend`, so it silently gets the CUDA default.**
`sLSTMLayerConfig` subclasses `sLSTMCellConfig`, so `backend="vanilla"` is a one-keyword
change.

### ROCm findings — empirically tested on this machine

Tested with the project venv (`torch 2.9.1+rocm6.4`, `torch.version.hip = 6.4.43484`,
`torch.cuda.is_available() → True`). No packages were installed; nothing in the user's
repo was modified (work done on a throwaway clone in the scratchpad).

**Blocker 1 — import-time crash, hits every backend.**
`src/cuda_init.py` line 30 runs at *module import*, guarded by `torch.cuda.is_available()`
— which is **True on ROCm**:

```python
os.environ["CUDA_LIB"] = os.path.join(
    os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0], "lib")
```

On torch 2.9 the signature is now `include_paths(device_type: str = 'cpu')`, so this
raises `TypeError: include_paths() got an unexpected keyword argument 'cuda'`. And
`cell.py` imports `cuda_init` unconditionally (`from .src.cuda_init import load`), so
**even the vanilla backend cannot be imported without patching this line.**
Fix is one token: `include_paths("cuda")`.

**Blocker 2 — the CUDA backend genuinely does not build on ROCm.**
After patching blocker 1, requesting `backend="cuda"` fails. Torch's hipify does convert
the sources, but the xLSTM config hard-codes nvcc-only flags and an NVIDIA arch:

```
clang++: error: unknown argument: '-Xptxas'
clang++: error: unknown argument: '--extra-device-vectorization'
clang++: error: no such file or directory: 'arch=compute_80,code=compute_80'
fatal error: hipsparse/hipsparse.h: No such file or directory
RuntimeError: Error building extension 'slstm_HS100BS8NH2...'
```

So: nvcc-specific flags (`-Xptxas`, `-res-usage`, `--extra-device-vectorization`),
`-gencode arch=compute_80` (A100), and a missing hipSPARSE dev header. Porting the CUDA
kernels to ROCm is not a small job.

**Result — the vanilla backend WORKS on this ROCm GPU.** Verified end-to-end,
forward *and* backward, with the exact P-sLSTM stack config (2 blocks, `embedding_dim=100`,
`num_heads=2`, `conv1d_kernel_size=8`, `bias_init="powerlaw_blockdependent"`,
`proj_factor=1.3`, gelu):

```
vanilla backend on cpu:  OK  out=(4, 6, 100)
vanilla backend on cuda: OK  out=(4, 6, 100)     # 'cuda' device == ROCm here
```

**ROCm risk verdict: LOW, with two one-line fixes.**
1. Patch `cuda_init.py` line 30 (or delete the vendored `src/cuda/` tree entirely and stub
   the `load` import).
2. Pass `backend="vanilla"` into `sLSTMLayerConfig`.

Also set `dtype="float32"`: `sLSTMCellConfig.dtype` defaults to `"bfloat16"`, which is
tuned for the CUDA kernel. (The float32 path is what was tested above.)

**Performance caveat:** the vanilla backend is a sequential Python loop over the sequence
axis, so it is far slower per step than the fused kernel. **But in P-sLSTM the recurrence
length is the number of patches `N` (≈6–21), not the raw lookback.** The loop is therefore
short and the penalty is small — a structural property of the patching design, and a
notable point in favour of vendoring this. (Exact speed ratio: **UNVERIFIED**, not
benchmarked.)

---

## 7. Repo layout and vendorability

Clone: `https://github.com/Eleanorkong/P-sLSTM` (shallow clone read in full).

| Path | Lines | What |
|---|---|---|
| `models/P_sLSTM.py` | **73** | The entire model. Patching + embedding + block stack + linear head. |
| `models/sLSTM.py` | 63 | Ablation: sLSTM without patching. |
| `models/xlstm/**` (Python) | **2701** total | Vendored NX-AI xlstm v1.0.4. |
| `models/xlstm/blocks/slstm/cell.py` | 786 | Both backends + config dataclass. Largest single file. |
| `models/xlstm/blocks/slstm/layer.py` | 161 | conv1d + multi-head recurrent + GroupNorm. |
| `models/xlstm/blocks/slstm/block.py` | 40 | Residual block wrapper. |
| `models/xlstm/blocks/slstm/src/vanilla/slstm.py` | 39 | Pure-PyTorch gate math. |
| `models/xlstm/blocks/slstm/src/cuda/*` | ~7 files | CUDA kernels — **droppable**. |
| `exp/exp_main.py` | — | Training/val/test loop (Time-Series-Library derived). |
| `data_provider/data_loader.py` | — | Datasets + StandardScaler. |
| `run_longExp.py` | — | Argparse entry point. |
| `scripts/EXP-LongForecasting/P_sLSTM/*.sh` | 5 files | Repro scripts (weather, ettm1, electricity, solar, PEMS_03). |
| `Appendices.pdf` | — | sLSTM equations, stabilizer proof, hyper-parameter tables. |
| `layers/`, `models/{Autoformer,Informer,DLinear,Transformer}.py`, `utils/masking.py` | — | Baseline/TSLib leftovers, unused by P-sLSTM. |

**Vendorability: high.** The novel contribution is 73 lines. The dependency is the
vendored xLSTM sLSTM stack — realistically `blocks/slstm/{cell,layer,block}.py`,
`blocks/xlstm_block.py`, `xlstm_block_stack.py`, `components/*` (~1000–1500 lines of the
2701), and none of the mLSTM or LM-model code. The CUDA tree can be deleted outright once
`backend="vanilla"` is pinned. Nothing outside `models/` is needed if you supply your own
data pipeline and training loop — which panelclv already has.

---

## 8. Stated limitations

Verbatim / near-verbatim from the paper:

- **Noise.** On PEMS03: "A potential explanation is that PEMS03 is a very noisy series and
  P-sLSTM does not include any denoising mechanisms."
- **No parallelism.** "there are still known limitations of LSTM/RNNs, such as that they
  cannot be computed in parallel. To help models perform parallel computing, we can
  consider adding mLSTM, another LSTM structure that can perform parallel computing."
- **Patching is crude.** Future work: "more complex patching techniques to preserve as
  much of the original periodicity of time series as possible."
- **Channel independence discards cross-series structure.** Future work: "capture
  multivariate correlations among time series data." Table 4 shows CI beating
  channel-mixing (CM) decisively on Weather — CM overfits badly (train MSE 0.049 vs test
  0.227).
- **Patch size is sensitive.** Performance improves with patch size up to "an experimental
  optimal solution" then degrades.
- **Sparse / intermittent / count-valued series: NOT DISCUSSED AT ALL.** No limitation is
  stated because the regime is never considered.

Efficiency claim (Table 5): training time per dataset, P-sLSTM vs iTransformer — Weather
52.54s vs 79.12s, ETTm1 93.13s vs 114.25s. (These are CUDA-backend numbers.)

---

## 9. Explicitly UNVERIFIED

- Random seeding / run-to-run reproducibility — no seeding code found, but absence in the
  two files read is not proof of absence repo-wide.
- Exact speed penalty of the vanilla backend vs the CUDA kernel — not benchmarked.
- Whether the published AAAI camera-ready differs from arXiv v1 (`arxiv.org/html/2408.10006v1`
  was the HTML source used for some experimental details).
- The script-vs-appendix hyper-parameter disagreements (Electricity patch size and conv
  size; the appendix's "1e-3 initial LR" vs the scripts' 6e-5–5e-4) are reported as found,
  not adjudicated.
- Whether `P-sLSTM` numbers reproduce — no training run was attempted.
