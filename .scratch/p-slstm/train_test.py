"""P-sLSTM trunk + categorical head — training test on the electronics panel.

What this is testing: whether P-sLSTM's contribution (patching + channel
independence + an sLSTM stack) learns anything about sparse, zero-inflated
transaction counts once its linear regression head is replaced by a softmax over
count classes and its MSE loss by cross-entropy.

Shape of the task, matching what CLAUDE.md mandates for every model here:
one-step-ahead categorical prediction. Each training example is a lookback
window of `seq_len` periods ending at t, and the label is the count class at
t+1. The panel's `samples_f` is already the AR-shifted input (verified:
samples_f == calibration_f[:, :-1] and targets == calibration tx[:, 1:]), so a
window ending at index t sees nothing past t. That keeps it leak-free and makes
the model steppable by the Monte Carlo rollout later.
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from xlstm.components.feedforward import FeedForwardConfig

SEED = int(os.environ.get("SEED", 0))
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
SEQ_LEN    = 24    # lookback in weeks
PATCH_SIZE = 8
STRIDE     = 4     # -> patch_num = (24-8)//4 + 1 = 5 recurrent steps
EMB_DIM    = 32
NUM_BLOCKS = 2
NUM_HEADS  = 2
CONV_K     = 4
INCOME_EMB = 4
LR         = 1e-3
WEIGHT_DECAY = 1e-4
BATCH      = 256
EPOCHS     = 15

PATCH_NUM = (SEQ_LEN - PATCH_SIZE) // STRIDE + 1

# ---------------------------------------------------------------------------
# Data: windows over the panel, train/val split strictly by time
# ---------------------------------------------------------------------------
d = np.load("/home/virthian/Desktop/Thesis/panelclv/Datasets/electronics_panel.npz",
            allow_pickle=True)

feats   = d["samples_f"]          # (N, 103, 6) float — AR-shifted inputs
income  = d["samples_c"][:, 0, 0] # (N,) int — constant per customer, verified
targets = d["targets"].astype(np.int64)  # (N, 103) class index in {0,1,2}
N_CUST, N_STEP, N_CHAN = feats.shape
N_CLASS = int(targets.max()) + 1
N_INCOME = int(d["n_income"])

N_TRAIN_STEP = int(d["inner_targets"].shape[1])   # 95 — the inner training window

# Standardize float channels on TRAINING-window statistics only.
mu = feats[:, :N_TRAIN_STEP].mean(axis=(0, 1), keepdims=True)
sd = feats[:, :N_TRAIN_STEP].std(axis=(0, 1), keepdims=True)
sd[sd == 0] = 1.0
feats_z = (feats - mu) / sd

def windows(t_lo, t_hi):
    """Every (customer, t) pair with t in [t_lo, t_hi): window is feats[t-SEQ_LEN+1 : t+1]."""
    ts = np.arange(max(t_lo, SEQ_LEN - 1), t_hi)
    X = np.stack([feats_z[:, t - SEQ_LEN + 1 : t + 1] for t in ts], axis=1)  # (N, |ts|, L, C)
    y = targets[:, ts]                                                       # (N, |ts|)
    inc = np.repeat(income[:, None], len(ts), axis=1)
    return (X.reshape(-1, SEQ_LEN, N_CHAN),
            y.reshape(-1),
            inc.reshape(-1))

Xtr, ytr, itr = windows(0, N_TRAIN_STEP)              # weeks 23..94
Xva, yva, iva = windows(N_TRAIN_STEP, N_STEP)         # weeks 95..102 — never trained on

dev = "cuda" if torch.cuda.is_available() else "cpu"
to = lambda a, dt: torch.as_tensor(a, dtype=dt, device=dev)
Xtr, ytr, itr = to(Xtr, torch.float32), to(ytr, torch.long), to(itr, torch.long)
Xva, yva, iva = to(Xva, torch.float32), to(yva, torch.long), to(iva, torch.long)

# ---------------------------------------------------------------------------
# Model: P-sLSTM trunk, categorical head
# ---------------------------------------------------------------------------
class PsLSTMClassifier(nn.Module):
    """Patching + channel-independent sLSTM trunk -> softmax over count classes.

    Two deliberate departures from upstream, both forced by the task:
      * the head mixes channels. Upstream is strictly channel-independent because
        every channel forecasts itself; here only the target channel is forecast,
        from all channels, so the per-channel trunk outputs are concatenated
        before the head.
      * the head emits K logits for one step, not `pred_len` floats. This is what
        makes the model steppable by an AR rollout instead of direct multi-step.
    """

    def __init__(self):
        super().__init__()
        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=NUM_HEADS,
                    conv1d_kernel_size=CONV_K,
                    bias_init="powerlaw_blockdependent",
                    backend="vanilla",   # the CUDA kernels do not build on ROCm
                    dtype="float32",     # cell default is bfloat16, tuned for that kernel
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=PATCH_NUM,    # the recurrence is over patches, not raw weeks
            num_blocks=NUM_BLOCKS,
            embedding_dim=EMB_DIM,
            slstm_at="all",
        )
        self.embedding = nn.Linear(PATCH_SIZE, EMB_DIM)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.income_emb = nn.Embedding(N_INCOME, INCOME_EMB)
        self.head = nn.Linear(N_CHAN * PATCH_NUM * EMB_DIM + INCOME_EMB, N_CLASS)

    def forward(self, x, inc):
        # x: (B, L, C) — one lookback window per customer-period
        B, L, C = x.shape
        x = rearrange(x, "b l c -> (b c) l")
        x = x.unfold(dimension=-1, size=PATCH_SIZE, step=STRIDE)  # (B*C, N_patch, patch)
        x = self.embedding(x)                                     # (B*C, N_patch, E)
        x = self.xlstm_stack(x)                                   # (B*C, N_patch, E)
        x = x.reshape(B, C * PATCH_NUM * EMB_DIM)                 # channels rejoin here
        x = torch.cat([x, self.income_emb(inc)], dim=-1)
        return self.head(x)                                       # (B, K) logits

model = PsLSTMClassifier().to(dev)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lossf = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Baseline: the class prior fitted on the training window. Accuracy is
# meaningless at 97.5% zeros, so cross-entropy against this is the real bar.
# ---------------------------------------------------------------------------
prior = torch.bincount(ytr, minlength=N_CLASS).float(); prior /= prior.sum()
prior_ce_va = float(nn.functional.nll_loss(
    torch.log(prior).expand(len(yva), -1), yva))

print(f"device={dev}  params={n_par:,}  patches={PATCH_NUM}")
print(f"train windows={len(ytr):,}  val windows={len(yva):,}  classes={N_CLASS}")
print(f"train class prior = {prior.tolist()}")
print(f"BASELINE  val CE (class prior) = {prior_ce_va:.5f}\n")

@torch.no_grad()
def evaluate(X, y, inc):
    model.eval()
    ce, n, logits_all = 0.0, 0, []
    for i in range(0, len(y), 4096):
        lg = model(X[i:i+4096], inc[i:i+4096])
        ce += float(lossf(lg, y[i:i+4096])) * len(lg); n += len(lg)
        logits_all.append(lg)
    lg = torch.cat(logits_all)
    p = lg.softmax(-1)
    # Expected count under the predicted distribution vs the realised count:
    # the aggregate quantity the thesis actually scores.
    exp_count = (p * torch.arange(N_CLASS, device=dev)).sum(-1).sum()
    act_count = y.float().sum()
    return ce / n, float(exp_count), float(act_count)

t0 = time.time()
hist = []
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(ytr), device=dev)
    tot, seen = 0.0, 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        opt.zero_grad()
        loss = lossf(model(Xtr[idx], itr[idx]), ytr[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += float(loss.detach()) * len(idx); seen += len(idx)
    va_ce, exp_c, act_c = evaluate(Xva, yva, iva)
    hist.append((ep, tot/seen, va_ce))
    bias = 100 * (exp_c - act_c) / act_c
    print(f"epoch {ep:2d}  train CE {tot/seen:.5f}  val CE {va_ce:.5f}"
          f"  ({'BEATS' if va_ce < prior_ce_va else 'above'} prior)"
          f"  val expected/actual counts {exp_c:7.1f}/{act_c:7.1f}  bias {bias:+6.1f}%")

print(f"\nwall clock {time.time()-t0:.1f}s for {EPOCHS} epochs")
best = min(hist, key=lambda h: h[2])
print(f"best val CE {best[2]:.5f} at epoch {best[0]}  vs prior {prior_ce_va:.5f}"
      f"  ({100*(1-best[2]/prior_ce_va):+.1f}% vs baseline)")
