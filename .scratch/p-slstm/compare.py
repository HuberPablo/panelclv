"""P-sLSTM vs the LSTM vs the Pareto/NBD benchmark, on one shared dataset.

Every model is fit and scored through the package's own machinery: the ADR-0001
temporal calibration split, the ADR-0008 full-calibration refit, the Monte Carlo
rollout, and `compute_forecast_metrics` as the single scoring authority. The only
thing written here is the P-sLSTM model itself and the windowed training loop its
architecture requires.

Protocol, identical for both neural models:
  1. temporal split at `val_start_idx` — train on transitions whose target period
     is before it, score cross-entropy on the suffix.
  2. early stopping on validation CE (patience 5, max 50 epochs) -> best_epoch.
  3. warm-started refit on the FULL calibration window for best_epoch+1 epochs.
  4. 30-path Monte Carlo rollout over the 52-week holdout, paths averaged.
  5. rmse / bias_percent / mape_aggregate from compute_forecast_metrics.

Seeds 0,1,2 for each model; results reported as a distribution.
"""

import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/virthian/Desktop/Thesis/panelclv/src")

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange

from panelclv.registry import build_model, rollout_for
from panelclv.trials.loaders import split_calibration, refit_loader
from panelclv.training import fit_model, refit_full_calibration
from panelclv.models.monte_carlo_forecasting import (
    forecast_attention, compute_forecast_metrics,
)
from panelclv.benchmarks.pareto_nbd import pareto_forecast

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from xlstm.components.feedforward import FeedForwardConfig

SCRATCH = "/tmp/claude-1000/-home-virthian-Desktop-Thesis-panelclv/52dbdc45-f08b-4176-bad7-8c10eb57ab3d/scratchpad"
CKPT = f"{SCRATCH}/ckpt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
N_SIMS = 30
MAX_EPOCHS, PATIENCE = 50, 5
LR, WD = 1e-3, 1e-4

# P-sLSTM trunk knobs — the same point tested in training-test-p-slstm.md.
SEQ_LEN, PATCH_SIZE, STRIDE = 24, 8, 4
EMB_DIM, NUM_BLOCKS, NUM_HEADS, CONV_K = 32, 2, 2, 4
PATCH_NUM = (SEQ_LEN - PATCH_SIZE) // STRIDE + 1

with open(f"{SCRATCH}/elec_data.pkl", "rb") as f:
    data = pickle.load(f)

K = data["embedded_cols"][data["target_col"]]     # softmax head size = 7
TGT = int(data["target_idx"])
S_VAL = int(data["val_start_idx"])
F = int(data["F"])

# Calibration statistics for the target channel. P-sLSTM patches raw counts, so it
# standardizes them INSIDE the model: the rollout writes raw sampled class indices
# into the target channel, and normalizing outside would feed the model raw counts
# at forecast time after training it on z-scores.
_cal_tgt = data["calibration"][:, :, TGT]
TGT_MEAN, TGT_STD = float(_cal_tgt.mean()), float(_cal_tgt.std()) or 1.0


# ---------------------------------------------------------------------------
# P-sLSTM: upstream trunk, categorical head
# ---------------------------------------------------------------------------
class PsLSTM(nn.Module):
    """Patching + channel-independent sLSTM trunk -> logits over count classes."""

    def __init__(self):
        super().__init__()
        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=NUM_HEADS, conv1d_kernel_size=CONV_K,
                    bias_init="powerlaw_blockdependent",
                    backend="vanilla",   # the CUDA kernels do not build on ROCm
                    dtype="float32",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=PATCH_NUM, num_blocks=NUM_BLOCKS,
            embedding_dim=EMB_DIM, slstm_at="all",
        )
        self.embedding = nn.Linear(PATCH_SIZE, EMB_DIM)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.head = nn.Linear(F * PATCH_NUM * EMB_DIM, K)
        self.register_buffer("mu", torch.tensor(TGT_MEAN))
        self.register_buffer("sd", torch.tensor(TGT_STD))

    def logits(self, w):
        """w: (B, SEQ_LEN, F) lookback window -> (B, K) logits for the next period."""
        B, L, C = w.shape
        w = w.clone()
        w[:, :, TGT] = (w[:, :, TGT] - self.mu) / self.sd   # standardize the count channel
        x = rearrange(w, "b l c -> (b c) l")
        x = x.unfold(dimension=-1, size=PATCH_SIZE, step=STRIDE)   # (B*C, N_patch, patch)
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        return self.head(x.reshape(B, C * PATCH_NUM * EMB_DIM))

    def to_rollout(self):
        return RolloutPsLSTM(self)


class RolloutPsLSTM(nn.Module):
    """Sampling wrapper with the stateless-rollout contract `simulate_attention_path`
    expects: forward(context, only_last=True) -> ((N, 1, 1) sample, None).

    P-sLSTM is stateless like the Transformer, so it rolls out through the growing-
    context simulator — but it only ever reads the last SEQ_LEN periods of that
    context, which is the fixed lookback its patching is defined on.
    """

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, x, state=None, only_last=True):
        window = x[:, -SEQ_LEN:, :]                       # (N, SEQ_LEN, F)
        probs = torch.softmax(self.trunk.logits(window), dim=-1)
        sample = dist.Categorical(probs=probs).sample().float()   # (N,)
        return sample.view(-1, 1, 1), None


def pslstm_windows(t_lo, t_hi):
    """AR transitions t in [t_lo, t_hi) as (window, label).

    The window is samples[:, t-SEQ_LEN+1 : t+1] — the lookback ENDING at t — and the
    label is targets[:, t], the count one period later. Transitions before
    SEQ_LEN-1 have no full lookback and are dropped: a cost of the fixed-window
    design that the LSTM, which carries state from period 0, does not pay.
    """
    X = data["samples"]                                # (N, T-1, F)
    y = data["targets"].squeeze(-1).astype(np.int64)   # (N, T-1)
    ts = np.arange(max(t_lo, SEQ_LEN - 1), t_hi)
    W = np.stack([X[:, t - SEQ_LEN + 1 : t + 1] for t in ts], axis=1)
    return (torch.from_numpy(W.reshape(-1, SEQ_LEN, F)),
            torch.from_numpy(y[:, ts].reshape(-1)))


def train_pslstm(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PsLSTM().to(DEV)

    # Same temporal boundary split_calibration uses: train on transitions 0..S_VAL-2,
    # score on S_VAL-1..T-2. No validation period is ever trained on.
    Xtr, ytr = pslstm_windows(0, S_VAL - 1)
    Xva, yva = pslstm_windows(S_VAL - 1, data["samples"].shape[1])
    Xtr, ytr, Xva, yva = (t.to(DEV) for t in (Xtr, ytr, Xva, yva))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()

    def val_ce():
        model.eval()
        with torch.no_grad():
            tot = sum(float(lossf(model.logits(Xva[i:i+4096]), yva[i:i+4096])) * len(yva[i:i+4096])
                      for i in range(0, len(yva), 4096))
        return tot / len(yva)

    best, best_ep, bad, best_state = float("inf"), 0, 0, None
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr), device=DEV)
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            opt.zero_grad()
            loss = lossf(model.logits(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = val_ce()
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"    [P-sLSTM] best val CE {best:.5f} at epoch {best_ep+1}")

    # ADR-0008 refit: warm-started fine-tune on the FULL calibration window,
    # best_epoch+1 epochs, no validation.
    Xf, yf = pslstm_windows(0, data["samples"].shape[1])
    Xf, yf = Xf.to(DEV), yf.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    for _ in range(best_ep + 1):
        model.train()
        perm = torch.randperm(len(yf), device=DEV)
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            opt.zero_grad()
            loss = lossf(model.logits(Xf[idx]), yf[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


# ---------------------------------------------------------------------------
# LSTM: built, fit and refit entirely through the package
# ---------------------------------------------------------------------------
LSTM_PARAMS = {   # mid-range point of the registry's search space; no Optuna run
    "embedding_dim": 128, "lstm_hidden_size": 64, "dense_units": 64, "dropout": 0.1,
}

def train_lstm(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    split = split_calibration(data, batch_size=128)
    model = build_model("lstm", LSTM_PARAMS, split.recipe)
    fit = fit_model(
        model, split.train_loader, split.val_loader, num_target_classes=K,
        n_epochs=MAX_EPOCHS, patience=PATIENCE, learning_rate=LR, weight_decay=WD,
        device=DEV, checkpoint_dir=CKPT, model_name=f"lstm_seed{seed}",
        val_score_start=split.recipe["val_score_start"], verbose=False,
    )
    print(f"    [LSTM] best val CE {fit.best_val_loss:.5f} at epoch {fit.best_epoch+1}")
    refit_full_calibration(
        model, refit_loader(data, batch_size=128), num_target_classes=K,
        n_epochs=fit.best_epoch + 1, learning_rate=LR, weight_decay=WD, device=DEV,
        checkpoint_dir=CKPT, model_name=f"lstm_refit_seed{seed}",
        warm_start_state=model.state_dict(), verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
results = {"lstm": [], "p_slstm": [], "pareto_nbd": []}
timings = {}

for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    t0 = time.time()
    lstm = train_lstm(seed)
    fc = rollout_for("lstm")(lstm.to_rollout(), data, n_simulations=N_SIMS,
                             device=DEV, seed=seed, return_simulations=False)
    m = compute_forecast_metrics(fc["actual"], fc["prediction_mean"])
    results["lstm"].append(m); print(f"    [LSTM]    {m}")
    timings.setdefault("lstm", []).append(time.time() - t0)

    t0 = time.time()
    ps = train_pslstm(seed)
    fc = forecast_attention(ps.to_rollout(), data, n_simulations=N_SIMS,
                            device=DEV, seed=seed, return_simulations=False)
    m = compute_forecast_metrics(fc["actual"], fc["prediction_mean"])
    results["p_slstm"].append(m); print(f"    [P-sLSTM] {m}")
    timings.setdefault("p_slstm", []).append(time.time() - t0)

# Pareto/NBD: one MCMC fit per seed (no training, no rollout).
if os.environ.get("SKIP_PARETO"):
    results.pop("pareto_nbd")
    SEEDS = []
for seed in SEEDS:
    t0 = time.time()
    fc = pareto_forecast(data, seed=seed)
    m = compute_forecast_metrics(fc["actual"], fc["prediction_mean"])
    results["pareto_nbd"].append(m)
    print(f"[Pareto/NBD seed {seed}] {m}")
    timings.setdefault("pareto_nbd", []).append(time.time() - t0)

print("\n" + "=" * 78)
print(f"{'model':<12} {'metric':<16} {'mean':>10} {'std':>9} {'min':>10} {'max':>10}")
print("=" * 78)
summary = {}
for name, runs in results.items():
    summary[name] = {}
    for metric in ("rmse", "bias_percent", "mape_aggregate"):
        v = np.array([r[metric] for r in runs])
        summary[name][metric] = dict(mean=float(v.mean()), std=float(v.std()),
                                     min=float(v.min()), max=float(v.max()),
                                     runs=[float(x) for x in v])
        print(f"{name:<12} {metric:<16} {v.mean():>10.4f} {v.std():>9.4f} "
              f"{v.min():>10.4f} {v.max():>10.4f}")
    print("-" * 78)
for k, v in timings.items():
    print(f"{k:<12} mean wall clock {np.mean(v):.1f}s")

with open(f"{SCRATCH}/comparison_{os.environ.get("TAG","a")}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {SCRATCH}/comparison.json")
