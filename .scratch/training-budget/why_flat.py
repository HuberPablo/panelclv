"""Why is the temporal validation curve flat, when the customer-wise one is not?

The obvious candidate: 98.6% of the cells are zeros, so aggregate cross-entropy is
dominated by how well the model predicts silence. If the zero part saturates in the first
epoch while the non-zero part keeps improving, then the number early stopping watches
stops moving long before the model stops learning — and a selection criterion computed on
the cells that carry the signal would still have something to see.

Trains once with early stopping off, under BOTH splits, and decomposes the validation
cross-entropy each epoch into the cells whose true count is zero and the cells whose true
count is positive.

    PYTHONPATH=src .../python .scratch/training-budget/why_flat.py
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from run_real_panel_benchmarks import build_data                    # noqa: E402
from panelclv.registry import build_model                           # noqa: E402
from panelclv.trials import split_calibration                       # noqa: E402

PARAMS = {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32}
N_EPOCHS = 120


def loaders_temporal(data):
    """ADR-0001: train on the prefix, score the validation suffix only."""
    split = split_calibration(data, PARAMS["batch_size"])
    return split.train_loader, split.val_loader, split.recipe, split.recipe["val_score_start"]


def loaders_customer(data, seed=0):
    """The notebook's split: a random 10% of customers, every period scored."""
    X, y = data["samples"], data["targets"].squeeze(-1).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(X))
    n_valid = round(len(order) * 0.1)
    va, tr = order[:n_valid], order[n_valid:]
    mk = lambda idx, sh: DataLoader(                                    # noqa: E731
        TensorDataset(torch.from_numpy(X[idx].copy()), torch.from_numpy(y[idx].copy())),
        batch_size=PARAMS["batch_size"], shuffle=sh)
    recipe = {"seq_cols": data["seq_cols"], "embedded_cols": data["embedded_cols"],
              "target_col": data["target_col"], "seq_len": X.shape[1]}
    return mk(tr, True), mk(va, False), recipe, 0


@torch.no_grad()
def decomposed_val_loss(model, loader, device, score_start):
    """Validation CE overall, on zero-count cells, and on positive-count cells."""
    model.eval()
    sums, counts = {"all": 0.0, "zero": 0.0, "pos": 0.0}, {"all": 0, "zero": 0, "pos": 0}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)[:, score_start:, :]
        target = yb[:, score_start:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             target.reshape(-1), reduction="none")
        flat = target.reshape(-1)
        for key, mask in (("all", torch.ones_like(flat, dtype=torch.bool)),
                          ("zero", flat == 0), ("pos", flat > 0)):
            sums[key] += float(ce[mask].sum()); counts[key] += int(mask.sum())
    return {k: (sums[k] / counts[k] if counts[k] else float("nan")) for k in sums}, counts


device = "cuda" if torch.cuda.is_available() else "cpu"
data = build_data("electronics", "2y")
rows = []
for split_name, make in (("temporal", loaders_temporal), ("customer_wise", loaders_customer)):
    train_loader, val_loader, recipe, score_start = make(data)
    model = build_model("valendin_lstm", {**PARAMS, "n_epochs": N_EPOCHS}, recipe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PARAMS["learning_rate"],
                            weight_decay=PARAMS["weight_decay"])
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val, counts = decomposed_val_loss(model, val_loader, device, score_start)
        rows.append(dict(split=split_name, epoch=epoch, **val))
    print(f"{split_name}: scored cells {counts['all']:,} "
          f"({counts['pos']:,} positive = {counts['pos']/counts['all']:.2%})", flush=True)

d = pd.DataFrame(rows)
d.to_csv(RESULTS / "why_flat.csv", index=False)
print("\nValidation CE by epoch, decomposed (mean over cells):")
show = [0, 1, 2, 4, 9, 19, 39, 79, 119]
for split_name, g in d.groupby("split"):
    g = g.set_index("epoch")
    print(f"\n  {split_name}")
    print("   epoch |      all |     zero |      pos")
    for e in show:
        if e in g.index:
            r = g.loc[e]
            print(f"   {e:5d} | {r['all']:8.5f} | {r['zero']:8.5f} | {r['pos']:8.4f}")
    first, best = g.iloc[0], g.loc[g["all"].idxmin()]
    print(f"   improvement from epoch 0 to the best epoch: "
          f"all {100*(first['all']-best['all'])/first['all']:.1f}%, "
          f"zero {100*(first['zero']-best['zero'])/first['zero']:.1f}%, "
          f"pos {100*(first['pos']-best['pos'])/first['pos']:.1f}%")
