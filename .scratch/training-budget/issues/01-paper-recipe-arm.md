# 01 — The `paper` arm: train the way the paper trains

Status: `ready-for-agent`

## Where the numbers come from

`Original_paper_model/banking_transactions_demo.ipynb`, the reference notebook ADR-0004
freezes the architecture against:

| setting | notebook | cell |
| --- | --- | --- |
| optimizer | `Adam()` — all defaults, so lr 1e-3 and **no weight decay** | 15 |
| loss | `sparse_categorical_crossentropy` | 15 |
| batch size | `BATCH_SIZE_TRAIN = 32` | 10 |
| early stopping | `EarlyStopping(monitor='val_loss', min_delta=0., patience=5, restore_best_weights=True)` | 15 |
| epoch budget | `MAX_EPOCHS = 150` | 15 |
| how long it runs | "This example takes about 100 epochs in total"; "the final validation loss should end up around 0.44 after ±90 epochs" | 15, 16 |
| widths | `memory_units = 128`, `dense_units = 128` | 12 |

`scripts/validate_valendin_lstm.py` already encodes exactly this recipe and reproduces the
notebook's published validation loss, so it is known to train correctly in this codebase.
It has never been used for a study.

## The parameters to run

**ValendinLSTM** (architecture frozen by ADR-0004, so only the recipe is set):

```python
search_space = {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32}
training     = {"n_epochs": 150, "patience": 5, "loss_type": "cross_entropy"}
n_trials     = 1
```

**LSTM** (`MultinomialLSTMModel`, at the notebook's widths):

```python
search_space = {"embedder": "valendin", "lstm_hidden_size": 128, "dense_units": 128,
                "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 0.0,
                "batch_size": 32}
training     = {"n_epochs": 150, "patience": 5, "loss_type": "cross_entropy"}
n_trials     = 1
```

## Notes that matter when reading the result

- A scalar in `search_space` is pinned by the registry's spec mini-language, so **batch 32
  is reachable without touching the registry**, even though the searched set is
  `{64, 128, 256}`. Whether to add 32 to that set is ticket 06's decision, not this one's.
- `weight_decay=0.0` makes AdamW identical to the notebook's Adam.
- `embedding_dim` is not sampled when `embedder="valendin"` — every feature keeps its own
  sqrt(cardinality)+1 vector.
- **The validation split stays temporal** (ADR-0001). The notebook holds out a random 10%
  of customers; this arm is about the training recipe, not the split, and mixing the two
  would make the result unattributable.
- **`L-paper` is not the published architecture.** `MultinomialLSTMModel` adds LayerNorm,
  a projection to a common width and a summed context (see `benchmarks/valendin_lstm.py`).
  It is our model at the paper's widths and recipe; `VL-paper` is the published one.

## Measured before launch (2026-09-20): the recipe alone is not enough

Three real `paper` studies were run on the workstation before renting anything
(`Studies/training_budget__ValendinLSTM__electronics__paper__r0{0,1,2}`):

| replication | best epoch | val CE | bias % | MAPE |
| --- | ---: | ---: | ---: | ---: |
| r00 | 1 | 0.09304 | +44.6 | 78.3 |
| r01 | 1 | 0.09337 | +42.7 | 70.8 |
| r02 | 1 | 0.09337 | +20.2 | 69.0 |

**The paper's recipe stops at epoch 1 on electronics** — 7 epochs trained, weights from
epoch 1 — and lands *worse* than `archive` (0.0886). Patience 5 at batch 32 fires on the
plateau even sooner in epochs than patience 7 at batch 256 does.

So the recipe's settings do not reproduce the recipe's *training*. The notebook reports
~90 epochs on its own data; ours quits at 1 because the electronics validation curve is
flat from the start (`docs/training-budget.md` §2). Copying the settings copies the
stopping rule too, and the stopping rule is the thing under investigation.

That is what the **`paper90`** arm is for: the same pinned recipe with `min_epochs=90`,
the notebook's own figure. It is the only arm that trains the paper's model for the
paper's number of epochs. `paper` stays in the design as the literal reading — the two
together separate "the settings" from "the training".
