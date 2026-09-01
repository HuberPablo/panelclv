# Backpropagation

How a gradient gets from the loss back to the weights in `panelclv`: what the autograd
graph is built over, what the five selectable losses differentiate, where the graph is
deliberately cut, and which periods are allowed to move a weight at all.

Read `CONTEXT.md` first for the vocabulary (*panel*, *period*, *calibration window*,
*validation window*, *holdout window*, *rollout*, *refit*). This document assumes it,
and assumes the categorical-head contract `CLAUDE.md` states: logits `(B, T, K)`, a
transaction count as a class and never a quantity, cross-entropy on a class index,
evaluation by sampling-and-averaging.

Everything below was **read from the source**, not measured. No claim is made here
about gradient magnitudes, convergence speed or training stability — see §9. Line
numbers were resolved against the working tree at the time of writing; where a shape is
concrete rather than symbolic it comes from the golden fixture in
`tests/test_golden_end_to_end.py`, by way of `docs/running-a-model.md`.

This document expands one bullet. `docs/running-a-model.md:363` says "Gradients flow
back through all 38 steps" and stops there. Everything from §4 on is what that sentence
was compressing.

## Contents

1. [The idea in plain terms](#1-the-idea-in-plain-terms)
2. [What the graph is built over](#2-what-the-graph-is-built-over)
3. [What the loss actually measures](#3-what-the-loss-actually-measures)
4. [The backward pass, line by line](#4-the-backward-pass-line-by-line)
5. [Where the graph stops: the rollout is not differentiated](#5-where-the-graph-stops-the-rollout-is-not-differentiated)
6. [Which periods are allowed to move a weight](#6-which-periods-are-allowed-to-move-a-weight)
7. [Reference: the gradient-flow table](#7-reference-the-gradient-flow-table)
8. [Invariants a new model must preserve](#8-invariants-a-new-model-must-preserve)
9. [What this document does not establish](#9-what-this-document-does-not-establish)

---

## 1. The idea in plain terms

A model is a pile of numbers — the **weights** — and a recipe for turning inputs into
outputs using them. The forward pass runs that recipe and then compresses everything it
produced into a single number, the **loss**, which is low when the model was right and
high when it was wrong.

Backpropagation answers exactly one question about that number:

> If I nudge this one weight by a hair, does the loss go up or down, and how fast?

That quantity is the weight's **gradient**, written `∂L/∂θ`. Collect it for every weight
and you have a direction in which the whole pile should move. An optimiser then takes a
small step in the opposite direction — downhill — and the model is slightly less wrong
than it was.

**The reason it is called *backward* is arithmetic, not metaphor.** The recipe is a
chain of simple operations, each feeding the next: embed, multiply, add, squash, score.
The chain rule says the derivative of the whole chain is the product of the derivatives
of its links. Computing that product from the far end — from the loss, backwards toward
the inputs — lets every link reuse the result its successor already computed. Computing
it from the front instead would redo that work once per weight. The saving is the entire
reason deep learning is affordable.

To make this possible, PyTorch records the chain as it happens. Every operation on a
tensor that requires a gradient appends a node to a **graph**, storing what was done and
which intermediate values the reverse pass will need. `loss.backward()` walks that graph
from the loss back to the leaves, filling in each parameter's `.grad`, and then discards
it. The next forward pass builds a fresh one.

Three consequences that the rest of this document is really about:

- **The graph only contains what the forward pass actually touched.** An operation done
  in numpy, or under `torch.no_grad()`, leaves no trace and can never receive a gradient.
- **Sampling is not an operation the chain rule can cross.** Drawing a category from a
  distribution is a discontinuous jump; there is no derivative of "which class came out"
  with respect to the probabilities, unless you build one deliberately.
- **What you differentiate is what you get.** The model improves at the quantity in the
  loss, and at nothing else.

That last point is where this package gets interesting, so it is worth stating up front:
**the numbers this project is judged on are never differentiated.** `rmse`,
`bias_percent` and `mape_aggregate` are computed downstream of a sampler, in numpy, long
after the graph is gone (§5). The network is trained as a per-cell classifier and only
then pressed into service as a forecaster.

---

## 2. What the graph is built over

### A batch is customers × periods

`trials/loaders.py` is where numpy becomes tensors. A batch is a plain 2-tuple:

| tensor | shape | dtype | meaning |
| --- | --- | --- | --- |
| `samples` | `(B, T, F)` | float32 | B customers, T periods, F feature channels |
| `targets` | `(B, T)` | int64 | the class index of the *next* period's transaction count |

**B is customers and T is periods, and only one of those two axes is ever shuffled.**
The training loader is built with `shuffle=shuffle_train`, which permutes rows — that
is, customers. Time is never permuted; each customer's history sits intact inside its
own row. This is what makes a recurrent unroll meaningful at all.

### Teacher forcing is two lines of slicing

There is no teacher-forcing *flag* anywhere in the package. It is structural, and it is
built in `data_preparation/panel_dataset.py:936-937`:

```python
    samples     = calibration[:, :-1, :]
    targets     = calibration[:, 1:, target_idx:target_idx + 1]
```

Two views of one array, offset by a single period. `samples` drops the last period;
`targets` drops the first and keeps only the target channel. The consequence is the whole
autoregressive construction: **the transaction count is both an input and a label.** The
row for period *t* carries the true count at *t* and is asked for the count at *t+1*.

So during training the model is fed ground truth at every position and never sees its own
output. Its errors cannot accumulate. That is a different regime from the rollout, and
the gap between them is the exposure bias discussed in `docs/loss-functions.md` §4.3.

### There is no padding and no loss mask

**A repo-wide search for `ignore_index`, `pad_sequence` or `key_padding_mask` returns
nothing.** The only `masked_fill` in the package is the Transformer's causal attention
mask, which is not a loss mask.

This is deliberate and it is bought upstream. `prepare_dataset` refuses to build a
dataset whose customers have unequal period counts
(`data_preparation/panel_dataset.py:846-858`):

```python
    if train_counts.nunique() != 1:
        raise ValueError(
            "Customers have inconsistent training-period counts:\n"
            f"{train_counts.value_counts()}"
        )
```

Because a panel is rectangular by definition — one row per customer per period, whether
or not they transacted — every sequence is the same length and the tensor is dense. What
that buys the backward pass is that **every cell of `(B, T)` is a real observation**. No
position has to be excluded from the mean, so nothing has to be masked, so nothing can
be masked *wrongly*. The one thing that does get excluded — the validation window — is
excluded by slicing the tensor rather than by masking the loss (§6).

### The shape chain, embedder to head

Both developed architectures agree on the endpoints and differ only in the middle.
Taking the LSTM (`models/multinomial_lstm.py:125-138`):

```python
    def forward(self, x: torch.Tensor, state=None):
        # The embedder turns (B, T, F) into (B, T, embedder.output_dim); which
        # features were summed, concatenated or projected is its business.
        encoded_input = self.embedder(x)

        # lstm_out: (B, T, lstm_hidden_size). `state` is the LSTM recurrent
        # (hidden, cell) state, threaded across autoregressive rollout steps.
        lstm_out, state = self.lstm(encoded_input, state)
        lstm_out = self.dropout(lstm_out)

        dense_out = self.dense(lstm_out)
        logits = self.output_layer(dense_out)

        return logits, state
```

| # | Operation | Shape after | What it means |
| --- | --- | --- | --- |
| 1 | input batch | `(B, T, F)` | every channel float32, including the categorical ones |
| 2 | `embedder(x)` | `(B, T, output_dim)` | categoricals looked up, covariates carried through |
| 3 | `nn.LSTM` / encoder stack | `(B, T, H)` or `(B, T, d_model)` | one vector per customer per period |
| 4 | dropout, dense | `(B, T, dense_units)` | LSTM only; the Transformer normalises instead |
| 5 | `output_layer` | `(B, T, K)` | raw logits — one score per count class |

**Step 5 produces raw scores, never probabilities.** `softmax` appears nowhere in the
training path; it is folded into the criterion at train time and applied explicitly only
at rollout time. This matters for the backward pass because the fused
log-softmax-plus-NLL used by `nn.CrossEntropyLoss` is both numerically stabler and has a
famously clean derivative — see §3.

`nn.LSTM` is constructed with `batch_first=True` and consumes all T periods in one call.
The recurrent state is created as zeros, threaded internally across the T positions, and
discarded on return: the training wrapper drops it with `logits, _`. Nothing carries
between batches, because each customer's whole sequence is already inside its own row.

### K is not the model's choice

`K = num_target_classes` is read off the embedder, never computed by the model. The tie
is enforced structurally in `Embedder.__init__` (`models/embedders.py:87-96`):

```python
        if target_col not in embedded_cols:
            raise ValueError(
                f"target_col {target_col!r} must appear in embedded_cols "
                f"(its cardinality drives the output head size)"
            )
        ...
        self.num_target_classes: int = int(embedded_cols[target_col])
```

**One integer sizes both ends of the network.** The same `K` is the number of rows in the
target column's `nn.Embedding` on the input side and the width of the final `nn.Linear`
on the output side. A model in which those disagree is not constructible — which is the
`CLAUDE.md` contract, enforced rather than merely documented.

Upstream, `K` is the observed maximum count plus one, capped by `clip_target_upper` when
set. It is small: 5 on CDNOW, 7 on the electronics panel (`docs/loss-functions.md` §2).
That smallness is why a categorical head over counts is tractable at all.

---

## 3. What the loss actually measures

`docs/loss-functions.md` is the authority on *why* these losses and what they cost the
forecast — the four-constraint contract C1–C4, the properness argument, the measured
class statistics. This section covers only what is needed to see the gradient, and defers
everything else to it.

### The reshape is where B and T stop mattering

`training/loop.py:106`:

```python
        loss = criterion(output.reshape(-1, num_target_classes), targets.reshape(-1))
```

Logits `(B, T, K)` become `(B*T, K)`; targets `(B, T)` become `(B*T,)`. Every criterion in
the package takes that flat pair — the contract is stated at the top of
`models/losses.py`:

```
All criterions take (logits, targets) — same call signature as
`nn.CrossEntropyLoss`, with shapes `(B*T, K)` and `(B*T,)` (long indices) —
so they're drop-in replacements inside `training.loop.fit_model`.
```

**The consequence for the gradient is that a (customer, period) cell is the unit of
account.** The loss is a plain mean over all `B*T` cells: a customer with a long history
contributes proportionally more than a short one only because it has more cells, and an
early period counts exactly as much as a late one. No position is weighted, skipped or
discounted.

### The five losses and what each differentiates

Selected by the `loss_type` string and built once per fit by `build_criterion`
(`models/losses.py:220-251`). Writing `q = softmax(logits)` and `y` for the true class:

| `loss_type` | Loss per cell | Gradient reaches the logits through |
| --- | --- | --- |
| `cross_entropy` | `−log q_y` | the fused log-softmax; `∂L/∂logits = q − onehot(y)` |
| `weighted_ce` | `w_y · (−log q_y)`, normalised by the summed weight | as above, scaled per class by a constant `w_y` |
| `focal` | `α_y · (1 − q_y)^γ · (−log q_y)` | `log_softmax`, plus the `(1−q_y)^γ` factor, itself a function of `q` |
| `emd` | `Σ_k (F_q(k) − 1{y ≤ k})²` | `softmax` then `cumsum` only |
| `ce_emd` | `CE + λ · EMD` | both paths, summed |

Three of those rows deserve a closer look.

**Cross-entropy has the derivative that makes everything else cheap.** Because the
softmax is fused into the criterion rather than applied in the model, the gradient
arriving at the logits is just `q − onehot(y)` — predicted probability minus truth. It is
bounded, it is zero exactly when the model is right, and it needs no special handling.
Every other loss here is a modification of that signal.

**The EMD loss differentiates a cumulative distribution** (`models/losses.py:85-91`):

```python
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)
        target_onehot = F.one_hot(targets, num_classes=K).float()
        cdf_pred = probs.cumsum(dim=-1)
        cdf_true = target_onehot.cumsum(dim=-1)
        return ((cdf_pred - cdf_true) ** 2).sum(dim=-1).mean()
```

`target_onehot` is built from integer labels and is a **constant** as far as autograd is
concerned — it has no gradient and never did. The whole gradient path is
`softmax → cumsum → square → sum`. The `cumsum` is the interesting link: because class
`k`'s probability appears in every cumulative term from `k` onward, an error at one class
propagates a gradient to all the classes below it. That is precisely the ordinal
behaviour the loss exists for — predicting 0 when the truth is 10 is punished far harder
than predicting 0 when the truth is 1 — and it is why plain cross-entropy, which treats
the classes as unordered names, cannot express it.

**`ce_emd` is a sum, so its gradient is a sum** (`models/losses.py:129-130`):

```python
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.emd_weight * self.emd(logits, targets)
```

Both terms reach the same logits; autograd accumulates the two contributions into one
`.grad`. `λ = 0` recovers cross-entropy bit for bit, which the test suite pins with
`torch.equal` rather than an approximate comparison. `λ` itself is tuned — it is declared
in the registry search space and resolved by Optuna like any other hyperparameter, so a
study chooses how ordinal it wants to be.

One thing the table does not say and should: **when `loss_type="ce_emd"`, class weights
are silently not applied.** `build_criterion` does not forward `class_weights` into that
branch, so the CE term there is unweighted even if weights were computed. This is a real
behavioural detail, not a bug to fix in passing; it is recorded here because a reader
comparing `weighted_ce` against `ce_emd` would otherwise assume they share a term.

### Class weights are a constant, not a parameter

`compute_class_weights` (`models/losses.py:208-212`) is inverse frequency, renormalised
to sum to `K`:

```python
    t = torch.as_tensor(targets).flatten().long()
    counts = torch.bincount(t, minlength=num_classes).float().clamp(min=1.0)
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return weights
```

It runs **once, before training**, from label counts. It carries no gradient and appears
in the graph only as a constant multiplier. `FocalLoss` stores its `alpha` as a
registered buffer for the same reason: buffers move with `.to(device)` but are not
parameters and are never optimised.

Note the default `training_only=True`, which slices the labels to the training-window
prefix before counting. The validation window's class mix must not influence the
training signal, for the same reason it must not influence the weights themselves
(ADR-0001).

---

## 4. The backward pass, line by line

Every gradient in this package is produced by one function. `train_one_epoch`
(`training/loop.py:100-110`) is the only place `backward()` is called anywhere in
`src/panelclv/`; `refit_full_calibration` reuses it unchanged, so there is exactly one
implementation to understand:

```python
        optimizer.zero_grad(set_to_none=True)
        output = model(samples)
        # Some training wrappers return (logits, _) — be robust to that.
        if isinstance(output, tuple):
            output = output[0]

        loss = criterion(output.reshape(-1, num_target_classes), targets.reshape(-1))
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
```

**`zero_grad(set_to_none=True)`.** PyTorch *accumulates* into `.grad` rather than
overwriting it, so last batch's gradients must be cleared or they would be added to this
batch's. `set_to_none=True` discards the buffers outright instead of filling them with
zeros: it is cheaper, it frees the memory, and it makes a parameter that received no
gradient at all distinguishable (its `.grad` is `None`, not a zero tensor). The optimiser
skips such parameters rather than applying weight decay to them.

**The forward call builds the graph as a side effect.** There is no separate "record"
step. By the time `model(samples)` returns, every operation from the embedding lookup to
the final `nn.Linear` has appended a node, and the intermediate activations the reverse
pass will need are being held in memory. This is why `seq_len` costs memory as well as
compute: the graph holds T periods' worth of LSTM intermediates for every customer in the
batch.

**`loss.backward()` is one call and it spans everything.** All T periods are unrolled
inside a single `nn.LSTM` invocation, so a single backward walks the recurrence from
period T−1 back to period 0 — full backpropagation-through-time, no truncation. This is
worth stating flatly because it is the sort of thing usually configured: **there is no
`detach()` anywhere in the training path.** A search across `training/`, `trials/` and
`data_preparation/` for `detach()` returns nothing. Nothing cuts the recurrence into
windows, and no state is carried between batches to be cut.

**`clip_grad_norm_` rescales, it does not clip elementwise.** It computes one global
L2 norm across every parameter's gradient and, if that norm exceeds `max_norm`, scales
*all* of them down by the same factor. The direction of the update is preserved exactly;
only its length is capped. Its position is the textbook one and the only correct one —
after `backward()`, so there is something to clip, and before `step()`, so the optimiser
sees the clipped values. The default is `grad_clip=1.0`.

**`optimizer.step()` applies AdamW** (`training/loop.py:261-263`):

```python
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
```

`model.parameters()` means *every* parameter, embedding tables included. **Nothing is
frozen anywhere in the package**: no `requires_grad = False`, no parameter groups, no
layer-wise rates. The warm-started refit (ADR-0008) loads weights but freezes nothing.

Two properties of this call that the document should not leave implicit. AdamW is
*decoupled* weight decay — the decay is applied directly to the weights rather than added
to the gradient, so it does not interact with Adam's per-parameter scaling. And **there
is no learning-rate schedule.** A search for `scheduler` or `lr_scheduler` across the
training path returns nothing; the rate is constant from the first batch to the last. The
learning rate and weight decay are instead chosen per trial by Optuna over the ranges the
registry declares (`registry/model_registry.py:337-339`): `learning_rate` log-uniform on
`(1e-4, 3e-3)`, `weight_decay` log-uniform on `(1e-6, 1e-2)`.

One optimiser step per batch. No gradient accumulation.

### How the gradient reaches the features

Below the backbone, the route splits by embedder, and the two available strategies give
covariates very different treatment (ADR-0005).

Under **`ValendinEmbedder`** — the registry default for both developed models — the
forward is pure concatenation (`models/embedders.py:244-257`):

```python
        chunks = [
            self._emb_modules[self._emb_index[col]](x[:, :, i].long())
            if col in self.embedded_cols
            else x[:, :, i : i + 1].float()
            for i, col in enumerate(self.seq_cols)
        ]
        return torch.cat(chunks, dim=-1)
```

**A covariate under this strategy has no learnable parameter of its own.** It contributes
one raw channel to the concatenated vector, untouched. Its only gradient path is the
corresponding column of the LSTM's input-to-hidden matrix `W_ih`. That single fact is why
numeric channels must be standardised before they ever reach the model: the column's
contribution to the pre-activation sum, and therefore the gradient arriving at its
weights, scales with the column's raw magnitude. `docs/feature_engineering.md` develops
that argument, including why the following `LayerNorm` cannot repair it; this document
does not restate it.

Under **`ProjectedEmbedder`** the covariates get a shared `Linear` of their own and the
embedded columns are summed into a common representation, so the gradient has a richer
route. Which embedder a model uses is part of its identity and is itself searchable.

**The previous transaction count reaches an embedding *row*, not a weight.** Because the
target column is required to be embedded, its channel is cast with `.long()` and used as
an index. The backward pass of an embedding lookup is a scatter-add: only the rows for
count classes that actually appeared in the batch receive any gradient at all. On these
panels, where the count distribution is dominated by zero, the row for class 0 is updated
on essentially every batch and the row for the largest class may go many batches
untouched.

### What is deliberately outside the graph

Two things inside the training path never receive a gradient, both on purpose.

The accuracy bookkeeping is fenced off (`training/loop.py:115-118`):

```python
        with torch.no_grad():
            preds = output.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()
```

`argmax` has no useful derivative, and this is diagnostics — it runs after `step()` and
must not extend the graph.

The Transformer's positional encoding is a **registered buffer**, not a parameter
(`models/multinomial_transformer.py:83`): `self.register_buffer("pe", pe.unsqueeze(0))`.
The sinusoids are fixed by construction, move with the model across devices, appear in no
optimiser, and are never learned.

---

## 5. Where the graph stops: the rollout is not differentiated

**A forecast in this package is produced entirely outside autograd.** Not "mostly" — the
chain is broken in three independent places, any one of which would be sufficient on its
own.

**First, the simulators run under `torch.inference_mode()`**
(`models/monte_carlo_forecasting.py:155`, and again at `:257` for the attention path):

```python
    with torch.inference_mode():
        # Step 1: warmup → its last-position sample IS the holdout step 0 forecast,
        # and `state` now summarises the whole calibration window.
        out, state = model(calib_tensor, state=None)
        previous_sample    = out[:, -1, 0]                    # (N,)
        sampled_path[:, 0] = previous_sample
```

`inference_mode` is stronger than `no_grad`: it disables version-counter tracking as
well, and tensors produced inside it are permanently barred from a later graph. No
operation in the warm-up or the step loop is recorded.

**Second, the sample is a hard draw.** The rollout model's forward
(`models/multinomial_lstm.py:223-227`, and identically in the other two):

```python
    def forward(self, x: torch.Tensor, state=None):
        logits, state = self.backbone(x, state)
        probs = torch.softmax(logits, dim=-1)
        sample = dist.Categorical(probs=probs).sample().unsqueeze(-1).float()
        return sample, state
```

`.sample()` is not reparameterised. There is no straight-through estimator and no
Gumbel-softmax anywhere in the repository. Even with the graph enabled, a gradient could
not cross this line: the output is an integer class drawn from `probs`, and "which class
came out" is not a differentiable function of `probs`.

**Third, the AR features leave torch entirely.** At every rollout step the derived
covariates are recomputed from the *sampled* history, and that recomputation happens in
numpy (`models/monte_carlo_forecasting.py:170`):

```python
                feats = ar_state.update(previous_sample.detach().cpu().numpy())
```

The path average is numpy too — simulated paths are stacked and meaned as `np.float64`
arrays, and `compute_forecast_metrics` never sees a tensor.

### What follows from that

**The loss is not scored on the thing it trains.** The network is optimised to be a
well-calibrated per-cell classifier; the forecast is what a non-differentiable sampler
does with that classifier afterwards. `rmse`, `bias_percent` and `mape_aggregate` — the
numbers the thesis reports — have no gradient and could not be optimised directly even
in principle without replacing the sampler.

That gap is the motivation for the whole loss-selection question. Since the rollout's
expected count is `E_q[y] = Σ_k k · q_k`, the forecast is only unbiased if `q` is an
honest estimate of the true class distribution — which is why the loss must stay strictly
proper, and why weighting or focal reshaping, both of which deliberately distort `q`,
cost forecast accuracy even when they improve classification metrics. That argument
belongs to `docs/loss-functions.md`; what this document adds is the mechanical reason it
cannot be sidestepped by simply differentiating the metric instead.

It is also why ADR-0003, which tried to select trials on rollout quality, was a search
concern rather than a gradient one — Optuna can score a non-differentiable objective
where backpropagation cannot. That decision has since been retired.

**One shared-weights subtlety.** `to_rollout()` hands over the *same* backbone rather
than a copy (ADR-0007), and the docstring at `models/multinomial_lstm.py:175-191` states
the consequence plainly: the simulator calls `.eval()` on the rollout model, which puts
the trained model's backbone in eval too. Every current caller hands the rollout model on
and stops using the trained one, so nothing is surprised by it — but resuming training
after `to_rollout()` would need an explicit `.train()`. Dropout would otherwise stay off
and the gradients would be computed under a different network than the one intended.

---

## 6. Which periods are allowed to move a weight

The validation window must never update weights (ADR-0001, and `CONTEXT.md` under
**Validation window**). The package enforces this twice over, and the stronger of the two
mechanisms is the one that never involves the loss at all.

**The training loader is physically truncated** (`trials/loaders.py:104-109`):

```python
    X = data["samples"]                                 # (N, T-1, F) float32
    y = data["targets"].squeeze(-1).astype(np.int64)    # (N, T-1) class indices

    # Train on the prefix transitions only (targets at periods 1..s-1); validate on the
    # full sequence but score only the suffix (see recipe["val_score_start"]).
    X_train, y_train = X[:, : s - 1], y[:, : s - 1]
```

Validation-window periods are not masked out of the loss — they are **not in the tensor**.
They never enter a forward pass, never appear in the graph, and cannot contribute a
gradient by any route. This is a categorically stronger guarantee than zeroing their
loss contribution would be, and it is worth preferring for exactly that reason.

**Validation slices after the forward pass instead** (`training/loop.py:161-170`):

```python
            # Keep only the validation-window steps (the suffix). The prefix steps
            # were warm-up context for the state, not part of the validation score.
            if val_score_start:
                output = output[:, val_score_start:]
                targets = targets[:, val_score_start:]

            loss = criterion(output.reshape(-1, num_target_classes), targets.reshape(-1))
```

The asymmetry is deliberate. The validation loader feeds the **full** sequence so the
recurrent state is warmed on the training prefix — a model scored on the validation
window from a cold state would be scored on a handicap it never has in practice — but
only the suffix positions are averaged into the number. The pass runs inside
`torch.inference_mode()` and under `model.eval()`, so it carries no gradient regardless
of the slicing; the slicing is about what is *measured*, not about what is learned.

**Selection happens on that number, and the chosen weights are put back**
(`training/loop.py:329-334`, then `:355-361`):

```python
        improved = (val_metrics["loss"] + 1e-4) < best_val_loss
        if improved:
            ...
            best_state = copy.deepcopy(model.state_dict())
```

```python
    torch.save(best_state, checkpoint_path)
    # Put the selected weights back into the object as well as onto disk. Patience
    # means the loop keeps training past its best epoch by design, so without this
    # the returned model holds the LAST epoch's weights while its own checkpoint
    # holds the best ones — two answers to one question. `to_rollout()` reads the
    # object (ADR-0007), so the difference is a quietly wrong forecast.
    model.load_state_dict(best_state)
```

Selection is on validation loss with a `1e-4` minimum improvement; accuracy and weighted
F1 are recorded but never select. Because patience means training deliberately continues
past the best epoch, the last epoch's weights are usually *not* the selected ones, and
the reload is what keeps the object and its checkpoint from disagreeing.

**The refit is the exception that proves the rule.** `refit_full_calibration` (ADR-0008)
trains on every transition, validation window included, using `refit_loader` — which
applies no truncation. It has no validation pass, no early stopping and no best-state
selection; it runs a fixed number of epochs from a warm start and saves the final
weights. This is legitimate precisely because selection has already happened: the study
chose the architecture and the stopping point using a held-out window, and the refit then
lets the weights also learn the most recent periods instead of merely conditioning on
them at forecast time. The holdout window is untouched throughout.

---

## 7. Reference: the gradient-flow table

Every stage a tensor passes through, from panel to reported metric.

| Stage | In the graph? | Where |
| --- | --- | --- |
| Panel → `(N, T, F)` tensors, AR features, standardisation | no — numpy/pandas, no torch import | `data_preparation/panel_dataset.py`, `ar_features.py` |
| Class-weight statistics | no — computed once, a constant | `models/losses.py:208-212` |
| Embedding lookup (target and other categoricals) | **yes** — scatter-add into the rows used | `models/embedders.py` |
| Covariate channel, `ValendinEmbedder` | **yes**, but only via the backbone's `W_ih` — no own parameter | `models/embedders.py:244-257` |
| Covariate projection, `ProjectedEmbedder` | **yes** — shared `Linear` + `LayerNorm` | `models/embedders.py:154-158` |
| LSTM unroll / Transformer encoder stack | **yes** — full BPTT over all T periods | `models/multinomial_lstm.py:132`, `models/multinomial_transformer.py:184` |
| Positional encoding | no — a registered buffer, not a parameter | `models/multinomial_transformer.py:83` |
| Causal attention mask | no — a constant `(T, T)` additive mask | `models/multinomial_transformer.py:158-162` |
| Output head → logits `(B, T, K)` | **yes** | `models/multinomial_lstm.py:136` |
| Reshape to `(B*T, K)` and loss reduction | **yes** — the root of the graph | `training/loop.py:106` |
| Gradient clipping, optimiser step | acts *on* gradients, not part of the graph | `training/loop.py:108-110` |
| Accuracy / F1 bookkeeping | no — `argmax` under `no_grad` | `training/loop.py:115-118` |
| Validation pass | no — `eval()` + `inference_mode()` | `training/loop.py:145`, `:153` |
| Rollout warm-up and step loop | no — `inference_mode()` | `models/monte_carlo_forecasting.py:155`, `:257` |
| Categorical sampling | no — a hard, non-reparameterised draw | `models/multinomial_lstm.py:226` |
| AR-feature recomputation during a rollout | no — `.detach().cpu().numpy()` | `models/monte_carlo_forecasting.py:170` |
| Averaging simulated paths | no — numpy | `models/monte_carlo_forecasting.py:400-401` |
| `rmse`, `bias_percent`, `mape_aggregate` | no — numpy, the graph is long gone | `models/monte_carlo_forecasting.py` → `compute_forecast_metrics` |

Read down the "in the graph?" column and the shape of the project appears: a narrow band
of differentiable computation between two wide non-differentiable ones. Features are
built before it; forecasts and scores are built after it.

---

## 8. Invariants a new model must preserve

Adding a model means adding one registry entry (ADR-0006). These are the properties that
entry's builder must satisfy for the single training loop to work on it.

1. **`forward` returns raw logits shaped `(B, T, K)`.** No softmax in the training path —
   the criterion applies it. A model that returns probabilities will be scored as if they
   were logits and will train to nonsense.
2. **`K` comes from the embedder, never recomputed.** Read `embedder.num_target_classes`.
   The same integer must size the target's embedding table and the output head; the
   `Embedder` base class already refuses any other arrangement.
3. **The criterion contract is flat.** Criterions receive `(B*T, K)` against `(B*T,)`,
   and `training/loop.py:106` is what makes that true. This reshape is load-bearing, and
   here is the trap: `FocalLoss` indexes with `gather(1, ...)` at `models/losses.py:63`
   while the line above it uses `dim=-1`. That is correct only for a flat 2-D input. A
   `(B, T, K)` tensor handed to it directly would gather along the period axis and
   silently return a plausible-looking wrong number rather than raising. Any new call
   site must reshape first.
4. **A rollout must be reconstructible from sampled history.** Every feature the model
   reads at holdout step *t* must be genuinely known in advance or computable from the
   model's own samples — otherwise the rollout cannot be run without leaking the answer.
   `docs/feature_engineering.md` is the authority; the training-time consequence is that
   AR features are precomputed constants from the true past, and their forecast-time
   definitions must match exactly.
5. **`to_rollout()` shares the backbone.** The trained model hands over its weights rather
   than being copied beside one (ADR-0007). Sharing means the simulator's `.eval()`
   applies to the trained model too, so any code that trains after calling `to_rollout()`
   must call `.train()` explicitly.
6. **A model that cannot be trained by `train_one_epoch` unchanged does not fit.** There
   is one backward pass in the package. A model needing gradient accumulation, a
   learning-rate schedule, frozen stages or truncated BPTT would require changing the
   shared loop, which is a design decision and belongs in an ADR — not in a builder.

---

## 9. What this document does not establish

Everything above was read from the source and from the existing documentation. Nothing
here was measured.

- **No claim is made about gradient magnitudes, vanishing or exploding gradients, or
  training stability** on either panel. The standardisation argument in §4 is a statement
  about what the arithmetic implies, taken from `docs/feature_engineering.md`; it is not
  a measurement of observed gradient norms. Whether `grad_clip=1.0` ever actually binds
  during a run is **UNVERIFIED**.
- **No claim is made about convergence** — how many epochs a fit typically runs, how often
  early stopping fires before the epoch budget, or how far past the best epoch patience
  usually carries training.
- **The concrete shapes** (`(8, 38, 5)`, a 52-period warm-up, `K = 5`) come from the
  golden fixture and the two configured panels by way of `docs/running-a-model.md` and
  `docs/loss-functions.md`. They are illustrative sizes, not properties of the design.
- **Line numbers drift.** They were resolved against the working tree at the time of
  writing. `docs/loss-functions.md` already carries stale ones — its loss table predates
  the insertion of `CrossEntropyPlusEMDLoss` — so treat any number here as a pointer to a
  named construct, and trust the name over the number.
- **The `ce_emd` weighting behaviour noted in §3** (class weights not forwarded into that
  branch) is read from `build_criterion`'s dispatch. Whether that is intended or merely
  untested is not established here.
- **The `FocalLoss` `gather(1, ...)` fragility in §8** is a latent hazard, not an observed
  failure. Every current call site reshapes first, so no run has been affected.
