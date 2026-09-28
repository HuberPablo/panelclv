| Tree | Hyperparameter | ρ with \|bias\|, pooled (descriptive) | ρ with \|bias\|, mean within-cell [95% CI], cells | ρ with MAPE, pooled (descriptive) | ρ with MAPE, mean within-cell [95% CI], cells |
| --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | `batch_size` | +0.48 | +0.05 [−0.15, +0.26], 12 | +0.49 | +0.07 [−0.13, +0.28], 12 |
| LSTM `no_ar` | `learning_rate` | −0.15 | +0.12 [−0.09, +0.35], 16 | −0.16 | +0.12 [−0.10, +0.35], 16 |
| LSTM `no_ar` | `lstm_hidden_size` | −0.20 | +0.11 [−0.09, +0.30], 15 | −0.21 | +0.08 [−0.11, +0.25], 15 |
| LSTM `no_ar` | `dense_units` | +0.28 | −0.02 [−0.24, +0.18], 15 | +0.28 | −0.03 [−0.24, +0.17], 15 |
| LSTM `no_ar` | `dropout` | −0.05 | +0.14 [−0.06, +0.32], 16 | −0.05 | +0.12 [−0.07, +0.30], 16 |
| LSTM `ar_bounded` | `batch_size` | +0.07 | +0.07 [−0.09, +0.23], 13 | +0.06 | +0.10 [−0.04, +0.26], 13 |
| LSTM `ar_bounded` | `learning_rate` | −0.02 | +0.10 [−0.02, +0.21], 16 | −0.10 | +0.03 [−0.06, +0.12], 16 |
| LSTM `ar_bounded` | `lstm_hidden_size` | 0 | +0.04 [−0.13, +0.20], 16 | −0.04 | +0.08 [−0.10, +0.25], 16 |
| LSTM `ar_bounded` | `dense_units` | −0.28 | −0.15 [−0.34, +0.02], 15 | −0.22 | **−0.20 [−0.37, −0.02], 15** |
| LSTM `ar_bounded` | `dropout` | −0.02 | +0.09 [−0.08, +0.25], 16 | +0.03 | +0.10 [−0.08, +0.27], 16 |
| Transformer `no_ar` | `batch_size` | −0.04 | −0.07 [−0.26, +0.14], 13 | −0.03 | −0.05 [−0.25, +0.17], 13 |
| Transformer `no_ar` | `learning_rate` | −0.09 | −0.03 [−0.18, +0.12], 16 | +0.05 | +0.03 [−0.14, +0.20], 16 |
| Transformer `no_ar` | `d_model` | −0.01 | not tested: varies in 2 of 16 cells | −0.03 | not tested: varies in 2 of 16 cells |
| Transformer `no_ar` | `nhead` | +0.08 | +0.03 [−0.14, +0.18], 16 | +0.05 | +0.02 [−0.15, +0.19], 16 |
| Transformer `no_ar` | `num_encoder_layers` | +0.24 | **+0.20 [+0.08, +0.31], 16** | +0.29 | **+0.22 [+0.13, +0.32], 16** |
| Transformer `no_ar` | `dropout` | +0.17 | +0.08 [−0.05, +0.22], 16 | +0.13 | +0.09 [−0.05, +0.23], 16 |
| Transformer `ar_bounded` | `batch_size` | +0.03 | +0.07 [−0.09, +0.21], 16 | +0.04 | +0.04 [−0.12, +0.19], 16 |
| Transformer `ar_bounded` | `learning_rate` | −0.05 | −0.14 [−0.30, +0.03], 16 | +0.13 | −0.07 [−0.24, +0.09], 16 |
| Transformer `ar_bounded` | `d_model` | −0.13 | not tested: varies in 1 of 16 cells | −0.13 | not tested: varies in 1 of 16 cells |
| Transformer `ar_bounded` | `nhead` | −0.05 | −0.04 [−0.20, +0.11], 16 | −0.01 | −0.06 [−0.22, +0.08], 16 |
| Transformer `ar_bounded` | `num_encoder_layers` | −0.01 | +0.01 [−0.17, +0.20], 16 | −0.10 | +0.02 [−0.15, +0.21], 16 |
| Transformer `ar_bounded` | `dropout` | −0.01 | −0.05 [−0.23, +0.12], 16 | +0.03 | −0.07 [−0.23, +0.11], 16 |
