| Tree | Hyperparameter | ρ with \|bias\|, pooled | ρ with \|bias\| within cells: positive / negative of cells, range | ρ with MAPE, pooled | ρ with MAPE within cells: positive / negative of cells, range |
| --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | `batch_size` | +0.48 | 5 / 7 of 12, −0.63 to +0.62 | +0.49 | 5 / 5 of 12, −0.60 to +0.62 |
| LSTM `no_ar` | `learning_rate` | −0.15 | 9 / 7 of 16, −0.61 to +0.93 | −0.16 | 8 / 8 of 16, −0.61 to +0.87 |
| LSTM `no_ar` | `lstm_hidden_size` | −0.20 | 9 / 6 of 15, −0.66 to +0.64 | −0.21 | 9 / 6 of 15, −0.66 to +0.57 |
| LSTM `no_ar` | `dense_units` | +0.28 | 8 / 7 of 15, −0.73 to +0.57 | +0.28 | 8 / 7 of 15, −0.78 to +0.43 |
| LSTM `no_ar` | `dropout` | −0.05 | 12 / 4 of 16, −0.76 to +0.66 | −0.05 | 11 / 5 of 16, −0.76 to +0.66 |
| LSTM `ar_bounded` | `batch_size` | +0.07 | 7 / 6 of 13, −0.43 to +0.59 | +0.06 | 6 / 6 of 13, −0.32 to +0.71 |
| LSTM `ar_bounded` | `learning_rate` | −0.02 | 10 / 6 of 16, −0.44 to +0.47 | −0.10 | 10 / 6 of 16, −0.39 to +0.32 |
| LSTM `ar_bounded` | `lstm_hidden_size` | 0 | 9 / 7 of 16, −0.69 to +0.54 | −0.04 | 9 / 7 of 16, −0.45 to +0.62 |
| LSTM `ar_bounded` | `dense_units` | −0.28 | 5 / 10 of 15, −0.85 to +0.51 | −0.22 | 5 / 10 of 15, −0.62 to +0.49 |
| LSTM `ar_bounded` | `dropout` | −0.02 | 12 / 4 of 16, −0.61 to +0.59 | +0.03 | 11 / 5 of 16, −0.61 to +0.56 |
| Transformer `no_ar` | `batch_size` | −0.04 | 5 / 8 of 13, −0.50 to +0.70 | −0.03 | 5 / 8 of 13, −0.52 to +0.70 |
| Transformer `no_ar` | `learning_rate` | −0.09 | 8 / 8 of 16, −0.54 to +0.54 | +0.05 | 8 / 8 of 16, −0.54 to +0.57 |
| Transformer `no_ar` | `d_model` | −0.01 | 1 / 1 of 2, −0.17 to +0.17 | −0.03 | 1 / 1 of 2, −0.06 to +0.09 |
| Transformer `no_ar` | `nhead` | +0.08 | 10 / 6 of 16, −0.76 to +0.66 | +0.05 | 10 / 6 of 16, −0.72 to +0.78 |
| Transformer `no_ar` | `num_encoder_layers` | +0.24 | 13 / 3 of 16, −0.29 to +0.57 | +0.29 | 13 / 2 of 16, −0.05 to +0.57 |
| Transformer `no_ar` | `dropout` | +0.17 | 10 / 6 of 16, −0.35 to +0.60 | +0.13 | 10 / 6 of 16, −0.44 to +0.65 |
| Transformer `ar_bounded` | `batch_size` | +0.03 | 11 / 5 of 16, −0.61 to +0.57 | +0.04 | 11 / 5 of 16, −0.61 to +0.53 |
| Transformer `ar_bounded` | `learning_rate` | −0.05 | 4 / 12 of 16, −0.75 to +0.52 | +0.13 | 7 / 9 of 16, −0.70 to +0.58 |
| Transformer `ar_bounded` | `d_model` | −0.13 | 0 / 1 of 1, −0.52 to −0.52 | −0.13 | 0 / 1 of 1, −0.52 to −0.52 |
| Transformer `ar_bounded` | `nhead` | −0.05 | 8 / 7 of 16, −0.76 to +0.44 | −0.01 | 8 / 7 of 16, −0.76 to +0.32 |
| Transformer `ar_bounded` | `num_encoder_layers` | −0.01 | 8 / 8 of 16, −0.54 to +0.80 | −0.10 | 9 / 7 of 16, −0.49 to +0.79 |
| Transformer `ar_bounded` | `dropout` | −0.01 | 8 / 8 of 16, −0.70 to +0.62 | +0.03 | 7 / 9 of 16, −0.66 to +0.68 |
