| Tree | Hyperparameter | ρ with \|bias\|, pooled | ρ with \|bias\|, within cells | ρ with MAPE, pooled | ρ with MAPE, within cells |
| --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | `batch_size` | **+0.48, p <10⁻⁴** | +0.04, p 0.655 | **+0.49, p <10⁻⁴** | +0.05, p 0.513 |
| LSTM `no_ar` | `learning_rate` | −0.15, p 0.056 | +0.12, p 0.123 | −0.16, p 0.043 | +0.12, p 0.133 |
| LSTM `no_ar` | `lstm_hidden_size` | −0.20, p 0.012 | +0.10, p 0.190 | −0.21, p 0.007 | +0.08, p 0.346 |
| LSTM `no_ar` | `dense_units` | **+0.28, p 0.0003** | −0.05, p 0.508 | **+0.28, p 0.0003** | −0.06, p 0.469 |
| LSTM `no_ar` | `dropout` | −0.05, p 0.524 | +0.14, p 0.069 | −0.05, p 0.567 | +0.13, p 0.102 |
| LSTM `ar_bounded` | `batch_size` | +0.07, p 0.390 | +0.06, p 0.485 | +0.06, p 0.427 | +0.08, p 0.300 |
| LSTM `ar_bounded` | `learning_rate` | −0.02, p 0.803 | +0.10, p 0.221 | −0.10, p 0.214 | +0.03, p 0.669 |
| LSTM `ar_bounded` | `lstm_hidden_size` | 0, p 0.969 | +0.01, p 0.892 | −0.04, p 0.616 | +0.04, p 0.582 |
| LSTM `ar_bounded` | `dense_units` | **−0.28, p 0.0003** | −0.15, p 0.061 | −0.22, p 0.006 | −0.19, p 0.014 |
| LSTM `ar_bounded` | `dropout` | −0.02, p 0.787 | +0.07, p 0.354 | +0.03, p 0.736 | +0.10, p 0.230 |
| Transformer `no_ar` | `batch_size` | −0.04, p 0.637 | −0.07, p 0.366 | −0.03, p 0.730 | −0.06, p 0.457 |
| Transformer `no_ar` | `learning_rate` | −0.09, p 0.270 | −0.03, p 0.740 | +0.05, p 0.557 | +0.03, p 0.710 |
| Transformer `no_ar` | `d_model` | −0.01, p 0.886 | +0.01, p 0.928 | −0.03, p 0.675 | +0.01, p 0.934 |
| Transformer `no_ar` | `nhead` | +0.08, p 0.285 | +0.02, p 0.798 | +0.05, p 0.552 | +0.02, p 0.843 |
| Transformer `no_ar` | `num_encoder_layers` | **+0.24, p 0.002** | +0.18, p 0.020 | **+0.29, p 0.0002** | +0.21, p 0.008 |
| Transformer `no_ar` | `dropout` | +0.17, p 0.027 | +0.08, p 0.297 | +0.13, p 0.113 | +0.09, p 0.240 |
| Transformer `ar_bounded` | `batch_size` | +0.03, p 0.699 | +0.02, p 0.818 | +0.04, p 0.639 | −0.01, p 0.872 |
| Transformer `ar_bounded` | `learning_rate` | −0.05, p 0.545 | −0.14, p 0.088 | +0.13, p 0.100 | −0.07, p 0.349 |
| Transformer `ar_bounded` | `d_model` | −0.13, p 0.098 | −0.08, p 0.325 | −0.13, p 0.093 | −0.08, p 0.325 |
| Transformer `ar_bounded` | `nhead` | −0.05, p 0.541 | −0.06, p 0.480 | −0.01, p 0.892 | −0.08, p 0.312 |
| Transformer `ar_bounded` | `num_encoder_layers` | −0.01, p 0.946 | 0, p 0.996 | −0.10, p 0.193 | +0.02, p 0.845 |
| Transformer `ar_bounded` | `dropout` | −0.01, p 0.927 | −0.04, p 0.603 | +0.03, p 0.687 | −0.06, p 0.472 |
