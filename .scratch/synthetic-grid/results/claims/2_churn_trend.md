| Tree | Customers | Rate | \|bias\| at churn 20 / 40 / 60 / 80% | MAPE at churn 20 / 40 / 60 / 80% | \|bias\| rises at every step | MAPE rises at every step |
| --- | --- | --- | --- | --- | --- | --- |
| Pareto/NBD | 1,000 | 0.01 | 12 / 34 / 41 / 54 | 48 / 66 / 88 / 159 | yes | yes |
| Pareto/NBD | 1,000 | 0.05 | 8 / 15 / 12 / 21 | 35 / 39 / 42 / 58 | no | yes |
| Pareto/NBD | 1,000 | 0.10 | 3 / 5 / 9 / 21 | 31 / 32 / 37 / 49 | yes | yes |
| Pareto/NBD | 1,000 | 0.30 | 14 / 14 / 13 / 13 | 30 / 31 / 32 / 35 | no | yes |
| LSTM `no_ar` | 1,000 | 0.01 | 59 / 132 / 290 / 755 | 71 / 136 / 291 / 759 | yes | yes |
| LSTM `no_ar` | 1,000 | 0.05 | 42 / 98 / 157 / 497 | 45 / 99 / 158 / 497 | yes | yes |
| LSTM `no_ar` | 1,000 | 0.10 | 31 / 57 / 61 / 225 | 33 / 58 / 69 / 233 | yes | yes |
| LSTM `no_ar` | 1,000 | 0.30 | 17 / 9 / 9 / 17 | 19 / 14 / 18 / 32 | no | no |
| LSTM `ar_bounded` | 1,000 | 0.01 | 47 / 114 / 190 / 576 | 65 / 121 / 196 / 591 | yes | yes |
| LSTM `ar_bounded` | 1,000 | 0.05 | 47 / 52 / 61 / 27 | 48 / 56 / 67 / 60 | no | no |
| LSTM `ar_bounded` | 1,000 | 0.10 | 30 / 28 / 21 / 15 | 32 / 34 / 33 / 45 | no | no |
| LSTM `ar_bounded` | 1,000 | 0.30 | 10 / 6 / 10 / 19 | 12 / 13 / 19 / 35 | no | yes |
| Transformer `no_ar` | 1,000 | 0.01 | 37 / 76 / 102 / 121 | 58 / 96 / 129 / 197 | yes | yes |
| Transformer `no_ar` | 1,000 | 0.05 | 72 / 108 / 144 / 111 | 75 / 110 / 149 / 122 | no | no |
| Transformer `no_ar` | 1,000 | 0.10 | 56 / 91 / 99 / 265 | 58 / 94 / 103 / 265 | yes | yes |
| Transformer `no_ar` | 1,000 | 0.30 | 24 / 29 / 50 / 88 | 36 / 36 / 57 / 96 | yes | no |
| Transformer `ar_bounded` | 1,000 | 0.01 | 44 / 58 / 60 / 90 | 62 / 75 / 98 / 176 | yes | yes |
| Transformer `ar_bounded` | 1,000 | 0.05 | 63 / 54 / 69 / 84 | 68 / 65 / 77 / 102 | no | no |
| Transformer `ar_bounded` | 1,000 | 0.10 | 45 / 33 / 113 / 90 | 52 / 41 / 115 / 102 | no | no |
| Transformer `ar_bounded` | 1,000 | 0.30 | 27 / 35 / 62 / 46 | 37 / 42 / 67 / 54 | no | no |
| Pareto/NBD | 3,000 | 0.01 | 11 / 22 / 30 / 22 | 37 / 46 / 62 / 78 | no | yes |
| Pareto/NBD | 3,000 | 0.05 | 7 / 10 / 7 / 8 | 32 / 33 / 34 / 39 | no | yes |
| Pareto/NBD | 3,000 | 0.10 | 2 / 2 / 3 / 7 | 30 / 30 / 32 / 35 | no | yes |
| Pareto/NBD | 3,000 | 0.30 | 15 / 14 / 13 / 12 | 30 / 30 / 30 / 31 | no | no |
| LSTM `no_ar` | 3,000 | 0.01 | 83 / 131 / 261 / 399 | 84 / 132 / 261 / 399 | yes | yes |
| LSTM `no_ar` | 3,000 | 0.05 | 25 / 69 / 102 / 274 | 26 / 69 / 105 / 277 | yes | yes |
| LSTM `no_ar` | 3,000 | 0.10 | 17 / 27 / 8 / 22 | 18 / 29 / 17 / 32 | no | no |
| LSTM `no_ar` | 3,000 | 0.30 | 6 / 4 / 7 / 23 | 8 / 9 / 12 / 28 | no | yes |
| LSTM `ar_bounded` | 3,000 | 0.01 | 25 / 37 / 72 / 127 | 38 / 52 / 89 / 152 | yes | yes |
| LSTM `ar_bounded` | 3,000 | 0.05 | 24 / 25 / 22 / 16 | 26 / 29 / 33 / 37 | no | yes |
| LSTM `ar_bounded` | 3,000 | 0.10 | 11 / 6 / 7 / 14 | 14 / 12 / 16 / 32 | no | no |
| LSTM `ar_bounded` | 3,000 | 0.30 | 3 / 7 / 13 / 24 | 6 / 10 / 18 / 31 | yes | yes |
