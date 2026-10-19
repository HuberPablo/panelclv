## T1 one-timers

#### One-timers: exactly one purchase in calibration

| panel | n customers (% of cohort) | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---|---|---|
| cdnow 2y | 1417 (60%) | 0.24 | 0.37 → **1.50×** [1.32, 1.71] | 1.04 → **4.25×** [3.87, 4.61] | 0.17 → **0.68×** [0.67, 0.69] |
| electronics 2y | 150 (18%) | 0.82 | 2.58 → **3.14×** [2.99, 3.27] | 1.40 → **1.71×** [1.66, 1.75] | 0.15 → **0.18×** [0.18, 0.18] |
| electronics 3y | 111 (13%) | 1.30 | not run | 0.93 → **0.72×** [0.69, 0.75] | 0.12 → **0.10×** [0.10, 0.10] |
| electronic_5y | 861 (23%) | 0.15 | 0.24 → **1.62×** [1.52, 1.72] | 0.27 → **1.86×** [1.77, 1.94] | 0.06 → **0.43×** [0.43, 0.43] |
| gift 2y | 1058 (51%) | 0.23 | 0.31 → **1.34×** [1.24, 1.42] | 0.37 → **1.59×** [1.53, 1.65] | 0.22 → **0.94×** [0.94, 0.95] |
| gift 3y | 891 (43%) | 0.15 | not run | 0.24 → **1.63×** [1.57, 1.69] | 0.13 → **0.92×** [0.91, 0.92] |
| multichannel 2y | 1028 (73%) | 0.09 | 0.27 → **2.87×** [2.48, 3.28] | 0.18 → **1.87×** [1.82, 1.93] | 0.08 → **0.86×** [0.85, 0.87] |
| multichannel 3y | 951 (68%) | 0.05 | not run | 0.11 → **2.37×** [2.28, 2.45] | 0.06 → **1.15×** [1.14, 1.16] |

## T2 recent / recent-heavy

#### Recent buyers: last purchase in the final 5 calibration weeks (0–4 weeks before holdout)

| panel | n customers (% of cohort) | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---|---|---|
| cdnow 2y | 194 (8%) | 3.71 | 2.24 → **0.61×** [0.53, 0.69] | 1.87 → **0.50×** [0.49, 0.52] | 3.19 → **0.86×** [0.85, 0.87] |
| electronics 2y | 99 (12%) | 4.33 | 2.63 → **0.61×** [0.58, 0.63] | 2.92 → **0.67×** [0.64, 0.71] | 1.65 → **0.38×** [0.38, 0.38] |
| electronics 3y | 91 (11%) | 4.47 | not run | 3.52 → **0.79×** [0.75, 0.82] | 1.39 → **0.31×** [0.31, 0.31] |
| electronic_5y | 237 (6%) | 2.22 | 2.36 → **1.07×** [1.03, 1.10] | 2.08 → **0.94×** [0.90, 0.97] | 1.59 → **0.72×** [0.72, 0.72] |
| gift 2y | 65 (3%) | 2.20 | 1.07 → **0.49×** [0.45, 0.52] | 0.74 → **0.34×** [0.32, 0.35] | 1.19 → **0.54×** [0.54, 0.54] |
| gift 3y | 65 (3%) | 2.48 | not run | 1.31 → **0.53×** [0.51, 0.54] | 1.41 → **0.57×** [0.57, 0.57] |
| multichannel 2y | 48 (3%) | 0.58 | 0.27 → **0.47×** [0.40, 0.54] | 0.33 → **0.56×** [0.53, 0.59] | 0.58 → **1.00×** [0.99, 1.00] |
| multichannel 3y | 41 (3%) | 0.61 | not run | 0.47 → **0.77×** [0.74, 0.81] | 0.64 → **1.05×** [1.04, 1.05] |

#### Recent heavy buyers: ≥5 purchases in calibration AND last purchase ≤13 weeks before holdout

| panel | n customers (% of cohort) | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---|---|---|
| cdnow 2y | 157 (7%) | 4.65 | 2.76 → **0.59×** [0.52, 0.67] | 2.45 → **0.53×** [0.50, 0.55] | 4.17 → **0.90×** [0.89, 0.91] |
| electronics 2y | 110 (13%) | 4.59 | 2.62 → **0.57×** [0.55, 0.59] | 2.91 → **0.63×** [0.60, 0.67] | 1.81 → **0.39×** [0.39, 0.40] |
| electronics 3y | 119 (14%) | 4.82 | not run | 3.57 → **0.74×** [0.71, 0.77] | 1.50 → **0.31×** [0.31, 0.31] |
| electronic_5y | 339 (9%) | 2.34 | 2.38 → **1.02×** [0.98, 1.06] | 2.13 → **0.91×** [0.88, 0.94] | 1.77 → **0.76×** [0.76, 0.76] |
| gift 2y | 93 (5%) | 2.35 | 1.39 → **0.59×** [0.54, 0.63] | 0.91 → **0.39×** [0.37, 0.40] | 1.68 → **0.71×** [0.71, 0.72] |
| gift 3y | 171 (8%) | 2.16 | not run | 1.43 → **0.66×** [0.65, 0.68] | 1.52 → **0.70×** [0.70, 0.70] |
| multichannel 2y | 15 (1%) | 1.73 | 0.27 → **0.16×** [0.14, 0.18] | 0.33 → **0.19×** [0.18, 0.21] | 1.29 → **0.75×** [0.74, 0.75] |
| multichannel 3y | 20 (1%) | 1.00 | not run | 0.61 → **0.61×** [0.58, 0.64] | 1.20 → **1.20×** [1.19, 1.21] |

#### Recent heavy buyers: forecast in the last holdout quarter ÷ first holdout quarter

| panel | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---:|---:|
| cdnow 2y | 0.61 | 0.34 | 0.34 | 0.76 |
| electronics 2y | 1.25 | 0.79 | 0.53 | 0.92 |
| electronics 3y | 2.30 | not run | 0.88 | 0.95 |
| electronic_5y | 0.62 | 0.64 | 0.60 | 0.93 |
| gift 2y | 1.25 | 1.29 | 1.37 | 0.95 |
| gift 3y | 2.01 | not run | 1.41 | 0.95 |
| multichannel 2y | 0.67 | 0.64 | 0.53 | 0.91 |
| multichannel 3y | 1.67 | not run | 0.73 | 0.93 |

#### One-timers: forecast in the last holdout quarter ÷ first holdout quarter

| panel | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---:|---:|
| cdnow 2y | 0.80 | 1.27 | 3.61 | 0.76 |
| electronics 2y | 0.74 | 0.84 | 1.23 | 0.92 |
| electronics 3y | 0.89 | not run | 1.08 | 0.95 |
| electronic_5y | 0.57 | 0.55 | 0.52 | 0.93 |
| gift 2y | 1.50 | 2.29 | 2.95 | 0.95 |
| gift 3y | 2.00 | not run | 1.85 | 0.95 |
| multichannel 2y | 2.29 | 0.67 | 1.24 | 0.91 |
| multichannel 3y | 2.50 | not run | 1.35 | 0.93 |

## T3 separation

| panel | 'cold' group (n) | 'hot' group (n) | actual: hot / cold per customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---|---|---|---|---|
| cdnow 2y | F1 R14-52w (1417) | F5+ R<=13w (157) | 4.65 / 0.24 = **19.0×** | 2.76 / 0.37 = **7.9×** | 2.45 / 1.04 = **3.1×** | 4.17 / 0.17 = **25.0×** |
| electronics 2y | F1 R>52w (150) | F5+ R<=13w (110) | 4.59 / 0.82 = **5.6×** | 2.62 / 2.58 = **1.0×** | 2.91 / 1.40 = **2.1×** | 1.81 / 0.15 = **12.4×** |
| electronics 3y | F1 R>52w (111) | F5+ R<=13w (119) | 4.82 / 1.30 = **3.7×** | not run | 3.57 / 0.93 = **3.9×** | 1.50 / 0.12 = **12.0×** |
| electronic_5y | F1 R>52w (861) | F5+ R<=13w (339) | 2.34 / 0.15 = **16.0×** | 2.38 / 0.24 = **10.2×** | 2.13 / 0.27 = **7.9×** | 1.77 / 0.06 = **28.4×** |
| gift 2y | F1 R>52w (1058) | F5+ R<=13w (93) | 2.35 / 0.23 = **10.3×** | 1.39 / 0.31 = **4.6×** | 0.91 / 0.37 = **2.6×** | 1.68 / 0.22 = **7.8×** |
| gift 3y | F1 R>52w (891) | F5+ R<=13w (171) | 2.16 / 0.15 = **14.8×** | not run | 1.43 / 0.24 = **6.2×** | 1.52 / 0.13 = **11.4×** |
| multichannel 2y | F1 R>52w (1028) | F5+ R<=13w (15) | 1.73 / 0.09 = **18.4×** | 0.27 / 0.27 = **1.0×** | 0.33 / 0.18 = **1.9×** | 1.29 / 0.08 = **15.9×** |
| multichannel 3y | F1 R>52w (951) | F5+ R<=13w (20) | 1.00 / 0.05 = **20.7×** | not run | 0.61 / 0.11 = **5.5×** | 1.20 / 0.06 = **21.6×** |

## T4 recency gradient (predicted tx / customer; actual first)


#### F2-4 purchases in calibration

| panel | recency band | n | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---:|---:|
| cdnow 2y | R<=13w | 299 | 1.40 | 0.83 | 0.71 | 1.46 |
| cdnow 2y | R14-52w | 448 | 0.76 | 0.47 | 1.06 | 0.58 |
| electronics 2y | R<=13w | 27 | 2.37 | 2.61 | 2.17 | 0.89 |
| electronics 2y | R14-52w | 48 | 0.96 | 2.58 | 1.62 | 0.75 |
| electronics 2y | R>52w | 234 | 1.07 | 2.58 | 1.43 | 0.22 |
| electronics 3y | R<=13w | 12 | 0.67 | not run | 1.89 | 0.66 |
| electronics 3y | R14-52w | 43 | 0.67 | not run | 1.25 | 0.58 |
| electronics 3y | R>52w | 204 | 0.72 | not run | 0.97 | 0.21 |
| electronic_5y | R<=13w | 100 | 0.58 | 0.90 | 0.81 | 0.55 |
| electronic_5y | R14-52w | 254 | 0.50 | 0.58 | 0.52 | 0.48 |
| electronic_5y | R>52w | 1287 | 0.25 | 0.32 | 0.33 | 0.20 |
| gift 2y | R<=13w | 277 | 1.17 | 0.70 | 0.57 | 0.79 |
| gift 2y | R14-52w | 255 | 0.76 | 0.53 | 0.52 | 0.69 |
| gift 2y | R>52w | 336 | 0.37 | 0.41 | 0.42 | 0.52 |
| gift 3y | R<=13w | 184 | 0.83 | not run | 0.84 | 0.66 |
| gift 3y | R14-52w | 193 | 0.61 | not run | 0.53 | 0.58 |
| gift 3y | R>52w | 490 | 0.30 | not run | 0.31 | 0.38 |
| multichannel 2y | R<=13w | 72 | 0.39 | 0.27 | 0.30 | 0.52 |
| multichannel 2y | R14-52w | 111 | 0.30 | 0.27 | 0.21 | 0.44 |
| multichannel 2y | R>52w | 165 | 0.21 | 0.27 | 0.19 | 0.26 |
| multichannel 3y | R<=13w | 57 | 0.25 | not run | 0.41 | 0.45 |
| multichannel 3y | R14-52w | 71 | 0.34 | not run | 0.29 | 0.38 |
| multichannel 3y | R>52w | 276 | 0.15 | not run | 0.19 | 0.22 |

#### F5+ purchases in calibration

| panel | recency band | n | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---:|---:|
| cdnow 2y | R<=13w | 157 | 4.65 | 2.76 | 2.45 | 4.17 |
| cdnow 2y | R14-52w | 36 | 1.72 | 0.90 | 1.61 | 1.11 |
| electronics 2y | R<=13w | 110 | 4.59 | 2.62 | 2.91 | 1.81 |
| electronics 2y | R14-52w | 133 | 2.35 | 2.58 | 1.90 | 1.23 |
| electronics 2y | R>52w | 127 | 1.31 | 2.58 | 1.50 | 0.36 |
| electronics 3y | R<=13w | 119 | 4.82 | not run | 3.57 | 1.50 |
| electronics 3y | R14-52w | 161 | 2.91 | not run | 2.11 | 1.22 |
| electronics 3y | R>52w | 179 | 0.96 | not run | 1.12 | 0.40 |
| electronic_5y | R<=13w | 339 | 2.34 | 2.38 | 2.13 | 1.77 |
| electronic_5y | R14-52w | 420 | 1.10 | 1.23 | 1.16 | 1.20 |
| electronic_5y | R>52w | 494 | 0.54 | 0.56 | 0.54 | 0.42 |
| gift 2y | R<=13w | 93 | 2.35 | 1.39 | 0.91 | 1.68 |
| gift 2y | R14-52w | 34 | 1.18 | 1.16 | 0.80 | 1.51 |
| gift 2y | R>52w | 9 | 0.44 | 0.86 | 0.61 | 1.03 |
| gift 3y | R<=13w | 171 | 2.16 | not run | 1.43 | 1.52 |
| gift 3y | R14-52w | 94 | 1.14 | not run | 1.04 | 1.33 |
| gift 3y | R>52w | 39 | 0.36 | not run | 0.51 | 0.79 |
| multichannel 2y | R<=13w | 15 | 1.73 | 0.27 | 0.33 | 1.29 |
| multichannel 2y | R14-52w | 7 | 1.14 | 0.27 | 0.25 | 1.07 |
| multichannel 2y | R>52w | 4 | 0.25 | 0.27 | 0.21 | 0.69 |
| multichannel 3y | R<=13w | 20 | 1.00 | not run | 0.61 | 1.20 |
| multichannel 3y | R14-52w | 16 | 1.38 | not run | 0.49 | 0.94 |
| multichannel 3y | R>52w | 11 | 0.45 | not run | 0.39 | 0.52 |

## T4b within-group discrimination, F2-4 repeat buyers (AUC holdout buyer vs non-buyer; Spearman)

| panel | ValendinLSTM AUC / ρ | LSTM + bounded32 AUC / ρ | Pareto/NBD AUC / ρ |
|---|---|---|---|
| cdnow 2y | 0.60 / 0.18 | 0.43 / -0.12 | 0.63 / 0.23 |
| electronics 2y | 0.50 / 0.00 | 0.54 / 0.07 | 0.56 / 0.10 |
| electronics 3y | not run | 0.54 / 0.04 | 0.55 / 0.05 |
| electronic_5y | 0.60 / 0.13 | 0.60 / 0.13 | 0.60 / 0.14 |
| gift 2y | 0.60 / 0.18 | 0.57 / 0.13 | 0.58 / 0.15 |
| gift 3y | not run | 0.61 / 0.19 | 0.62 / 0.19 |
| multichannel 2y | 0.52 / 0.02 | 0.52 / 0.03 | 0.56 / 0.09 |
| multichannel 3y | not run | 0.58 / 0.12 | 0.59 / 0.12 |

## T5 panel characteristics and differentiation

| panel | customers | calibration weeks | zero cells, calibration | tx / customer, calibration | one-timers | ≥5 tx | silent >52 w at cut-off | actual separation | separation captured: V / B / P | Spearman: V / B / P | forecast CV: V / B / P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| cdnow 2y | 2357 | 39 | 95.0% | 2.03 | 60% | 8% | 0% | 19.0× | 41% / 16% / 131% | 0.40 / 0.09 / 0.45 | 1.34 / 0.66 / 1.71 |
| electronics 2y | 829 | 104 | 97.5% | 5.65 | 18% | 45% | 62% | 5.6× | 18% / 37% / 221% | 0.03 / 0.26 / 0.31 | 0.05 / 0.36 / 1.11 |
| electronics 3y | 829 | 156 | 97.9% | 7.32 | 13% | 55% | 60% | 3.7× | – / 104% / 324% | – / 0.32 / 0.33 | – / 0.73 / 1.10 |
| electronic_5y | 3755 | 260 | 98.4% | 4.55 | 23% | 33% | 70% | 16.0× | 64% / 50% / 177% | 0.41 / 0.40 / 0.40 | 1.27 / 1.14 / 1.40 |
| gift 2y | 2062 | 104 | 98.1% | 2.04 | 51% | 7% | 68% | 10.3× | 45% / 25% / 76% | 0.37 / 0.33 / 0.38 | 0.64 / 0.37 / 0.83 |
| gift 3y | 2062 | 156 | 98.4% | 2.60 | 43% | 15% | 69% | 14.8× | – / 42% / 77% | – / 0.42 / 0.44 | – / 0.87 / 1.06 |
| multichannel 2y | 1402 | 104 | 98.6% | 1.44 | 73% | 2% | 85% | 18.4× | 6% / 10% / 87% | 0.00 / 0.10 / 0.19 | 0.09 / 0.23 / 1.21 |
| multichannel 3y | 1402 | 156 | 99.0% | 1.60 | 68% | 3% | 88% | 20.7× | – / 26% / 105% | – / 0.24 / 0.25 | – / 0.69 / 1.41 |

## T6 effects vs Pareto/NBD (independent bootstrap)

| panel | model (n vs 20) | log2 separation | log2 pred/act, one-timers | log2 pred/act, recent buyers 0–4 w | RMSE of totals, recent buyers | RMSE of totals, one-timers |
|---|---|---|---|---|---|---|
| cdnow 2y | ValendinLSTM (20) | **-1.73 [-1.91, -1.55]** | **+1.08 [+0.90, +1.26]** | **-0.56 [-0.73, -0.38]** | **+0.44 [+0.31, +0.57]** | **+0.01 [+0.01, +0.03]** |
| cdnow 2y | LSTM+bounded32 (100) | **-3.25 [-3.40, -3.09]** | **+2.45 [+2.28, +2.60]** | **-0.80 [-0.86, -0.74]** | **+0.47 [+0.42, +0.51]** | **+0.40 [+0.34, +0.45]** |
| electronics 2y | ValendinLSTM (20) | **-3.60 [-3.62, -3.59]** | **+4.13 [+4.05, +4.19]** | **+0.67 [+0.60, +0.73]** | **-0.17 [-0.20, -0.13]** | **+0.55 [+0.47, +0.61]** |
| electronics 2y | LSTM+bounded32 (100) | **-2.62 [-2.69, -2.54]** | **+3.24 [+3.20, +3.29]** | **+0.77 [+0.70, +0.85]** | **-0.33 [-0.38, -0.29]** | **-0.02 [-0.03, -0.01]** |
| electronics 3y | LSTM+bounded32 (100) | **-1.65 [-1.70, -1.60]** | **+2.86 [+2.80, +2.93]** | **+1.31 [+1.25, +1.37]** | **-0.80 [-0.83, -0.77]** | **-0.20 [-0.20, -0.19]** |
| electronic_5y | ValendinLSTM (20) | **-1.48 [-1.56, -1.41]** | **+1.91 [+1.82, +2.00]** | **+0.56 [+0.52, +0.62]** | **-0.31 [-0.34, -0.28]** | **+0.00 [+0.00, +0.01]** |
| electronic_5y | LSTM+bounded32 (20) | **-1.85 [-1.92, -1.78]** | **+2.11 [+2.04, +2.18]** | **+0.38 [+0.33, +0.43]** | **-0.20 [-0.26, -0.14]** | **+0.01 [+0.01, +0.01]** |
| gift 2y | ValendinLSTM (20) | **-0.79 [-0.91, -0.68]** | **+0.49 [+0.39, +0.58]** | **-0.18 [-0.30, -0.07]** | **+0.12 [+0.09, +0.15]** | **+0.01 [+0.00, +0.01]** |
| gift 2y | LSTM+bounded32 (100) | **-1.64 [-1.72, -1.57]** | **+0.73 [+0.67, +0.78]** | **-0.71 [-0.76, -0.66]** | **+0.31 [+0.30, +0.33]** | **+0.02 [+0.01, +0.02]** |
| gift 3y | LSTM+bounded32 (100) | **-0.90 [-0.95, -0.85]** | **+0.81 [+0.76, +0.86]** | **-0.12 [-0.16, -0.08]** | **+0.51 [+0.48, +0.53]** | **+0.01 [+0.01, +0.01]** |
| multichannel 2y | ValendinLSTM (20) | **-3.98 [-4.00, -3.96]** | **+1.66 [+1.45, +1.87]** | **-1.16 [-1.38, -0.95]** | **+0.08 [+0.07, +0.09]** | **+0.05 [+0.03, +0.07]** |
| multichannel 2y | LSTM+bounded32 (100) | **-3.17 [-3.26, -3.07]** | **+1.11 [+1.06, +1.15]** | **-0.90 [-0.99, -0.82]** | **+0.06 [+0.06, +0.06]** | **+0.01 [+0.01, +0.01]** |
| multichannel 3y | LSTM+bounded32 (100) | **-2.05 [-2.14, -1.96]** | **+1.02 [+0.97, +1.07]** | **-0.48 [-0.55, -0.41]** | **+0.12 [+0.11, +0.12]** | **+0.01 [+0.01, +0.01]** |

## T7 contribution to aggregate bias (percentage points of the panel's actual total)

| panel | model | total bias | one-timers | F2-4 | F5+ |
|---|---|---:|---:|---:|---:|
| cdnow 2y | ValendinLSTM | -24 | +9 | -16 | -17 |
| cdnow 2y | LSTM+bounded32 | +37 | +59 | -4 | -18 |
| cdnow 2y | ParetoNBD | -14 | -6 | -3 | -5 |
| electronics 2y | ValendinLSTM | +46 | +18 | +30 | -2 |
| electronics 2y | LSTM+bounded32 | -2 | +6 | +7 | -15 |
| electronics 2y | ParetoNBD | -63 | -7 | -17 | -39 |
| electronics 3y | LSTM+bounded32 | -13 | -3 | +6 | -16 |
| electronics 3y | ParetoNBD | -65 | -8 | -7 | -50 |
| electronic_5y | ValendinLSTM | +14 | +4 | +7 | +4 |
| electronic_5y | LSTM+bounded32 | +9 | +5 | +6 | -2 |
| electronic_5y | ParetoNBD | -17 | -3 | -3 | -10 |
| gift 2y | ValendinLSTM | -16 | +7 | -15 | -8 |
| gift 2y | LSTM+bounded32 | -19 | +13 | -18 | -13 |
| gift 2y | ParetoNBD | -11 | -1 | -6 | -4 |
| gift 3y | LSTM+bounded32 | -5 | +8 | -1 | -12 |
| gift 3y | ParetoNBD | -8 | -1 | +0 | -7 |
| multichannel 2y | ValendinLSTM | +67 | +80 | -1 | -12 |
| multichannel 2y | LSTM+bounded32 | +17 | +37 | -8 | -12 |
| multichannel 2y | ParetoNBD | +6 | -6 | +15 | -2 |
| multichannel 3y | LSTM+bounded32 | +33 | +36 | +10 | -13 |
| multichannel 3y | ParetoNBD | +23 | +4 | +20 | -1 |

## T8 captured separation, 3y vs 2y (LSTM + bounded32, independent bootstrap)

| panel | captured 2y | captured 3y | Δ log2 captured (3y − 2y) [95% CI] |
|---|---:|---:|---|
| electronics | 37% | 104% | **+1.52 [+1.43, +1.60]** |
| gift | 25% | 42% | **+0.76 [+0.67, +0.85]** |
| multichannel | 10% | 26% | **+1.39 [+1.26, +1.52]** |
