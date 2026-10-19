
#### CDNOW — 39 weeks calibration, 39 weeks holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 |
|---|---:|---:|---:|---:|---:|
| *All customers* | | | | | |
| all | 2357 | 1895 | 1627 (0.86×) | 1445 (0.76×) | 2599 (1.37×) |
| *Frequency (calibration purchases)* | | | | | |
| 1 purchase | 1417 | 346 | 236 (0.68×) | 520 (1.50×) | 1472 (4.25×) |
| 2 purchases | 435 | 309 | 276 (0.89×) | 200 (0.65×) | 377 (1.22×) |
| 3-4 purchases | 312 | 448 | 420 (0.94×) | 260 (0.58×) | 308 (0.69×) |
| 5+ purchases | 193 | 792 | 695 (0.88×) | 465 (0.59×) | 442 (0.56×) |
| *Recency within frequency (weeks silent)* | | | | | |
| 2 purchases, silent ≤ 13 w | 125 | 117 | 127 (1.09×) | 70 (0.60×) | 62 (0.53×) |
| 2 purchases, silent 14–38 w | 310 | 192 | 148 (0.77×) | 130 (0.68×) | 315 (1.64×) |
| 3-4 purchases, silent ≤ 13 w | 174 | 301 | 310 (1.03×) | 180 (0.60×) | 150 (0.50×) |
| 3-4 purchases, silent 14–38 w | 138 | 147 | 111 (0.75×) | 80 (0.54×) | 159 (1.08×) |
| 5+ purchases, silent ≤ 13 w | 157 | 730 | 655 (0.90×) | 433 (0.59×) | 384 (0.53×) |
| 5+ purchases, silent 14–38 w | 36 | 62 | 40 (0.64×) | 32 (0.52×) | 58 (0.93×) |
| *Overdue (silence ÷ own usual gap)* | | | | | |
| on schedule | 389 | 991 | 869 (0.88×) | 568 (0.57×) | 488 (0.49×) |
| overdue 1-3x | 222 | 312 | 342 (1.09×) | 185 (0.59×) | 232 (0.74×) |
| overdue >3x | 317 | 246 | 179 (0.73×) | 168 (0.68×) | 392 (1.59×) |
| *Momentum* | | | | | |
| rising | 277 | 570 | 504 (0.88×) | 362 (0.64×) | 302 (0.53×) |
| declining | 651 | 979 | 885 (0.90×) | 558 (0.57×) | 810 (0.83×) |
| one purchase week | 1429 | 346 | 238 (0.69×) | 525 (1.52×) | 1486 (4.30×) |
| *Valendin et al. (2022) groups* | | | | | |
| At Risk | 173 | 0 | 268 | 167 | 213 |
| Opportunity | 171 | 875 | 194 (0.22×) | 155 (0.18×) | 209 (0.24×) |
| Other | 2013 | 1020 | 1165 (1.14×) | 1124 (1.10×) | 2177 (2.13×) |

#### electronics — 2 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 | Transformer |
|---|---:|---:|---:|---:|---:|---:|
| *All customers* | | | | | | |
| all | 829 | 1467 | 541 (0.37×) | 2142 (1.46×) | 1443 (0.98×) | 1504 (1.03×) |
| *Frequency (calibration purchases)* | | | | | | |
| 1 purchase | 150 | 123 | 22 (0.18×) | 386 (3.14×) | 210 (1.71×) | 240 (1.95×) |
| 2 purchases | 134 | 141 | 35 (0.25×) | 346 (2.45×) | 196 (1.39×) | 221 (1.57×) |
| 3-4 purchases | 175 | 219 | 77 (0.35×) | 452 (2.06×) | 274 (1.25×) | 300 (1.37×) |
| 5+ purchases | 370 | 984 | 408 (0.41×) | 959 (0.97×) | 763 (0.78×) | 743 (0.76×) |
| *Recency within frequency (weeks silent)* | | | | | | |
| 2 purchases, silent ≤ 13 w | 7† | 29 | 5 (0.19×) | 18 (0.63×) | 15 (0.50×) | 14 (0.49×) |
| 2 purchases, silent 14–52 w | 12† | 2 | 8 (4.02×) | 31 (15.45×) | 19 (9.33×) | 20 (10.04×) |
| 2 purchases, silent > 52 w | 115 | 110 | 21 (0.19×) | 296 (2.69×) | 162 (1.48×) | 187 (1.70×) |
| 3-4 purchases, silent ≤ 13 w | 20† | 35 | 19 (0.54×) | 52 (1.49×) | 44 (1.25×) | 42 (1.20×) |
| 3-4 purchases, silent 14–52 w | 36 | 44 | 28 (0.64×) | 93 (2.12×) | 59 (1.35×) | 63 (1.43×) |
| 3-4 purchases, silent > 52 w | 119 | 140 | 30 (0.21×) | 307 (2.19×) | 171 (1.22×) | 195 (1.39×) |
| 5+ purchases, silent ≤ 13 w | 110 | 505 | 199 (0.39×) | 289 (0.57×) | 320 (0.63×) | 271 (0.54×) |
| 5+ purchases, silent 14–52 w | 133 | 312 | 163 (0.52×) | 343 (1.10×) | 252 (0.81×) | 254 (0.81×) |
| 5+ purchases, silent > 52 w | 127 | 167 | 45 (0.27×) | 327 (1.96×) | 191 (1.14×) | 218 (1.30×) |
| *Overdue (silence ÷ own usual gap)* | | | | | | |
| on schedule | 222 | 661 | 290 (0.44×) | 578 (0.88×) | 523 (0.79×) | 476 (0.72×) |
| overdue 1-3x | 132 | 285 | 143 (0.50×) | 340 (1.19×) | 234 (0.82×) | 245 (0.86×) |
| overdue >3x | 156 | 196 | 62 (0.32×) | 402 (2.05×) | 239 (1.22×) | 273 (1.39×) |
| *Momentum* | | | | | | |
| rising | 159 | 523 | 201 (0.38×) | 414 (0.79×) | 368 (0.70×) | 330 (0.63×) |
| declining | 351 | 619 | 294 (0.47×) | 906 (1.46×) | 628 (1.01×) | 663 (1.07×) |
| one purchase week | 319 | 325 | 47 (0.14×) | 822 (2.53×) | 447 (1.38×) | 511 (1.57×) |
| *Valendin et al. (2022) groups* | | | | | | |
| At Risk | 140 | 0 | 119 | 362 | 256 | 263 |
| Opportunity | 81 | 611 | 37 (0.06×) | 209 (0.34×) | 132 (0.22×) | 141 (0.23×) |
| Other | 608 | 856 | 385 (0.45×) | 1571 (1.84×) | 1055 (1.23×) | 1100 (1.28×) |

#### gift — 2 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 |
|---|---:|---:|---:|---:|---:|
| *All customers* | | | | | |
| all | 2062 | 1146 | 1015 (0.89×) | 967 (0.84×) | 934 (0.81×) |
| *Frequency (calibration purchases)* | | | | | |
| 1 purchase | 1058 | 243 | 229 (0.94×) | 325 (1.34×) | 386 (1.59×) |
| 2 purchases | 453 | 227 | 220 (0.97×) | 185 (0.81×) | 195 (0.86×) |
| 3-4 purchases | 415 | 413 | 350 (0.85×) | 280 (0.68×) | 235 (0.57×) |
| 5+ purchases | 136 | 263 | 217 (0.82×) | 176 (0.67×) | 117 (0.45×) |
| *Recency within frequency (weeks silent)* | | | | | |
| 2 purchases, silent ≤ 13 w | 94 | 78 | 50 (0.64×) | 48 (0.61×) | 45 (0.57×) |
| 2 purchases, silent 14–52 w | 116 | 63 | 59 (0.94×) | 48 (0.76×) | 53 (0.84×) |
| 2 purchases, silent > 52 w | 243 | 86 | 110 (1.28×) | 90 (1.04×) | 98 (1.13×) |
| 3-4 purchases, silent ≤ 13 w | 183 | 245 | 168 (0.68×) | 146 (0.60×) | 112 (0.46×) |
| 3-4 purchases, silent 14–52 w | 139 | 130 | 117 (0.90×) | 87 (0.67×) | 80 (0.61×) |
| 3-4 purchases, silent > 52 w | 93 | 38 | 65 (1.71×) | 47 (1.24×) | 43 (1.14×) |
| 5+ purchases, silent ≤ 13 w | 93 | 219 | 156 (0.71×) | 129 (0.59×) | 85 (0.39×) |
| 5+ purchases, silent 14–52 w | 34 | 40 | 51 (1.28×) | 39 (0.99×) | 27 (0.68×) |
| 5+ purchases, silent > 52 w | 9† | 4 | 9 (2.31×) | 8 (1.93×) | 5 (1.37×) |
| *Overdue (silence ÷ own usual gap)* | | | | | |
| on schedule | 537 | 675 | 481 (0.71×) | 408 (0.60×) | 329 (0.49×) |
| overdue 1-3x | 268 | 160 | 183 (1.14×) | 137 (0.86×) | 126 (0.79×) |
| overdue >3x | 195 | 67 | 122 (1.83×) | 95 (1.42×) | 90 (1.35×) |
| *Momentum* | | | | | |
| rising | 360 | 462 | 292 (0.63×) | 260 (0.56×) | 215 (0.47×) |
| declining | 640 | 440 | 493 (1.12×) | 380 (0.86×) | 331 (0.75×) |
| one purchase week | 1062 | 244 | 230 (0.94×) | 326 (1.34×) | 388 (1.59×) |
| *Valendin et al. (2022) groups* | | | | | |
| At Risk | 236 | 0 | 218 | 171 | 138 |
| Opportunity | 89 | 355 | 45 (0.13×) | 44 (0.12×) | 42 (0.12×) |
| Other | 1737 | 791 | 752 (0.95×) | 752 (0.95×) | 754 (0.95×) |

#### multichannel — 2 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 |
|---|---:|---:|---:|---:|---:|
| *All customers* | | | | | |
| all | 1402 | 228 | 243 (1.06×) | 380 (1.67×) | 267 (1.17×) |
| *Frequency (calibration purchases)* | | | | | |
| 1 purchase | 1028 | 97 | 83 (0.86×) | 278 (2.87×) | 182 (1.87×) |
| 2 purchases | 242 | 53 | 69 (1.30×) | 66 (1.24×) | 51 (0.97×) |
| 3-4 purchases | 106 | 43 | 61 (1.41×) | 29 (0.67×) | 26 (0.61×) |
| 5+ purchases | 26† | 35 | 30 (0.85×) | 7 (0.20×) | 8 (0.22×) |
| *Recency within frequency (weeks silent)* | | | | | |
| 2 purchases, silent ≤ 13 w | 41 | 20 | 16 (0.78×) | 11 (0.56×) | 12 (0.60×) |
| 2 purchases, silent 14–52 w | 64 | 7 | 22 (3.10×) | 17 (2.48×) | 13 (1.89×) |
| 2 purchases, silent > 52 w | 137 | 26 | 32 (1.22×) | 37 (1.43×) | 26 (1.01×) |
| 3-4 purchases, silent ≤ 13 w | 31 | 8 | 22 (2.71×) | 8 (1.05×) | 10 (1.24×) |
| 3-4 purchases, silent 14–52 w | 47 | 26 | 27 (1.06×) | 13 (0.49×) | 10 (0.40×) |
| 3-4 purchases, silent > 52 w | 28† | 9 | 11 (1.27×) | 8 (0.84×) | 6 (0.63×) |
| 5+ purchases, silent ≤ 13 w | 15† | 26 | 19 (0.75×) | 4 (0.16×) | 5 (0.19×) |
| 5+ purchases, silent 14–52 w | 7† | 8 | 8 (0.94×) | 2 (0.23×) | 2 (0.22×) |
| 5+ purchases, silent > 52 w | 4† | 1 | 3 (2.75×) | 1 (1.08×) | 1 (0.84×) |
| *Overdue (silence ÷ own usual gap)* | | | | | |
| on schedule | 164 | 75 | 91 (1.22×) | 45 (0.59×) | 44 (0.58×) |
| overdue 1-3x | 102 | 39 | 40 (1.02×) | 28 (0.71×) | 20 (0.53×) |
| overdue >3x | 97 | 14 | 27 (1.95×) | 26 (1.87×) | 19 (1.36×) |
| *Momentum* | | | | | |
| rising | 129 | 63 | 64 (1.01×) | 35 (0.56×) | 34 (0.53×) |
| declining | 234 | 65 | 95 (1.46×) | 63 (0.98×) | 49 (0.76×) |
| one purchase week | 1039 | 100 | 84 (0.84×) | 281 (2.81×) | 184 (1.84×) |
| *Valendin et al. (2022) groups* | | | | | |
| At Risk | 288 | 0 | 110 | 78 | 64 |
| Opportunity | 20† | 50 | 2 (0.04×) | 5 (0.11×) | 4 (0.07×) |
| Other | 1094 | 178 | 130 (0.73×) | 297 (1.67×) | 199 (1.12×) |

#### electronics — 3 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | LSTM + bounded32 |
|---|---:|---:|---:|---:|
| *All customers* | | | | |
| all | 829 | 1541 | 536 (0.35×) | 1343 (0.87×) |
| *Frequency (calibration purchases)* | | | | |
| 1 purchase | 111 | 144 | 14 (0.10×) | 103 (0.72×) |
| 2 purchases | 109 | 54 | 25 (0.46×) | 109 (2.01×) |
| 3-4 purchases | 150 | 130 | 51 (0.39×) | 166 (1.28×) |
| 5+ purchases | 459 | 1213 | 445 (0.37×) | 965 (0.80×) |
| *Recency within frequency (weeks silent)* | | | | |
| 2 purchases, silent ≤ 13 w | 1† | 0 | 1 | 2 |
| 2 purchases, silent 14–52 w | 15† | 21 | 8 (0.37×) | 18 (0.87×) |
| 2 purchases, silent > 52 w | 93 | 33 | 17 (0.51×) | 89 (2.69×) |
| 3-4 purchases, silent ≤ 13 w | 11† | 8 | 7 (0.92×) | 21 (2.63×) |
| 3-4 purchases, silent 14–52 w | 28† | 8 | 17 (2.13×) | 35 (4.41×) |
| 3-4 purchases, silent > 52 w | 111 | 114 | 27 (0.23×) | 110 (0.96×) |
| 5+ purchases, silent ≤ 13 w | 119 | 574 | 179 (0.31×) | 424 (0.74×) |
| 5+ purchases, silent 14–52 w | 161 | 468 | 196 (0.42×) | 340 (0.73×) |
| 5+ purchases, silent > 52 w | 179 | 171 | 71 (0.41×) | 201 (1.17×) |
| *Overdue (silence ÷ own usual gap)* | | | | |
| on schedule | 273 | 833 | 297 (0.36×) | 676 (0.81×) |
| overdue 1-3x | 149 | 312 | 139 (0.45×) | 246 (0.79×) |
| overdue >3x | 181 | 153 | 71 (0.46×) | 210 (1.37×) |
| *Momentum* | | | | |
| rising | 221 | 632 | 238 (0.38×) | 541 (0.86×) |
| declining | 382 | 666 | 269 (0.40×) | 591 (0.89×) |
| one purchase week | 226 | 243 | 28 (0.12×) | 211 (0.87×) |
| *Valendin et al. (2022) groups* | | | | |
| At Risk | 136 | 0 | 133 | 279 |
| Opportunity | 71 | 722 | 39 (0.05×) | 115 (0.16×) |
| Other | 622 | 819 | 364 (0.44×) | 949 (1.16×) |

#### gift — 3 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | LSTM + bounded32 |
|---|---:|---:|---:|---:|
| *All customers* | | | | |
| all | 2062 | 1040 | 956 (0.92×) | 983 (0.95×) |
| *Frequency (calibration purchases)* | | | | |
| 1 purchase | 891 | 130 | 119 (0.92×) | 212 (1.63×) |
| 2 purchases | 413 | 126 | 139 (1.10×) | 142 (1.13×) |
| 3-4 purchases | 454 | 293 | 283 (0.96×) | 267 (0.91×) |
| 5+ purchases | 304 | 491 | 416 (0.85×) | 363 (0.74×) |
| *Recency within frequency (weeks silent)* | | | | |
| 2 purchases, silent ≤ 13 w | 45 | 23 | 19 (0.83×) | 29 (1.24×) |
| 2 purchases, silent 14–52 w | 73 | 31 | 30 (0.96×) | 31 (0.99×) |
| 2 purchases, silent > 52 w | 295 | 72 | 90 (1.25×) | 83 (1.15×) |
| 3-4 purchases, silent ≤ 13 w | 139 | 130 | 102 (0.79×) | 126 (0.97×) |
| 3-4 purchases, silent 14–52 w | 120 | 87 | 82 (0.94×) | 71 (0.82×) |
| 3-4 purchases, silent > 52 w | 195 | 76 | 98 (1.29×) | 70 (0.92×) |
| 5+ purchases, silent ≤ 13 w | 171 | 370 | 260 (0.70×) | 245 (0.66×) |
| 5+ purchases, silent 14–52 w | 94 | 107 | 125 (1.17×) | 98 (0.91×) |
| 5+ purchases, silent > 52 w | 39 | 14 | 31 (2.21×) | 20 (1.42×) |
| *Overdue (silence ÷ own usual gap)* | | | | |
| on schedule | 629 | 675 | 545 (0.81×) | 559 (0.83×) |
| overdue 1-3x | 257 | 158 | 178 (1.13×) | 121 (0.77×) |
| overdue >3x | 282 | 77 | 114 (1.48×) | 91 (1.18×) |
| *Momentum* | | | | |
| rising | 500 | 570 | 432 (0.76×) | 444 (0.78×) |
| declining | 668 | 340 | 405 (1.19×) | 326 (0.96×) |
| one purchase week | 894 | 130 | 119 (0.92×) | 213 (1.64×) |
| *Valendin et al. (2022) groups* | | | | |
| At Risk | 380 | 0 | 281 | 247 |
| Opportunity | 39 | 123 | 12 (0.09×) | 15 (0.12×) |
| Other | 1643 | 917 | 663 (0.72×) | 721 (0.79×) |

#### multichannel — 3 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | LSTM + bounded32 |
|---|---:|---:|---:|---:|
| *All customers* | | | | |
| all | 1402 | 173 | 212 (1.23×) | 231 (1.33×) |
| *Frequency (calibration purchases)* | | | | |
| 1 purchase | 951 | 46 | 53 (1.15×) | 109 (2.37×) |
| 2 purchases | 262 | 41 | 54 (1.32×) | 47 (1.16×) |
| 3-4 purchases | 142 | 39 | 60 (1.55×) | 50 (1.28×) |
| 5+ purchases | 47 | 47 | 45 (0.95×) | 24 (0.52×) |
| *Recency within frequency (weeks silent)* | | | | |
| 2 purchases, silent ≤ 13 w | 22† | 4 | 7 (1.70×) | 7 (1.69×) |
| 2 purchases, silent 14–52 w | 39 | 11 | 11 (1.01×) | 9 (0.79×) |
| 2 purchases, silent > 52 w | 201 | 26 | 36 (1.39×) | 32 (1.23×) |
| 3-4 purchases, silent ≤ 13 w | 35 | 10 | 19 (1.87×) | 17 (1.65×) |
| 3-4 purchases, silent 14–52 w | 32 | 13 | 16 (1.22×) | 12 (0.93×) |
| 3-4 purchases, silent > 52 w | 75 | 16 | 26 (1.61×) | 21 (1.33×) |
| 5+ purchases, silent ≤ 13 w | 20† | 20 | 24 (1.20×) | 12 (0.61×) |
| 5+ purchases, silent 14–52 w | 16† | 22 | 15 (0.68×) | 8 (0.36×) |
| 5+ purchases, silent > 52 w | 11† | 5 | 6 (1.14×) | 4 (0.85×) |
| *Overdue (silence ÷ own usual gap)* | | | | |
| on schedule | 188 | 76 | 90 (1.18×) | 65 (0.86×) |
| overdue 1-3x | 124 | 29 | 43 (1.47×) | 29 (1.01×) |
| overdue >3x | 129 | 22 | 26 (1.20×) | 26 (1.19×) |
| *Momentum* | | | | |
| rising | 152 | 63 | 65 (1.04×) | 48 (0.77×) |
| declining | 289 | 64 | 93 (1.46×) | 72 (1.13×) |
| one purchase week | 961 | 46 | 53 (1.16×) | 110 (2.39×) |
| *Valendin et al. (2022) groups* | | | | |
| At Risk | 357 | 0 | 109 | 88 |
| Opportunity | 4† | 9 | 0 (0.02×) | 0 (0.05×) |
| Other | 1041 | 164 | 102 (0.62×) | 142 (0.87×) |

#### electronic_5y — 5 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 | Transformer |
|---|---:|---:|---:|---:|---:|---:|
| *All customers* | | | | | | |
| all | 3755 | 2153 | 1796 (0.83×) | 2451 (1.14×) | 2340 (1.09×) | 2422 (1.12×) |
| *Frequency (calibration purchases)* | | | | | | |
| 1 purchase | 861 | 126 | 54 (0.43×) | 204 (1.62×) | 234 (1.86×) | 172 (1.37×) |
| 2 purchases | 690 | 163 | 118 (0.72×) | 215 (1.32×) | 218 (1.34×) | 198 (1.21×) |
| 3-4 purchases | 951 | 344 | 314 (0.91×) | 432 (1.26×) | 413 (1.20×) | 424 (1.23×) |
| 5+ purchases | 1253 | 1520 | 1311 (0.86×) | 1600 (1.05×) | 1476 (0.97×) | 1627 (1.07×) |
| *Recency within frequency (weeks silent)* | | | | | | |
| 2 purchases, silent ≤ 13 w | 17† | 8 | 6 (0.76×) | 13 (1.57×) | 11 (1.33×) | 9 (1.09×) |
| 2 purchases, silent 14–52 w | 61 | 26 | 21 (0.80×) | 29 (1.10×) | 25 (0.95×) | 20 (0.78×) |
| 2 purchases, silent > 52 w | 612 | 129 | 91 (0.71×) | 174 (1.35×) | 183 (1.42×) | 169 (1.31×) |
| 3-4 purchases, silent ≤ 13 w | 83 | 50 | 49 (0.97×) | 78 (1.56×) | 70 (1.40×) | 59 (1.19×) |
| 3-4 purchases, silent 14–52 w | 193 | 101 | 101 (1.00×) | 119 (1.18×) | 107 (1.06×) | 98 (0.97×) |
| 3-4 purchases, silent > 52 w | 675 | 193 | 164 (0.85×) | 235 (1.22×) | 236 (1.22×) | 267 (1.38×) |
| 5+ purchases, silent ≤ 13 w | 339 | 793 | 600 (0.76×) | 808 (1.02×) | 723 (0.91×) | 715 (0.90×) |
| 5+ purchases, silent 14–52 w | 420 | 462 | 502 (1.09×) | 516 (1.12×) | 486 (1.05×) | 542 (1.17×) |
| 5+ purchases, silent > 52 w | 494 | 265 | 208 (0.79×) | 275 (1.04×) | 266 (1.01×) | 370 (1.39×) |
| *Overdue (silence ÷ own usual gap)* | | | | | | |
| on schedule | 1125 | 1232 | 1049 (0.85×) | 1359 (1.10×) | 1242 (1.01×) | 1184 (0.96×) |
| overdue 1-3x | 751 | 441 | 495 (1.12×) | 499 (1.13×) | 475 (1.08×) | 584 (1.32×) |
| overdue >3x | 972 | 344 | 196 (0.57×) | 378 (1.10×) | 376 (1.09×) | 472 (1.37×) |
| *Momentum* | | | | | | |
| rising | 925 | 1001 | 799 (0.80×) | 1098 (1.10×) | 994 (0.99×) | 937 (0.94×) |
| declining | 1923 | 1016 | 940 (0.93×) | 1138 (1.12×) | 1100 (1.08×) | 1303 (1.28×) |
| one purchase week | 907 | 136 | 57 (0.42×) | 215 (1.58×) | 246 (1.81×) | 182 (1.34×) |
| *Valendin et al. (2022) groups* | | | | | | |
| At Risk | 651 | 0 | 500 | 585 | 558 | 640 |
| Opportunity | 45 | 192 | 11 (0.06×) | 20 (0.10×) | 19 (0.10×) | 16 (0.08×) |
| Other | 3059 | 1961 | 1286 (0.66×) | 1846 (0.94×) | 1763 (0.90×) | 1765 (0.90×) |
