# Machine-learning forecasts of individual customer transactions, 2018–2026

A literature note: which methods beyond the ones already in the thesis review forecast
per-customer transaction counts, purchase timing, non-contractual churn or CLV with modern
machine learning, and what they report. The papers already covered in the review are
excluded here and only named where another paper benchmarks against them: Schmittlein,
Morrison & Colombo 1987; Fader, Hardie & Lee 2005; Platzer & Reutterer 2016; Bachmann,
Meierer & Näf 2021; Malthouse 2009; Xie & Huang 2020; Sarkar & De Bruyn 2021; Mena et al.
2019; Valendin, Reutterer, Platzer & Kalcher 2022; Kurbiel 2024; Kiyakoglu & Aydin 2024;
Näf, Mbelson & Meierer 2026; Kim, McCarthy & Lee 2026; Tudoran et al. 2024.

**Method.** Web search (standard, then extended where thin), the Semantic Scholar and
OpenAlex APIs (citation lists of Valendin et al. 2022, Bauer & Jannach 2021 and Chen et al.
2018; DOI lookups), then the primary source for every paper: arXiv full text, the author or
publisher PDF converted with `pdftotext`, or the GitHub repository. Where only an abstract
or landing page could be reached, the entry says **[abstract only]** and nothing beyond the
abstract is attributed to it. Every number is the paper's own **claim**, reproduced from its
tables or abstract; none was re-computed. Author–year citations; the URL each statement was
checked against follows it.

**Scope.** Individual-level targets only. Session-level purchase-intent classifiers from
clickstreams (Toth et al. 2017, Sheil et al. 2018), contractual churn, and recommender
papers are listed in §9 as adjacent and not counted. Twenty-two papers appear in the
comparison table in §11; the rest have shorter entries in §5.4, §7.4, §8 and §9.

---

## 1. Summary

Three findings stand out.

1. **Almost nothing benchmarks a neural *count* model against Pareto/NBD on the thesis's
   terms.** Of the papers below, only Chen, Keng & Moreno (2018), Xie (2019), Bauer &
   Jannach (2021), Chen et al. (2018), Chen, Ng, Smyl & Steininger (2024), Gadgil et al.
   (2023), Dew & Ansari (2018) and Sarkar & De Bruyn (2026) put a BTYD model in the
   comparison, and most of them forecast *revenue* or a *binary* event, not a count per
   period. The count-per-period, rollout-style evaluation of Valendin et al. (2022) has no
   published replication; the closest are Sarkar & De Bruyn (2026), who simulate forward
   with a dual LSTM, and Näf et al. (2026), already in the review.
2. **No Transformer-versus-Pareto/NBD comparison on a transaction panel was found** in a
   peer-reviewed venue. Lu & Kannan (2025) use a Transformer on touchpoint sequences for
   conversion, not counts; USE-LTV (Yang et al. 2025) is an ads-revenue regressor; West &
   Lu (2026) is an MSI talk behind a login. Likewise no application of Chronos, TimesFM or
   Moirai to customer panels was found; the foundation-model work on transactions (§8) is
   about representation learning for banking tasks.
3. **The hybrid direction is live and points both ways.** Chou et al. (2022) feed BG/BB
   estimates into a Lasso and report beating two RNNs; Xie (2019) embeds the Pareto/NBD
   likelihood in a network's loss; Ghosh & Schein (2026) replace MCMC with amortised
   variational inference; Yan & Resnick (2023) ensemble "generative" and "discriminative"
   models in production. The deep-learning papers that win do so by a margin that is often
   small, single-run, and on the authors' own benchmark.

---

## 2. Hybrids of BTYD models and machine learning

### 2.1 Chou, Chuang, Chou & Liang (2022) — Lasso-BG/BB **[abstract and highlights only]**

- *Citation.* Chou, P., Chuang, H. H.-C., Chou, Y.-C. & Liang, T.-P. (2022), "Predictive
  analytics for customer repurchase: Interdisciplinary integration of buy till you die
  modeling and machine learning", *European Journal of Operational Research* 296(2),
  635–651, DOI 10.1016/j.ejor.2021.04.021
  (https://ideas.repec.org/a/eee/ejores/v296y2022i2p635-651.html).
- *Target.* Customer repurchase propensity and frequency in a discrete-time non-contractual
  setting (same source).
- *Model.* BG/BB estimates (parameters and predictions) are fed as inputs, with about 100
  transaction-derived features (RFM, returns/cancellations, payment tool, device), into a
  high-dimensional Lasso regression; a CART–ANN variant shows the same pattern (same
  source).
- *Benchmarks.* BG/BB alone, Lasso alone, and in robustness checks "two sophisticated
  recurrent neural networks" (same source; names of the RNNs not accessible).
- *Data.* "A large online retail dataset"; size and window lengths not accessible.
- *Headline (claim).* The integrated Lasso-BG/BB improves on both BG/BB and Lasso alone,
  stays interpretable, outperforms the two RNNs, and the BG/BB outputs are the most
  influential of the ~100 predictors (same source). Numbers not accessible.
- *Code.* Not stated.
- *Caveat.* Paywalled; the RNN robustness check could not be inspected, so it is unknown
  whether the RNNs were tuned or how the target was defined for them.

### 2.2 Xie (2019) — Pareto/NBD likelihood inside a neural network's loss

- *Citation.* Xie, S.-M. (2019), "Neural Network Based Parameter Estimation Method for the
  Pareto/NBD Model", arXiv:1911.01919 (https://arxiv.org/abs/1911.01919).
- *Target.* Per-customer number of repeat purchases in the holdout and active/inactive
  status (PDF §3).
- *Model.* A feed-forward network ("NNA") that outputs the individual Pareto/NBD rates
  (μ_i, λ_i); eight loss variants: MAE, MSE, the Pareto/NBD likelihood, the likelihood
  plus MAE/MSE, and a likelihood-ratio form plus MAE/MSE (PDF §3).
- *Benchmarks.* Pareto/NBD by maximum likelihood.
- *Data.* CDNOW (23,570 customers) and "E-tailing", a Taiwanese online clothing retailer
  sampled to 24,000 customers; 60 % of customers for training, 40 % held out (PDF Table 1).
  The split is by customer, not by time.
- *Headline (claims, PDF Table 4, CDNOW).* Pareto/NBD: inactive-status accuracy 76.56 %,
  multi-class accuracy on the repeat-purchase count 67.14 %, MAE 0.690, predicted total
  repeat purchases 4,105 against an actual 7,634. Best network, NNA (Likelihood + MSE):
  76.68 %, 61.09 %, MAE 0.896, predicted 8,526. The abstract summarises this as the
  Pareto/NBD "underestimates repeat purchases at both individual and aggregate levels"
  and the network "performs much better".
- *Code.* Not stated.
- *Caveat.* The table shows the opposite of the abstract on individual-level error: the
  Pareto/NBD has the lowest MAE and the highest exact-count accuracy, and the network wins
  only on the aggregate total. Single run; customer-wise split; a preprint that was never
  published.

### 2.3 Ghosh & Schein (2026) — amortised variational inference for non-Poisson BTYD **[abstract only]**

- *Citation.* Ghosh, S. & Schein, A. (2026), "Scalable Amortized Variational Inference for
  Non-Poisson Buy-'Til-You-Die Models", arXiv:2608.19022
  (https://arxiv.org/abs/2608.19022).
- *Target.* Not named in the abstract beyond "predictive performance".
- *Model.* A BTYD family with Weibull-renewal transaction timing (the Pareto/GGG
  direction), fitted by amortised variational inference instead of MCMC; covariates are
  added through gradient-based inference (same source).
- *Benchmarks.* "The current state-of-the-art" fit by MCMC, unnamed in the abstract.
- *Data.* 5 million online-retail customers (proprietary) and 4 million donors from the
  2020 US election cycle (public).
- *Headline (claim).* Fits the 5-million-customer panel in 8 minutes against an estimated
  3–4 days for the MCMC baseline, "with no appreciable change" in interpretation or
  predictive performance (same source).
- *Code.* Not stated.
- *Caveat.* The abstract reports speed, not accuracy; it cites Valendin et al. 2022, which
  is how it surfaced, but no comparison with an LSTM is mentioned.

### 2.4 Chen, Keng & Moreno (2018) — MAT-RNN, a Weibull-output RNN against Pareto/NBD on CDNOW

- *Citation.* Chen, T., Keng, B. & Moreno, J. (2018), "Multivariate Arrival Times with
  Recurrent Neural Networks for Personalized Demand Forecasting", arXiv:1812.11444; ICDM
  2018 DMS workshop (https://arxiv.org/abs/1812.11444; full text read at
  https://ar5iv.labs.arxiv.org/html/1812.11444).
- *Target.* Time until each customer's next purchase of each product, jointly over
  products, with right-censoring (ar5iv §3).
- *Model.* Two LSTM layers, then a dense layer that outputs a Weibull scale and shape per
  product; negative log-likelihood with the density for observed and the survival function
  for censored steps (ar5iv §3).
- *Benchmarks.* Pareto/NBD (on CDNOW only), WTTE-RNN, a squared-loss RNN, and a
  per-product random forest (ar5iv §4).
- *Data.* CDNOW weekly, training 1997-01-01 to 1998-05-31, predicting the following month;
  a retailer with five product categories of 6k–341k customers, 78 training weeks and a
  4-week test horizon; CMAPSS engine data as a regression check (ar5iv §4).
- *Headline (claims).* ROC-AUC for "buys in the test window": MAT-RNN 0.84 vs Pareto/NBD
  0.80 on CDNOW; on the retailer MAT-RNN beats the random forest in four of five categories
  (e.g. deli 0.892 vs 0.847) and loses on one (bars 0.817 vs 0.830) (ar5iv Tables 2–3).
- *Code.* https://github.com/rubikloud/matrnn (2 stars, 13 commits; no licence shown).
- *Caveat.* The Pareto/NBD is reduced to a binary "any purchase" classifier for the
  comparison; one month of holdout; a workshop paper that was not developed further.

### 2.5 Yan & Resnick (2023) — a production ensemble of generative and discriminative models **[abstract only]**

- *Citation.* Yan, Y. & Resnick, N. (2023), "A high-performance turnkey system for customer
  lifetime value prediction in retail brands", *Quantitative Marketing and Economics*, DOI
  10.1007/s11129-023-09272-x (abstract via https://api.openalex.org/works/doi:10.1007/s11129-023-09272-x).
- *Target.* Future customer value with a churn stage, updated daily (abstract).
- *Model.* Encodings and embeddings for high-cardinality inputs; a multi-stage churn–CLV
  framework; a feature-weighted ensemble of "generative and discriminative" models
  (abstract). Which generative model is not named in the abstract.
- *Data.* Twelve retail brands on the Amperity platform, six evaluation intervals from June
  2020 to September 2022 (abstract).
- *Headline (claim).* "Consistently surpasses benchmark performances" on all twelve brands;
  numbers not accessible (Springer PDF is behind a bot wall).
- *Code.* Not stated.
- *Caveat.* Vendor paper on proprietary panels; benchmarks unnamed in the abstract.

---

## 3. Recurrent sequence models for per-period value or counts (the Valendin lineage)

### 3.1 Bauer & Jannach (2021) — encoder–decoder GRU stacked with gradient boosting, against BG/NBD

- *Citation.* Bauer, J. & Jannach, D. (2021), "Improved Customer Lifetime Value Prediction
  with Sequence-To-Sequence Learning and Feature-Based Models", *ACM Transactions on
  Knowledge Discovery from Data*, DOI 10.1145/3441444 (author PDF
  https://web-ainf.aau.at/pub/jannach/files/Journal_TKDD_2021.pdf).
- *Target.* Profit (revenue for the public set) per customer per future time step, summed
  over the horizon: four weeks on daily data for the public set, six months on weekly data
  for the proprietary set (PDF §4.1, §5.1.1). A regression target, not a count.
- *Model.* Two encoder GRU layers and a decoder GRU producing the future per-period profit
  sequence, with temporal convolutions and customer embeddings; a GBM on engineered
  features; the final model stacks the two ("GBM-S2S") (PDF §4.4–4.5).
- *Benchmarks.* Adjusted customer-mean / ARMA, BG/NBD, a Markov-chain CLV model, and the
  embeddings + random-forest method of Chamberlain et al. 2017 (PDF §5.2.1).
- *Data.* UCI Online Retail ("UKRetail", 4,373 customers, 3,284 after removing wholesale
  outliers, about one year of daily data) and "Children", a European e-commerce company
  with 1,081,443 customers (1,073,872 after denoising); last six months held out (PDF
  Table 4, §5.1).
- *Headline (claims, PDF Table 5).* RMSE / MAE. UKRetail: BG/NBD 69.17 / 41.54; S2S 65.36
  / 34.55; GBM-S2S 64.95 / 30.80. Children: BG/NBD 20.14 / 8.09; GBM 12.50 / 3.42; S2S
  12.16 / 3.91; GBM-S2S 11.65 / 3.14. Differences on Children significant at p < 10⁻⁵; on
  UKRetail only "a clear tendency (p < 0.2)" (PDF §5.3.1).
- *Code.* https://bitbucket.org/Josef-Bauer/clv_pred/src/master/ (PDF footnote 10).
- *Caveat.* Revenue regression on a heavily filtered panel (20 % of UKRetail customers
  removed as wholesale); BG/NBD with Gamma-Gamma is the only probabilistic benchmark; the
  S2S model wins on RMSE and the GBM on MAE, so "which is better" depends on the metric.

### 3.2 Sarkar & De Bruyn (2026) — dual LSTM with Monte Carlo rollout for CLV distributions **[abstract only]**

- *Citation.* Sarkar, M. & De Bruyn, A. (2026), "A dual-LSTM model for CLV distributions:
  Diverging paths and bimodalities", *Journal of the Academy of Marketing Science*, DOI
  10.1007/s11747-026-01193-x (abstract via Crossref,
  https://api.crossref.org/works/10.1007/s11747-026-01193-x).
- *Target.* The full distribution of each donor's future value, generated by simulating
  customer responses and firm solicitations forward (abstract).
- *Model.* Two interacting LSTMs, one for customer behaviour and one for the firm's
  marketing actions, rolled out by Monte Carlo simulation (abstract).
- *Benchmarks.* Six, "including the Pareto/NBD, hierarchical Bayesian extensions, and
  state-of-the-art deep learning alternatives" (abstract; the deep-learning benchmarks are
  not named in the abstract).
- *Data.* Nine years of individual-level data on 21,939 donors (abstract).
- *Headline (claim).* Beats the six benchmarks on 19 of 24 evaluation metrics; ranking by
  the right tail of the CLV distribution lifts targeting by up to 8.3 % (abstract).
- *Code.* Not stated.
- *Caveat.* Paywalled, so the 24 metrics and the window lengths could not be read; this is
  the closest published design to the thesis's sample-and-feed-back rollout, by the authors
  of the 2021 LSTM response model already in the review.

### 3.3 Chen, Ng, Smyl & Steininger (2024) — a dilated multi-cell RNN against BTYD and LightGBM at Uber and Meta

- *Citation.* Chen, H., Ng, E., Smyl, S. & Steininger, G. (2024), "Predicting Customer
  Lifetime Value Using Recurrent Neural Net", arXiv:2412.20295 (v2 Jan 2025), first
  presented at the KDD 2022 Customer Journey Optimization workshop
  (https://arxiv.org/abs/2412.20295; PDF read).
- *Target.* Acquisition LTV (new users, Uber) and rolling LTV (existing users, Meta Quest):
  gross revenue per user over 1/4/13/26-week horizons (PDF §2, §5.2).
- *Model.* A multi-layer dilated RNN whose cell carries a delayed controlling state, from
  Smyl's ES-dRNN line; periods with no transactions are sub-sampled and calendar features
  added (PDF §4.3–4.4).
- *Benchmarks.* XGBoost (Uber); an unnamed BTYD model and LightGBM (Meta) (PDF Tables 2–3).
- *Data.* Proprietary; Meta Quest user series up to four years long with 100+ features;
  sizes not reported.
- *Headline (claims).* Uber RMSE: XGBoost 2,292; 2-cell RNN 2,830; 2-cell RNN with city
  embeddings 2,271. Meta adjusted-SMAPE at 4 / 13 / 26 weeks: BTYD 53.6 / 70.5 / 78.3 %;
  LightGBM 30.4 / 53.7 / 61.1 %; dRNN 22.9 / 39.8 / 51.2 % (PDF Tables 2–3).
- *Code.* Not stated.
- *Caveat.* Which BTYD model and how it was fitted is not stated; a four-page industry note
  with no variance reported and no public data.

### 3.4 Chen, Guitart, Fernández del Río & Periáñez (2018) — DNN and CNN against Pareto/NBD and MBG/CNBD-k in a mobile game

- *Citation.* Chen, P. P., Guitart, A., Fernández del Río, A. & Periáñez, Á. (2018),
  "Customer Lifetime Value in Video Games Using Deep Learning and Parametric Models", *IEEE
  Big Data 2018*, 2134–2140, arXiv:1811.12799 (full text read at
  https://arxiv.org/html/1811.12799v1).
- *Target.* Each paying player's spend over the next 365 days (regression).
- *Model.* A multilayer perceptron on 203 engineered features; a CNN on the raw daily
  series. No LSTM (listed as future work).
- *Benchmarks.* Pareto/NBD and MBG/CNBD-k, each with an "average" or Gamma-Gamma spend
  model; BG/NBD and BG/CNBD were fitted but not tabulated.
- *Data.* *Age of Ishtaria*, 2014-09-24 to 2017-05-01; 2,505 paying users who churned
  (nine days inactive) in the last year; 80/20 split.
- *Headline (claims, Table II).* RMSLE / NRMSE / SMAPE / % error: Pareto/NBD + average
  9.35 / 1.88 / 95.65 / 8.96 %; MBG/CNBD-k + average 3.46 / 1.68 / 75.53 / 7.96 %; DNN
  1.82 / 1.12 / 72.99 / 5.82 %; CNN 1.84 / 1.05 / 73.76 / 5.72 %. On the top-20 % spenders
  the % error is 33.35 % (Pareto/NBD) vs 15.64 % (CNN) (Table III).
- *Code.* Not stated.
- *Caveat.* The abstract says CNNs "perform better", the text says DNN and CNN are "almost
  identical"; the sample is 2,505 churned payers, split by customer; single run.

### 3.5 Gadgil, Gill & Abdelmoniem (2023) — stacked tree ensembles against BG/NBD on Online Retail II

- *Citation.* Gadgil, K., Gill, S. S. & Abdelmoniem, A. M. (2023), "A Meta-learning based
  Stacked Regression Approach for Customer Lifetime Value Prediction", arXiv:2308.08502;
  also *Journal of Economy and Technology* 2023 (PDF read from
  https://arxiv.org/pdf/2308.08502).
- *Target.* The number of transactions per customer in the succeeding three months (PDF
  §IV.b, Table 3) — a count, though the paper calls it CLV.
- *Model.* A stacked regressor over random forest and XGBoost; LightGBM and a DNN as
  further baselines.
- *Benchmarks.* BG/NBD.
- *Data.* UCI Online Retail II, 5,850 customers, 2009-12 to 2011-12 (PDF Table 1).
- *Headline (claims, PDF Table 5).* RMSE / MAE: BG/NBD 1.62 / 0.90; LightGBM 1.95 / 0.88;
  DNN 1.53 / 0.83; random forest 1.44 / 0.87; XGBoost 1.34 / 0.83; stacked 1.37 / 0.82.
- *Code.* Not stated.
- *Caveat.* One split, no variance, no significance test; the gap between BG/NBD and the
  best ML model is 0.08 in MAE on a count target; the authors themselves say the LightGBM
  edge "cannot be considered significant".

### 3.6 The `rfm2lstm` repository and what cites Valendin et al. (2022)

- The reference implementation is https://github.com/valendin/rfm2lstm: Python,
  TensorFlow + TensorFlow Probability, GPL-3.0, 21 stars, 2 forks, 26 commits, with a demo
  on the Czech banking dataset from data.world (`banking_transactions_demo.ipynb`) and a
  README invitation to send openly shareable panels. No fork or downstream repository that
  extends it was found.
- Semantic Scholar lists 23 citing papers for DOI 10.1016/j.ijresmar.2022.02.007 (query
  https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.ijresmar.2022.02.007/citations).
  The ones that forecast customer behaviour are Sarkar & De Bruyn 2026 (§3.2), Ghosh &
  Schein 2026 (§2.3), Hruschka 2025 (§9), Simon 2025 (§4.2), Lu & Kannan 2025 (§6.1), and
  Kiyakoglu & Aydin 2024 and Tudoran et al. 2024 (already in the review). **No replication
  of the 2022 study was found.** Näf et al. 2026 (in the review) use the authors'
  implementation as their LSTM benchmark and call it "a particularly strong benchmark"
  because it derives seasonal regularities (https://arxiv.org/pdf/2604.22636).

---

## 4. Bayesian and nonparametric extensions of the benchmark itself

### 4.1 Dew & Ansari (2018) — the Gaussian-process propensity model (GPPM)

- *Citation.* Dew, R. & Ansari, A. (2018), "Bayesian Nonparametric Customer Base Analysis
  with Model-Based Visualizations", *Marketing Science* 37(2), 216–235, DOI
  10.1287/mksc.2017.1050 (author PDF
  https://business.columbia.edu/sites/default/files-efs/pubfiles/26107/ansari_Bayesian_Nonparametric.pdf).
- *Target.* Daily spend incidence per customer, modelled through latent GP functions of
  calendar time, recency, lifetime and purchase number; the forecasting comparison is on
  the daily spending series of the customer base over a 30-day holdout (PDF §3.3).
- *Model.* Individual purchase propensity as a sum of GP-distributed latent functions with
  a random effect, estimated in Stan.
- *Benchmarks.* Pareto/NBD, BG/NBD, two log-logistic hazard models, a latent-propensity
  model (LPM) and a state-space propensity model (SSPM) (PDF §3.3.1).
- *Data.* Two free-to-play mobile games ("Life Simulator", "City Builder"); 10,000 users
  each who installed in a 30-day window; 8,000 in calibration and 2,000 held out
  cross-sectionally, plus the last 30 days held out in calendar time (PDF §3.1, §3.3).
- *Headline (claims, PDF Table 2, holdout).* MAPE / RMSE on daily spending. Life
  Simulator: GPPM 0.24 / 22.54; Pareto/NBD 0.33 / 32.10; BG/NBD 0.31 / 30.04. City Builder:
  GPPM 0.32 / 20.97; Pareto/NBD 0.45 / 27.80; BG/NBD 0.61 / 37.41. The SSPM is the closest
  competitor (0.17 / 20.59 and 0.38 / 27.16).
- *Code.* Stan code in Online Appendix C; a Shiny app with every benchmark fit at
  https://rdew.shinyapps.io/gppm_benchmarks/ (PDF Figure 10 notes).
- *Caveat.* Evaluated on aggregate daily series, not per-customer error; a 30-day holdout;
  MCMC on 8,000 customers, so scale is the cost.

### 4.2 Simon (2025) — which Pareto/NBD estimator, and timing forecasts **[abstract only]**

- *Citation.* Simon, L. (2025), "A generalised comparison of Pareto/NBD based forecasts
  using MCMC, maximum likelihood, and heuristics", *Journal of Business Economics* 95,
  1079–1105, DOI 10.1007/s11573-025-01237-8 (abstract via Crossref).
- *Headline (claim).* Model-based forecasts beat heuristics for the number of purchases,
  activity, and top-segment membership; MCMC is slightly better than MLE and adds
  intervals; "forecasting the timing of a customer's next purchase yields deviations that
  are too large to be used in practice" (abstract).
- *Why it is here.* Not a machine-learning paper, but it is the most recent statement that
  the benchmark's own next-purchase-time forecast is unusable, which bounds what the
  point-process papers in §5 are competing against.

---

## 5. Survival and point-process models of purchase timing

### 5.1 Grob, Cardoso, Liu, Little & Chamberlain (2018) — RNN survival model for return time at ASOS

- *Citation.* Grob, G. L., Cardoso, Â., Liu, C. H. B., Little, D. A. & Chamberlain, B. P.
  (2018), "A Recurrent Neural Network Survival Model: Predicting Web User Return Time",
  *ECML PKDD 2018*, LNCS 11053, 152–168, arXiv:1807.04098 (full text read at
  https://ar5iv.labs.arxiv.org/html/1807.04098).
- *Target.* Days until a user's next session, with right-censoring for users who do not
  return in the prediction window.
- *Model.* An LSTM recurrent marked temporal point process (RMTPP-style hazard
  λ*(t) = exp(v·h + w(t − t_j) + b)) trained with a survival likelihood that includes the
  censored users.
- *Benchmarks.* Time since last session; an LSTM with MSE on returning users only; Cox
  proportional hazards, each with and without an absence-conditioned correction.
- *Data.* ASOS, 38,716 training and 9,680 test users over about 1.5 years; 63.6 % return.
- *Headline (claims, Table 1).* RMSE on returning users (days) / concordance /
  non-returning AUC: RNN-MSE 28.69 / 0.706 / 0.763; Cox 49.99 / 0.816 / 0.793; RNNSM 59.99
  / 0.739 / 0.796.
- *Code.* https://github.com/grobgl/rnnsm.
- *Caveat.* The proposed model has the best non-returning AUC but worse concordance than
  Cox and worse RMSE than the plain RNN; the authors note it overestimates return times.

### 5.2 Equihua, Nordmark, Ali & Lausen (2023) — deep survival for non-contractual retail churn

- *Citation.* Equihua, J. P., Nordmark, H., Ali, M. & Lausen, B. (2023), "Modelling
  customer churn for the retail industry in a deep learning based sequential framework",
  arXiv:2304.00575 (PDF read; no journal version found).
- *Target.* Per-customer survival function for the time to next purchase; churn as "no
  purchase before day t" (t = 180 days for the retail data).
- *Model.* An LSTM plus MLP that outputs the rate of a per-customer exponential
  inter-purchase distribution, trained on the censored likelihood (PDF §3).
- *Benchmarks.* Cox proportional hazards and an individual-level Kaplan–Meier estimator.
- *Data.* A large retailer, 17.8 million transactions by 2.8 million customers, mean
  inter-event time 80 days; plus a synthetic panel (PDF §4).
- *Headline (claims, Tables 4–5, retail test).* Accuracy: RNN 0.654, Cox 0.485, KM 0.592.
  C-index: RNN 0.391, Cox 0.580, KM 0.555. Brier at 180 days: RNN 0.303, Cox 0.070.
- *Code.* Not stated.
- *Caveat.* The RNN wins on accuracy and loses on C-index and Brier score, i.e. on every
  proper survival metric; the exponential assumption is noted by the authors as
  under-estimating survival for loyal segments.

### 5.3 Manzoor & Akoglu (2017) — RUSH!, a time-varying multivariate Hawkes process for when and what customers buy next

- *Citation.* Manzoor, E. & Akoglu, L. (2017), "RUSH! Targeted Time-limited Coupons via
  Purchase Forecasts", *KDD 2017*, 1923–1931, DOI 10.1145/3097983.3098104 (author PDF
  https://www.andrew.cmu.edu/user/lakoglu/pubs/17-kdd-rush.pdf).
- *Target.* The time and merchant category of a customer's next purchase.
- *Model.* A multidimensional Hawkes process with time-varying base rates built from 28
  temporal features (hour, weekday, pay-day), self- and cross-excitation across ten
  categories (PDF §3).
- *Benchmarks.* Poisson, time-varying Poisson, and plain Hawkes processes.
- *Data.* 199,109 prepaid-card accounts of a US bank, 24.9 million purchases 2013–2016,
  reduced to 2.8 million transactions in ten categories (PDF §2).
- *Headline (claims, PDF §5).* Predictive log-likelihood improves by 9.53 units for time
  variation, 16.28 for memory, and 30.52 for both together; the model dominates the
  baselines in a coupon cost–benefit simulation.
- *Code.* https://github.com/emaadmanzoor/rush/.
- *Caveat.* Pre-2018 and not neural, but it is the cleanest customer-level Hawkes model of
  purchase timing found; evaluation is likelihood and simulated profit, not count error.

### 5.4 Neural temporal point processes applied to purchases — shorter entries

- **Bai et al. (2019), CTRec**, *SIGIR 2019*, 675–684: a "demand-aware Hawkes process" with
  a CNN for short-term and self-attention for long-term periodic demand, which "estimates
  the purchase time of a product given a user" on four commercial datasets (ACM abstract,
  https://unpaywall.org/10.1145%2F3331184.3331199) **[abstract only]**. Recommender
  framing; timing metrics not accessible.
- **Vassøy, Ruocco, de Souza da Silva & Aune (2018)**, arXiv:1812.01276: a hierarchical RNN
  for next-item recommendation with a jointly trained point process for the time until the
  user's next session; "improves" an unnamed return-time baseline on two unnamed datasets
  (https://arxiv.org/abs/1812.01276) **[abstract only]**.
- **Šeputis, Gamper & Paulavičius (2022)**, arXiv:2208.08623: reproduces published neural
  TPPs on a consumer-behaviour dataset about twice the largest public benchmark and reports
  that they "cannot forecast rare events" and that SDE-based variants "do not scale easily"
  (https://arxiv.org/abs/2208.08623) **[abstract only]**. A useful negative result.
- **Türkmen, Wang & Januschowski (2019)**, "Intermittent Demand Forecasting with Deep
  Renewal Processes", NeurIPS 2019 TPP workshop, arXiv:1911.10416: RNNs that parameterise
  conditional inter-demand time and size distributions (https://arxiv.org/abs/1911.10416)
  **[abstract only]**. SKU-level, not customer-level, but the same renewal-process view of a
  sparse count series.

### 5.5 Martínez, Schmuck, Pereverzyev, Pirker & Haltmeier (2020) — gradient boosting for next-month purchase

- *Citation.* *European Journal of Operational Research* 281(3), 588–596, DOI
  10.1016/j.ejor.2018.04.034 (https://ideas.repec.org/a/eee/ejores/v281y2020i3p588-596.html)
  **[abstract only]**.
- *Target.* Whether a customer purchases in the next month (binary).
- *Model.* Monthly-refreshed timing and value features; gradient tree boosting is "the best
  performing method".
- *Data.* More than 10,000 customers and about 200,000 purchases.
- *Headline (claim).* Accuracy 89 %, AUC 0.95 on the test set.
- *Caveat.* No BTYD benchmark despite citing Fader & Hardie and Platzer & Reutterer; binary
  target.

---

## 6. Transformers and attention on customer sequences

### 6.1 Lu & Kannan (2025) — Transformer with heterogeneous-mixture multi-head attention on touchpoint journeys

- *Citation.* Lu, Z. & Kannan, P. K. (2025), "AI for Customer Journeys: A Transformer
  Approach", *Journal of Marketing Research*, DOI 10.1177/00222437251347268; SSRN 4684617
  (earlier title "Measuring the Synergy Across Customer Touchpoints using Transformers")
  (abstract via https://api.semanticscholar.org/graph/v1/paper/DOI:10.1177/00222437251347268).
- *Target.* The next touchpoint and conversion in a multichannel journey (abstract).
- *Model.* Touchpoints as tokens; a heterogeneous-mixture multi-head self-attention that
  weights attention heads per customer.
- *Benchmarks.* Hidden Markov models, point-process models, and LSTMs (abstract).
- *Data.* A large hospitality firm, more than 92,000 users and 500,000 touchpoints
  (https://www.rhsmith.umd.edu/research/ai-customer-journeys-transformer-approach).
- *Headline (claim).* "More accurate predictions ... outperforming" HMM, point-process and
  LSTM models; simulations "further establish the model's superiority" (abstract). A
  conversion AUC of 0.92 against 0.85 for the LSTM appears in search-engine summaries of a
  university press release and was **not verified** in the paper or the UMD page.
- *Code.* Not stated.
- *Caveat.* Touchpoint and conversion sequences, not transaction counts; the full text was
  not accessible (SSRN and SAGE both blocked).

### 6.2 West & Lu (2026) — Transformers for CLV with marketing interventions **[title only]**

MSI Forum, 4 May 2026, "Attention to the Journey: Transformer Models for Customer Lifetime
Value Prediction in the Presence of Marketing Interventions"
(https://www.msi.org/presentation/msi-forum-2026-summary-attention-to-the-journey-transformer-models-for-customer-lifetime-value-prediction-in-the-presence-of-marketing-interventions/).
A search-engine summary says it is benchmarked against RFM-based stochastic models and
HMMs; the page itself is behind a login, so nothing more is attributed. Worth watching: it
is the only Transformer-for-CLV work with a stochastic-model benchmark found.

### 6.3 Yang et al. (2025) — USE-LTV, a Transformer LTV regressor in Baidu ads **[abstract only]**

*CIKM 2025*, DOI 10.1145/3746252.3760876. A Transformer over "uncertain" behaviour
sequences with a loss tailored to a long-tailed exponential LTV distribution; "11.64 % NMAE
improvement for a-year LTV compared to the state-of-the-art method" on Baidu data (abstract
via Semantic Scholar). Revenue regression, no BTYD benchmark, proprietary data.

### 6.4 van Maasakkers, Fok & Donkers (2023) — GRUs for next-basket prediction, and a Transformer follow-up in progress

*Expert Systems with Applications* 212, 118795 (https://personal.eur.nl/dfok/). Product
choice in large assortments with a GRU, not counts. The first author's 2026 dissertation
"Deep Learning Approaches for Customer Analytics" describes "generative transformer"-style
models for baskets and click sequences that "outperform traditional statistical
approaches" with no numbers given
(https://www.eur.nl/en/ese/news/using-ai-models-predict-customer-purchases); a Tinbergen
talk is titled "Market Basket Analysis for Large Assortments with Transformers". No paper
yet.

---

## 7. Industrial LTV regression: loss design rather than architecture

These papers forecast revenue per customer with a feed-forward or tree model. They matter
for the thesis only for how they treat zero-inflation and heavy tails, which is the same
problem a count head faces.

### 7.1 Wang, Liu & Miao (2019) — the zero-inflated lognormal (ZILN) loss

- *Citation.* Wang, X., Liu, T. & Miao, J. (2019), "A Deep Probabilistic Model for Customer
  Lifetime Value Prediction", arXiv:1912.07753, Google (PDF read).
- *Target.* Total spend in the 12 months after a customer's first purchase, excluding that
  purchase (PDF §3.1, §5.1).
- *Model.* A linear model or a two-hidden-layer DNN (64, 32) whose three outputs are the
  return probability and the lognormal μ, σ of spend given return; the ZILN negative
  log-likelihood replaces MSE (PDF §3).
- *Benchmarks.* The same models with MSE; ranking by first-purchase value as a floor.
- *Data.* Kaggle Acquire Valued Shoppers (311K customers; the twenty largest companies;
  cohort first purchasing 2012-03 to 2012-07; 80/20 split) and KDD Cup 1998 (about 200K
  lapsed donors) (PDF §5).
- *Headline (claims).* Spearman correlation with true LTV averages 23.9 % higher than MSE
  for the linear model and 48.0 % for the DNN; normalised Gini improves 23.1 % over the
  first-purchase baseline for DNN-ZILN; decile MAPE is 60–69 % lower than MSE (PDF Tables
  1–3). KDD-98 results are averaged over 50 training runs because of run-to-run variation.
- *Code.* https://github.com/google/lifetime_value (Apache-2.0, 379 stars; archived by the
  owner on 18 April 2026).
- *Caveat.* No probabilistic-model benchmark; new-customer LTV, where RFM carries no
  information; one of the few papers that reports multiple runs.

### 7.2 Chamberlain, Cardoso, Liu, Pagliari & Deisenroth (2017) — embeddings and random forests at ASOS

- *Citation.* *KDD 2017*, 1753–1762, arXiv:1703.02596 (full text read at
  https://ar5iv.labs.arxiv.org/html/1703.02596).
- *Target.* Net sales over the next year; churn = no order in a year.
- *Model.* Random forest on 132 handcrafted features, plus skip-gram customer embeddings
  from co-viewing; DNN and logistic–DNN hybrids tried for churn only.
- *Headline (claims).* Churn AUC 0.798; CLTV Spearman 0.56 (0.46 excluding zero-value
  customers); embeddings give a "significant uplift" shown only in figures; the hybrids did
  not beat the random forest and were judged "commercially unviable" at the size needed.
- *Code.* Not stated.
- *Caveat.* Pre-2018 and single retailer; included because Bauer & Jannach re-implement it
  as a baseline.

### 7.3 Sifa, Runge, Bauckhage & Klapper (2018) — DNN plus SMOTE in a freemium app **[abstract only]**

*HICSS-51*, DOI 10.24251/hicss.2018.115 (abstract via OpenAlex; the ScholarSpace PDF is
behind a bot check). Individual-level LTV in a non-contractual freemium setting with
"common machine learning methods"; SMOTE oversampling "improves prediction performance for
premium and high-value users, especially when used in combination with deep neural
networks". No numbers accessible, no BTYD benchmark.

### 7.4 Platform LTV papers, 2024–2026 — one line each

- **Weng et al. (2024), OptDist**, *CIKM 2024*, arXiv:2408.08585: learns which of several
  candidate output distributions to use per customer; "outperforms state-of-the-art
  baselines" on two public and one private dataset; numbers not on the abstract page
  (https://arxiv.org/abs/2408.08585) **[abstract only]**.
- **Wan et al. (2025), TTF**, AAAI 2026 IAAI track, arXiv:2511.17639: channel-level LTV
  curves at Douyin from short inputs; "MAPEp decreased by 4.3 % and MAPEa by 3.2 %" against
  the deployed model (https://arxiv.org/abs/2511.17639) **[abstract only]**. Aggregate, not
  individual.
- **Chen et al. (2025), GRePO-LTV**, *KDD 2025 ADS track*, arXiv:2506.11037: graph
  representation plus Pareto-optimal multi-task learning for WeChat mini-game LTV where
  purchase rates are "as low as 0.1 %"; no numbers on the abstract page
  (https://arxiv.org/abs/2506.11037) **[abstract only]**.
- **Yuan (2025), SHORE**, arXiv:2506.10487: ordinal classification plus Huber loss for
  LTV-60 in games, "47.91 % relative reduction in prediction error" online; **withdrawn by
  arXiv** for a licensing problem (https://arxiv.org/abs/2506.10487). Do not cite.
- **İşeri, Keskin & Raştak (2025)**, *ICEEECS 2025*, 64–68, DOI 10.5220/0014305900004848:
  Hepsiburada; three-class purchase frequency next month (1, 2–3, 4+); a LightGBM gate
  plus XGBoost regressor reaches accuracy 0.641 vs 0.628 for the best single classifier,
  with no significance test; Pareto/NBD and BG/NBD are discussed but not run
  (https://scitepress.org/publishedPapers/2025/143059/pdf/index.html). The only recent
  paper with a categorical count target, and it uses trees.

---

## 8. Foundation models and LLMs on transaction data

No paper was found that applies a time-series foundation model (Chronos, TimesFM, Moirai)
to per-customer transaction panels, nor one that pits an LLM against Pareto/NBD. What
exists is representation learning on event sequences, evaluated on classification tasks:

- **Skalski et al. (2024)**, "Towards a Foundation Purchasing Model", arXiv:2401.01641:
  generative autoregressive pre-training on card transactions from 180 issuing banks (5.1
  billion transactions), evaluated on fraud and other downstream tasks
  (https://arxiv.org/abs/2401.01641) **[abstract only]**.
- **Polleti et al. (2025)**, "Open Banking Foundational Model", arXiv:2511.12154:
  masked-language-model pre-training on transaction sequences; "outperforms classical
  feature engineering and discrete event sequence methods" in data-scarce settings
  (https://arxiv.org/html/2511.12154v1) **[abstract only]**.
- **PRAGMA (Revolut, 2026)**, arXiv:2604.08649, and the **Large Behavior Model** (Modecrua
  et al. 2026, arXiv:2607.06993): Transformer / LLM pre-training on verbalised or tokenised
  banking and retail histories; evaluated on purchase prediction, basket completion and
  promotion response against "frontier general-purpose language models", numbers not on the
  abstract page (https://arxiv.org/abs/2607.06993) **[abstract only]**.
- **Ergul, Balcisoy & Bozkaya (2025)**, arXiv:2502.15724: LoRA-fine-tuned Mistral predicts
  the next merchant category from bank transactions; F1 above "traditional sequential
  models" in three categories (https://arxiv.org/abs/2502.15724) **[abstract only]**.

None forecasts a count per period, none reports calibration of a count distribution, and
none uses a BTYD benchmark.

---

## 9. Adjacent deep models of purchase incidence (not counted)

- **Xia, Chatterjee & May (2019)**, "Using Conditional Restricted Boltzmann Machines to
  Model Complex Consumer Shopping Patterns", *Marketing Science* 38(4), 711–727: CRBMs for
  intertemporal cross-category purchase incidence on a very large panel
  (https://ideas.repec.org/a/inm/ormksc/v38y2019i4p711-727.html) **[abstract only]**.
  Category incidence, not counts; no BTYD benchmark.
- **Gabel & Timoshenko (2022)**, "Product Choice with Large Assortments: A Scalable
  Deep-Learning Model", *Management Science* 68(3), 1808–1827: a product-choice network with
  bottleneck layers, time-series filters for repurchase intervals and weight sharing;
  validated on loyalty-card data with experimental discounts
  (https://pure.eur.nl/en/publications/product-choice-with-large-assortments-a-scalable-deep-learning-mo/)
  **[abstract only]**.
- **Hruschka (2025)**, "Multicategory choice modeling by recurrent neural nets", *Journal
  of Retailing and Consumer Services* 85, 104310: the three RNN variants against MLPs and
  multivariate logit on holdout binary cross-entropy for weekly category incidence
  (abstract via OpenAlex) **[abstract only]**.
- **Toth, Tan, Di Fabbrizio & Datta (2017)**, SIGIR eCom (https://ceur-ws.org/Vol-2311/paper_6.pdf),
  and **Sheil, Rana & Reilly (2018)**, arXiv:1807.08207: session-level purchase-intent
  classification from clickstreams with mixtures of RNNs and multi-layer RNNs. Not
  customer-base analysis.
- **Batislam-style and Jasek et al. (2019)** comparisons of probabilistic models only
  (https://doaj.org/article/038ea3f90f6245ae9de45c1ef66b4d0c) are outside scope.

---

## 10. Contradictions and recurring patterns

- **Abstract versus table.** Xie (2019) and Equihua et al. (2023) both claim the network
  "performs better" while their own tables show the benchmark winning on the per-customer
  error (MAE, multi-class accuracy) or on proper survival metrics (C-index, Brier). Chen et
  al. (2018) headline the CNN but report it tied with a feature MLP. Grob et al. (2018)
  win on one column and lose on two.
- **Which metric decides.** Bauer & Jannach's S2S wins on RMSE and their GBM on MAE;
  Gadgil et al.'s LightGBM has the worst RMSE and the second-best MAE. A result that flips
  with the metric is a recurring feature of this literature, which is a reason for the
  thesis's single scoring authority.
- **Hybrids versus pure networks.** Chou et al. (2022) report a Lasso-BG/BB beating two
  RNNs; Valendin et al. (2022) and Sarkar & De Bruyn (2026) report an LSTM beating the
  Pareto/NBD and hierarchical-Bayes extensions. Nobody has run both designs on the same
  panel.
- **Where the BTYD model loses most.** Every paper that tabulates it shows the probabilistic
  model losing most on the right tail (top-20 % spenders in Chen et al. 2018, aggregate
  repeat purchases in Xie 2019, long horizons in Chen, Ng et al. 2024) and least on the
  central per-customer error.
- **Timing is hard for everyone.** Simon (2025) finds Pareto/NBD next-purchase timing
  unusable; the neural TPP papers report better likelihoods or AUCs on return time but
  large RMSEs (Grob et al. 2018: 60 days), and Šeputis et al. (2022) report that neural
  TPPs cannot forecast rare events at scale.
- **Variance is rarely reported.** Only Wang et al. (2019, 50 runs) and Bauer & Jannach
  (2021, significance tests) address run-to-run variation; the rest are single fits.

---

## 11. Comparison table

| Study | Target | Model | Benchmarks | Datasets | Headline result (claim) | Code |
|---|---|---|---|---|---|---|
| Chou et al. 2022, EJOR | Repurchase propensity/frequency (discrete) | Lasso on BG/BB outputs + ~100 features | BG/BB, Lasso, two RNNs | Large online retailer (n/r) | Lasso-BG/BB beats BG/BB, Lasso and both RNNs; BG/BB outputs most influential | not stated |
| Xie 2019, arXiv | Holdout repeat-purchase count; active status | FFNN emitting Pareto/NBD rates, likelihood in loss | Pareto/NBD MLE | CDNOW 23,570; E-tailing 24,000; 60/40 by customer | P/NBD MAE 0.690, total 4,105 vs actual 7,634; best NNA MAE 0.896, total 8,526 | not stated |
| Ghosh & Schein 2026, arXiv | n/r ("predictive performance") | Weibull-renewal BTYD, amortised VI | MCMC state of the art | 5M retail customers; 4M donors | 8 min vs 3–4 days, "no appreciable change" | not stated |
| Chen, Keng & Moreno 2018, ICDM-W | Time to next purchase per product (censored) | 2-layer LSTM → Weibull params | Pareto/NBD, WTTE-RNN, RF | CDNOW (1 month holdout); retailer 5 categories, 78 wk train / 4 wk test | ROC-AUC 0.84 vs P/NBD 0.80 on CDNOW | github rubikloud/matrnn |
| Yan & Resnick 2023, QME | Churn-staged CLV, daily refresh | Ensemble generative + discriminative | unnamed | 12 brands, 6 intervals 2020–22 | "consistently surpasses benchmarks" | not stated |
| Bauer & Jannach 2021, TKDD | Profit per period, summed (4 wk daily / 6 mo weekly) | Enc–dec GRU + GBM stack | ACMB/ARMA, BG/NBD, Markov chain, Embeddings-RF | UKRetail 3,284; Children 1.07M | Children RMSE/MAE 11.65/3.14 vs BG/NBD 20.14/8.09 (p<10⁻⁵); UKRetail 64.95/30.80 vs 69.17/41.54 (p<0.2) | bitbucket Josef-Bauer/clv_pred |
| Sarkar & De Bruyn 2026, JAMS | Distribution of future value by simulation | Dual LSTM, Monte Carlo rollout | Pareto/NBD, HB extensions, deep learning (6) | 21,939 donors, 9 years | Best on 19 of 24 metrics; targeting lift up to 8.3 % | not stated |
| Chen, Ng, Smyl & Steininger 2024, arXiv | Revenue per user, 1–26 wk | Dilated multi-cell RNN | BTYD (unnamed), LightGBM, XGBoost | Uber, Meta Quest (n/r) | aSMAPE 4 wk: BTYD 53.6 %, LGB 30.4 %, dRNN 22.9 % | not stated |
| Chen, Guitart et al. 2018, IEEE Big Data | 365-day spend, paying players | MLP (203 features); CNN on raw series | Pareto/NBD, MBG/CNBD-k (+avg/GG) | Age of Ishtaria, 2,505 churned payers | RMSLE: P/NBD 9.35, MBG/CNBD-k 3.46, DNN 1.82, CNN 1.84 | not stated |
| Gadgil et al. 2023, arXiv | Transactions in next 3 months | Stacked RF + XGBoost | BG/NBD, LightGBM, DNN | Online Retail II, 5,850 | RMSE/MAE: BG/NBD 1.62/0.90; XGBoost 1.34/0.83; stacked 1.37/0.82 | not stated |
| Dew & Ansari 2018, Mkt Sci | Daily spend incidence; aggregate daily series | GP propensity model (Stan) | Pareto/NBD, BG/NBD, log-logistic, LPM, SSPM | 2 mobile games, 8,000 + 2,000 users, 30-day holdout | Holdout MAPE 0.24/0.32 vs P/NBD 0.33/0.45, BG/NBD 0.31/0.61 | Stan in appendix; Shiny app |
| Simon 2025, JBE | P/NBD counts, activity, timing | Pareto/NBD MCMC vs MLE vs heuristics | heuristics | n/r | MCMC slightly > MLE; timing forecasts "too large to be used" | not stated |
| Grob et al. 2018, ECML PKDD | Days to next session (censored) | LSTM RMTPP-style survival | last-session gap, RNN-MSE, Cox | ASOS 38,716 / 9,680 users | Non-return AUC 0.796 vs Cox 0.793; concordance 0.739 vs 0.816 | github grobgl/rnnsm |
| Equihua et al. 2023, arXiv | Survival to next purchase; churn at 180 d | LSTM → exponential rate | Cox, individual KM | Retailer 2.8M customers, 17.8M trans. | Accuracy 0.654 vs Cox 0.485; C-index 0.391 vs 0.580 | not stated |
| Manzoor & Akoglu 2017, KDD | Time and category of next purchase | Time-varying multivariate Hawkes | Poisson, TV-Poisson, Hawkes | 199,109 prepaid accounts, 2.8M trans. | Log-lik +30.52 over Poisson; dominates in coupon simulation | github emaadmanzoor/rush |
| Bai et al. 2019, SIGIR | Purchase time of product for user | Demand-aware Hawkes + CNN + self-attention | n/r | 4 commercial datasets | "learns purchase cycles" | n/r |
| Martínez et al. 2020, EJOR | Purchase next month (binary) | Gradient tree boosting | other ML | >10,000 customers, ~200,000 purchases | Accuracy 89 %, AUC 0.95 | not stated |
| Lu & Kannan 2025, JMR | Next touchpoint, conversion | Transformer, heterogeneous-mixture attention | HMM, point process, LSTM | Hospitality, 92,000 users, 500,000 touchpoints | Outperforms all three (AUC 0.92 vs 0.85 LSTM in press summary, unverified) | not stated |
| Yang et al. 2025, CIKM | One-year ad-customer LTV | Transformer + long-tail loss | "state of the art" | Baidu ads | NMAE −11.64 % | not stated |
| Wang, Liu & Miao 2019, arXiv | 12-month spend after first purchase | DNN/linear with ZILN loss | same with MSE; first-purchase rank | Kaggle AVS top-20 companies; KDD Cup 98 | Spearman +48 % (DNN) over MSE; Gini +23.1 % over baseline; 50 runs on KDD-98 | github google/lifetime_value (archived) |
| Chamberlain et al. 2017, KDD | 1-year net sales; churn | RF + skip-gram embeddings; DNN hybrids | handcrafted-feature RF | ASOS (n/r) | Churn AUC 0.798; CLTV Spearman 0.56; hybrids not viable | not stated |
| İşeri et al. 2025, ICEEECS | Purchase-frequency class next month (1 / 2–3 / 4+) | LightGBM gate + XGBoost regressor | own single models (P/NBD, BG/NBD discussed only) | Hepsiburada, "several million rows" | Accuracy 0.641 vs 0.628; 4+ recall 0.33 vs 0.29 | not stated |

n/r = not reported in the accessible text.

---

## 12. What this means for the thesis

- The thesis's design — categorical count head, class-index target, sampling-and-averaging
  rollout against a Pareto/NBD fitted on the same calibration window — has no published
  precedent besides Valendin et al. (2022) itself; Sarkar & De Bruyn (2026) is the nearest
  relative and should be cited as such once the full text is obtained.
- A Transformer on a transaction panel benchmarked against Pareto/NBD would be a first in
  the peer-reviewed literature as far as this search found; Lu & Kannan (2025) and West &
  Lu (2026) are the citations that show the architecture entering marketing, on different
  targets.
- The one count-target ML paper on a public panel, Gadgil et al. (2023), finds BG/NBD within
  0.08 MAE of the best tree ensemble on Online Retail II with no significance test; the
  thesis's bootstrap Δ-of-means protocol is the obvious answer to that weakness and should
  be presented that way.
- Expect the benchmark to lose on the right tail and the aggregate and to hold on the
  per-customer centre (Xie 2019; Chen et al. 2018; Chen, Ng et al. 2024); report both the
  per-customer RMSE and the aggregate bias, which `compute_forecast_metrics` already does.
- Several papers' conclusions flip between metrics; keeping one scoring authority and
  reporting it on every model is a methodological contribution worth stating explicitly.
- Chou et al. (2022) is the strongest counter-evidence to "deep beats BTYD": a BG/BB-fed
  Lasso beating two RNNs on a large retailer. A hybrid arm (BTYD outputs as covariates) is
  the cheapest extension with a literature anchor.
- Run-to-run variance is almost never reported in this literature (Wang et al. 2019 is the
  exception); the thesis's replication-across-seeds reporting fills a documented gap.
- Simon (2025) establishes that Pareto/NBD timing forecasts are unusable in practice; a
  per-period count rollout sidesteps timing and should be framed that way rather than as a
  timing model.
- The neural TPP literature (Grob et al. 2018; Šeputis et al. 2022) reports poor rare-event
  and long-gap behaviour, which argues against adding a point-process arm to the thesis.
- Time-series foundation models have no customer-panel application in the literature;
  mentioning this absence is enough, and testing Chronos or TimesFM zero-shot would be a
  separate study, not a benchmark the thesis is missing.
