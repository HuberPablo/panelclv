# Cohort analysis

How the CLV literature groups customers into cohorts, and which of those groupings our
real panels can support.

**Vocabulary.** In this project a **cohort** is the set of customers a model is fit on
(`CONTEXT.md`). Each real panel is one *acquisition cohort*: every customer whose first
purchase falls in a fixed window (`docs/datasets.md`, "Acquisition cohort"). The
literature uses "cohort" in two senses:

- **acquisition-time cohorts**, defined by *when* a customer was acquired;
- **characteristic cohorts**, defined by *how* the customer was acquired or what the
  first purchase looked like: channel, promotion, first spend, first category,
  demographics.

This doc calls a variable that splits a cohort into groups a **cohort dimension**. It is
not a behavioural cluster: a behavioural cluster is computed from calibration behaviour
(`CONTEXT.md`). A cohort dimension is fixed at acquisition.

**Sources.** Two literature searches, run on 2026-10-14, read primary sources: author
PDFs, working papers, SSRN, arXiv, and the PDFs in `../Papers/`. **[abstract only]**
marks a paper whose full text could not be reached; nothing beyond its abstract is
attributed to it. **[secondary]** marks a claim known only through another source. Where
a working-paper version was read, section and page numbers refer to that version.

---

## 1. Literature review

### 1.1 Starting point: the cohorts in our real panels

All seven panels follow the single-cohort design of the source papers. Each customer
belongs to the cohort whose first purchase falls in the window, and anyone who bought
earlier is excluded as left-censored.

| Panel | Acquisition window | Span (weeks) | N | Cohort dimensions in the panel |
|---|---|---|---|---|
| electronics | 1998-12-02 .. 1999-11-30 | 52 | 3,755 | acquisition week; `first_spend`; `income`, `gender`, `age_band`, `children` |
| gift | 2001-03-01 .. 2001-05-31 | 13 | 918 | acquisition week; `first_spend`; `first_method` (channel); `age_code`, `income_code` |
| multichannel | 2005-01-01 .. 2005-03-31 | 13 | 1,379 | acquisition week; `first_spend`; `first_channel`, `first_division` |
| books | 2008-01-01 .. 2008-03-31 | 13 | 1,218 | acquisition week; `first_spend`; `first_category` |
| apparel | 1996-08-11 .. 1996-12-31 | 19 | 14,358 | acquisition week; `first_spend` |
| vod | 2011-01-01 .. 2011-03-31 | 13 | 4,843 | acquisition week; `first_spend` |
| game | 2008-01-01 .. 2008-03-31 | 14 | 6,173 | acquisition week; `first_spend` |

The table shows three things.

- **Only one acquisition cohort per panel.** No panel holds a second vintage, so the
  multi-cohort questions in §1.2 (pooling, cross-cohort shifts) cannot be asked of the
  panels as built. The raw files span several years (apparel 10, electronics 6,
  multichannel 11.5), so later acquisition cohorts could be built with the same rule.
- **Within-cohort acquisition timing exists.** Customers enter across a 13- to 52-week
  window, and the panel records the week each one entered.
- **The characteristic dimensions are the ones the literature uses most** (§1.3): first
  spend in every panel, first-purchase channel in two, first category in one, and
  demographics in two.

**A first look.** On the `2y` panels (104 calibration weeks, 52 holdout weeks), mean
holdout transactions per customer differ by cohort dimension as follows. Groups with
fewer than 30 customers are omitted. Acquisition month is week of first purchase ÷ 4.33.
Holdout is the same calendar year for every customer, so holdout figures are not
confounded by exposure the way calibration totals are.

| Panel | Dimension | Holdout mean, lowest → highest group | Ratio |
|---|---|---|---|
| multichannel | `first_division` | 0.04 (id 5) → 0.25 (id 1) | 7.0× |
| multichannel | `first_channel` | 0.05 (web) → 0.26 (phone) | 5.2× |
| gift | `age_code` | 0.20 (unmatched) → 1.02 (code 6) | 5.1× |
| gift | `income_code` | 0.21 (unmatched) → 0.93 (code 8) | 4.4× |
| electronics | `age_band` | 0.28 (missing) → 0.88 (55–64) | 3.2× |
| gift | `first_method` | 0.23 (internet) → 0.53 (store) | 2.3× |
| electronics | acquisition month | 0.48 → 0.93 | 1.9× |
| electronics | `income` | 0.47 (missing) → 0.82 (code 9) | 1.8× |
| books | `first_category` | 1.20 → 2.28 | 1.9× |
| vod | `first_spend` tercile | 5.72 (low) → 7.98 (mid) | 1.4× |
| gift | `first_spend` tercile | 0.37 (high) → 0.54 (low) | 1.4× |
| apparel | acquisition month | 0.94 (last) → 1.19 (first) | 1.3× |
| apparel | `first_spend` tercile | 0.97 (low) → 1.23 (high) | 1.3× |
| electronics | `first_spend` tercile | 0.68 → 0.71 | 1.1× |
| books, vod | acquisition month | — | 1.1× |

Three readings guided the literature search. The search confirmed each of them (§1.5).

1. **Channel is the largest effect wherever it is recorded.** In multichannel and gift,
   web- or internet-acquired customers are the least active in holdout.
2. **Missing demographics are informative.** The *unmatched* or *missing* code is the
   least active group in every demographic column of electronics and gift.
3. **First spend and acquisition timing matter less**, and point in different
   directions across panels.

These are descriptive splits, not effects: each dimension is looked at alone, and none
of the gaps is tested.

### 1.2 Acquisition-time cohorts

**1.2.1 The standard BTYD design: one cohort in a fixed window.** Almost every
benchmarking paper fits and forecasts on a single cohort whose first purchase falls in
one calendar window. The window is usually a quarter.

| Paper | Cohort | Window |
|---|---|---|
| Fader, Hardie & Lee (2005a, §7; 2005b, §1) | CDNOW, 23,570 first-time buyers; estimation on a 1/10 systematic sample of 2,357 | Q1 1997 |
| Batislam, Denizel & Filiztekin (2007, Table 1) | Turkish grocer: Cohort1, 5,479 customers; Cohort2, 6,276 | Aug–Oct 2001; Nov 2001 – Jan 2002 |
| Platzer & Reutterer (2016) [secondary, via BTYDplus `groceryElog`] | Online grocer, 1,525 customers | Q1 2006 |
| Bachmann, Meierer & Näf (2021, Tables 5, 10, 15) | Multichannel 1,402; electronics 829; sporting goods 1,071 | Jan–Mar 2005, 1999 and 2008 respectively |
| Valendin et al. (2022, §3.1, Table 3) | Eight datasets | Q1 (CDNOW, groceries), H1 (charity, 21,166), one year (electronics 3,782; blood donations 11,887; yogurt 31,216; sunscreen 7,794) |
| Meierer et al. (2026, §4) | CLVTools `apparelTrans`, 600 customers | One day, 2005-01-02 |

The window is not chosen to compare vintages. Its purpose is to give every customer a
known birth date and a common observation window, so that recency and frequency are
comparable. Fader, Hardie & Lee name this as a limitation. They "looked at only one
cohort", and managers "need to run the model across multiple cohorts", grouping "by date
of initial purchase … or geography, demographics, or mode of acquisition" (2005b, §5).

**1.2.2 Quasi cohorts.** When the data do not record acquisition dates, a left filter
stands in: the customer's first purchase falls in the window *after a washout period
with no activity*.

| Paper | Washout before the window |
|---|---|
| Batislam et al. (2007, working paper) | 13 months: transactions before August 2001 are dropped |
| Platzer & Reutterer (2016) [secondary]; Valendin et al. (2022, §3.1) | 2 years; first purchase in Q1 2006 |
| Kim, McCarthy & Lee (2026, §4, fn. 2) | 1 year; 2016 is a washout year |

`docs/dataset-characterisation-literature.md` §2.2 gives Batislam's washout as 11
months. The working paper says 13.

**1.2.3 Multi-cohort accounting: valuing the whole base.** In valuation work the cohort
is the unit of account. The customer base is the sum of cohorts acquired in successive
periods.

- **Gupta, Lehmann & Stuart (2004)** sum the lifetime value of cohort *k*, acquired at
  time *k*, over all *k* (eqs 3–7, Table 1). They work from quarterly public filings,
  so the cohorts are in effect quarterly. Every cohort is given the same margin and
  retention pattern; the authors note that relaxing this "increases the model
  complexity significantly".
- **McCarthy, Fader & Hardie (2017)** define a matrix C(m, m′): the number of customers
  acquired in month *m* still active in month *m′*. Monthly cohorts are inferred from
  quarterly disclosures. Retention is a Weibull hazard with gamma heterogeneity, plus
  seasonal and macro covariates, but no cohort-specific parameter: cross-cohort effects
  are left out because aggregate data cannot identify them (§6).
- **McCarthy & Fader (2018)** fit weekly acquisition cohorts to quarterly disclosures
  for Overstock and Wayfair. For a customer acquired in Q1 2017 they report post-
  acquisition profit against acquisition cost: $47 vs $38 at Overstock and $59 vs $69
  at Wayfair (Fig. 7).
- **The Customer Cohort Chart (C3)** [secondary] stacks each period's revenue or active
  customers by acquisition cohort. It separates growth from the existing base from
  growth from acquisition. It is the headline analysis of Fader, Hardie & Ross's *The
  Customer-Base Audit* (2022) and of McCarthy & Fader (2020, HBR), and is usually drawn
  with annual cohorts. Neither text was read; verify before citing details.
- **Kim, McCarthy & Lee (2026)** define cohorts by calendar week of acquisition across
  966 firms. Each cohort-week's revenue is decomposed into acquisitions × repeat orders
  per cohort member × average order value (eq. 1).

**1.2.4 Granularity.** The cohort width follows the setting's natural cycle:

| Width | Setting | Papers |
|---|---|---|
| Year | Annual giving, campaign cycles | Fader, Hardie & Shang (2010): six cohorts, 1995–2000; Gopalakrishnan et al. (2017): eleven cohorts, 1990–2000; Fader & Hardie (2010), Table 1; C3 practice |
| Quarter | Non-contractual retail benchmarks | CDNOW, Bachmann et al., Batislam et al., Gupta et al. |
| Month | Subscriptions, billing cycles | Schweidel, Fader & Bradlow (2008); McCarthy, Fader & Hardie (2017) |
| Week | Card panels, recent valuation work | McCarthy & Fader (2018); Oblander & McCarthy (2023); Kim et al. (2026) |
| Event | A structural break | Tudoran et al. (2024): acquired before vs during COVID |

Gopalakrishnan et al. chose annual cohorts explicitly to avoid seasonal distortion
(2011 working paper, citing Fader, Hardie & Shang 2010).

**1.2.5 How multiple cohorts are modelled.** From most to least common:

1. **Pool all cohorts** into one model.
   - Fader, Hardie & Shang (2010, §3.2): the pooled parameters are "remarkably
     similar" to the 1995-only fit.
   - Fader & Hardie (2007) call pooling the natural start.
2. **Fit one model per cohort.**
   - Fader, Hardie & Lee (2005a, §8) advise it: apply the model "separately to customer
     cohorts defined by the time (e.g., quarter) of acquisition, acquisition channel,
     etc."
   - CLVTools calls this "common practice" (Meierer et al. 2026, §4).
   - The cost, per Fader & Hardie (2007): "every new cohort has one less period of
     information than its temporal predecessor".
3. **Cohort covariates.**
   - Schweidel et al. (2008) scale the hazard by a cubic in the cohort index.
   - Kim et al. (2026) benchmark a Pareto/NBD with a cohort-number covariate.
4. **A hierarchy or changepoint across cohorts.** Gopalakrishnan, Bradlow & Fader
   (2017).
5. **An age-period-cohort decomposition.** Oblander & McCarthy (2023) separate cohort,
   tenure and calendar effects with three-way fixed effects.

Bachmann et al. (2021, future research) note that latent-attrition studies "assume that
the cohorts are independent of each other", and call cohort fixed effects in a pooled
model "not an optimal solution".

**1.2.6 Does vintage matter? The evidence is mixed.**

- **Shifts found:**
  - Gopalakrishnan et al. (2017): a changepoint at the 1998 donor cohort, driven by
    dropout. The 1998–99 cohorts drop out more, and more heterogeneously. The authors
    read it as market saturation: the best prospects were acquired first. Pooling across
    the shift causes "old data" aggregation bias [published version: abstract only;
    details from the 2011 working paper].
  - Tudoran et al. (2024): separate pre-COVID and COVID-cohort models beat a pooled
    model out of sample. About 67% of COVID-acquired customers stopped within a year.
  - Kim et al. (2026): early and recent weekly cohorts show "markedly different decay
    rates".
- **No shift found:**
  - Schweidel et al. (2008): no cross-cohort term enters any winning model across
    seven telecom services.
  - Fader, Hardie & Shang (2010): the cohorts pool.
- **Seasonal vintage:**
  - Padilla & Ascarza (2021, §3.3) find holiday-acquired customers transact less.
  - Meierer et al. (2026) suggest that Q4 / Black Friday cohorts "may have worse
    retention", as an illustration only.
  - No other peer-reviewed study isolating season-of-acquisition effects was found.

### 1.3 Characteristic cohorts

Fader, Hardie & Lee (2005a) and Fader & Hardie (2007) both list acquisition channel and
product class beside acquisition date as cohort definitions. The literature on each
characteristic follows, ordered by how often it is studied.

**1.3.1 Acquisition channel.** This is the most studied characteristic. Its main effect
is on **retention**.

- **Verhoef & Donkers (2005)**, Dutch insurer, 3,317 customers acquired H1 1999.
  - Every channel retains worse than co-insurance (an employer or interest-group deal).
  - Predicted retention against the service-type average (Table 3): co-insurance +8.2
    pp, TV/radio −9.8, direct mail −7.6, website +0.1, word of mouth +1.4.
  - Channel explains little of cross-buying (pseudo-R² 0.05).
- **Villanueva, Yoo & Hanssens (2008)**, web-hosting firm, aggregate VAR.
  - Word-of-mouth customers bring "nearly twice as much long-term value" as
    marketing-induced ones.
  - Simulated 10-week value per 1,000 new customers: $20,554 vs $10,419 (MSI working
    paper, Table 5).
- **Schmitt, Skiera & Van den Bulte (2011)**, German bank, 5,181 referred and 4,633
  non-referred customers.
  - Referred customers' churn hazard is 18% lower after controls, and the gap does not
    erode.
  - Their margin is about 25% higher at first, but the gap closes after about 857 days.
  - Six-year CLV is at least 16% higher (Table 2).
- **Chan, Wu & Xie (2011)** [abstract only]: customers acquired through Google search
  ads transact more often than customers from other channels.
- **Online vs offline first purchase.** The sign depends on context.
  - Bachmann et al. (2021, Table 6), on the multichannel data our panel is built from:
    first-purchase channel has no effect on purchasing (γ = 0.210, n.s.), but
    online-acquired customers "churn faster" (γ_attr = 1.907, p < .01).
  - Meierer et al. (2026), CLVTools apparel data: offline-acquired customers buy more
    (0.624, p < .001) but drop out faster (0.791, p < .01).
  - Padilla & Ascarza (2021): the predicted top-decile spenders are more often
    offline-acquired.

**1.3.2 Acquisition promotion, price or free trial.** In subscription settings the
effect is consistently negative. In retail it is mixed.

- **Lewis (2006)** [abstract only], newspaper and online grocer: acquisition discount
  depth lowers repeat buying and customer asset value. Customers acquired at a 35%
  discount are worth about half as much.
- **Datta, Foubert & Van Heerde (2015)**, digital TV, 16,512 customers.
  - Free-trial customers churn more: 48% vs 34% over the window.
  - Expected survival is 19.66 vs 25.73 months.
  - Net CLV is 59% lower: €101 vs €245.
  - They respond more to marketing.
- **Anderson & Simester (2004)**, catalogue field experiments.
  - Prospects acquired with deeper discounts later bought *more* units: an index of 114
    and 134 in two studies.
  - But they bought cheaper items: price index 96 and 90 (Tables 5, 7, 8).
- **Padilla & Ascarza (2021)**: a discounted first purchase predicts more repeat
  purchases, unless the basket held no new product.

**1.3.3 First-purchase spend.** This is the most BTYD-native first-impression variable.
Its effect is consistently on **purchase frequency, not dropout**.

- **Abe (2009)**, CDNOW (2,357 customers, Table 3).
  - The initial purchase amount raises the purchase rate λ: 3.21 per $1,000, 95%
    interval [1.59, 4.90].
  - It has no effect on dropout: −0.21, [−2.52, 1.98].
  - In a CD-chain dataset, also positive on λ: 0.38, [0.13, 0.63] (Table 10).
- **Voigt & Hinz (2016)**, three freemium platforms.
  - Each extra € of initial purchase multiplies 365-day CLV by 1.012–1.035.
  - Each extra day until the first purchase lowers it.
- **Padilla & Ascarza (2021)**: the lowest quartile of price paid buys less often.
- **Fader, Hardie & Lee (2005b, §2.2)**: on the same CDNOW cohort, *average repeat*
  spend is nearly uncorrelated with frequency (0.11). The link is from the *first*
  purchase, not from spend in general.

**1.3.4 First-purchase product or category.** The effects are real, but their direction
depends on the product.

- **Verhoef & Donkers (2005)**: customers whose first product was housing insurance
  defect less.
- **Padilla & Ascarza (2021, §3.3)**, beauty retailer, 13,473 customers: a first basket
  with a new product or with hair care predicts more repeat purchases.
- **"Harbinger" non-core first products** predicting non-return (Anderson, Chen, Liu &
  Simester) [secondary, press summaries only].

**1.3.5 Demographics and geography.** These are the most commonly *available*
covariates and the weakest and least consistent ones.

- **Abe (2009)**: age and gender have no effect in either of two datasets.
- **Bachmann et al. (2021, Table 11)**, on the electronics data our panel is built from:
  - gender has no effect on either process;
  - higher income lowers attrition (γ_attr = −0.978, p < .05).
- **Verhoef & Donkers (2005)**: women and older, wealthier and less urban customers
  defect less.
- **Meierer et al. (2026)**, apparel: women buy more and drop out less.
- **Chamberlain et al. (2017)**, ASOS: demographics carry 7.8% of random-forest feature
  importance, against 60% for purchase history (Table 1). Country is the strongest
  demographic.
- **Valendin et al. (2022)**: their Extended LSTM embeds the electronics demographics.
  They report that "the most important signals are already contained in the basic
  transaction log"; covariates add "further polish".

No paper read treats a *missing* demographic as a group of its own.

**1.3.6 Other characteristics.** Each appears in a single paper.

- **Payment method and time to first purchase.** Credit-card payers and customers who
  pay sooner are worth more (Voigt & Hinz 2016).
- **Referral × demographics.** The referral advantage holds for men and women and for
  ages up to 55, but not for customers over 55 (Schmitt et al. 2011, Table 3).

### 1.4 Behavioural segments are not cohorts

Customers are also grouped by behaviour after acquisition.

- **RFM cells:** Fader, Hardie & Lee (2005b), Fig. 1, recency × frequency in weeks
  1–39.
- **"One-time buyers" vs "power users":** Vanderveld et al. (2016) [abstract only].
- **Customers at risk or with opportunity:** Valendin et al. (2022).

These groupings are built from the calibration history that a BTYD model already
conditions on through (x, t_x, T). In the literature they are model outputs or
evaluation slices. In this project the same distinction separates a behavioural cluster
(an input computed from calibration) and the customer groups of the segment analysis
(an evaluation slice) from a cohort dimension (fixed at acquisition).

### 1.5 Summary: the most common cohorts in CLV

| Rank | Cohort | Typical definition | Main effect in the literature | In our panels |
|---|---|---|---|---|
| 1 | **Acquisition window** (single cohort) | First purchase in one quarter; also half-year, year or a quasi-cohort washout | Not compared. It is the unit of analysis | All seven, one cohort each |
| 2 | **Acquisition vintage** (multi-cohort) | Year (donations, C3), month (subscriptions), week (card panels) | Mixed: shifts in donations, COVID and card panels; none in telecom | Not built; the raw files could support it |
| 3 | **Acquisition channel / mode** | Store vs web vs phone vs mail; referral vs paid; search ads | Mostly **retention**. Referral and word of mouth retain better; online-acquired churn faster at the catalogue merchant | gift (`first_method`), multichannel (`first_channel`, `first_division`). The largest holdout gaps in §1.1 |
| 4 | **Acquisition promotion / free trial** | Discount depth, free trial | Negative on CLV in subscriptions; mixed in retail | Not recorded |
| 5 | **First-purchase spend** | Amount or basket size of the first purchase | Positive on **frequency**, not on dropout | All seven (`first_spend`). Small, mixed gaps |
| 6 | **First-purchase category** | Product class of the first basket | Real, but product-specific | books (`first_category`) |
| 7 | **Season of acquisition** | Holiday or Q4 acquisition | Lower activity (one study) | Partly: electronics' window opens in December |
| 8 | **Demographics / geography** | Age, gender, income, region | Weak and inconsistent; marginal once history is observed | electronics, gift. Missing codes are the least active group |

Taken together, the literature points to a split between two kinds of characteristic:

- characteristics of **how a customer was recruited** (channel, referral, discount,
  trial) mostly move **dropout**;
- characteristics of **what the customer did on day one** (first spend, first category)
  mostly move **purchase frequency**.

Our panels agree on channel. Their largest within-cohort gaps sit on the first-purchase
channel: multichannel 5.2×, gift 2.3×. Web or internet acquisition is the least active
group in both, as Bachmann et al. found on the multichannel data. They agree on income
in electronics. They disagree on first spend, which separates customers in apparel and
vod but not in electronics or books, and runs the other way in gift.

---

## References

- Abe, M. (2009). "Counting your customers" one by one: A hierarchical Bayes extension
  to the Pareto/NBD model. *Marketing Science* 28(3), 541–553.
- Anderson, E. T. & Simester, D. I. (2004). Long-run effects of promotion depth on new
  versus established customers: Three field studies. *Marketing Science* 23(1), 4–20.
- Bachmann, P., Meierer, M. & Näf, J. (2021). The role of time-varying contextual
  factors in latent attrition models for customer base analysis. *Marketing Science*
  40(4), 783–809.
- Batislam, E. P., Denizel, M. & Filiztekin, A. (2007). Empirical validation and
  comparison of models for customer base analysis. *IJRM* 24(3), 201–209.
- Chamberlain, B. P., Cardoso, Â., Liu, C. H. B., Pagliari, R. & Deisenroth, M. P.
  (2017). Customer lifetime value prediction using embeddings. *KDD '17*, 1753–1762.
- Chan, T. Y., Wu, C. & Xie, Y. (2011). Measuring the lifetime value of customers
  acquired from Google search advertising. *Marketing Science* 30(5), 837–850.
  [abstract only]
- Datta, H., Foubert, B. & Van Heerde, H. J. (2015). The challenge of retaining
  customers acquired with free trials. *JMR* 52(2), 217–234.
- Fader, P. S. & Hardie, B. G. S. (2007). How to project customer retention. *Journal
  of Interactive Marketing* 21(1), 76–90.
- Fader, P. S. & Hardie, B. G. S. (2007). Incorporating time-invariant covariates into
  the Pareto/NBD and BG/NBD models. Note 019, brucehardie.com.
- Fader, P. S. & Hardie, B. G. S. (2010). Customer-base valuation in a contractual
  setting: The perils of ignoring heterogeneity. *Marketing Science* 29(1), 85–93.
- Fader, P. S., Hardie, B. G. S. & Lee, K. L. (2005a). "Counting your customers" the
  easy way: An alternative to the Pareto/NBD model. *Marketing Science* 24(2), 275–284.
- Fader, P. S., Hardie, B. G. S. & Lee, K. L. (2005b). RFM and CLV: Using iso-value
  curves for customer base analysis. *JMR* 42(4), 415–430.
- Fader, P. S., Hardie, B. G. S. & Ross, M. (2022). *The Customer-Base Audit*. Wharton
  School Press. [secondary]
- Fader, P. S., Hardie, B. G. S. & Shang, J. (2010). Customer-base analysis in a
  discrete-time noncontractual setting. *Marketing Science* 29(6), 1086–1108.
- Gopalakrishnan, A., Bradlow, E. T. & Fader, P. S. (2017). A cross-cohort changepoint
  model for customer-base analysis. *Marketing Science* 36(2), 195–213. [published
  version: abstract only]
- Gupta, S., Lehmann, D. R. & Stuart, J. A. (2004). Valuing customers. *JMR* 41(1),
  7–18.
- Kim, McCarthy & Lee (2026). Forecasting revenue with its customer-base drivers.
  arXiv:2608.02911.
- Lewis, M. (2006). Customer acquisition promotions and customer asset value. *JMR*
  43(2), 195–203. [abstract only]
- McCarthy, D. M. & Fader, P. S. (2018). Customer-based corporate valuation for publicly
  traded non-contractual firms. *JMR* 55(5).
- McCarthy, D. M. & Fader, P. S. (2020). How to value a company by analyzing its
  customers. *Harvard Business Review* 98(1). [secondary]
- McCarthy, D. M., Fader, P. S. & Hardie, B. G. S. (2017). Valuing subscription-based
  businesses using publicly disclosed customer data. *Journal of Marketing* 81(1),
  17–35.
- Meierer, M., Bachmann, P., Näf, J., Schilter, P. & Algesheimer, R. (2026). Estimating
  individual customer lifetime values with R: The CLVTools package. arXiv:2602.09845.
- Oblander, E. S. & McCarthy, D. M. (2023). Frontiers: Estimating the long-term impact
  of major events on consumption patterns: Evidence from COVID-19. *Marketing Science*
  42(5), 839–852. [published version: abstract only]
- Padilla, N. & Ascarza, E. (2021). Overcoming the cold start problem of CRM using a
  probabilistic machine learning approach. *JMR* 58(5), 981–1006.
- Platzer, M. & Reutterer, T. (2016). Ticking away the moments: Timing regularity helps
  to better predict customer activity. *Marketing Science* 35(5), 779–799. [secondary]
- Schmitt, P., Skiera, B. & Van den Bulte, C. (2011). Referral programs and customer
  value. *Journal of Marketing* 75(1), 46–59.
- Schweidel, D. A., Fader, P. S. & Bradlow, E. T. (2008). Understanding service
  retention within and across cohorts using limited information. *Journal of
  Marketing* 72(1), 82–94.
- Tudoran, A. A. et al. (2024). Understanding consumer behavior
  during and after a pandemic: Implications for CLV prediction models. *Journal of
  Business Research* 174, 114527.
- Valendin, J., Reutterer, T., Platzer, M. & Kalcher, K. (2022). Customer base analysis
  with recurrent neural networks. *IJRM* 39(4), 988–1018.
- Vanderveld, A., Pandey, A., Han, A. & Parekh, R. (2016). An engagement-based customer
  lifetime value system for e-commerce. *KDD '16*, 293–302. [abstract only]
- Verhoef, P. C. & Donkers, B. (2005). The effect of acquisition channels on customer
  loyalty and cross-buying. *Journal of Interactive Marketing* 19(2), 31–43.
- Villanueva, J., Yoo, S. & Hanssens, D. M. (2008). The impact of marketing-induced
  versus word-of-mouth customer acquisition on customer equity growth. *JMR* 45(1),
  48–59.
- Voigt, S. & Hinz, O. (2016). Making digital freemium business models a success:
  Predicting customers' lifetime value via initial purchase information. *Business &
  Information Systems Engineering* 58(2), 107–118.
