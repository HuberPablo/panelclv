# 13 — The cohort rule is stated more narrowly than it behaves

**Status:** ready-for-agent

Three places describe the filter as excluding one set of customers. It excludes a strictly
larger one, and on synthetic panels the difference is most of the cohort.

## Doc claim

`CONTEXT.md:20-22`:

> **Cohort**: The set of customers a model is fit on. **Customers first seen only in the
> holdout are excluded**, so every model sees the same cohort.

`src/panelclv/data_preparation/panel_dataset.py:556-560`:

> `train_panel` is already the calibration slice (`period_start <= training_end`), so a
> positive total over it is **exactly** "first purchase <= training_end". **Customers first
> seen only in the holdout** sum to 0 here and are excluded …

`docs/feature_engineering.md:410-414` states the equivalence outright:

> Keeps only customers with at least one transaction during calibration — **equivalently**,
> first purchase ≤ `training_end`. … Customers first seen in the holdout are unknown at
> forecast time and would otherwise present the model with an all-zero history.

That "equivalently" is the assumption under audit: it holds for a panel derived from a
transaction log, and not for one where a customer can exist with no transactions at all.

## Code reality

`select_active_cohort` (`src/panelclv/data_preparation/panel_dataset.py:553-572`) drops
**every** customer whose calibration transactions sum to zero:

```python
totals = train_panel.groupby(id_col)[target_col].sum()
active = totals[totals > 0].index
```

On a panel built *from a transaction log*, "zero in calibration" and "first seen in the
holdout" coincide, which is presumably where the wording comes from. On a **panel**, as
`CONTEXT.md:11-13` defines one — "one row per customer per period, covering every period in
the window whether or not the customer transacted" — they do not. A customer present
throughout with zero transactions in calibration is dropped and was never "first seen in the
holdout"; a never-buyer is dropped and is never seen at all.

That is not hypothetical here. `notebooks/Pareto_Datasets.ipynb` prints, on a synthetic
Pareto/NBD panel:

> never-buyers (0 tx all-window): 465 / 1000 -- dropped later by require_calibration_activity

So on the synthetic grids — the Pareto/NBD comparison the thesis leans on — the filter
removes 46% of the generated customers, none of whom match the documented description.

## Why the precision matters

The synthetic panels exist to vary a *generating* parameter and watch the error move
(`CONTEXT.md`, "Grid"). Silently conditioning the cohort on calibration activity changes the
churn and never-buyer mix each cell is actually scored on, relative to the parameters the
cell was generated with. That is a fine and deliberate choice — it is the Valendin cohort
rule, applied to both windows so every model sees the same customers — but it should be
stated as what it is.

## Fix

State the rule as the code implements it, in all three places:

> Customers with no transaction anywhere in the calibration window are excluded — which on a
> real panel is "first seen only in the holdout", and on a synthetic panel also removes
> never-buyers.

Then add one clause on the consequence for the synthetic grids, since that is where the two
readings diverge, and where a reader is most likely to be surprised.

## Verified alongside

The other half of the `CONTEXT.md` sentence — "so every model sees the same cohort" — does
hold: the filter is applied to `train_panel` and `holdout_panel` together
(`panel_dataset.py:808-810`), and Pareto/NBD is fitted on the same filtered `train_panel`
and `ids` (`studies/runner.py:226-230`).
