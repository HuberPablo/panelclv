import numpy as np, pandas as pd
from autoseqmodels.data_prep.loader import _coerce_merge_key

# --- Dates (change these to re-define the train / holdout windows;
#     every downstream cell derives its T_CAL / T_HOLD from here).
training_start, training_end = pd.Timestamp("1999-01-01"), pd.Timestamp("2000-12-31")
holdout_start,  holdout_end  = pd.Timestamp("2001-01-01"), pd.Timestamp("2002-12-31")

WEEKS_PER_YEAR = 52
T_CAL  = (training_end.year - training_start.year + 1) * WEEKS_PER_YEAR
T_HOLD = (holdout_end.year  - holdout_start.year  + 1) * WEEKS_PER_YEAR

# --- Load + clean
tx_df, cov_df = loader.load_table(
    path="Datasets/Electronics_Retailer_data.Rdata",
    r_base_object_name="mydata", r_covariates_object_name="covariates.dynamic")
tx_df["Id"], cov_df["Id"] = _coerce_merge_key(tx_df["Id"], cov_df["Id"])
tx_df["Date"]      = pd.to_datetime(tx_df["Date"])
cov_df["Cov.Date"] = pd.to_datetime(cov_df["Cov.Date"])

# --- Cohort: first purchase <= training_end
cohort = tx_df.groupby("Id")["Date"].min()
cohort = cohort[cohort <= training_end].index

# --- Weekly transaction counts on a full (Id x year x week) grid
def add_yw(df, col):
    df["year"] = df[col].dt.year
    df["week"] = (df[col].dt.dayofyear // 7).clip(upper=WEEKS_PER_YEAR - 1)
add_yw(tx_df, "Date")
weekly = (tx_df[tx_df["Id"].isin(cohort)]
          .groupby(["Id","year","week"]).size().reset_index(name="Transactions"))
years, weeks = range(training_start.year, holdout_end.year+1), range(WEEKS_PER_YEAR)
panel = pd.MultiIndex.from_product([cohort, years, weeks],
                                    names=["Id","year","week"]).to_frame(index=False)
panel = panel.merge(weekly, on=["Id","year","week"], how="left").fillna({"Transactions": 0})

# Merging
# --- Merge covariates (Gender/Income static, high.season time-varying)
add_yw(cov_df, "Cov.Date")
cov_w = (cov_df[cov_df["Id"].isin(cohort)]
         .groupby(["Id","year","week"], as_index=False)
         .agg({"Gender":"first", "Income":"first", "high.season":"max"}))
panel = panel.merge(cov_w, on=["Id","year","week"], how="left").sort_values(["Id","year","week"])
for c in ["Gender","Income","high.season"]:
    panel[c] = panel.groupby("Id")[c].ffill().bfill()



