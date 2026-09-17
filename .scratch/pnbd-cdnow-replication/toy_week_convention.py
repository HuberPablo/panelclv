"""Where does customer time 0 sit? A toy panel run through the real `_build_cbs`.

`docs/pareto-nbd-cdnow-replication.md` §9(b) suspected the Pareto/NBD forecast window was
one period early. It is not. `t_x` and `T_cal` are both differences of week LABELS, so
whatever within-week anchor you assume cancels out of both, and the arithmetic is exact
under the convention that a week's transactions happen at its end.

This prints the toy that settles it: four calibration weeks, two holdout weeks, one
customer active in calibration weeks 1 and 3. Read the output twice —

  * anchor = end of the week   -> time 0 is 2020-01-13, T_cal=3 ends at 2020-02-03 (the
                                  end of calibration), and column 0 is holdout week 1. OK.
  * anchor = start of the week -> time 0 is 2020-01-06, T_cal=3 ends at 2020-01-27, and
                                  column 0 lands on the last calibration week. Not OK.

The code implements the first. `check_window_alignment.py` prints the same thing for the
four real panels; it is written in start-of-week coordinates, so its "shift = 1 period"
column is the second reading, not a defect.
"""
import sys
import pandas as pd
sys.path.insert(0, "src")
from panelclv.benchmarks.pareto_nbd import _build_cbs

weeks = pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27",   # calibration
                        "2020-02-03", "2020-02-10"])                              # holdout
# One customer: bought in calibration week 1 and week 3, nothing else.
counts = [1, 0, 1, 0, 0, 0]
panel = pd.DataFrame({"Id": 1, "period_start": weeks, "Transactions": counts})
cal = panel[panel["period_start"] <= pd.Timestamp("2020-01-27")]

print("calibration panel (what the fitter is given):")
print(cal.to_string(index=False))

cbs = _build_cbs(cal, id_col="Id", target_col="Transactions",
                 time_col="period_start", period_in_days=7.0)
print("\nthe three numbers the model reduces this customer to:")
print(cbs.to_string())

T = float(cbs["T_cal"].iloc[0])
label = pd.Timestamp("2020-01-06")            # the customer's first active week's LABEL
holdout = weeks[4:]

# The same T_cal, read under both within-week anchors. `_build_cbs` never states one;
# the point of the toy is that only one of them makes the windows line up.
for anchor, origin in (("end of the week", label + pd.Timedelta(weeks=1)),
                       ("start of the week", label)):
    print(f"\n--- if a week's transactions happen at the {anchor} ---")
    print(f"  customer time 0        = {origin.date()}")
    print(f"  T_cal = {T} means observation ends {(origin + pd.Timedelta(weeks=T)).date()}"
          f"   (calibration really ends 2020-02-03)")
    for t in (1, 2):
        lo = origin + pd.Timedelta(weeks=T + t - 1)
        hi = origin + pd.Timedelta(weeks=T + t)
        want = holdout[t - 1]
        ok = "OK " if lo == want else "NOT"
        print(f"  {ok} column {t-1}: [{lo.date()}, {hi.date()})"
              f"  vs holdout week {t}: [{want.date()}, {(want + pd.Timedelta(weeks=1)).date()})")
