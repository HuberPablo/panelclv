"""Do the Pareto/NBD forecast columns cover the holdout weeks, or the week before?

_build_cbs sets cal_end = max(period_start) of the calibration panel, and the
forecaster puts holdout period t on customer time [T_cal+t-1, T_cal+t].  Print the
calendar week each of those actually is, against the holdout panel's own weeks.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, "scripts")
from run_real_panel_benchmarks import build_data, CALIBRATIONS

for cal, panels in (("2y", ["cdnow", "electronics", "gift", "multichannel"]),
                    ("3y", ["electronics", "gift", "multichannel"])):
    for name in panels:
        data = build_data(name, cal)
        tp, hp = data["train_panel"], data["holdout_panel"]
        cal_starts = np.sort(pd.to_datetime(tp["period_start"]).unique())
        hold_starts = np.sort(pd.to_datetime(hp["period_start"]).unique())
        cal_end = pd.Timestamp(cal_starts[-1])
        P = pd.Timedelta(days=7)
        col0 = (cal_end, cal_end + P)                  # forecast column 0's calendar span
        hold0 = (pd.Timestamp(hold_starts[0]), pd.Timestamp(hold_starts[0]) + P)
        shift = (hold0[0] - col0[0]).days / 7
        print(f"{cal} {name:13s} T_CAL={data['T_CAL']:3d} T_HOLD={data['T_HOLD']:3d} "
              f"| cal weeks {pd.Timestamp(cal_starts[0]).date()}..{cal_end.date()} "
              f"| forecast col0 covers [{col0[0].date()}, {col0[1].date()}) "
              f"| holdout col0 is [{hold0[0].date()}, {hold0[1].date()}) "
              f"| shift = {shift:.0f} period(s)")
