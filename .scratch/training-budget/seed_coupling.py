"""How far does the replication RNG coupling reach into the archive?

`forecast_recurrent` seeds the global RNG before sampling, and each replication's
forecast runs immediately before the next replication trains. If that determines the
next replication's weight init and shuffling, two suites sharing a seed list should
produce IDENTICAL winning validation losses from replication 2 onward.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "Studies"

rows = []
for mfile in ROOT.glob("*/*/metrics.csv"):
    suite, model = mfile.parents[1].name, mfile.parents[0].name
    try:
        df = pd.read_csv(mfile)
    except Exception:
        continue
    if "objective" not in df or "study" not in df:
        continue
    for _, r in df.iterrows():
        rows.append(dict(suite=suite, model=model, family=suite.split("__")[0],
                         study=int(r["study"]), seed=r.get("seed"),
                         objective=float(r["objective"])))
d = pd.DataFrame(rows)
print(f"{len(d)} (suite, study) rows from {d.suite.nunique()} suites\n")

# Within a family+model, group by study index: if the coupling determines training,
# suites of the same family that share a seed list repeat the same objective.
print("Exact repeats of the winning validation loss across suites, by study index")
print("(a repeat means two suites trained an identical model at that replication):\n")
out = []
for (fam, model), g in d.groupby(["family", "model"]):
    if g.suite.nunique() < 3:
        continue
    per_study = []
    for study, gg in g.groupby("study"):
        vals = gg.objective.round(12)
        if len(vals) < 3:
            continue
        # share of suites whose objective is not unique at this study index
        dup = vals.duplicated(keep=False).mean()
        per_study.append((study, len(vals), dup))
    if not per_study:
        continue
    ps = pd.DataFrame(per_study, columns=["study", "n_suites", "dup_share"])
    out.append(dict(family=fam, model=model, suites=g.suite.nunique(),
                    study1=ps.loc[ps.study == 1, "dup_share"].mean(),
                    study2plus=ps.loc[ps.study > 1, "dup_share"].mean(),
                    n_study_idx=len(ps)))
res = pd.DataFrame(out).sort_values("study2plus", ascending=False)
print(res.round(3).to_string(index=False))
print("\nstudy1 = share of suites sharing an identical objective at replication 1;")
print("study2plus = the same, averaged over replications 2..n.")
