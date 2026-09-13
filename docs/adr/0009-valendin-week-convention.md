# Every panel numbers weeks the way Valendin's notebook does

A date's week is `dayofyear // 7`, capped at 51 — the grid of the Valendin et al.
reference notebook, which Markus's code and our electronics panel also use. The package
previously used `(dayofyear − 1) // 7`, and the panels were mixed: electronics on one
rule, CDNOW, gift and multichannel on the other. That mix was the actual defect. The same
week number meant different days on different datasets, and `prepare_dataset` cuts
windows on `period_calendar.week_start`, which is only correct for panels built on the
rule it inverts. A mid-year cut on a panel from the other rule moves a whole week across
the boundary.

We chose Valendin's rule over keeping ours so a week in this project is the same seven
days as in the reference code the benchmark reproduces. `period_calendar` holds the rule
and its inverse; every panel builder (`scripts/build_cdnow_panel.py`,
`scripts/build_rdata_panel.py`) reads it from there.

## Consequences

- Week 0 is six days (Jan 1..6) and week 51 is nine, ten in a leap year. Their counts
  are correspondingly low and high, on every dataset alike.
- CDNOW's calibration can no longer be Fader & Hardie's Jan 1..Sep 30 1997 exactly:
  1997 week 39 opens on Sep 30, so calibration is weeks 0..38 = Jan 1..Sep 29 and Sep 30
  falls in the holdout. In exchange the data's last day, 1998-06-30, now closes a week,
  so the holdout is 39 weeks, as published, instead of 38.
- Electronics was already on this rule and its windows start on Jan 1, where both rules
  agree, so its tensors are unchanged.
- Results recorded before the switch on CDNOW (38-week holdout) and on the gift and
  multichannel probes were computed on the old weeks and are not directly comparable to
  new runs.
