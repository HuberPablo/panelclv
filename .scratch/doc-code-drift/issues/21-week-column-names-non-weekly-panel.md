# 21 — The prediction CSV writes `week_*` columns whatever the panel's frequency is

**Status:** needs-triage

## Doc claim

`CONTEXT.md:15-19` makes the unit a **Period**, explicitly frequency-agnostic, and puts "week"
on the avoid list:

> **Period**: The time unit a panel is measured in — weekly throughout this project, **though
> the package supports daily and monthly**.
> _Avoid_: **week**, timestep, bucket

The support is real: `_VALID_FREQUENCIES` (`src/panelclv/configs/panel_config.py:106`) accepts
daily / weekly / monthly, `_DAYS_PER_PERIOD` (`data_preparation/period_calendar.py:52-57`)
carries the table, and `_resolve_time_index` (`panel_dataset.py:219-249`) has the branches.

## Code reality

The canonical on-disk prediction format hardcodes the weekly spelling.
`src/panelclv/predictions/prediction_csv.py:90`:

```python
columns = [f"week_{i + week_offset}" for i in range(n_weeks)]
```

with the parameter itself named `week_offset` and the count `n_weeks`. The same spelling is
baked into `predictions/__init__.py:3-4` ("``week_0..week_{T-1}``"),
`evaluation.plot_weekly_aggregated`, and `WEEKS_PER_YEAR`
(`data_preparation/period_calendar.py:41`).

So a **daily** panel writes a CSV whose columns are `week_0 … week_N`, one per day.

## Why `needs-triage`

Nothing is numerically wrong — the columns are positional and every reader
(`load_predictions_from_csv`, `reduce_to_customer_period`, the suite readers) treats them
positionally. This is a labelling and vocabulary question, and the decision is about cost:

**(a) Leave it, and record why.** Every panel in this project is weekly; the format is already
written into every archived `Studies/` CSV; renaming would either break every stored result or
force a compatibility shim in the reader. If this is the call, say so in
`predictions/__init__.py` and add "the prediction CSV's `week_*` columns" as a stated exception
in `CONTEXT.md`'s Period entry — an exception on the record is not drift.

**(b) Rename to `period_*`.** Cleaner against the vocabulary and correct on a daily panel, but
it needs a reader that accepts both spellings so archived suites still load, and a sweep of
`week_offset` / `n_weeks` / `plot_weekly_aggregated`.

(a) is almost certainly right given the archive, but it is currently *undocumented* silence
rather than a decision — which is what puts this on the list at all.

## Note

`WEEKS_PER_YEAR` (`period_calendar.py:41`) is a separate question: it is used for annual
aggregation in the validation gate and the drift check, where 52 is a weekly-panel assumption
rather than a column name. Worth checking whether anything reads it on a non-weekly path
before touching either.
