# 14 — `CONTEXT.md`: declared and derived covariates are not "standardised the same way"

**Status:** ready-for-agent

## Doc claim

`CONTEXT.md:29-35`, defining **Covariate**:

> A covariate is either **declared** — a panel column named in one of a `PanelConfig`'s roles
> — or **derived** — computed from the target's own past. The distinction is about where the
> value comes from, not how the model reads it: both occupy a channel of the same tensor,
> **both are standardised the same way**, and the covariate-subset search drops either.

## Code reality

`standardize_covariates` excludes every column in `embedded_cols`
(`src/panelclv/data_preparation/panel_dataset.py:477-480`):

> Scope — only the columns the model actually treats as numeric:
>   * everything in ``embedded_cols`` is excluded: those are integer class indices, cast with
>     ``.long()`` and used as embedding-table lookups, so rescaling them would corrupt the
>     lookup outright;

A **declared** covariate may be embedded — that is exactly what `PanelConfig.embedded_cols`
is for, and the golden fixture embeds `week` in some configurations. A **derived** AR feature
never is: `configs/ar_feature_names.py` produces continuous quantities (recency, cumulative
counts, tenure, rate), and nothing embeds them.

So a declared categorical covariate goes through the embedding table untouched, while a
derived AR feature is centred and scaled. The two are not standardised the same way, and
cannot be — rescaling a class index would break the lookup.

## Which parts of the sentence do hold

Worth keeping in the rewrite, because two of the three clauses are correct:

- "both occupy a channel of the same tensor" — true, both live in `seq_cols`.
- "the covariate-subset search drops either" — true
  (`tuning/optuna_tuning.py:128-152`, `validate_removable_features`).

## Fix

`CONTEXT.md:34-35` — replace the middle clause. The real invariant is about *channels*, not
about standardisation:

> … both occupy a channel of the same tensor, both are read through the same embedder seam
> (embedded if declared as categorical, standardised if numeric — see
> `docs/feature_engineering.md` §5), and the covariate-subset search drops either.

The entry's actual point — that the declared/derived distinction is about *provenance*, not
about how the model consumes them — survives intact; it just should not rest on a claim that
has an exception built into the transform.
