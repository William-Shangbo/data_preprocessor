# 2026-04-09 EWMA Rank Incident Postmortem

## Summary

During `daily_data_ewma` generation for `d30+`, the job repeatedly appeared to hang around the `ewma_30hl` stage. Investigation showed that the bottleneck was not the EWMA recurrence itself but the downstream cross-sectional `rank(...)` call over the full daily frame.

## Impact

- `daily_data_ewma/d30..d59` could not be produced reliably with the original parallel configuration.
- The operator experience was poor because logs stopped for a long time at `calculating ewma_30hl`.
- The historical implementation also exhibited incorrect behavior for some `all-NaN` feature groups.

## Root Cause

### 1. Implementation-level bottleneck

The original unweighted `rank` path used:

```python
df.groupby(by)[cols].rank(...)
df.groupby(by)[cols].transform("count")
```

This is semantically "rank by timepoint", but operationally it asks pandas to materialize large intermediate frames for the entire day (`119500 x 384`) at once.

### 2. All-NaN group instability

For some columns that were entirely `NaN` inside a `(dateid, timeid)` group, the previous implementation could emit a mixture of `NaN` and `0.0` instead of a stable all-`NaN` result. This made the old path unsuitable as a gold reference for regression comparison.

### 3. Resource pressure amplified by parallelism

Under `num_workers=2`, each worker was already holding several large frames:

- raw daily slice
- imputed daily frame
- cross-day concatenated frame
- EWMA feature frame
- rank intermediates

This substantially increased memory pressure and made the slow rank stage look like a deadlock.

## Fix

The production unweighted `rank` path now:

1. iterates group-by-group,
2. computes rank/count normalization on the smaller group frame,
3. explicitly preserves `all-NaN` groups as `NaN`,
4. preserves singleton semantics (`0.5`) and average-tie ranking.

## Experiments

### d33

- Old implementation: about `29s`
- Group-wise implementation: about `2.9s`
- The main observed semantic difference was the corrected handling of `all-NaN` groups.

### d34

The revised group-wise rank successfully passed the previously stuck `ewma_30hl` stage in isolation:

- impute: about `3.8s`
- ewma_30hl: about `2.4s`
- rank: about `3.0s`

## Required Pre-Production Tests

Any general-purpose ranking or normalization function must have archived checks for:

1. `all-NaN` group behavior
2. singleton non-NaN group behavior
3. tie handling
4. mixed NaN/non-NaN groups
5. consistency across representative real slices (`d33`, `d34`)
6. timing/memory measurements on realistic production-sized inputs

## Archival Policy

Before promoting a shared utility into production:

1. Add a focused regression script under `ljcomp/dev/`
2. Record at least one real-data benchmark
3. Save a postmortem if the change was triggered by a production incident
4. Restate previously produced outputs when semantics change
