# PaymentOps AML v0.9.4 Confirmatory Protocol

## Purpose
Confirm on untouched IBM AML Medium data the no-bank-identity CatBoost
configuration selected from the v0.9.3 Small-data exploratory work.

## Locked configuration
- Direct `From Bank`, `To Bank`, and `bank_pair` categoricals are excluded.
- Payment currency, receiving currency, payment format, numerical transaction
  context, and label-free historical aggregates are retained.
- CatBoost uses GPU device `0`.
- Candidate weighting is selected only on the HI-Medium tuning partition.
- Chronological partitions use whole timestamp groups.

## Primary promotion inference
Promotion uses a paired contiguous transaction-mass bootstrap:
- Timestamp-preserving blocks target 25,000 rows.
- Complete blocks are sampled with replacement until at least 300,000 rows.
- 400 resamples, fixed seed 20260624.
- Promotion requires a positive 2.5th-percentile Capture@1% uplift and
  acceptable Brier-score change.

## Secondary diagnostic
A paired equal-calendar-time 6-hour bootstrap is reported as a stability
diagnostic. It cannot override the primary confirmatory promotion decision.

## Frozen transfer
HI-Medium model and sigmoid calibration artifacts are frozen. LI-Medium labels
must not be used for fit, tuning, calibration, or threshold selection.
