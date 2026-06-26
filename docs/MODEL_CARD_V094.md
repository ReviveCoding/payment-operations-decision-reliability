# Model Card: PaymentOps AML v0.9.4

## Intended use
Rank transactions for analyst review at a fixed 1% capacity. The system is a
benchmark champion-challenger framework, not an autonomous financial-crime
decision engine.

## Configuration
- Candidate: `catboost.core.CatBoostClassifier`
- Recorded training: GPU device `0`, 900 iterations, selected weight mode `none`
- Baseline: regularized logistic regression
- Calibration: held-out sigmoid calibration
- Direct raw bank/corridor categoricals excluded: `From Bank`, `To Bank`,
  `bank_pair`
- Retained direct categoricals: payment currency, receiving currency, payment
  format
- Historical features use preceding transactions only
- Strict chronological 60/20/20 split with complete timestamp groups

## Primary promotion inference
- paired timestamp-preserving transaction-mass bootstrap
- target block size: 25,000 rows
- at least 300,000 rows per paired resample
- 400 resamples, seed `20260624`
- promotion requires Capture@1% lower bound > 0 and acceptable Brier change

## Confirmatory outcomes

| Evaluation | Capture@1% baseline | Capture@1% candidate | Uplift | Primary CI | Decision |
|---|---:|---:|---:|---:|---|
| HI-Medium | 71.78% | 85.91% | +14.14pp | [+6.37pp, +28.00pp] | Candidate |
| Frozen HI-Medium to LI-Medium | 49.51% | 68.12% | +18.61pp | [+6.71pp, +31.98pp] | Candidate |

The frozen LI-Medium result captured 704 more positives at the same review
budget. No LI-Medium labels were used for fitting, tuning, calibration, or
threshold selection.

## Evidence integrity
- manifest SHA-256:
  `CF2A6527779CD5D0331ACC20DF77C63D5E7EEEFE25C52455D9AD3050A1E507D1`
- summary SHA-256:
  `FDBE4A788D889C25CA98265A86BBB54ED510B981AF8F188767A09160B9DB73D5`

## Limitations
- IBM AML is synthetic benchmark data, not production-bank validation.
- The result covers one locked configuration and documented seed.
- The 6-hour equal-calendar-time diagnostic has a 0.00% lower bound in both
  Medium runs and remains a secondary reported stability measure.
- Medium datasets, trained models, predictions, and local evidence remain
  untracked.