# Frozen v0.9.4 Confirmatory Evidence

## Integrity
| Artifact | SHA-256 |
|---|---|
| Evidence manifest | `CF2A6527779CD5D0331ACC20DF77C63D5E7EEEFE25C52455D9AD3050A1E507D1` |
| Evidence summary | `FDBE4A788D889C25CA98265A86BBB54ED510B981AF8F188767A09160B9DB73D5` |

The underlying local evidence directory remains untracked because it includes
large models and prediction artifacts.

## Protocol
- no direct raw bank/corridor categorical identity
- review budget: 1%
- primary: paired transaction-mass bootstrap
- 25,000-row timestamp-preserving blocks
- 400 paired resamples, seed `20260624`
- secondary: 6-hour equal-calendar-time bootstrap diagnostic

## Results
| Evaluation | Uplift | Primary interval | Non-positive resamples | Champion |
|---|---:|---:|---:|---|
| HI-Medium | +14.14pp | [+6.37pp, +28.00pp] | 0 / 400 | Candidate |
| Frozen HI-Medium to LI-Medium | +18.61pp | [+6.71pp, +31.98pp] | 0 / 400 | Candidate |

Frozen transfer used LI-Medium only for label-free history and final evaluation;
LI labels were not used for fitting, tuning, calibration, or threshold
selection.