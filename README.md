# PaymentOps AML Decision Reliability Platform

A reliability-first AML transaction-ranking framework combining leakage-safe
temporal features, calibrated risk ranking, fixed review-capacity evaluation,
raw-entity ablations, promotion gates, and frozen cross-regime transfer.

The repository retains an earlier release-security overlay as a supporting
engineering component. Current model-performance claims are based only on the
v0.9.4 AML confirmatory protocol.

## v0.9.4 confirmatory evidence

![Capture@1% comparison across v0.9.4 Medium evaluations](docs/assets/v094_capture_at_1pct.svg)

*Figure 1. At a fixed 1% review budget, the selected no-bank-identity candidate improves capture both in HI-Medium confirmation and in frozen HI-to-LI transfer. Primary intervals use 400 paired transaction-mass resamples.*

| Evaluation | Baseline Capture@1% | Candidate Capture@1% | Uplift | Primary transaction-mass CI | Champion |
|---|---:|---:|---:|---:|---|
| HI-Medium chronological confirmation | 71.78% | 85.91% | +14.14pp | [+6.37pp, +28.00pp] | Candidate |
| Frozen HI-Medium to LI-Medium | 49.51% | 68.12% | +18.61pp | [+6.71pp, +31.98pp] | Candidate |

The frozen LI-Medium candidate captured 2,577 of 3,783 positives at a fixed
1% review capacity, compared with 1,873 for the frozen baseline: 704
additional positives. Both primary analyses used 400 paired resamples and had
zero non-positive uplift replicates.

The primary analysis uses paired transaction-mass bootstrap inference with
timestamp-preserving blocks. A 6-hour equal-calendar-time bootstrap is retained
as a separate stability diagnostic and does not override the pre-registered
v0.9.4 promotion rule.

## Selected model policy

- Candidate: CatBoostClassifier, GPU device `0` in the recorded runs
- Baseline: regularized logistic regression
- Calibration: held-out sigmoid calibration partition
- Direct raw `From Bank`, `To Bank`, and `bank_pair` categoricals: excluded
- Retained context: payment currency, receiving currency, payment format,
  transaction context, account history, account-pair history, and label-free
  bank-pair historical aggregates
- Split: strict chronological 60/20/20 with complete timestamp groups
- Review budget: top 1% of held-out transactions

## Reproduce

Install:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -c constraints/ci.txt -r requirements-modeling.txt `
  pytest pytest-cov ruff mypy build setuptools wheel
python -m pip install -e .
```

HI-Medium confirmation:

```powershell
.\scripts\run_ibm_aml_v094.ps1 `
  -Mode HiMediumConfirmatoryNoBankIdentity `
  -InputPath "C:\path\to\HI-Medium_Trans.csv" `
  -OutputDir ".local-run\ibm-aml-hi-medium-confirmatory-no-bank-v094"
```

Frozen HI-Medium to LI-Medium transfer:

```powershell
$Py = ".\.venv\Scripts\python.exe"
& $Py ".\scripts\run_frozen_hi_to_li_v094.py" `
  --repo-root "." `
  --source-run ".local-run\ibm-aml-hi-medium-confirmatory-no-bank-v094" `
  --target-input "C:\path\to\LI-Medium_Trans.csv" `
  --output-dir ".local-run\ibm-aml-hi-medium-no-bank-to-li-medium-frozen-v094"
```

## Evidence boundary

IBM AML is a synthetic transaction benchmark. Results support only controlled
benchmark claims for the declared chronological protocols. They do not establish
production-bank efficacy, regulatory compliance, or autonomous decision
readiness.

See [the v0.9.4 protocol](docs/v094_confirmatory_protocol.md),
[the model card](docs/MODEL_CARD_V094.md), [the frozen evidence
summary](docs/V094_EVIDENCE.md), and [limitations](docs/known_limitations.md).

## Repository layout

```text
contracts/    Versioned experiment contracts
docs/         Protocol, model card, frozen evidence, limitations, legacy context
scripts/      Profiling, training/evaluation, frozen transfer, qualification
src/          Modeling and release-security implementation
tests/        Unit, split-integrity, leakage-control, and protocol tests
data/         Small synthetic fixture; Medium inputs are external
```