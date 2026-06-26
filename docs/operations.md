# Operations

## Environment
- Python 3.11+
- Modeling dependencies: `requirements-modeling.txt`
- Recorded v0.9.4 training used CatBoost GPU device `0`
- IBM AML Medium datasets are external and intentionally untracked

## Quality checks
```powershell
$Py = ".\.venv\Scripts\python.exe"
& $Py -m ruff format --check .
& $Py -m ruff check .
& $Py -m pytest -q
& $Py -m mypy src
```

## Confirmatory commands
Use the command examples in the repository README. Do not reuse a previous
training output directory for a new model run. Frozen transfer must load the
HI-Medium source artifacts and must not fit, tune, calibrate, or choose
thresholds with LI labels.

## Security-overlay component
The retained release-security utility requires
`PAYMENT_OPS_MANIFEST_HMAC_KEY` for manifest verification. Do not store secrets
in source, contracts, models, or logs.