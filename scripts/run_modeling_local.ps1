[CmdletBinding()]
param(
    [string]$InputPath = "data\iso20022_analytics.csv",
    [string]$ConfigPath = "contracts\payment_risk_experiment_cpu.json",
    [string]$OutputDir = "",
    [switch]$SkipDependencyInstall
)

Set-StrictMode -Version Latest
# Native tools can write expected progress messages to stderr. Exit codes are authoritative.
$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtualenv Python was not found: $Python. Create/install the project venv first."
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = ".local-run\model-uplift-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}

function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE: $Python $($Arguments -join ' ')"
    }
}

Set-Location -LiteralPath $RepoRoot
if (-not $SkipDependencyInstall) {
    Invoke-PythonChecked -Arguments @("-m", "pip", "install", "-r", "requirements-modeling.txt")
}
Invoke-PythonChecked -Arguments @(
    "scripts\run_champion_challenger.py",
    "--input", $InputPath,
    "--output-dir", $OutputDir,
    "--config", $ConfigPath
)

Write-Host ""
Write-Host "MODEL UPLIFT EXPERIMENT COMPLETE"
Write-Host "Evidence: $(Join-Path $RepoRoot "$OutputDir\reports\experiment_metrics.json")"
Write-Host "Decision: $(Join-Path $RepoRoot "$OutputDir\reports\promotion_decision.json")"
