[CmdletBinding()]
param(
    [ValidateSet("HiMediumConfirmatoryNoBankIdentity")]
    [string]$Mode = "HiMediumConfirmatoryNoBankIdentity",
    [string]$InputPath = "C:\Users\bjw-0\Downloads\Project_Data\ibm_aml\HI-Medium_Trans.csv",
    [string]$OutputDir = "",
    [switch]$SkipDependencyInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtualenv Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
    throw "HI-Medium IBM AML input not found: $InputPath"
}

$ConfigName = "payment_risk_experiment_ibm_aml_medium_confirmatory_no_bank_identity_v094.json"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = ".local-run\ibm-aml-hi-medium-confirmatory-no-bank-identity-v094-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}

function Invoke-PythonChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $Python $($Arguments -join ' ')"
    }
}

Set-Location -LiteralPath $RepoRoot
if (-not $SkipDependencyInstall) {
    Invoke-PythonChecked -Arguments @("-m", "pip", "install", "-r", "requirements-modeling.txt")
}

$ProfilePath = Join-Path $OutputDir "reports\ibm_aml_input_profile.json"
Invoke-PythonChecked -Arguments @(
    "scripts\profile_aml_data.py",
    "--input", $InputPath,
    "--output", $ProfilePath
)
Invoke-PythonChecked -Arguments @(
    "scripts\run_champion_challenger_v094.py",
    "--repo-root", $RepoRoot,
    "--input", $InputPath,
    "--output-dir", $OutputDir,
    "--config", (Join-Path "contracts" $ConfigName)
)

Write-Host ""
Write-Host "IBM AML V0.9.4 MEDIUM CONFIRMATORY EXPERIMENT COMPLETE"
$MetricsPath = Join-Path $RepoRoot ($OutputDir + "\reports\experiment_metrics.json")
$DecisionPath = Join-Path $RepoRoot ($OutputDir + "\reports\promotion_decision.json")
$ProtocolPath = Join-Path $RepoRoot ($OutputDir + "\reports\pre_registered_protocol.json")
Write-Host "Metrics: $MetricsPath"
Write-Host "Decision: $DecisionPath"
Write-Host "Pre-registration: $ProtocolPath"
