[CmdletBinding()]
param(
    [ValidateSet("Probe", "FullGpu", "LiStress")]
    [string]$Mode = "Probe",
    [string]$InputPath = "C:\Users\bjw-0\Downloads\Project_Data\ibm_aml\HI-Small_Trans.csv",
    [string]$OutputDir = "",
    [switch]$SkipDependencyInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Virtualenv Python not found: $Python" }
if (-not (Test-Path -LiteralPath $InputPath)) { throw "IBM AML input not found: $InputPath" }

$ConfigName = switch ($Mode) {
    "Probe" { "payment_risk_experiment_ibm_aml_probe.json" }
    "FullGpu" { "payment_risk_experiment_ibm_aml_catboost_gpu.json" }
    "LiStress" { "payment_risk_experiment_ibm_aml_li_stress.json" }
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = ".local-run\ibm-aml-$($Mode.ToLowerInvariant())-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}

function Invoke-PythonChecked {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed ($LASTEXITCODE): $Python $($Arguments -join ' ')" }
}

Set-Location -LiteralPath $RepoRoot
if (-not $SkipDependencyInstall) {
    Invoke-PythonChecked -Arguments @("-m", "pip", "install", "-r", "requirements-modeling.txt")
}

$ProfilePath = Join-Path $OutputDir "reports\ibm_aml_input_profile.json"
Invoke-PythonChecked -Arguments @(
    "scripts\profile_aml_data.py", "--input", $InputPath, "--output", $ProfilePath
)
Invoke-PythonChecked -Arguments @(
    "scripts\run_champion_challenger.py",
    "--input", $InputPath,
    "--output-dir", $OutputDir,
    "--config", (Join-Path "contracts" $ConfigName)
)

Write-Host ""
Write-Host "IBM AML CHAMPION-CHALLENGER EXPERIMENT COMPLETE"
Write-Host "Input profile: $(Join-Path $RepoRoot $ProfilePath)"
Write-Host "Metrics: $(Join-Path $RepoRoot "$OutputDir\reports\experiment_metrics.json")"
Write-Host "Decision: $(Join-Path $RepoRoot "$OutputDir\reports\promotion_decision.json")"
