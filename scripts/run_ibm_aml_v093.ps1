[CmdletBinding()]
param(
    [ValidateSet("HiFullGpu", "LiStress", "HiNoBankIdentity", "LiNoBankIdentity")]
    [string]$Mode = "HiFullGpu",
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
    "HiFullGpu" { "payment_risk_experiment_ibm_aml_catboost_gpu_v093.json" }
    "LiStress" { "payment_risk_experiment_ibm_aml_li_stress_v093.json" }
    "HiNoBankIdentity" { "payment_risk_experiment_ibm_aml_catboost_gpu_no_bank_identity_v093.json" }
    "LiNoBankIdentity" { "payment_risk_experiment_ibm_aml_li_stress_no_bank_identity_v093.json" }
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = ".local-run\ibm-aml-$($Mode.ToLowerInvariant())-v093-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
function Invoke-PythonChecked {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)
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
Invoke-PythonChecked -Arguments @("scripts\profile_aml_data.py", "--input", $InputPath, "--output", $ProfilePath)
Invoke-PythonChecked -Arguments @(
    "scripts\run_champion_challenger.py",
    "--input", $InputPath,
    "--output-dir", $OutputDir,
    "--config", (Join-Path "contracts" $ConfigName)
)

Write-Host ""
Write-Host "IBM AML V0.9.3 CHAMPION-CHALLENGER EXPERIMENT COMPLETE"
$MetricsPath = Join-Path $RepoRoot ($OutputDir + "\reports\experiment_metrics.json")
$DecisionPath = Join-Path $RepoRoot ($OutputDir + "\reports\promotion_decision.json")
$ProvenancePath = Join-Path $RepoRoot ($OutputDir + "\reports\candidate_model_metadata.json")
Write-Host "Metrics: $MetricsPath"
Write-Host "Decision: $DecisionPath"
Write-Host "Candidate provenance: $ProvenancePath"
