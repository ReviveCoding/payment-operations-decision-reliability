param(
  [string]$VenvPath = ".venv-qualification",
  [string]$PythonVersion = "3.11"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
py -$PythonVersion -m venv (Join-Path $Root $VenvPath)
$Python = Join-Path $Root "$VenvPath\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install --no-cache-dir -c (Join-Path $Root "constraints\ci.txt") `
  $Root pytest pytest-cov ruff mypy build setuptools wheel
& $Python (Join-Path $Root "scripts\qualify_local.py") --profile standard
