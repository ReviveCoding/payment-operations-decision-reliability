#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PAYMENT_OPS_QUALIFY_VENV:-$ROOT/.venv-qualification}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --no-cache-dir -c "$ROOT/constraints/ci.txt" \
  "$ROOT" pytest pytest-cov ruff mypy build setuptools wheel
"$VENV/bin/python" "$ROOT/scripts/qualify_local.py" --profile standard
