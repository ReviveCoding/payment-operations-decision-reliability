from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from payment_ops_hardening.cli import main
from payment_ops_hardening.release_security import finalize_release_security

KEY = "c" * 32


def _write_release(root: Path) -> list[str]:
    paths = ["models/model.bin", "data/features.csv", "reports/decision.json"]
    (root / "models").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "reports").mkdir()
    (root / "models/model.bin").write_bytes(b"model")
    (root / "data/features.csv").write_text(
        "case_id,score\na,0.1\nb,0.2\n", encoding="utf-8"
    )
    (root / "reports/decision.json").write_text(
        json.dumps({"run_id": "r1", "state": "PROMOTE"}), encoding="utf-8"
    )
    (root / "release_manifest.json").write_text(
        json.dumps({"run_id": "r1", "release_state": "PROMOTE"}),
        encoding="utf-8",
    )
    return paths


def _write_contract(path: Path, paths: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "require_signature": True,
                "required_paths": paths,
                "content_contracts": (
                    {
                        "data/features.csv": {
                            "type": "csv",
                            "required_columns": ["case_id", "score"],
                            "min_rows": 2,
                        }
                    }
                    if "data/features.csv" in paths
                    else {}
                ),
                "expected_key_id": "cli-key",
                "minimum_release_sequence": 5,
                "allowed_release_states": ["PROMOTE"],
                "reject_untracked_files": True,
                "allowed_untracked_paths": [],
                "json_value_equalities": [],
            }
        ),
        encoding="utf-8",
    )


def test_cli_verifies_signed_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    paths = _write_release(release)
    finalize_release_security(
        release,
        required_paths=paths,
        content_contracts={
            "data/features.csv": {
                "type": "csv",
                "required_columns": ["case_id", "score"],
                "min_rows": 2,
            }
        },
        key=KEY,
        key_id="cli-key",
        require_signature=True,
        release_sequence=5,
        allowed_release_states=["PROMOTE"],
        reject_untracked_files=True,
    )
    contract = tmp_path / "contract.json"
    _write_contract(contract, paths)
    monkeypatch.setenv("PAYMENT_OPS_MANIFEST_HMAC_KEY", KEY)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "payment-ops-hardening-validate",
            str(release),
            "--contract-file",
            str(contract),
        ],
    )
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["manifest_authentication"]["status"] == "VERIFIED"
    assert result["release_sequence"] == 5


def test_cli_rejects_key_id_override_conflicting_with_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract = tmp_path / "contract.json"
    _write_contract(contract, ["models/model.bin"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "payment-ops-hardening-validate",
            str(tmp_path / "release"),
            "--contract-file",
            str(contract),
            "--expected-key-id",
            "different-key",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "conflicts with the trusted contract" in capsys.readouterr().err


def test_cli_reports_package_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["payment-ops-hardening-validate", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "payment-ops-hardening-validate 0.8.2"
