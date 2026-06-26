import json
from pathlib import Path

import pytest

from payment_ops_hardening.release_contract import (
    ReleaseContractError,
    load_release_contract,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_release_contract_normalizes_policy(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    _write(
        path,
        {
            "schema_version": "1.1",
            "require_signature": False,
            "required_paths": ["models/model.bin", "data/features.csv"],
            "content_contracts": {
                "data/features.csv": {
                    "type": "csv",
                    "required_columns": ["case_id", "score"],
                    "min_rows": 1,
                }
            },
            "expected_key_id": "2026-q2",
            "allowed_release_states": ["PROMOTE"],
            "reject_untracked_files": True,
        },
    )
    result = load_release_contract(path)
    assert result["expected_key_id"] == "2026-q2"
    assert result["allowed_release_states"] == ["PROMOTE"]
    assert result["reject_untracked_files"] is True
    assert result["content_contracts"]["data/features.csv"]["min_rows"] == 1


def test_release_contract_rejects_unknown_contract_path(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    _write(
        path,
        {
            "schema_version": "1.1",
            "require_signature": False,
            "required_paths": ["models/model.bin"],
            "content_contracts": {"data/features.csv": {"type": "csv"}},
        },
    )
    with pytest.raises(ReleaseContractError, match="outside required_paths"):
        load_release_contract(path)


def test_release_contract_rejects_string_instead_of_column_list(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    _write(
        path,
        {
            "schema_version": "1.1",
            "require_signature": False,
            "required_paths": ["data/features.csv"],
            "content_contracts": {
                "data/features.csv": {
                    "type": "csv",
                    "required_columns": "case_id",
                }
            },
        },
    )
    with pytest.raises(ReleaseContractError, match="JSON list of strings"):
        load_release_contract(path)


def test_release_contract_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        '{"schema_version":"1.0","required_paths":["a"],"required_paths":["b"]}',
        encoding="utf-8",
    )
    with pytest.raises(ReleaseContractError, match="duplicate JSON key"):
        load_release_contract(path)


def test_release_contract_requires_boolean_untracked_policy(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    _write(
        path,
        {
            "schema_version": "1.1",
            "require_signature": False,
            "required_paths": ["a"],
            "reject_untracked_files": "yes",
        },
    )
    with pytest.raises(ReleaseContractError, match="must be boolean"):
        load_release_contract(path)


def test_v04_contract_template_renders_and_validates(tmp_path):
    from payment_ops_hardening.contract_template import render_release_contract_template

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "release-contract.json"
    contract = render_release_contract_template(
        root / "contracts/payment_ops_v04_release_contract.template.json",
        output,
        {
            "__SELECTED_MODEL__": "logistic_calibrated",
            "__EXPECTED_KEY_ID__": "paymentops-2026-q3",
            "__MINIMUM_RELEASE_SEQUENCE__": 201,
        },
    )
    assert contract["minimum_release_sequence"] == 201
    assert contract["expected_key_id"] == "paymentops-2026-q3"
    assert "models/logistic_calibrated.joblib" in contract["required_paths"]


def test_contract_template_rejects_missing_replacement(tmp_path):
    from payment_ops_hardening.contract_template import (
        ContractTemplateError,
        render_release_contract_template,
    )

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ContractTemplateError, match="replacement mismatch"):
        render_release_contract_template(
            root / "contracts/payment_ops_v04_release_contract.template.json",
            tmp_path / "release-contract.json",
            {
                "__SELECTED_MODEL__": "logistic_calibrated",
                "__EXPECTED_KEY_ID__": "paymentops-2026-q3",
            },
        )


def test_release_contract_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    _write(
        path,
        {
            "schema_version": "1.1",
            "require_signature": False,
            "required_paths": ["a"],
            "typo_policy": True,
        },
    )
    with pytest.raises(ReleaseContractError, match="unknown release contract fields"):
        load_release_contract(path)


def test_signed_contract_requires_expected_key_id(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    _write(
        path,
        {
            "schema_version": "1.1",
            "require_signature": True,
            "required_paths": ["a"],
        },
    )
    with pytest.raises(
        ReleaseContractError, match="requires a non-empty expected_key_id"
    ):
        load_release_contract(path)


def test_release_contract_hardlink_is_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    _write(
        original,
        {
            "schema_version": "1.1",
            "require_signature": False,
            "required_paths": ["a"],
        },
    )
    linked = tmp_path / "contract.json"
    try:
        linked.hardlink_to(original)
    except (OSError, NotImplementedError):
        pytest.skip("hard links are unavailable")
    with pytest.raises(ReleaseContractError, match="hard-linked"):
        load_release_contract(linked)
