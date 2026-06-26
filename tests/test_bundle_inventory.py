import json
from pathlib import Path

import pytest

from payment_ops_hardening.bundle_inventory import (
    BundleInventoryError,
    build_inventory,
    verify_inventory,
)


def _release(tmp_path: Path) -> list[str]:
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "models/model.bin").write_bytes(b"model")
    (tmp_path / "data/features.csv").write_text("case_id,score\na,0.1\nb,0.2\n")
    (tmp_path / "reports/decision.json").write_text(json.dumps({"state": "PROMOTE"}))
    return ["models/model.bin", "data/features.csv", "reports/decision.json"]


def test_inventory_verifies_hash_size_and_csv_contract(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    inventory = build_inventory(
        tmp_path, paths, csv_contract_paths=["data/features.csv"]
    )
    result = verify_inventory(tmp_path, inventory, expected_paths=paths)
    assert result["verified_files"] == 3
    assert result["verified_csv_contracts"] == 1


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    inventory = build_inventory(tmp_path, paths)
    (tmp_path / "models/model.bin").unlink()
    with pytest.raises(BundleInventoryError, match="missing"):
        verify_inventory(tmp_path, inventory)


def test_content_tampering_is_rejected(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    inventory = build_inventory(tmp_path, paths)
    (tmp_path / "models/model.bin").write_bytes(b"tampered")
    with pytest.raises(BundleInventoryError, match="mismatch"):
        verify_inventory(tmp_path, inventory)


def test_csv_row_truncation_is_rejected(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    inventory = build_inventory(
        tmp_path, paths, csv_contract_paths=["data/features.csv"]
    )
    (tmp_path / "data/features.csv").write_text("case_id,score\na,0.1\n")
    with pytest.raises(BundleInventoryError, match="mismatch"):
        verify_inventory(tmp_path, inventory)


def test_expected_path_set_must_match(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    inventory = build_inventory(tmp_path, paths)
    with pytest.raises(BundleInventoryError, match="path-set mismatch"):
        verify_inventory(tmp_path, inventory, expected_paths=paths + ["missing.json"])


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside")
    with pytest.raises(BundleInventoryError, match="unsafe"):
        build_inventory(tmp_path, ["../outside.txt"])


def test_policy_contract_rejects_bad_csv_before_signing(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    contracts = {
        "data/features.csv": {
            "type": "csv",
            "required_columns": ["case_id", "score", "decision_time"],
            "min_rows": 2,
        }
    }
    with pytest.raises(BundleInventoryError, match="required columns missing"):
        build_inventory(tmp_path, paths, content_contracts=contracts)


def test_policy_contract_rejects_empty_csv_before_signing(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    (tmp_path / "data/features.csv").write_text("case_id,score\n")
    contracts = {
        "data/features.csv": {
            "type": "csv",
            "required_columns": ["case_id", "score"],
            "min_rows": 1,
        }
    }
    with pytest.raises(BundleInventoryError, match="fewer rows"):
        build_inventory(tmp_path, paths, content_contracts=contracts)


def test_json_contract_rejects_missing_required_key(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    contracts = {
        "reports/decision.json": {
            "type": "json",
            "json_type": "object",
            "required_keys": ["state", "selected_model"],
            "min_items": 2,
        }
    }
    with pytest.raises(BundleInventoryError, match="required keys missing"):
        build_inventory(tmp_path, paths, content_contracts=contracts)


def test_expected_content_contract_must_match_embedded_policy(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    embedded = {
        "data/features.csv": {
            "type": "csv",
            "required_columns": ["case_id", "score"],
            "min_rows": 1,
        }
    }
    inventory = build_inventory(tmp_path, paths, content_contracts=embedded)
    stronger = {
        "data/features.csv": {
            "type": "csv",
            "required_columns": ["case_id", "score"],
            "min_rows": 2,
        }
    }
    with pytest.raises(BundleInventoryError, match="contract-set mismatch"):
        verify_inventory(
            tmp_path,
            inventory,
            expected_paths=paths,
            expected_contracts=stronger,
        )


def test_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "actual.bin"
    target.write_bytes(b"value")
    link = tmp_path / "linked.bin"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(BundleInventoryError, match="symlink"):
        build_inventory(tmp_path, ["linked.bin"])


def test_unknown_inventory_schema_is_rejected(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    inventory = build_inventory(tmp_path, paths)
    inventory["schema_version"] = "999"
    with pytest.raises(BundleInventoryError, match="schema_version"):
        verify_inventory(tmp_path, inventory)


def test_csv_row_width_mismatch_is_rejected_before_signing(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    (tmp_path / "data/features.csv").write_text("case_id,score\na,0.1\nb\n")
    contracts = {
        "data/features.csv": {
            "type": "csv",
            "required_columns": ["case_id", "score"],
            "min_rows": 1,
        }
    }
    with pytest.raises(BundleInventoryError, match="row width mismatch"):
        build_inventory(tmp_path, paths, content_contracts=contracts)


def test_duplicate_json_keys_are_rejected_before_signing(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    (tmp_path / "reports/decision.json").write_text(
        '{"state":"PROMOTE","state":"HOLD"}'
    )
    contracts = {
        "reports/decision.json": {
            "type": "json",
            "json_type": "object",
            "required_keys": ["state"],
        }
    }
    with pytest.raises(BundleInventoryError, match="duplicate JSON key"):
        build_inventory(tmp_path, paths, content_contracts=contracts)


def test_non_finite_json_is_rejected_before_signing(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    (tmp_path / "reports/decision.json").write_text('{"score": NaN}')
    contracts = {
        "reports/decision.json": {
            "type": "json",
            "json_type": "object",
            "required_keys": ["score"],
        }
    }
    with pytest.raises(BundleInventoryError, match="non-finite"):
        build_inventory(tmp_path, paths, content_contracts=contracts)


def test_backslash_path_is_rejected_for_cross_platform_portability(
    tmp_path: Path,
) -> None:
    with pytest.raises(BundleInventoryError, match="non-portable"):
        build_inventory(tmp_path, [r"models\\model.bin"])


def test_parent_directory_symlink_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "model.bin").write_bytes(b"model")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(BundleInventoryError, match="symlink"):
        build_inventory(tmp_path, ["alias/model.bin"])


def test_unknown_content_contract_field_is_rejected(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    with pytest.raises(
        BundleInventoryError, match="unknown csv content contract fields"
    ):
        build_inventory(
            tmp_path,
            paths,
            content_contracts={
                "data/features.csv": {
                    "type": "csv",
                    "required_column": ["case_id"],
                }
            },
        )


def test_numeric_contract_values_must_be_actual_integers(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    with pytest.raises(BundleInventoryError, match="non-negative integer"):
        build_inventory(
            tmp_path,
            paths,
            content_contracts={"data/features.csv": {"type": "csv", "min_rows": "2"}},
        )


def test_inventory_rejects_unknown_entry_fields(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    inventory = build_inventory(tmp_path, paths)
    inventory["files"][0]["ignored_policy"] = True
    with pytest.raises(BundleInventoryError, match="unknown inventory entry fields"):
        verify_inventory(tmp_path, inventory)


def test_hard_linked_artifact_is_rejected(tmp_path: Path) -> None:
    import os

    paths = _release(tmp_path)
    alias = tmp_path / "models/model-alias.bin"
    try:
        os.link(tmp_path / "models/model.bin", alias)
    except OSError:
        pytest.skip("hard links are unavailable")
    with pytest.raises(BundleInventoryError, match="hard-linked"):
        build_inventory(tmp_path, paths)


def test_release_root_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _release(real)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(BundleInventoryError, match="release root"):
        build_inventory(linked, ["models/model.bin"])
