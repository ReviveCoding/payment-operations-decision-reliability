import json
from pathlib import Path

import pytest

from payment_ops_hardening.release_security import (
    ReleaseSecurityError,
    finalize_release_security,
    verify_release_security,
)
from payment_ops_hardening.semantic_invariants import (
    SemanticInvariantError,
    normalize_json_value_equalities,
    resolve_json_pointer,
)

KEY = "e" * 32


def _release(root: Path, *, status_run_id: str = "r1") -> list[str]:
    (root / "reports").mkdir()
    (root / "release_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "release_state": "PROMOTE",
                "selected_model": "logistic",
                "selected_policy": "urgency",
            }
        ),
        encoding="utf-8",
    )
    (root / "run_status.json").write_text(
        json.dumps(
            {
                "run_id": status_run_id,
                "status": "SUCCESS",
                "release_state": "PROMOTE",
            }
        ),
        encoding="utf-8",
    )
    (root / "reports/release_decision.json").write_text(
        json.dumps(
            {
                "release_state": "PROMOTE",
                "selected_model": "logistic",
                "selected_policy": "urgency",
            }
        ),
        encoding="utf-8",
    )
    return ["run_status.json", "reports/release_decision.json"]


def _equalities() -> list[dict]:
    return [
        {
            "name": "run_id_consistency",
            "references": [
                {"path": "release_manifest.json", "pointer": "/run_id"},
                {"path": "run_status.json", "pointer": "/run_id"},
            ],
        },
        {
            "name": "release_state_consistency",
            "references": [
                {"path": "release_manifest.json", "pointer": "/release_state"},
                {"path": "run_status.json", "pointer": "/release_state"},
                {
                    "path": "reports/release_decision.json",
                    "pointer": "/release_state",
                },
            ],
        },
    ]


def test_json_pointer_distinguishes_boolean_and_integer() -> None:
    assert resolve_json_pointer({"value": True}, "/value") is True


def test_invalid_json_pointer_escape_is_rejected() -> None:
    with pytest.raises(SemanticInvariantError, match="invalid JSON pointer escape"):
        normalize_json_value_equalities(
            [
                {
                    "name": "bad",
                    "references": [
                        {"path": "a.json", "pointer": "/a~2b"},
                        {"path": "b.json", "pointer": "/value"},
                    ],
                }
            ]
        )


def test_finalize_blocks_cross_file_run_id_mismatch(tmp_path: Path) -> None:
    paths = _release(tmp_path, status_run_id="different")
    with pytest.raises(ReleaseSecurityError, match="run_id_consistency"):
        finalize_release_security(
            tmp_path,
            required_paths=paths,
            key=KEY,
            json_value_equalities=_equalities(),
        )


def test_verify_blocks_mismatch_even_when_bundle_was_finalized_without_policy(
    tmp_path: Path,
) -> None:
    paths = _release(tmp_path, status_run_id="different")
    finalize_release_security(tmp_path, required_paths=paths, key=KEY)
    with pytest.raises(ReleaseSecurityError, match="run_id_consistency"):
        verify_release_security(
            tmp_path,
            key=KEY,
            expected_paths=paths,
            json_value_equalities=_equalities(),
        )


def test_finalize_and_verify_report_equality_evidence(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    finalized = finalize_release_security(
        tmp_path,
        required_paths=paths,
        key=KEY,
        json_value_equalities=_equalities(),
    )
    verified = verify_release_security(
        tmp_path,
        key=KEY,
        expected_paths=paths,
        json_value_equalities=_equalities(),
    )
    assert finalized["verified_json_value_equalities"] == 2
    assert [item["name"] for item in verified["json_value_equalities"]] == [
        "run_id_consistency",
        "release_state_consistency",
    ]


def test_path_objects_are_normalized_for_equality_policy(tmp_path: Path) -> None:
    paths = _release(tmp_path)
    path_objects = [Path(path) for path in paths]
    result = finalize_release_security(
        tmp_path,
        required_paths=path_objects,
        key=KEY,
        json_value_equalities=_equalities(),
    )
    assert result["verified_json_value_equalities"] == 2
