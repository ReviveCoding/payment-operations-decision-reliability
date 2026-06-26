import json
from pathlib import Path

import pytest
from filelock import FileLock

import payment_ops_hardening.release_security as release_security_module
from payment_ops_hardening.release_security import (
    ReleaseSecurityError,
    finalize_release_security,
    verify_release_security,
)

KEY = "s" * 32
PATHS = ["models/model.bin", "data/features.csv", "reports/decision.json"]


def _release(root: Path) -> None:
    (root / "models").mkdir()
    (root / "data").mkdir()
    (root / "reports").mkdir()
    (root / "models/model.bin").write_bytes(b"model")
    (root / "data/features.csv").write_text("case_id,score\na,0.1\nb,0.2\n")
    (root / "reports/decision.json").write_text('{"state":"PROMOTE"}')
    (root / "release_manifest.json").write_text(
        json.dumps({"run_id": "r1", "release_state": "PROMOTE"})
    )


def test_finalize_and_verify_signed_release(tmp_path: Path) -> None:
    _release(tmp_path)
    final = finalize_release_security(
        tmp_path,
        required_paths=PATHS,
        csv_contract_paths=["data/features.csv"],
        key=KEY,
        require_signature=True,
    )
    result = verify_release_security(
        tmp_path, key=KEY, require_signature=True, expected_paths=PATHS
    )
    assert final["inventoried_files"] == 3
    assert result["manifest_authentication"]["status"] == "VERIFIED"


def test_manifest_and_signature_replacement_without_key_fails(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(
        tmp_path, required_paths=PATHS, key=KEY, require_signature=True
    )
    manifest_path = tmp_path / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["release_state"] = "HOLD"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReleaseSecurityError, match="signature verification failed"):
        verify_release_security(tmp_path, key=KEY, require_signature=True)


def test_inventory_file_tampering_is_rejected_before_artifact_load(
    tmp_path: Path,
) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS)
    inventory_path = tmp_path / "release_bundle_inventory.json"
    inventory_path.write_text("{}")
    with pytest.raises(ReleaseSecurityError, match="inventory checksum mismatch"):
        verify_release_security(tmp_path)


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS)
    (tmp_path / "models/model.bin").write_bytes(b"malicious")
    with pytest.raises(ReleaseSecurityError, match="mismatch"):
        verify_release_security(tmp_path)


def test_signature_required_without_key_fails_at_finalize(tmp_path: Path) -> None:
    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="no HMAC key"):
        finalize_release_security(
            tmp_path, required_paths=PATHS, require_signature=True
        )


def test_signed_release_cannot_silently_downgrade_to_unsigned(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)
    (tmp_path / "release_manifest.sig").unlink()
    with pytest.raises(ReleaseSecurityError, match="required but missing"):
        verify_release_security(tmp_path, key=KEY)


def test_bad_key_fails_before_writing_security_artifacts(tmp_path: Path) -> None:
    _release(tmp_path)
    original_manifest = (tmp_path / "release_manifest.json").read_bytes()
    with pytest.raises(ReleaseSecurityError, match="at least 32 bytes"):
        finalize_release_security(tmp_path, required_paths=PATHS, key="short")
    assert (tmp_path / "release_manifest.json").read_bytes() == original_manifest
    assert not (tmp_path / "release_bundle_inventory.json").exists()
    assert not (tmp_path / "release_manifest.sig").exists()


def test_signed_release_requires_verification_key(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)
    with pytest.raises(ReleaseSecurityError, match="signed release requires"):
        verify_release_security(tmp_path)


def test_finalize_rejects_invalid_content_before_security_files(tmp_path: Path) -> None:
    _release(tmp_path)
    original_manifest = (tmp_path / "release_manifest.json").read_bytes()
    contracts = {
        "data/features.csv": {
            "type": "csv",
            "required_columns": ["case_id", "score", "missing"],
            "min_rows": 1,
        }
    }
    with pytest.raises(ReleaseSecurityError, match="required columns missing"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            content_contracts=contracts,
            key=KEY,
        )
    assert (tmp_path / "release_manifest.json").read_bytes() == original_manifest
    assert not (tmp_path / "release_bundle_inventory.json").exists()
    assert not (tmp_path / "release_manifest.sig").exists()


def test_key_id_is_verified_for_rotation_policy(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(
        tmp_path,
        required_paths=PATHS,
        key=KEY,
        key_id="2026-q2",
    )
    result = verify_release_security(
        tmp_path,
        key=KEY,
        expected_key_id="2026-q2",
    )
    assert result["key_id"] == "2026-q2"
    with pytest.raises(ReleaseSecurityError, match="key_id mismatch"):
        verify_release_security(
            tmp_path,
            key=KEY,
            expected_key_id="2026-q3",
        )


def test_signature_algorithm_metadata_tampering_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)
    manifest_path = tmp_path / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bundle_security"]["signature_algorithm"] = "none"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        ReleaseSecurityError, match="unsupported manifest signature algorithm"
    ):
        verify_release_security(tmp_path, key=KEY)


def test_finalize_failure_restores_previous_security_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY, key_id="old")
    original = {
        name: (tmp_path / name).read_bytes()
        for name in [
            "release_manifest.json",
            "release_bundle_inventory.json",
            "release_manifest.sig",
        ]
    }

    def fail_signature(*args, **kwargs):
        raise RuntimeError("simulated signing failure")

    monkeypatch.setattr(
        release_security_module, "manifest_signature_digest", fail_signature
    )
    with pytest.raises(ReleaseSecurityError, match="simulated signing failure"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            key=KEY,
            key_id="new",
        )
    for name, payload in original.items():
        assert (tmp_path / name).read_bytes() == payload


def test_concurrent_security_writer_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    lock_path = tmp_path.parent / f".{tmp_path.name}.release_security.lock"
    with FileLock(str(lock_path), timeout=0):
        with pytest.raises(ReleaseSecurityError, match="owns the lock"):
            finalize_release_security(
                tmp_path,
                required_paths=PATHS,
                key=KEY,
            )


def test_signed_release_cannot_be_refinalized_unsigned(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)
    original_manifest = (tmp_path / "release_manifest.json").read_bytes()
    original_signature = (tmp_path / "release_manifest.sig").read_bytes()
    with pytest.raises(
        ReleaseSecurityError, match="cannot be re-finalized as unsigned"
    ):
        finalize_release_security(tmp_path, required_paths=PATHS)
    assert (tmp_path / "release_manifest.json").read_bytes() == original_manifest
    assert (tmp_path / "release_manifest.sig").read_bytes() == original_signature


def test_duplicate_manifest_keys_are_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    (tmp_path / "release_manifest.json").write_text(
        '{"run_id":"r1","run_id":"r2","release_state":"PROMOTE"}'
    )
    with pytest.raises(ReleaseSecurityError, match="duplicate JSON key"):
        finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)


def test_non_finite_manifest_number_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    (tmp_path / "release_manifest.json").write_text(
        '{"run_id":"r1","release_state":"PROMOTE","metric":NaN}'
    )
    with pytest.raises(ReleaseSecurityError, match="non-finite"):
        finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)


def test_hold_release_is_blocked_by_allowed_state_policy(tmp_path: Path) -> None:
    _release(tmp_path)
    manifest_path = tmp_path / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["release_state"] = "HOLD"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReleaseSecurityError, match="not allowed"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            key=KEY,
            allowed_release_states=["PROMOTE"],
        )


def test_untracked_file_is_blocked_at_finalize(tmp_path: Path) -> None:
    _release(tmp_path)
    (tmp_path / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ReleaseSecurityError, match="untracked"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            key=KEY,
            reject_untracked_files=True,
        )


def test_untracked_file_is_blocked_after_release(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(
        tmp_path,
        required_paths=PATHS,
        key=KEY,
        reject_untracked_files=True,
    )
    (tmp_path / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ReleaseSecurityError, match="untracked"):
        verify_release_security(
            tmp_path,
            key=KEY,
            expected_paths=PATHS,
            reject_untracked_files=True,
        )


def test_allowed_untracked_operational_file_is_reported(tmp_path: Path) -> None:
    _release(tmp_path)
    (tmp_path / "runtime.log").write_text("ok")
    finalize_release_security(
        tmp_path,
        required_paths=PATHS,
        key=KEY,
        reject_untracked_files=True,
        allowed_untracked_paths=["runtime.log"],
    )
    result = verify_release_security(
        tmp_path,
        key=KEY,
        expected_paths=PATHS,
        reject_untracked_files=True,
        allowed_untracked_paths=["runtime.log"],
    )
    assert result["inventory"]["file_set"]["allowed_untracked_files"] == ["runtime.log"]


def test_minimum_release_sequence_blocks_replay(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(
        tmp_path,
        required_paths=PATHS,
        key=KEY,
        release_sequence=10,
    )
    result = verify_release_security(
        tmp_path,
        key=KEY,
        minimum_release_sequence=10,
    )
    assert result["release_sequence"] == 10
    with pytest.raises(ReleaseSecurityError, match="below the deployment minimum"):
        verify_release_security(
            tmp_path,
            key=KEY,
            minimum_release_sequence=11,
        )


def test_release_sequence_must_be_nonnegative_integer(tmp_path: Path) -> None:
    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="release_sequence"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            key=KEY,
            release_sequence=-1,
        )


def test_manifest_symlink_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    external = tmp_path.parent / f"{tmp_path.name}-external-manifest.json"
    external.write_text('{"run_id":"outside","release_state":"PROMOTE"}')
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ReleaseSecurityError, match="symlink"):
        finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)


def test_signature_symlink_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)
    signature = tmp_path / "release_manifest.sig"
    external = tmp_path.parent / f"{tmp_path.name}-external-signature.sig"
    external.write_bytes(signature.read_bytes())
    signature.unlink()
    try:
        signature.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ReleaseSecurityError, match="symlink"):
        verify_release_security(tmp_path, key=KEY)


def test_generator_required_paths_are_materialized_once(tmp_path: Path) -> None:
    _release(tmp_path)
    required_generator = (item for item in PATHS)
    result = finalize_release_security(
        tmp_path,
        required_paths=required_generator,
        key=KEY,
        reject_untracked_files=True,
    )
    assert result["inventoried_files"] == len(PATHS)


def test_negative_lock_timeout_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="lock_timeout"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            key=KEY,
            lock_timeout=-1,
        )


def test_empty_allowed_release_states_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="allowed_release_states"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            key=KEY,
            allowed_release_states=[],
        )


def test_required_paths_scalar_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="iterable of paths, not a scalar"):
        finalize_release_security(tmp_path, required_paths="models/model.bin")


def test_allowed_states_scalar_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="not a scalar"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            allowed_release_states="PROMOTE",
        )


def test_environment_signature_requirement_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release(tmp_path)
    monkeypatch.setenv("PAYMENT_OPS_REQUIRE_MANIFEST_SIGNATURE", "true")
    with pytest.raises(ReleaseSecurityError, match="no HMAC key"):
        finalize_release_security(tmp_path, required_paths=PATHS)


def test_refinalization_preserves_key_id_and_sequence(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(
        tmp_path,
        required_paths=PATHS,
        key=KEY,
        key_id="2026-q2",
        release_sequence=8,
    )
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)
    result = verify_release_security(
        tmp_path,
        key=KEY,
        expected_key_id="2026-q2",
        minimum_release_sequence=8,
    )
    assert result["key_id"] == "2026-q2"
    assert result["release_sequence"] == 8


def test_refinalization_cannot_decrease_sequence(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(
        tmp_path,
        required_paths=PATHS,
        key=KEY,
        key_id="2026-q2",
        release_sequence=8,
    )
    with pytest.raises(ReleaseSecurityError, match="cannot decrease"):
        finalize_release_security(
            tmp_path,
            required_paths=PATHS,
            key=KEY,
            release_sequence=7,
        )
    result = verify_release_security(
        tmp_path,
        key=KEY,
        expected_key_id="2026-q2",
        minimum_release_sequence=8,
    )
    assert result["release_sequence"] == 8


def test_pending_transaction_is_recovered_before_verification(tmp_path: Path) -> None:
    _release(tmp_path)
    finalize_release_security(tmp_path, required_paths=PATHS, key=KEY)
    manifest_path = tmp_path / "release_manifest.json"
    inventory_path = tmp_path / "release_bundle_inventory.json"
    signature_path = tmp_path / "release_manifest.sig"
    original = {
        manifest_path: manifest_path.read_bytes(),
        inventory_path: inventory_path.read_bytes(),
        signature_path: signature_path.read_bytes(),
    }
    transaction = release_security_module._write_transaction(
        tmp_path.resolve(), original
    )
    manifest_path.write_text('{"broken":true}', encoding="utf-8")
    inventory_path.write_text('{"broken":true}', encoding="utf-8")
    signature_path.write_text("0" * 64, encoding="ascii")
    result = verify_release_security(tmp_path, key=KEY)
    assert result["manifest_authentication"]["status"] == "VERIFIED"
    assert not transaction.exists()
    assert manifest_path.read_bytes() == original[manifest_path]
    assert inventory_path.read_bytes() == original[inventory_path]
    assert signature_path.read_bytes() == original[signature_path]
