from pathlib import Path

import pytest

import payment_ops_hardening.verified_loader as loader_module
from payment_ops_hardening.release_security import (
    ReleaseSecurityError,
    finalize_release_security,
)
from payment_ops_hardening.verified_loader import read_verified_artifact

KEY = "v" * 32


def _release(root: Path) -> None:
    (root / "models").mkdir()
    (root / "models/model.bin").write_bytes(b"trusted-model")
    (root / "release_manifest.json").write_text(
        '{"run_id":"r1","release_state":"PROMOTE"}',
        encoding="utf-8",
    )
    finalize_release_security(
        root,
        required_paths=["models/model.bin"],
        content_contracts={"models/model.bin": {"type": "binary", "min_size_bytes": 1}},
        key=KEY,
        key_id="k1",
        release_sequence=5,
        allowed_release_states=["PROMOTE"],
        reject_untracked_files=True,
    )


def test_verified_loader_returns_inventory_matched_bytes(tmp_path: Path) -> None:
    _release(tmp_path)
    payload = read_verified_artifact(
        tmp_path,
        "models/model.bin",
        key=KEY,
        expected_key_id="k1",
        minimum_release_sequence=5,
        allowed_release_states=["PROMOTE"],
        expected_paths=["models/model.bin"],
    )
    assert payload == b"trusted-model"


def test_verified_loader_detects_post_verification_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release(tmp_path)
    real_verify = loader_module.verify_release_security

    def verify_then_swap(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        (tmp_path / "models/model.bin").write_bytes(b"swapped-model")
        return result

    monkeypatch.setattr(loader_module, "verify_release_security", verify_then_swap)
    with pytest.raises(ReleaseSecurityError, match="changed after verification"):
        read_verified_artifact(tmp_path, "models/model.bin", key=KEY)


def test_verified_loader_rejects_noninventoried_path(tmp_path: Path) -> None:
    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="not present"):
        read_verified_artifact(tmp_path, "models/other.bin", key=KEY)


def test_verified_loader_rejects_scalar_artifact_paths(tmp_path: Path) -> None:
    from payment_ops_hardening.verified_loader import read_verified_artifacts

    _release(tmp_path)
    with pytest.raises(ReleaseSecurityError, match="not a scalar"):
        read_verified_artifacts(tmp_path, "models/model.bin", key=KEY)
