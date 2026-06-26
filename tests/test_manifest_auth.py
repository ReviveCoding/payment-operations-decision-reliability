from pathlib import Path

import pytest

from payment_ops_hardening.manifest_auth import (
    ManifestAuthenticationError,
    sign_manifest,
    verify_manifest_signature,
)

KEY = "k" * 32


def test_signed_manifest_verifies(tmp_path: Path) -> None:
    manifest = {"run_id": "r1", "release_state": "PROMOTE"}
    sign_manifest(manifest, tmp_path, key=KEY)
    result = verify_manifest_signature(
        manifest, tmp_path, key=KEY, require_signature=True
    )
    assert result["status"] == "VERIFIED"


def test_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    manifest = {"run_id": "r1", "release_state": "PROMOTE"}
    sign_manifest(manifest, tmp_path, key=KEY)
    manifest["release_state"] = "HOLD"
    with pytest.raises(ManifestAuthenticationError, match="verification failed"):
        verify_manifest_signature(manifest, tmp_path, key=KEY, require_signature=True)


def test_wrong_key_is_rejected(tmp_path: Path) -> None:
    manifest = {"run_id": "r1"}
    sign_manifest(manifest, tmp_path, key=KEY)
    with pytest.raises(ManifestAuthenticationError, match="verification failed"):
        verify_manifest_signature(manifest, tmp_path, key="x" * 32)


def test_unsigned_bundle_can_be_explicitly_allowed(tmp_path: Path) -> None:
    result = verify_manifest_signature(
        {"run_id": "r1"}, tmp_path, require_signature=False
    )
    assert result["status"] == "UNSIGNED_ALLOWED"


def test_unsigned_bundle_is_rejected_when_required(tmp_path: Path) -> None:
    with pytest.raises(ManifestAuthenticationError, match="required but missing"):
        verify_manifest_signature({"run_id": "r1"}, tmp_path, require_signature=True)


def test_short_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestAuthenticationError, match="at least 32 bytes"):
        sign_manifest({"run_id": "r1"}, tmp_path, key="short")


def test_key_id_has_portable_safe_format(tmp_path: Path) -> None:
    from payment_ops_hardening.manifest_auth import normalized_key_id

    assert normalized_key_id("paymentops-2026:q2") == "paymentops-2026:q2"
    with pytest.raises(ManifestAuthenticationError, match="must match"):
        normalized_key_id("../../unsafe")


def test_signature_required_environment_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from payment_ops_hardening.manifest_auth import signature_required

    monkeypatch.setenv("PAYMENT_OPS_REQUIRE_MANIFEST_SIGNATURE", "tru")
    with pytest.raises(ManifestAuthenticationError, match="must be one of"):
        signature_required()


def test_signature_hardlink_is_rejected(tmp_path: Path) -> None:
    manifest = {"run_id": "r1"}
    sign_manifest(manifest, tmp_path, key=KEY)
    signature = tmp_path / "release_manifest.sig"
    linked = tmp_path / "signature-copy.sig"
    try:
        linked.hardlink_to(signature)
    except (OSError, NotImplementedError):
        pytest.skip("hard links are unavailable")
    signature.unlink()
    signature.hardlink_to(linked)
    with pytest.raises(ManifestAuthenticationError, match="unreadable or unstable"):
        verify_manifest_signature(manifest, tmp_path, key=KEY, require_signature=True)
