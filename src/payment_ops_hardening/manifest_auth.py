from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any

from payment_ops_hardening.atomic_io import atomic_write_bytes
from payment_ops_hardening.secure_io import StableFileError, read_stable_bytes

SIGNATURE_FILE = "release_manifest.sig"
SIGNATURE_ALGORITHM = "hmac-sha256"
KEY_ENV = "PAYMENT_OPS_MANIFEST_HMAC_KEY"
KEY_ID_ENV = "PAYMENT_OPS_MANIFEST_HMAC_KEY_ID"
REQUIRE_ENV = "PAYMENT_OPS_REQUIRE_MANIFEST_SIGNATURE"
_KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ManifestAuthenticationError(RuntimeError):
    """Raised when release-manifest authenticity cannot be established."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ManifestAuthenticationError(
            f"manifest is not strict JSON serializable: {exc}"
        ) from exc


def validated_key_bytes(key: str | bytes | None = None) -> bytes | None:
    supplied = key if key is not None else os.getenv(KEY_ENV)
    if supplied is None:
        return None
    if not isinstance(supplied, (str, bytes)):
        raise ManifestAuthenticationError("manifest HMAC key must be text or bytes")
    encoded = supplied if isinstance(supplied, bytes) else supplied.encode("utf-8")
    if len(encoded) < 32:
        raise ManifestAuthenticationError(
            "manifest HMAC key must contain at least 32 bytes"
        )
    return encoded


def normalized_key_id(key_id: str | None = None) -> str | None:
    supplied = key_id if key_id is not None else os.getenv(KEY_ID_ENV)
    if supplied is None:
        return None
    if not isinstance(supplied, str):
        raise ManifestAuthenticationError("manifest HMAC key_id must be text")
    value = supplied.strip()
    if not _KEY_ID_PATTERN.fullmatch(value):
        raise ManifestAuthenticationError(
            "manifest HMAC key_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
        )
    return value


def _safe_signature_path(root: str | Path) -> Path:
    release_root = Path(root)
    if release_root.is_symlink() or bool(
        getattr(release_root, "is_junction", lambda: False)()
    ):
        raise ManifestAuthenticationError(
            "release root cannot be a symlink or junction"
        )
    release_root = release_root.resolve()
    candidate = release_root / SIGNATURE_FILE
    if candidate.is_symlink() or bool(
        getattr(candidate, "is_junction", lambda: False)()
    ):
        raise ManifestAuthenticationError("manifest signature cannot be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(release_root)
    except ValueError as exc:
        raise ManifestAuthenticationError(
            "manifest signature path escapes release root"
        ) from exc
    return resolved


def signature_required(explicit: bool | None = None) -> bool:
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise ManifestAuthenticationError("require_signature must be boolean")
        return explicit
    raw = os.getenv(REQUIRE_ENV)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ManifestAuthenticationError(
        f"{REQUIRE_ENV} must be one of 1/0, true/false, yes/no, or on/off"
    )


def manifest_signature_digest(manifest: dict, key: str | bytes | None = None) -> str:
    secret = validated_key_bytes(key)
    if secret is None:
        raise ManifestAuthenticationError(f"{KEY_ENV} is not configured")
    return hmac.new(secret, canonical_json_bytes(manifest), hashlib.sha256).hexdigest()


def sign_manifest(
    manifest: dict,
    root: str | Path,
    *,
    key: str | bytes | None = None,
) -> str:
    digest = manifest_signature_digest(manifest, key)
    atomic_write_bytes(_safe_signature_path(root), f"{digest}\n".encode("ascii"))
    return digest


def verify_manifest_signature(
    manifest: dict,
    root: str | Path,
    *,
    key: str | bytes | None = None,
    require_signature: bool | None = None,
) -> dict:
    signature_path = _safe_signature_path(root)
    required = signature_required(require_signature)
    if not signature_path.is_file():
        if required:
            raise ManifestAuthenticationError(
                "release manifest signature is required but missing"
            )
        return {"status": "UNSIGNED_ALLOWED", "algorithm": None}

    secret = validated_key_bytes(key)
    if secret is None:
        raise ManifestAuthenticationError(
            f"signed release requires {KEY_ENV} for verification"
        )
    try:
        observed = (
            read_stable_bytes(signature_path, reject_hardlinks=True)
            .decode("ascii")
            .strip()
        )
    except (StableFileError, UnicodeError) as exc:
        raise ManifestAuthenticationError(
            "manifest signature is unreadable or unstable"
        ) from exc
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ManifestAuthenticationError(
            "manifest signature is not a lowercase SHA-256 hex digest"
        )
    expected = hmac.new(
        secret, canonical_json_bytes(manifest), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise ManifestAuthenticationError("manifest signature verification failed")
    return {"status": "VERIFIED", "algorithm": SIGNATURE_ALGORITHM}
