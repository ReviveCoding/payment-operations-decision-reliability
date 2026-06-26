from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from filelock import FileLock, Timeout

from payment_ops_hardening.bundle_inventory import safe_relative_path
from payment_ops_hardening.secure_io import StableFileError, read_stable_bytes
from payment_ops_hardening.release_security import (
    ReleaseSecurityError,
    release_lock_path,
    verify_release_security,
)


def _materialize_artifact_paths(values: Iterable[str | Path]) -> list[str | Path]:
    if isinstance(values, (str, bytes, Path)):
        raise ReleaseSecurityError(
            "artifact_paths must be an iterable of paths, not a scalar"
        )
    try:
        result = list(values)
    except TypeError as exc:
        raise ReleaseSecurityError("artifact_paths must be iterable") from exc
    if not result or any(not isinstance(item, (str, Path)) for item in result):
        raise ReleaseSecurityError(
            "artifact_paths must contain one or more strings or Paths"
        )
    return result


def read_verified_artifacts(
    root: str | Path,
    artifact_paths: Iterable[str | Path],
    **verification_kwargs,
) -> dict:
    """Verify and read exact artifact bytes while holding the release lock.

    Loading joblib from ``io.BytesIO`` and XGBoost from ``bytearray`` of these bytes removes
    the path-replacement window between verification and deserialization. The lock prevents
    cooperating writers from replacing the bundle between verification and the byte read;
    size/hash checks still protect against non-cooperating mutation.
    """
    release_root = Path(root).resolve()
    requested_paths = _materialize_artifact_paths(artifact_paths)
    lock_timeout = verification_kwargs.get("lock_timeout", 0.0)
    try:
        timeout_value = float(lock_timeout)
    except (TypeError, ValueError) as exc:
        raise ReleaseSecurityError(
            "lock_timeout must be a non-negative number"
        ) from exc
    lock_path = release_lock_path(release_root)
    try:
        with FileLock(str(lock_path), timeout=timeout_value):
            verification = verify_release_security(
                release_root,
                _lock_acquired=True,
                **verification_kwargs,
            )
            expected = verification["inventory"].get("artifacts", {})
            payloads: dict[str, bytes] = {}
            for requested in requested_paths:
                relative = Path(requested).as_posix()
                metadata = expected.get(relative)
                if not isinstance(metadata, dict):
                    raise ReleaseSecurityError(
                        f"artifact is not present in verified inventory: {relative}"
                    )
                path = safe_relative_path(release_root, relative)
                try:
                    payload = read_stable_bytes(path, reject_hardlinks=True)
                except StableFileError as exc:
                    raise ReleaseSecurityError(
                        f"verified artifact cannot be read stably: {relative}: {exc}"
                    ) from exc
                if len(payload) != metadata.get("size_bytes"):
                    raise ReleaseSecurityError(
                        f"verified artifact size changed after verification: {relative}"
                    )
                if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
                    raise ReleaseSecurityError(
                        f"verified artifact checksum changed after verification: {relative}"
                    )
                payloads[relative] = payload
            return {"verification": verification, "artifacts": payloads}
    except Timeout as exc:
        raise ReleaseSecurityError(
            f"another release-security operation owns the lock: {lock_path}"
        ) from exc


def read_verified_artifact(
    root: str | Path,
    artifact_path: str | Path,
    **verification_kwargs,
) -> bytes:
    relative = Path(artifact_path).as_posix()
    result = read_verified_artifacts(
        root,
        [relative],
        **verification_kwargs,
    )
    return result["artifacts"][relative]
