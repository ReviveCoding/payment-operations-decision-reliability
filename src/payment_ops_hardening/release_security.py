from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable, Mapping

from filelock import FileLock, Timeout

from payment_ops_hardening.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    fsync_directory,
)
from payment_ops_hardening.bundle_inventory import (
    INVENTORY_FILE,
    INVENTORY_SCHEMA_VERSION,
    BundleInventoryError,
    build_inventory,
    normalize_relative_path,
    safe_relative_path,
    sha256_file,
    verify_inventory,
    verify_no_untracked_files,
    write_inventory,
)
from payment_ops_hardening.manifest_auth import (
    ManifestAuthenticationError,
    SIGNATURE_ALGORITHM,
    SIGNATURE_FILE,
    manifest_signature_digest,
    normalized_key_id,
    signature_required,
    validated_key_bytes,
    verify_manifest_signature,
)
from payment_ops_hardening.semantic_invariants import (
    SemanticInvariantError,
    normalize_json_value_equalities,
    verify_json_value_equalities,
)
from payment_ops_hardening.strict_json import (
    StrictJSONError,
    read_strict_json,
    strict_json_loads,
)

SECURITY_SCHEMA_VERSION = "2.1"
LOCK_FILE = ".release_security.lock"
TRANSACTION_SCHEMA_VERSION = "1.0"
TRANSACTION_SUFFIX = ".release_security.transaction.json"
CONTROL_PATHS = {"release_manifest.json", INVENTORY_FILE, SIGNATURE_FILE}
_SECURITY_FIELDS = {
    "schema_version",
    "inventory_file",
    "inventory_schema_version",
    "inventory_sha256",
    "signature_algorithm",
    "signature_file",
    "signature_required",
    "key_id",
    "release_sequence",
}


class ReleaseSecurityError(RuntimeError):
    """Unified release-hardening error exposed to pipelines and serving code."""


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _release_root(root: str | Path, *, create: bool = False) -> Path:
    source = Path(root)
    if _is_link_like(source):
        raise ReleaseSecurityError(
            f"release root cannot be a symlink or junction: {source}"
        )
    if create:
        source.mkdir(parents=True, exist_ok=True)
    if not source.exists() or not source.is_dir():
        raise ReleaseSecurityError(f"release root is not a directory: {source}")
    return source.resolve()


def _safe_sibling_control_path(root: Path, suffix: str, label: str) -> Path:
    candidate = root.parent / f".{root.name}{suffix}"
    if _is_link_like(candidate):
        raise ReleaseSecurityError(f"{label} cannot be a symlink or junction")
    return candidate


def release_lock_path(root: str | Path) -> Path:
    release_root = _release_root(root)
    return _safe_sibling_control_path(release_root, LOCK_FILE, "release lock")


def _transaction_path(root: Path) -> Path:
    return _safe_sibling_control_path(
        root, TRANSACTION_SUFFIX, "release-security transaction file"
    )


def _read_json(path: Path) -> dict:
    try:
        value = read_strict_json(path)
    except StrictJSONError as exc:
        raise ReleaseSecurityError(f"invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseSecurityError(f"JSON object required: {path}")
    return value


def _snapshot(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _restore(path: Path, value: bytes | None) -> None:
    if value is None:
        path.unlink(missing_ok=True)
        fsync_directory(path.parent)
    else:
        atomic_write_bytes(path, value)


def _encode_snapshot(value: bytes | None) -> str | None:
    return None if value is None else base64.b64encode(value).decode("ascii")


def _decode_snapshot(value: object, label: str) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReleaseSecurityError(f"invalid transaction snapshot for {label}")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ReleaseSecurityError(f"invalid transaction snapshot for {label}") from exc


def _write_transaction(root: Path, original: dict[Path, bytes | None]) -> Path:
    path = _transaction_path(root)
    controls = {item.name: _encode_snapshot(value) for item, value in original.items()}
    atomic_write_json(
        path,
        {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "release_root": str(root),
            "controls": controls,
        },
    )
    return path


def _recover_pending_transaction(root: Path) -> bool:
    path = _transaction_path(root)
    if not path.exists():
        return False
    transaction = _read_json(path)
    if set(transaction) != {"schema_version", "release_root", "controls"}:
        raise ReleaseSecurityError("release-security transaction has unknown fields")
    if transaction.get("schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise ReleaseSecurityError("unsupported release-security transaction schema")
    if transaction.get("release_root") != str(root):
        raise ReleaseSecurityError("release-security transaction targets another root")
    controls = transaction.get("controls")
    if not isinstance(controls, dict) or set(controls) != CONTROL_PATHS:
        raise ReleaseSecurityError(
            "release-security transaction control set is invalid"
        )
    for name in sorted(CONTROL_PATHS):
        target = safe_relative_path(root, name)
        _restore(target, _decode_snapshot(controls[name], name))
    path.unlink(missing_ok=True)
    fsync_directory(path.parent)
    return True


def _materialize_paths(values: Iterable[str | Path], field: str) -> list[str | Path]:
    if isinstance(values, (str, bytes, Path)):
        raise ReleaseSecurityError(
            f"{field} must be an iterable of paths, not a scalar"
        )
    try:
        result = list(values)
    except TypeError as exc:
        raise ReleaseSecurityError(f"{field} must be iterable") from exc
    if not result and field == "required_paths":
        raise ReleaseSecurityError("required_paths cannot be empty")
    if any(not isinstance(item, (str, Path)) for item in result):
        raise ReleaseSecurityError(f"{field} must contain only strings or Paths")
    return result


def _materialize_states(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ReleaseSecurityError(
            "allowed_release_states must be an iterable of states, not a scalar"
        )
    try:
        result = list(values)
    except TypeError as exc:
        raise ReleaseSecurityError("allowed_release_states must be iterable") from exc
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ReleaseSecurityError(
            "allowed_release_states must contain non-empty strings"
        )
    if len(result) != len(set(result)):
        raise ReleaseSecurityError("allowed_release_states must be unique")
    return result


def _validate_lock_timeout(value: float) -> float:
    if isinstance(value, bool):
        raise ReleaseSecurityError("lock_timeout must be a non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseSecurityError(
            "lock_timeout must be a non-negative number"
        ) from exc
    if result < 0:
        raise ReleaseSecurityError("lock_timeout must be a non-negative number")
    return result


def _validate_security_contract(security: dict) -> tuple[bool, str | None]:
    unknown = sorted(set(security) - _SECURITY_FIELDS)
    if unknown:
        raise ReleaseSecurityError(f"unknown bundle_security fields: {unknown}")
    missing = sorted(_SECURITY_FIELDS - set(security))
    if missing:
        raise ReleaseSecurityError(f"bundle_security fields are missing: {missing}")
    if security.get("schema_version") != SECURITY_SCHEMA_VERSION:
        raise ReleaseSecurityError(
            f"unsupported bundle_security schema_version: {security.get('schema_version')!r}"
        )
    if security.get("inventory_file") != INVENTORY_FILE:
        raise ReleaseSecurityError("unexpected release inventory filename")
    if security.get("inventory_schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ReleaseSecurityError("unexpected release inventory schema version")
    inventory_hash = security.get("inventory_sha256")
    if (
        not isinstance(inventory_hash, str)
        or len(inventory_hash) != 64
        or any(character not in "0123456789abcdef" for character in inventory_hash)
    ):
        raise ReleaseSecurityError("release inventory checksum is invalid")
    if not isinstance(security.get("signature_required"), bool):
        raise ReleaseSecurityError("signature_required must be boolean")

    algorithm = security.get("signature_algorithm")
    signature_file = security.get("signature_file")
    key_id = security.get("key_id")
    declared_signed = (
        algorithm is not None or signature_file is not None or key_id is not None
    )
    if declared_signed:
        if algorithm != SIGNATURE_ALGORITHM:
            raise ReleaseSecurityError("unsupported manifest signature algorithm")
        if signature_file != SIGNATURE_FILE:
            raise ReleaseSecurityError("unexpected manifest signature filename")
        try:
            normalized = normalized_key_id(key_id)
        except ManifestAuthenticationError as exc:
            raise ReleaseSecurityError(str(exc)) from exc
        if normalized != key_id:
            raise ReleaseSecurityError("signed release key_id is not normalized")
    elif security["signature_required"]:
        raise ReleaseSecurityError(
            "signature is required but signing metadata is absent"
        )
    release_sequence = security.get("release_sequence")
    if release_sequence is not None and (
        isinstance(release_sequence, bool)
        or not isinstance(release_sequence, int)
        or release_sequence < 0
    ):
        raise ReleaseSecurityError("release_sequence must be a non-negative integer")
    return declared_signed, key_id if isinstance(key_id, str) else None


def _previous_security_metadata(
    manifest_snapshot: bytes | None, signature_snapshot: bytes | None
) -> dict:
    if manifest_snapshot is None:
        return {
            "signed": signature_snapshot is not None,
            "key_id": None,
            "sequence": None,
        }
    try:
        manifest = strict_json_loads(manifest_snapshot.decode("utf-8"))
    except (UnicodeError, StrictJSONError):
        return {
            "signed": signature_snapshot is not None,
            "key_id": None,
            "sequence": None,
        }
    security = manifest.get("bundle_security") if isinstance(manifest, dict) else None
    if not isinstance(security, dict):
        return {
            "signed": signature_snapshot is not None,
            "key_id": None,
            "sequence": None,
        }
    declared_signed = any(
        security.get(field) is not None
        for field in ("signature_algorithm", "signature_file", "key_id")
    )
    sequence = security.get("release_sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, (int, type(None))):
        sequence = None
    return {
        "signed": bool(signature_snapshot is not None or declared_signed),
        "key_id": security.get("key_id")
        if isinstance(security.get("key_id"), str)
        else None,
        "sequence": sequence,
    }


def _verify_file_set(
    root: Path,
    inventoried_paths: Iterable[str | Path],
    allowed_untracked_paths: Iterable[str | Path],
) -> dict:
    user_allowed = {str(item) for item in allowed_untracked_paths}
    result = verify_no_untracked_files(
        root,
        inventoried_paths,
        allowed_untracked_paths=CONTROL_PATHS | user_allowed,
    )
    observed_allowed = set(result["allowed_untracked_files"])
    result["control_files"] = sorted(observed_allowed & CONTROL_PATHS)
    result["allowed_untracked_files"] = sorted(observed_allowed - CONTROL_PATHS)
    return result


def finalize_release_security(
    root: str | Path,
    *,
    required_paths: Iterable[str | Path],
    content_contracts: Mapping[str | Path, Mapping[str, object]] | None = None,
    csv_contract_paths: Iterable[str | Path] = (),
    key: str | bytes | None = None,
    key_id: str | None = None,
    require_signature: bool | None = None,
    release_sequence: int | None = None,
    allowed_release_states: Iterable[str] | None = None,
    reject_untracked_files: bool = False,
    allowed_untracked_paths: Iterable[str | Path] = (),
    json_value_equalities: Iterable[Mapping[str, object]] | None = None,
    lock_timeout: float = 0.0,
) -> dict:
    release_root = _release_root(root, create=True)
    required_path_list = _materialize_paths(required_paths, "required_paths")
    csv_contract_path_list = _materialize_paths(
        csv_contract_paths, "csv_contract_paths"
    )
    allowed_untracked_path_list = _materialize_paths(
        allowed_untracked_paths, "allowed_untracked_paths"
    )
    allowed_state_list = _materialize_states(allowed_release_states)
    try:
        normalized_equalities = normalize_json_value_equalities(json_value_equalities)
    except SemanticInvariantError as exc:
        raise ReleaseSecurityError(str(exc)) from exc
    if require_signature is not None and not isinstance(require_signature, bool):
        raise ReleaseSecurityError("require_signature must be boolean or null")
    if not isinstance(reject_untracked_files, bool):
        raise ReleaseSecurityError("reject_untracked_files must be boolean")
    lock_timeout_value = _validate_lock_timeout(lock_timeout)
    try:
        manifest_path = safe_relative_path(release_root, "release_manifest.json")
        inventory_path = safe_relative_path(release_root, INVENTORY_FILE)
        signature_path = safe_relative_path(release_root, SIGNATURE_FILE)
    except BundleInventoryError as exc:
        raise ReleaseSecurityError(str(exc)) from exc
    lock_path = release_lock_path(release_root)
    lock = FileLock(str(lock_path), timeout=lock_timeout_value)

    try:
        with lock:
            _recover_pending_transaction(release_root)
            original = {
                manifest_path: _snapshot(manifest_path),
                inventory_path: _snapshot(inventory_path),
                signature_path: _snapshot(signature_path),
            }
            manifest = _read_json(manifest_path)
            if allowed_state_list is not None and manifest.get(
                "release_state"
            ) not in set(allowed_state_list):
                raise ReleaseSecurityError(
                    f"release_state {manifest.get('release_state')!r} is not allowed"
                )
            if release_sequence is not None and (
                isinstance(release_sequence, bool)
                or not isinstance(release_sequence, int)
                or release_sequence < 0
            ):
                raise ReleaseSecurityError(
                    "release_sequence must be a non-negative integer"
                )
            previous = _previous_security_metadata(
                original[manifest_path], original[signature_path]
            )
            secret = validated_key_bytes(key)
            resolved_key_id = normalized_key_id(key_id)
            effective_required = bool(
                signature_required(None) or require_signature or previous["signed"]
            )
            if previous["signed"] and resolved_key_id is None:
                resolved_key_id = previous["key_id"]
            if previous["sequence"] is not None:
                if release_sequence is None:
                    release_sequence = previous["sequence"]
                elif release_sequence < previous["sequence"]:
                    raise ReleaseSecurityError(
                        "release_sequence cannot decrease during re-finalization; "
                        f"previous={previous['sequence']}, requested={release_sequence}"
                    )
            if previous["signed"] and secret is None:
                raise ReleaseSecurityError(
                    "signed release cannot be re-finalized as unsigned"
                )
            if effective_required and secret is None:
                raise ManifestAuthenticationError(
                    "signature required but no HMAC key was supplied"
                )
            if resolved_key_id is not None and secret is None:
                raise ManifestAuthenticationError(
                    "key_id cannot be set without an HMAC key"
                )
            if secret is not None and resolved_key_id is None:
                resolved_key_id = "default"
            invariant_allowed_paths = {
                normalize_relative_path(path) for path in required_path_list
            } | {"release_manifest.json"}
            try:
                invariant_evidence = verify_json_value_equalities(
                    release_root,
                    normalized_equalities,
                    allowed_paths=invariant_allowed_paths,
                )
            except SemanticInvariantError as exc:
                raise ReleaseSecurityError(str(exc)) from exc
            inventory = build_inventory(
                release_root,
                required_path_list,
                content_contracts=content_contracts,
                csv_contract_paths=csv_contract_path_list,
            )
            if reject_untracked_files:
                _verify_file_set(
                    release_root,
                    required_path_list,
                    allowed_untracked_path_list,
                )
            transaction_path = _write_transaction(release_root, original)
            try:
                write_inventory(release_root, inventory)
                # Re-verify payload bytes immediately before committing the signed manifest.
                verify_inventory(
                    release_root,
                    inventory,
                    expected_paths=required_path_list,
                    expected_contracts=content_contracts,
                )
                if reject_untracked_files:
                    _verify_file_set(
                        release_root,
                        required_path_list,
                        allowed_untracked_path_list,
                    )
                inventory_hash = sha256_file(inventory_path)
                manifest["bundle_security"] = {
                    "schema_version": SECURITY_SCHEMA_VERSION,
                    "inventory_file": INVENTORY_FILE,
                    "inventory_schema_version": INVENTORY_SCHEMA_VERSION,
                    "inventory_sha256": inventory_hash,
                    "signature_algorithm": SIGNATURE_ALGORITHM
                    if secret is not None
                    else None,
                    "signature_file": SIGNATURE_FILE if secret is not None else None,
                    "signature_required": bool(
                        secret is not None or effective_required
                    ),
                    "key_id": resolved_key_id,
                    "release_sequence": release_sequence,
                }
                signature = None
                if secret is not None:
                    signature = manifest_signature_digest(manifest, secret)
                    atomic_write_bytes(signature_path, f"{signature}\n".encode("ascii"))
                else:
                    signature_path.unlink(missing_ok=True)
                    fsync_directory(signature_path.parent)
                atomic_write_json(
                    manifest_path, manifest
                )  # commit marker, written last
                try:
                    invariant_evidence = verify_json_value_equalities(
                        release_root,
                        normalized_equalities,
                        allowed_paths=invariant_allowed_paths,
                    )
                except SemanticInvariantError as exc:
                    raise ReleaseSecurityError(str(exc)) from exc
                transaction_path.unlink(missing_ok=True)
                fsync_directory(transaction_path.parent)
            except BaseException:
                for path, value in original.items():
                    _restore(path, value)
                transaction_path.unlink(missing_ok=True)
                fsync_directory(transaction_path.parent)
                raise
    except Timeout as exc:
        raise ReleaseSecurityError(
            f"another release-security operation owns the lock: {lock_path}"
        ) from exc
    except (
        BundleInventoryError,
        ManifestAuthenticationError,
        SemanticInvariantError,
    ) as exc:
        raise ReleaseSecurityError(str(exc)) from exc
    except ReleaseSecurityError:
        raise
    except Exception as exc:
        raise ReleaseSecurityError(
            f"release security finalization failed: {exc}"
        ) from exc

    return {
        "inventory_file": str(inventory_path),
        "inventory_sha256": manifest["bundle_security"]["inventory_sha256"],
        "signature": signature,
        "key_id": resolved_key_id,
        "inventoried_files": len(inventory["files"]),
        "verified_json_value_equalities": len(invariant_evidence),
    }


def _verify_release_security_locked(
    release_root: Path,
    *,
    key: str | bytes | None,
    require_signature: bool | None,
    expected_key_id: str | None,
    minimum_release_sequence: int | None,
    expected_path_list: list[str | Path] | None,
    expected_contracts: Mapping[str | Path, Mapping[str, object]] | None,
    allowed_state_list: list[str] | None,
    reject_untracked_files: bool,
    allowed_untracked_path_list: list[str | Path],
    normalized_equalities: list[dict],
) -> dict:
    _recover_pending_transaction(release_root)
    manifest_path = safe_relative_path(release_root, "release_manifest.json")
    manifest = _read_json(manifest_path)
    if allowed_state_list is not None and manifest.get("release_state") not in set(
        allowed_state_list
    ):
        raise ReleaseSecurityError(
            f"release_state {manifest.get('release_state')!r} is not allowed"
        )
    security = manifest.get("bundle_security")
    if not isinstance(security, dict):
        raise ReleaseSecurityError("release manifest has no bundle_security contract")
    declared_signed, declared_key_id = _validate_security_contract(security)
    observed_sequence = security.get("release_sequence")
    if minimum_release_sequence is not None:
        if (
            isinstance(minimum_release_sequence, bool)
            or not isinstance(minimum_release_sequence, int)
            or minimum_release_sequence < 0
        ):
            raise ReleaseSecurityError(
                "minimum_release_sequence must be a non-negative integer"
            )
        if observed_sequence is None or observed_sequence < minimum_release_sequence:
            raise ReleaseSecurityError(
                "release sequence is below the deployment minimum; "
                f"minimum={minimum_release_sequence}, observed={observed_sequence}"
            )
    normalized_expected_key_id = normalized_key_id(expected_key_id)
    if (
        normalized_expected_key_id is not None
        and declared_key_id != normalized_expected_key_id
    ):
        raise ReleaseSecurityError(
            "release key_id mismatch; "
            f"expected={normalized_expected_key_id!r}, observed={declared_key_id!r}"
        )
    inventory_path = safe_relative_path(release_root, INVENTORY_FILE)
    if not inventory_path.is_file():
        raise ReleaseSecurityError("release bundle inventory is missing")
    if sha256_file(inventory_path) != security["inventory_sha256"]:
        raise ReleaseSecurityError("release bundle inventory checksum mismatch")
    inventory = _read_json(inventory_path)
    inventory_result = verify_inventory(
        release_root,
        inventory,
        expected_paths=expected_path_list,
        expected_contracts=expected_contracts,
    )
    if reject_untracked_files:
        inventory_result["file_set"] = _verify_file_set(
            release_root,
            inventory_result["paths"],
            allowed_untracked_path_list,
        )
    if require_signature is not None and not isinstance(require_signature, bool):
        raise ReleaseSecurityError("require_signature must be boolean or null")
    effective_signature_requirement = bool(
        signature_required(None)
        or security["signature_required"]
        or declared_signed
        or require_signature
    )
    signature_result = verify_manifest_signature(
        manifest,
        release_root,
        key=key,
        require_signature=effective_signature_requirement,
    )
    invariant_allowed_paths = (
        {normalize_relative_path(path) for path in expected_path_list}
        if expected_path_list is not None
        else set(inventory_result["paths"])
    ) | {"release_manifest.json"}
    try:
        invariant_evidence = verify_json_value_equalities(
            release_root,
            normalized_equalities,
            allowed_paths=invariant_allowed_paths,
        )
    except SemanticInvariantError as exc:
        raise ReleaseSecurityError(str(exc)) from exc
    return {
        "run_id": manifest.get("run_id"),
        "release_state": manifest.get("release_state"),
        "key_id": declared_key_id,
        "release_sequence": observed_sequence,
        "inventory": inventory_result,
        "manifest_authentication": signature_result,
        "json_value_equalities": invariant_evidence,
    }


def verify_release_security(
    root: str | Path,
    *,
    key: str | bytes | None = None,
    require_signature: bool | None = None,
    expected_key_id: str | None = None,
    minimum_release_sequence: int | None = None,
    expected_paths: Iterable[str | Path] | None = None,
    expected_contracts: Mapping[str | Path, Mapping[str, object]] | None = None,
    allowed_release_states: Iterable[str] | None = None,
    reject_untracked_files: bool = False,
    allowed_untracked_paths: Iterable[str | Path] = (),
    json_value_equalities: Iterable[Mapping[str, object]] | None = None,
    lock_timeout: float = 0.0,
    _lock_acquired: bool = False,
) -> dict:
    release_root = _release_root(root)
    expected_path_list = (
        None
        if expected_paths is None
        else _materialize_paths(expected_paths, "expected_paths")
    )
    allowed_untracked_path_list = _materialize_paths(
        allowed_untracked_paths, "allowed_untracked_paths"
    )
    allowed_state_list = _materialize_states(allowed_release_states)
    try:
        normalized_equalities = normalize_json_value_equalities(json_value_equalities)
    except SemanticInvariantError as exc:
        raise ReleaseSecurityError(str(exc)) from exc
    if require_signature is not None and not isinstance(require_signature, bool):
        raise ReleaseSecurityError("require_signature must be boolean or null")
    if not isinstance(reject_untracked_files, bool):
        raise ReleaseSecurityError("reject_untracked_files must be boolean")
    lock_timeout_value = _validate_lock_timeout(lock_timeout)

    def verify_once() -> dict:
        return _verify_release_security_locked(
            release_root,
            key=key,
            require_signature=require_signature,
            expected_key_id=expected_key_id,
            minimum_release_sequence=minimum_release_sequence,
            expected_path_list=expected_path_list,
            expected_contracts=expected_contracts,
            allowed_state_list=allowed_state_list,
            reject_untracked_files=reject_untracked_files,
            allowed_untracked_path_list=allowed_untracked_path_list,
            normalized_equalities=normalized_equalities,
        )

    if _lock_acquired:
        return verify_once()
    lock_path = release_lock_path(release_root)
    try:
        with FileLock(str(lock_path), timeout=lock_timeout_value):
            return verify_once()
    except Timeout as exc:
        raise ReleaseSecurityError(
            f"another release-security operation owns the lock: {lock_path}"
        ) from exc
    except (
        BundleInventoryError,
        ManifestAuthenticationError,
        SemanticInvariantError,
    ) as exc:
        raise ReleaseSecurityError(str(exc)) from exc
