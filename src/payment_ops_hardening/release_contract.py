from __future__ import annotations

from pathlib import Path
from typing import Any

from payment_ops_hardening.bundle_inventory import (
    BundleInventoryError,
    normalize_contracts,
)
from payment_ops_hardening.manifest_auth import (
    ManifestAuthenticationError,
    normalized_key_id,
)
from payment_ops_hardening.secure_io import StableFileError, read_stable_bytes
from payment_ops_hardening.semantic_invariants import (
    SemanticInvariantError,
    normalize_json_value_equalities,
)
from payment_ops_hardening.strict_json import StrictJSONError, strict_json_loads

CONTRACT_SCHEMA_VERSION = "1.2"
_SUPPORTED_CONTRACT_SCHEMA_VERSIONS = {"1.1", CONTRACT_SCHEMA_VERSION}
_CONTRACT_FIELDS = {
    "schema_version",
    "require_signature",
    "required_paths",
    "content_contracts",
    "expected_key_id",
    "minimum_release_sequence",
    "allowed_release_states",
    "reject_untracked_files",
    "allowed_untracked_paths",
    "json_value_equalities",
}


class ReleaseContractError(RuntimeError):
    """Raised when the independent release policy file is malformed."""


def _string_list(value: object, field: str, *, required: bool = False) -> list[str]:
    if value is None:
        if required:
            raise ReleaseContractError(f"{field} is required")
        return []
    if not isinstance(value, list):
        raise ReleaseContractError(f"{field} must be a JSON list")
    if any(not isinstance(item, str) for item in value):
        raise ReleaseContractError(f"{field} must contain only strings")
    if any(not item for item in value) or len(value) != len(set(value)):
        raise ReleaseContractError(f"{field} must contain unique non-empty strings")
    if required and not value:
        raise ReleaseContractError(f"{field} must be non-empty")
    return list(value)


def load_release_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    is_junction = getattr(source, "is_junction", None)
    if source.is_symlink() or bool(is_junction and is_junction()):
        raise ReleaseContractError("release contract cannot be a symlink or junction")
    if not source.is_file():
        raise ReleaseContractError(f"release contract is not a regular file: {source}")
    try:
        if source.stat(follow_symlinks=False).st_nlink > 1:
            raise ReleaseContractError("release contract cannot be hard-linked")
    except OSError as exc:
        raise ReleaseContractError(
            f"cannot inspect release contract: {source}"
        ) from exc
    try:
        payload = read_stable_bytes(source, reject_hardlinks=True)
        value = strict_json_loads(payload.decode("utf-8"))
    except (StableFileError, StrictJSONError, UnicodeError) as exc:
        raise ReleaseContractError(
            f"invalid or unstable release contract file: {source}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReleaseContractError("release contract must contain a JSON object")
    unknown = sorted(set(value) - _CONTRACT_FIELDS)
    if unknown:
        raise ReleaseContractError(f"unknown release contract fields: {unknown}")
    observed_schema = value.get("schema_version")
    if observed_schema not in _SUPPORTED_CONTRACT_SCHEMA_VERSIONS:
        raise ReleaseContractError(
            f"unsupported release contract schema_version: {observed_schema!r}"
        )
    require_signature = value.get("require_signature")
    if not isinstance(require_signature, bool):
        raise ReleaseContractError("require_signature must be boolean")

    required_paths = _string_list(
        value.get("required_paths"), "required_paths", required=True
    )
    content_contracts = value.get("content_contracts", {})
    if not isinstance(content_contracts, dict):
        raise ReleaseContractError("content_contracts must be an object")
    try:
        normalized_contracts = normalize_contracts(content_contracts)
    except BundleInventoryError as exc:
        raise ReleaseContractError(str(exc)) from exc
    unknown_paths = sorted(set(normalized_contracts) - set(required_paths))
    if unknown_paths:
        raise ReleaseContractError(
            f"content_contracts reference paths outside required_paths: {unknown_paths}"
        )

    expected_key_id = value.get("expected_key_id")
    if require_signature and (
        not isinstance(expected_key_id, str) or not expected_key_id.strip()
    ):
        raise ReleaseContractError(
            "signed release contract requires a non-empty expected_key_id"
        )
    if expected_key_id is not None:
        try:
            expected_key_id = normalized_key_id(expected_key_id)
        except ManifestAuthenticationError as exc:
            raise ReleaseContractError(str(exc)) from exc

    minimum_release_sequence = value.get("minimum_release_sequence")
    if minimum_release_sequence is not None and (
        isinstance(minimum_release_sequence, bool)
        or not isinstance(minimum_release_sequence, int)
        or minimum_release_sequence < 0
    ):
        raise ReleaseContractError(
            "minimum_release_sequence must be a non-negative integer"
        )

    allowed_states = _string_list(
        value.get("allowed_release_states", ["PROMOTE"]),
        "allowed_release_states",
        required=True,
    )
    reject_untracked = value.get("reject_untracked_files", True)
    if not isinstance(reject_untracked, bool):
        raise ReleaseContractError("reject_untracked_files must be boolean")
    allowed_untracked = _string_list(
        value.get("allowed_untracked_paths", []),
        "allowed_untracked_paths",
    )
    try:
        json_value_equalities = normalize_json_value_equalities(
            value.get("json_value_equalities", [])
        )
    except SemanticInvariantError as exc:
        raise ReleaseContractError(str(exc)) from exc
    allowed_invariant_paths = set(required_paths) | {"release_manifest.json"}
    invariant_paths = {
        reference["path"]
        for equality in json_value_equalities
        for reference in equality["references"]
    }
    outside_invariant_paths = sorted(invariant_paths - allowed_invariant_paths)
    if outside_invariant_paths:
        raise ReleaseContractError(
            "json_value_equalities reference paths outside required_paths/control paths: "
            f"{outside_invariant_paths}"
        )
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "require_signature": require_signature,
        "required_paths": required_paths,
        "content_contracts": normalized_contracts,
        "expected_key_id": expected_key_id,
        "minimum_release_sequence": minimum_release_sequence,
        "allowed_release_states": allowed_states,
        "reject_untracked_files": reject_untracked,
        "allowed_untracked_paths": allowed_untracked,
        "json_value_equalities": json_value_equalities,
    }
