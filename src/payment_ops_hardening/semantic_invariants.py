from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from payment_ops_hardening.bundle_inventory import (
    BundleInventoryError,
    normalize_relative_path,
    safe_relative_path,
)
from payment_ops_hardening.manifest_auth import canonical_json_bytes
from payment_ops_hardening.secure_io import StableFileError, read_stable_bytes
from payment_ops_hardening.strict_json import StrictJSONError, strict_json_loads

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_EQUALITY_FIELDS = {"name", "references"}
_REFERENCE_FIELDS = {"path", "pointer"}


class SemanticInvariantError(RuntimeError):
    """Raised when cross-artifact semantic invariants are malformed or violated."""


def _decode_pointer_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        char = token[index]
        if char != "~":
            result.append(char)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise SemanticInvariantError(
                f"invalid JSON pointer escape in token: {token!r}"
            )
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def normalize_json_pointer(pointer: object) -> str:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise SemanticInvariantError("JSON pointer must be a string beginning with '/'")
    for token in pointer[1:].split("/"):
        _decode_pointer_token(token)
    return pointer


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    current = document
    for token in (_decode_pointer_token(part) for part in pointer[1:].split("/")):
        if isinstance(current, dict):
            if token not in current:
                raise SemanticInvariantError(f"JSON pointer does not exist: {pointer}")
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-" or not token.isdigit():
                raise SemanticInvariantError(
                    f"invalid JSON array index in pointer: {pointer}"
                )
            index = int(token)
            if index >= len(current):
                raise SemanticInvariantError(
                    f"JSON pointer index is out of range: {pointer}"
                )
            current = current[index]
            continue
        raise SemanticInvariantError(
            f"JSON pointer traverses a scalar value: {pointer}"
        )
    return current


def normalize_json_value_equalities(
    equalities: Iterable[Mapping[str, object]] | None,
) -> list[dict]:
    if equalities is None:
        return []
    if isinstance(equalities, (str, bytes, Mapping)):
        raise SemanticInvariantError(
            "json_value_equalities must be an iterable of equality objects"
        )
    try:
        values = list(equalities)
    except TypeError as exc:
        raise SemanticInvariantError("json_value_equalities must be iterable") from exc

    normalized: list[dict] = []
    names: set[str] = set()
    for equality in values:
        if not isinstance(equality, Mapping):
            raise SemanticInvariantError("each JSON equality must be an object")
        unknown = sorted(set(equality) - _EQUALITY_FIELDS)
        missing = sorted(_EQUALITY_FIELDS - set(equality))
        if unknown or missing:
            raise SemanticInvariantError(
                f"invalid JSON equality fields; missing={missing}, unknown={unknown}"
            )
        name = equality["name"]
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise SemanticInvariantError(
                "JSON equality name must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
            )
        if name in names:
            raise SemanticInvariantError(f"duplicate JSON equality name: {name!r}")
        names.add(name)

        references = equality["references"]
        if isinstance(references, (str, bytes, Mapping)) or not isinstance(
            references, Iterable
        ):
            raise SemanticInvariantError("JSON equality references must be a list")
        reference_values = list(references)
        if len(reference_values) < 2:
            raise SemanticInvariantError(
                "JSON equality requires at least two references"
            )
        normalized_references: list[dict] = []
        seen_references: set[tuple[str, str]] = set()
        for reference in reference_values:
            if not isinstance(reference, Mapping):
                raise SemanticInvariantError(
                    "JSON equality reference must be an object"
                )
            unknown_ref = sorted(set(reference) - _REFERENCE_FIELDS)
            missing_ref = sorted(_REFERENCE_FIELDS - set(reference))
            if unknown_ref or missing_ref:
                raise SemanticInvariantError(
                    "invalid JSON equality reference fields; "
                    f"missing={missing_ref}, unknown={unknown_ref}"
                )
            try:
                path = normalize_relative_path(reference["path"])
            except BundleInventoryError as exc:
                raise SemanticInvariantError(str(exc)) from exc
            pointer = normalize_json_pointer(reference["pointer"])
            identity = (path, pointer)
            if identity in seen_references:
                raise SemanticInvariantError(
                    f"duplicate JSON equality reference: {path}{pointer}"
                )
            seen_references.add(identity)
            normalized_references.append({"path": path, "pointer": pointer})
        normalized.append({"name": name, "references": normalized_references})
    return normalized


def verify_json_value_equalities(
    root: str | Path,
    equalities: Iterable[Mapping[str, object]] | None,
    *,
    allowed_paths: Iterable[str | Path] | None = None,
) -> list[dict]:
    normalized = normalize_json_value_equalities(equalities)
    if not normalized:
        return []
    allowed: set[str] | None = None
    if allowed_paths is not None:
        if isinstance(allowed_paths, (str, bytes, Path)):
            raise SemanticInvariantError("allowed_paths must be an iterable of paths")
        try:
            allowed = {normalize_relative_path(path) for path in allowed_paths}
        except (TypeError, BundleInventoryError) as exc:
            raise SemanticInvariantError(str(exc)) from exc

    release_root = Path(root).resolve()
    documents: dict[str, Any] = {}
    evidence: list[dict] = []
    for equality in normalized:
        canonical_values: list[bytes] = []
        for reference in equality["references"]:
            relative = reference["path"]
            if allowed is not None and relative not in allowed:
                raise SemanticInvariantError(
                    f"JSON equality references a path outside policy: {relative}"
                )
            if relative not in documents:
                try:
                    path = safe_relative_path(release_root, relative)
                    payload = read_stable_bytes(path, reject_hardlinks=True)
                    documents[relative] = strict_json_loads(payload.decode("utf-8"))
                except (
                    BundleInventoryError,
                    StableFileError,
                    StrictJSONError,
                    UnicodeError,
                ) as exc:
                    raise SemanticInvariantError(
                        f"cannot read JSON equality source {relative}: {exc}"
                    ) from exc
            value = resolve_json_pointer(documents[relative], reference["pointer"])
            canonical_values.append(canonical_json_bytes(value))
        first = canonical_values[0]
        if any(value != first for value in canonical_values[1:]):
            refs = [f"{ref['path']}#{ref['pointer']}" for ref in equality["references"]]
            raise SemanticInvariantError(
                f"JSON equality {equality['name']!r} failed across references: {refs}"
            )
        evidence.append(
            {
                "name": equality["name"],
                "references": equality["references"],
                "value_sha256": hashlib.sha256(first).hexdigest(),
            }
        )
    return evidence
