from __future__ import annotations

import csv
import os
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from payment_ops_hardening.atomic_io import atomic_write_json
from payment_ops_hardening.secure_io import StableFileError, read_stable_file
from payment_ops_hardening.strict_json import StrictJSONError, read_strict_json

INVENTORY_FILE = "release_bundle_inventory.json"
INVENTORY_SCHEMA_VERSION = "2.1"
_INVENTORY_KEYS = {"schema_version", "files"}
_ENTRY_BASE_KEYS = {"path", "size_bytes", "sha256"}
_ENTRY_OPTIONAL_KEYS = {"content_contract", "observed_content"}
_CONTRACT_KEYS = {
    "binary": {"type", "min_size_bytes"},
    "csv": {
        "type",
        "min_size_bytes",
        "required_columns",
        "exact_columns",
        "min_rows",
        "max_rows",
    },
    "json": {
        "type",
        "min_size_bytes",
        "json_type",
        "required_keys",
        "min_items",
    },
}


class BundleInventoryError(RuntimeError):
    """Raised when a release-bundle inventory is incomplete or invalid."""


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _validate_release_root(root: str | Path, *, create: bool = False) -> Path:
    source = Path(root)
    if _is_link_like(source):
        raise BundleInventoryError(
            f"release root cannot be a symlink or junction: {source}"
        )
    if create:
        source.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise BundleInventoryError(f"release root does not exist: {source}")
    if not source.is_dir():
        raise BundleInventoryError(f"release root is not a directory: {source}")
    return source.resolve()


def _reject_hardlink(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        links = path.stat(follow_symlinks=False).st_nlink
    except (OSError, TypeError):
        return
    if links > 1:
        raise BundleInventoryError(
            f"hard-linked release artifacts are not allowed: {path}"
        )


def _stable_file_metadata(path: Path) -> tuple[os.stat_result, str, os.stat_result]:
    """Hash a stable regular file through a no-follow descriptor when supported."""
    try:
        result = read_stable_file(path, capture_payload=False, reject_hardlinks=True)
    except StableFileError as exc:
        raise BundleInventoryError(str(exc)) from exc
    return result.before, result.sha256, result.after


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # chunk_size retained for API compatibility; stable hashing intentionally uses 1 MiB.
    del chunk_size
    return _stable_file_metadata(Path(path))[1]


def _normalize_relative_text(value: str | Path) -> str:
    if isinstance(value, Path):
        if value.is_absolute() or value.drive or ".." in value.parts:
            raise BundleInventoryError(f"unsafe release path: {value}")
        raw = value.as_posix()
    elif isinstance(value, str):
        raw = value
        if "\\" in raw:
            raise BundleInventoryError(f"non-portable release path separator: {value}")
    else:
        raise BundleInventoryError(
            f"release path must be a string or pathlib.Path, got {type(value).__name__}"
        )
    if not raw or raw == "." or "\x00" in raw:
        raise BundleInventoryError(f"empty or invalid release path: {value!r}")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise BundleInventoryError(f"unsafe release path: {value}")
    if any(":" in part for part in relative.parts):
        raise BundleInventoryError(f"non-portable release path component: {value}")
    text = relative.as_posix()
    if not text or text == ".":
        raise BundleInventoryError(f"empty release path: {value!r}")
    return text


def normalize_relative_path(value: str | Path) -> str:
    """Public fail-closed normalization used by external policy modules."""
    return _normalize_relative_text(value)


def safe_relative_path(root: Path, value: str | Path) -> Path:
    relative_text = _normalize_relative_text(value)
    root_resolved = _validate_release_root(root)
    current = root_resolved
    for part in PurePosixPath(relative_text).parts:
        current = current / part
        if _is_link_like(current):
            raise BundleInventoryError(
                f"symlink or junction release artifacts are not allowed: {value}"
            )
    resolved = current.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise BundleInventoryError(f"release path escapes root: {value}") from exc
    _reject_hardlink(resolved)
    return resolved


def _csv_observation(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                columns = next(reader)
            except StopIteration:
                return {"columns": [], "data_rows": 0}
            if not columns or any(column == "" for column in columns):
                raise BundleInventoryError(f"CSV header contains empty columns: {path}")
            if len(columns) != len(set(columns)):
                raise BundleInventoryError(f"duplicate CSV columns: {path}")
            data_rows = 0
            for row_number, row in enumerate(reader, start=2):
                if not row or not any(cell != "" for cell in row):
                    continue
                if len(row) != len(columns):
                    raise BundleInventoryError(
                        f"CSV row width mismatch for {path} at row {row_number}"
                    )
                data_rows += 1
            return {"columns": columns, "data_rows": data_rows}
    except BundleInventoryError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BundleInventoryError(f"invalid CSV artifact: {path}") from exc


def _json_observation(path: Path) -> dict:
    try:
        value = read_strict_json(path)
    except StrictJSONError as exc:
        raise BundleInventoryError(f"invalid JSON artifact: {path}: {exc}") from exc
    if isinstance(value, dict):
        return {"json_type": "object", "keys": sorted(value), "items": len(value)}
    if isinstance(value, list):
        return {"json_type": "array", "items": len(value)}
    return {"json_type": type(value).__name__, "items": None}


def _sequence_of_strings(value: object, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise BundleInventoryError(f"{field} must be a JSON list of strings")
    if any(not isinstance(item, str) for item in value):
        raise BundleInventoryError(f"{field} must contain only strings")
    if len(value) != len(set(value)) or any(not item for item in value):
        raise BundleInventoryError(f"{field} must contain unique non-empty strings")
    return list(value)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BundleInventoryError(f"{field} must be a non-negative integer")
    return value


def _reject_unknown_keys(
    value: Mapping[str, object], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BundleInventoryError(f"unknown {label} fields: {unknown}")


def _normalize_contract(contract: Mapping[str, object]) -> dict:
    if not isinstance(contract, Mapping):
        raise BundleInventoryError("content contract must be an object")
    kind_value = contract.get("type")
    if not isinstance(kind_value, str):
        raise BundleInventoryError("content contract type must be a string")
    kind = kind_value.lower()
    if kind not in _CONTRACT_KEYS:
        raise BundleInventoryError(f"unsupported content contract type: {kind!r}")
    _reject_unknown_keys(contract, _CONTRACT_KEYS[kind], f"{kind} content contract")
    normalized: dict[str, object] = {"type": kind}
    normalized["min_size_bytes"] = _nonnegative_int(
        contract.get("min_size_bytes", 0), "min_size_bytes"
    )

    if kind == "csv":
        required = _sequence_of_strings(
            contract.get("required_columns", []), "required_columns"
        )
        exact = contract.get("exact_columns")
        if exact is not None:
            exact_columns = _sequence_of_strings(exact, "exact_columns")
            if required and not set(required).issubset(exact_columns):
                raise BundleInventoryError(
                    "required_columns must be included in exact_columns"
                )
            normalized["exact_columns"] = exact_columns
        normalized["required_columns"] = required
        min_rows = _nonnegative_int(contract.get("min_rows", 0), "min_rows")
        normalized["min_rows"] = min_rows
        max_rows = contract.get("max_rows")
        if max_rows is not None:
            max_rows_int = _nonnegative_int(max_rows, "max_rows")
            if max_rows_int < min_rows:
                raise BundleInventoryError("max_rows cannot be smaller than min_rows")
            normalized["max_rows"] = max_rows_int

    if kind == "json":
        expected_type = contract.get("json_type", "object")
        if not isinstance(expected_type, str) or expected_type not in {
            "object",
            "array",
        }:
            raise BundleInventoryError("json_type must be 'object' or 'array'")
        normalized["json_type"] = expected_type
        required_keys = _sequence_of_strings(
            contract.get("required_keys", []), "required_keys"
        )
        if expected_type != "object" and required_keys:
            raise BundleInventoryError("required_keys are valid only for JSON objects")
        normalized["required_keys"] = required_keys
        normalized["min_items"] = _nonnegative_int(
            contract.get("min_items", 0), "min_items"
        )
    return normalized


def normalize_contracts(
    contracts: Mapping[str | Path, Mapping[str, object]] | None,
) -> dict[str, dict]:
    if contracts is None:
        return {}
    if not isinstance(contracts, Mapping):
        raise BundleInventoryError("content_contracts must be an object")
    normalized: dict[str, dict] = {}
    for path, contract in contracts.items():
        relative_text = _normalize_relative_text(path)
        if relative_text in normalized:
            raise BundleInventoryError(
                f"duplicate content contract path: {relative_text}"
            )
        normalized[relative_text] = _normalize_contract(contract)
    return normalized


def _validate_contract(path: Path, contract: dict) -> dict:
    size = path.stat(follow_symlinks=False).st_size
    if size < contract.get("min_size_bytes", 0):
        raise BundleInventoryError(f"artifact smaller than contract minimum: {path}")
    kind = contract["type"]
    if kind == "binary":
        return {"size_bytes": size}
    if kind == "csv":
        observed = _csv_observation(path)
        columns = observed["columns"]
        missing = sorted(set(contract.get("required_columns", [])) - set(columns))
        if missing:
            raise BundleInventoryError(
                f"CSV required columns missing for {path}: {missing}"
            )
        exact = contract.get("exact_columns")
        if exact is not None and columns != exact:
            raise BundleInventoryError(f"CSV exact column contract mismatch: {path}")
        rows = observed["data_rows"]
        if rows < contract.get("min_rows", 0):
            raise BundleInventoryError(
                f"CSV has fewer rows than contract minimum: {path}"
            )
        if "max_rows" in contract and rows > contract["max_rows"]:
            raise BundleInventoryError(f"CSV exceeds contract maximum rows: {path}")
        return observed
    observed = _json_observation(path)
    if observed["json_type"] != contract["json_type"]:
        raise BundleInventoryError(f"JSON type contract mismatch: {path}")
    if observed["json_type"] == "object":
        missing_keys = sorted(
            set(contract.get("required_keys", [])) - set(observed["keys"])
        )
        if missing_keys:
            raise BundleInventoryError(
                f"JSON required keys missing for {path}: {missing_keys}"
            )
    if (observed["items"] or 0) < contract.get("min_items", 0):
        raise BundleInventoryError(
            f"JSON has fewer items than contract minimum: {path}"
        )
    return observed


def _materialize_path_iterable(
    values: Iterable[str | Path], field: str
) -> list[str | Path]:
    if isinstance(values, (str, bytes, Path)):
        raise BundleInventoryError(
            f"{field} must be an iterable of paths, not a scalar"
        )
    try:
        result = list(values)
    except TypeError as exc:
        raise BundleInventoryError(f"{field} must be an iterable of paths") from exc
    return result


def build_inventory(
    root: str | Path,
    required_paths: Iterable[str | Path],
    *,
    content_contracts: Mapping[str | Path, Mapping[str, object]] | None = None,
    csv_contract_paths: Iterable[str | Path] = (),
) -> dict:
    release_root = _validate_release_root(root)
    required = _materialize_path_iterable(required_paths, "required_paths")
    normalized_required = [_normalize_relative_text(item) for item in required]
    if len(normalized_required) != len(set(normalized_required)):
        raise BundleInventoryError("required_paths must be unique after normalization")
    normalized_paths = sorted(normalized_required)
    if not normalized_paths:
        raise BundleInventoryError("release inventory cannot be empty")
    contracts = normalize_contracts(content_contracts)
    csv_values = _materialize_path_iterable(csv_contract_paths, "csv_contract_paths")
    normalized_csv = [_normalize_relative_text(item) for item in csv_values]
    if len(normalized_csv) != len(set(normalized_csv)):
        raise BundleInventoryError(
            "csv_contract_paths must be unique after normalization"
        )
    for relative_text in normalized_csv:
        contracts.setdefault(relative_text, _normalize_contract({"type": "csv"}))
    unknown_contracts = sorted(set(contracts) - set(normalized_paths))
    if unknown_contracts:
        raise BundleInventoryError(
            f"content contracts reference non-inventoried paths: {unknown_contracts}"
        )

    files: list[dict] = []
    for relative_text in normalized_paths:
        path = safe_relative_path(release_root, relative_text)
        if not path.is_file():
            raise BundleInventoryError(
                f"required release artifact is missing: {relative_text}"
            )
        before, digest, _ = _stable_file_metadata(path)
        entry = {
            "path": relative_text,
            "size_bytes": before.st_size,
            "sha256": digest,
        }
        contract = contracts.get(relative_text)
        if contract is not None:
            entry["content_contract"] = contract
            entry["observed_content"] = _validate_contract(path, contract)
            # Content parsing may take time; prove the bytes remained stable through it.
            after_parse, digest_after_parse, _ = _stable_file_metadata(path)
            if before.st_size != after_parse.st_size or digest != digest_after_parse:
                raise BundleInventoryError(
                    f"release artifact changed during content validation: {relative_text}"
                )
        files.append(entry)
    return {"schema_version": INVENTORY_SCHEMA_VERSION, "files": files}


def write_inventory(root: str | Path, inventory: dict) -> Path:
    path = _validate_release_root(root) / INVENTORY_FILE
    atomic_write_json(path, inventory)
    return path


def verify_inventory(
    root: str | Path,
    inventory: dict,
    *,
    expected_paths: Iterable[str | Path] | None = None,
    expected_contracts: Mapping[str | Path, Mapping[str, object]] | None = None,
) -> dict:
    release_root = _validate_release_root(root)
    if not isinstance(inventory, dict):
        raise BundleInventoryError("release inventory must be an object")
    _reject_unknown_keys(inventory, _INVENTORY_KEYS, "inventory")
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise BundleInventoryError(
            f"unsupported inventory schema_version: {inventory.get('schema_version')!r}"
        )
    entries = inventory.get("files")
    if not isinstance(entries, list) or not entries:
        raise BundleInventoryError("release inventory has no file entries")
    expected = None
    if expected_paths is not None:
        expected_list = _materialize_path_iterable(expected_paths, "expected_paths")
        normalized_expected = [_normalize_relative_text(item) for item in expected_list]
        if len(normalized_expected) != len(set(normalized_expected)):
            raise BundleInventoryError(
                "expected_paths must be unique after normalization"
            )
        expected = set(normalized_expected)
    normalized_expected_contracts = normalize_contracts(expected_contracts)

    observed_paths: set[str] = set()
    embedded_contracts: dict[str, dict] = {}
    verified_artifacts: dict[str, dict] = {}
    verified_contracts = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise BundleInventoryError("release inventory contains a non-object entry")
        allowed_keys = _ENTRY_BASE_KEYS | _ENTRY_OPTIONAL_KEYS
        _reject_unknown_keys(entry, allowed_keys, "inventory entry")
        missing_base = sorted(_ENTRY_BASE_KEYS - set(entry))
        if missing_base:
            raise BundleInventoryError(
                f"inventory entry missing fields: {missing_base}"
            )
        if ("content_contract" in entry) != ("observed_content" in entry):
            raise BundleInventoryError(
                "content_contract and observed_content must appear together"
            )
        relative_text = _normalize_relative_text(entry["path"])
        if relative_text in observed_paths:
            raise BundleInventoryError(f"duplicate inventory path: {relative_text!r}")
        observed_paths.add(relative_text)
        if isinstance(entry["size_bytes"], bool) or not isinstance(
            entry["size_bytes"], int
        ):
            raise BundleInventoryError(f"invalid inventory size for {relative_text}")
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise BundleInventoryError(
                f"invalid inventory checksum for {relative_text}"
            )
        path = safe_relative_path(release_root, relative_text)
        if not path.is_file():
            raise BundleInventoryError(
                f"inventoried artifact is missing: {relative_text}"
            )
        before, actual_digest, _ = _stable_file_metadata(path)
        if before.st_size != entry["size_bytes"]:
            raise BundleInventoryError(f"size mismatch: {relative_text}")
        if actual_digest != digest:
            raise BundleInventoryError(f"checksum mismatch: {relative_text}")
        verified_artifacts[relative_text] = {
            "size_bytes": entry["size_bytes"],
            "sha256": digest,
        }
        if "content_contract" in entry:
            contract = _normalize_contract(entry["content_contract"])
            embedded_contracts[relative_text] = contract
            actual_observation = _validate_contract(path, contract)
            if actual_observation != entry["observed_content"]:
                raise BundleInventoryError(
                    f"content observation mismatch: {relative_text}"
                )
            _, digest_after_parse, _ = _stable_file_metadata(path)
            if digest_after_parse != digest:
                raise BundleInventoryError(
                    f"artifact changed during content verification: {relative_text}"
                )
            verified_contracts += 1

    if expected is not None:
        missing = sorted(expected - observed_paths)
        unexpected = sorted(observed_paths - expected)
        if missing or unexpected:
            raise BundleInventoryError(
                f"inventory path-set mismatch; missing={missing}, unexpected={unexpected}"
            )
    if (
        normalized_expected_contracts
        and embedded_contracts != normalized_expected_contracts
    ):
        missing = sorted(set(normalized_expected_contracts) - set(embedded_contracts))
        unexpected = sorted(
            set(embedded_contracts) - set(normalized_expected_contracts)
        )
        changed = sorted(
            path
            for path in set(embedded_contracts) & set(normalized_expected_contracts)
            if embedded_contracts[path] != normalized_expected_contracts[path]
        )
        raise BundleInventoryError(
            "content contract-set mismatch; "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    return {
        "verified_files": len(entries),
        "verified_content_contracts": verified_contracts,
        "verified_csv_contracts": sum(
            contract.get("type") == "csv" for contract in embedded_contracts.values()
        ),
        "artifacts": verified_artifacts,
        "paths": sorted(observed_paths),
    }


def list_release_files(root: str | Path) -> set[str]:
    """Return regular files under root and reject symlinks, junctions, and hard links."""
    release_root = _validate_release_root(root)
    discovered: set[str] = set()
    for path in release_root.rglob("*"):
        if _is_link_like(path):
            raise BundleInventoryError(
                f"symlink or junction found in release bundle: {path}"
            )
        if path.is_file():
            _reject_hardlink(path)
            discovered.add(path.relative_to(release_root).as_posix())
    return discovered


def verify_no_untracked_files(
    root: str | Path,
    inventoried_paths: Iterable[str | Path],
    *,
    allowed_untracked_paths: Iterable[str | Path] = (),
) -> dict:
    inventoried_values = _materialize_path_iterable(
        inventoried_paths, "inventoried_paths"
    )
    allowed_values = _materialize_path_iterable(
        allowed_untracked_paths, "allowed_untracked_paths"
    )
    inventoried = {_normalize_relative_text(item) for item in inventoried_values}
    allowed = {_normalize_relative_text(item) for item in allowed_values}
    actual = list_release_files(root)
    untracked = sorted(actual - inventoried - allowed)
    missing = sorted(inventoried - actual)
    if missing or untracked:
        raise BundleInventoryError(
            f"release file-set mismatch; missing={missing}, untracked={untracked}"
        )
    return {
        "actual_files": len(actual),
        "inventoried_files": len(inventoried),
        "allowed_untracked_files": sorted(actual & allowed),
    }
