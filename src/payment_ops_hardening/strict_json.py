from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024


class StrictJSONError(ValueError):
    """Raised when JSON is ambiguous or outside the strict release format."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON number is not allowed: {value}")


def strict_json_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, RecursionError, MemoryError, ValueError) as exc:
        raise StrictJSONError(str(exc)) from exc


def read_strict_json(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> Any:
    source = Path(path)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise StrictJSONError("max_bytes must be a positive integer")
    try:
        size = source.stat().st_size
        if size > max_bytes:
            raise StrictJSONError(
                f"JSON file exceeds maximum size {max_bytes} bytes: {source}"
            )
        payload = source.read_bytes()
    except StrictJSONError:
        raise
    except OSError as exc:
        raise StrictJSONError(f"cannot read JSON file: {source}") from exc
    if len(payload) > max_bytes:
        raise StrictJSONError(
            f"JSON file exceeds maximum size {max_bytes} bytes: {source}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise StrictJSONError(f"JSON file is not valid UTF-8: {source}") from exc
    return strict_json_loads(text)
