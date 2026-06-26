from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CHUNK_SIZE = 1024 * 1024
_STABLE_FIELDS = (
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_ino",
    "st_dev",
    "st_nlink",
)


class StableFileError(RuntimeError):
    """Raised when a file cannot be read as one stable regular-file object."""


@dataclass(frozen=True)
class StableFileRead:
    before: os.stat_result
    after: os.stat_result
    sha256: str
    payload: bytes | None


def _same_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field, None) == getattr(right, field, None)
        for field in _STABLE_FIELDS
    )


def _identity_matches(path_stat: os.stat_result, fd_stat: os.stat_result) -> bool:
    identity_fields = ("st_dev", "st_ino")
    comparable = [
        field
        for field in identity_fields
        if getattr(path_stat, field, None) not in (None, 0)
        and getattr(fd_stat, field, None) not in (None, 0)
    ]
    return all(
        getattr(path_stat, field) == getattr(fd_stat, field) for field in comparable
    )


def read_stable_file(
    path: str | Path,
    *,
    capture_payload: bool,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    reject_hardlinks: bool = True,
) -> StableFileRead:
    """Read one stable regular file through a descriptor, rejecting path swaps.

    On POSIX, ``O_NOFOLLOW`` blocks a final-component symlink race. On platforms without
    that flag, the pre-open path metadata, opened descriptor identity, descriptor metadata,
    and post-read path metadata must all agree before any bytes are returned.
    """
    source = Path(path)
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise StableFileError("chunk_size must be a positive integer")
    try:
        before_path = source.stat(follow_symlinks=False)
    except OSError as exc:
        raise StableFileError(f"cannot inspect file before read: {source}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise StableFileError(f"file is not a regular file: {source}")
    if reject_hardlinks and before_path.st_nlink > 1:
        raise StableFileError(f"hard-linked files are not allowed: {source}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise StableFileError(f"cannot open stable regular file: {source}") from exc

    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture_payload else None
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise StableFileError(f"opened object is not a regular file: {source}")
        if reject_hardlinks and opened_before.st_nlink > 1:
            raise StableFileError(f"hard-linked files are not allowed: {source}")
        if not _identity_matches(before_path, opened_before):
            raise StableFileError(f"file path changed before descriptor open: {source}")

        while True:
            chunk = os.read(descriptor, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    except OSError as exc:
        raise StableFileError(f"cannot read stable regular file: {source}") from exc
    finally:
        os.close(descriptor)

    try:
        after_path = source.stat(follow_symlinks=False)
    except OSError as exc:
        raise StableFileError(f"file path disappeared after read: {source}") from exc
    if not stat.S_ISREG(after_path.st_mode):
        raise StableFileError(f"file path is no longer a regular file: {source}")
    if reject_hardlinks and after_path.st_nlink > 1:
        raise StableFileError(f"hard-linked files are not allowed: {source}")
    if not _same_metadata(opened_before, opened_after):
        raise StableFileError(f"file descriptor changed while being read: {source}")
    if not _same_metadata(before_path, after_path) or not _identity_matches(
        after_path, opened_after
    ):
        raise StableFileError(f"file path changed while being read: {source}")

    return StableFileRead(
        before=opened_before,
        after=opened_after,
        sha256=digest.hexdigest(),
        payload=b"".join(chunks) if chunks is not None else None,
    )


def read_stable_bytes(
    path: str | Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    reject_hardlinks: bool = True,
) -> bytes:
    result = read_stable_file(
        path,
        capture_payload=True,
        chunk_size=chunk_size,
        reject_hardlinks=reject_hardlinks,
    )
    assert result.payload is not None
    return result.payload
