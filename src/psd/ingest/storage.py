"""Object storage for trajectory bodies (TASK-011).

Spec Section 10.2 puts bodies in S3-compatible object storage and metadata in Postgres,
and Section 15.2 notes that a local filesystem is fine for the reproduction path. Both
implementations sit behind one small interface so the reproduction path needs no cloud
account and the platform path needs no code change.

Keys are partitioned `tenant/domain/date/` per component C-01, and writes are
content-addressed so redelivery of identical bytes is a no-op rather than a second copy.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    """Append-only blob storage."""

    def put(self, key: str, data: bytes) -> str:
        """Write `data` at `key` and return its URI. Idempotent on identical content."""
        ...

    def get(self, key: str) -> bytes | None: ...

    def exists(self, key: str) -> bool: ...


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def body_key(tenant_id: str, domain: str, content_hash: str, when: datetime | None = None) -> str:
    """`tenant/domain/date/hash.json`, per component C-01's partitioning note."""
    stamp = (when or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"{tenant_id}/{domain}/{stamp}/{content_hash}.json"


def quarantine_key(
    tenant_id: str, domain: str, content_hash: str, when: datetime | None = None
) -> str:
    """Spec Section 10.3: a blocking violation moves the record to a quarantine prefix.

    Separate prefix, not a flag on the body, so that "never enters a corpus" is a
    property of where the bytes live rather than of a query someone might forget to
    filter.
    """
    stamp = (when or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"_quarantine/{tenant_id}/{domain}/{stamp}/{content_hash}.json"


class FilesystemObjectStore:
    """Local-filesystem object store.

    Adequate for the reproduction path and for tests. Writes are content-addressed and
    write-once: re-putting identical bytes at the same key is a no-op, and putting
    DIFFERENT bytes at an existing key raises, because a content-addressed store whose
    contents can change under an address is not content-addressed.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        # Refuse a key that escapes the root. Keys derive from tenant and domain ids,
        # which are external input.
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError(f"key {key!r} escapes the object store root")
        return path

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(
                    f"{key!r} already exists with different content; object storage is write-once"
                )
            return f"file://{path}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"file://{path}"

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.exists() else None

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
