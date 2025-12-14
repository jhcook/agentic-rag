"""Canonical indexed text artifacts.

This module persists *only* the extracted, canonical text that was indexed.
Artifacts live under `cache/indexed/` (configurable via `RAG_INDEXED_DIR`).

Design constraints:
- No JSONL store.
- Artifacts must match exactly what is embedded (embed from bytes read from disk).
- Artifacts must be deleted alongside vector index deletions.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


INDEXED_DIR = os.getenv("RAG_INDEXED_DIR", "./cache/indexed")


@dataclass(frozen=True)
class IndexedArtifact:
    """Represents the on-disk canonical text artifact for a URI."""

    uri: str
    path: Path
    text_sha256: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def _sha256_uri(uri: str) -> str:
    return hashlib.sha256(uri.encode("utf-8", errors="strict")).hexdigest()


def artifact_path_for_uri(uri: str) -> Path:
    """Return the canonical artifact path for a URI."""

    base = Path(INDEXED_DIR)
    return base / f"{_sha256_uri(uri)}.txt"


def write_indexed_text(*, uri: str, text: str) -> IndexedArtifact:
    """Write canonical indexed text for a URI.

    Writes atomically (temp file then rename). Returns artifact metadata.
    """

    target = artifact_path_for_uri(uri)
    target.parent.mkdir(parents=True, exist_ok=True)

    text_sha256 = _sha256_text(text)

    # Atomic-ish write
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        tmp_path.replace(target)
    finally:
        try:
            if tmp_path.exists() and tmp_path != target:
                tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return IndexedArtifact(uri=uri, path=target, text_sha256=text_sha256)


def read_indexed_text(uri: str) -> Optional[str]:
    """Read canonical indexed text for a URI, or None if missing."""

    path = artifact_path_for_uri(uri)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_indexed_bytes(uri: str) -> Optional[bytes]:
    """Read canonical indexed text bytes for a URI, or None if missing."""

    path = artifact_path_for_uri(uri)
    if not path.exists():
        return None
    return path.read_bytes()


def delete_indexed_text(uri: str) -> bool:
    """Delete canonical indexed text for a URI (best-effort)."""

    path = artifact_path_for_uri(uri)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def clear_indexed_dir() -> int:
    """Delete all artifacts under INDEXED_DIR. Returns deleted file count."""

    base = Path(INDEXED_DIR)
    if not base.exists():
        return 0

    deleted = 0
    for child in base.glob("*.txt"):
        try:
            child.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted
