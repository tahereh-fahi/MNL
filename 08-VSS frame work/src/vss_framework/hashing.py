"""Integrity helpers used by every source adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Integrity check failed for {path}: expected {expected_sha256}, got {actual}"
        )
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}

