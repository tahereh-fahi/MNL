"""Integrity controls shared by health producers and their export."""
from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .runtime import distribution_version
from .resources import PACKAGE_ROOT


def implementation_receipt() -> dict[str, Any]:
    """Fingerprint transitive package code, not just the calling module."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "opencv-python", "pandas", "scipy", "torch", "torchvision", "easyocr"):
        try:
            versions[name] = distribution_version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {"code_sha256": {
        p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p)
        for p in sorted(PACKAGE_ROOT.rglob("*.py")) if "__pycache__" not in p.parts
    }, "runtime": versions, "reuse_allowed": False}


def verified_outputs(directory: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    directory = directory.resolve()
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    paths = {}
    for key, record in manifest.get("outputs", {}).items():
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or unsafe health upstream output: {key}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Upstream integrity mismatch: {key}")
        paths[key] = path
    if not paths:
        raise ValueError("Upstream manifest has no verifiable outputs")
    return manifest, paths


def verify_inventory(path: Path) -> dict[str, Any]:
    """Require an intact producer manifest; never infer origin from a filename."""
    manifest, outputs = verified_outputs(path.parent)
    if outputs.get("canonical_inventory_events") != path.resolve():
        raise ValueError("Inventory must be its producer's canonical_inventory_events output")
    return manifest
