"""Isolated, integrity-checked diagnostic stages. Prepared by Tahereh Fahi."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .hashing import sha256_file
from .io import write_json
from .resources import PACKAGE_ROOT


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes(names: list[str]) -> dict[str, str]:
    # Explicit dependencies keep unrelated analytical changes out of this cache.
    names = sorted(set(names + ["validation_cache.py", "hashing.py", "io.py", "resources.py"]))
    return {name: sha256_file(PACKAGE_ROOT / name) for name in names}


def verified_receipt(directory: Path, identity: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads((directory / "stage_receipt.json").read_text())
        if not isinstance(receipt, dict):
            return None
        if receipt.get("identity") != identity or receipt.get("execution_status") != "complete":
            return None
        outputs = receipt.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            return None
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                  if p.is_file() and p != directory / "stage_receipt.json"}
        if actual != set(outputs):
            return None
        for name, expected in outputs.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected:
                return None
        return receipt
    except (OSError, ValueError, TypeError, KeyError):
        return None


def run_stage(root: Path, name: str, inputs: dict[str, Any],
              action: Callable[[Path], None], *, resume: bool = True) -> dict[str, Any]:
    identity = fingerprint(inputs)
    attempt = 0
    while True:
        directory = root / (f"{name}_{identity[:16]}" + (f"_attempt{attempt}" if attempt else ""))
        if not directory.exists():
            break
        receipt = verified_receipt(directory, identity) if resume else None
        if receipt:
            print(json.dumps({"stage": name, "execution_status": "verified_cache"}), flush=True)
            return {**receipt, "directory": directory.name, "reused": True}
        attempt += 1
    directory.mkdir(parents=True)
    print(json.dumps({"stage": name, "execution_status": "running"}), flush=True)
    started = time.monotonic()
    action(directory)
    outputs = {p.relative_to(directory).as_posix(): sha256_file(p)
               for p in sorted(directory.rglob("*")) if p.is_file()}
    if not outputs:
        raise RuntimeError(f"Diagnostic stage {name} produced no artifacts")
    receipt = {"prepared_by": "Tahereh Fahi", "identity": identity, "inputs": inputs,
               "execution_status": "complete", "validation_status": "not_validated",
               "publication_ready": False, "execution_seconds": time.monotonic() - started,
               "outputs": outputs}
    write_json(directory / "stage_receipt.json", receipt)
    return {**receipt, "directory": directory.name, "reused": False}
