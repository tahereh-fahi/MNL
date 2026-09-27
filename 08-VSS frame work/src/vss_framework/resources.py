"""Resolve framework-owned code/assets independently of the workspace layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
MODULES = {"gem_xp": "gems", "inventory": "inventory", "weapons": "weapons", "hud_clock": "hud"}


def asset_path(name: str) -> Path:
    return PACKAGE_ROOT / "assets" / name


def detector_path(name: str) -> Path:
    return PACKAGE_ROOT / "detectors" / (MODULES[name] + ".py")


def resolve_path(value: str | Path, workspace_root: Path) -> Path:
    text = str(value)
    if text.startswith("framework:"):
        path = (PACKAGE_ROOT / text.removeprefix("framework:")).resolve()
        if not path.is_relative_to(PACKAGE_ROOT):
            raise ValueError("Framework resource escapes package")
        return path
    if text.startswith("VSS frame work/"):
        text = "08-VSS frame work/" + text.removeprefix("VSS frame work/")
    return (workspace_root / text).resolve()


def source_reference(path: Path, workspace_root: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(PACKAGE_ROOT):
        return "framework:" + path.relative_to(PACKAGE_ROOT).as_posix()
    if path.is_relative_to(workspace_root.resolve()):
        return path.relative_to(workspace_root.resolve()).as_posix()
    return path.name


def worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    # The source checkout works without installation; installed wheels use this
    # same package root. No sibling project is added to the import path.
    env["PYTHONPATH"] = str(PACKAGE_ROOT.parent)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def module_command(command: list[str], detector: str) -> list[str]:
    return [command[0], "-m", "vss_framework.detectors." + MODULES[detector], *command[2:]]


def load_runtime_config(path: Path) -> dict[str, Any]:
    """Bind historical detector declarations to their internal implementations.

    Historical source-artifact specifications remain available to the explicit
    bootstrap importer. Fresh runs generate their own intermediate artifacts.
    """
    config = json.loads(path.read_text(encoding="utf-8"))
    detectors = config.setdefault("detectors", {})
    defaults = {
        "gem_xp": {"script": "framework:detectors/gems.py", "template_dir": "framework:assets/gems"},
        "inventory": {"script": "framework:detectors/inventory.py", "weapon_icon_dir": "framework:assets/weapon_icons", "passive_icon_dir": "framework:assets/passive_icons", "weapon_manifest": "framework:assets/manifests/wiki_weapon_manifest.csv", "passive_manifest": "framework:assets/manifests/wiki_passive_item_manifest.csv"},
        "hud_clock": {"script": "framework:detectors/hud.py", "template_paths": ["framework:assets/hud/skull_icon.jpg", "framework:assets/hud/skull_anchor.jpg"]},
    }
    legacy_prefixes = ("01_kill_counter_and_time_stamp/", "02_gems/", "02_blue_gems/", "03_weapons/")
    for name, values in defaults.items():
        if name not in detectors:
            continue
        detector = detectors[name]
        for key, default in values.items():
            old = detector.get(key)
            if old is None or (isinstance(old, str) and old.startswith(legacy_prefixes)) or (isinstance(old, list) and all(str(x).startswith(legacy_prefixes) for x in old)):
                detector[key] = default
        if detector["script"] != values["script"]:
            raise ValueError(f"Unsupported detector script for {name}; use the framework implementation")
        detector.pop("pipeline_module_dir", None)
    return config
