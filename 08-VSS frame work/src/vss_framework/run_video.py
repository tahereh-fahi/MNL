"""Execute framework-owned detectors from video, with verified stage caching.

Prepared by Tahereh Fahi. Saved detector results are optional accelerators;
the execution graph can generate every intermediate input from the video.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .hashing import sha256_file
from . import __version__
from .runtime import distribution_version
from .io import write_json
from .resources import PACKAGE_ROOT, asset_path, load_runtime_config, resolve_path


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_fingerprint() -> str:
    files = sorted(p for p in PACKAGE_ROOT.rglob("*") if p.is_file() and (p.suffix == ".py" or "assets" in p.parts))
    return fingerprint({p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p) for p in files if "__pycache__" not in p.parts})


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "opencv-python", "easyocr", "torch", "torchvision"):
        try:
            versions[name] = distribution_version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def resource_fingerprints(config: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    result = {}
    for detector_name, detector in config["detectors"].items():
        for key in ("template_dir", "weapon_icon_dir", "passive_icon_dir", "weapon_manifest", "passive_manifest", "template_paths"):
            if key not in detector:
                continue
            values = detector[key] if isinstance(detector[key], list) else [detector[key]]
            for index, value in enumerate(values):
                path = resolve_path(value, workspace_root)
                if not path.exists():
                    raise FileNotFoundError(f"Missing {detector_name} resource: {key}")
                files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
                result[f"{detector_name}.{key}.{index}"] = {p.relative_to(path).as_posix() if path.is_dir() else p.name: sha256_file(p) for p in files}
    return result


def verified_stage(directory: Path, identity: str) -> bool:
    receipt_path = directory / "stage_receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["identity"] != identity or receipt["status"] != "complete" or not receipt["outputs"]:
            return False
        for name, expected in receipt["outputs"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256_file(path) != expected:
                return False
        return True
    except (KeyError, ValueError, OSError):
        return False


def prepare_config(path: Path, workspace_root: Path, video_override: Path | None = None) -> dict[str, Any]:
    config = load_runtime_config(path)
    if config.get("policies", {}).get("use_human_coded_ground_truth"):
        raise ValueError("Video extraction cannot use Human-coded ground truth")
    if config.get("policies", {}).get("multi_run_video"):
        raise ValueError("This recording contains multiple game runs; use its separate run configurations so inventory and levels reset at each run boundary")
    video = video_override.resolve() if video_override else resolve_path(config["dataset"]["video"]["path"], workspace_root)
    if not video.is_file():
        raise FileNotFoundError(f"Source video is missing: {video.name}")
    observed_sha = sha256_file(video)
    if observed_sha != config["dataset"]["video"]["sha256"]:
        raise ValueError(f"Video hash mismatch for {video.name}")
    from .video import OpenCVVideoReader
    with OpenCVVideoReader(video, config["dataset"]["video_asset_id"]) as reader:
        actual = reader.metadata
    # A renamed source file is accepted only through an explicit override and
    # an exact video hash match; video identity never comes from its filename.
    config["dataset"]["video"]["path"] = str(video)
    config["dataset"]["duration_ms"] = actual.duration_ms
    config["dataset"]["fps"] = actual.fps
    detectors = config.setdefault("detectors", {})
    gem = detectors.setdefault("gem_xp", {})
    gem.setdefault("template_profile", "legacy")
    if "initial_level" not in gem:
        raise ValueError("Fresh XP/inventory extraction requires detectors.gem_xp.initial_level; telemetry-only configurations do not establish a starting game level")
    gem["inventory_events"] = None
    inventory = detectors.setdefault("inventory", {})
    inventory.setdefault("sample_fps", 10.0)
    inventory.setdefault("video_key", config["dataset"]["video_asset_id"])
    inventory.pop("xp_events", None)
    inventory.pop("weapon_timeline", None)
    hud = detectors.setdefault("hud_clock", {})
    for key, value in {"sample_offsets_seconds": [.3, .5, .7], "min_ocr_confidence": 0., "min_timer_observed_rate": .5, "min_kill_observed_rate": .4, "initial_kill_state": 0, "evidence_every_seconds": 60, "max_evidence_frames": 250}.items():
        hud.setdefault(key, value)
    config["sources"] = {}
    return config


def run_video(*, config_path: Path, workspace_root: Path, output_dir: Path,
              video_override: Path | None = None, max_seconds: float | None = None,
              resume: bool = True, stages: str = "all") -> dict[str, Any]:
    started = time.monotonic()
    if stages not in {"all", "inventory-gems"}:
        raise ValueError("stages must be all or inventory-gems")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be a positive finite number")
    config = prepare_config(config_path, workspace_root, video_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "runtime_config.json"
    write_json(runtime_path, config)
    config = load_runtime_config(runtime_path)
    write_json(runtime_path, config)
    identity = {"video_sha256": config["dataset"]["video"]["sha256"], "implementation": implementation_fingerprint(),
                "dependencies": dependency_versions(), "resources": resource_fingerprints(config, workspace_root),
                "dataset": {k: v for k, v in config["dataset"].items() if k != "video"},
                "settings": json.loads(json.dumps(config["detectors"])), "max_seconds": max_seconds}
    records: dict[str, Any] = {}

    def stage(name: str, action: Callable[[Path], Any], dependencies: tuple[str, ...] = ()) -> Path:
        stage_id = fingerprint({"run": identity, "stage": name, "dependencies": {key: records[key]["outputs"] for key in dependencies}})
        base = output_dir / (name + "_" + stage_id[:16])
        directory = base
        attempt = 0
        while directory.exists():
            if resume and name not in {"health", "health_attribution"} and verified_stage(directory, stage_id):
                receipt = json.loads((directory / "stage_receipt.json").read_text())
                records[name] = {**receipt, "directory": directory.name, "reused": True}
                print(json.dumps({"stage": name, "status": "verified_cache"}), flush=True)
                return directory
            attempt += 1
            directory = base.with_name(base.name + f"_attempt{attempt}")
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        before = time.monotonic()
        try:
            result = action(directory)
        except Exception as exc:
            write_json(output_dir / "progress.json", {"prepared_by": "Tahereh Fahi", "stages": records,
                "status": "failed", "failed_stage": name, "error_type": type(exc).__name__})
            raise
        outputs = {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}
        if not outputs:
            raise RuntimeError(f"Stage {name} produced no outputs")
        receipt = {"prepared_by": "Tahereh Fahi", "status": "complete", "identity": stage_id, "outputs": outputs,
                   "quality_status": result.get("worker_qc_status", "not_applicable") if isinstance(result, dict) else "not_applicable",
                   "execution_seconds": time.monotonic() - before}
        write_json(directory / "stage_receipt.json", receipt)
        records[name] = {**receipt, "directory": directory.name, "reused": False}
        write_json(output_dir / "progress.json", {"prepared_by": "Tahereh Fahi", "stages": records, "status": "running"})
        return directory

    from .detectors.legacy_gem_xp import run_legacy_gem_xp_detector
    from .detectors.legacy_inventory import run_legacy_inventory_detector
    from .detectors.weapons import record_video_weapons
    common = {"workspace_root": workspace_root, "config_path": runtime_path}
    initial = stage("initial_gems", lambda out: run_legacy_gem_xp_detector(**common, output_dir=out, max_seconds=max_seconds))
    xp_paths = list((initial / "worker").glob("*xp_ab_events.csv"))
    if len(xp_paths) != 1:
        raise RuntimeError("Initial XP extraction must produce exactly one event file")
    video_path = Path(config["dataset"]["video"]["path"])
    weapons = stage("weapons", lambda out: record_video_weapons(video_path, out,
        resolve_path(config["detectors"]["inventory"]["weapon_icon_dir"], workspace_root),
        resolve_path(config["detectors"]["inventory"]["weapon_manifest"], workspace_root),
        end_second=max_seconds, debug_every_seconds=None))
    config["detectors"]["inventory"]["xp_events"] = str(xp_paths[0])
    config["detectors"]["inventory"]["weapon_timeline"] = str(weapons / "weapon_timeline.csv")
    write_json(runtime_path, config)
    base_inventory = stage("inventory", lambda out: run_legacy_inventory_detector(**common, output_dir=out, max_seconds=max_seconds), ("initial_gems", "weapons"))
    inventory_csv = base_inventory / "worker/inventory_events.csv"
    event_files = [base_inventory / "canonical_inventory_events.jsonl"]

    if stages == "all":
        from .detectors.legacy_hud import run_legacy_hud_detector
        from .telemetry_scan import scan_video4_telemetry
        from .chest_lifecycle import scan_video4_chests
        from .chest_rewards import identify_chest_rewards
        from .gold_fever import scan_video4_gold_fever
        from .status_events import scan_video4_status_events
        from .menu_actions import scan_video4_menu_actions
        from .inventory_reconciliation import reconcile_inventory_with_chest_rewards
        stage("hud", lambda out: run_legacy_hud_detector(**common, output_dir=out, max_seconds=None if max_seconds is None else math.ceil(max_seconds)))
        stage("telemetry", lambda out: scan_video4_telemetry(**common, output_dir=out, start_second=0., max_seconds=max_seconds, sample_fps=2.))
        chests = stage("chests", lambda out: scan_video4_chests(**common, output_dir=out, sample_fps=2., end_second=max_seconds))
        rewards = stage("chest_rewards", lambda out: identify_chest_rewards(**common, output_dir=out, chest_events_path=chests / "chest_events.jsonl", automated_inventory_path=inventory_csv), ("chests", "inventory"))
        reconciled = stage("inventory_reconciled", lambda out: reconcile_inventory_with_chest_rewards(
            base_inventory_path=inventory_csv, chest_rewards_path=rewards / "chest_reward_events.jsonl",
            output_dir=out, fps=float(config["dataset"]["fps"]),
            level_observations_path=base_inventory / "worker/initial_level_observations.csv"),
            ("inventory", "chest_rewards"))
        inventory_csv = reconciled / "inventory_events.csv"
        fever = stage("gold_fever", lambda out: scan_video4_gold_fever(**common, output_dir=out, sample_fps=2., end_second=max_seconds))
        status = stage("status", lambda out: scan_video4_status_events(**common, output_dir=out, sample_fps=2., end_second=max_seconds))
        menus = stage("menu_actions", lambda out: scan_video4_menu_actions(**common, output_dir=out, menu_audit_path=base_inventory / "worker/level_up_menu_audit.csv"), ("inventory",))
        from .health_calibration import calibrate_video4_health
        from .health_attribution import attribute_video4_health
        health = stage("health", lambda out: calibrate_video4_health(
            **common, output_dir=out, sample_fps=2., max_seconds=max_seconds))
        # Use this run's canonical base inventory, never the standalone CLI's
        # historical default. Chest-based recovery attribution is not inferred.
        health_attribution = stage("health_attribution", lambda out: attribute_video4_health(
            health_run_dir=health, inventory_events_path=base_inventory / "canonical_inventory_events.jsonl",
            output_dir=out), ("health", "inventory"))
        event_files.append(health_attribution / "attributed_health_events.jsonl")
        event_files.extend([chests / "chest_events.jsonl", rewards / "chest_reward_events.jsonl", fever / "gold_fever_effect_candidates.jsonl", status / "status_events.jsonl", menus / "menu_action_events.jsonl"])

    config["detectors"]["gem_xp"]["inventory_events"] = str(inventory_csv)
    write_json(runtime_path, config)
    gems = stage("gems", lambda out: run_legacy_gem_xp_detector(**common, output_dir=out, max_seconds=max_seconds),
                 ("inventory_reconciled",) if stages == "all" else ("inventory",))

    from .dashboard_projection import project_gem_events, project_inventory_events
    from .dashboard_release import build_dashboard_release
    from .game_level import apply_gameplay_interruptions, build_xp_progress
    from .gameplay_pause_export import build_gameplay_pauses
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    gem_rows = read_csv(next((gems / "worker").glob("*xp_ab_events.csv")))
    inventory_rows = read_csv(inventory_csv)
    level_inventory_rows = [row for row in inventory_rows if row.get("event_source") in {"level_up", "level_up_retrospective"}]
    def release(out: Path) -> None:
        out.mkdir(parents=True)
        missing = [p.name for p in event_files if not p.is_file()]
        if missing:
            raise RuntimeError(f"Missing release inputs: {missing}")
        payload = build_dashboard_release(video_asset_id=config["dataset"]["video_asset_id"], session_id=config["dataset"]["session_id"],
            duration_ms=min(config["dataset"]["duration_ms"], round(max_seconds * 1000)) if max_seconds else config["dataset"]["duration_ms"],
            event_files=event_files, gem_events=project_gem_events(gem_rows, 0, config["dataset"].get("segment_id", "video")), inventory_events=project_inventory_events(level_inventory_rows))
        write_json(out / "dashboard_release.json", payload)
        signal_path = next((gems / "worker").glob("*xp_frame_signal.csv"))
        with signal_path.open(newline="", encoding="utf-8") as handle:
            chest_rows = (
                [json.loads(line) for line in (chests / "chest_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                if stages == "all" else None
            )
            pauses = build_gameplay_pauses(
                csv.DictReader(handle),
                float(config["dataset"]["fps"]),
                chest_rows,
                level_inventory_rows,
            )
        write_json(out / "gameplay_pauses.json", pauses)
        progress = build_xp_progress(gem_rows, initial_level=int(config["detectors"]["gem_xp"]["initial_level"]))
        write_json(out / "xp_progress.json", apply_gameplay_interruptions(progress, pauses["intervals"]))
        if stages == "all":
            from .health_export import build_health_payload
            write_json(out / "health.json", build_health_payload(health, health_attribution))
    stage("release", release, tuple(records))
    manifest = {"prepared_by": "Tahereh Fahi", "framework_version": __version__, "status": "complete", "video_sha256": identity["video_sha256"],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "implementation_sha256": identity["implementation"],
                "run_scope": "full_video" if max_seconds is None else "smoke_prefix", "stage_scope": stages, "stages": records,
                "publication_ready": False,
                "validation_status": "not_validated",
                "quality_failed_stages": [name for name, r in records.items() if r.get("quality_status") == "failed"],
                "execution_seconds": time.monotonic() - started, "prompt_workflow_elapsed_seconds": None,
                "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "progress.json", {"prepared_by": "Tahereh Fahi", "stages": records,
        "status": "complete", "publication_ready": manifest["publication_ready"],
        "quality_failed_stages": manifest["quality_failed_stages"]})
    return manifest
