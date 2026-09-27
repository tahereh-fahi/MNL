"""Run the existing automated inventory recorder under framework control."""

from __future__ import annotations

from ..resources import load_runtime_config, resolve_path, source_reference, module_command, worker_environment

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..adapters import adapt_inventory_events
from ..hashing import sha256_file
from ..io import write_json, write_jsonl
from ..level_up_transactions import resolve_level_up_transactions
from ..xp_level_scan import code_hashes
from .instant_reward import resolve_instant_reward_transactions


WRAPPER_VERSION = "0.2.0"


def build_legacy_inventory_command(
    *,
    python_executable: Path,
    script_path: Path,
    video_path: Path,
    xp_events_path: Path,
    output_dir: Path,
    weapon_icon_dir: Path,
    passive_icon_dir: Path,
    weapon_manifest_path: Path,
    passive_manifest_path: Path,
    weapon_timeline_path: Path,
    video_key: str,
    sample_fps: float,
    start_second: float = 0.0,
    end_second: float | None = None,
) -> list[str]:
    command = [
        str(python_executable), str(script_path),
        "--video", str(video_path),
        "--xp-events", str(xp_events_path),
        "--output-dir", str(output_dir),
        "--weapon-icon-dir", str(weapon_icon_dir),
        "--passive-icon-dir", str(passive_icon_dir),
        "--weapon-manifest", str(weapon_manifest_path),
        "--passive-manifest", str(passive_manifest_path),
        "--weapon-timeline", str(weapon_timeline_path),
        "--video-key", video_key,
        "--sample-fps", format(sample_fps, ".12g"),
    ]
    if start_second > 0:
        command.extend(["--start", format(start_second, ".12g")])
    if end_second is not None:
        command.extend(["--end", format(end_second, ".12g")])
    return command


def _relative(path: Path, root: Path) -> str:
    return source_reference(path, root)


def run_legacy_inventory_detector(
    *,
    workspace_root: Path,
    config_path: Path,
    output_dir: Path,
    python_executable: Path | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    config = load_runtime_config(config_path)
    dataset = config["dataset"]
    detector = config["detectors"]["inventory"]
    paths = {
        key: resolve_path(detector[key], workspace_root)
        for key in (
            "script", "xp_events", "weapon_icon_dir", "passive_icon_dir",
            "weapon_manifest", "passive_manifest", "weapon_timeline",
        )
    }
    signal_name = paths["xp_events"].name.replace(
        "_xp_ab_events.csv", "_xp_frame_signal.csv"
    )
    if signal_name != paths["xp_events"].name:
        pause_signal = paths["xp_events"].with_name(signal_name)
        if pause_signal.is_file():
            paths["xp_frame_signal"] = pause_signal
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    if not video_path.is_file() or not all(path.exists() for path in paths.values()):
        raise FileNotFoundError("An automated inventory input is missing")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    video_sha = sha256_file(video_path)
    if video_sha != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")

    input_records = {
        key: {
            "path": _relative(path, workspace_root),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for key, path in paths.items()
    }
    implementation = code_hashes()
    identity = {
        "wrapper_version": WRAPPER_VERSION,
        "video_sha256": video_sha,
        "inputs": input_records,
        "code_sha256": implementation,
        "sample_fps": detector["sample_fps"],
        "start_second": float(detector.get("start_second", 0.0)),
        "max_seconds": max_seconds,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_id = f"run_inventory_{digest[:20]}"
    worker_dir = output_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    command = build_legacy_inventory_command(
        python_executable=Path(python_executable or sys.executable).absolute(),
        script_path=paths["script"],
        video_path=video_path,
        xp_events_path=paths["xp_events"],
        output_dir=worker_dir,
        weapon_icon_dir=paths["weapon_icon_dir"],
        passive_icon_dir=paths["passive_icon_dir"],
        weapon_manifest_path=paths["weapon_manifest"],
        passive_manifest_path=paths["passive_manifest"],
        weapon_timeline_path=paths["weapon_timeline"],
        video_key=detector["video_key"],
        sample_fps=float(detector["sample_fps"]),
        start_second=float(detector.get("start_second", 0.0)),
        end_second=max_seconds,
    )
    completed = subprocess.run(module_command(command, "inventory"), check=False, env=worker_environment())
    if completed.returncode:
        raise RuntimeError(f"Inventory worker exited with status {completed.returncode}")
    inventory_path = worker_dir / "inventory_events.csv"
    audit_path = worker_dir / "level_up_menu_audit.csv"
    level_observations_path = worker_dir / "initial_level_observations.csv"
    if not all(path.is_file() for path in (
        inventory_path, audit_path, level_observations_path
    )):
        raise RuntimeError("Inventory worker did not produce its required outputs")

    events = list(adapt_inventory_events(
        inventory_path,
        "worker/inventory_events.csv",
        video_asset_id=dataset["video_asset_id"],
        session_id=dataset["session_id"],
        processing_run_id=run_id,
    ))
    if max_seconds is not None:
        boundary_ms = round(max_seconds * 1000)
        events = [event for event in events if event.time_lower_ms < boundary_ms]
    else:
        boundary_ms = None
    transaction_events = resolve_level_up_transactions(
        menu_audit_path=audit_path,
        inventory_events_path=inventory_path,
        video_asset_id=dataset["video_asset_id"],
        session_id=dataset["session_id"],
        processing_run_id=run_id,
        source_artifact="worker/level_up_menu_audit.csv",
        end_ms=boundary_ms,
    )
    transaction_events, instant_reward_events = resolve_instant_reward_transactions(
        video_path=video_path,
        inventory_script_path=paths["script"],
        transactions=transaction_events,
    )
    superseded_inventory_ids = {
        str(event.attributes["inventory_event_id"])
        for event in transaction_events
        if event.action == "select_instant_reward"
        and event.attributes.get("inventory_event_id")
    }
    if superseded_inventory_ids:
        events = [
            event for event in events
            if not event.evidence
            or event.evidence[0].source_record_key not in superseded_inventory_ids
        ]
    events.extend(transaction_events)
    events.extend(instant_reward_events)
    events.sort(key=lambda event: (event.time_lower_ms, event.event_id))
    canonical_path = output_dir / "canonical_inventory_events.jsonl"
    count = write_jsonl(canonical_path, (event.to_dict() for event in events))
    type_counts = Counter(event.event_type for event in events)
    review_counts = Counter(event.publication_status.value for event in events)
    manifest = {
        "artifact_type": "vss_framework_legacy_inventory_run",
        "framework_version": "0.6.0",
        "wrapper_version": WRAPPER_VERSION,
        "processing_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "run_scope": "full_video" if max_seconds is None else "smoke_prefix",
        "max_seconds": max_seconds,
        "algorithm_policy": "framework_inventory_with_shared_pause_gate",
        "code_sha256": implementation,
        "inputs": {
            "video": {"path": dataset["video"]["path"], "sha256": video_sha},
            **input_records,
        },
        "counts": {
            "canonical_events": count,
            "by_event_type": dict(sorted(type_counts.items())),
            "by_publication_status": dict(sorted(review_counts.items())),
            "level_up_transactions": len(transaction_events),
            "resolved_level_up_transactions": sum(
                event.publication_status.value != "unresolved"
                for event in transaction_events
            ),
            "unresolved_level_up_transactions": sum(
                event.publication_status.value == "unresolved"
                for event in transaction_events
            ),
            "instant_reward_events": len(instant_reward_events),
            "superseded_false_inventory_source_events": len(
                superseded_inventory_ids
            ),
        },
        "outputs": {
            "canonical_inventory_events": {
                "path": canonical_path.name,
                "sha256": sha256_file(canonical_path),
                "row_count": count,
            },
            "inventory_events": {
                "path": "worker/inventory_events.csv",
                "sha256": sha256_file(inventory_path),
            },
            "level_up_menu_audit": {
                "path": "worker/level_up_menu_audit.csv",
                "sha256": sha256_file(audit_path),
            },
            "initial_level_observations": {
                "path": "worker/initial_level_observations.csv",
                "sha256": sha256_file(level_observations_path),
            },
        },
        "policies": {
            "human_coded_ground_truth_used": False,
            "human_validation_loaded": False,
            "treasure_chest_audit_loaded": False,
            "database_write_performed": False,
            "blank_is_zero": False,
            "level_up_transition_selection_excluded": True,
            "stable_menu_choice_extraction_retained": True,
        },
        "limitations": [
            "Banish, Reroll, and Skip are not yet explicitly classified.",
            "Ambiguous item identities retain their worker review status.",
        ],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"},
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
