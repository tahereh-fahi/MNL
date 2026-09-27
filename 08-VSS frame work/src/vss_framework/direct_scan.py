"""Run detector wrappers directly against an interval of source video."""

from __future__ import annotations

from .resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import adapt_automated_signals
from .detectors import ScreenStateDetector, XPBarDetector
from .hashing import sha256_file
from .io import write_json, write_jsonl
from .video import OpenCVVideoReader


DIRECT_SCAN_VERSION = "0.2.0"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, workspace_root: Path) -> str:
    return source_reference(path, workspace_root)


def _run_id(
    *, video_sha256: str, start_ms: int, end_ms: int | None, sample_fps: float
) -> str:
    key = (
        f"{DIRECT_SCAN_VERSION}|{video_sha256}|{start_ms}|{end_ms}|{sample_fps}"
    ).encode("utf-8")
    return f"scan_video4_{hashlib.sha256(key).hexdigest()[:20]}"


def scan_video4_interval(
    *,
    workspace_root: Path,
    config_path: Path,
    output_dir: Path,
    start_ms: int,
    end_ms: int | None,
    sample_fps: float,
    verify_video_hash: bool = True,
    include_cached_signals: bool = False,
) -> dict[str, Any]:
    """Sample Video 4 and emit direct, frame-timed screen observations."""

    config = load_runtime_config(config_path)
    dataset = config["dataset"]
    video_config = dataset["video"]
    video_path = (workspace_root / video_config["path"]).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    expected_video_sha = str(video_config["sha256"])
    if verify_video_hash:
        actual_video_sha = sha256_file(video_path)
        if actual_video_sha != expected_video_sha:
            raise ValueError(
                f"Video SHA-256 mismatch: expected {expected_video_sha}, got {actual_video_sha}"
            )
    else:
        actual_video_sha = expected_video_sha

    inventory_script = detector_path("inventory")
    weapon_script = detector_path("weapons")
    screen_detector = ScreenStateDetector(
        legacy_scripts_dir=inventory_script.parent,
        source_artifact=_relative(inventory_script, workspace_root),
    )
    xp_script = detector_path("gem_xp")
    xp_detector = XPBarDetector(
        legacy_scripts_dir=xp_script.parent,
        source_artifact=_relative(xp_script, workspace_root),
    )
    frame_observations = []
    with OpenCVVideoReader(video_path, dataset["video_asset_id"]) as reader:
        metadata = reader.metadata
        effective_end_ms = min(
            metadata.duration_ms, metadata.duration_ms if end_ms is None else end_ms
        )
        for packet in reader.iter_packets(
            start_ms=start_ms, end_ms=end_ms, sample_fps=sample_fps
        ):
            frame_observations.extend(screen_detector.observe(packet))
            frame_observations.extend(xp_detector.observe(packet))

    canonical_spec = {"path": None, "sha256": None}
    canonical_sha = None
    signal_observations = []
    if include_cached_signals:
        canonical_spec = config["sources"]["canonical_observations"]
        canonical_path = resolve_path(canonical_spec["path"], workspace_root)
        canonical_sha = sha256_file(canonical_path)
        if canonical_sha != canonical_spec["sha256"]:
            raise ValueError("Canonical automated observation source failed SHA-256 verification")
        signal_observations = list(adapt_automated_signals(
            canonical_path, canonical_spec["path"], start_ms=start_ms, end_ms=effective_end_ms,
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_observations_path = output_dir / "frame_observations.jsonl"
    frame_observation_count = write_jsonl(
        frame_observations_path,
        (observation.to_dict() for observation in frame_observations),
    )
    signals_path = output_dir / "signal_observations.jsonl"
    signal_observation_count = write_jsonl(
        signals_path, (observation.to_dict() for observation in signal_observations)
    )
    screen_observations = [
        item for item in frame_observations if item.observation_type == "screen_state"
    ]
    xp_observations = [
        item for item in frame_observations if item.observation_type == "xp_bar_progress"
    ]
    state_counts = Counter(str(observation.value) for observation in screen_observations)
    visibility_counts = Counter(
        observation.visibility.value for observation in frame_observations
    )
    signal_counts = Counter(item.observable_code for item in signal_observations)
    manifest = {
        "artifact_type": "vss_framework_direct_video_scan",
        "framework_version": DIRECT_SCAN_VERSION,
        "processing_run_id": _run_id(
            video_sha256=actual_video_sha,
            start_ms=start_ms,
            end_ms=end_ms,
            sample_fps=sample_fps,
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "dataset": {
            "video_asset_id": dataset["video_asset_id"],
            "session_id": dataset["session_id"],
        },
        "scan": {
            "interval_semantics": "half_open_[start,end)",
            "start_ms": start_ms,
            "end_ms": effective_end_ms,
            "sample_fps": sample_fps,
            "native_fps": metadata.fps,
            "native_frame_count": metadata.frame_count,
            "frame_width": metadata.width,
            "frame_height": metadata.height,
        },
        "detectors": [
            {
                "name": screen_detector.name,
                "version": screen_detector.version,
                "reused_entry_point": "level_up_option_rectangles",
                "source_artifact": _relative(inventory_script, workspace_root),
                "source_sha256": sha256_file(inventory_script),
                "dependency_artifact": _relative(weapon_script, workspace_root),
                "dependency_sha256": sha256_file(weapon_script),
            },
            {
                "name": xp_detector.name,
                "version": xp_detector.version,
                "reused_entry_points": [
                    "measure_xp_bar_progress",
                    "gameplay_hud_score",
                    "level_up_overlay_score",
                ],
                "source_artifact": _relative(xp_script, workspace_root),
                "source_sha256": sha256_file(xp_script),
            },
            {
                "name": "sealed_automated_signal_adapter",
                "enabled": include_cached_signals,
                "version": "0.1.0",
                "observable_codes": ["kill_counter", "game_clock", "gem_pickup"],
                "source_artifact": canonical_spec["path"],
                "source_sha256": canonical_sha,
            },
        ],
        "source_integrity": {
            "video": {
                "path": video_config["path"],
                "expected_sha256": expected_video_sha,
                "observed_sha256": actual_video_sha if verify_video_hash else None,
                "verified": verify_video_hash,
            },
            "canonical_automated_observations": {
                "path": canonical_spec["path"],
                "expected_sha256": canonical_spec["sha256"],
                "observed_sha256": canonical_sha,
                "verified": include_cached_signals,
            },
        },
        "counts": {
            "frame_observations": frame_observation_count,
            "signal_observations": signal_observation_count,
            "screen_state_observations": len(screen_observations),
            "xp_bar_observations": len(xp_observations),
            "accepted_xp_bar_observations": sum(
                bool(item.attributes.get("accepted")) for item in xp_observations
            ),
            "by_state": dict(sorted(state_counts.items())),
            "by_visibility": dict(sorted(visibility_counts.items())),
            "signals_by_observable": dict(sorted(signal_counts.items())),
            "observed_kill_counter_signals": sum(
                item.observable_code == "kill_counter" and item.observed
                for item in signal_observations
            ),
        },
        "outputs": {
            "frame_observations": {
                "path": frame_observations_path.name,
                "sha256": sha256_file(frame_observations_path),
                "row_count": frame_observation_count,
            },
            "signal_observations": {
                "path": signals_path.name,
                "sha256": sha256_file(signals_path),
                "row_count": signal_observation_count,
            },
        },
        "policies": {
            "human_coded_ground_truth_used": False,
            "blank_is_zero": False,
            "unknown_frames_are_preserved": True,
        },
        "limitations": [
            "Screen-state observations identify Level-Up menus and otherwise report only HUD visibility.",
            "HUD visibility alone is not treated as proof of ordinary gameplay.",
            "Unknown screen states remain unknown; they are not coerced to gameplay.",
            "One sampled frame is an observation, not a resolved event.",
            "Kill Counter and game-clock signals preserve observed versus carried or unavailable state.",
            "Gem quantities remain XP-linked estimates rather than exact physical pickup counts.",
        ],
        "metadata_verification": {
            "prepared_by": "Tahereh Fahi",
            "human_ground_truth_used": False,
        },
    }
    manifest_path = output_dir / "scan_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
