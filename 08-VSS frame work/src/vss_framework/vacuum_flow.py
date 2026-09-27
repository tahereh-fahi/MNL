"""Detect mass radial gem flow compatible with a Vacuum pickup."""

from __future__ import annotations

from .resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

import hashlib
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .hashing import sha256_file
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision
from .video import OpenCVVideoReader
from .gameplay_state import gameplay_pause_evidence

DETECTOR_VERSION = "0.2.0"


def gem_like_centroids(frame: Any) -> list[tuple[float, float]]:
    """Return small saturated blue/green/red component centers at 640x360."""
    import cv2
    import numpy as np
    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colored = (saturation >= 125) & (value >= 85) & (
        ((hue <= 8) | (hue >= 170)) | ((hue >= 35) & (hue <= 95)) | ((hue >= 96) & (hue <= 125)))
    # HUD and borders are not world pickups.
    colored[:32, :] = False
    colored[:, :12] = False
    colored[:, -12:] = False
    count, _, stats, centers = cv2.connectedComponentsWithStats(colored.astype(np.uint8), 8)
    result = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if 3 <= area <= 130 and 2 <= width <= 24 and 2 <= height <= 28:
            result.append((float(centers[index, 0]), float(centers[index, 1])))
    return result


def _load_legacy_gem_detector(script_path: Path, template_dir: Path, frame_height: int) -> tuple[Any, Any, Any]:
    """Reuse the established label-free gem template detector unchanged."""
    from .detectors import gems as module
    config = module.Config()
    templates = module.load_template_bank(template_dir, frame_height, profile="legacy")
    return module, config, templates


def radial_flow_features(previous: Sequence[tuple[float, float]], current: Sequence[tuple[float, float]],
                         *, dt_seconds: float, center: tuple[float, float] = (320, 190),
                         previous_center: tuple[float, float] | None = None,
                         maximum_match_distance: float = 34.0) -> dict[str, float | int]:
    """Greedily match colored components and measure inward radial velocity."""
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    available = set(range(len(current)))
    prior_center = center if previous_center is None else previous_center
    center_delta = (center[0] - prior_center[0], center[1] - prior_center[1])
    inward_speeds: list[float] = []
    outer_matches = 0
    for px, py in previous:
        if not available:
            break
        predicted_x, predicted_y = px + center_delta[0], py + center_delta[1]
        index = min(available, key=lambda candidate: math.hypot(current[candidate][0] - predicted_x, current[candidate][1] - predicted_y))
        cx, cy = current[index]
        displacement = math.hypot(cx - predicted_x, cy - predicted_y)
        if displacement > maximum_match_distance:
            continue
        available.remove(index)
        prior_radius = math.hypot(px - prior_center[0], py - prior_center[1])
        current_radius = math.hypot(cx - center[0], cy - center[1])
        if prior_radius >= 70:
            outer_matches += 1
            inward_speeds.append((prior_radius - current_radius) / dt_seconds)
    positive = [speed for speed in inward_speeds if speed > 2]
    return {"previous_components": len(previous), "current_components": len(current),
            "outer_matches": outer_matches,
            "inward_fraction": len(positive) / outer_matches if outer_matches else 0.0,
            "median_inward_speed": median(positive) if positive else 0.0,
            "net_median_radial_speed": median(inward_speeds) if inward_speeds else 0.0}


def mark_vacuum_like_rows(rows: list[dict[str, Any]], *, sample_fps: float,
                          baseline_seconds: float = 8.0) -> None:
    history: deque[int] = deque(maxlen=max(5, round(baseline_seconds * sample_fps)))
    for row in rows:
        if not row.get("observable"):
            history.clear()
            row["vacuum_like"] = False
            continue
        baseline = median(history) if history else 0
        row["outer_match_baseline"] = baseline
        # Vacuum must be both numerous and unusually broad compared with the
        # recent local attraction stream.
        row["vacuum_like"] = (int(row["outer_matches"]) >= max(10, baseline * 1.6)
                              and float(row["inward_fraction"]) >= .68
                              and float(row["median_inward_speed"]) >= 35)
        history.append(int(row["outer_matches"]))


def resolve_vacuum_intervals(rows: Sequence[dict[str, Any]], *, processing_run_id: str,
                             video_asset_id: str, session_id: str, sample_period_ms: int,
                             minimum_samples: int = 3) -> list[CanonicalEvent]:
    groups: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] = []
    for row in rows:
        contiguous = active and int(row["media_time_ms"]) - int(active[-1]["media_time_ms"]) <= sample_period_ms * 1.5
        if row.get("vacuum_like") and (not active or contiguous):
            active.append(row)
        else:
            if active:
                groups.append(active)
            active = [row] if row.get("vacuum_like") else []
    if active:
        groups.append(active)
    events = []
    for index, group in enumerate(groups, start=1):
        if len(group) < minimum_samples:
            continue
        start_components = int(group[0]["previous_components"])
        end_components = int(group[-1]["current_components"])
        depletion_fraction = (start_components - end_components) / max(start_components, 1)
        if depletion_fraction < .25:
            continue
        start, end = int(group[0]["media_time_ms"]), int(group[-1]["media_time_ms"]) + sample_period_ms
        events.append(CanonicalEvent(
            event_id=f"vss_vacuum_flow_{index}_{start}", video_asset_id=video_asset_id,
            session_id=session_id, event_family="effect", event_type="vacuum_flow_candidate",
            time_lower_ms=start, time_upper_ms=end, anchor_time_ms=start,
            temporal_precision=TemporalPrecision.INTERVAL, evidence_grade=EvidenceGrade.C,
            publication_status=PublicationStatus.NEEDS_REVIEW,
            inference_method="persistent_mass_radial_gem_like_flow", processing_run_id=processing_run_id,
            evidence=(EvidenceReference(source_artifact="vacuum_flow_signals.jsonl",
                source_record_key=f"rows:{group[0]['frame_number']}-{group[-1]['frame_number']}",
                modalities=("color_components", "radial_motion"),
                details={"sample_count": len(group),
                         "median_outer_matches": median(int(row["outer_matches"]) for row in group),
                         "median_inward_fraction": median(float(row["inward_fraction"]) for row in group),
                         "median_inward_speed": median(float(row["median_inward_speed"]) for row in group),
                         "start_components": start_components, "end_components": end_components,
                         "depletion_fraction": depletion_fraction}),),
            attributes={"compatible_pickup_types": ["vacuum"], "vacuum_confirmed": False,
                        "gem_identity_is_color_geometry_candidate": True, "imputed": False}))
    return events


def scan_video4_vacuum_flow(*, workspace_root: Path, config_path: Path, output_dir: Path,
                            sample_fps: float = 5.0, start_second: float = 0.0,
                            end_second: float | None = None) -> dict[str, Any]:
    config = load_runtime_config(config_path)
    dataset = config["dataset"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    gem_script = resolve_path(config["detectors"]["gem_xp"]["script"], workspace_root)
    template_dir = resolve_path(config["detectors"]["gem_xp"]["template_dir"], workspace_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if sha256_file(video_path) != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    rows: list[dict[str, Any]] = []
    previous_points = None
    previous_center = None
    previous_time = None
    with OpenCVVideoReader(video_path, dataset["video_asset_id"]) as reader:
        gem_module, gem_config, gem_templates = _load_legacy_gem_detector(gem_script, template_dir, reader.metadata.height)
        for packet in reader.iter_packets(start_ms=round(start_second * 1000),
                end_ms=None if end_second is None else round(end_second * 1000), sample_fps=sample_fps):
            pause = gameplay_pause_evidence(packet.image)
            menu = bool(pause["blocked"])
            if menu:
                points, player_center = [], (320.0, 190.0)
            else:
                anchor = gem_module.detect_player_anchor(packet.image, gem_config)
                detections = gem_module.detect_gems_near_player(
                    packet.image, gem_templates, anchor, gem_config,
                    search_radius_1440p=1100.0, enforce_search_circle=True)
                scale_x, scale_y = 640.0 / reader.metadata.width, 360.0 / reader.metadata.height
                points = [(row.cx * scale_x, row.cy * scale_y) for row in detections]
                player_center = (anchor.x * scale_x, anchor.y * scale_y)
            if previous_points is None or menu:
                row = {"frame_number": packet.frame_number, "media_time_ms": packet.media_time_ms,
                       "observable": False, "reason": pause["reason"] if menu else "no_contiguous_gameplay_pair"}
            else:
                features = radial_flow_features(previous_points, points,
                    dt_seconds=(packet.media_time_ms - previous_time) / 1000, center=player_center,
                    previous_center=previous_center)
                row = {"frame_number": packet.frame_number, "media_time_ms": packet.media_time_ms,
                       "observable": True, **features}
            row["gameplay_pause"] = pause
            rows.append(row)
            previous_points = None if menu else points
            previous_center = None if menu else player_center
            previous_time = None if menu else packet.media_time_ms
    mark_vacuum_like_rows(rows, sample_fps=sample_fps)
    sample_period_ms = math.ceil(1000 / sample_fps)
    identity = f"{DETECTOR_VERSION}|{dataset['video']['sha256']}|{sample_fps}|{start_second}|{end_second}"
    run_id = f"run_vacuum_flow_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    events = resolve_vacuum_intervals(rows, processing_run_id=run_id,
        video_asset_id=dataset["video_asset_id"], session_id=dataset["session_id"], sample_period_ms=sample_period_ms)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path, events_path = output_dir / "vacuum_flow_signals.jsonl", output_dir / "vacuum_flow_candidates.jsonl"
    write_jsonl(signals_path, rows)
    write_jsonl(events_path, (event.to_dict() for event in events))
    manifest = {"artifact_type": "vss_framework_vacuum_flow_run", "framework_version": "0.11.0",
        "detector_version": DETECTOR_VERSION, "processing_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "prepared_by": "Tahereh Fahi",
        "configuration": {"sample_fps": sample_fps, "start_second": start_second, "end_second": end_second,
                          "minimum_consecutive_samples": 3,
                          "minimum_outer_matches": 10, "minimum_inward_fraction": .68,
                          "minimum_inward_speed_640px_per_second": 35, "minimum_depletion_fraction": .25},
        "counts": {"sampled_frames": len(rows), "observable_pairs": sum(bool(row.get("observable")) for row in rows),
                   "vacuum_like_samples": sum(bool(row.get("vacuum_like")) for row in rows),
                   "vacuum_flow_candidates": len(events)},
        "source_integrity": {"video_sha256": dataset["video"]["sha256"],
                             "legacy_gem_detector_sha256": sha256_file(gem_script),
                             "detector_source_sha256": sha256_file(Path(__file__)),
                             "gameplay_state_source_sha256": sha256_file(Path(__file__).with_name("gameplay_state.py"))},
        "outputs": {"signals": {"path": signals_path.name, "sha256": sha256_file(signals_path), "row_count": len(rows)},
                    "candidates": {"path": events_path.name, "sha256": sha256_file(events_path), "row_count": len(events)}},
        "policies": {"human_coded_ground_truth_used": False, "imputation_performed": False,
                     "vacuum_identity_auto_accepted": False, "database_write_performed": False},
        "limitations": ["Template-confirmed gem detections can still contain visually similar false matches.",
                        "Normal Magnet attraction and weapon effects can produce radial motion; persistence and count gates reduce but do not eliminate this ambiguity."],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
