"""Self-calibrate the player health bar from automated Video 4 observations."""

from __future__ import annotations

from .resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

import hashlib
import json
import math
import time
from dataclasses import asdict
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .hashing import sha256_file
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, SignalObservation, TemporalPrecision, Visibility
from .telemetry_scan import measure_health_bar_width
from .health_provenance import implementation_receipt
from .gameplay_state import gameplay_pause_evidence

DETECTOR_VERSION = "0.4.0"
FRAMEWORK_VERSION = "0.8.0"


def estimate_full_health_width(widths: Sequence[float], *, bin_radius: int = 1) -> tuple[float, dict[str, Any]]:
    """Choose the highest repeatedly observed width cluster, never a lone maximum."""
    clean = [float(value) for value in widths if math.isfinite(value) and value > 0]
    if len(clean) < 10:
        raise ValueError("At least 10 valid health-bar widths are required for calibration")
    rounded = Counter(round(value) for value in clean)
    support = {
        center: sum(rounded.get(candidate, 0) for candidate in range(center - bin_radius, center + bin_radius + 1))
        for center in rounded
    }
    minimum_support = max(5, math.ceil(0.02 * len(clean)))
    peak_support = max(support.values())
    # A higher but weak cluster is usually compression/geometry noise. Require
    # upper candidates to retain at least half the support of the dominant
    # cluster before preferring their larger width.
    eligible = [center for center, count in support.items() if count >= minimum_support and count >= 0.50 * peak_support]
    if not eligible:
        raise ValueError("No repeated upper-width cluster supports an automatic calibration")
    center = max(eligible)
    cluster = sorted(value for value in clean if abs(value - center) <= bin_radius)
    reference = cluster[len(cluster) // 2]
    return reference, {
        "method": "highest_repeated_width_cluster",
        "rounded_cluster_center_px": center,
        "cluster_radius_px": bin_radius,
        "cluster_support": len(cluster),
        "observation_count": len(clean),
        "support_fraction": len(cluster) / len(clean),
        "minimum_cluster_support": minimum_support,
    }


def health_percent(width: float, full_width: float) -> float:
    if full_width <= 0:
        raise ValueError("full_width must be positive")
    return max(0.0, min(100.0, 100.0 * float(width) / float(full_width)))


def reject_isolated_short_widths(samples: list[dict[str, Any]], *, threshold: float = 40.0) -> int:
    """Reject short red components unless another short candidate is adjacent."""
    original = [row["width"] for row in samples]
    rejected = 0
    for index, row in enumerate(samples):
        width = original[index]
        row["raw_width_candidate"] = width
        if row.get("excluded_from_gameplay"):
            # Preserve the screen-state exclusion rather than replacing it
            # with the generic missing-component reason below.
            continue
        if width is None or width >= threshold:
            row["rejection_reason"] = None
            continue
        adjacent_short = any(
            original[candidate] is not None and original[candidate] < threshold
            for candidate in (index - 1, index + 1)
            if 0 <= candidate < len(original)
        )
        if not adjacent_short:
            row["width"] = None
            row["confidence"] = 0.0
            row["rejection_reason"] = "isolated_short_component"
            rejected += 1
        else:
            row["rejection_reason"] = None
    return rejected


def extract_health_events(
    observations: Sequence[SignalObservation], *, processing_run_id: str,
    minimum_change: float = 3.0, maximum_gap_ms: int = 1500,
) -> list[CanonicalEvent]:
    """Emit persistent HP changes; gaps are boundaries and never filled."""
    events: list[CanonicalEvent] = []
    for index in range(1, len(observations) - 1):
        previous, current, following = observations[index - 1:index + 2]
        if not (previous.observed and current.observed and following.observed):
            continue
        if len({row.attributes.get("gameplay_segment_id") for row in (previous, current, following)}) != 1:
            continue
        if current.time_lower_ms - previous.time_lower_ms > maximum_gap_ms or following.time_lower_ms - current.time_lower_ms > maximum_gap_ms:
            continue
        assert previous.numeric_value is not None and current.numeric_value is not None and following.numeric_value is not None
        delta = float(current.numeric_value) - float(previous.numeric_value)
        if abs(delta) < minimum_change:
            continue
        # The change must persist into the next direct observation. This drops
        # single-frame red-component mistakes without smoothing or imputation.
        if delta < 0 and float(following.numeric_value) > float(current.numeric_value) + 1.5:
            continue
        if delta > 0 and float(following.numeric_value) < float(current.numeric_value) - 1.5:
            continue
        event_type = "hp_loss" if delta < 0 else "hp_recovery"
        events.append(CanonicalEvent(
            event_id=f"vss_{event_type}_{current.frame_number}",
            video_asset_id=current.video_asset_id, session_id=current.session_id,
            event_family="punishment" if delta < 0 else "recovery", event_type=event_type,
            time_lower_ms=previous.time_lower_ms, time_upper_ms=current.time_upper_ms,
            anchor_time_ms=current.time_lower_ms, temporal_precision=TemporalPrecision.BOUNDED,
            evidence_grade=EvidenceGrade.B, publication_status=PublicationStatus.AUTO_ACCEPTED,
            inference_method="persistent_self_calibrated_health_change", processing_run_id=processing_run_id,
            frame_number=current.frame_number, quantity=abs(delta), unit="percentage_points",
            evidence=(EvidenceReference(
                source_artifact=current.source_artifact, source_record_key=current.source_record_key,
                modalities=("health_bar",), details={"previous_percent": previous.numeric_value, "current_percent": current.numeric_value, "following_percent": following.numeric_value},
            ),),
            attributes={"direction": "decrease" if delta < 0 else "increase", "imputed": False, "persistence_confirmed": True},
        ))
    return events


def _load_config(path: Path) -> Any:
    from .detectors.gems import Config
    return Config()


def calibrate_video4_health(*, workspace_root: Path, config_path: Path, output_dir: Path, sample_fps: float = 2.0, max_seconds: float | None = None) -> dict[str, Any]:
    """Sample the whole pinned video, calibrate full width, and publish HP percent."""
    import cv2

    started = time.monotonic()
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    config = load_runtime_config(config_path)
    if config.get("policies", {}).get("use_human_coded_ground_truth"):
        raise ValueError("Health extraction cannot consume human-coded ground truth")
    receipt = implementation_receipt()
    dataset = config["dataset"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    geometry_path = resolve_path(config["detectors"]["gem_xp"]["script"], workspace_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if sha256_file(video_path) != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")

    hp_config = _load_config(geometry_path)
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Cannot read video metadata: {video_path}")
    step = fps / sample_fps
    samples: list[dict[str, Any]] = []
    gameplay_segment_id = 0
    was_paused = False
    position = 0.0
    limit = frame_count if max_seconds is None else min(frame_count, math.ceil(max_seconds * fps))
    while round(position) < limit:
        frame_number = round(position)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if ok:
            pause = gameplay_pause_evidence(frame)
            paused = bool(pause["blocked"])
            if paused and not was_paused:
                gameplay_segment_id += 1
            was_paused = paused
            width, confidence = (None, 0.0) if paused else measure_health_bar_width(frame, hp_config)
            samples.append({"frame_number": frame_number, "media_time_ms": round(1000 * frame_number / fps),
                            "width": width, "confidence": confidence,
                            "excluded_from_gameplay": paused, "screen_state": pause["phase"],
                            "rejection_reason": pause["reason"] if paused else None,
                            "gameplay_segment_id": gameplay_segment_id})
        position += step
    capture.release()

    isolated_short_rejections = reject_isolated_short_widths(samples)

    observed_widths = [row["width"] for row in samples if row["width"] is not None]
    full_width, calibration = estimate_full_health_width(observed_widths)
    observations: list[SignalObservation] = []
    for row in samples:
        observed = row["width"] is not None
        observations.append(SignalObservation(
            observation_id=f"vss_player_health_percent_{row['frame_number']}",
            video_asset_id=dataset["video_asset_id"], session_id=dataset["session_id"],
            detector_name="self_calibrated_player_health", detector_version=DETECTOR_VERSION,
            observable_code="player_health_percent", time_lower_ms=row["media_time_ms"], time_upper_ms=row["media_time_ms"],
            temporal_precision=TemporalPrecision.FRAME,
            visibility=Visibility.VISIBLE if observed else Visibility.UNKNOWN,
            evidence_grade=EvidenceGrade.B if observed else EvidenceGrade.UNRESOLVED,
            observed=observed, source_artifact="VSS frame work/src/vss_framework/health_calibration.py",
            source_record_key=f"frame:{row['frame_number']}",
            numeric_value=health_percent(row["width"], full_width) if observed else None,
            unit="percent_of_calibrated_full_bar", frame_number=row["frame_number"],
            attributes={"raw_red_fill_width_1440p_px": row.get("raw_width_candidate"), "confidence": row["confidence"], "rejection_reason": row.get("rejection_reason"),
                        "excluded_from_gameplay": row["excluded_from_gameplay"], "screen_state": row["screen_state"],
                        "gameplay_segment_id": row["gameplay_segment_id"],
                        "full_bar_reference_width_1440p_px": full_width, "calibration_method": calibration["method"], "imputed": False},
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = output_dir / "health_configuration.json"
    write_json(configuration_path, config)
    observations_path = output_dir / "health_observations.jsonl"
    calibration_path = output_dir / "health_calibration.json"
    write_jsonl(observations_path, (row.to_dict() for row in observations))
    write_json(calibration_path, {"prepared_by": "Tahereh Fahi", "full_bar_reference_width_1440p_px": full_width, **calibration})
    identity = json.dumps({"version": DETECTOR_VERSION, "video": dataset['video']['sha256'],
                           "sample_fps": sample_fps, "full_width": full_width,
                           "max_seconds": max_seconds, "implementation": receipt,
                           "config_sha256": sha256_file(config_path)}, sort_keys=True)
    processing_run_id = f"run_health_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    events = extract_health_events(
        observations, processing_run_id=processing_run_id,
        maximum_gap_ms=math.ceil(1500 / sample_fps),
    )
    events_path = output_dir / "health_events.jsonl"
    write_jsonl(events_path, (event.to_dict() for event in events))
    manifest = {
        "artifact_type": "vss_framework_health_calibration_run", "framework_version": FRAMEWORK_VERSION,
        "processing_run_id": processing_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "prepared_by": "Tahereh Fahi",
        "configuration": {"sample_fps": sample_fps},
        "provenance": {**receipt, "config_sha256": sha256_file(config_path),
                       "effective_geometry": asdict(hp_config),
                       "max_seconds": max_seconds,
                       "video_asset_id": dataset["video_asset_id"], "session_id": dataset["session_id"]},
        "run_scope": "full_video" if max_seconds is None else "smoke_prefix",
        "publication_ready": False,
        "execution_seconds": time.monotonic() - started,
        "prompt_workflow_elapsed_seconds": None,
        "calibration": {"full_bar_reference_width_1440p_px": full_width, **calibration},
        "counts": {"sampled_frames": len(samples), "observed_health_frames": len(observed_widths), "missing_health_frames": len(samples) - len(observed_widths),
                   "level_up_pause_excluded_frames": sum(row["excluded_from_gameplay"] for row in samples),
                   "isolated_short_components_rejected": isolated_short_rejections, "hp_loss_events": sum(event.event_type == "hp_loss" for event in events), "hp_recovery_events": sum(event.event_type == "hp_recovery" for event in events)},
        "source_integrity": {"video_sha256": dataset["video"]["sha256"], "geometry_source_sha256": sha256_file(geometry_path), "detector_source_sha256": sha256_file(Path(__file__))},
        "outputs": {
            "health_configuration": {"path": configuration_path.name, "sha256": sha256_file(configuration_path)},
            "health_observations": {"path": observations_path.name, "sha256": sha256_file(observations_path), "row_count": len(observations)},
            "health_calibration": {"path": calibration_path.name, "sha256": sha256_file(calibration_path)},
            "health_events": {"path": events_path.name, "sha256": sha256_file(events_path), "row_count": len(events)},
        },
        "policies": {"human_coded_ground_truth_used": False, "imputation_performed": False, "database_write_performed": False},
        "limitations": [
            "Percentage is relative to the video's automatically inferred full-bar width, not an OCR reading of numeric HP.",
            "The current red-component detector may return missing when the visible fill is shorter than its geometry threshold.",
            "Missing observations remain missing and are not interpreted as zero health.",
            "Level-up transitions and menus are excluded from health calibration and change detection.",
        ],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"},
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
