"""Conservative sprite tracking for visible world-pickup candidates."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .hashing import sha256_file
from .gameplay_state import gameplay_pause_evidence
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision
from .video import OpenCVVideoReader

DETECTOR_VERSION = "0.2.0"
FRAMEWORK_VERSION = "0.9.0"


@dataclass(frozen=True)
class PickupDetection:
    code: str
    frame_number: int
    media_time_ms: int
    x: float
    y: float
    score: float
    scale: float


@dataclass
class PickupTrack:
    track_id: int
    code: str
    detections: list[PickupDetection] = field(default_factory=list)

    @property
    def last(self) -> PickupDetection:
        return self.detections[-1]


def non_maximum_suppression(detections: Sequence[PickupDetection], radius: float = 24.0) -> list[PickupDetection]:
    kept: list[PickupDetection] = []
    for candidate in sorted(detections, key=lambda row: row.score, reverse=True):
        if all(math.hypot(candidate.x - row.x, candidate.y - row.y) > radius for row in kept):
            kept.append(candidate)
    return kept


def associate_tracks(
    frames: Sequence[Sequence[PickupDetection]], *, maximum_distance: float = 100.0,
    maximum_gap_ms: int = 1250,
    interrupted_sample_times_ms: Sequence[int] = (),
) -> list[PickupTrack]:
    """Associate visible sprites without joining tracks across excluded UI."""
    tracks: list[PickupTrack] = []
    interrupted = sorted(interrupted_sample_times_ms)
    next_id = 1
    for detections in frames:
        used: set[int] = set()
        for detection in sorted(detections, key=lambda row: row.score, reverse=True):
            if bisect_left(interrupted, detection.media_time_ms) != bisect_right(interrupted, detection.media_time_ms):
                continue
            choices = [
                track for track in tracks
                if track.track_id not in used and track.code == detection.code
                and 0 < detection.media_time_ms - track.last.media_time_ms <= maximum_gap_ms
                and bisect_right(interrupted, track.last.media_time_ms) == bisect_right(interrupted, detection.media_time_ms)
                and math.hypot(detection.x - track.last.x, detection.y - track.last.y) <= maximum_distance
            ]
            if choices:
                track = min(choices, key=lambda row: math.hypot(detection.x - row.last.x, detection.y - row.last.y))
            else:
                track = PickupTrack(next_id, detection.code)
                next_id += 1
                tracks.append(track)
            track.detections.append(detection)
            used.add(track.track_id)
    return tracks


def resolve_pickup_events(
    tracks: Iterable[PickupTrack], *, processing_run_id: str, video_asset_id: str,
    session_id: str, player_center: tuple[float, float], final_sample_ms: int,
    minimum_observations: int = 2, collection_radius: float = 150.0,
    sample_period_ms: int = 500,
    interrupted_sample_times_ms: Sequence[int] = (),
) -> list[CanonicalEvent]:
    """Resolve only persistent tracks that disappear close to the player."""
    events: list[CanonicalEvent] = []
    interrupted = sorted(interrupted_sample_times_ms)
    for track in tracks:
        if len(track.detections) < minimum_observations or track.last.media_time_ms >= final_sample_ms - sample_period_ms:
            continue
        # A menu hiding a previously visible sprite is unavailable evidence,
        # not evidence that the object was collected by the player.
        start, end = track.last.media_time_ms, track.last.media_time_ms + sample_period_ms
        if bisect_left(interrupted, start) != bisect_right(interrupted, end):
            continue
        distance = math.hypot(track.last.x - player_center[0], track.last.y - player_center[1])
        if distance > collection_radius:
            continue
        first, last = track.detections[0], track.last
        mean_score = sum(row.score for row in track.detections) / len(track.detections)
        # Disappearance near the player is strong enough for a candidate, but
        # not direct proof of collection because occlusion is still possible.
        events.append(CanonicalEvent(
            event_id=f"vss_world_pickup_{track.code}_{track.track_id}_{last.frame_number}",
            video_asset_id=video_asset_id, session_id=session_id,
            event_family="world_pickup", event_type=track.code,
            time_lower_ms=last.media_time_ms, time_upper_ms=min(final_sample_ms, last.media_time_ms + sample_period_ms),
            anchor_time_ms=last.media_time_ms, temporal_precision=TemporalPrecision.BOUNDED,
            evidence_grade=EvidenceGrade.C, publication_status=PublicationStatus.NEEDS_REVIEW,
            inference_method="persistent_sprite_track_disappeared_near_player",
            processing_run_id=processing_run_id, frame_number=last.frame_number,
            action="pickup", acquisition_source="world_pickup", quantity=1, unit="physical_pickup_candidate",
            evidence=(EvidenceReference(
                source_artifact="world_pickup_detections.jsonl", source_record_key=f"track:{track.track_id}",
                modalities=("sprite_template", "temporal_tracking", "player_proximity"),
                details={"observations": len(track.detections), "first_seen_ms": first.media_time_ms,
                         "last_seen_ms": last.media_time_ms, "mean_score": mean_score,
                         "last_distance_to_player_px": distance},
            ),),
            attributes={"candidate_only": True, "occlusion_not_excluded": True, "imputed": False},
        ))
    return events


def is_large_level_up_overlay(frame: Any) -> bool:
    """Compatibility entry point covering small animated and stable panels."""
    return bool(gameplay_pause_evidence(frame)["blocked"])


def _load_templates(asset_dir: Path, scales: Sequence[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cv2
    manifest_path = asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    templates: list[dict[str, Any]] = []
    for asset in manifest["assets"]:
        path = asset_dir / asset["file"]
        if sha256_file(path) != asset["sha256"]:
            raise ValueError(f"Reference sprite hash mismatch: {path}")
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            raise ValueError(f"Reference sprite must have an alpha channel: {path}")
        alpha = rgba[:, :, 3]
        x, y, w, h = cv2.boundingRect((alpha > 0).astype("uint8"))
        rgba = rgba[y:y+h, x:x+w]
        for scale in scales:
            width, height = max(4, round(w * scale)), max(4, round(h * scale))
            resized = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_NEAREST)
            templates.append({"code": asset["code"], "scale": scale,
                              "bgr": resized[:, :, :3], "mask": resized[:, :, 3]})
    return templates, manifest


def alpha_composite(frame: Any, rgba: Any, x: int, y: int) -> Any:
    """Composite an RGBA sprite into a BGR frame for detector calibration."""
    import numpy as np
    result = frame.copy()
    height, width = rgba.shape[:2]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    target = result[y:y + height, x:x + width].astype(np.float32)
    result[y:y + height, x:x + width] = (rgba[:, :, :3] * alpha + target * (1.0 - alpha)).astype("uint8")
    return result


def calibrate_match_thresholds(
    frames: Sequence[Any], templates: Sequence[dict[str, Any]], *, margin: float = 0.04,
) -> dict[str, dict[str, float | bool]]:
    """Separate injected positives from untouched-video hard negatives."""
    import cv2
    import numpy as np
    results: dict[str, dict[str, float | bool]] = {}
    for template in templates:
        key = f"{template['code']}@{template['scale']:.4f}"
        negative_scores: list[float] = []
        positive_scores: list[float] = []
        sprite = np.dstack((template["bgr"], template["mask"]))
        th, tw = template["bgr"].shape[:2]
        for frame in frames:
            height, width = frame.shape[:2]
            working = cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            roi = working[max(0, height // 4 - 360):min(working.shape[0], height // 4 + 360),
                          max(0, width // 4 - 360):min(working.shape[1], width // 4 + 360)]
            if roi.shape[0] < th or roi.shape[1] < tw:
                continue
            negative_scores.append(float(cv2.minMaxLoc(cv2.matchTemplate(roi, template["bgr"], cv2.TM_CCOEFF_NORMED))[1]))
            x = max(2, roi.shape[1] // 2 - tw // 2)
            y = max(2, roi.shape[0] // 2 - th // 2)
            injected = alpha_composite(roi, sprite, x, y)
            local = injected[max(0, y - 3):min(injected.shape[0], y + th + 3), max(0, x - 3):min(injected.shape[1], x + tw + 3)]
            positive_scores.append(float(cv2.minMaxLoc(cv2.matchTemplate(local, template["bgr"], cv2.TM_CCOEFF_NORMED))[1]))
        negative_max = max(negative_scores, default=1.0)
        positive_min = min(positive_scores, default=0.0)
        separable = positive_min > negative_max + margin
        results[key] = {"negative_max": negative_max, "positive_min": positive_min,
                        "margin": positive_min - negative_max, "separable": separable,
                        "threshold": (negative_max + positive_min) / 2 if separable else 1.0}
    return results


def calibrate_video4_world_pickups(
    *, workspace_root: Path, config_path: Path, asset_dir: Path, output_path: Path,
    frame_times_seconds: Sequence[float] = (60, 180, 300, 420, 540, 660, 780, 1020),
    scales: Sequence[float] = (0.25, 0.5),
) -> dict[str, Any]:
    import cv2
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    if sha256_file(video_path) != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    processing_scale = .5
    templates, manifest = _load_templates(asset_dir, tuple(scale * processing_scale for scale in scales))
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[Any] = []
    for second in frame_times_seconds:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
        ok, frame = capture.read()
        if ok and not is_large_level_up_overlay(frame):
            frames.append(frame)
    capture.release()
    results = calibrate_match_thresholds(frames, templates)
    payload = {"artifact_type": "vss_world_pickup_self_calibration", "calibration_version": "1.0.0",
               "prepared_by": "Tahereh Fahi", "video_asset_id": dataset["video_asset_id"],
               "video_sha256": dataset["video"]["sha256"], "human_coded_ground_truth_used": False,
               "method": "alpha-injected positives versus untouched-video hard negatives",
               "sample_times_seconds": list(frame_times_seconds), "usable_frame_count": len(frames),
               "source_scale_values": list(scales), "processing_scale": processing_scale,
               "reference_manifest_sha256": sha256_file(asset_dir / "manifest.json"),
               "thresholds": results,
               "counts": {"template_variants": len(results),
                          "separable_variants": sum(bool(row["separable"]) for row in results.values())},
               "limitations": ["Injected positives validate appearance matching but do not prove that each pickup occurs in Video 4.",
                               "Untouched calibration frames can contain unknown real pickups; this makes the negative test conservative."],
               "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    write_json(output_path, payload)
    return payload


def match_frame(
    frame: Any, templates: Sequence[dict[str, Any]], *, frame_number: int,
    media_time_ms: int, threshold: float = 0.86, search_radius: int = 720,
    maximum_per_code: int = 12, processing_scale: float = 0.5,
    calibrated_thresholds: dict[str, float] | None = None,
) -> list[PickupDetection]:
    import cv2
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    left, top = max(0, center_x - search_radius), max(0, center_y - search_radius)
    right, bottom = min(width, center_x + search_radius), min(height, center_y + search_radius)
    crop = frame[top:bottom, left:right]
    if processing_scale != 1.0:
        crop = cv2.resize(crop, None, fx=processing_scale, fy=processing_scale, interpolation=cv2.INTER_AREA)
    found: list[PickupDetection] = []
    for template in templates:
        template_key = f"{template['code']}@{template['scale']:.4f}"
        active_threshold = calibrated_thresholds.get(template_key, 1.0) if calibrated_thresholds is not None else threshold
        if active_threshold >= 1.0:
            continue
        th, tw = template["bgr"].shape[:2]
        if th > crop.shape[0] or tw > crop.shape[1]:
            continue
        # Tight alpha bounds keep transparent padding small. The unmasked
        # coefficient path is dramatically faster than OpenCV's masked path.
        scores = cv2.matchTemplate(crop, template["bgr"], cv2.TM_CCOEFF_NORMED)
        scores[~__import__("numpy").isfinite(scores)] = 0
        for _ in range(maximum_per_code):
            _, score, _, location = cv2.minMaxLoc(scores)
            if score < active_threshold:
                break
            x = left + (location[0] + tw / 2) / processing_scale
            y = top + (location[1] + th / 2) / processing_scale
            found.append(PickupDetection(template["code"], frame_number, media_time_ms, x, y, float(score), template["scale"] / processing_scale))
            x0, y0 = max(0, location[0] - tw // 2), max(0, location[1] - th // 2)
            scores[y0:min(scores.shape[0], location[1] + th), x0:min(scores.shape[1], location[0] + tw)] = 0
    by_code: dict[str, list[PickupDetection]] = {}
    for row in found:
        by_code.setdefault(row.code, []).append(row)
    return [row for rows in by_code.values() for row in non_maximum_suppression(rows)]


def scan_video4_world_pickups(
    *, workspace_root: Path, config_path: Path, asset_dir: Path, output_dir: Path,
    start_second: float = 0.0, end_second: float | None = None, sample_fps: float = 2.0,
    threshold: float = 0.86, scales: Sequence[float] = (0.25, 0.5),
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if sha256_file(video_path) != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    processing_scale = 0.5
    templates, asset_manifest = _load_templates(asset_dir, tuple(scale * processing_scale for scale in scales))
    calibrated_thresholds: dict[str, float] | None = None
    if calibration_path is not None:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration["video_sha256"] != dataset["video"]["sha256"]:
            raise ValueError("World-pickup calibration belongs to a different video")
        calibrated_thresholds = {key: float(row["threshold"]) for key, row in calibration["thresholds"].items()}
    identity = f"{DETECTOR_VERSION}|{dataset['video']['sha256']}|{sample_fps}|{threshold}|{list(scales)}"
    run_id = f"run_world_pickups_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    frame_rows: list[list[PickupDetection]] = []
    pause_rows: list[dict[str, Any]] = []
    interrupted_times: list[int] = []
    with OpenCVVideoReader(video_path, dataset["video_asset_id"]) as reader:
        width, height = reader.metadata.width, reader.metadata.height
        for packet in reader.iter_packets(start_ms=round(start_second * 1000), end_ms=None if end_second is None else round(end_second * 1000), sample_fps=sample_fps):
            pause = gameplay_pause_evidence(packet.image)
            if pause["blocked"]:
                interrupted_times.append(packet.media_time_ms)
                pause_rows.append({"frame_number": packet.frame_number,
                                   "media_time_ms": packet.media_time_ms, **pause})
                frame_rows.append([])
            else:
                frame_rows.append(match_frame(packet.image, templates, frame_number=packet.frame_number,
                                              media_time_ms=packet.media_time_ms, threshold=threshold,
                                              search_radius=round(720 * height / 1440), processing_scale=processing_scale,
                                              calibrated_thresholds=calibrated_thresholds))
    flattened = [row for frame in frame_rows for row in frame]
    tracks = associate_tracks(frame_rows, maximum_distance=round(100 * height / 1440),
                              maximum_gap_ms=math.ceil(1250 / sample_fps),
                              interrupted_sample_times_ms=interrupted_times)
    requested_end_ms = round(end_second * 1000) if end_second is not None else dataset["duration_ms"]
    final_sample_ms = min(requested_end_ms, dataset["duration_ms"])
    sample_period_ms = math.ceil(1000 / sample_fps)
    events = resolve_pickup_events(tracks, processing_run_id=run_id,
        video_asset_id=dataset["video_asset_id"], session_id=dataset["session_id"],
        player_center=(width / 2, height / 2), final_sample_ms=final_sample_ms,
        collection_radius=150 * height / 1440, sample_period_ms=sample_period_ms,
        interrupted_sample_times_ms=interrupted_times)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_path, events_path = output_dir / "world_pickup_detections.jsonl", output_dir / "world_pickup_candidates.jsonl"
    write_jsonl(detections_path, ({**row.__dict__, "track_candidates_are_resolved_separately": True} for row in flattened))
    write_jsonl(events_path, (event.to_dict() for event in events))
    pauses_path = output_dir / "excluded_level_up_samples.jsonl"
    write_jsonl(pauses_path, pause_rows)
    by_code = {code: sum(row.code == code for row in flattened) for code in sorted({row.code for row in flattened})}
    manifest = {
        "artifact_type": "vss_framework_world_pickup_run", "framework_version": FRAMEWORK_VERSION,
        "processing_run_id": run_id, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "configuration": {"start_second": start_second, "end_second": end_second, "sample_fps": sample_fps,
                          "match_threshold": threshold, "template_scales_in_source_frame": list(scales),
                          "processing_scale": processing_scale,
                          "calibration_path": calibration_path.name if calibration_path is not None else None,
                          "candidate_rule": "persistent track disappears near player"},
        "counts": {"detections": len(flattened), "tracks": len(tracks), "pickup_candidates": len(events),
                   "excluded_level_up_samples": len(pause_rows), "detections_by_code": by_code},
        "source_integrity": {"video_sha256": dataset["video"]["sha256"],
                             "asset_manifest_sha256": sha256_file(asset_dir / "manifest.json"),
                             "detector_source_sha256": sha256_file(Path(__file__)),
                             "gameplay_state_source_sha256": sha256_file(Path(__file__).with_name("gameplay_state.py"))},
        "outputs": {"detections": {"path": detections_path.name, "sha256": sha256_file(detections_path), "row_count": len(flattened)},
                    "pickup_candidates": {"path": events_path.name, "sha256": sha256_file(events_path), "row_count": len(events)},
                    "excluded_level_up_samples": {"path": pauses_path.name, "sha256": sha256_file(pauses_path), "row_count": len(pause_rows)}},
        "policies": {"human_coded_ground_truth_used": False, "blank_means_zero": False,
                     "database_write_performed": False, "auto_accepted_pickups": False},
        "limitations": ["Template presence is not itself a pickup.",
                        "Disappearance near the assumed player center can still be caused by occlusion; candidates require review until effect corroboration is added.",
                        "Only the pinned reference sprites and configured scales are searched.",
                        "Observed level-up interruptions break tracks and cannot create disappearance candidates; boundaries between sampled frames remain uncertain."],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"},
        "reference_asset_count": len(asset_manifest["assets"]),
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
