"""Video-only freeze-effect candidates for possible Orologion use."""

from __future__ import annotations

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


def aligned_motion_features(previous: Any, current: Any) -> dict[str, float]:
    """Measure residual motion after translation-only camera compensation."""
    import cv2
    import numpy as np
    size = (320, 180)
    prior = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    now = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    prior_gray = cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY).astype(np.float32)
    now_gray = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift, response = cv2.phaseCorrelate(prior_gray, now_gray)
    transform = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
    aligned = cv2.warpAffine(now, transform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    difference = cv2.absdiff(prior, aligned)
    magnitude = difference.max(axis=2)
    yy, xx = np.ogrid[:size[1], :size[0]]
    # Broadcasting both terms in one expression creates the full 180x320 mask.
    gameplay = ((yy >= 22) & (yy < 170) &
                ((xx - 160) ** 2 + (yy - 95) ** 2 >= 28 ** 2))
    values = magnitude[gameplay]
    hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
    pale = (((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 185)) |
            ((hsv[:, :, 0] >= 78) & (hsv[:, :, 0] <= 105) &
             (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 145)))
    return {"residual_motion_fraction": float((values >= 20).mean()),
            "residual_motion_median": float(np.median(values)),
            "pale_cyan_fraction": float(pale[gameplay].mean()),
            "camera_shift_x": float(shift[0]), "camera_shift_y": float(shift[1]),
            "alignment_response": float(response)}


def mark_freeze_like_rows(rows: list[dict[str, Any]], *, baseline_seconds: float,
                          sample_fps: float, motion_ratio: float = 0.42,
                          minimum_tint_lift: float = 0.008) -> None:
    history: deque[dict[str, Any]] = deque(maxlen=max(4, round(baseline_seconds * sample_fps)))
    for row in rows:
        if not row.get("observable", False):
            history.clear()
            row["freeze_like"] = False
            continue
        if len(history) < max(3, round(2 * sample_fps)):
            row["freeze_like"] = False
        else:
            motion_baseline = median(float(item["residual_motion_fraction"]) for item in history)
            tint_baseline = median(float(item["pale_cyan_fraction"]) for item in history)
            row["motion_baseline"] = motion_baseline
            row["tint_baseline"] = tint_baseline
            row["motion_ratio"] = float(row["residual_motion_fraction"]) / max(motion_baseline, 1e-6)
            row["tint_lift"] = float(row["pale_cyan_fraction"]) - tint_baseline
            row["freeze_like"] = (motion_baseline >= .015 and row["motion_ratio"] <= motion_ratio
                                  and row["tint_lift"] >= minimum_tint_lift
                                  and float(row["alignment_response"]) >= .05)
        history.append(row)


def resolve_freeze_intervals(rows: Sequence[dict[str, Any]], *, processing_run_id: str,
                             video_asset_id: str, session_id: str,
                             sample_period_ms: int, minimum_duration_ms: int = 1500) -> list[CanonicalEvent]:
    groups: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] = []
    for row in rows:
        contiguous = active and int(row["media_time_ms"]) - int(active[-1]["media_time_ms"]) <= sample_period_ms * 1.5
        if row.get("freeze_like") and (not active or contiguous):
            active.append(row)
        else:
            if active:
                groups.append(active)
            active = [row] if row.get("freeze_like") else []
    if active:
        groups.append(active)
    events: list[CanonicalEvent] = []
    for index, group in enumerate(groups, start=1):
        start = int(group[0]["media_time_ms"])
        end = int(group[-1]["media_time_ms"]) + sample_period_ms
        if end - start < minimum_duration_ms:
            continue
        events.append(CanonicalEvent(
            event_id=f"vss_freeze_effect_{index}_{start}", video_asset_id=video_asset_id,
            session_id=session_id, event_family="effect", event_type="freeze_effect_candidate",
            time_lower_ms=start, time_upper_ms=end, anchor_time_ms=start,
            temporal_precision=TemporalPrecision.INTERVAL, evidence_grade=EvidenceGrade.C,
            publication_status=PublicationStatus.NEEDS_REVIEW,
            inference_method="camera_compensated_motion_suppression_plus_frozen_tint",
            processing_run_id=processing_run_id,
            evidence=(EvidenceReference(source_artifact="freeze_signals.jsonl",
                source_record_key=f"rows:{group[0]['frame_number']}-{group[-1]['frame_number']}",
                modalities=("frame_motion", "color_state"),
                details={"sample_count": len(group),
                         "median_motion_ratio": median(float(row["motion_ratio"]) for row in group),
                         "median_tint_lift": median(float(row["tint_lift"]) for row in group)}),),
            attributes={"compatible_pickup_types": ["orologion"], "orologion_confirmed": False,
                        "enemy_identity_not_tracked": True, "imputed": False},
        ))
    return events


def scan_video4_freeze_effect(*, workspace_root: Path, config_path: Path, output_dir: Path,
                              sample_fps: float = 4.0, baseline_seconds: float = 8.0) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if sha256_file(video_path) != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    rows: list[dict[str, Any]] = []
    previous = None
    with OpenCVVideoReader(video_path, dataset["video_asset_id"]) as reader:
        for packet in reader.iter_packets(sample_fps=sample_fps):
            pause = gameplay_pause_evidence(packet.image)
            menu = bool(pause["blocked"])
            if previous is None or menu:
                row = {"frame_number": packet.frame_number, "media_time_ms": packet.media_time_ms,
                       "observable": False, "reason": pause["reason"] if menu else "no_contiguous_gameplay_pair"}
            else:
                row = {"frame_number": packet.frame_number, "media_time_ms": packet.media_time_ms,
                       "observable": True, **aligned_motion_features(previous, packet.image)}
            row["gameplay_pause"] = pause
            rows.append(row)
            previous = None if menu else packet.image
    mark_freeze_like_rows(rows, baseline_seconds=baseline_seconds, sample_fps=sample_fps)
    sample_period_ms = math.ceil(1000 / sample_fps)
    identity = f"{DETECTOR_VERSION}|{dataset['video']['sha256']}|{sample_fps}|{baseline_seconds}"
    run_id = f"run_freeze_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    events = resolve_freeze_intervals(rows, processing_run_id=run_id,
        video_asset_id=dataset["video_asset_id"], session_id=dataset["session_id"],
        sample_period_ms=sample_period_ms)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path, events_path = output_dir / "freeze_signals.jsonl", output_dir / "freeze_effect_candidates.jsonl"
    write_jsonl(signals_path, rows)
    write_jsonl(events_path, (event.to_dict() for event in events))
    manifest = {"artifact_type": "vss_framework_freeze_effect_run", "framework_version": "0.10.0",
        "detector_version": DETECTOR_VERSION, "processing_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "prepared_by": "Tahereh Fahi",
        "configuration": {"sample_fps": sample_fps, "baseline_seconds": baseline_seconds,
                          "minimum_duration_ms": 1500, "camera_compensation": "translation_phase_correlation"},
        "counts": {"sampled_frames": len(rows), "observable_pairs": sum(bool(row.get("observable")) for row in rows),
                   "freeze_like_samples": sum(bool(row.get("freeze_like")) for row in rows),
                   "freeze_effect_candidates": len(events)},
        "source_integrity": {"video_sha256": dataset["video"]["sha256"],
                             "detector_source_sha256": sha256_file(Path(__file__)),
                             "gameplay_state_source_sha256": sha256_file(Path(__file__).with_name("gameplay_state.py"))},
        "outputs": {"signals": {"path": signals_path.name, "sha256": sha256_file(signals_path), "row_count": len(rows)},
                    "candidates": {"path": events_path.name, "sha256": sha256_file(events_path), "row_count": len(events)}},
        "policies": {"human_coded_ground_truth_used": False, "imputation_performed": False,
                     "orologion_identity_auto_accepted": False, "database_write_performed": False},
        "limitations": ["The detector observes a freeze-like visual state, not the Orologion pickup itself.",
                        "Player weapons and dense effects can obscure enemy motion; enemy identities are not yet tracked."],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
