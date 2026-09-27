"""Video-only detection of the persistent Gold Fever HUD gauge."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .hashing import sha256_file
from .gameplay_state import gameplay_pause_evidence
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision
from .video import OpenCVVideoReader

DETECTOR_VERSION = "0.4.0"


def gold_fever_overlay_features(frame: Any) -> dict[str, Any]:
    """Measure the distinctive wide gold gauge fixed to the bottom HUD edge."""
    import cv2
    import numpy as np

    image = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gold = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 38) &
            (hsv[:, :, 1] >= 130) & (hsv[:, :, 2] >= 150)).astype(np.uint8)
    bottom = gold[317:359]
    row_coverage = bottom.mean(axis=1)
    peak_index = int(np.argmax(row_coverage))
    peak = float(row_coverage[peak_index])
    # The real gauge is several adjacent pixels tall. Requiring neighboring-row
    # support rejects single decorative edges and transient gold projectiles.
    lo, hi = max(0, peak_index - 2), min(len(row_coverage), peak_index + 3)
    supported = float(np.median(row_coverage[lo:hi]))
    # The meter itself shrinks as Gold Fever runs down, so its width alone
    # cannot represent the state for the full interval.  The persistent
    # “Gold Fever!” label at bottom left remains visible through the tail.
    # Keep this local measurement separate from the wide-meter signal.
    label_gold_fraction = float(gold[335:360, 20:175].mean())
    from .detectors.gems import gameplay_hud_score

    pause = gameplay_pause_evidence(frame)
    return {
        "bottom_gold_fraction": float(bottom.mean()),
        "peak_gold_row_coverage": peak,
        "supported_gold_row_coverage": supported,
        "peak_row_y_360": float(317 + peak_index),
        "gold_fever_label_gold_fraction": label_gold_fraction,
        "gameplay_hud_score": float(gameplay_hud_score(frame)),
        "gameplay_paused": bool(pause["blocked"]),
        "gameplay_pause": pause,
    }


def mark_gold_fever_rows(rows: list[dict[str, Any]], *, minimum_peak: float = 0.60,
                         minimum_supported: float = 0.42,
                         minimum_bottom_fraction: float = 0.06,
                         minimum_label_fraction: float = 0.02) -> None:
    for row in rows:
        row["gold_fever_gauge_like"] = (
            float(row["peak_gold_row_coverage"]) >= minimum_peak
            and float(row["supported_gold_row_coverage"]) >= minimum_supported
            and float(row["bottom_gold_fraction"]) >= minimum_bottom_fraction
            and float(row["peak_row_y_360"]) >= 344
        )
        row["gold_fever_label_like"] = (
            float(row.get("gold_fever_label_gold_fraction", 0.0)) >= minimum_label_fraction
        )
        row["gold_fever_visible_raw"] = row["gold_fever_gauge_like"] or row["gold_fever_label_like"]
        # A frozen gauge can remain visible behind a menu; its pixels remain
        # available for review but cannot establish active effect duration.
        row["gold_fever_like"] = (
            row["gold_fever_visible_raw"]
            and not row.get("gameplay_paused", False)
            and float(row.get("gameplay_hud_score", 1.0)) >= 0.90
        )


def resolve_gold_fever_intervals(rows: Sequence[dict[str, Any]], *, processing_run_id: str,
                                 video_asset_id: str, session_id: str,
                                 sample_period_ms: int, minimum_duration_ms: int = 1000,
                                 bridge_gap_ms: int = 750) -> list[CanonicalEvent]:
    groups: list[list[dict[str, Any]]] = []
    interrupted = False
    for row in rows:
        if row.get("gameplay_paused"):
            interrupted = True
            continue
        if not row.get("gold_fever_like"):
            continue
        if interrupted or not groups or int(row["media_time_ms"]) - int(groups[-1][-1]["media_time_ms"]) > bridge_gap_ms:
            groups.append([row])
        else:
            groups[-1].append(row)
        interrupted = False
    events: list[CanonicalEvent] = []
    for index, group in enumerate(groups, start=1):
        start = int(group[0]["media_time_ms"])
        end = int(group[-1]["media_time_ms"]) + sample_period_ms
        # A wide gold HUD element can occasionally resemble the meter.  A
        # publishable Gold Fever interval must also contain the fixed label;
        # this avoids promoting meter-only false positives while allowing the
        # label to carry the interval after the meter has depleted.
        if (end - start < minimum_duration_ms
                or not any(row.get("gold_fever_label_like") for row in group)):
            continue
        events.append(CanonicalEvent(
            event_id=f"vss_gold_fever_{index}_{start}", video_asset_id=video_asset_id,
            session_id=session_id, event_family="effect", event_type="gold_fever_effect_candidate",
            time_lower_ms=start, time_upper_ms=end, anchor_time_ms=start,
            temporal_precision=TemporalPrecision.INTERVAL, evidence_grade=EvidenceGrade.B,
            publication_status=PublicationStatus.NEEDS_REVIEW,
            inference_method="persistent_bottom_gold_fever_hud_gauge",
            processing_run_id=processing_run_id,
            evidence=(EvidenceReference(source_artifact="gold_fever_signals.jsonl",
                source_record_key=f"rows:{group[0]['frame_number']}-{group[-1]['frame_number']}",
                modalities=("hud_overlay", "color_geometry"),
                details={"sample_count": len(group),
                         "median_peak_gold_row_coverage": median(float(row["peak_gold_row_coverage"]) for row in group)}),),
            attributes={"gold_fever_state_observed": True, "gilded_clover_confirmed": False,
                        "level_up_pause_samples_excluded": True, "imputed": False},
        ))
    return events


def scan_video4_gold_fever(*, workspace_root: Path, config_path: Path, output_dir: Path,
                           sample_fps: float = 4.0, start_second: float = 0.0,
                           end_second: float | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if sha256_file(video_path) != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    rows: list[dict[str, Any]] = []
    with OpenCVVideoReader(video_path, dataset["video_asset_id"]) as reader:
        for packet in reader.iter_packets(start_ms=round(start_second * 1000),
                                          end_ms=None if end_second is None else round(end_second * 1000),
                                          sample_fps=sample_fps):
            rows.append({"frame_number": packet.frame_number, "media_time_ms": packet.media_time_ms,
                         **gold_fever_overlay_features(packet.image)})
    mark_gold_fever_rows(rows)
    sample_period_ms = math.ceil(1000 / sample_fps)
    identity = f"{DETECTOR_VERSION}|{dataset['video']['sha256']}|{sample_fps}|{start_second}|{end_second}"
    run_id = f"run_gold_fever_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    events = resolve_gold_fever_intervals(rows, processing_run_id=run_id,
        video_asset_id=dataset["video_asset_id"], session_id=dataset["session_id"],
        sample_period_ms=sample_period_ms)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path = output_dir / "gold_fever_signals.jsonl"
    events_path = output_dir / "gold_fever_effect_candidates.jsonl"
    write_jsonl(signals_path, rows)
    write_jsonl(events_path, (event.to_dict() for event in events))
    manifest = {"artifact_type": "vss_framework_gold_fever_run", "framework_version": "0.12.0",
        "detector_version": DETECTOR_VERSION, "processing_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "prepared_by": "Tahereh Fahi",
        "configuration": {"sample_fps": sample_fps, "start_second": start_second,
                          "end_second": end_second, "minimum_duration_ms": 1000,
                          "bridge_gap_ms": 750, "minimum_peak_row_coverage": 0.60,
                          "minimum_supported_row_coverage": 0.42},
        "counts": {"sampled_frames": len(rows),
                   "excluded_level_up_samples": sum(bool(row.get("gameplay_paused")) for row in rows),
                   "excluded_non_gameplay_hud_samples": sum(
                       float(row.get("gameplay_hud_score", 1.0)) < 0.90 for row in rows
                   ),
                   "gold_fever_like_samples": sum(bool(row["gold_fever_like"]) for row in rows),
                   "gold_fever_effect_candidates": len(events)},
        "source_integrity": {"video_sha256": dataset["video"]["sha256"],
                             "detector_source_sha256": sha256_file(Path(__file__)),
                             "gameplay_state_source_sha256": sha256_file(Path(__file__).with_name("gameplay_state.py")),
                             "gameplay_hud_source_sha256": sha256_file(Path(__file__).with_name("detectors") / "gems.py")},
        "outputs": {"signals": {"path": signals_path.name, "sha256": sha256_file(signals_path), "row_count": len(rows)},
                    "candidates": {"path": events_path.name, "sha256": sha256_file(events_path), "row_count": len(events)}},
        "policies": {"human_coded_ground_truth_used": False, "imputation_performed": False,
                     "gilded_clover_identity_auto_accepted": False, "database_write_performed": False},
        "limitations": ["The detector observes the Gold Fever HUD state, not the pickup that caused it.",
                        "Short or heavily occluded gauge appearances may be missed.",
                        "Frames without a confirmed gameplay XP-bar HUD are excluded.",
                        "A visible gauge behind a level-up menu is retained as raw evidence and excluded from active effect intervals; an interruption can split one effect into multiple observed intervals."],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
