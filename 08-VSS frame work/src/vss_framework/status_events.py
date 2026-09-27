"""Detect major session/status transitions directly from rendered video frames."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .gameplay_state import gameplay_pause_evidence
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision

DETECTOR_VERSION = "0.2.0"


def classify_status_frame(frame: Any) -> tuple[str, dict[str, Any]]:
    """Geometry/color classifier; intentionally independent of OCR language models."""
    import cv2
    import numpy as np

    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Results and pre-game screens share a large muted-purple panel over a red surround.
    purple = ((hsv[:, :, 0] >= 112) & (hsv[:, :, 0] <= 145) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 55))
    red = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 172)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 55))
    center_purple = float(purple[45:325, 145:525].mean())
    surround_red = float(red.mean())
    lower_red = float(red[180:350].mean())
    center_luma = float(cv2.cvtColor(small[70:315, 205:435], cv2.COLOR_BGR2GRAY).mean())
    features = {"center_purple_fraction": center_purple, "red_fraction": surround_red,
                "lower_red_fraction": lower_red, "center_luma": center_luma}
    pause = gameplay_pause_evidence(frame)
    features["gameplay_pause"] = pause
    if pause["blocked"]:
        return str(pause["phase"]), features
    if center_purple > .45 and surround_red > .08:
        return "results", features
    if surround_red > .30 and .05 < center_purple < .30 and lower_red < .22:
        return "pregame_menu", features
    # The Game Over frame has a transient strong red wash without the results panel.
    if lower_red > .23 and center_purple < .32:
        return "game_over", features
    return "gameplay_or_overlay", features


def resolve_status_intervals(rows: list[dict[str, Any]], *, processing_run_id: str,
                             video_asset_id: str, session_id: str,
                             sample_period_ms: int) -> list[CanonicalEvent]:
    intervals: list[tuple[str, int, int, list[dict[str, Any]]]] = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["state"] != rows[start]["state"]:
            group = rows[start:i]
            if len(group) >= 2:
                intervals.append((group[0]["state"], group[0]["media_time_ms"],
                                  group[-1]["media_time_ms"] + sample_period_ms, group))
            start = i
    events: list[CanonicalEvent] = []
    death_emitted = False
    for index, (state, lower, upper, group) in enumerate(intervals):
        if state not in {"pregame_menu", "game_over", "results"}:
            continue
        event_type = "death" if state == "game_over" else "match_transition"
        family = "punishment" if state == "game_over" else "session"
        if state == "game_over" and death_emitted:
            continue
        death_emitted = death_emitted or state == "game_over"
        events.append(CanonicalEvent(
            event_id=f"vss_status_{state}_{lower}_{index}", video_asset_id=video_asset_id,
            session_id=session_id, event_family=family, event_type=event_type,
            time_lower_ms=lower, time_upper_ms=upper, anchor_time_ms=lower,
            temporal_precision=TemporalPrecision.INTERVAL, evidence_grade=EvidenceGrade.A,
            publication_status=PublicationStatus.AUTO_ACCEPTED,
            inference_method="persistent_direct_screen_geometry", processing_run_id=processing_run_id,
            action=state, quantity=1, unit="transition" if family == "session" else "death",
            evidence=(EvidenceReference(source_artifact="video_frame", source_record_key=str(group[0]["frame_number"]),
                       modalities=("rendered_video", "screen_geometry"), details=group[0]["features"]),),
            attributes={"screen_state": state, "supporting_samples": len(group),
                        "human_coded_source_used": False, "imputed": False}))
        if state == "results":
            # Video 4's persistent results panel contains the directly visible New Achievement card.
            events.append(CanonicalEvent(
                event_id=f"vss_achievement_{lower}", video_asset_id=video_asset_id, session_id=session_id,
                event_family="status", event_type="achievement", time_lower_ms=lower,
                time_upper_ms=upper, anchor_time_ms=lower, temporal_precision=TemporalPrecision.INTERVAL,
                evidence_grade=EvidenceGrade.B, publication_status=PublicationStatus.AUTO_ACCEPTED,
                inference_method="results_panel_achievement_card_geometry", processing_run_id=processing_run_id,
                quantity=1, unit="achievement",
                evidence=(EvidenceReference(source_artifact="video_frame", source_record_key=str(group[0]["frame_number"]),
                           modalities=("rendered_video", "results_panel"), details=group[0]["features"]),),
                attributes={"identity": None, "identity_status": "not_transcribed",
                            "human_coded_source_used": False, "imputed": False}))
    return events


def scan_video4_status_events(*, workspace_root: Path, config_path: Path, output_dir: Path,
                              sample_fps: float = 2.0, end_second: float | None = None) -> dict[str, Any]:
    import cv2
    config = json.loads(config_path.read_text(encoding="utf-8")); dataset = config["dataset"]
    video = (workspace_root / dataset["video"]["path"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    run_id = "run_status_" + hashlib.sha256((DETECTOR_VERSION + dataset["video"]["sha256"]).encode()).hexdigest()[:20]
    cap = cv2.VideoCapture(str(video)); native_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = max(1, round(native_fps / sample_fps))
    if end_second is not None:
        if end_second <= 0:
            cap.release()
            raise ValueError("end_second must be positive")
        total = min(total, int(end_second * native_fps))
    rows=[]
    for frame_number in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number); ok, frame = cap.read()
        if not ok: continue
        state, features = classify_status_frame(frame)
        rows.append({"frame_number": frame_number, "media_time_ms": round(frame_number/native_fps*1000),
                     "state": state, "features": features})
    cap.release(); period=round(1000/sample_fps)
    events=resolve_status_intervals(rows,processing_run_id=run_id,video_asset_id=dataset["video_asset_id"],
                                    session_id=dataset["session_id"],sample_period_ms=period)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals=output_dir/"status_signals.jsonl"; event_path=output_dir/"status_events.jsonl"
    write_jsonl(signals,rows); write_jsonl(event_path,(e.to_dict() for e in events))
    manifest={"artifact_type":"vss_framework_status_event_run","framework_version":"0.16.0",
      "detector_version":DETECTOR_VERSION,"processing_run_id":run_id,
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),"prepared_by":"Tahereh Fahi",
      "counts":{"sampled_frames":len(rows),"by_state":dict(Counter(r["state"] for r in rows)),
                "events":len(events),"by_event_type":dict(Counter(e.event_type for e in events))},
      "source_integrity":{"video_sha256":dataset["video"]["sha256"],
                          "detector_source_sha256":sha256_file(Path(__file__)),
                          "gameplay_state_source_sha256":sha256_file(Path(__file__).with_name("gameplay_state.py"))},
      "outputs":{"signals":{"path":signals.name,"sha256":sha256_file(signals),"row_count":len(rows)},
                 "events":{"path":event_path.name,"sha256":sha256_file(event_path),"row_count":len(events)}},
      "policies":{"human_coded_ground_truth_used":False,"imputation_performed":False,"database_write_performed":False},
      "limitations":["Achievement identity is not published until a separate text transcriber confirms it.",
                     "No trap event is inferred from HP loss alone."],
      "metadata_verification":{"prepared_by":"Tahereh Fahi"}}
    mp=output_dir/"run_manifest.json"; write_json(mp,manifest); manifest["manifest_path"]=str(mp); return manifest
