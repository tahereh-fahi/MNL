"""Direct visual detection of Treasure Chest animation intervals and tiers."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .hashing import sha256_file
from .gameplay_state import level_up_pause_evidence
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision
from .video import OpenCVVideoReader

DETECTOR_VERSION = "0.2.0"


def chest_overlay_features(frame: Any) -> dict[str, Any]:
    """Measure the chest panel, central beam, and revealed reward orbs."""
    import cv2
    import numpy as np

    image = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    center = hsv[40:315, 300:340]
    left = hsv[40:315, 225:285]
    right = hsv[40:315, 355:415]
    sides = np.concatenate((left.reshape(-1, 3), right.reshape(-1, 3)))
    center_beam = ((center[:, :, 0] >= 110) & (center[:, :, 0] <= 145) &
                   (center[:, :, 1] >= 80) & (center[:, :, 2] >= 70))
    purple_panel = ((sides[:, 0] >= 108) & (sides[:, 0] <= 145) &
                    (sides[:, 1] >= 45) & (sides[:, 2] >= 65))

    reward_region = hsv[45:240, 215:425]
    red_orange = ((((reward_region[:, :, 0] <= 15) | (reward_region[:, :, 0] >= 170)) &
                   (reward_region[:, :, 1] >= 120) & (reward_region[:, :, 2] >= 130))).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(red_orange)
    reward_orbs = 0
    for _, _, width, height, area in stats[1:count]:
        if 300 <= area <= 900 and 22 <= width <= 36 and 22 <= height <= 36:
            reward_orbs += 1
    pause = level_up_pause_evidence(frame)
    return {"center_beam_fraction": float(center_beam.mean()),
            "purple_panel_fraction": float(purple_panel.mean()),
            "reward_orb_count": reward_orbs,
            "level_up_pause": bool(pause["blocked"]), "gameplay_pause": pause}


def mark_chest_rows(rows: list[dict[str, Any]], *, minimum_center_beam: float = .35,
                    minimum_panel: float = .45) -> None:
    for row in rows:
        row["chest_overlay_like"] = (not row.get("level_up_pause", False)
                                     and float(row["center_beam_fraction"]) >= minimum_center_beam
                                     and float(row["purple_panel_fraction"]) >= minimum_panel)


def resolve_chest_intervals(rows: Sequence[dict[str, Any]], *, processing_run_id: str,
                            video_asset_id: str, session_id: str, sample_period_ms: int,
                            minimum_duration_ms: int = 1500, bridge_gap_ms: int = 750) -> list[CanonicalEvent]:
    groups: list[list[dict[str, Any]]] = []
    interrupted = False
    for row in rows:
        if row.get("level_up_pause"):
            interrupted = True
            continue
        if not row.get("chest_overlay_like"):
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
        if end - start < minimum_duration_ms:
            continue
        reward_count = max(int(row["reward_orb_count"]) for row in group)
        tier = {1: 1, 3: 2, 5: 3}.get(reward_count)
        # The panel geometry alone also occurs in other animated overlays.
        # Publish a chest only after its characteristic 1/3/5 reward reveal.
        if tier is None:
            continue
        event_type = f"loot_box_tier_{tier}"
        events.append(CanonicalEvent(
            event_id=f"vss_chest_{index}_{start}", video_asset_id=video_asset_id,
            session_id=session_id, event_family="chest", event_type=event_type,
            time_lower_ms=start, time_upper_ms=end, anchor_time_ms=start,
            temporal_precision=TemporalPrecision.INTERVAL,
            evidence_grade=EvidenceGrade.A,
            publication_status=PublicationStatus.AUTO_ACCEPTED,
            inference_method="persistent_chest_panel_plus_reward_orb_count",
            processing_run_id=processing_run_id, quantity=reward_count or None, unit="revealed_rewards",
            evidence=(EvidenceReference(source_artifact="chest_signals.jsonl",
                source_record_key=f"rows:{group[0]['frame_number']}-{group[-1]['frame_number']}",
                modalities=("chest_panel", "reward_reveal"),
                details={"sample_count": len(group), "maximum_reward_orb_count": reward_count,
                         "median_center_beam_fraction": median(float(row["center_beam_fraction"]) for row in group)}),),
            attributes={"tier": tier, "reward_count": reward_count or None,
                        "reward_identities_resolved": False, "imputed": False},
        ))
    return events


def scan_video4_chests(*, workspace_root: Path, config_path: Path, output_dir: Path,
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
                         **chest_overlay_features(packet.image)})
    mark_chest_rows(rows)
    sample_period_ms = math.ceil(1000 / sample_fps)
    identity = f"{DETECTOR_VERSION}|{dataset['video']['sha256']}|{sample_fps}|{start_second}|{end_second}"
    run_id = f"run_chests_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    events = resolve_chest_intervals(rows, processing_run_id=run_id,
        video_asset_id=dataset["video_asset_id"], session_id=dataset["session_id"],
        sample_period_ms=sample_period_ms)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path, events_path = output_dir / "chest_signals.jsonl", output_dir / "chest_events.jsonl"
    write_jsonl(signals_path, rows)
    write_jsonl(events_path, (event.to_dict() for event in events))
    by_type: dict[str, int] = {}
    for event in events:
        by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
    manifest = {"artifact_type": "vss_framework_chest_lifecycle_run", "framework_version": "0.13.0",
        "detector_version": DETECTOR_VERSION, "processing_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "prepared_by": "Tahereh Fahi",
        "configuration": {"sample_fps": sample_fps, "start_second": start_second, "end_second": end_second,
                          "minimum_duration_ms": 1500, "bridge_gap_ms": 750},
        "counts": {"sampled_frames": len(rows), "chest_overlay_samples": sum(bool(row["chest_overlay_like"]) for row in rows),
                   "excluded_level_up_samples": sum(bool(row.get("level_up_pause")) for row in rows),
                   "chest_intervals": len(events), "by_event_type": by_type},
        "source_integrity": {"video_sha256": dataset["video"]["sha256"],
                             "detector_source_sha256": sha256_file(Path(__file__)),
                             "gameplay_state_source_sha256": sha256_file(Path(__file__).with_name("gameplay_state.py"))},
        "outputs": {"signals": {"path": signals_path.name, "sha256": sha256_file(signals_path), "row_count": len(rows)},
                    "events": {"path": events_path.name, "sha256": sha256_file(events_path), "row_count": len(events)}},
        "policies": {"human_coded_ground_truth_used": False, "imputation_performed": False,
                     "reward_identity_inferred": False, "database_write_performed": False},
        "limitations": ["Reward count is derived from simultaneously visible reward orbs; reward identities require a separate detector.",
                        "A chest whose 1/3/5 reward reveal is never observable is not published, preventing panel-only false positives."],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
