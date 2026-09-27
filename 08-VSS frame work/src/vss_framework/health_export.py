"""Export an explicitly selected, integrity-checked health run for the dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .health_provenance import verified_outputs
from .io import iter_jsonl


def build_health_payload(health_dir: Path, attribution_dir: Path) -> dict[str, Any]:
    health, hp = verified_outputs(health_dir)
    attribution, ap = verified_outputs(attribution_dir)
    expected = {
        "health_run_manifest_sha256": sha256_file(health_dir / "run_manifest.json"),
        "health_observations_sha256": sha256_file(hp["health_observations"]),
        "health_events_sha256": sha256_file(hp["health_events"]),
    }
    if any(attribution.get("sources", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Attribution does not match the selected health run")
    observations = list(iter_jsonl(hp["health_observations"]))
    events = list(iter_jsonl(ap["attributed_health_events"]))
    if not observations:
        raise ValueError("No sampled health observations to export")
    return {
        "schemaVersion": "mnl-video4-health-v1", "preparedBy": "Tahereh Fahi",
        "publicationReady": False,
        "source": {
            "videoAssetId": observations[0]["video_asset_id"],
            "durationMs": max(r["time_upper_ms"] for r in observations),
            "healthProcessingRunId": health["processing_run_id"],
            "attributionProcessingRunId": attribution["processing_run_id"],
            "healthManifestSha256": expected["health_run_manifest_sha256"],
            "attributionManifestSha256": sha256_file(attribution_dir / "run_manifest.json"),
            "inventoryEventsSha256": attribution["sources"]["inventory_events_sha256"],
            "automationIntegrity": "not_certified", "accuracy": "not_validated",
            "imputationPerformed": False,
        },
        "calibration": health["calibration"],
        "summary": {**health["counts"], "recoveryAttributions": attribution["counts"]["recovery_attributions"]},
        "observations": [{
            "mediaTimeMs": r["time_lower_ms"], "healthPercent": r["numeric_value"],
            "observed": r["observed"], "confidence": r["attributes"]["confidence"],
            "rawWidthPx": r["attributes"]["raw_red_fill_width_1440p_px"],
            "rejectionReason": r["attributes"].get("rejection_reason"),
            "excludedFromGameplay": r["attributes"].get("excluded_from_gameplay"),
            "gameplaySegmentId": r["attributes"].get("gameplay_segment_id"),
            "screenState": r["attributes"].get("screen_state"),
        } for r in observations],
        "events": [{
            "eventId": r["event_id"], "eventType": r["event_type"],
            "startMs": r["time_lower_ms"], "endMs": r["time_upper_ms"],
            "mediaTimeMs": r["anchor_time_ms"], "quantity": r["quantity"],
            "evidenceGrade": r["evidence_grade"], "publicationStatus": r["publication_status"],
            "previousPercent": r["evidence"][0]["details"]["previous_percent"],
            "currentPercent": r["evidence"][0]["details"]["current_percent"],
            "attribution": r.get("attributes", {}).get("attribution"),
            "evidence": r["evidence"], "processingRunId": r["processing_run_id"],
            "inferenceMethod": r["inference_method"],
        } for r in events],
    }
