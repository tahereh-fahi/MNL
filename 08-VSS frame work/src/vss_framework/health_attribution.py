"""Attribute automated HP recovery events without inventing causal evidence."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .hashing import sha256_file
from .io import iter_jsonl, write_json, write_jsonl
from .health_provenance import implementation_receipt, verified_outputs, verify_inventory


ATTRIBUTION_VERSION = "0.2.0"


def attribute_recovery_event(event: dict[str, Any], inventory_events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return a cautious attribution; temporal compatibility is not causality."""
    anchor = int(event["anchor_time_ms"])
    quantity = float(event.get("quantity") or 0)
    floor_chickens = [row for row in inventory_events if row.get("event_type") == "floor_chicken"]
    prior_chicken = [row for row in floor_chickens if 0 <= anchor - int(row["anchor_time_ms"]) <= 5_000]
    if prior_chicken:
        source = min(prior_chicken, key=lambda row: anchor - int(row["anchor_time_ms"]))
        return {
            "cause_code": "floor_chicken_temporal_match",
            "cause_label": "Floor Chicken",
            "causal_status": "strong_temporal_candidate",
            "evidence_grade": source.get("evidence_grade", "unresolved"),
            "latency_ms": anchor - int(source["anchor_time_ms"]),
            "source_event_id": source["event_id"],
            "explanation": "Automated Floor Chicken selection precedes the persistent HP increase by no more than five seconds.",
        }

    pummarola = [
        row for row in inventory_events
        if row.get("item_name") == "Pummarola"
        and row.get("event_type") in {"new_passive_item", "passive_item_upgrade"}
        and int(row["anchor_time_ms"]) <= anchor
    ]
    if pummarola and quantity <= 8:
        source = max(pummarola, key=lambda row: int(row["anchor_time_ms"]))
        return {
            "cause_code": "pummarola_compatible_regeneration",
            "cause_label": "Regeneration compatible with Pummarola",
            "causal_status": "compatible_not_confirmed",
            "evidence_grade": "B",
            "latency_ms": anchor - int(source["anchor_time_ms"]),
            "source_event_id": source["event_id"],
            "explanation": "Pummarola is already present and the increase is small, but the video does not isolate Pummarola from other Recovery sources.",
        }

    nearby_menu = [
        row for row in inventory_events
        if row.get("event_type") == "level_up_transaction"
        and abs(anchor - int(row["anchor_time_ms"])) <= 2_000
    ]
    if nearby_menu:
        source = min(nearby_menu, key=lambda row: abs(anchor - int(row["anchor_time_ms"])))
        return {
            "cause_code": "unresolved_near_level_up_menu",
            "cause_label": "Unresolved near Level-Up menu",
            "causal_status": "temporal_overlap_only",
            "evidence_grade": "unresolved",
            "latency_ms": anchor - int(source["anchor_time_ms"]),
            "source_event_id": source["event_id"],
            "explanation": "A Level-Up menu is nearby in media time, but a menu is not evidence of healing.",
        }

    if quantity <= 8:
        return {
            "cause_code": "gradual_regeneration_candidate",
            "cause_label": "Gradual regeneration candidate",
            "causal_status": "pattern_only",
            "evidence_grade": "C",
            "latency_ms": None,
            "source_event_id": None,
            "explanation": "The persistent increase is small and compatible with regeneration, but no automated source event identifies its cause.",
        }
    return {
        "cause_code": "unresolved_recovery_source",
        "cause_label": "Unresolved recovery source",
        "causal_status": "unresolved",
        "evidence_grade": "unresolved",
        "latency_ms": None,
        "source_event_id": None,
        "explanation": "No sufficiently close automated reward or inventory evidence explains this HP increase.",
    }


def attribute_video4_health(*, health_run_dir: Path, inventory_events_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    health_manifest_path = health_run_dir / "run_manifest.json"
    health_manifest, health_outputs = verified_outputs(health_run_dir)
    inventory_manifest = verify_inventory(inventory_events_path)
    provenance = health_manifest.get("provenance", {})
    if not provenance.get("video_asset_id") or not provenance.get("code_sha256"):
        raise ValueError("Fresh health provenance is required; rerun health calibration")
    if inventory_manifest.get("inputs", {}).get("video", {}).get("sha256") != health_manifest["source_integrity"]["video_sha256"]:
        raise ValueError("Health and inventory must refer to the same video hash")
    observations_path = health_outputs["health_observations"]
    events_path = health_outputs["health_events"]
    inventory_events = list(iter_jsonl(inventory_events_path))
    events = list(iter_jsonl(events_path))
    for row in [*inventory_events, *events]:
        if row.get("video_asset_id") != provenance["video_asset_id"] or row.get("session_id") != provenance["session_id"]:
            raise ValueError("Health/inventory video or session mismatch")
    # Uncertain upstream inventory cannot become confident causal evidence.
    eligible_inventory = [row for row in inventory_events
                          if row.get("publication_status") == "auto_accepted"
                          and row.get("evidence_grade") in {"A", "B"}]
    enriched = []
    for event in events:
        row = dict(event)
        if row["event_type"] == "hp_recovery":
            row["attributes"] = {**row.get("attributes", {}), "attribution": attribute_recovery_event(row, eligible_inventory)}
        enriched.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "attributed_health_events.jsonl"
    write_jsonl(output_path, enriched)
    attribution_counts = Counter(
        row["attributes"]["attribution"]["cause_code"]
        for row in enriched if row["event_type"] == "hp_recovery"
    )
    receipt = implementation_receipt()
    identity = json.dumps({"version": ATTRIBUTION_VERSION, "health": sha256_file(events_path),
                           "inventory": sha256_file(inventory_events_path), "implementation": receipt}, sort_keys=True)
    manifest = {
        "artifact_type": "vss_framework_health_attribution_run",
        "framework_version": "0.8.0",
        "processing_run_id": f"run_health_attribution_{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "provenance": {**receipt, "inventory_manifest_sha256": sha256_file(inventory_events_path.parent / "run_manifest.json"),
                       "inventory_context": "supplied_verified_producer_output",
                       "inventory_origin_certified": False},
        "run_scope": health_manifest.get("run_scope", "unknown"),
        "publication_ready": False,
        "execution_seconds": time.monotonic() - started,
        "prompt_workflow_elapsed_seconds": None,
        "counts": {"health_events": len(enriched), "recovery_events": sum(row["event_type"] == "hp_recovery" for row in enriched), "recovery_attributions": dict(sorted(attribution_counts.items()))},
        "sources": {
            "health_run_manifest_sha256": sha256_file(health_manifest_path),
            "health_observations_sha256": sha256_file(observations_path),
            "health_events_sha256": sha256_file(events_path),
            "inventory_events_sha256": sha256_file(inventory_events_path),
        },
        "outputs": {"attributed_health_events": {"path": output_path.name, "sha256": sha256_file(output_path), "row_count": len(enriched)}},
        "policies": {"human_coded_ground_truth_used": False, "causal_claims_made": False, "unresolved_causes_preserved": True, "database_write_performed": False},
        "metadata_verification": {"prepared_by": "Tahereh Fahi"},
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
