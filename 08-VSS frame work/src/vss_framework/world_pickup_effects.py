"""Effect corroboration for world pickups without Human-coded ground truth."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .hashing import sha256_file
from .io import iter_jsonl, write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision

RESOLVER_VERSION = "0.2.0"


def extract_currency_gains(observations: Sequence[dict[str, Any]], *, processing_run_id: str,
                           maximum_gap_ms: int = 2_000, minimum_ocr_confidence: float = 0.8) -> list[CanonicalEvent]:
    """Convert consecutive direct Coin Counter increases into gain evidence."""
    # Retain unavailable samples so an interruption cannot disappear from the
    # sequence and create a spurious before/after gain.
    rows = sorted((row for row in observations if row.get("observable_code") == "coin_counter"),
                  key=lambda row: int(row["time_lower_ms"]))
    events: list[CanonicalEvent] = []
    for index in range(1, len(rows) - 1):
        previous, current, following = rows[index - 1:index + 2]
        triple = (previous, current, following)
        if any(not row.get("observed") or row.get("numeric_value") is None
               or row.get("attributes", {}).get("excluded_from_gameplay", False)
               for row in triple):
            continue
        if len({row.get("attributes", {}).get("gameplay_segment_id") for row in triple}) != 1:
            continue
        start, end = int(previous["time_lower_ms"]), int(current["time_lower_ms"])
        if not 0 < end - start <= maximum_gap_ms:
            continue
        previous_confidence = float(previous.get("attributes", {}).get("confidence") or 0)
        current_confidence = float(current.get("attributes", {}).get("confidence") or 0)
        if min(previous_confidence, current_confidence) < minimum_ocr_confidence:
            continue
        following_time = int(following["time_lower_ms"])
        if not 0 < following_time - end <= maximum_gap_ms or int(following["numeric_value"]) != int(current["numeric_value"]):
            continue
        delta = int(current["numeric_value"]) - int(previous["numeric_value"])
        if delta <= 0:
            continue
        events.append(CanonicalEvent(
            event_id=f"vss_currency_gain_{current.get('frame_number', end)}",
            video_asset_id=current["video_asset_id"], session_id=current["session_id"],
            event_family="currency", event_type="currency_gain",
            time_lower_ms=start, time_upper_ms=end, anchor_time_ms=end,
            temporal_precision=TemporalPrecision.BOUNDED, evidence_grade=EvidenceGrade.B,
            publication_status=PublicationStatus.NEEDS_REVIEW,
            inference_method="direct_monotonic_coin_counter_increase", processing_run_id=processing_run_id,
            frame_number=current.get("frame_number"), quantity=delta, unit="gold_coins",
            evidence=(EvidenceReference(source_artifact=current["source_artifact"],
                source_record_key=current["source_record_key"], modalities=("coin_counter_ocr",),
                details={"previous_value": previous["numeric_value"], "current_value": current["numeric_value"]}),),
            attributes={"physical_pickup_identity": None, "other_currency_sources_not_excluded": True,
                        "previous_ocr_confidence": previous_confidence, "current_ocr_confidence": current_confidence,
                        "new_value_confirmed_at_ms": following_time,
                        "imputed": False},
        ))
    return events


def extract_abrupt_healing_candidates(health_events: Sequence[dict[str, Any]], *, processing_run_id: str,
                                      minimum_percentage_points: float = 10.0) -> list[CanonicalEvent]:
    """Large persistent recoveries support healing, not a unique item identity."""
    events: list[CanonicalEvent] = []
    for source in health_events:
        if source.get("event_type") != "hp_recovery" or float(source.get("quantity") or 0) < minimum_percentage_points:
            continue
        events.append(CanonicalEvent(
            event_id=f"vss_abrupt_healing_{source.get('frame_number', source['anchor_time_ms'])}",
            video_asset_id=source["video_asset_id"], session_id=source["session_id"],
            event_family="recovery", event_type="abrupt_healing_effect",
            time_lower_ms=int(source["time_lower_ms"]), time_upper_ms=int(source["time_upper_ms"]),
            anchor_time_ms=int(source["anchor_time_ms"]), temporal_precision=TemporalPrecision.BOUNDED,
            evidence_grade=EvidenceGrade.C, publication_status=PublicationStatus.NEEDS_REVIEW,
            inference_method="large_persistent_health_bar_increase", processing_run_id=processing_run_id,
            frame_number=source.get("frame_number"), quantity=float(source["quantity"]), unit="percentage_points",
            evidence=(EvidenceReference(source_artifact="health_events.jsonl", source_record_key=source["event_id"],
                modalities=("health_bar",), details={"source_event_type": "hp_recovery"}),),
            attributes={"compatible_pickup_types": ["floor_chicken"], "floor_chicken_confirmed": False,
                        "other_healing_sources_not_excluded": True, "imputed": False},
        ))
    return events


def corroborate_effects(effect_events: Sequence[CanonicalEvent], sprite_candidates: Sequence[dict[str, Any]],
                        *, maximum_lag_ms: int = 2_000) -> list[CanonicalEvent]:
    """Promote identity only when a matching sprite candidate overlaps an effect."""
    result: list[CanonicalEvent] = []
    for event in effect_events:
        expected = "floor_chicken" if event.event_type == "abrupt_healing_effect" else None
        matches = [row for row in sprite_candidates if expected and row.get("event_type") == expected
                   and abs(int(row["anchor_time_ms"]) - int(event.anchor_time_ms or 0)) <= maximum_lag_ms]
        if not matches:
            result.append(event)
            continue
        source = min(matches, key=lambda row: abs(int(row["anchor_time_ms"]) - int(event.anchor_time_ms or 0)))
        result.append(CanonicalEvent(**{**event.__dict__,
            "event_type": expected, "event_family": "world_pickup",
            "evidence_grade": EvidenceGrade.B, "publication_status": PublicationStatus.AUTO_ACCEPTED,
            "inference_method": "sprite_disappearance_plus_matching_direct_effect",
            "evidence": event.evidence + (EvidenceReference(source_artifact="world_pickup_candidates.jsonl",
                source_record_key=source["event_id"], modalities=("sprite_template", "temporal_tracking")),),
            "attributes": {**event.attributes, "floor_chicken_confirmed": True,
                           "corroborating_sprite_event_id": source["event_id"]}}))
    return result


def resolve_video4_world_pickup_effects(*, health_events_path: Path, sprite_candidates_path: Path,
                                        output_dir: Path, telemetry_path: Path | None = None) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    telemetry_sha = sha256_file(telemetry_path) if telemetry_path is not None else "none"
    identity = f"{RESOLVER_VERSION}|{sha256_file(health_events_path)}|{sha256_file(sprite_candidates_path)}|{telemetry_sha}"
    run_id = f"run_pickup_effects_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    health = list(iter_jsonl(health_events_path))
    sprites = list(iter_jsonl(sprite_candidates_path))
    effects = extract_abrupt_healing_candidates(health, processing_run_id=run_id)
    if telemetry_path is not None:
        effects.extend(extract_currency_gains(list(iter_jsonl(telemetry_path)), processing_run_id=run_id))
    resolved = corroborate_effects(effects, sprites)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "world_pickup_effect_events.jsonl"
    write_jsonl(output_path, (row.to_dict() for row in resolved))
    counts = Counter(row.event_type for row in resolved)
    sources = {"health_events_sha256": sha256_file(health_events_path),
               "sprite_candidates_sha256": sha256_file(sprite_candidates_path),
               "resolver_source_sha256": sha256_file(Path(__file__))}
    if telemetry_path is not None:
        sources["telemetry_sha256"] = sha256_file(telemetry_path)
    manifest = {"artifact_type": "vss_framework_world_pickup_effect_resolution",
        "framework_version": "0.9.0", "resolver_version": RESOLVER_VERSION,
        "processing_run_id": run_id, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi", "counts": {"events": len(resolved), "by_type": dict(sorted(counts.items()))},
        "sources": sources,
        "outputs": {"effect_events": {"path": output_path.name, "sha256": sha256_file(output_path), "row_count": len(resolved)}},
        "policies": {"human_coded_ground_truth_used": False, "blank_means_zero": False,
                     "unconfirmed_identity_published": False, "database_write_performed": False},
        "limitations": ["A health jump alone does not uniquely identify Floor Chicken.",
                        "A Coin Counter increase does not distinguish a physical pickup from every other currency source."],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
