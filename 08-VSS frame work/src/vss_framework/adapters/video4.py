"""Reuse existing Video 4 outputs without treating Human coding as ground truth."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..hashing import verify_file
from ..io import iter_jsonl
from ..models import (
    CanonicalEvent,
    EvidenceGrade,
    EvidenceReference,
    PublicationStatus,
    TemporalPrecision,
)


def resolve_and_verify_sources(
    workspace_root: Path, config: dict[str, Any]
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    resolved: dict[str, Path] = {}
    verified: list[dict[str, object]] = []

    video_spec = config["dataset"]["video"]
    video_path = workspace_root / video_spec["path"]
    resolved["video"] = video_path
    video_record = verify_file(video_path, video_spec["sha256"])
    video_record["path"] = video_spec["path"]
    video_record["role"] = "video"
    verified.append(video_record)

    for role, spec in config["sources"].items():
        path = workspace_root / spec["path"]
        resolved[role] = path
        record = verify_file(path, spec["sha256"])
        record["path"] = spec["path"]
        record["role"] = role
        verified.append(record)
    return resolved, verified


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return int(float(text))


def _gem_grade(row: dict[str, Any]) -> tuple[EvidenceGrade, PublicationStatus]:
    attributes = row.get("attributes_json") or {}
    if attributes.get("needs_review"):
        return EvidenceGrade.C, PublicationStatus.NEEDS_REVIEW
    modalities = set((row.get("evidence_json") or {}).get("modalities") or [])
    if "visual" in modalities and "experience_bar_change" in modalities:
        return EvidenceGrade.A, PublicationStatus.AUTO_ACCEPTED
    return EvidenceGrade.B, PublicationStatus.AUTO_ACCEPTED


def _adapt_gem_events(
    observations_path: Path,
    source_label: str,
    *,
    video_asset_id: str,
    session_id: str,
    processing_run_id: str,
) -> Iterable[CanonicalEvent]:
    for row in iter_jsonl(observations_path):
        if row.get("observable_code") != "gem_pickup":
            continue
        attributes = row.get("attributes_json") or {}
        gem_type = str(attributes.get("gem_type") or "unresolved").lower()
        if gem_type not in {"blue", "green", "red", "unresolved"}:
            gem_type = "unresolved"
        start_ms = int(row["media_start_ms"])
        end_ms = int(row["media_end_ms"])
        grade, publication = _gem_grade(row)
        evidence_json = row.get("evidence_json") or {}
        yield CanonicalEvent(
            event_id=f"vss_{row['observation_id']}",
            video_asset_id=video_asset_id,
            session_id=session_id,
            event_family="gem",
            event_type=f"{gem_type}_gem_pickup",
            time_lower_ms=start_ms,
            time_upper_ms=end_ms,
            anchor_time_ms=end_ms,
            temporal_precision=TemporalPrecision.WINDOW,
            evidence_grade=grade,
            publication_status=publication,
            inference_method=str(attributes.get("color_evidence") or "xp_linked_pickup"),
            processing_run_id=processing_run_id,
            frame_number=_int_or_none(row.get("frame_number")),
            game_time_ms=_int_or_none(row.get("game_time_ms")),
            character_level=_int_or_none(attributes.get("hud_level")),
            item_name=f"{gem_type} gem",
            item_type="experience_gem",
            action="pickup",
            acquisition_source="world_pickup",
            quantity=float(row.get("quantity") or 0),
            quantity_min=0.0 if grade == EvidenceGrade.C else float(row.get("quantity") or 0),
            quantity_max=float(row.get("quantity") or 0),
            unit="estimated_gems",
            evidence=(
                EvidenceReference(
                    source_artifact=source_label,
                    source_record_key=str(row.get("source_record_key") or row["observation_id"]),
                    modalities=tuple(evidence_json.get("modalities") or ()),
                    details={
                        "source_observation_id": row["observation_id"],
                        "needs_review": bool(attributes.get("needs_review")),
                        "heuristic_confidence": row.get("confidence"),
                    },
                ),
            ),
            attributes={
                "quantity_semantics": "xp_linked_estimate_not_exact_physical_count",
                "source_event_key": attributes.get("event_key"),
                "review_reason": attributes.get("review_reason"),
            },
        )


def _inventory_event_type(row: dict[str, str]) -> str:
    event_type = row["event_type"].strip().lower()
    item_type = row["item_type"].strip().lower()
    if event_type == "evolution":
        return "weapon_evolution"
    if event_type == "new" and item_type == "weapon":
        return "new_weapon"
    if event_type == "new" and item_type == "passive_item":
        return "new_passive_item"
    if event_type == "upgrade" and item_type == "weapon":
        return "weapon_upgrade"
    if event_type == "upgrade" and item_type == "passive_item":
        return "passive_item_upgrade"
    return "inventory_change"


def _adapt_inventory_events(
    inventory_path: Path,
    source_label: str,
    *,
    video_asset_id: str,
    session_id: str,
    processing_run_id: str,
) -> Iterable[CanonicalEvent]:
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"].strip().lower() == "initial_state":
                continue
            source_id = row["event_id"]
            time_ms = int(round(float(row["video_second"]) * 1000))
            needs_review = _bool(row.get("needs_review"))
            confidence = row.get("confidence", "").strip().lower()
            grade = (
                EvidenceGrade.C
                if needs_review
                else EvidenceGrade.A
                if confidence == "high"
                else EvidenceGrade.B
            )
            publication = (
                PublicationStatus.NEEDS_REVIEW
                if needs_review
                else PublicationStatus.AUTO_ACCEPTED
            )
            acquisition_source = row["event_source"].strip().lower()
            selection_event_id = None
            if acquisition_source.startswith("level_up"):
                selection_event_id = f"vss_{source_id}_selection"
                yield CanonicalEvent(
                    event_id=selection_event_id,
                    video_asset_id=video_asset_id,
                    session_id=session_id,
                    event_family="progression",
                    event_type="level_up_selection",
                    time_lower_ms=time_ms,
                    time_upper_ms=time_ms,
                    anchor_time_ms=time_ms,
                    temporal_precision=TemporalPrecision.FRAME,
                    evidence_grade=grade,
                    publication_status=publication,
                    inference_method="final_selected_option_frame",
                    processing_run_id=processing_run_id,
                    frame_number=_int_or_none(row.get("frame_number")),
                    character_level=_int_or_none(row.get("character_level")),
                    action="selection",
                    acquisition_source="level_up",
                    quantity=1,
                    unit="selection",
                    evidence=(
                        EvidenceReference(
                            source_artifact=source_label,
                            source_record_key=source_id,
                            modalities=("level_up_menu", "inventory"),
                        ),
                    ),
                    attributes={
                        "selected_item_identity": row.get("item_after") or None,
                        "classification_confidence_label": confidence or None,
                    },
                )

            yield CanonicalEvent(
                event_id=f"vss_{source_id}_outcome",
                video_asset_id=video_asset_id,
                session_id=session_id,
                event_family="inventory",
                event_type=_inventory_event_type(row),
                time_lower_ms=time_ms,
                time_upper_ms=time_ms,
                anchor_time_ms=time_ms,
                temporal_precision=TemporalPrecision.FRAME,
                evidence_grade=grade,
                publication_status=publication,
                inference_method=(
                    row.get("inference_method")
                    or ("verified_chest_audit" if acquisition_source == "treasure_chest" else "final_selected_option")
                ),
                processing_run_id=processing_run_id,
                frame_number=_int_or_none(row.get("frame_number")),
                character_level=_int_or_none(row.get("character_level")),
                item_name=row.get("item_after") or None,
                item_type=row.get("item_type") or None,
                action=row.get("event_type") or None,
                acquisition_source=acquisition_source,
                quantity=1,
                unit="inventory_transition",
                evidence=(
                    EvidenceReference(
                        source_artifact=source_label,
                        source_record_key=source_id,
                        modalities=("level_up_menu", "inventory")
                        if acquisition_source.startswith("level_up")
                        else ("inventory", "verified_chest_audit"),
                    ),
                ),
                attributes={
                    "selection_event_id": selection_event_id,
                    "slot": row.get("slot") or None,
                    "item_before": row.get("item_before") or None,
                    "level_before": _int_or_none(row.get("level_before")),
                    "level_after": _int_or_none(row.get("level_after")),
                    "classification_confidence_label": confidence or None,
                },
            )


def adapt_inventory_events(
    inventory_path: Path,
    source_label: str,
    *,
    video_asset_id: str,
    session_id: str,
    processing_run_id: str,
) -> Iterable[CanonicalEvent]:
    """Public adapter for an automated inventory recorder output."""

    return _adapt_inventory_events(
        inventory_path,
        source_label,
        video_asset_id=video_asset_id,
        session_id=session_id,
        processing_run_id=processing_run_id,
    )


def _adapt_retrospective_outcomes(
    outcomes_path: Path,
    source_label: str,
    *,
    video_asset_id: str,
    session_id: str,
    processing_run_id: str,
) -> Iterable[CanonicalEvent]:
    payload = json.loads(outcomes_path.read_text(encoding="utf-8"))
    for row in payload.get("nonInventoryOutcomes", []):
        if row.get("dataset") != "video4":
            continue
        source_id = str(row["eventId"])
        time_ms = int(row["mediaTimeMs"])
        needs_review = bool(row.get("needsReview"))
        grade = EvidenceGrade.C if needs_review else EvidenceGrade.B
        publication = PublicationStatus.NEEDS_REVIEW
        outcome = str(row["outcome"])
        event_type = {
            "Big Coin Bag": "big_coin_bag",
            "Floor Chicken": "floor_chicken",
        }.get(outcome, "consumable_reward")
        selection_id = f"vss_{source_id}_selection"
        common_evidence = (
            EvidenceReference(
                source_artifact=source_label,
                source_record_key=source_id,
                modalities=("level_up_menu", "reward_option"),
            ),
        )
        yield CanonicalEvent(
            event_id=selection_id,
            video_asset_id=video_asset_id,
            session_id=session_id,
            event_family="progression",
            event_type="level_up_selection",
            time_lower_ms=time_ms,
            time_upper_ms=time_ms,
            anchor_time_ms=time_ms,
            temporal_precision=TemporalPrecision.FRAME,
            evidence_grade=grade,
            publication_status=publication,
            inference_method="reviewed_reward_option",
            processing_run_id=processing_run_id,
            frame_number=int(row["frameNumber"]),
            character_level=int(row["resultingLevel"]),
            action="selection",
            acquisition_source="level_up",
            quantity=1,
            unit="selection",
            evidence=common_evidence,
            attributes={"selected_outcome": outcome},
        )
        yield CanonicalEvent(
            event_id=f"vss_{source_id}_outcome",
            video_asset_id=video_asset_id,
            session_id=session_id,
            event_family="currency" if outcome == "Big Coin Bag" else "consumable",
            event_type=event_type,
            time_lower_ms=time_ms,
            time_upper_ms=time_ms,
            anchor_time_ms=time_ms,
            temporal_precision=TemporalPrecision.FRAME,
            evidence_grade=grade,
            publication_status=publication,
            inference_method="reviewed_reward_option",
            processing_run_id=processing_run_id,
            frame_number=int(row["frameNumber"]),
            character_level=int(row["resultingLevel"]),
            item_name=outcome,
            item_type="instant_reward",
            action="acquired",
            acquisition_source="level_up",
            quantity=1,
            unit="reward",
            evidence=common_evidence,
            attributes={"selection_event_id": selection_id},
        )


def build_video4_events(
    paths: dict[str, Path], config: dict[str, Any], processing_run_id: str
) -> list[CanonicalEvent]:
    dataset = config["dataset"]
    kwargs = {
        "video_asset_id": dataset["video_asset_id"],
        "session_id": dataset["session_id"],
        "processing_run_id": processing_run_id,
    }
    sources = config["sources"]
    events = [
        *_adapt_gem_events(
            paths["canonical_observations"],
            sources["canonical_observations"]["path"],
            **kwargs,
        ),
        *_adapt_inventory_events(
            paths["inventory_events"],
            sources["inventory_events"]["path"],
            **kwargs,
        ),
        *_adapt_retrospective_outcomes(
            paths["retrospective_outcomes"],
            sources["retrospective_outcomes"]["path"],
            **kwargs,
        ),
    ]
    events.sort(key=lambda event: (event.time_lower_ms, event.event_id))
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Duplicate canonical event IDs were generated")
    return events
