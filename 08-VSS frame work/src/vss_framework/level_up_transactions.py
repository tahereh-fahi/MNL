"""Resolve automated Level-Up menu observations into transaction records."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from .models import (
    CanonicalEvent,
    EvidenceGrade,
    EvidenceReference,
    PublicationStatus,
    TemporalPrecision,
)


CURSOR_MARGIN_THRESHOLD = 20.0
MATCH_TOLERANCE_MS = 250


def _boolean(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _milliseconds(value: object) -> int:
    return round(float(str(value)) * 1000)


def resolve_level_up_transactions(
    *,
    menu_audit_path: Path,
    inventory_events_path: Path,
    video_asset_id: str,
    session_id: str,
    processing_run_id: str,
    source_artifact: str,
    end_ms: int | None = None,
) -> list[CanonicalEvent]:
    """Create one transaction per menu with an observable final pointer.

    A missing inventory transition is not sufficient evidence to distinguish
    Banish, Skip, or an instant reward. Such transactions remain unresolved.
    """

    with inventory_events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        inventory_rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("event_source", "")).startswith("level_up")
            and row.get("event_type") != "initial_state"
        ]
    inventory_by_time = [(_milliseconds(row["video_second"]), row) for row in inventory_rows]

    def owned_level_before(item_name: str, time_ms: int) -> tuple[int | None, int | None]:
        candidates = []
        for event_time, event_row in inventory_by_time:
            if event_time >= time_ms or event_row.get("item_after") != item_name:
                continue
            try:
                level = int(float(event_row.get("level_after") or ""))
                maximum = int(float(event_row.get("normal_max_level") or ""))
            except ValueError:
                continue
            candidates.append((event_time, level, maximum))
        if not candidates:
            return None, None
        _, level, maximum = max(candidates)
        return level, maximum

    transactions: list[CanonicalEvent] = []
    with menu_audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            end_time_ms = _milliseconds(row["menu_end_second"])
            if end_ms is not None and end_time_ms >= end_ms:
                continue
            cursor_score = float(row.get("cursor_score") or 0)
            if cursor_score < CURSOR_MARGIN_THRESHOLD:
                continue
            nearest = [
                (abs(event_time - end_time_ms), event_row)
                for event_time, event_row in inventory_by_time
                if abs(event_time - end_time_ms) <= MATCH_TOLERANCE_MS
            ]
            match = min(nearest, default=None, key=lambda item: item[0])
            needs_review = _boolean(row.get("needs_review"))
            start_time_ms = _milliseconds(row["menu_start_second"])
            candidate = row.get("suggested_item") or ""
            option_count = int(float(row.get("option_count") or 0))
            if (
                match is None
                and transactions
                and transactions[-1].action == "select_reward"
                and transactions[-1].item_name == candidate
                and transactions[-1].attributes.get("option_count") == option_count
                and 0 <= start_time_ms - transactions[-1].time_upper_ms <= 250
            ):
                # The same menu was split at an overlapping XP search-window
                # boundary.  It is not a second level, but its final frame is
                # the completed selection time and must be retained.
                previous = transactions[-1]
                attributes = dict(previous.attributes)
                attributes["collapsed_menu_record_count"] = int(
                    attributes.get("collapsed_menu_record_count", 1)
                ) + 1
                transactions[-1] = replace(
                    previous,
                    time_upper_ms=end_time_ms,
                    anchor_time_ms=end_time_ms,
                    frame_number=int(float(row["frame_number"])),
                    evidence=previous.evidence + (
                        EvidenceReference(
                            source_artifact=source_artifact,
                            source_record_key=f"menu:{index}",
                            modalities=("level_up_menu", "selection_pointer"),
                            details={"collapsed_as_continuation": True},
                        ),
                    ),
                    attributes=attributes,
                )
                continue
            evidence = (
                EvidenceReference(
                    source_artifact=source_artifact,
                    source_record_key=f"menu:{index}",
                    modalities=("level_up_menu", "selection_pointer", "inventory_state"),
                    details={
                        "cursor_score": cursor_score,
                        "selected_index": int(float(row["selected_index"])),
                    },
                ),
            )
            if match is not None:
                inventory = match[1]
                action = "select_reward"
                grade = EvidenceGrade.C if needs_review else EvidenceGrade.A
                status = (
                    PublicationStatus.NEEDS_REVIEW
                    if needs_review else PublicationStatus.AUTO_ACCEPTED
                )
                attributes = {
                    "resolved_action": action,
                    "selected_item": inventory.get("item_after") or None,
                    "inventory_event_id": inventory.get("event_id"),
                    "inventory_event_type": inventory.get("event_type"),
                    "possible_actions": [action],
                    "option_count": option_count,
                }
                item_name = inventory.get("item_after") or None
                character_level = int(float(inventory["character_level"]))
            else:
                owned_level, normal_max = owned_level_before(candidate, end_time_ms)
                is_owned_at_max = (
                    option_count == 1
                    and
                    owned_level is not None
                    and normal_max is not None
                    and owned_level >= normal_max
                )
                if is_owned_at_max:
                    action = "banish"
                    grade = EvidenceGrade.B
                    status = PublicationStatus.AUTO_ACCEPTED
                    attributes = {
                        "resolved_action": "banish",
                        "selected_item": candidate,
                        "owned_level_before": owned_level,
                        "normal_max_level": normal_max,
                        "option_count": option_count,
                        "possible_actions": ["banish"],
                        "reason": "selected_owned_item_already_at_normal_max",
                    }
                else:
                    action = "unresolved_no_inventory_transition"
                    grade = EvidenceGrade.UNRESOLVED
                    status = PublicationStatus.UNRESOLVED
                    attributes = {
                        "resolved_action": None,
                        "selected_item_candidate": candidate or None,
                        "possible_actions": ["banish", "skip", "instant_reward"],
                        "reason": "menu_closed_without_corresponding_inventory_transition",
                        "reroll_requires_continuous_menu_content_change": True,
                        "option_count": option_count,
                    }
                item_name = None
                character_level = None
            transactions.append(CanonicalEvent(
                event_id=f"vss_level_up_transaction_{index:04d}_{end_time_ms}",
                video_asset_id=video_asset_id,
                session_id=session_id,
                event_family="menu_action",
                event_type="level_up_transaction",
                time_lower_ms=start_time_ms,
                time_upper_ms=end_time_ms,
                anchor_time_ms=end_time_ms,
                temporal_precision=TemporalPrecision.INTERVAL,
                evidence_grade=grade,
                publication_status=status,
                inference_method=(
                    "menu_pointer_plus_inventory_transition"
                    if match is not None else "menu_close_without_inventory_transition"
                ),
                processing_run_id=processing_run_id,
                frame_number=int(float(row["frame_number"])),
                character_level=character_level,
                item_name=item_name,
                action=action,
                acquisition_source="level_up",
                quantity=1,
                unit="transaction",
                evidence=evidence,
                attributes=attributes,
            ))
    return transactions
