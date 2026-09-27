"""Compile dashboard-ready releases from canonical Automated framework events."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import iter_jsonl


GEM_LABELS = {
    "blue": "Blue XP gem",
    "green": "Green XP gem",
    "red": "Red XP gem",
    "unresolved": "Unresolved XP pickup",
}

EVIDENCE_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
EVOLUTION_CLUSTER_MS = 10_000
LUCKY_TRANSACTION_ANCHOR_TOLERANCE_MS = 50
LUCKY_FRAGMENT_GAP_MS = 500
INVENTORY_CANONICAL_ANCHOR_TOLERANCE_MS = 1


def _event_value(event: Mapping[str, Any], dashboard_key: str, canonical_key: str) -> Any:
    return event.get(dashboard_key) if dashboard_key in event else event.get(canonical_key)


def canonicalize_lucky_level_ups(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Publish one Lucky interval for each completed level-up transaction.

    The menu audit can split one four-choice screen into adjacent rows while the
    selection highlight changes.  A real Lucky level-up must still correspond
    to exactly one completed level-up transaction.  Keep transaction-backed
    detections, absorb a short following orphan fragment into that interval,
    and reject unpaired detections instead of inventing another level increase.
    Both canonical-event and dashboard-event field names are supported.
    """

    copied = [dict(event) for event in events]
    lucky = sorted(
        (event for event in copied if _event_value(event, "eventType", "event_type") == "lucky_level_up"),
        key=lambda event: int(_event_value(event, "startMs", "time_lower_ms") or 0),
    )
    if not lucky:
        return copied

    transactions = [
        event for event in copied
        if _event_value(event, "eventType", "event_type") == "level_up_transaction"
        and str(event.get("action") or "") not in {"banish", "reroll", "skip"}
    ]
    # Some source releases do not yet include transaction events. Preserve
    # their Lucky evidence until that detector is available instead of erasing it.
    if not transactions:
        return copied

    ordinary = [event for event in copied if _event_value(event, "eventType", "event_type") != "lucky_level_up"]
    canonical: list[dict[str, Any]] = []
    canonical_by_transaction: dict[str, dict[str, Any]] = {}
    for event in lucky:
        start = int(_event_value(event, "startMs", "time_lower_ms") or 0)
        end = int(_event_value(event, "endMs", "time_upper_ms") or 0)
        anchor = int(_event_value(event, "anchorMs", "anchor_time_ms") or 0)
        matching = sorted(
            (
                transaction for transaction in transactions
                if (
                    abs(int(_event_value(transaction, "anchorMs", "anchor_time_ms") or 0) - anchor)
                    <= LUCKY_TRANSACTION_ANCHOR_TOLERANCE_MS
                    or (
                        start <= int(_event_value(transaction, "endMs", "time_upper_ms") or 0)
                        and end >= int(_event_value(transaction, "startMs", "time_lower_ms") or 0)
                    )
                )
            ),
            key=lambda transaction: abs(int(_event_value(transaction, "anchorMs", "anchor_time_ms") or 0) - anchor),
        )
        event_id_key = "eventId" if "eventId" in event else "event_id"
        end_key = "endMs" if "endMs" in event else "time_upper_ms"
        support_key = "supportingEventIds" if "eventId" in event else "supporting_event_ids"
        transaction_key = "sourceTransactionEventId" if "eventId" in event else "source_transaction_event_id"

        if matching:
            transaction = matching[0]
            transaction_id = str(_event_value(transaction, "eventId", "event_id"))
            transaction_anchor = int(_event_value(transaction, "anchorMs", "anchor_time_ms") or anchor)
            existing = canonical_by_transaction.get(transaction_id)
            if existing is not None:
                existing_end_key = "endMs" if "endMs" in existing else "time_upper_ms"
                existing[existing_end_key] = max(int(existing[existing_end_key]), end)
                existing_support_key = "supportingEventIds" if "eventId" in existing else "supporting_event_ids"
                support = set(existing.get(existing_support_key) or [existing["eventId" if "eventId" in existing else "event_id"]])
                support.update(event.get(support_key) or [event[event_id_key]])
                existing[existing_support_key] = sorted(support)
                existing["evidenceCount" if "eventId" in existing else "evidence_count"] = len(support)
                existing["canonicalized"] = True
                continue
            event[transaction_key] = transaction_id
            event["anchorMs" if "eventId" in event else "anchor_time_ms"] = transaction_anchor
            event[end_key] = max(end, transaction_anchor)
            event[support_key] = sorted(set(event.get(support_key) or [event[event_id_key]]))
            event["evidenceCount" if "eventId" in event else "evidence_count"] = len(event[support_key])
            canonical.append(event)
            canonical_by_transaction[transaction_id] = event
            continue

        previous = canonical[-1] if canonical else None
        if previous is None:
            continue
        gap = int(_event_value(event, "startMs", "time_lower_ms") or 0) - int(
            _event_value(previous, "endMs", "time_upper_ms") or 0
        )
        if not 0 <= gap <= LUCKY_FRAGMENT_GAP_MS:
            continue
        previous_end_key = "endMs" if "endMs" in previous else "time_upper_ms"
        previous[previous_end_key] = max(int(previous[previous_end_key]), int(event[end_key]))
        previous_support_key = "supportingEventIds" if "eventId" in previous else "supporting_event_ids"
        previous_support = set(previous.get(previous_support_key) or [previous["eventId" if "eventId" in previous else "event_id"]])
        previous_support.update(event.get(support_key) or [event[event_id_key]])
        previous[previous_support_key] = sorted(previous_support)
        previous["evidenceCount" if "eventId" in previous else "evidence_count"] = len(previous_support)
        previous["canonicalized"] = True

    return sorted(
        ordinary + canonical,
        key=lambda event: (
            int(_event_value(event, "anchorMs", "anchor_time_ms") or 0),
            str(_event_value(event, "eventType", "event_type") or ""),
            str(_event_value(event, "eventId", "event_id") or ""),
        ),
    )


def canonicalize_weapon_evolutions(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse multiple detector observations of one evolution into one event.

    Chest-reward recognition and subsequent inventory-transition recognition are
    supporting evidence for the same outcome when they name the same evolved
    weapon within one chest sequence.  Prefer the chest-reward timestamp, retain
    the strongest evidence grade, and preserve every source id for auditability.
    """

    ordinary = [dict(event) for event in events if event.get("eventType") != "weapon_evolution"]
    evolutions = sorted(
        (dict(event) for event in events if event.get("eventType") == "weapon_evolution"),
        key=lambda event: (str(event.get("itemName") or ""), int(event.get("anchorMs") or 0)),
    )
    clusters: list[list[dict[str, Any]]] = []
    for event in evolutions:
        if (
            clusters
            and clusters[-1][0].get("itemName") == event.get("itemName")
            and int(event.get("anchorMs") or 0) - int(clusters[-1][-1].get("anchorMs") or 0) <= EVOLUTION_CLUSTER_MS
        ):
            clusters[-1].append(event)
        else:
            clusters.append([event])

    canonical = []
    for cluster in clusters:
        primary = min(
            cluster,
            key=lambda event: (
                0 if "chest_reward" in str(event.get("eventId") or "") else 1,
                int(event.get("anchorMs") or 0),
            ),
        )
        merged = dict(primary)
        merged["evidenceGrade"] = min(
            (str(event.get("evidenceGrade") or "D") for event in cluster),
            key=lambda grade: EVIDENCE_GRADE_ORDER.get(grade, 99),
        )
        merged["publicationStatus"] = (
            "auto_accepted"
            if any(event.get("publicationStatus") == "auto_accepted" for event in cluster)
            else "needs_review"
        )
        supporting_ids = {
            str(event_id)
            for event in cluster
            for event_id in (event.get("supportingEventIds") or [event["eventId"]])
        }
        merged["supportingEventIds"] = sorted(supporting_ids)
        merged["evidenceCount"] = len(supporting_ids)
        merged["canonicalized"] = len(supporting_ids) > 1
        canonical.append(merged)
    result = ordinary + canonical
    chests = [event for event in result if str(event.get("eventType") or "").startswith("loot_box_tier_")]
    for evolution in canonical:
        containing = next(
            (
                chest for chest in chests
                if int(chest.get("startMs") or 0) <= int(evolution.get("anchorMs") or 0) <= int(chest.get("endMs") or 0)
            ),
            None,
        )
        if containing:
            evolution["parentChestEventId"] = containing["eventId"]
    return sorted(result, key=lambda row: (row["anchorMs"], row["eventType"], row["eventId"]))


def refresh_dashboard_release(release: Mapping[str, Any]) -> dict[str, Any]:
    """Return a release with canonical reward events and refreshed counts."""

    result = dict(release)
    events = canonicalize_lucky_level_ups(canonicalize_weapon_evolutions(release.get("events", [])))
    by_type = dict(sorted(Counter(row["eventType"] for row in events).items()))
    accepted = sum(row.get("publicationStatus") == "auto_accepted" for row in events)
    result["events"] = events
    result["summary"] = {
        **dict(release.get("summary", {})),
        "eventCount": len(events),
        "acceptedCount": accepted,
        "reviewCount": len(events) - accepted,
        "byEventType": by_type,
    }
    result["coverage"] = [
        {
            **dict(row),
            "published_count": by_type.get(row.get("event_type"), 0),
        }
        for row in release.get("coverage", [])
    ]
    return result


def project_canonical_event(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventId": row["event_id"],
        "eventType": row["event_type"],
        "family": row["event_family"],
        "startMs": row["time_lower_ms"],
        "endMs": row["time_upper_ms"],
        "anchorMs": row.get("anchor_time_ms") if row.get("anchor_time_ms") is not None else row["time_lower_ms"],
        "quantity": row.get("quantity"),
        "itemName": row.get("item_name"),
        "action": row.get("action"),
        "evidenceGrade": row["evidence_grade"],
        "publicationStatus": row["publication_status"],
        **({"evidence": row.get("evidence", []), "attributes": row.get("attributes", {}),
            "processingRunId": row.get("processing_run_id"), "inferenceMethod": row.get("inference_method")}
           if row["event_type"] in {"hp_loss", "hp_recovery"} else {}),
    }


def project_exact_gem_events(gem_events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for source in gem_events:
        for gem_type, label in GEM_LABELS.items():
            quantity = int((source.get("quantities") or {}).get(gem_type) or 0)
            if quantity <= 0:
                continue
            projected.append(
                {
                    "eventId": f"{source['eventKey']}_{gem_type}",
                    "eventType": f"{gem_type}_gem_pickup",
                    "family": "gem",
                    "startMs": source["videoTimeMs"],
                    "endMs": source["videoTimeMs"],
                    "anchorMs": source["videoTimeMs"],
                    "quantity": quantity,
                    "itemName": label,
                    "action": "picked up",
                    "evidenceGrade": "C" if source.get("needsReview") else "B",
                    "publicationStatus": "needs_review" if source.get("needsReview") else "auto_accepted",
                }
            )
    return projected


def project_inventory_selection_events(
    inventory_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert selected inventory outcomes to canonical dashboard events."""

    projected = []
    for source in inventory_events:
        needs_review = bool(source.get("needsReview"))
        confidence = str(source.get("confidenceLabel") or "low").casefold()
        grade = "B" if confidence == "high" and not needs_review else "C"
        common = {
            "startMs": int(source["mediaTimeMs"]),
            "endMs": int(source["mediaTimeMs"]),
            "anchorMs": int(source["mediaTimeMs"]),
            "quantity": 1,
            "itemName": source.get("itemName"),
            "action": source.get("action"),
            "evidenceGrade": grade,
            "publicationStatus": "needs_review" if needs_review else "auto_accepted",
        }
        projected.extend(
            [
                {
                    **common,
                    "eventId": f"{source['eventId']}_transaction",
                    "eventType": "level_up_transaction",
                    "family": "progression",
                },
                {
                    **common,
                    "eventId": str(source["eventId"]),
                    "eventType": "level_up_selection",
                    "family": "inventory",
                },
            ]
        )
    return projected


def _unrepresented_inventory_events(
    projected: Sequence[Mapping[str, Any]], canonical: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Use inventory projections only where no canonical level-up event exists.

    The inventory CSV and canonical inventory JSONL describe the same selection
    with different event IDs.  ID-based deduplication therefore counts it twice.
    Match the frame anchor and event type, allowing one millisecond for the
    CSV projection's rounding versus the canonical event's truncation.  For
    transactions, also require a selected-reward action and a compatible item
    identity so a distinct menu action or conflicting result is not discarded.
    """

    canonical_by_kind_and_anchor: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for event in canonical:
        kind = str(event.get("eventType"))
        if kind not in {"level_up_selection", "level_up_transaction"}:
            continue
        key = (kind, int(event["anchorMs"]))
        canonical_by_kind_and_anchor.setdefault(key, []).append(event)

    remaining = []
    for event in projected:
        key = (str(event["eventType"]), int(event["anchorMs"]))
        matches = [
            match
            for anchor in range(
                key[1] - INVENTORY_CANONICAL_ANCHOR_TOLERANCE_MS,
                key[1] + INVENTORY_CANONICAL_ANCHOR_TOLERANCE_MS + 1,
            )
            for match in canonical_by_kind_and_anchor.get((key[0], anchor), ())
        ]
        if key[0] == "level_up_transaction":
            matches = [
                match for match in matches
                if match.get("action") == "select_reward"
                and (not match.get("itemName") or not event.get("itemName")
                     or match["itemName"] == event["itemName"])
            ]
        if not matches:
            remaining.append(dict(event))
    return remaining


def build_dashboard_release(
    *, video_asset_id: str, session_id: str, duration_ms: int,
    event_files: Sequence[Path], gem_events: Sequence[Mapping[str, Any]] = (),
    event_sources: Sequence[Mapping[str, Any]] = (),
    inventory_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one data-complete release; unresolved source rows remain excluded."""

    events = []
    sources = [{"path": path} for path in event_files] + [dict(source) for source in event_sources]
    for source in sources:
        path = Path(source["path"])
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            if row.get("publication_status") == "unresolved":
                continue
            maximum_character_level = source.get("maximumCharacterLevel")
            if maximum_character_level is not None and int(row.get("character_level") or 0) > int(maximum_character_level):
                continue
            event = project_canonical_event(row)
            offset_ms = int(source.get("offsetMs") or 0)
            if offset_ms:
                for key in ("startMs", "endMs", "anchorMs"):
                    event[key] += offset_ms
            event["eventId"] = f"{source.get('eventIdPrefix') or ''}{event['eventId']}"
            events.append(event)
    canonical_events = list(events)
    events.extend(project_exact_gem_events(gem_events))
    events.extend(_unrepresented_inventory_events(
        project_inventory_selection_events(inventory_events), canonical_events,
    ))
    unique = {event["eventId"]: event for event in events}
    events = canonicalize_lucky_level_ups(canonicalize_weapon_evolutions(list(unique.values())))
    by_type = dict(sorted(Counter(row["eventType"] for row in events).items()))
    gem_types = {f"{gem_type}_gem_pickup" for gem_type in GEM_LABELS}
    coverage = []
    for event_type, count in by_type.items():
        coverage.append(
            {
                "event_type": event_type,
                "status": "events_published",
                "published_count": count,
                "note": (
                    "Exact automated XP/gem events from the framework detector."
                    if event_type in gem_types else None
                ),
            }
        )
    accepted = sum(row["publicationStatus"] == "auto_accepted" for row in events)
    return {
        "schemaVersion": "vss-framework-release-v1",
        "preparedBy": "Tahereh Fahi",
        "videoAssetId": video_asset_id,
        "sessionId": session_id,
        "durationMs": duration_ms,
        "humanCodedGroundTruthUsed": False,
        "databaseUsed": False,
        "summary": {
            "eventCount": len(events),
            "acceptedCount": accepted,
            "reviewCount": len(events) - accepted,
            "byEventType": by_type,
        },
        "coverage": coverage,
        "events": events,
    }


def merge_dashboard_releases(
    *, video_asset_id: str, session_id: str, duration_ms: int,
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge separately processed video parts on one media-time axis."""

    events: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments, start=1):
        release = segment["release"]
        offset_ms = int(segment.get("offsetMs") or 0)
        prefix = str(segment.get("eventIdPrefix") or f"part{segment_index}_")
        for source in release.get("events", []):
            event = dict(source)
            event["eventId"] = f"{prefix}{event['eventId']}"
            for key in ("startMs", "endMs", "anchorMs"):
                if isinstance(event.get(key), (int, float)):
                    event[key] += offset_ms
            events.append(event)

    unique = {event["eventId"]: event for event in events}
    events = canonicalize_lucky_level_ups(canonicalize_weapon_evolutions(list(unique.values())))
    by_type = dict(sorted(Counter(row["eventType"] for row in events).items()))
    accepted = sum(row["publicationStatus"] == "auto_accepted" for row in events)
    return {
        "schemaVersion": "vss-framework-release-v1",
        "preparedBy": "Tahereh Fahi",
        "videoAssetId": video_asset_id,
        "sessionId": session_id,
        "durationMs": duration_ms,
        "humanCodedGroundTruthUsed": False,
        "databaseUsed": False,
        "summary": {
            "eventCount": len(events),
            "acceptedCount": accepted,
            "reviewCount": len(events) - accepted,
            "byEventType": by_type,
        },
        "coverage": [
            {
                "event_type": event_type,
                "status": "events_published",
                "published_count": count,
                "note": "Merged from separately processed source-video segments.",
            }
            for event_type, count in by_type.items()
        ],
        "events": events,
    }
