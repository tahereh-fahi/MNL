"""Export auditable pause intervals from the fresh per-frame XP scan."""
from __future__ import annotations

import math
from typing import Iterable, Mapping


MAX_FRAGMENT_GAP_FRAMES = 1
MAX_ORPHAN_TRANSITION_FRAMES = 2
MAX_LEADING_TRANSITION_GAP_FRAMES = 4
MAX_LEADING_TRANSITION_FRAMES = 6
MAX_TREASURE_GAP_SECONDS = 0.25
CHEST_INTERVAL_ASSOCIATION_SECONDS = 0.50
LEVEL_UP_LEADING_ASSOCIATION_SECONDS = 0.75
LEVEL_UP_TRAILING_TRANSITION_SECONDS = 2.50


def _is_short_transition_fragment(interval: Mapping) -> bool:
    phases = {str(phase) for phase in interval["phases"]}
    return (
        int(interval["endFrameExclusive"]) - int(interval["startFrame"])
        <= MAX_ORPHAN_TRANSITION_FRAMES
        and bool(phases)
        and all(phase.endswith("_transition") for phase in phases)
    )


def _is_leading_transition_fragment(interval: Mapping) -> bool:
    phases = {str(phase) for phase in interval["phases"]}
    return (
        int(interval["endFrameExclusive"]) - int(interval["startFrame"])
        <= MAX_LEADING_TRANSITION_FRAMES
        and bool(phases)
        and all(phase.endswith("_transition") for phase in phases)
    )


def _is_transition_dominant_level_up_fragment(interval: Mapping) -> bool:
    """Return true for a short level-up onset mostly seen as transition.

    A level-up panel can become visible for one sampled frame, disappear from
    the classifier for a frame, and then become stable.  That first component
    is still an onset fragment, not a separate menu transaction.  Requiring
    transition frames to outnumber menu frames keeps substantive adjacent
    menus separate.
    """
    phases = {str(phase) for phase in interval["phases"]}
    counts = interval.get("_phaseFrameCounts", {})
    transition_frames = sum(
        int(count) for phase, count in counts.items()
        if str(phase).endswith("_transition")
    )
    menu_frames = int(counts.get("level_up_menu", 0))
    return (
        int(interval["endFrameExclusive"]) - int(interval["startFrame"])
        <= MAX_LEADING_TRANSITION_FRAMES
        and bool(phases)
        and phases <= {"level_up_transition", "level_up_menu"}
        and transition_frames > menu_frames
    )


def _classify_interval(interval: dict) -> None:
    phases = {str(phase) for phase in interval["phases"]}
    if "level_up_menu" in phases:
        event_type = "level_up_pause"
    elif "treasure_menu" in phases:
        event_type = "treasure_pause"
    elif phases and all(phase.startswith("treasure_") for phase in phases):
        event_type = "treasure_pause"
    else:
        event_type = "unresolved_interruption_pause"
    interval["eventType"] = event_type
    interval["eventId"] = f"{event_type}_{interval['startFrame']}"


def _is_treasure_interval(interval: Mapping) -> bool:
    phases = {str(phase) for phase in interval["phases"]}
    return "treasure_menu" in phases or (
        bool(phases) and all(phase.startswith("treasure_") for phase in phases)
    )


def _merge_interval(previous: dict, interval: Mapping, fps: float, *, reason: str) -> None:
    gap = int(interval["startFrame"]) - int(previous["endFrameExclusive"])
    components = previous.pop("mergedComponents", [
        {
            "startFrame": previous["startFrame"],
            "endFrameExclusive": previous["endFrameExclusive"],
            "phases": list(previous["phases"]),
        }
    ])
    components.append({
        "startFrame": interval["startFrame"],
        "endFrameExclusive": interval["endFrameExclusive"],
        "phases": list(interval["phases"]),
    })
    previous["endFrameExclusive"] = interval["endFrameExclusive"]
    previous["endMs"] = round(int(interval["endFrameExclusive"]) / fps * 1000)
    for phase in interval["phases"]:
        if phase not in previous["phases"]:
            previous["phases"].append(phase)
    previous_counts = previous.setdefault("_phaseFrameCounts", {})
    for phase, count in interval.get("_phaseFrameCounts", {}).items():
        previous_counts[phase] = int(previous_counts.get(phase, 0)) + int(count)
    previous["mergedComponents"] = components
    previous["mergeReason"] = reason
    previous["mergedGapFrames"] = previous.get("mergedGapFrames", 0) + gap


def _merge_fragmented_intervals(intervals: list[dict], fps: float) -> list[dict]:
    """Join UI fragments separated by brief classifier dropouts.

    Treasure animations can momentarily lose every visual anchor while coins,
    flashes, or the opening chest occlude the panel.  Those gaps are part of
    one interaction, so treasure-labelled components may bridge up to 250 ms.
    The broader rule is deliberately treasure-only: two adjacent level-up
    menus must remain separate transactions.
    """
    merged: list[dict] = []
    max_treasure_gap_frames = max(1, round(fps * MAX_TREASURE_GAP_SECONDS))
    for interval in intervals:
        if merged:
            previous = merged[-1]
            gap = interval["startFrame"] - previous["endFrameExclusive"]
            if (
                0 <= gap <= max_treasure_gap_frames
                and _is_treasure_interval(previous)
                and _is_treasure_interval(interval)
            ):
                _merge_interval(
                    previous,
                    interval,
                    fps,
                    reason="treasure_components_across_brief_visual_dropout",
                )
                continue
            if (
                0 <= gap <= MAX_FRAGMENT_GAP_FRAMES
                and (
                    _is_short_transition_fragment(previous)
                    or _is_short_transition_fragment(interval)
                )
            ):
                _merge_interval(
                    previous,
                    interval,
                    fps,
                    reason="short_transition_fragment_across_one_frame_gap",
                )
                continue
            if (
                0 <= gap <= MAX_LEADING_TRANSITION_GAP_FRAMES
                and _is_leading_transition_fragment(previous)
                and "level_up_menu" in interval["phases"]
            ):
                _merge_interval(
                    previous,
                    interval,
                    fps,
                    reason="leading_transition_fragment_before_level_up_menu",
                )
                continue
            if (
                0 <= gap <= MAX_LEADING_TRANSITION_GAP_FRAMES
                and _is_transition_dominant_level_up_fragment(previous)
                and "level_up_menu" in interval["phases"]
            ):
                _merge_interval(
                    previous,
                    interval,
                    fps,
                    reason="transition_dominant_level_up_onset_before_stable_menu",
                )
                continue
            if (
                0 <= gap <= MAX_LEADING_TRANSITION_GAP_FRAMES
                and "level_up_menu" in previous["phases"]
                and _is_leading_transition_fragment(interval)
            ):
                _merge_interval(
                    previous,
                    interval,
                    fps,
                    reason="trailing_transition_fragment_after_level_up_menu",
                )
                continue
        merged.append(interval)
    return merged


def _reconcile_treasure_pauses(intervals: list[dict], chest_events: Iterable[Mapping], fps: float) -> list[dict]:
    """Use accepted chest lifecycles to bridge heavily occluded animations."""
    output = list(intervals)
    association_ms = round(CHEST_INTERVAL_ASSOCIATION_SECONDS * 1000)
    matched_ids: set[int] = set()
    additions: list[dict] = []
    for chest in sorted(chest_events, key=lambda row: int(row["time_lower_ms"])):
        if str(chest.get("publication_status")) != "auto_accepted":
            continue
        lower = int(chest["time_lower_ms"])
        upper = int(chest["time_upper_ms"])
        matched = [
            interval for interval in output
            if interval.get("eventType") == "treasure_pause"
            and int(interval["endMs"]) >= lower - association_ms
            and int(interval["startMs"]) <= upper + association_ms
        ]
        matched_ids.update(id(interval) for interval in matched)
        start_ms = min([lower, *(int(interval["startMs"]) for interval in matched)])
        end_ms = max([upper, *(int(interval["endMs"]) for interval in matched)])
        phases: list[str] = []
        components: list[dict] = []
        for interval in matched:
            for phase in interval.get("phases", []):
                if phase not in phases:
                    phases.append(phase)
            components.extend(interval.get("mergedComponents", [{
                "startFrame": interval["startFrame"],
                "endFrameExclusive": interval["endFrameExclusive"],
                "phases": list(interval.get("phases", [])),
            }]))
        if "treasure_menu" not in phases:
            phases.append("treasure_menu")
        start_frame = math.floor(start_ms / 1000 * fps)
        end_frame = math.ceil(end_ms / 1000 * fps)
        additions.append({
            "eventId": f"treasure_pause_{start_frame}",
            "eventType": "treasure_pause",
            "startFrame": start_frame,
            "endFrameExclusive": end_frame,
            "startMs": start_ms,
            "endMs": end_ms,
            "phases": phases,
            "mergedComponents": components,
            "mergeReason": "accepted_chest_lifecycle_bridges_visual_occlusion",
            "sourceChestEventId": chest.get("event_id"),
            "chestEvidenceIntervalMs": [lower, upper],
        })
    retained = []
    for interval in output:
        if id(interval) in matched_ids:
            continue
        if interval.get("eventType") == "treasure_pause":
            interval = dict(interval)
            interval["eventType"] = "unresolved_interruption_pause"
            interval["eventId"] = f"unresolved_interruption_pause_{interval['startFrame']}"
            interval["unmatchedTreasureVisualEvidence"] = True
        retained.append(interval)
    return sorted([*retained, *additions], key=lambda interval: int(interval["startFrame"]))


def _reconcile_level_up_pauses(
    intervals: list[dict], level_up_events: Iterable[Mapping], fps: float
) -> list[dict]:
    """Join visual fragments belonging to an accepted inventory transaction.

    A selected Level Up card can collapse through several compressed or
    partially transparent frames.  Those frames may intermittently lose both
    the purple panel and the option-row geometry, creating many apparent
    pauses after one recorded selection.  The inventory event supplies the
    transaction anchor; it never creates a pause without visual evidence.

    Menu-bearing components must occur at or immediately before the selection.
    The wider post-selection allowance is restricted to transition-only
    components, so a later stable menu remains a separate transaction.
    """
    events = []
    for event in level_up_events:
        if str(event.get("event_source", "")) not in {
            "level_up", "level_up_retrospective"
        }:
            continue
        try:
            event_ms = round(float(event["video_second"]) * 1000)
        except (KeyError, TypeError, ValueError):
            continue
        events.append((event_ms, event))
    if not events:
        return intervals

    assignments: dict[int, list[dict]] = {index: [] for index in range(len(events))}
    unmatched: list[dict] = []
    leading_ms = round(LEVEL_UP_LEADING_ASSOCIATION_SECONDS * 1000)
    trailing_ms = round(LEVEL_UP_TRAILING_TRANSITION_SECONDS * 1000)
    for interval in intervals:
        phases = {str(phase) for phase in interval.get("phases", [])}
        if not phases or not phases <= {"level_up_transition", "level_up_menu"}:
            unmatched.append(interval)
            continue
        transition_only = phases == {"level_up_transition"}
        candidates: list[tuple[int, int]] = []
        for index, (event_ms, _) in enumerate(events):
            start_ms = int(interval["startMs"])
            end_ms = int(interval["endMs"])
            if start_ms <= event_ms <= end_ms:
                distance = 0
            elif end_ms < event_ms and event_ms - end_ms <= leading_ms:
                distance = event_ms - end_ms
            elif (
                transition_only
                and start_ms > event_ms
                and start_ms - event_ms <= trailing_ms
            ):
                distance = start_ms - event_ms
            else:
                continue
            candidates.append((distance, index))
        if not candidates:
            unmatched.append(interval)
            continue
        _, best_index = min(candidates)
        assignments[best_index].append(interval)

    reconciled: list[dict] = []
    for index, matched in assignments.items():
        if not matched:
            continue
        _, event = events[index]
        matched.sort(key=lambda item: int(item["startFrame"]))
        components: list[dict] = []
        phases: list[str] = []
        for interval in matched:
            parts = interval.get("mergedComponents") or [{
                "startFrame": interval["startFrame"],
                "endFrameExclusive": interval["endFrameExclusive"],
                "phases": list(interval["phases"]),
            }]
            components.extend(parts)
            for phase in interval["phases"]:
                if phase not in phases:
                    phases.append(phase)
        start_frame = min(int(item["startFrame"]) for item in matched)
        end_frame = max(int(item["endFrameExclusive"]) for item in matched)
        source_event_ids = [
            str(other.get("event_id") or f"inventory_level_up_{other_index}")
            for other_index, (other_ms, other) in enumerate(events)
            if round(start_frame / fps * 1000) <= other_ms <= round(end_frame / fps * 1000)
        ]
        event_id = str(event.get("event_id") or f"inventory_level_up_{index}")
        if event_id not in source_event_ids:
            source_event_ids.insert(0, event_id)
        reconciled.append({
            "eventId": f"level_up_pause_{start_frame}",
            "eventType": "level_up_pause",
            "startFrame": start_frame,
            "endFrameExclusive": end_frame,
            "startMs": round(start_frame / fps * 1000),
            "endMs": round(end_frame / fps * 1000),
            "phases": phases,
            "mergedComponents": components,
            "mergedGapFrames": sum(
                max(0, int(right["startFrame"]) - int(left["endFrameExclusive"]))
                for left, right in zip(matched, matched[1:])
            ),
            "mergeReason": "accepted_inventory_level_up_lifecycle",
            "sourceInventoryEventId": event_id,
            "sourceInventoryEventIds": source_event_ids,
        })
    return sorted([*unmatched, *reconciled], key=lambda item: int(item["startFrame"]))


def build_gameplay_pauses(
    rows: Iterable[Mapping],
    fps: float,
    chest_events: Iterable[Mapping] | None = None,
    level_up_events: Iterable[Mapping] | None = None,
) -> dict:
    intervals = []
    current = None
    version = None
    for row in rows:
        version = row.get("gameplay_state_version") or version
        blocked = str(row.get("gameplay_paused", "")).lower() in {"1", "true"}
        frame = int(row["frame_index"])
        if not blocked:
            current = None
            continue
        if current is None or current["endFrameExclusive"] != frame:
            current = {"eventId": f"interruption_pause_{frame}", "eventType": "unresolved_interruption_pause",
                       "startFrame": frame, "endFrameExclusive": frame + 1,
                       "startMs": round(frame / fps * 1000),
                       "endMs": round((frame + 1) / fps * 1000), "phases": [],
                       "_phaseFrameCounts": {}}
            intervals.append(current)
        current["endFrameExclusive"] = frame + 1
        current["endMs"] = round((frame + 1) / fps * 1000)
        phase = str(row.get("gameplay_phase", "unclassified"))
        if phase not in current["phases"]:
            current["phases"].append(phase)
        current["_phaseFrameCounts"][phase] = current["_phaseFrameCounts"].get(phase, 0) + 1
    intervals = _merge_fragmented_intervals(intervals, fps)
    for interval in intervals:
        _classify_interval(interval)
    if level_up_events is not None:
        intervals = _reconcile_level_up_pauses(intervals, level_up_events, fps)
    if chest_events is not None:
        intervals = _reconcile_treasure_pauses(intervals, chest_events, fps)
    for interval in intervals:
        interval.pop("_phaseFrameCounts", None)
    return {"prepared_by": "Tahereh Fahi", "classifier_version": version,
            "publication_ready": False, "frame_indexing": "zero_based_decoded_source_frames",
            "interval_semantics": "half_open_start_inclusive_end_exclusive",
            "policy": "No gameplay measurement inside level-up or treasure UI; event-specific evidence remains observable",
            "boundary_policy": "Level-up evidence retained; gem quantity at boundary unobserved",
            "merge_policy": (
                "Merge transition-only components of at most two frames across "
                "one frame; merge treasure components across visual dropouts of "
                "at most 250 ms; join short transition-dominant level-up onset "
                "or trailing fragments of at most six frames to the same menu "
                "across at most four frames; accepted chest "
                "lifecycles bridge longer visually occluded treasure animation; "
                "accepted inventory transactions join fragmented Level Up "
                "selection and collapse animation"
            ),
            "intervals": intervals}
