"""Reconstruct player game-level progression from automated XP/HUD evidence.

Game level is deliberately separate from weapon and passive-item levels.  The
resolver accepts only consecutive player-level transitions and preserves gaps
as unresolved evidence instead of turning noisy HUD OCR jumps into levels.
"""

from __future__ import annotations

from copy import deepcopy
from bisect import bisect_right
from typing import Any, Mapping, Sequence


GAMEPLAY_INTERRUPTION_EVENT_TYPES = {
    "level_up_pause",
    "level_up_transaction",
    "lucky_level_up",
    "match_transition",
    "death",
}

NON_COMPLETING_LEVEL_UP_ACTIONS = {"banish", "reroll", "skip"}


def _boolean(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _integer(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_gameplay_interruption_intervals(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return merged media-time intervals during which gameplay cannot advance XP."""

    intervals: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("eventType") or event.get("event_type") or "")
        if not (
            event_type in GAMEPLAY_INTERRUPTION_EVENT_TYPES
            or event_type.startswith("loot_box_tier_")
        ):
            continue
        start = _integer(event.get("startMs") if "startMs" in event else event.get("time_lower_ms"))
        end_value = event.get("endMs") if "endMs" in event else event.get("time_upper_ms")
        anchor_value = event.get("anchorMs") if "anchorMs" in event else event.get("anchor_time_ms")
        available_ends = [value for value in (_integer(end_value), _integer(anchor_value)) if value is not None]
        if start is None or not available_ends:
            continue
        end = max(available_ends)
        if end <= start:
            continue
        intervals.append(
            {
                "startMs": start,
                "endMs": end,
                "eventTypes": [event_type],
                "eventIds": [str(event.get("eventId") or event.get("event_id") or "")],
            }
        )

    intervals.sort(key=lambda interval: (interval["startMs"], interval["endMs"]))
    merged: list[dict[str, Any]] = []
    for interval in intervals:
        previous = merged[-1] if merged else None
        if previous is None or interval["startMs"] > previous["endMs"]:
            merged.append(deepcopy(interval))
            continue
        previous["endMs"] = max(previous["endMs"], interval["endMs"])
        previous["eventTypes"] = sorted(set(previous["eventTypes"] + interval["eventTypes"]))
        previous["eventIds"] = sorted(
            value for value in set(previous["eventIds"] + interval["eventIds"]) if value
        )
    return merged


def apply_gameplay_interruptions(
    progress: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze XP fill across verified non-gameplay intervals.

    Measured points inside a blocking interval are suppressed. Carry-forward
    anchors preserve a flat XP value, while confirmed 100 -> 0 level resets
    remain state transitions rather than XP gained during stopped play.
    """

    result = deepcopy(dict(progress))
    intervals = build_gameplay_interruption_intervals(events)
    original_points = sorted(
        (
            deepcopy(dict(point))
            for point in progress.get("points", [])
            if not point.get("interruptionHold")
        ),
        key=lambda point: (
            int(point["mediaTimeMs"]),
            0 if point.get("boundaryAnchor") == "level_end" else 1,
        ),
    )
    source_points = list(original_points)
    changes = sorted(
        (dict(change) for change in progress.get("levelChanges", [])),
        key=lambda change: int(change["mediaTimeMs"]),
    )
    holds: list[dict[str, Any]] = []

    def state_at(time_ms: int) -> dict[str, Any] | None:
        eligible = [point for point in source_points if int(point["mediaTimeMs"]) <= time_ms]
        return eligible[-1] if eligible else None

    for interval_index, interval in enumerate(intervals):
        boundary_times = sorted(
            {
                int(change["mediaTimeMs"])
                for change in changes
                if interval["startMs"] < int(change["mediaTimeMs"]) < interval["endMs"]
            }
        )
        cuts = [interval["startMs"], *boundary_times, interval["endMs"]]
        for start_ms, end_ms in zip(cuts, cuts[1:]):
            if end_ms <= start_ms:
                continue
            state = state_at(start_ms)
            if state is None:
                continue
            hold_id = f"interruption_{interval_index}_{start_ms}_{end_ms}"
            hold = {
                "startMs": start_ms,
                "endMs": end_ms,
                "hudLevel": int(state["hudLevel"]),
                "progressPercent": float(state["progressPercent"]),
                "eventTypes": list(interval["eventTypes"]),
                "eventIds": list(interval["eventIds"]),
                "policy": "last_observation_carried_forward_no_xp_gain",
            }
            holds.append(hold)
            for endpoint, time_ms in (("start", start_ms), ("end", end_ms)):
                original_points.append(
                    {
                        "mediaTimeMs": time_ms,
                        "frameNumber": None,
                        "hudLevel": hold["hudLevel"],
                        "progressPercent": hold["progressPercent"],
                        "quality": float(state.get("quality", 1.0)),
                        "breakBefore": endpoint == "start",
                        "sourceType": "Automated",
                        "levelSource": "gameplay_interruption_hold",
                        "interruptionHold": endpoint,
                        "interruptionId": hold_id,
                        "interruptionEventTypes": list(interval["eventTypes"]),
                    }
                )

    def strictly_inside_interruption(point: Mapping[str, Any]) -> bool:
        if point.get("boundaryAnchor") or point.get("interruptionHold"):
            return False
        time_ms = int(point["mediaTimeMs"])
        return any(interval["startMs"] < time_ms < interval["endMs"] for interval in intervals)

    original_points = [point for point in original_points if not strictly_inside_interruption(point)]
    original_points.sort(
        key=lambda point: (
            int(point["mediaTimeMs"]),
            0 if point.get("boundaryAnchor") == "level_end" else 1,
            0 if point.get("interruptionHold") == "start" else 1,
        )
    )
    deduplicated: list[dict[str, Any]] = []
    for point in original_points:
        identity = (
            int(point["mediaTimeMs"]),
            int(point["hudLevel"]),
            round(float(point["progressPercent"]), 4),
            str(point.get("interruptionId") or ""),
        )
        if deduplicated:
            prior_identity = (
                int(deduplicated[-1]["mediaTimeMs"]),
                int(deduplicated[-1]["hudLevel"]),
                round(float(deduplicated[-1]["progressPercent"]), 4),
                str(deduplicated[-1].get("interruptionId") or ""),
            )
            if identity == prior_identity:
                if point.get("boundaryAnchor") or point.get("interruptionHold"):
                    deduplicated[-1] = point
                continue
        deduplicated.append(point)

    previous: dict[str, Any] | None = None
    for point in deduplicated:
        point["breakBefore"] = (
            previous is None
            or point.get("interruptionHold") == "start"
            or point["hudLevel"] != previous["hudLevel"]
            or int(point["mediaTimeMs"]) - int(previous["mediaTimeMs"]) > 2_000
            or float(point["progressPercent"]) < float(previous["progressPercent"]) - 25
        )
        previous = point

    def overlaps_interruption(segment: Mapping[str, Any]) -> bool:
        return any(
            interval["startMs"] < int(segment["endMs"])
            and interval["endMs"] > int(segment["startMs"])
            for interval in intervals
        )

    result["points"] = deduplicated
    result["lessReliableSegments"] = [
        deepcopy(dict(segment))
        for segment in progress.get("lessReliableSegments", [])
        if not overlaps_interruption(segment)
    ]
    result["interruptionIntervals"] = intervals
    result["interruptionHolds"] = holds
    result["gameplayInterruptionRule"] = (
        "XP gain is zero during verified non-gameplay intervals; the last observed fill is carried forward, while confirmed level resets remain state transitions."
    )
    return result


def reconcile_game_levels_from_completed_selections(
    progress: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Correct an undercounted game-level sequence with completed menus.

    One resolved level-up transaction that closes with a selected reward is one
    completed player level.  This is especially important when several queued
    level-up menus appear back-to-back: the XP bar may expose only the first
    reset, while every completed menu still advances the player level.

    The correction is deliberately conservative.  It replaces the XP-derived
    sequence only when completed selections outnumber the published XP level
    changes.  Otherwise the original evidence remains authoritative.
    """

    result = deepcopy(dict(progress))
    original_changes = sorted(
        (deepcopy(dict(change)) for change in progress.get("levelChanges", [])),
        key=lambda change: int(change["mediaTimeMs"]),
    )
    transactions = sorted(
        (
            deepcopy(dict(event))
            for event in events
            if str(event.get("eventType") or event.get("event_type") or "")
            == "level_up_transaction"
            and str(event.get("action") or "").casefold()
            not in NON_COMPLETING_LEVEL_UP_ACTIONS
            and _integer(event.get("endMs") if "endMs" in event else event.get("time_upper_ms"))
            is not None
        ),
        key=lambda event: int(
            event.get("endMs") if "endMs" in event else event.get("time_upper_ms")
        ),
    )
    already_reconciled = bool(original_changes) and all(
        change.get("evidenceMethod") == "completed_level_up_transaction"
        for change in original_changes
    )
    if len(transactions) < len(original_changes) or (
        len(transactions) == len(original_changes) and not already_reconciled
    ):
        result["gameLevelReconciliationApplied"] = already_reconciled
        result["completedLevelUpTransactionCount"] = len(transactions)
        return result

    original_points = [
        deepcopy(dict(point))
        for point in progress.get("points", [])
        if not point.get("boundaryAnchor") and not point.get("interruptionHold")
    ]
    initial_level = _integer(original_changes[0].get("fromLevel")) if original_changes else None
    if initial_level is None:
        observed_levels = [
            _integer(point.get("hudLevel")) for point in original_points
            if _integer(point.get("hudLevel")) is not None
        ]
        initial_level = min(observed_levels) if observed_levels else 1

    completion_times = [
        int(event.get("endMs") if "endMs" in event else event.get("time_upper_ms"))
        for event in transactions
    ]
    changes: list[dict[str, Any]] = []
    for index, (event, time_ms) in enumerate(zip(transactions, completion_times), start=1):
        changes.append(
            {
                "mediaTimeMs": time_ms,
                "frameNumber": None,
                "fromLevel": initial_level + index - 1,
                "toLevel": initial_level + index,
                "progressPercent": 100,
                "sourceType": "Automated",
                "needsReview": str(event.get("publicationStatus") or "") == "needs_review",
                "evidenceMethod": "completed_level_up_transaction",
                "timingPrecision": "frame",
                "sourceEventId": str(event.get("eventId") or event.get("event_id") or ""),
            }
        )

    for point in original_points:
        point["hudLevel"] = initial_level + bisect_right(completion_times, int(point["mediaTimeMs"]))
        point["levelReconciledBy"] = "completed_level_up_transaction"

    initial_time_ms = min(
        (int(point["mediaTimeMs"]) for point in original_points),
        default=0,
    )
    original_points.append(
        {
            "mediaTimeMs": initial_time_ms,
            "frameNumber": None,
            "hudLevel": initial_level,
            "progressPercent": 0.0,
            "quality": 1.0,
            "breakBefore": False,
            "sourceType": "Automated",
            "levelSource": "segment_start",
            "boundaryAnchor": "level_start",
            "needsReview": False,
        }
    )

    for change in changes:
        for level, fill, role in (
            (change["fromLevel"], 100.0, "level_end"),
            (change["toLevel"], 0.0, "level_start"),
        ):
            original_points.append(
                {
                    "mediaTimeMs": change["mediaTimeMs"],
                    "frameNumber": None,
                    "hudLevel": level,
                    "progressPercent": fill,
                    "quality": 1.0,
                    "breakBefore": False,
                    "sourceType": "Automated",
                    "levelSource": "completed_level_up_transaction",
                    "boundaryAnchor": role,
                    "needsReview": change["needsReview"],
                    "sourceEventId": change["sourceEventId"],
                }
            )

    anchor_order = {"level_end": 0, "level_start": 1}
    original_points.sort(
        key=lambda point: (
            int(point["mediaTimeMs"]),
            anchor_order.get(str(point.get("boundaryAnchor")), 2),
            int(point.get("frameNumber") or -1),
        )
    )
    deduplicated: list[dict[str, Any]] = []
    for point in original_points:
        identity = (
            int(point["mediaTimeMs"]),
            int(point["hudLevel"]),
            round(float(point["progressPercent"]), 4),
        )
        if deduplicated and identity == (
            int(deduplicated[-1]["mediaTimeMs"]),
            int(deduplicated[-1]["hudLevel"]),
            round(float(deduplicated[-1]["progressPercent"]), 4),
        ):
            if point.get("boundaryAnchor"):
                deduplicated[-1] = point
            continue
        deduplicated.append(point)

    previous: dict[str, Any] | None = None
    for point in deduplicated:
        point["breakBefore"] = (
            previous is None
            or point["hudLevel"] != previous["hudLevel"]
            or int(point["mediaTimeMs"]) - int(previous["mediaTimeMs"]) > 2_000
            or float(point["progressPercent"]) < float(previous["progressPercent"]) - 25
        )
        previous = point

    reconciled_segments: list[dict[str, Any]] = []
    for source_segment in progress.get("lessReliableSegments", []):
        segment = deepcopy(dict(source_segment))
        start_ms = int(segment["startMs"])
        end_ms = int(segment["endMs"])
        start_level = initial_level + bisect_right(completion_times, start_ms)
        end_level = initial_level + bisect_right(completion_times, end_ms)
        if start_level != end_level:
            continue
        segment["hudLevel"] = start_level
        for point in segment.get("points", []):
            point["hudLevel"] = initial_level + bisect_right(
                completion_times, int(point["mediaTimeMs"])
            )
            point["levelReconciledBy"] = "completed_level_up_transaction"
        reconciled_segments.append(segment)

    result["points"] = deduplicated
    result["levelChanges"] = changes
    result["lessReliableSegments"] = reconciled_segments
    result["gameLevelReconciliationApplied"] = True
    result["completedLevelUpTransactionCount"] = len(transactions)
    result["gameLevelReconciliationRule"] = (
        "When completed level-up selections outnumber XP-reset level changes, each completed selection advances player game level by one; XP remains the within-level 0–100 fill signal."
    )
    return result


def build_xp_progress(
    rows: Sequence[Mapping[str, Any]],
    session_start_ms: int = 0,
    *,
    quality_threshold: float = 0.85,
    initial_level: int | None = None,
    frame_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project legacy XP rows into reliable progress and game-level changes.

    The first usable transition establishes the segment's starting level when
    ``initial_level`` is not supplied.  After that, only ``L -> L+1`` is
    published.  Regressions and jumps remain in ``unresolvedLevelChanges`` so
    downstream code can audit them without visualizing false levels.
    """

    points: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    first_reliable_by_level: dict[int, dict[str, Any]] = {}

    for source_index, row in enumerate(rows):
        hud_level = _integer(row.get("hud_level"))
        inferred_level = _integer(row.get("inferred_level"))
        reset_level = _integer(row.get("reset_inferred_level"))
        # XP resets provide the stable level identity for the bar. Prefer that
        # sequence when it is internally consistent; small HUD numerals are
        # frequently misread (for example 4 as 5 or 16 as 19).
        reset_level_supported = inferred_level is not None and reset_level == inferred_level
        level = inferred_level if reset_level_supported else hud_level or inferred_level
        if level is None:
            continue
        time_ms = session_start_ms + round(_number(row.get("video_time_b")) * 1000)
        frame_number = _integer(row.get("frame_b"))
        hud_accepted_field = row.get("hud_level_ocr_accepted")
        hud_accepted = hud_accepted_field is None or _boolean(hud_accepted_field)
        level_supported = reset_level_supported or hud_accepted
        level_source = "reset_inference" if reset_level_supported else "hud_ocr"
        boundary = _boolean(row.get("level_up_boundary_candidate"))
        if boundary and reset_level_supported:
            before = int(reset_level)
            after = before + 1
        else:
            before = _integer(row.get("hud_level_before")) or level
            after = _integer(row.get("hud_level_after")) or level

        def append_endpoint(
            *,
            suffix: str,
            endpoint_level: int,
            progress_key: str,
            fallback_progress_key: str | None = None,
        ) -> None:
            quality = _optional_number(row.get(f"xp_quality_{suffix}"))
            progress = _optional_number(row.get(progress_key))
            if progress is None and fallback_progress_key is not None:
                progress = _optional_number(row.get(fallback_progress_key))
            endpoint_time = _optional_number(row.get(f"video_time_{suffix}"))
            if (
                not level_supported
                or quality is None
                or quality < quality_threshold
                or progress is None
                or endpoint_time is None
            ):
                return
            points.append(
                {
                    "mediaTimeMs": session_start_ms + round(endpoint_time * 1000),
                    "frameNumber": _integer(row.get(f"frame_{suffix}")),
                    "hudLevel": endpoint_level,
                    "progressPercent": max(0.0, min(100.0, progress)),
                    "quality": quality,
                    "breakBefore": False,
                    "sourceType": "Automated",
                    "levelSource": level_source,
                    "endpoint": suffix.upper(),
                }
            )

        # Each detector row describes a measured A -> B fill interval. Keeping
        # only B made isolated one-point SVG polylines and discarded the 100 ->
        # 0 reset evidence at level boundaries.
        append_endpoint(
            suffix="a",
            endpoint_level=before if boundary else level,
            progress_key="xp_bar_progress_a_percent",
            fallback_progress_key="xp_progress_a",
        )
        append_endpoint(
            suffix="b",
            endpoint_level=after if boundary else level,
            progress_key="xp_bar_progress_b_effective_percent",
            fallback_progress_key="xp_bar_progress_b_percent",
        )
        if any(point["mediaTimeMs"] == time_ms and point["hudLevel"] == (after if boundary else level) for point in points[-2:]):
            first_reliable_by_level.setdefault(
                after if boundary else level,
                {
                    "mediaTimeMs": time_ms,
                    "frameNumber": frame_number,
                    "fromLevel": None,
                    "toLevel": after if boundary else level,
                    "progressPercent": 100,
                    "sourceType": "Automated",
                    "needsReview": True,
                    "evidenceMethod": "first_visible_hud_level",
                    "sourceRowIndex": source_index,
                },
            )

        if (
            boundary
            and before is not None
            and after is not None
            and after > before
        ):
            candidates.append(
                {
                    "mediaTimeMs": time_ms,
                    "frameNumber": frame_number,
                    "fromLevel": before,
                    "toLevel": after,
                    "progressPercent": 100,
                    "sourceType": "Automated",
                    "needsReview": _boolean(row.get("needs_review")) or _boolean(row.get("post_level_up_review")),
                    "evidenceMethod": "xp_reset_boundary",
                    "sourceRowIndex": source_index,
                }
            )
    candidates.extend(first_reliable_by_level.values())
    candidates.sort(
        key=lambda row: (
            int(row["mediaTimeMs"]),
            row["evidenceMethod"] == "first_visible_hud_level",
            int(row["sourceRowIndex"]),
        )
    )
    changes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    expected_from = initial_level
    if expected_from is None and points:
        expected_from = int(points[0]["hudLevel"])
    for candidate_index, candidate in enumerate(candidates):
        if expected_from is None:
            expected_from = candidate["fromLevel"] if candidate["fromLevel"] is not None else candidate["toLevel"]
        if candidate["toLevel"] <= expected_from:
            continue
        exact_consecutive = (
            candidate["fromLevel"] == expected_from
            and candidate["toLevel"] == expected_from + 1
        )
        first_visible_next = (
            candidate["evidenceMethod"] == "first_visible_hud_level"
            and candidate["toLevel"] == expected_from + 1
        )
        locally_consecutive_after_gap = (
            candidate["fromLevel"] is not None
            and candidate["fromLevel"] > expected_from
            and candidate["toLevel"] == candidate["fromLevel"] + 1
            and not any(
                future["fromLevel"] is not None
                and expected_from <= future["fromLevel"] < candidate["fromLevel"]
                and future["toLevel"] == future["fromLevel"] + 1
                for future in candidates[candidate_index + 1:]
            )
        )
        if exact_consecutive or first_visible_next or locally_consecutive_after_gap:
            event = dict(candidate)
            event.pop("sourceRowIndex", None)
            if first_visible_next:
                event["fromLevel"] = expected_from
                event["needsReview"] = True
                event["timingPrecision"] = "upper_bound"
            if locally_consecutive_after_gap:
                event["gapBefore"] = True
                event["missingPrecedingLevels"] = list(
                    range(expected_from + 1, int(event["fromLevel"]) + 1)
                )
                event["needsReview"] = True
            changes.append(event)
            expected_from = event["toLevel"]
        else:
            unresolved.append(
                {
                    **candidate,
                    "expectedFromLevel": expected_from,
                    "reason": (
                        "first_visible_hud_level_skips_intermediate_levels"
                        if candidate["fromLevel"] is None
                        else "non_consecutive_hud_or_xp_transition"
                        if candidate["toLevel"] != candidate["fromLevel"] + 1
                        else "transition_conflicts_with_monotonic_game_level"
                    ),
                }
            )

    # Materialize the semantics of a completed level: the old bar reaches
    # 100%, and the new level begins at 0%. These are detector boundary
    # anchors, not interpolated samples.
    for change in changes:
        from_level = _integer(change.get("fromLevel"))
        to_level = _integer(change.get("toLevel"))
        if from_level is None or to_level is None:
            continue
        for anchor_level, anchor_progress, anchor_role in (
            (from_level, 100.0, "level_end"),
            (to_level, 0.0, "level_start"),
        ):
            points.append(
                {
                    "mediaTimeMs": int(change["mediaTimeMs"]),
                    "frameNumber": change.get("frameNumber"),
                    "hudLevel": anchor_level,
                    "progressPercent": anchor_progress,
                    "quality": 1.0,
                    "breakBefore": False,
                    "sourceType": "Automated",
                    "levelSource": "xp_reset_boundary",
                    "boundaryAnchor": anchor_role,
                    "needsReview": bool(change.get("needsReview")),
                }
            )

    # A recording segment may begin before the first detected XP increase, so
    # its initial level has no preceding reset event to supply the 0% anchor.
    # Preserve that logical start explicitly without inventing intermediate
    # measurements across the unobserved interval.
    if points:
        first_observed = min(points, key=lambda point: int(point["mediaTimeMs"]))
        first_level = int(initial_level if initial_level is not None else first_observed["hudLevel"])
        if not any(
            int(point["hudLevel"]) == first_level
            and float(point["progressPercent"]) == 0.0
            for point in points
        ):
            points.append(
                {
                    "mediaTimeMs": session_start_ms,
                    "frameNumber": None,
                    "hudLevel": first_level,
                    "progressPercent": 0.0,
                    "quality": 1.0,
                    "breakBefore": False,
                    "sourceType": "Automated",
                    "levelSource": "segment_start",
                    "boundaryAnchor": "level_start",
                    "needsReview": True,
                }
            )

    anchor_order = {"level_end": 0, "level_start": 1}
    points.sort(
        key=lambda point: (
            int(point["mediaTimeMs"]),
            anchor_order.get(str(point.get("boundaryAnchor")), 0),
            int(point.get("frameNumber") or -1),
        )
    )
    deduplicated: list[dict[str, Any]] = []
    for point in points:
        identity = (
            int(point["mediaTimeMs"]),
            int(point["hudLevel"]),
            round(float(point["progressPercent"]), 4),
        )
        if deduplicated and identity == (
            int(deduplicated[-1]["mediaTimeMs"]),
            int(deduplicated[-1]["hudLevel"]),
            round(float(deduplicated[-1]["progressPercent"]), 4),
        ):
            if point.get("boundaryAnchor"):
                deduplicated[-1] = point
            continue
        deduplicated.append(point)
    points = deduplicated
    previous_point: dict[str, Any] | None = None
    for point in points:
        point["breakBefore"] = (
            previous_point is None
            or point["hudLevel"] != previous_point["hudLevel"]
            or int(point["mediaTimeMs"]) - int(previous_point["mediaTimeMs"]) > 2_000
            or float(point["progressPercent"]) < float(previous_point["progressPercent"]) - 25
        )
        previous_point = point

    less_reliable = build_less_reliable_xp_segments(
        points, frame_rows or (), session_start_ms=session_start_ms
    )
    return {
        "sourceType": "Automated",
        "timingPrecision": "frame",
        "progressMeaning": "XP bar fill percentage within the detected player HUD level",
        "levelMeaning": "player game level only; weapon and passive-item levels are excluded",
        "reliabilityRule": (
            f"XP quality >= {quality_threshold:.2f}; game-level events must form consecutive L to L+1 transitions"
        ),
        "points": points,
        "lessReliableSegments": less_reliable,
        "levelChanges": changes,
        "unresolvedLevelChanges": unresolved,
    }


def build_less_reliable_xp_segments(
    reliable_points: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    session_start_ms: int = 0,
) -> list[dict[str, Any]]:
    """Bridge same-level gaps with measured XP values, visibly marked lower reliability."""

    segments: list[dict[str, Any]] = []
    for start, end in zip(reliable_points, reliable_points[1:]):
        same_level_gap = (
            bool(end.get("breakBefore"))
            and start.get("hudLevel") == end.get("hudLevel")
            and int(end["mediaTimeMs"]) - int(start["mediaTimeMs"]) > 2_000
            and float(end["progressPercent"]) >= float(start["progressPercent"])
        )
        if not same_level_gap:
            continue
        measured = []
        for row in frame_rows:
            time_ms = session_start_ms + round(_number(row.get("video_time")) * 1000)
            if not int(start["mediaTimeMs"]) < time_ms < int(end["mediaTimeMs"]):
                continue
            if not _boolean(row.get("xp_measurement_valid")):
                continue
            progress = _number(row.get("xp_progress_percent"), -1)
            quality = _number(row.get("xp_quality"), -1)
            if not 0 <= progress <= 100 or not 0 <= quality <= 1:
                continue
            measured.append(
                {
                    "mediaTimeMs": time_ms,
                    "frameNumber": _integer(row.get("frame_index")),
                    "hudLevel": start["hudLevel"],
                    "progressPercent": round(progress, 4),
                    "quality": round(quality, 4),
                    "sourceType": "Automated",
                    "reliability": "lower",
                    "levelSource": "same_level_endpoint_inference",
                }
            )
        if measured:
            segments.append(
                {
                    "hudLevel": start["hudLevel"],
                    "startMs": start["mediaTimeMs"],
                    "endMs": end["mediaTimeMs"],
                    "points": [dict(start), *measured, dict(end)],
                }
            )
    return segments


def merge_xp_progress(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge separately processed XP/Game-Level segments on one session axis."""

    if not segments:
        return build_xp_progress([])
    result = {
        key: value
        for key, value in dict(segments[0]["progress"]).items()
        if key not in {"points", "levelChanges", "unresolvedLevelChanges", "lessReliableSegments", "interruptionIntervals", "interruptionHolds"}
    }
    result.update({"points": [], "levelChanges": [], "unresolvedLevelChanges": [], "lessReliableSegments": [], "interruptionIntervals": [], "interruptionHolds": []})

    def shift_row(source: Mapping[str, Any], offset_ms: int, level_offset: int) -> dict[str, Any]:
        row = dict(source)
        for key in ("mediaTimeMs", "startMs", "endMs"):
            if isinstance(row.get(key), (int, float)):
                row[key] += offset_ms
        for key in ("hudLevel", "fromLevel", "toLevel", "expectedFromLevel"):
            if isinstance(row.get(key), (int, float)):
                row[key] += level_offset
        if isinstance(row.get("missingPrecedingLevels"), list):
            row["missingPrecedingLevels"] = [level + level_offset for level in row["missingPrecedingLevels"]]
        if isinstance(row.get("points"), list):
            row["points"] = [shift_row(point, offset_ms, level_offset) for point in row["points"]]
        return row

    for segment in segments:
        progress = segment["progress"]
        offset_ms = int(segment.get("offsetMs") or 0)
        level_offset = int(segment.get("levelOffset") or 0)
        for key in ("points", "levelChanges", "unresolvedLevelChanges", "lessReliableSegments", "interruptionIntervals", "interruptionHolds"):
            result[key].extend(
                shift_row(row, offset_ms, level_offset) for row in progress.get(key, [])
            )
    result["points"].sort(key=lambda row: int(row["mediaTimeMs"]))
    result["levelChanges"].sort(key=lambda row: int(row["mediaTimeMs"]))
    result["unresolvedLevelChanges"].sort(key=lambda row: int(row["mediaTimeMs"]))
    result["lessReliableSegments"].sort(key=lambda row: int(row["startMs"]))
    result["interruptionIntervals"].sort(key=lambda row: int(row["startMs"]))
    result["interruptionHolds"].sort(key=lambda row: int(row["startMs"]))
    return result
