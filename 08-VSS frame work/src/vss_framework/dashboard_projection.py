"""Source-independent projections used by the research dashboard."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


GEM_COLORS = frozenset({"blue", "green", "red"})
UNRESOLVED_CLOSEST_COLOR_MIN_CONFIDENCE = 0.25


def _boolean(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _integer(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(str(value)))


def _unresolved_closest_color(row: Mapping[str, Any], quantity: int, confidence: float) -> dict[str, Any]:
    """Expose a tentative nearest colour without reclassifying an event."""

    candidate = next(
        (
            str(row.get(field) or "").strip().casefold()
            for field in (
                "magnet_entry_likely_color",
                "percentage_assist_color",
                "percentage_color_candidate",
            )
            if str(row.get(field) or "").strip().casefold() in GEM_COLORS
        ),
        None,
    )
    assigned = candidate is not None and confidence >= UNRESOLVED_CLOSEST_COLOR_MIN_CONFIDENCE
    return {
        "closestColorCandidate": candidate,
        "closestColorConfidence": confidence if candidate is not None else None,
        "closestColorQuantity": quantity if assigned else 0,
        "closestColorAssigned": assigned,
        "closestColorAssignmentThreshold": UNRESOLVED_CLOSEST_COLOR_MIN_CONFIDENCE,
    }


def project_inventory_events(
    rows: Sequence[Mapping[str, Any]], session_start_ms: int = 0,
) -> list[dict[str, Any]]:
    """Project persistent inventory changes without confusing item and game levels."""

    events: list[dict[str, Any]] = []
    for row in rows:
        if row.get("event_type") == "initial_state":
            continue
        media_ms = session_start_ms + round(float(row["video_second"]) * 1000)
        events.append(
            {
                "eventId": row["event_id"],
                "mediaTimeMs": media_ms,
                "frameNumber": int(float(row["frame_number"])),
                "observedCharacterLevel": _integer(row.get("character_level")),
                # Backward-compatible dashboard field.  Its name is historical;
                # item levels remain in itemLevelBefore/itemLevelAfter.
                "resultingLevel": _integer(row.get("character_level")),
                "action": "new" if row.get("event_type") == "new" else "upgrade",
                "itemType": row.get("item_type"),
                "slot": row.get("slot") or "unknown",
                "itemName": row.get("item_after"),
                "itemBefore": row.get("item_before") or None,
                "itemAfter": row.get("item_after") or None,
                "itemLevelBefore": _integer(row.get("level_before")),
                "itemLevelAfter": _integer(row.get("level_after")) or 1,
                "confidenceLabel": row.get("confidence"),
                "needsReview": _boolean(row.get("needs_review")),
            }
        )
    return events


def project_gem_events(
    rows: Sequence[Mapping[str, Any]], session_start_ms: int, segment: str,
    source_id_start: int = 1, *, event_key_prefix: str = "",
) -> list[dict[str, Any]]:
    """Project automated XP-linked gem estimates while preserving uncertainty."""

    events: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=source_id_start):
        quantities = {
            "blue": int(float(row.get("collected_blue_gems") or 0)),
            "green": int(float(row.get("collected_green_gems") or 0)),
            "red": int(float(row.get("collected_red_gems") or 0)),
            "unresolved": int(float(row.get("unresolved_collected_gems") or 0)),
        }
        total = sum(quantities.values())
        if total <= 0:
            continue
        confidence = float(row.get("confidence") or 0)
        unresolved_assignment = _unresolved_closest_color(
            row, quantities["unresolved"], confidence,
        ) if quantities["unresolved"] else None
        events.append(
            {
                "sourceEventId": index,
                "eventKey": f"{event_key_prefix}{segment}_{row['event_key']}",
                "videoTimeMs": session_start_ms + round(float(row["video_time_b"]) * 1000),
                "frameNumber": int(float(row["frame_b"])),
                "quantities": quantities,
                "totalQuantity": total,
                "confidenceByType": {key: confidence for key, value in quantities.items() if value > 0},
                "needsReview": _boolean(row.get("needs_review")),
                "reviewReason": row.get("review_reason") or None,
                "previewAvailable": False,
                "unresolvedClosestColor": unresolved_assignment,
            }
        )
    return events


def build_five_second_counter_windows(
    telemetry: Sequence[Mapping[str, Any]], duration_ms: int, *, bin_ms: int = 5_000,
    time_key: str = "mediaStartMs", value_key: str = "killCounter",
) -> list[dict[str, Any]]:
    """Build half-open counter-delta bins without manufacturing missing values."""

    ordered = sorted((dict(row) for row in telemetry), key=lambda row: int(row[time_key]))
    windows: list[dict[str, Any]] = []
    for start in range(0, duration_ms, bin_ms):
        end = min(duration_ms, start + bin_ms)
        observed = [row for row in ordered if start <= int(row[time_key]) <= end]
        prior = [row for row in ordered if int(row[time_key]) <= start]
        through_end = [row for row in ordered if int(row[time_key]) <= end]
        if not prior or not through_end:
            start_counter = end_counter = 0
            available = False
        else:
            start_counter = int(prior[-1][value_key])
            end_counter = int(through_end[-1][value_key])
            available = bool(observed)
        windows.append(
            {
                "startMs": start,
                "endMs": end,
                "counterStart": start_counter,
                "counterEnd": end_counter,
                "delta": max(0, end_counter - start_counter) if available else None,
                "observed": available,
                "intervalSemantics": "[startMs,endMs)",
            }
        )
    return windows


def centered_window_mean(
    points: Sequence[Mapping[str, Any]], *, time_key: str, value_key: str,
    half_window_ms: int = 15_000,
) -> list[dict[str, float]]:
    """Return a centered 30-second mean when ``half_window_ms`` is 15 seconds."""

    ordered = sorted(points, key=lambda row: float(row[time_key]))
    result: list[dict[str, float]] = []
    for point in ordered:
        center = float(point[time_key])
        window = [row for row in ordered if abs(float(row[time_key]) - center) <= half_window_ms]
        result.append({"centerMs": center, "value": sum(float(row[value_key]) for row in window) / len(window)})
    return result


def build_gold_counter_trajectory(
    points: Sequence[Mapping[str, Any]], *, bin_ms: int = 5_000,
    maximum_mean_gap_ms: int = 10_000, half_window_ms: int = 15_000,
) -> dict[str, list[dict[str, Any]]]:
    """Build cumulative Gold bins, Gold deltas, and its centered mean.

    The Gold bars are changes in the cumulative counter, not a moving mean of
    the changes.  The overlaid trend is the centered 30-second mean of the
    cumulative counter itself.  Missing bins and counter regressions remain
    missing rather than being converted to zero.
    """

    latest_by_bin: dict[int, dict[str, Any]] = {}
    for source in sorted(points, key=lambda row: int(row["mediaTimeMs"])):
        observed_at = int(source["mediaTimeMs"])
        start_ms = (observed_at // bin_ms) * bin_ms
        current = latest_by_bin.get(start_ms)
        if current is None or observed_at >= int(current["observedAtMs"]):
            latest_by_bin[start_ms] = {
                "startMs": start_ms,
                "endMs": start_ms + bin_ms,
                "observedAtMs": observed_at,
                "value": source["value"],
            }

    counter_bins = sorted(latest_by_bin.values(), key=lambda row: int(row["startMs"]))
    delta_bins: list[dict[str, Any]] = []
    for previous, current in zip(counter_bins, counter_bins[1:]):
        if int(current["startMs"]) != int(previous["startMs"]) + bin_ms:
            continue
        if float(current["value"]) < float(previous["value"]):
            continue
        delta_bins.append(
            {
                **current,
                "counterStart": previous["value"],
                "goldDelta": float(current["value"]) - float(previous["value"]),
            }
        )

    segments: list[list[dict[str, Any]]] = []
    for counter_bin in counter_bins:
        previous = segments[-1][-1] if segments else None
        if previous is None or int(counter_bin["startMs"]) - int(previous["startMs"]) > maximum_mean_gap_ms:
            segments.append([counter_bin])
        else:
            segments[-1].append(counter_bin)

    means: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        for bin_index, counter_bin in enumerate(segment):
            center_ms = int(counter_bin["startMs"]) + bin_ms // 2
            neighbors = [
                candidate for candidate in segment
                if abs((int(candidate["startMs"]) + bin_ms // 2) - center_ms) <= half_window_ms
            ]
            means.append(
                {
                    "centerMs": center_ms,
                    "value": sum(float(candidate["value"]) for candidate in neighbors) / len(neighbors),
                    "breakBefore": bin_index == 0,
                    "segmentIndex": segment_index,
                }
            )

    return {
        "counterBins": counter_bins,
        "fiveSecondDeltaBins": delta_bins,
        "centered30SecondCounterMean": means,
    }


def build_reward_trajectory(
    telemetry: Sequence[Mapping[str, Any]],
    gem_events: Sequence[Mapping[str, Any]],
    duration_ms: int,
    *,
    bin_ms: int = 5_000,
    counter_semantics: str = "observed_span",
) -> list[dict[str, Any]]:
    """Build the dashboard's native five-second Kill and gem trajectory.

    ``observed_span`` preserves the historical per-bin first/last observation
    behavior. ``boundary_snapshot`` samples the latest counter at each bin
    boundary, which is appropriate for already aligned cumulative telemetry.
    """

    if counter_semantics not in {"observed_span", "boundary_snapshot"}:
        raise ValueError("counter_semantics must be observed_span or boundary_snapshot")
    ordered_telemetry = sorted(telemetry, key=lambda row: int(row["mediaStartMs"]))

    def counter_at_or_before(time_ms: int) -> int | None:
        prior = [row for row in ordered_telemetry if int(row["mediaStartMs"]) <= time_ms]
        return int(prior[-1]["killCounter"]) if prior else None

    trajectory = []
    for start in range(0, ((duration_ms + bin_ms - 1) // bin_ms) * bin_ms, bin_ms):
        end = min(start + bin_ms, duration_ms)
        observed = [row for row in ordered_telemetry if start <= int(row["mediaStartMs"]) < end]
        events = [row for row in gem_events if start <= int(row["videoTimeMs"]) < end]
        quantities = {
            key: sum(int((event.get("quantities") or {}).get(key) or 0) for event in events)
            for key in ("blue", "green", "red", "unresolved")
        }
        if counter_semantics == "boundary_snapshot":
            start_snapshot = counter_at_or_before(start)
            end_snapshot = counter_at_or_before(end)
            start_counter = start_snapshot if start_snapshot is not None else 0
            end_counter = end_snapshot if end_snapshot is not None else start_counter
        else:
            start_counter = int(observed[0]["killCounter"]) if observed else 0
            end_counter = int(observed[-1]["killCounter"]) if observed else start_counter
        trajectory.append(
            {
                "startMs": start,
                "endMs": end,
                "killCounterStart": start_counter,
                "killCounterEnd": end_counter,
                "kills": max(0, end_counter - start_counter),
                "gameClockStartMs": observed[0].get("gameClockMs") if observed else None,
                "gameClockEndMs": observed[-1].get("gameClockMs") if observed else None,
                "gemEventCount": len(events),
                "gemObservationCount": len(events),
                "gemEventsNeedingReview": sum(bool(event.get("needsReview")) for event in events),
                "gems": {**quantities, "total": sum(quantities.values())},
                "sourceObserved": bool(observed),
            }
        )
    return trajectory
