"""Adapter worker for cached XP-confirmed collected-gem events."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import observation
from .io_utils import read_csv


CATEGORY_COLUMNS = {
    "blue": "collected_blue_gems",
    "green": "collected_green_gems",
    "red": "collected_red_gems",
    "unresolved": "unresolved_collected_gems",
}
GEM_COLORS = frozenset({"blue", "green", "red"})
# This gate is deliberately applied to the detector's published confidence, not
# to a colour hint alone.  A hint identifies the nearest candidate; it does not
# make an unresolved pickup a confirmed colour classification.
UNRESOLVED_CLOSEST_COLOR_MIN_CONFIDENCE = 0.25
REQUIRED_COLUMNS = {
    "event_id",
    "event_key",
    "frame_a",
    "frame_b",
    "video_time_a",
    "video_time_b",
    "collected_gems_total",
    "color_evidence",
    "confidence",
    "needs_review",
    "review_reason",
    *CATEGORY_COLUMNS.values(),
}


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip().lower() in {"", "nan", "na", "null", "none"}:
        return None
    return int(float(value))


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "nan", "na", "null", "none"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _boolean(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"", "0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _confidence_method(color_evidence: str) -> str:
    if "trajectory" in color_evidence:
        return "trajectory_heuristic"
    if "visual" in color_evidence:
        return "visual_disappearance_heuristic"
    if "percentage" in color_evidence:
        return "percentage_assist_calibration"
    if "xp" in color_evidence:
        return "xp_magnitude_fallback"
    return "ambiguous_heuristic"


def _unresolved_closest_color(
    row: Mapping[str, Any], *, quantity: int, confidence: float,
) -> dict[str, Any]:
    """Return an auditable optional colour assignment for an unresolved pickup.

    The source category is never changed: callers retain it as ``unresolved``.
    Only an explicit upstream colour candidate is eligible, and only when the
    detector confidence meets the publication threshold.
    """

    candidate_fields = (
        ("magnet_entry_likely_color", "magnet_entry_likely_color"),
        ("percentage_assist_color", "percentage_assist_color"),
        ("percentage_color_candidate", "percentage_color_candidate"),
    )
    candidate = None
    source = None
    for field, label in candidate_fields:
        value = str(row.get(field) or "").strip().casefold()
        if value in GEM_COLORS:
            candidate = value
            source = label
            break

    assigned = candidate is not None and confidence >= UNRESOLVED_CLOSEST_COLOR_MIN_CONFIDENCE
    return {
        "closest_color_candidate": candidate,
        "closest_color_candidate_source": source,
        "closest_color_confidence": confidence if candidate is not None else None,
        "closest_color_quantity": quantity if assigned else 0,
        "closest_color_assigned": assigned,
        "closest_color_assignment_threshold": UNRESOLVED_CLOSEST_COLOR_MIN_CONFIDENCE,
    }


def normalize_gems(
    *,
    source_csv: Path,
    source_csv_relative: str,
    processing_run_id: str,
    session_id: str,
    video_asset_id: str,
    duration_ms: int,
    fps: float,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv(source_csv)
    if not rows:
        raise ValueError("Gem source CSV is empty.")
    missing_columns = REQUIRED_COLUMNS.difference(rows[0])
    if missing_columns:
        raise ValueError(f"Gem source is missing columns: {sorted(missing_columns)}")

    event_ids = [int(row["event_id"]) for row in rows]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Gem event_id values are not unique.")
    if event_ids != sorted(event_ids):
        raise ValueError("Gem event_id values are not sorted.")
    frame_numbers = [int(row["frame_b"]) for row in rows]
    media_start_times_ms = [
        int(round(float(row["video_time_a"]) * 1000)) for row in rows
    ]
    media_times_ms = [int(round(float(row["video_time_b"]) * 1000)) for row in rows]
    if frame_numbers != sorted(frame_numbers) or media_times_ms != sorted(media_times_ms):
        raise ValueError("Gem events are not ordered by frame/media time.")
    if media_times_ms[-1] > duration_ms:
        raise ValueError("Gem event falls outside the video duration.")
    if fps <= 0:
        raise ValueError("Video FPS must be positive.")
    for row, media_start_ms, media_end_ms in zip(
        rows, media_start_times_ms, media_times_ms
    ):
        expected_start_ms = int(round(int(row["frame_a"]) / fps * 1000))
        expected_end_ms = int(round(int(row["frame_b"]) / fps * 1000))
        if abs(expected_start_ms - media_start_ms) > 2 or abs(
            expected_end_ms - media_end_ms
        ) > 2:
            raise ValueError(
                f"Gem event {row['event_id']} frame and media timestamps disagree."
            )

    observations: list[dict[str, Any]] = []
    quantity_totals = {category: 0 for category in CATEGORY_COLUMNS}
    review_event_count = 0
    balance_errors: list[dict[str, Any]] = []
    evidence_missing_count = 0
    for source_row, (row, event_id, media_start_ms, media_ms, frame_number) in enumerate(
        zip(rows, event_ids, media_start_times_ms, media_times_ms, frame_numbers),
        start=2,
    ):
        counts = {
            category: int(row[column])
            for category, column in CATEGORY_COLUMNS.items()
        }
        for category, quantity in counts.items():
            if quantity < 0:
                raise ValueError(
                    f"Gem event {event_id} has negative {category} quantity."
                )
            quantity_totals[category] += quantity
        total = int(row["collected_gems_total"])
        if sum(counts.values()) != total:
            balance_errors.append(
                {
                    "event_id": event_id,
                    "category_sum": sum(counts.values()),
                    "declared_total": total,
                }
            )
            continue
        confidence = _optional_float(row.get("confidence"))
        if confidence is None or not 0 <= confidence <= 1:
            raise ValueError(f"Gem event {event_id} has invalid confidence.")
        needs_review = _boolean(row.get("needs_review"))
        review_event_count += int(needs_review)
        color_evidence = row.get("color_evidence", "")
        modalities = ["experience_bar_change"]
        if "visual" in color_evidence or "trajectory" in color_evidence:
            modalities.append("visual")
        if row.get("hud_level_source") == "hud_ocr":
            modalities.append("ocr")
        evidence: dict[str, Any] = {
            "modalities": modalities,
            "source_csv": source_csv_relative,
            "source_row_number": source_row,
            "source_event_id": event_id,
            "frame_a": int(row["frame_a"]),
            "frame_b": frame_number,
        }
        evidence_paths = []
        for column in ("pair_image", "trajectory_image", "count_trajectory_image"):
            value = (row.get(column) or "").strip()
            if value:
                evidence[column] = value
                evidence_paths.append(value)
        if not evidence_paths:
            evidence_missing_count += 1

        attributes_common = {
            "gem_semantics": "collected_pickup_confirmed_by_xp_change",
            "xp_event_id": event_id,
            "event_key": row["event_key"],
            "event_time_policy": "xp_transition_end_for_binning",
            "xp_event_total_quantity": total,
            "all_category_quantities": counts,
            "hud_level": _optional_int(row.get("hud_level")),
            "hud_level_source": row.get("hud_level_source") or None,
            "xp_bar_increase_percent": _optional_float(
                row.get("xp_bar_increase_percent")
            ),
            "estimated_base_xp_gain": _optional_float(
                row.get("estimated_base_xp_gain")
            ),
            "color_evidence": color_evidence,
            "confidence_method": _confidence_method(color_evidence),
            "confidence_semantics": "detector_heuristic_not_calibrated_probability",
            "needs_review": needs_review,
            "review_reason": row.get("review_reason") or None,
            "percentage_assist_applied": _boolean(
                row.get("percentage_assist_applied")
            ),
            "xp_bar_saturated": _boolean(row.get("xp_bar_saturated")),
            "event_detection_reason": row.get("event_detection_reason") or None,
            "post_level_up_review": _boolean(row.get("post_level_up_review")),
            "level_up_boundary_candidate": _boolean(
                row.get("level_up_boundary_candidate")
            ),
            "xp_increase_censored": _boolean(row.get("xp_increase_censored")),
            "xp_full_inferred_from_level_change": _boolean(
                row.get("xp_full_inferred_from_level_change")
            ),
        }
        for category, quantity in counts.items():
            if quantity <= 0:
                continue
            attributes = dict(attributes_common)
            if category == "unresolved":
                attributes.update(
                    _unresolved_closest_color(
                        row, quantity=quantity, confidence=confidence,
                    )
                )
            attributes.update(
                {
                    "gem_type": category,
                    "category_quantity": quantity,
                    "count_batched": quantity > 1,
                }
            )
            observations.append(
                observation(
                    processing_run_id_value=processing_run_id,
                    source_record_key=f"xp_event:{row['event_key']}:{category}",
                    session_id=session_id,
                    video_asset_id=video_asset_id,
                    record_kind="point_event",
                    temporal_precision="window",
                    observable_code="gem_pickup",
                    media_start_ms=media_start_ms,
                    media_end_ms=media_ms,
                    session_start_ms=media_start_ms,
                    session_end_ms=media_ms,
                    frame_number=frame_number,
                    quantity=quantity,
                    confidence=confidence,
                    evidence_json=evidence,
                    attributes_json=attributes,
                )
            )

    if balance_errors:
        raise ValueError(
            f"{len(balance_errors)} gem events do not balance across categories."
        )
    total_quantity = sum(quantity_totals.values())
    issue_rows = [
        {
            "event_id": int(row["event_id"]),
            "event_key": row["event_key"],
            "video_time_ms": int(round(float(row["video_time_b"]) * 1000)),
            "needs_review": True,
            "review_reason": row.get("review_reason") or None,
        }
        for row in rows
        if _boolean(row.get("needs_review"))
    ]
    summary = {
        "source_event_rows": len(rows),
        "observation_rows": len(observations),
        "counts_by_observable": {"gem_pickup": len(observations)},
        "quantity_by_gem_type": quantity_totals,
        "total_gem_quantity": total_quantity,
        "needs_review_event_count": review_event_count,
        "needs_review_event_fraction": round(review_event_count / len(rows), 6),
        "unresolved_quantity_fraction": (
            round(quantity_totals["unresolved"] / total_quantity, 6)
            if total_quantity else None
        ),
        "events_without_preview_evidence": evidence_missing_count,
        "evidence_policy": config.get("evidence_policy", "none"),
        "human_labels_used_by_detector": False,
        "audio_used_by_detector": False,
    }
    if not total_quantity:
        # A recorded level-up boundary is not a measured pickup. An empty
        # pickup denominator is undefined, not a perfect zero-error result.
        summary["unresolved_quantity_fraction_denominator"] = 0
        summary["unresolved_quantity_fraction_status"] = "no_measured_pickups"
    return observations, summary, issue_rows
