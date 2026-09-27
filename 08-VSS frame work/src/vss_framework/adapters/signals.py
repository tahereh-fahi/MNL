"""Adapt sealed automated HUD and gem observations into framework signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..io import iter_jsonl
from ..models import (
    EvidenceGrade,
    SignalObservation,
    TemporalPrecision,
    Visibility,
)


SUPPORTED_OBSERVABLES = {"kill_counter", "game_clock", "gem_pickup"}


def _int_or_none(value: object) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    return int(float(str(value)))


def _signal_quality(row: dict[str, Any]) -> tuple[bool, Visibility, EvidenceGrade]:
    code = str(row["observable_code"])
    attributes = row.get("attributes_json") or {}
    evidence = row.get("evidence_json") or {}
    modalities = set(evidence.get("modalities") or [])
    if attributes.get("excluded_from_gameplay"):
        return False, Visibility.UNKNOWN, EvidenceGrade.UNRESOLVED
    if code == "kill_counter":
        observed = bool(attributes.get("source_value_observed"))
        return (
            observed,
            Visibility.VISIBLE if observed else Visibility.UNKNOWN,
            EvidenceGrade.A if observed else EvidenceGrade.C,
        )
    if code == "game_clock":
        observed = bool(attributes.get("source_ocr_observed", True))
        return (
            observed,
            Visibility.VISIBLE if observed else Visibility.PARTIAL,
            EvidenceGrade.A if observed else EvidenceGrade.B,
        )
    needs_review = bool(attributes.get("needs_review"))
    visual = "visual" in modalities
    return (
        True,
        Visibility.VISIBLE if visual else Visibility.PARTIAL,
        EvidenceGrade.C
        if needs_review
        else EvidenceGrade.A
        if visual and "experience_bar_change" in modalities
        else EvidenceGrade.B,
    )


def adapt_automated_signals(
    source_path: Path,
    source_label: str,
    *,
    start_ms: int,
    end_ms: int,
) -> Iterable[SignalObservation]:
    """Yield only automated signals overlapping ``[start_ms, end_ms)``."""

    for row in iter_jsonl(source_path):
        code = str(row.get("observable_code") or "")
        if code not in SUPPORTED_OBSERVABLES:
            continue
        if str(row.get("source_type") or "") != "automated":
            continue
        lower = int(row["media_start_ms"])
        upper = int(row["media_end_ms"])
        if upper <= start_ms or lower >= end_ms:
            continue
        observed, visibility, grade = _signal_quality(row)
        attributes = dict(row.get("attributes_json") or {})
        unit = attributes.get("unit")
        if code == "gem_pickup":
            unit = "estimated_gems"
        numeric_value = (
            row.get("quantity") if code == "gem_pickup" else row.get("numeric_value")
        )
        yield SignalObservation(
            observation_id=f"vss_signal_{row['observation_id']}",
            video_asset_id=str(row["video_asset_id"]),
            session_id=str(row["session_id"]),
            detector_name="sealed_automated_signal_adapter",
            detector_version="0.1.0",
            observable_code=code,
            time_lower_ms=lower,
            time_upper_ms=upper,
            temporal_precision=TemporalPrecision(str(row["temporal_precision"])),
            visibility=visibility,
            evidence_grade=grade,
            observed=observed,
            source_artifact=source_label,
            source_record_key=str(row.get("source_record_key") or row["observation_id"]),
            numeric_value=numeric_value,
            text_value=row.get("text_value"),
            unit=str(unit) if unit is not None else None,
            frame_number=_int_or_none(row.get("frame_number")),
            attributes={
                "upstream_observation_id": row["observation_id"],
                "source_type": "automated",
                "source_value_observed": attributes.get("source_value_observed"),
                "source_state_source": attributes.get("source_state_source"),
                "excluded_from_gameplay": attributes.get("excluded_from_gameplay", False),
                "exclusion_reason": attributes.get("exclusion_reason"),
                "gameplay_segment_id": attributes.get("gameplay_segment_id"),
                "needs_review": bool(attributes.get("needs_review")),
                "gem_type": attributes.get("gem_type"),
                "closest_color_candidate": attributes.get("closest_color_candidate"),
                "closest_color_candidate_source": attributes.get("closest_color_candidate_source"),
                "closest_color_confidence": attributes.get("closest_color_confidence"),
                "closest_color_quantity": attributes.get("closest_color_quantity"),
                "closest_color_assigned": attributes.get("closest_color_assigned"),
                "closest_color_assignment_threshold": attributes.get("closest_color_assignment_threshold"),
                "quantity_semantics": (
                    "xp_linked_estimate_not_exact_physical_count"
                    if code == "gem_pickup"
                    else None
                ),
            },
        )
