"""Adapter worker for cached kill-counter and HUD game-clock detections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .clock_qc import ClockSample, evaluate_clock_series, format_clock
from .contracts import observation
from .io_utils import read_csv


REQUIRED_COLUMNS = {
    "Video Second",
    "Time Stamp",
    "Kill Counter Quantity",
    "review_flag",
}


def _parse_bool(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"", "0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _optional_float(value: str | None) -> float | None:
    normalized = (value or "").strip().lower()
    if normalized in {"", "nan", "na", "null", "none"}:
        return None
    parsed = float(normalized)
    if not 0 <= parsed <= 1:
        raise ValueError(f"OCR confidence must be in [0,1]: {value!r}")
    return parsed


def _relative_evidence_path(source_csv_relative: str, value: str | None) -> str | None:
    raw = (value or "").strip().replace("\\", "/")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("HUD evidence path must be relative to the worker artifact.")
    return (Path(source_csv_relative).parent / candidate).as_posix()


def normalize_hud(
    *,
    source_csv: Path,
    source_csv_relative: str,
    processing_run_id: str,
    session_id: str,
    video_asset_id: str,
    duration_ms: int,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv(source_csv)
    if not rows:
        raise ValueError("HUD source CSV is empty.")
    missing_columns = REQUIRED_COLUMNS.difference(rows[0])
    if missing_columns:
        raise ValueError(f"HUD source is missing columns: {sorted(missing_columns)}")

    seconds = [int(row["Video Second"]) for row in rows]
    if seconds != list(range(seconds[0], seconds[0] + len(seconds))):
        raise ValueError("HUD source seconds are not contiguous.")
    enriched_source = {
        "timer_observed",
        "timer_confidence",
        "kill_observed",
        "kill_confidence",
        "kill_state_value",
        "kill_state_source",
    }.issubset(rows[0])
    kills = [
        int(row["kill_state_value"] if enriched_source else row["Kill Counter Quantity"])
        if (row["kill_state_value"] if enriched_source else row["Kill Counter Quantity"]).strip()
        else None
        for row in rows
    ]
    if any(value is not None and value < 0 for value in kills):
        raise ValueError("Kill counter contains a negative value.")
    if any(current is not None and previous is not None and current < previous
           for previous, current in zip(kills, kills[1:])):
        raise ValueError("Kill counter decreases in the source artifact.")

    raw_clocks = [row.get("Time Stamp") or None for row in rows]
    pause_schema = "excluded_from_gameplay" in rows[0]
    if pause_schema:
        raw_clocks = [None if _parse_bool(row.get("excluded_from_gameplay")) else value
                      for row, value in zip(rows, raw_clocks)]
    clock_config = config.get("clock_qc", {})
    clocks = evaluate_clock_series(
        raw_clocks,
        video_duration_ms=duration_ms,
        max_game_clock_rate=float(clock_config.get("max_game_clock_rate", 2.1)),
        jump_review_threshold_seconds=int(
            clock_config.get("jump_review_threshold_seconds", 3)
        ),
    )
    if pause_schema:
        clocks = [ClockSample(None, None, None, "excluded_level_up_pause",
                              ((row.get("exclusion_reason") or "level_up_pause"),))
                  if _parse_bool(row.get("excluded_from_gameplay")) else clock
                  for row, clock in zip(rows, clocks)]

    observations: list[dict[str, Any]] = []
    sample_offsets = list(config.get("source_sampling_offsets_seconds", []))
    for source_row, (row, second, kill_value, clock) in enumerate(
        zip(rows, seconds, kills, clocks), start=2
    ):
        media_start_ms = second * 1000
        media_end_ms = min((second + 1) * 1000, duration_ms)
        upstream_review = _parse_bool(row.get("review_flag"))
        excluded = _parse_bool(row.get("excluded_from_gameplay"))
        exclusion_reason = (row.get("exclusion_reason") or "").strip() or None
        segment_id = int(row["gameplay_segment_id"]) if row.get("gameplay_segment_id") else None
        pause_attributes = ({"excluded_from_gameplay": excluded, "exclusion_reason": exclusion_reason,
                             "gameplay_segment_id": segment_id} if pause_schema else {})
        kill_observed = (
            _parse_bool(row.get("kill_observed")) and not excluded if enriched_source else False
        )
        kill_state_source = (
            (row.get("kill_state_source") or "").strip()
            if enriched_source
            else "legacy_unknown"
        )
        kill_confidence = (
            _optional_float(row.get("kill_confidence")) if enriched_source else None
        )
        timer_confidence = (
            _optional_float(row.get("timer_confidence")) if enriched_source else None
        )
        kill_needs_review = not excluded and (
            kill_state_source != "observed"
            if enriched_source
            else upstream_review
            or clock.status in {"missing", "rejected", "review"}
        )
        shared_evidence = {
            "modalities": ["ocr", "hud"],
            "source_csv": source_csv_relative,
            "source_row_number": source_row,
        }
        sample_frame_path = _relative_evidence_path(
            source_csv_relative, row.get("sample_frame_path")
        )
        if sample_frame_path is not None:
            shared_evidence["sample_frame_path"] = sample_frame_path
        observations.append(
            observation(
                processing_run_id_value=processing_run_id,
                source_record_key=f"hud_second:{second}:kill_counter",
                session_id=session_id,
                video_asset_id=video_asset_id,
                record_kind="sampled_value",
                temporal_precision="window",
                observable_code="kill_counter",
                media_start_ms=media_start_ms,
                media_end_ms=media_end_ms,
                session_start_ms=media_start_ms,
                session_end_ms=media_end_ms,
                game_time_ms=(
                    clock.accepted_seconds * 1000
                    if clock.accepted_seconds is not None
                    else None
                ),
                numeric_value=None if excluded else kill_value,
                confidence=(
                    kill_confidence if enriched_source and kill_observed else None
                ),
                evidence_json=dict(shared_evidence),
                attributes_json={
                    "unit": "kills",
                    "is_cumulative": True,
                    "hud_region": "top_right",
                    "sample_window_semantics": "[start,end)",
                    "source_sampling_offsets_seconds": sample_offsets,
                    "source_output_rate_hz": 1,
                    "source_ocr_confidence_available": enriched_source,
                    "source_may_forward_fill_value": True,
                    "source_value_observed": kill_observed if enriched_source else None,
                    "source_state_source": kill_state_source,
                    **pause_attributes,
                    "source_kill_status": (
                        row.get("kill_status") if enriched_source else None
                    ),
                    "source_ocr_time_stamp": clock.raw_text,
                    "source_parsed_clock_seconds_before_qc": clock.parsed_seconds,
                    "upstream_review_flag": upstream_review,
                    "needs_review": kill_needs_review,
                    "game_clock_qc_status": clock.status,
                    "game_clock_qc_reasons": list(clock.reasons),
                },
            )
        )
        if not clock.usable_for_alignment and not excluded:
            continue
        clock_milliseconds = clock.accepted_seconds * 1000 if clock.accepted_seconds is not None and not excluded else None
        observations.append(
            observation(
                processing_run_id_value=processing_run_id,
                source_record_key=f"hud_second:{second}:game_clock",
                session_id=session_id,
                video_asset_id=video_asset_id,
                record_kind="sampled_value",
                temporal_precision="window",
                observable_code="game_clock",
                media_start_ms=media_start_ms,
                media_end_ms=media_end_ms,
                session_start_ms=media_start_ms,
                session_end_ms=media_end_ms,
                game_time_ms=clock_milliseconds,
                numeric_value=clock_milliseconds,
                text_value=format_clock(clock.accepted_seconds) if clock.accepted_seconds is not None and not excluded else None,
                confidence=timer_confidence,
                evidence_json=dict(shared_evidence),
                attributes_json={
                    "unit": "ms",
                    **pause_attributes,
                    "hud_region": "top_middle",
                    "raw_ocr_text": (
                        row.get("timer_raw_text") if enriched_source else clock.raw_text
                    ),
                    "parsed_seconds_before_qc": clock.parsed_seconds,
                    "qc_status": clock.status,
                    "qc_reasons": list(clock.reasons),
                    "usable_for_alignment": clock.usable_for_alignment,
                    "needs_review": clock.status in {"rejected", "review"},
                    "source_ocr_confidence_available": enriched_source,
                    "source_ocr_observed": (
                        _parse_bool(row.get("timer_observed"))
                        if enriched_source
                        else None
                    ),
                    "source_timer_status": (
                        row.get("timer_status") if enriched_source else None
                    ),
                    "source_timer_preprocessing_variant": (
                        (row.get("timer_preprocessing_variant") or "").strip()
                        or None
                        if enriched_source
                        else None
                    ),
                    "source_timer_normalization_policy": (
                        (row.get("timer_normalization_policy") or "").strip()
                        or None
                        if enriched_source
                        else None
                    ),
                    "source_timer_fallback_attempted": (
                        _parse_bool(row.get("timer_fallback_attempted"))
                        if enriched_source
                        else None
                    ),
                    "source_timer_primary_candidate_count": (
                        int(row["timer_primary_candidate_count"])
                        if enriched_source
                        and (row.get("timer_primary_candidate_count") or "").strip()
                        else None
                    ),
                    "source_timer_fallback_candidate_count": (
                        int(row["timer_fallback_candidate_count"])
                        if enriched_source
                        and (row.get("timer_fallback_candidate_count") or "").strip()
                        else None
                    ),
                    "source_may_forward_fill_value": not enriched_source,
                    "imputed_by_mvp": False,
                    "sample_window_semantics": "[start,end)",
                },
            )
        )

    status_counts = Counter(clock.status for clock in clocks)
    issue_rows = [
        {
            "video_second": second,
            "raw_text": clock.raw_text,
            "parsed_seconds": clock.parsed_seconds,
            "accepted_seconds": clock.accepted_seconds,
            "status": clock.status,
            "reasons": list(clock.reasons),
        }
        for second, clock in zip(seconds, clocks)
        if clock.status != "accepted"
    ]
    summary = {
        "source_rows": len(rows),
        "observation_rows": len(observations),
        "counts_by_observable": {
            "kill_counter": len(rows),
            "game_clock": sum(row["observable_code"] == "game_clock" for row in observations),
        },
        "final_kill_counter": kills[-1],
        "kill_counter_decreases": 0,
        **({"level_up_pause_excluded_seconds": sum(_parse_bool(row.get("excluded_from_gameplay")) for row in rows)} if pause_schema else {}),
        "clock_status_counts": dict(sorted(status_counts.items())),
        "clock_usable_count": sum(clock.usable_for_alignment for clock in clocks),
        "clock_usable_fraction": round(
            sum(clock.usable_for_alignment for clock in clocks)
            / max(1, sum(clock.status != "excluded_level_up_pause" for clock in clocks)), 6
        ),
        "clock_source_nonblank_count": sum(
            clock.raw_text is not None for clock in clocks
        ),
        "clock_imputed_by_mvp_count": 0,
        "source_confidence_available": enriched_source,
        "kill_state_source_counts": (
            dict(sorted(Counter(row["kill_state_source"] for row in rows).items()))
            if enriched_source
            else {"legacy_unknown": len(rows)}
        ),
        "kill_observed_count": (
            sum(_parse_bool(row.get("kill_observed")) for row in rows)
            if enriched_source
            else None
        ),
        "timer_observed_count": (
            sum(_parse_bool(row.get("timer_observed")) for row in rows)
            if enriched_source
            else None
        ),
        "timer_fallback_attempted_count": (
            sum(_parse_bool(row.get("timer_fallback_attempted")) for row in rows)
            if enriched_source
            else None
        ),
        "timer_fallback_recovered_count": (
            sum(
                _parse_bool(row.get("timer_fallback_attempted"))
                and _parse_bool(row.get("timer_observed"))
                for row in rows
            )
            if enriched_source
            else None
        ),
        "timer_unique_embedded_normalization_count": (
            sum(
                (row.get("timer_normalization_policy") or "").strip()
                == "unique_embedded_mm_ss"
                for row in rows
            )
            if enriched_source
            else None
        ),
    }
    return observations, summary, issue_rows
