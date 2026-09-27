"""Validation of raw HUD game-clock OCR without silent imputation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClockSample:
    raw_text: str | None
    parsed_seconds: int | None
    accepted_seconds: int | None
    status: str
    reasons: tuple[str, ...]

    @property
    def usable_for_alignment(self) -> bool:
        return self.accepted_seconds is not None


def parse_clock(raw_text: str | None) -> tuple[int | None, str | None]:
    if raw_text is None or raw_text.strip() == "":
        return None, "missing_ocr"
    text = raw_text.strip()
    parts = text.split(":")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        minutes, seconds = (int(part) for part in parts)
        if seconds >= 60:
            return None, "seconds_out_of_range"
        return minutes * 60 + seconds, None
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        hours, minutes, seconds = (int(part) for part in parts)
        if minutes >= 60 or seconds >= 60:
            return None, "clock_component_out_of_range"
        return hours * 3600 + minutes * 60 + seconds, None
    if not re.fullmatch(r"[0-9:]+", text):
        return None, "invalid_characters"
    return None, "invalid_clock_format"


def format_clock(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def evaluate_clock_series(
    raw_values: Sequence[str | None],
    *,
    video_duration_ms: int,
    max_game_clock_rate: float = 2.1,
    jump_review_threshold_seconds: int = 3,
) -> list[ClockSample]:
    parsed: list[int | None] = []
    parse_reasons: list[str | None] = []
    for raw in raw_values:
        value, reason = parse_clock(raw)
        parsed.append(value)
        parse_reasons.append(reason)

    previous_valid: list[int | None] = []
    cursor: int | None = None
    for index, value in enumerate(parsed):
        previous_valid.append(cursor)
        if value is not None:
            cursor = index

    next_valid: list[int | None] = [None] * len(parsed)
    cursor = None
    for index in range(len(parsed) - 1, -1, -1):
        next_valid[index] = cursor
        if parsed[index] is not None:
            cursor = index

    rejected: dict[int, list[str]] = {}
    maximum_clock = int((video_duration_ms / 1000) * max_game_clock_rate) + 2
    for index, value in enumerate(parsed):
        if value is None:
            continue
        if value > maximum_clock:
            rejected.setdefault(index, []).append("outside_video_clock_bound")
        previous_index = previous_valid[index]
        next_index = next_valid[index]
        if previous_index is None or next_index is None:
            continue
        previous_value = parsed[previous_index]
        next_value = parsed[next_index]
        assert previous_value is not None and next_value is not None
        is_isolated_dip = value < previous_value and next_value >= previous_value
        is_isolated_spike = value > next_value and next_value >= previous_value
        if is_isolated_dip or is_isolated_spike:
            rejected.setdefault(index, []).append("isolated_order_violation")

    last_accepted: int | None = None
    for index, value in enumerate(parsed):
        if value is None or index in rejected:
            continue
        if last_accepted is not None and value < last_accepted:
            rejected.setdefault(index, []).append("monotonicity_violation")
            continue
        last_accepted = value

    accepted_values: list[int | None] = [
        None if value is None or index in rejected else value
        for index, value in enumerate(parsed)
    ]
    review_reasons: dict[int, list[str]] = {}
    previous_index = None
    previous_value = None
    for index, value in enumerate(accepted_values):
        if value is None:
            continue
        if previous_index is not None and previous_value is not None:
            media_gap = index - previous_index
            clock_jump = value - previous_value
            allowed_jump = jump_review_threshold_seconds * max(1, media_gap)
            if clock_jump > allowed_jump:
                review_reasons.setdefault(index, []).append(
                    "large_positive_clock_discontinuity"
                )
        previous_index = index
        previous_value = value

    results: list[ClockSample] = []
    for index, raw in enumerate(raw_values):
        if parsed[index] is None:
            reason = parse_reasons[index] or "unparseable_clock"
            status = "missing" if reason == "missing_ocr" else "rejected"
            results.append(
                ClockSample(raw, None, None, status, (reason,))
            )
            continue
        if index in rejected:
            results.append(
                ClockSample(
                    raw,
                    parsed[index],
                    None,
                    "rejected",
                    tuple(sorted(set(rejected[index]))),
                )
            )
            continue
        reasons = tuple(sorted(set(review_reasons.get(index, []))))
        results.append(
            ClockSample(
                raw,
                parsed[index],
                parsed[index],
                "review" if reasons else "accepted",
                reasons,
            )
        )
    return results
