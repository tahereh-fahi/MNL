"""Reusable validation and repair rules for cumulative HUD counters."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def corroborate_repeated_counter_points(
    points: Sequence[Mapping[str, Any]], *, time_key: str = "mediaTimeMs",
    value_key: str = "value", maximum_gap_ms: int = 2_000,
) -> list[dict[str, Any]]:
    """Keep direct OCR values corroborated by an adjacent repeated sample."""

    ordered = sorted((dict(point) for point in points), key=lambda row: int(row[time_key]))
    result: list[dict[str, Any]] = []
    for index, point in enumerate(ordered):
        neighbors = ordered[max(0, index - 1):index] + ordered[index + 1:index + 2]
        if any(
            neighbor[value_key] == point[value_key]
            and abs(int(neighbor[time_key]) - int(point[time_key])) <= maximum_gap_ms
            for neighbor in neighbors
        ):
            result.append(point)
    return result


def restore_truncated_counter_from_raw(
    stored_value: int, raw_text: str | None, *, minimum_missing_magnitude: int = 1_000,
) -> tuple[int, str | None]:
    """Restore leading digits when the retained value is only an OCR suffix."""

    digits = "".join(character for character in str(raw_text or "") if character.isdigit())
    if not digits:
        return stored_value, None
    raw_value = int(digits)
    if raw_value - stored_value >= minimum_missing_magnitude and str(raw_value).endswith(str(stored_value)):
        return raw_value, "full_raw_ocr_restored"
    return stored_value, None


def repair_persistent_leading_place_shift(
    values: Sequence[int], *, minimum_jump: int = 1_000, confirmation_samples: int = 3,
) -> tuple[list[int], list[str | None]]:
    """Remove a persistent leading-place OCR substitution after a discontinuity.

    A correction is applied only when several following samples support the
    same power-of-ten offset and the corrected suffix remains monotonic.  This
    replaces per-video timestamp/offset constants while retaining an audit
    reason for every corrected sample.
    """

    repaired = list(values)
    reasons: list[str | None] = [None] * len(values)
    if len(values) < confirmation_samples + 1:
        return repaired, reasons
    for index in range(1, len(values)):
        jump = repaired[index] - repaired[index - 1]
        if jump < minimum_jump:
            continue
        magnitude = 10 ** max(1, len(str(abs(jump))) - 1)
        candidates = [magnitude * multiplier for multiplier in range(1, 10)]
        window = values[index:index + confirmation_samples]
        valid_offsets = []
        for offset in candidates:
            corrected = [value - offset for value in window]
            if (
                all(value >= 0 for value in corrected)
                and corrected[0] >= repaired[index - 1]
                and all(right >= left for left, right in zip(corrected, corrected[1:]))
                and corrected[0] - repaired[index - 1] < minimum_jump
            ):
                valid_offsets.append(offset)
        if not valid_offsets:
            continue
        offset = min(valid_offsets, key=lambda value: abs((values[index] - value) - repaired[index - 1]))
        for repair_index in range(index, len(values)):
            candidate = values[repair_index] - offset
            if candidate < 0 or (repair_index > index and candidate < repaired[repair_index - 1]):
                break
            repaired[repair_index] = candidate
            reasons[repair_index] = "persistent_leading_place_ocr_shift_corrected"
        break
    return repaired, reasons


def repair_isolated_counter_discontinuities(
    values: Sequence[int],
) -> tuple[list[int], list[str | None]]:
    """Carry over an isolated impossible spike/drop in a cumulative counter.

    A sample is changed only when the immediately following value returns to a
    value at or above the preceding accepted counter. This distinguishes a
    one-sample OCR discontinuity from a persistent change that needs a separate
    leading-place repair.
    """

    repaired = list(values)
    reasons: list[str | None] = [None] * len(values)
    for index in range(1, len(values) - 1):
        previous = repaired[index - 1]
        current = repaired[index]
        following = values[index + 1]
        isolated_spike = current > previous and following < current and following >= previous
        isolated_drop = current < previous and following >= previous
        if isolated_spike or isolated_drop:
            repaired[index] = previous
            reasons[index] = "isolated_cumulative_counter_ocr_discontinuity_rejected"
    return repaired, reasons


def repair_cumulative_counter_values(
    values: Sequence[int],
) -> tuple[list[int], list[str | None]]:
    """Apply the reusable cumulative-counter repairs in a stable order."""

    shifted, shift_reasons = repair_persistent_leading_place_shift(values)
    repaired, discontinuity_reasons = repair_isolated_counter_discontinuities(shifted)
    reasons = [late or early for early, late in zip(shift_reasons, discontinuity_reasons)]
    return repaired, reasons


def counter_quality_summary(values: Sequence[int]) -> dict[str, Any]:
    deltas = [right - left for left, right in zip(values, values[1:])]
    return {
        "count": len(values),
        "monotonic": all(delta >= 0 for delta in deltas),
        "negativeDeltaCount": sum(delta < 0 for delta in deltas),
        "largestIncrease": max(deltas, default=0),
        "repeatedValueCount": sum(count - 1 for count in Counter(values).values() if count > 1),
    }
