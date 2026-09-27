#!/usr/bin/env python3
"""Active, evidence-preserving HUD detector worker.

The worker extracts one-second samples of the on-screen game clock and
cumulative kill counter.  It keeps raw OCR evidence separate from the
stateful kill estimate so a carried value is never mistaken for an observed
value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PREPARED_BY = "Tahereh Fahi"
WORKER_NAME = "mnl-hud-detector"
WORKER_VERSION = "1.7.1"
OUTPUT_SCHEMA_VERSION = "1.7.0"

KILL_OVERSHOOT_LOOKAHEAD_SECONDS = 7
KILL_OVERSHOOT_MIN_SUPPORTING_SECONDS = 2
KILL_OVERSHOOT_MIN_SAMPLES_PER_SECOND = 2
KILL_LARGE_JUMP_MIN_INCREASE = 30
KILL_LARGE_JUMP_MIN_FRACTION = 0.10
KILL_LARGE_JUMP_LOOKAHEAD_SECONDS = 20
KILL_LARGE_JUMP_MIN_FUTURE_CONFIDENCE = 0.5
KILL_GAP_ALLOWANCE_PER_ELIGIBLE_SECOND = 2
KILL_GAP_ALLOWANCE_CAP = 200
KILL_GAP_JUMP_CONFIRMATION_SECONDS = 10
KILL_GAP_JUMP_MIN_FUTURE_CONFIDENCE = 0.3

SCRIPT_PATH = Path(__file__).resolve()
COMPONENT_DIR = SCRIPT_PATH.parents[1]
from ..resources import asset_path
from ..gameplay_state import gameplay_pause_evidence
from .gems import gameplay_hud_score
TEMPLATE_DIR = asset_path("hud")
SKULL_TEMPLATE_NAMES = ("skull_icon.jpg", "skull_anchor.jpg")

EXIT_OK = 0
EXIT_INPUT_OR_DEPENDENCY = 3
EXIT_PROCESSING = 4
EXIT_QC_FAILED = 5

SAMPLING_OFFSETS_DEFAULT = (0.3, 0.5, 0.7)
TIMER_PRIMARY_PREPROCESSING_VARIANT = "fixed_threshold_200_inverted_upscaled_2x"
TIMER_FALLBACK_PREPROCESSING_VARIANTS = (
    "raw_color_upscaled_2x",
    "grayscale_upscaled_2x",
    "otsu_binary_inverted_upscaled_2x",
)
MANAGED_OUTPUT_NAMES = (
    "hud_observations.csv",
    "ocr_candidates.jsonl",
    "evidence_index.csv",
    "qc.json",
    "manifest.json",
    "_SUCCESS",
)

CSV_FIELDS = (
    "video_name",
    "video_file",
    "video_duration_minutes",
    "processed_minutes",
    "Video Second",
    "Time Stamp",
    "Kill Counter Quantity",
    "review_flag",
    "timer_observed",
    "timer_raw_text",
    "timer_normalized_text",
    "timer_confidence",
    "timer_status",
    "timer_candidate_count",
    "timer_primary_candidate_count",
    "timer_fallback_candidate_count",
    "timer_fallback_attempted",
    "timer_preprocessing_variant",
    "timer_normalization_policy",
    "kill_observed",
    "kill_raw_text",
    "kill_observed_value",
    "kill_confidence",
    "kill_state_value",
    "kill_state_source",
    "kill_status",
    "kill_candidate_count",
    "decoded_sample_count",
    "gameplay_sample_count",
    "excluded_sample_count",
    "excluded_from_gameplay",
    "exclusion_reason",
    "gameplay_segment_id",
    "sample_frame_path",
)

EVIDENCE_FIELDS = (
    "video_second",
    "relative_path",
    "reason",
    "sha256",
    "timer_status",
    "kill_state_source",
)

EASYOCR_LANGUAGES = ("en",)
EASYOCR_DETECTION_NETWORK = "craft"
EASYOCR_RECOGNITION_NETWORK = "standard"


class WorkerError(RuntimeError):
    """Expected worker failure with a stable process exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class TimerSelection:
    observed: bool
    raw_text: Optional[str]
    normalized_text: Optional[str]
    confidence: Optional[float]
    status: str
    valid_candidate_count: int
    preprocessing_variant: Optional[str] = None
    normalization_policy: Optional[str] = None


@dataclass(frozen=True)
class KillSelection:
    observed: bool
    raw_text: Optional[str]
    observed_value: Optional[int]
    confidence: Optional[float]
    state_value: Optional[int]
    state_source: str
    status: str
    valid_candidate_count: int


def exclude_paused_hud_window(
    samples: Sequence[Mapping[str, Any]],
    timer: TimerSelection,
    kill: KillSelection,
) -> Tuple[TimerSelection, KillSelection]:
    """An overlapping pause invalidates the whole one-second HUD window."""
    if not any(sample.get("excluded_from_gameplay") for sample in samples):
        return timer, kill
    return (
        TimerSelection(False, None, None, None, "excluded_level_up_pause", 0),
        KillSelection(False, None, None, None, None, "excluded_level_up_pause",
                      "excluded_level_up_pause", 0),
    )


def exclude_non_gameplay_hud_window() -> Tuple[TimerSelection, KillSelection]:
    """Represent a pregame/loading second without inventing HUD values."""
    status = "excluded_non_gameplay_hud"
    return (
        TimerSelection(False, None, None, None, status, 0),
        KillSelection(False, None, None, None, None, status, status, 0),
    )


def advance_kill_state(
    previous_state: Optional[int],
    selection: KillSelection,
    *,
    excluded_window: bool,
) -> Optional[int]:
    """Retain a cumulative baseline through pauses without publishing it there.

    A level-up window has no usable HUD observation, so its output stays
    missing. The game is paused, however, and the last reliable cumulative
    count must remain the baseline for rejecting partial OCR at resume.
    """
    return previous_state if excluded_window else selection.state_value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_json_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_target(path: Path) -> Tuple[Any, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=str(path.parent),
        prefix=".%s." % path.name,
        suffix=".tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    handle, temporary = _atomic_target(path)
    try:
        with handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    handle, temporary = _atomic_target(path)
    count = 0
    try:
        with handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def write_csv_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> int:
    handle, temporary = _atomic_target(path)
    count = 0
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        dir=str(path.parent), prefix=".%s." % path.name, suffix=".tmp"
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def template_paths() -> Tuple[Path, ...]:
    """Return component-relative template targets, independent of the CWD."""

    return tuple(TEMPLATE_DIR / name for name in SKULL_TEMPLATE_NAMES)


def normalize_timer_text(text: Any) -> Optional[str]:
    normalized, _ = normalize_timer_text_with_policy(
        text, allow_unique_embedded=False
    )
    return normalized


def normalize_timer_text_with_policy(
    text: Any, *, allow_unique_embedded: bool
) -> Tuple[Optional[str], str]:
    """Normalize an OCR timer while recording the exact repair policy.

    Primary OCR retains the historical exact ``MM:SS`` behavior.  The bounded
    fallback may extract one embedded valid ``DD:DD`` substring, but only when
    there is exactly one such occurrence.  It never inserts a value from
    neighboring seconds and never forward-fills the timer.
    """

    if text is None:
        return None, "missing_text"
    cleaned = re.sub(r"[^0-9:]", "", str(text))
    match = re.fullmatch(r"(\d{2}):(\d{2})", cleaned)
    if match and int(match.group(2)) < 60:
        return cleaned, "exact_cleaned_mm_ss"
    if allow_unique_embedded:
        embedded = [
            candidate.group(1)
            for candidate in re.finditer(r"(?=(\d{2}:\d{2}))", cleaned)
            if int(candidate.group(1)[3:]) < 60
        ]
        if len(embedded) == 1:
            return embedded[0], "unique_embedded_mm_ss"
        if len(embedded) > 1:
            return None, "ambiguous_embedded_mm_ss"
    return None, "no_valid_mm_ss"


def annotate_timer_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    preprocessing_variant: str,
    allow_unique_embedded: bool,
    fallback: bool,
) -> None:
    for candidate in candidates:
        normalized, policy = normalize_timer_text_with_policy(
            candidate.get("text"),
            allow_unique_embedded=allow_unique_embedded,
        )
        candidate["preprocessing_variant"] = preprocessing_variant
        candidate["normalization_policy"] = policy
        candidate["normalized_text"] = normalized
        candidate["fallback"] = fallback


def normalize_kill_text(text: Any) -> Optional[int]:
    if text is None:
        return None
    cleaned = re.sub(r"[^0-9]", "", str(text))
    if not cleaned:
        return None
    return int(cleaned)


def _candidate_confidence(candidate: Mapping[str, Any]) -> float:
    try:
        confidence = float(candidate.get("confidence", candidate.get("conf", 0.0)))
    except (TypeError, ValueError):
        return 0.0
    return confidence if math.isfinite(confidence) else 0.0


def select_timer(
    candidates: Sequence[Mapping[str, Any]], min_confidence: float = 0.0
) -> TimerSelection:
    valid: List[Tuple[Mapping[str, Any], str, float]] = []
    for candidate in candidates:
        confidence = _candidate_confidence(candidate)
        if "normalized_text" in candidate:
            normalized = candidate.get("normalized_text")
        else:
            normalized = normalize_timer_text(candidate.get("text"))
        if normalized is not None and confidence >= min_confidence:
            valid.append((candidate, str(normalized), confidence))
    if not valid:
        return TimerSelection(False, None, None, None, "missing", 0)
    candidate, normalized, confidence = max(valid, key=lambda item: item[2])
    return TimerSelection(
        True,
        str(candidate.get("text", "")),
        normalized,
        confidence,
        "observed",
        len(valid),
        (
            str(candidate["preprocessing_variant"])
            if candidate.get("preprocessing_variant") is not None
            else None
        ),
        (
            str(candidate["normalization_policy"])
            if candidate.get("normalization_policy") is not None
            else "exact_cleaned_mm_ss"
        ),
    )


def select_kill(
    candidates: Sequence[Mapping[str, Any]],
    previous_state: Optional[int],
    min_confidence: float = 0.0,
    max_jump_override: Optional[int] = None,
) -> KillSelection:
    observed_digit_strings = {
        digits
        for candidate in candidates
        if (digits := re.sub(r"[^0-9]", "", str(candidate.get("text", ""))))
    }

    def transition_value(candidate: Mapping[str, Any]) -> Optional[int]:
        value = normalize_kill_text(candidate.get("text"))
        if value is None or previous_state is None:
            return value
        allowed_jump = max_jump_override or max(75, int(previous_state * 0.15))
        digits = re.sub(r"[^0-9]", "", str(candidate.get("text", "")))
        # Prefer a shorter prefix when it is independently observed elsewhere
        # in the same one-second window.  This catches the frequent skull-edge
        # suffix even while the cumulative count is still small enough that the
        # inflated value would pass the broad early-game jump allowance.
        supported_prefixes = [
            int(digits[:end])
            for end in range(1, len(digits))
            if digits[:end] in observed_digit_strings
            and previous_state <= int(digits[:end]) <= previous_state + allowed_jump
        ]
        if supported_prefixes:
            return min(
                supported_prefixes,
                key=lambda repaired: repaired - previous_state,
            )
        if value - previous_state <= allowed_jump:
            return value
        # Bright HUD/overlay edges can be read as one or more extra leading
        # digits. Accept a stripped suffix only when it forms a plausible
        # monotonic transition from the already observed cumulative counter.
        repaired_values: List[int] = []
        for start in range(1, len(digits)):
            suffix = int(digits[start:])
            if previous_state <= suffix <= previous_state + allowed_jump:
                repaired_values.append(suffix)
        # The skull immediately to the right of the counter is commonly read
        # as one or more trailing digits (14 -> 140, 26 -> 266).  Strip only
        # when the full OCR value is an implausible jump and the prefix is a
        # plausible monotonic transition.  This preserves a genuine 140 when
        # the prior state is already near 140.
        for end in range(len(digits) - 1, 0, -1):
            prefix = int(digits[:end])
            if previous_state <= prefix <= previous_state + allowed_jump:
                repaired_values.append(prefix)
        if repaired_values:
            return min(repaired_values, key=lambda repaired: repaired - previous_state)
        return value

    valid_by_sample: Dict[Any, Tuple[Mapping[str, Any], int, float]] = {}
    max_jump = (
        max_jump_override or max(75, int(previous_state * 0.15))
        if previous_state is not None
        else None
    )
    for candidate_index, candidate in enumerate(candidates):
        confidence = _candidate_confidence(candidate)
        value = transition_value(candidate)
        if value is None or confidence < min_confidence:
            continue
        if previous_state is not None and value < previous_state:
            continue
        if (
            previous_state is not None
            and previous_state > 10
            and max_jump is not None
            and value - previous_state > max_jump
        ):
            continue
        if candidate.get("frame_index") is not None:
            sample_key: Any = ("frame", int(candidate["frame_index"]))
        elif candidate.get("offset_seconds") is not None:
            sample_key = ("offset", float(candidate["offset_seconds"]))
        else:
            sample_key = ("candidate", candidate_index)
        previous = valid_by_sample.get(sample_key)
        if previous is None or confidence > previous[2]:
            valid_by_sample[sample_key] = (candidate, value, confidence)

    valid = list(valid_by_sample.values())

    if not valid:
        source = "carried_forward" if previous_state is not None else "missing"
        return KillSelection(
            False,
            None,
            None,
            None,
            previous_state,
            source,
            source,
            0,
        )

    sorted_values = sorted(item[1] for item in valid)
    # With an even number of decoded samples, prefer the lower median.  A
    # single inflated OCR value must not advance the cumulative state and then
    # poison later seconds through carry-forward.
    median_value = sorted_values[(len(sorted_values) - 1) // 2]
    matching = [item for item in valid if item[1] == median_value]
    candidate, value, confidence = max(matching, key=lambda item: item[2])
    return KillSelection(
        True,
        str(candidate.get("text", "")),
        value,
        confidence,
        value,
        "observed",
        "observed",
        len(valid),
    )


def _kill_candidates_from_record(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [
        candidate
        for sample in record.get("samples", [])
        for candidate in sample.get("kill_candidates", [])
    ]


def _record_is_excluded(record: Mapping[str, Any]) -> bool:
    return any(
        bool(sample.get("excluded_from_gameplay"))
        for sample in record.get("samples", [])
    )


def _record_is_pregame(record: Mapping[str, Any]) -> bool:
    return any(
        sample.get("exclusion_reason") == "pregame_or_loading_before_gameplay_hud"
        for sample in record.get("samples", [])
    )


def select_kill_sequence(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_state: Optional[int],
    min_confidence: float = 0.0,
    overshoot_lookahead_seconds: int = KILL_OVERSHOOT_LOOKAHEAD_SECONDS,
    overshoot_min_supporting_seconds: int = KILL_OVERSHOOT_MIN_SUPPORTING_SECONDS,
    overshoot_min_samples_per_second: int = KILL_OVERSHOOT_MIN_SAMPLES_PER_SECOND,
    large_jump_min_increase: int = KILL_LARGE_JUMP_MIN_INCREASE,
    large_jump_min_fraction: float = KILL_LARGE_JUMP_MIN_FRACTION,
    large_jump_lookahead_seconds: int = KILL_LARGE_JUMP_LOOKAHEAD_SECONDS,
    large_jump_min_future_confidence: float = KILL_LARGE_JUMP_MIN_FUTURE_CONFIDENCE,
    gap_allowance_per_eligible_second: int = KILL_GAP_ALLOWANCE_PER_ELIGIBLE_SECOND,
    gap_allowance_cap: int = KILL_GAP_ALLOWANCE_CAP,
    gap_jump_confirmation_seconds: int = KILL_GAP_JUMP_CONFIRMATION_SECONDS,
    gap_jump_min_future_confidence: float = KILL_GAP_JUMP_MIN_FUTURE_CONFIDENCE,
) -> List[KillSelection]:
    """Select a monotonic kill series without trusting isolated upward OCR spikes.

    A cumulative counter cannot decrease.  Accepting one inflated observation
    therefore poisons every later, correct-but-lower reading until the real
    counter catches up.  Before committing an increase, inspect bounded future
    windows using the *previously trusted* state.  The short window requires
    multiple samples per contradicting second.  A larger proposed jump also
    gets a longer window, requiring two later high-confidence observations
    below it, because heavy visual effects can briefly corrupt all samples.

    This is deterministic bounded lookahead, not interpolation: rejected
    seconds remain explicitly unobserved and the counter carries the last
    trusted state until a later observation is accepted.
    """

    selections: List[KillSelection] = []
    previous_state = initial_state
    last_observed_index: Optional[int] = None
    eligible_gap_seconds = 0
    lookahead = max(0, int(overshoot_lookahead_seconds))

    for index, record in enumerate(records):
        candidates = _kill_candidates_from_record(record)
        excluded = _record_is_excluded(record)
        base_jump = (
            max(75, int(previous_state * 0.15))
            if previous_state is not None else None
        )
        gap_seconds = eligible_gap_seconds if last_observed_index is not None else 0
        allowed_jump = (
            max(
                base_jump,
                min(
                    gap_allowance_cap,
                    base_jump + gap_allowance_per_eligible_second * gap_seconds,
                ),
            )
            if base_jump is not None else None
        )
        selection = select_kill(
            candidates,
            previous_state=previous_state,
            min_confidence=min_confidence,
            max_jump_override=allowed_jump,
        )

        if _record_is_pregame(record):
            _, selection = exclude_non_gameplay_hud_window()
        elif excluded:
            _, selection = exclude_paused_hud_window(
                record.get("samples", []),
                TimerSelection(False, None, None, None, "missing", 0),
                selection,
            )
        elif (
            selection.observed
            and previous_state is not None
            and selection.state_value is not None
            and selection.state_value > previous_state
            and lookahead > 0
        ):
            # Recovery after a long OCR gap may exceed the ordinary one-second
            # jump bound. Require two later readings at or above the proposed
            # value before publishing such a recovered observation.
            if (
                base_jump is not None
                and selection.state_value - previous_state > base_jump
            ):
                confirmations = 0
                for future in records[
                    index + 1 : index + 1 + gap_jump_confirmation_seconds
                ]:
                    if _record_is_excluded(future):
                        continue
                    future_selection = select_kill(
                        _kill_candidates_from_record(future),
                        previous_state=selection.state_value,
                        min_confidence=max(
                            min_confidence, gap_jump_min_future_confidence
                        ),
                    )
                    if future_selection.observed:
                        confirmations += 1
                    if confirmations >= 2:
                        break
                if confirmations < 2:
                    selection = KillSelection(
                        observed=False,
                        raw_text=None,
                        observed_value=None,
                        confidence=None,
                        state_value=previous_state,
                        state_source="carried_forward",
                        status="unconfirmed_gap_jump",
                        valid_candidate_count=selection.valid_candidate_count,
                    )

        if (
            selection.observed
            and previous_state is not None
            and selection.state_value is not None
            and selection.state_value > previous_state
            and lookahead > 0
        ):
            large_jump = (
                selection.state_value - previous_state
                >= max(
                    large_jump_min_increase,
                    math.ceil(previous_state * large_jump_min_fraction),
                )
            )
            window = max(
                lookahead,
                large_jump_lookahead_seconds if large_jump else 0,
            )
            short_contradictions = 0
            large_jump_contradictions = 0
            for distance, future in enumerate(
                records[index + 1 : index + 1 + window], start=1
            ):
                if _record_is_excluded(future):
                    continue
                future_selection = select_kill(
                    _kill_candidates_from_record(future),
                    previous_state=previous_state,
                    min_confidence=min_confidence,
                )
                contradictory = (
                    future_selection.observed
                    and future_selection.state_value is not None
                    and previous_state <= future_selection.state_value < selection.state_value
                )
                if contradictory and distance <= lookahead and (
                    future_selection.valid_candidate_count
                    >= overshoot_min_samples_per_second
                ):
                    short_contradictions += 1
                if contradictory and large_jump and (
                    future_selection.confidence is not None
                    and future_selection.confidence
                    >= large_jump_min_future_confidence
                ):
                    large_jump_contradictions += 1
                if (
                    short_contradictions >= overshoot_min_supporting_seconds
                    or large_jump_contradictions
                    >= overshoot_min_supporting_seconds
                ):
                    selection = KillSelection(
                        observed=False,
                        raw_text=None,
                        observed_value=None,
                        confidence=None,
                        state_value=previous_state,
                        state_source="carried_forward",
                        status="rejected_temporal_overshoot",
                        valid_candidate_count=selection.valid_candidate_count,
                    )
                    break

        previous_state = advance_kill_state(
            previous_state,
            selection,
            excluded_window=excluded,
        )
        if selection.observed:
            last_observed_index = index
            eligible_gap_seconds = 0
        elif not excluded:
            eligible_gap_seconds += 1
        selections.append(selection)

    return selections


def _kill_selection_payload(selection: KillSelection) -> Dict[str, Any]:
    return {
        "observed": selection.observed,
        "raw_text": selection.raw_text,
        "observed_value": selection.observed_value,
        "confidence": selection.confidence,
        "state_value": selection.state_value,
        "state_source": selection.state_source,
        "status": selection.status,
        "valid_candidate_count": selection.valid_candidate_count,
    }


def reconcile_kill_output_rows(
    output_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    *,
    initial_state: Optional[int],
    min_confidence: float,
) -> List[int]:
    """Apply bounded temporal kill selections to the worker's final ledgers."""

    selections = select_kill_sequence(
        candidate_rows,
        initial_state=initial_state,
        min_confidence=min_confidence,
    )
    changed_seconds: List[int] = []
    for row, candidate_row, selection in zip(
        output_rows, candidate_rows, selections
    ):
        payload = _kill_selection_payload(selection)
        if candidate_row["selection"]["kill"] != payload:
            changed_seconds.append(int(row["Video Second"]))
        candidate_row["selection"]["kill"] = payload
        row["Kill Counter Quantity"] = (
            selection.state_value if selection.state_value is not None else ""
        )
        row["kill_observed"] = selection.observed
        row["kill_raw_text"] = selection.raw_text or ""
        row["kill_observed_value"] = (
            selection.observed_value
            if selection.observed_value is not None
            else ""
        )
        row["kill_confidence"] = (
            round(selection.confidence, 8)
            if selection.confidence is not None
            else ""
        )
        row["kill_state_value"] = (
            selection.state_value if selection.state_value is not None else ""
        )
        row["kill_state_source"] = selection.state_source
        row["kill_status"] = selection.status
        row["review_flag"] = bool(
            not row["timer_observed"]
            or row["timer_fallback_attempted"]
            or selection.state_source != "observed"
            or int(row["decoded_sample_count"])
            != len(candidate_row.get("samples", []))
        )
    return changed_seconds


def parse_sampling_offsets(text: str) -> Tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("sample offsets must be comma-separated numbers") from error
    if not values or any(not 0.0 <= value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("every sample offset must satisfy 0 <= offset < 1")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("sample offsets must be unique and increasing")
    return values


def seconds_to_process(
    total_duration_seconds: float,
    full: bool,
    max_seconds: Optional[int],
) -> int:
    if not math.isfinite(total_duration_seconds) or total_duration_seconds <= 0:
        raise WorkerError("Video duration must be positive.", EXIT_INPUT_OR_DEPENDENCY)
    available_complete_seconds = max(1, int(total_duration_seconds))
    if full:
        return available_complete_seconds
    if max_seconds is None or max_seconds <= 0:
        raise WorkerError("--max-seconds must be a positive integer.", EXIT_INPUT_OR_DEPENDENCY)
    return min(max_seconds, available_complete_seconds)


def detect_gameplay_start_second(
    capture: Any,
    cv2_module: Any,
    fps: float,
    frame_count: int,
    processed_seconds: int,
    sample_offsets: Sequence[float],
    *,
    minimum_hud_score: float = 0.90,
    confirmation_seconds: int = 2,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """Find the first sustained gameplay HUD without using video-specific time.

    Menus and loading screens may expose digits or skull-like artwork in the
    eventual OCR regions.  The long gold-bordered XP bar is the shared visual
    prerequisite for HUD measurements.  Requiring it in multiple samples for
    consecutive seconds prevents a transient menu decoration from starting the
    cumulative kill state.
    """
    evidence: List[Dict[str, Any]] = []
    run: List[int] = []
    required_samples = max(1, min(2, len(sample_offsets)))
    for second in range(processed_seconds):
        scores: List[float] = []
        eligible_samples = 0
        for offset in sample_offsets:
            frame_index = min(frame_count - 1, int((second + offset) * fps))
            capture.set(cv2_module.CAP_PROP_POS_FRAMES, frame_index)
            decoded, frame = capture.read()
            if not decoded:
                continue
            score = float(gameplay_hud_score(frame))
            scores.append(score)
            if score >= minimum_hud_score and not gameplay_pause_evidence(frame)["blocked"]:
                eligible_samples += 1
        qualifies = eligible_samples >= required_samples
        evidence.append({
            "video_second": second,
            "decoded_sample_count": len(scores),
            "qualifying_sample_count": eligible_samples,
            "maximum_gameplay_hud_score": round(max(scores), 6) if scores else None,
            "qualifies": qualifies,
        })
        if qualifies:
            run.append(second)
            if len(run) >= confirmation_seconds:
                return run[-confirmation_seconds], evidence
        else:
            run.clear()
    return None, evidence


def clamp_roi(
    roi: Sequence[int], frame_width: int, frame_height: int
) -> Tuple[int, int, int, int]:
    x, y, width, height = (int(value) for value in roi)
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(1, min(width, frame_width - x))
    height = max(1, min(height, frame_height - y))
    return x, y, width, height


def fallback_rois(frame_width: int, frame_height: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    aspect_ratio = frame_width / frame_height if frame_height else 0.0
    if aspect_ratio >= 1.70:
        kills_right_pct = 0.793
        kills_width_pct = 0.060
    else:
        kills_right_pct = 0.828
        kills_width_pct = 0.065
    timer_roi = (
        int(0.455 * frame_width),
        int(0.045 * frame_height),
        int(0.090 * frame_width),
        int(0.050 * frame_height),
    )
    kills_width = int(kills_width_pct * frame_width)
    kills_roi = (
        int(kills_right_pct * frame_width) - kills_width,
        int(0.028 * frame_height),
        kills_width,
        int(0.045 * frame_height),
    )
    return (
        clamp_roi(timer_roi, frame_width, frame_height),
        clamp_roi(kills_roi, frame_width, frame_height),
    )


def crop_roi(frame: Any, roi: Sequence[int]) -> Any:
    x, y, width, height = roi
    return frame[y : y + height, x : x + width]


def _bbox_json(bbox: Any) -> List[List[float]]:
    return [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in bbox]


def ocr_results_to_candidates(results: Sequence[Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for bbox, text, confidence in results:
        candidates.append(
            {
                "text": str(text),
                "confidence": round(float(confidence), 8),
                "bbox": _bbox_json(bbox),
            }
        )
    return candidates


def preprocess_for_ocr(frame: Any, cv2: Any) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return threshold


def timer_fallback_preprocessing_variants(
    timer_crop: Any, cv2: Any
) -> Tuple[Tuple[str, Any], ...]:
    """Return deterministic fallback views of one cached timer crop."""

    raw_upscaled = cv2.resize(
        timer_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2GRAY)
    gray_upscaled = cv2.resize(
        gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    _, otsu = cv2.threshold(
        gray_upscaled,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return (
        (TIMER_FALLBACK_PREPROCESSING_VARIANTS[0], raw_upscaled),
        (TIMER_FALLBACK_PREPROCESSING_VARIANTS[1], gray_upscaled),
        (TIMER_FALLBACK_PREPROCESSING_VARIANTS[2], otsu),
    )


def run_timer_fallback_ocr(
    *,
    timer_crop: Any,
    reader: Any,
    cv2: Any,
    offset_seconds: float,
    frame_index: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """OCR one cached crop using named variants after all primary samples miss."""

    flat_candidates: List[Dict[str, Any]] = []
    variant_records: List[Dict[str, Any]] = []
    for variant_name, processed_crop in timer_fallback_preprocessing_variants(
        timer_crop, cv2
    ):
        candidates = ocr_results_to_candidates(
            reader.readtext(
                processed_crop,
                allowlist="0123456789:",
                detail=1,
            )
        )
        annotate_timer_candidates(
            candidates,
            preprocessing_variant=variant_name,
            allow_unique_embedded=True,
            fallback=True,
        )
        for candidate in candidates:
            candidate["offset_seconds"] = offset_seconds
            candidate["frame_index"] = frame_index
        flat_candidates.extend(candidates)
        variant_records.append(
            {
                "preprocessing_variant": variant_name,
                "candidates": candidates,
            }
        )
    return flat_candidates, variant_records


def _relative_component_path(path: Path) -> str:
    return path.resolve().relative_to(COMPONENT_DIR).as_posix()


def _find_skull_roi(
    frame: Any, cv2: Any, numpy: Any
) -> Tuple[Optional[Tuple[int, int, int, int]], float, Optional[str]]:
    frame_height, frame_width = frame.shape[:2]
    top_strip_height = min(frame_height, max(100, int(frame_height * 0.12)))
    # The kill counter is immediately to the left of the skull, while the coin
    # counter is farther right beside a different icon. Keep the search inside
    # the top HUD strip and around the skull's expected horizontal band. The
    # previous extreme-right search could match the coin icon and silently turn
    # currency into a plausible-looking "kill" series.
    search_x0 = int(frame_width * 0.70)
    search_x1 = int(frame_width * 0.88)
    search_gray = cv2.cvtColor(
        frame[:top_strip_height, search_x0:search_x1], cv2.COLOR_BGR2GRAY
    )
    expected_scale = frame_height / 1800.0
    scales = numpy.linspace(
        max(0.35, expected_scale * 0.55),
        min(1.35, expected_scale * 1.65),
        19,
    )
    best: Optional[Tuple[float, Any, int, int, Path]] = None
    for template_path in template_paths():
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        for scale in scales:
            scaled_width = max(8, int(template.shape[1] * scale))
            scaled_height = max(8, int(template.shape[0] * scale))
            if (
                scaled_width >= search_gray.shape[1]
                or scaled_height >= search_gray.shape[0]
            ):
                continue
            scaled = cv2.resize(
                template,
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_AREA,
            )
            match = cv2.matchTemplate(search_gray, scaled, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(match)
            if best is None or score > best[0]:
                best = (float(score), location, scaled_width, scaled_height, template_path)
    if best is None:
        return None, 0.0, None
    score, location, skull_width, skull_height, template_path = best
    relative_template = _relative_component_path(template_path)
    if score < 0.35:
        return None, score, relative_template
    roi = clamp_roi(
        (
            search_x0 + location[0],
            location[1],
            skull_width,
            skull_height,
        ),
        frame_width,
        frame_height,
    )
    return roi, score, relative_template


def _kill_roi_from_skull(
    skull_roi: Sequence[int], frame_width: int, frame_height: int
) -> Tuple[int, int, int, int]:
    skull_x, skull_y, skull_width, skull_height = skull_roi
    gap = max(3, int(frame_width * 0.003))
    # Keep enough room for a multi-digit counter without flooding the OCR crop
    # with background pixels. At 1080p the four-digit value is about 75 px wide.
    left_width = max(int(frame_width * 0.065), skull_width * 5)
    # The numeric baseline is aligned with the skull, but the glyph ascenders
    # begin noticeably above the icon at 1080p. Preserve the full digit height;
    # a tighter crop clips the tops of 2/3/4/5 and creates confident mutations.
    pad_y = max(12, int(frame_height * 0.016))
    return clamp_roi(
        (
            skull_x - gap - left_width,
            skull_y - pad_y,
            left_width,
            skull_height + 2 * pad_y,
        ),
        frame_width,
        frame_height,
    )


def find_hud_rois(
    frame: Any, reader: Any, cv2: Any, numpy: Any
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Dict[str, Any]]:
    frame_height, frame_width = frame.shape[:2]
    top_strip_height = min(frame_height, max(120, int(frame_height * 0.18)))
    top_candidates = ocr_results_to_candidates(
        reader.readtext(frame[:top_strip_height, :], detail=1)
    )
    timer_roi: Optional[Tuple[int, int, int, int]] = None
    kill_roi: Optional[Tuple[int, int, int, int]] = None
    timer_source = "fallback"
    kill_source = "fallback"
    timer_matches: List[Tuple[float, Tuple[int, int, int, int]]] = []
    numeric_matches: List[Tuple[float, Tuple[int, int, int, int]]] = []

    for candidate in top_candidates:
        bbox = candidate["bbox"]
        x = int(min(point[0] for point in bbox))
        y = int(min(point[1] for point in bbox))
        width = max(1, int(max(point[0] for point in bbox) - x))
        height = max(1, int(max(point[1] for point in bbox) - y))
        confidence = float(candidate["confidence"])
        if normalize_timer_text(candidate["text"]) is not None:
            pad_x = max(12, int(frame_width * 0.010))
            pad_y = max(6, int(frame_height * 0.008))
            timer_matches.append(
                (
                    confidence,
                    clamp_roi(
                        (
                            x - pad_x,
                            y - pad_y,
                            width + 2 * pad_x,
                            height + 2 * pad_y,
                        ),
                        frame_width,
                        frame_height,
                    ),
                )
            )
        raw_candidate_text = str(candidate["text"])
        if (
            x > frame_width // 2
            and y >= int(frame_height * 0.02)
            and not re.search(r"[A-Za-z]", raw_candidate_text)
            and normalize_kill_text(raw_candidate_text) is not None
        ):
            left_pad = max(35, int(frame_width * 0.025))
            right_pad = max(25, int(frame_width * 0.015))
            pad_y = max(6, int(frame_height * 0.008))
            numeric_matches.append(
                (
                    confidence,
                    clamp_roi(
                        (
                            x - left_pad,
                            y - pad_y,
                            width + left_pad + right_pad,
                            height + 2 * pad_y,
                        ),
                        frame_width,
                        frame_height,
                    ),
                )
            )

    if timer_matches:
        timer_roi = max(timer_matches, key=lambda item: item[0])[1]
        timer_source = "calibration_ocr"
    if numeric_matches:
        kill_roi = max(numeric_matches, key=lambda item: item[0])[1]
        kill_source = "calibration_ocr"

    skull_roi, skull_score, skull_template = _find_skull_roi(frame, cv2, numpy)
    # The skull anchor establishes the meaning of the number. A high-confidence
    # numeric OCR result alone cannot distinguish kills from coins, so it must
    # not override a valid skull-relative crop.
    if skull_roi is not None:
        kill_roi = _kill_roi_from_skull(skull_roi, frame_width, frame_height)
        kill_source = "skull_template"

    fallback_timer, fallback_kill = fallback_rois(frame_width, frame_height)
    if timer_roi is None:
        timer_roi = fallback_timer
    if kill_roi is None:
        kill_roi = fallback_kill

    metadata_payload = {
        "timer_roi": list(timer_roi),
        "kill_roi": list(kill_roi),
        "timer_roi_source": timer_source,
        "kill_roi_source": kill_source,
        "skull_roi": list(skull_roi) if skull_roi is not None else None,
        "skull_match_score": round(float(skull_score), 8),
        "skull_template": skull_template,
        "top_strip_ocr_candidates": top_candidates,
    }
    return timer_roi, kill_roi, metadata_payload


def _draw_evidence_frame(
    frame: Any,
    timer_roi: Sequence[int],
    kill_roi: Sequence[int],
    timer_text: str,
    kill_text: str,
    cv2: Any,
) -> Any:
    result = frame.copy()
    tx, ty, tw, th = timer_roi
    kx, ky, kw, kh = kill_roi
    cv2.rectangle(result, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 3)
    cv2.rectangle(result, (kx, ky), (kx + kw, ky + kh), (255, 0, 0), 3)
    cv2.putText(
        result,
        "Timer observed: %s" % timer_text,
        (tx, max(30, ty - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        result,
        "Kill state: %s" % kill_text,
        (kx, max(30, ky - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )
    return result


def _save_jpeg_atomic(path: Path, frame: Any, cv2: Any) -> str:
    encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise WorkerError("OpenCV could not encode an evidence frame.", EXIT_PROCESSING)
    write_bytes_atomic(path, bytes(buffer))
    return sha256_file(path)


def refresh_reconciled_kill_evidence(
    *,
    video_path: Path,
    output_dir: Path,
    output_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    changed_seconds: Sequence[int],
    timer_roi: Sequence[int],
    kill_roi: Sequence[int],
    max_evidence_frames: int,
    cv2: Any,
) -> None:
    """Redraw evidence whose kill state changed during temporal reconciliation."""

    if not changed_seconds:
        return
    evidence_by_second = {
        int(row["video_second"]): row
        for row in evidence_rows
        if isinstance(row.get("video_second"), int)
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise WorkerError(
            "Could not reopen the video to refresh reconciled HUD evidence.",
            EXIT_INPUT_OR_DEPENDENCY,
        )
    try:
        for second in changed_seconds:
            row = output_rows[second]
            candidate_row = candidate_rows[second]
            existing = evidence_by_second.get(second)
            if existing is None and len(evidence_rows) >= max_evidence_frames:
                continue
            decoded_samples = [
                sample
                for sample in candidate_row.get("samples", [])
                if sample.get("decoded") and sample.get("frame_index") is not None
            ]
            if not decoded_samples:
                continue
            sample = min(
                decoded_samples,
                key=lambda value: abs(float(value.get("offset_seconds", 0.0)) - 0.5),
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_index"]))
            decoded, frame = capture.read()
            if not decoded:
                continue
            relative_path = (
                str(existing["relative_path"])
                if existing is not None
                else "evidence/hud_second_%06d.jpg" % second
            )
            annotated = _draw_evidence_frame(
                frame,
                timer_roi,
                kill_roi,
                str(row["timer_normalized_text"] or "MISSING"),
                str(
                    row["kill_state_value"]
                    if row["kill_state_value"] != ""
                    else "MISSING"
                ),
                cv2,
            )
            evidence_hash = _save_jpeg_atomic(
                output_dir / relative_path, annotated, cv2
            )
            if existing is None:
                existing = {
                    "video_second": second,
                    "relative_path": relative_path,
                    "reason": "temporal_reconciliation",
                    "sha256": evidence_hash,
                    "timer_status": row["timer_status"],
                    "kill_state_source": row["kill_state_source"],
                }
                evidence_rows.append(existing)
                evidence_by_second[second] = existing
            else:
                reasons = set(str(existing.get("reason", "")).split("+"))
                reasons.discard("")
                reasons.add("temporal_reconciliation")
                existing.update(
                    {
                        "reason": "+".join(sorted(reasons)),
                        "sha256": evidence_hash,
                        "timer_status": row["timer_status"],
                        "kill_state_source": row["kill_state_source"],
                    }
                )
            row["sample_frame_path"] = relative_path
            candidate_row["evidence_frame"] = relative_path
    finally:
        capture.release()


def _load_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import cv2
        import easyocr
        import numpy
    except Exception as error:
        raise WorkerError(
            "Required OCR dependency is unavailable: %s" % error,
            EXIT_INPUT_OR_DEPENDENCY,
        ) from error
    return cv2, easyocr, numpy


def preflight_easyocr_models(
    easyocr_module: Any,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Resolve and hash the exact default English detector/recognizer weights."""

    easyocr_config = getattr(easyocr_module, "config", None)
    if easyocr_config is None:
        try:
            easyocr_config = importlib.import_module("easyocr.config")
        except Exception as error:
            raise WorkerError(
                "Could not inspect EasyOCR model configuration.",
                EXIT_INPUT_OR_DEPENDENCY,
            ) from error
    try:
        model_directory = Path(str(easyocr_config.MODULE_PATH)).expanduser() / "model"
        detector_spec = easyocr_config.detection_models[
            EASYOCR_DETECTION_NETWORK
        ]
        recognizer_spec = easyocr_config.recognition_models["gen2"]["english_g2"]
        effective_specs = (
            ("detector", detector_spec),
            ("recognizer", recognizer_spec),
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise WorkerError(
            "EasyOCR model configuration is incompatible with this worker.",
            EXIT_INPUT_OR_DEPENDENCY,
        ) from error

    records: List[Dict[str, Any]] = []
    for role, model_spec in effective_specs:
        try:
            configured_filename = str(model_spec["filename"])
        except (KeyError, TypeError) as error:
            raise WorkerError(
                "EasyOCR %s model configuration has no filename." % role,
                EXIT_INPUT_OR_DEPENDENCY,
            ) from error
        filename = Path(configured_filename).name
        model_path = model_directory / configured_filename
        if not model_path.is_file():
            raise WorkerError(
                "Required EasyOCR %s weight '%s' is missing; downloads are disabled."
                % (role, filename),
                EXIT_INPUT_OR_DEPENDENCY,
            )
        size_bytes = model_path.stat().st_size
        if size_bytes <= 0:
            raise WorkerError(
                "Required EasyOCR %s weight '%s' is empty."
                % (role, filename),
                EXIT_INPUT_OR_DEPENDENCY,
            )
        try:
            content_sha256 = sha256_file(model_path)
        except OSError as error:
            raise WorkerError(
                "Could not hash EasyOCR %s weight '%s'." % (role, filename),
                EXIT_INPUT_OR_DEPENDENCY,
            ) from error
        records.append(
            {
                "role": role,
                "filename": filename,
                "size_bytes": size_bytes,
                "sha256": content_sha256,
            }
        )
    return model_directory, records


def create_easyocr_reader(
    easyocr_module: Any, model_directory: Path, gpu: bool
) -> Any:
    try:
        return easyocr_module.Reader(
            list(EASYOCR_LANGUAGES),
            gpu=gpu,
            model_storage_directory=str(model_directory),
            detect_network=EASYOCR_DETECTION_NETWORK,
            recog_network=EASYOCR_RECOGNITION_NETWORK,
            download_enabled=False,
        )
    except Exception as error:
        raise WorkerError(
            "EasyOCR initialization failed with downloads disabled (%s)."
            % type(error).__name__,
            EXIT_INPUT_OR_DEPENDENCY,
        ) from error


def _dependency_version(package_name: str) -> Optional[str]:
    from ..runtime import distribution_version
    try:
        return distribution_version(package_name)
    except metadata.PackageNotFoundError:
        return None


def dependency_versions() -> Dict[str, Optional[str]]:
    return {
        "easyocr": _dependency_version("easyocr"),
        "numpy": _dependency_version("numpy"),
        "opencv_python": _dependency_version("opencv-python"),
        "python": sys.version.split()[0],
        "torch": _dependency_version("torch"),
    }


def _safe_template_manifest() -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for path in template_paths():
        if not path.is_file():
            continue
        result.append(
            {
                "path": _relative_component_path(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return result


def _validate_output_target(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise WorkerError("--output-dir is not a directory.", EXIT_INPUT_OR_DEPENDENCY)
    conflicts = [name for name in MANAGED_OUTPUT_NAMES if (output_dir / name).exists()]
    evidence_dir = output_dir / "evidence"
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        conflicts.append("evidence/")
    if conflicts:
        raise WorkerError(
            "Output directory already contains managed artifacts: %s"
            % ", ".join(conflicts),
            EXIT_INPUT_OR_DEPENDENCY,
        )


def _video_sha256(path: Path, skip: bool) -> Tuple[Optional[str], str]:
    if skip:
        return None, "skipped_by_cli"
    return sha256_file(path), "verified"


def _build_qc(
    rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    processed_seconds: int,
    decoded_sample_count: int,
    min_timer_observed_rate: float,
    min_kill_observed_rate: float,
) -> Dict[str, Any]:
    timer_observed_count = sum(bool(row["timer_observed"]) for row in rows)
    timer_fallback_attempted_count = sum(
        bool(row["timer_fallback_attempted"]) for row in rows
    )
    timer_fallback_recovered_count = sum(
        bool(row["timer_fallback_attempted"])
        and bool(row["timer_observed"])
        and row["timer_preprocessing_variant"]
        in TIMER_FALLBACK_PREPROCESSING_VARIANTS
        for row in rows
    )
    timer_fallback_unique_embedded_count = sum(
        bool(row["timer_fallback_attempted"])
        and bool(row["timer_observed"])
        and row["timer_normalization_policy"] == "unique_embedded_mm_ss"
        for row in rows
    )
    kill_observed_count = sum(bool(row["kill_observed"]) for row in rows)
    kill_temporal_overshoot_rejected_count = sum(
        row["kill_status"] == "rejected_temporal_overshoot"
        for row in rows
    )
    kill_gap_jump_unconfirmed_count = sum(
        row["kill_status"] == "unconfirmed_gap_jump" for row in rows
    )
    carried_count = sum(row["kill_state_source"] == "carried_forward" for row in rows)
    assumed_count = sum(row["kill_state_source"] == "assumed_initial_state" for row in rows)
    missing_kill_state_count = sum(row["kill_state_value"] == "" for row in rows)
    excluded_count = sum(bool(row.get("excluded_from_gameplay")) for row in rows)
    pregame_excluded_count = sum(
        row.get("exclusion_reason") == "pregame_or_loading_before_gameplay_hud"
        for row in rows
    )
    pause_excluded_count = excluded_count - pregame_excluded_count
    eligible_seconds = processed_seconds - excluded_count
    # Vampire Survivors renders no numeric glyph beside the skull while the
    # cumulative kill state is still the configured initial zero.  Those
    # anchored blank seconds are valid state coverage, not failed OCR.  Stop
    # this exemption at the first positive counter so intermittent OCR of the
    # rendered zero does not prematurely end the genuine zero-state period.
    # Later blank crops remain visible as genuine carried-forward observations.
    initial_blank_zero_count = 0
    for row in rows:
        if bool(row.get("excluded_from_gameplay")):
            continue
        try:
            kill_state_value = int(row["kill_state_value"])
        except (TypeError, ValueError):
            kill_state_value = None
        if kill_state_value is not None and kill_state_value > 0:
            break
        if (
            not bool(row["kill_observed"])
            and kill_state_value == 0
            and row["kill_state_source"] in {"carried_forward", "assumed_initial_state"}
        ):
            initial_blank_zero_count += 1
    kill_quality_eligible_seconds = max(0, eligible_seconds - initial_blank_zero_count)
    timer_rate = timer_observed_count / eligible_seconds if eligible_seconds else 0.0
    kill_rate = (
        kill_observed_count / kill_quality_eligible_seconds
        if kill_quality_eligible_seconds else 0.0
    )
    kill_states = [
        int(row["kill_state_value"])
        for row in rows
        if row["kill_state_value"] != ""
    ]
    checks = {
        "output_row_count_matches_requested_seconds": len(rows) == processed_seconds,
        "candidate_row_count_matches_output_rows": len(candidate_rows) == len(rows),
        "video_seconds_are_contiguous": [row["Video Second"] for row in rows]
        == list(range(processed_seconds)),
        "at_least_one_frame_decoded": decoded_sample_count > 0,
        "kill_state_is_monotonic": all(
            current >= previous for previous, current in zip(kill_states, kill_states[1:])
        ),
        "timer_observed_rate_meets_threshold": timer_rate >= min_timer_observed_rate,
        "kill_observed_rate_meets_threshold": kill_rate >= min_kill_observed_rate,
        "timer_is_never_forward_filled": all(
            row["timer_status"] in {
                "observed", "missing", "excluded_level_up_pause",
                "excluded_non_gameplay_hud",
            } for row in rows
        ),
        "kill_state_provenance_is_explicit": all(
            row["kill_state_source"]
            in {
                "observed", "carried_forward", "assumed_initial_state",
                "missing", "excluded_level_up_pause", "excluded_non_gameplay_hud",
            }
            for row in rows
        ),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    warnings: List[str] = []
    if timer_rate < 0.5:
        warnings.append("timer_observed_rate_below_0.5")
    if kill_rate < 0.5:
        warnings.append("kill_observed_rate_below_0.5")
    status = "failed" if failed_checks else ("warning" if warnings else "passed")
    return {
        "prepared_by": PREPARED_BY,
        "artifact_type": "hud_detector_qc",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": status,
        "checks": checks,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "metrics": {
            "requested_seconds": processed_seconds,
            "gameplay_eligible_seconds": eligible_seconds,
            "level_up_pause_excluded_seconds": pause_excluded_count,
            "pregame_or_loading_excluded_seconds": pregame_excluded_count,
            "output_rows": len(rows),
            "decoded_sample_count": decoded_sample_count,
            "timer_observed_count": timer_observed_count,
            "timer_missing_count": eligible_seconds - timer_observed_count,
            "timer_observed_rate": round(timer_rate, 8),
            "timer_fallback_attempted_count": timer_fallback_attempted_count,
            "timer_fallback_recovered_count": timer_fallback_recovered_count,
            "timer_fallback_unique_embedded_count": (
                timer_fallback_unique_embedded_count
            ),
            "kill_observed_count": kill_observed_count,
            "kill_observed_rate": round(kill_rate, 8),
            "kill_temporal_overshoot_rejected_count": (
                kill_temporal_overshoot_rejected_count
            ),
            "kill_gap_jump_unconfirmed_count": kill_gap_jump_unconfirmed_count,
            "kill_quality_eligible_seconds": kill_quality_eligible_seconds,
            "initial_blank_zero_exempt_seconds": initial_blank_zero_count,
            "kill_carried_forward_count": carried_count,
            "kill_assumed_initial_state_count": assumed_count,
            "kill_missing_state_count": missing_kill_state_count,
            "review_row_count": sum(bool(row["review_flag"]) for row in rows),
        },
        "quality_thresholds": {
            "min_timer_observed_rate": min_timer_observed_rate,
            "min_kill_observed_rate": min_kill_observed_rate,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Vampire Survivors HUD clock and cumulative kill state with "
            "raw OCR evidence and explicit observation provenance."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run_length = parser.add_mutually_exclusive_group(required=True)
    run_length.add_argument("--max-seconds", type=int)
    run_length.add_argument("--full", action="store_true")
    parser.add_argument(
        "--sample-offsets",
        type=parse_sampling_offsets,
        default=SAMPLING_OFFSETS_DEFAULT,
        help="Comma-separated offsets inside each second (default: 0.3,0.5,0.7).",
    )
    parser.add_argument("--min-ocr-confidence", type=float, default=0.0)
    parser.add_argument("--min-timer-observed-rate", type=float, default=0.0)
    parser.add_argument("--min-kill-observed-rate", type=float, default=0.0)
    parser.add_argument(
        "--evidence-every",
        type=int,
        default=30,
        help="Save an annotated sample every N video seconds; 0 disables periodic samples.",
    )
    parser.add_argument(
        "--max-evidence-frames",
        type=int,
        default=250,
        help="Upper bound including calibration and review frames.",
    )
    parser.add_argument(
        "--no-review-frames",
        action="store_true",
        help="Do not add evidence frames for missing or carried observations.",
    )
    parser.add_argument(
        "--initial-kill-state",
        type=int,
        default=0,
        help="Explicit state before second zero (default: 0 for a run starting at video start).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Ask EasyOCR to use a GPU. CPU is the reproducible default.",
    )
    parser.add_argument(
        "--skip-video-sha256",
        action="store_true",
        help="Development-only: record that source content hashing was skipped.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise WorkerError("--max-seconds must be positive.", EXIT_INPUT_OR_DEPENDENCY)
    if not 0.0 <= args.min_ocr_confidence <= 1.0:
        raise WorkerError("--min-ocr-confidence must be in [0, 1].", EXIT_INPUT_OR_DEPENDENCY)
    for field_name in ("min_timer_observed_rate", "min_kill_observed_rate"):
        value = getattr(args, field_name)
        if not 0.0 <= value <= 1.0:
            raise WorkerError("--%s must be in [0, 1]." % field_name.replace("_", "-"), EXIT_INPUT_OR_DEPENDENCY)
    if args.evidence_every < 0:
        raise WorkerError("--evidence-every cannot be negative.", EXIT_INPUT_OR_DEPENDENCY)
    if args.max_evidence_frames < 1:
        raise WorkerError("--max-evidence-frames must be at least 1.", EXIT_INPUT_OR_DEPENDENCY)
    if args.initial_kill_state < 0:
        raise WorkerError("--initial-kill-state cannot be negative.", EXIT_INPUT_OR_DEPENDENCY)


def calibration_candidate_seconds(
    gameplay_start_second: float, processed_seconds: int
) -> List[float]:
    """Return fixed checkpoints plus candidates near actual gameplay onset.

    Recordings can spend over a minute in pre-game screens.  A later fixed
    checkpoint may be visually occluded or have incomplete HUD anchors even
    though the sustained-HUD gate correctly found gameplay earlier.  Sampling
    several offsets from that detected boundary makes calibration independent
    of a particular video's intro duration.
    """
    fixed = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0)
    onset_offsets = (0.5, 1.0, 2.0, 5.0, 10.0)
    return sorted({
        float(second)
        for second in (
            *(value for value in fixed if value >= gameplay_start_second),
            *(gameplay_start_second + offset for offset in onset_offsets),
        )
        if gameplay_start_second <= second < processed_seconds
    })


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    started_at = utc_now()
    started_perf = time.perf_counter()
    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not video_path.is_file():
        raise WorkerError("Input video does not exist or is not a file.", EXIT_INPUT_OR_DEPENDENCY)
    _validate_output_target(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    cv2, easyocr, numpy = _load_dependencies()
    easyocr_model_directory, easyocr_model_weights = preflight_easyocr_models(
        easyocr
    )
    resolved_dependency_versions = dependency_versions()
    dependency_version_fingerprint = deterministic_json_fingerprint(
        resolved_dependency_versions
    )
    video_sha256, video_hash_status = _video_sha256(
        video_path, args.skip_video_sha256
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise WorkerError("OpenCV could not open the input video.", EXIT_INPUT_OR_DEPENDENCY)

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise WorkerError("Video FPS or frame count is invalid.", EXIT_INPUT_OR_DEPENDENCY)
        duration_seconds = frame_count / fps
        processed_seconds = seconds_to_process(
            duration_seconds, args.full, args.max_seconds
        )
        gameplay_start_second, gameplay_start_evidence = detect_gameplay_start_second(
            capture,
            cv2,
            fps,
            frame_count,
            processed_seconds,
            args.sample_offsets,
        )
        if gameplay_start_second is None:
            raise WorkerError(
                "Could not find a sustained gameplay HUD in the processed interval.",
                EXIT_PROCESSING,
            )
        reader = create_easyocr_reader(
            easyocr, easyocr_model_directory, gpu=args.gpu
        )
        candidate_seconds = calibration_candidate_seconds(
            gameplay_start_second, processed_seconds
        )
        if not candidate_seconds:
            candidate_seconds = [gameplay_start_second + 0.5]
        calibration_candidates: List[Tuple[Tuple[int, float, int], float, Any, Any, Any, Dict[str, Any]]] = []
        for candidate_second in candidate_seconds:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(candidate_second * fps))
            decoded, candidate_frame = capture.read()
            if not decoded:
                continue
            if gameplay_pause_evidence(candidate_frame)["blocked"]:
                continue
            if gameplay_hud_score(candidate_frame) < 0.90:
                continue
            candidate_timer_roi, candidate_kill_roi, candidate_metadata = find_hud_rois(
                candidate_frame, reader, cv2, numpy
            )
            candidate_score = (
                int(candidate_metadata["kill_roi_source"] == "skull_template"),
                float(candidate_metadata["skull_match_score"]),
                int(candidate_metadata["timer_roi_source"] == "calibration_ocr"),
            )
            calibration_candidates.append(
                (
                    candidate_score,
                    candidate_second,
                    candidate_frame,
                    candidate_timer_roi,
                    candidate_kill_roi,
                    candidate_metadata,
                )
            )
        if not calibration_candidates:
            raise WorkerError("Could not decode a calibration frame.", EXIT_PROCESSING)
        (
            _,
            calibration_second,
            calibration_frame,
            timer_roi,
            kill_roi,
            calibration,
        ) = max(calibration_candidates, key=lambda item: item[0])
        calibration["video_second"] = round(calibration_second, 3)
        calibration["candidate_seconds"] = candidate_seconds
        calibration["gameplay_start_second"] = gameplay_start_second
        calibration["gameplay_start_policy"] = (
            "two_consecutive_seconds_with_two_samples_at_gameplay_hud_score_gte_0.90"
        )
        calibration["gameplay_start_evidence"] = gameplay_start_evidence

        evidence_rows: List[Dict[str, Any]] = []
        calibration_relative = "evidence/calibration.jpg"
        calibration_image = _draw_evidence_frame(
            calibration_frame,
            timer_roi,
            kill_roi,
            "CALIBRATION",
            "CALIBRATION",
            cv2,
        )
        calibration_hash = _save_jpeg_atomic(
            output_dir / calibration_relative, calibration_image, cv2
        )
        evidence_rows.append(
            {
                "video_second": round(calibration_second, 3),
                "relative_path": calibration_relative,
                "reason": "calibration",
                "sha256": calibration_hash,
                "timer_status": "calibration",
                "kill_state_source": "calibration",
            }
        )

        output_rows: List[Dict[str, Any]] = []
        candidate_rows: List[Dict[str, Any]] = []
        previous_kill_state: Optional[int] = args.initial_kill_state
        decoded_sample_count = 0
        gameplay_segment_id = 0
        previous_window_paused = False

        for second in range(processed_seconds):
            timer_candidates: List[Dict[str, Any]] = []
            kill_candidates: List[Dict[str, Any]] = []
            samples: List[Dict[str, Any]] = []
            evidence_frame = None
            evidence_frame_offset: Optional[float] = None
            fallback_timer_crop = None
            fallback_timer_offset: Optional[float] = None
            fallback_timer_frame_index: Optional[int] = None

            for offset in args.sample_offsets:
                frame_index = min(frame_count - 1, int((second + offset) * fps))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                frame_decoded, frame = capture.read()
                sample_payload: Dict[str, Any] = {
                    "offset_seconds": offset,
                    "frame_index": frame_index,
                    "decoded": bool(frame_decoded),
                    "timer_candidates": [],
                    "kill_candidates": [],
                }
                if not frame_decoded:
                    samples.append(sample_payload)
                    continue
                decoded_sample_count += 1
                pause = gameplay_pause_evidence(frame)
                hud_score = float(gameplay_hud_score(frame))
                before_gameplay = second < gameplay_start_second
                sample_payload["gameplay_hud_score"] = round(hud_score, 6)
                sample_payload["screen_state"] = pause["phase"]
                sample_payload["excluded_from_gameplay"] = bool(
                    pause["blocked"] or before_gameplay
                )
                sample_payload["exclusion_reason"] = (
                    "pregame_or_loading_before_gameplay_hud"
                    if before_gameplay
                    else (pause["reason"] if pause["blocked"] else None)
                )
                if evidence_frame is None or abs(offset - 0.5) < abs(
                    (evidence_frame_offset or 0.0) - 0.5
                ):
                    evidence_frame = frame.copy()
                    evidence_frame_offset = offset
                if pause["blocked"] or before_gameplay:
                    samples.append(sample_payload)
                    continue
                raw_timer_crop = crop_roi(frame, timer_roi)
                if fallback_timer_crop is None or abs(offset - 0.5) < abs(
                    (fallback_timer_offset or 0.0) - 0.5
                ):
                    fallback_timer_crop = raw_timer_crop.copy()
                    fallback_timer_offset = offset
                    fallback_timer_frame_index = frame_index
                timer_crop = preprocess_for_ocr(raw_timer_crop, cv2)
                raw_kill_crop = crop_roi(frame, kill_roi)
                kill_crop = preprocess_for_ocr(raw_kill_crop, cv2)
                kill_crop_raw_upscaled = cv2.resize(
                    raw_kill_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                )
                timer_result = ocr_results_to_candidates(
                    reader.readtext(
                        timer_crop,
                        allowlist="0123456789:",
                        detail=1,
                    )
                )
                kill_result = ocr_results_to_candidates(
                    reader.readtext(
                        kill_crop,
                        allowlist="0123456789",
                        detail=1,
                    )
                )
                raw_kill_result = ocr_results_to_candidates(
                    reader.readtext(
                        kill_crop_raw_upscaled,
                        allowlist="0123456789",
                        detail=1,
                    )
                )
                for candidate in kill_result:
                    candidate["preprocessing_variant"] = "fixed_threshold_200_inverted_upscaled_2x"
                for candidate in raw_kill_result:
                    candidate["preprocessing_variant"] = "raw_color_upscaled_2x"
                kill_result.extend(raw_kill_result)
                annotate_timer_candidates(
                    timer_result,
                    preprocessing_variant=TIMER_PRIMARY_PREPROCESSING_VARIANT,
                    allow_unique_embedded=False,
                    fallback=False,
                )
                for candidate in timer_result:
                    candidate["offset_seconds"] = offset
                    candidate["frame_index"] = frame_index
                for candidate in kill_result:
                    candidate["offset_seconds"] = offset
                    candidate["frame_index"] = frame_index
                timer_candidates.extend(timer_result)
                kill_candidates.extend(kill_result)
                sample_payload["timer_candidates"] = timer_result
                sample_payload["kill_candidates"] = kill_result
                samples.append(sample_payload)

            excluded_window = any(sample.get("excluded_from_gameplay") for sample in samples)
            if excluded_window and not previous_window_paused:
                gameplay_segment_id += 1
            previous_window_paused = excluded_window
            timer_selection = select_timer(
                timer_candidates, min_confidence=args.min_ocr_confidence
            )
            primary_timer_candidate_count = len(timer_candidates)
            timer_fallback = {
                "attempted": False,
                "reason": None,
                "source_offset_seconds": fallback_timer_offset,
                "source_frame_index": fallback_timer_frame_index,
                "variants": [],
            }
            fallback_timer_candidates: List[Dict[str, Any]] = []
            if not timer_selection.observed and not excluded_window:
                if (
                    fallback_timer_crop is not None
                    and fallback_timer_offset is not None
                    and fallback_timer_frame_index is not None
                ):
                    timer_fallback["attempted"] = True
                    timer_fallback["reason"] = "no_valid_primary_timer_candidate"
                    (
                        fallback_timer_candidates,
                        timer_fallback["variants"],
                    ) = run_timer_fallback_ocr(
                        timer_crop=fallback_timer_crop,
                        reader=reader,
                        cv2=cv2,
                        offset_seconds=fallback_timer_offset,
                        frame_index=fallback_timer_frame_index,
                    )
                    timer_candidates.extend(fallback_timer_candidates)
                    timer_selection = select_timer(
                        fallback_timer_candidates,
                        min_confidence=args.min_ocr_confidence,
                    )
                else:
                    timer_fallback["reason"] = "no_decoded_timer_crop"
            kill_selection = select_kill(
                kill_candidates,
                previous_state=previous_kill_state,
                min_confidence=args.min_ocr_confidence,
            )
            pregame_window = second < gameplay_start_second
            if pregame_window:
                timer_selection, kill_selection = exclude_non_gameplay_hud_window()
                timer_fallback["reason"] = "pregame_or_loading_before_gameplay_hud"
            elif excluded_window:
                # The CSV window can contain both live and paused samples.
                # Exclude that window conservatively; its raw candidates remain
                # in the evidence ledger with their exact source frame indices.
                timer_selection, kill_selection = exclude_paused_hud_window(samples, timer_selection, kill_selection)
                timer_fallback["reason"] = "window_overlaps_level_up_pause"
            elif (
                second == 0
                and not kill_selection.observed
                and kill_selection.state_value == args.initial_kill_state
            ):
                kill_selection = KillSelection(
                    observed=False,
                    raw_text=None,
                    observed_value=None,
                    confidence=None,
                    state_value=args.initial_kill_state,
                    state_source="assumed_initial_state",
                    status="assumed_initial_state",
                    valid_candidate_count=0,
                )
            previous_kill_state = advance_kill_state(
                previous_kill_state,
                kill_selection,
                excluded_window=excluded_window,
            )

            periodic_evidence = (
                args.evidence_every > 0 and second % args.evidence_every == 0
            )
            review_evidence = (
                not args.no_review_frames
                and (
                    not timer_selection.observed
                    or timer_fallback["attempted"]
                    or kill_selection.state_source != "observed"
                )
            )
            sample_frame_path = ""
            if (
                evidence_frame is not None
                and len(evidence_rows) < args.max_evidence_frames
                and (periodic_evidence or review_evidence)
            ):
                reasons = []
                if periodic_evidence:
                    reasons.append("periodic")
                if review_evidence:
                    reasons.append("review")
                sample_frame_path = "evidence/hud_second_%06d.jpg" % second
                annotated = _draw_evidence_frame(
                    evidence_frame,
                    timer_roi,
                    kill_roi,
                    timer_selection.normalized_text or "MISSING",
                    (
                        str(kill_selection.state_value)
                        if kill_selection.state_value is not None
                        else "MISSING"
                    ),
                    cv2,
                )
                evidence_hash = _save_jpeg_atomic(
                    output_dir / sample_frame_path, annotated, cv2
                )
                evidence_rows.append(
                    {
                        "video_second": second,
                        "relative_path": sample_frame_path,
                        "reason": "+".join(reasons),
                        "sha256": evidence_hash,
                        "timer_status": timer_selection.status,
                        "kill_state_source": kill_selection.state_source,
                    }
                )

            review_flag = (
                not timer_selection.observed
                or timer_fallback["attempted"]
                or kill_selection.state_source != "observed"
                or sum(bool(sample["decoded"]) for sample in samples)
                != len(args.sample_offsets)
            )
            row = {
                "video_name": video_path.stem,
                "video_file": video_path.name,
                "video_duration_minutes": round(duration_seconds / 60.0, 6),
                "processed_minutes": round(processed_seconds / 60.0, 6),
                "Video Second": second,
                "Time Stamp": timer_selection.normalized_text or "",
                "Kill Counter Quantity": (
                    kill_selection.state_value
                    if kill_selection.state_value is not None
                    else ""
                ),
                "review_flag": review_flag,
                "timer_observed": timer_selection.observed,
                "timer_raw_text": timer_selection.raw_text or "",
                "timer_normalized_text": timer_selection.normalized_text or "",
                "timer_confidence": (
                    round(timer_selection.confidence, 8)
                    if timer_selection.confidence is not None
                    else ""
                ),
                "timer_status": timer_selection.status,
                "timer_candidate_count": len(timer_candidates),
                "timer_primary_candidate_count": primary_timer_candidate_count,
                "timer_fallback_candidate_count": len(fallback_timer_candidates),
                "timer_fallback_attempted": timer_fallback["attempted"],
                "timer_preprocessing_variant": (
                    timer_selection.preprocessing_variant or ""
                ),
                "timer_normalization_policy": (
                    timer_selection.normalization_policy or ""
                ),
                "kill_observed": kill_selection.observed,
                "kill_raw_text": kill_selection.raw_text or "",
                "kill_observed_value": (
                    kill_selection.observed_value
                    if kill_selection.observed_value is not None
                    else ""
                ),
                "kill_confidence": (
                    round(kill_selection.confidence, 8)
                    if kill_selection.confidence is not None
                    else ""
                ),
                "kill_state_value": (
                    kill_selection.state_value
                    if kill_selection.state_value is not None
                    else ""
                ),
                "kill_state_source": kill_selection.state_source,
                "kill_status": kill_selection.status,
                "kill_candidate_count": len(kill_candidates),
                "decoded_sample_count": sum(
                    bool(sample["decoded"]) for sample in samples
                ),
                "gameplay_sample_count": sum(bool(sample["decoded"]) and not sample.get("excluded_from_gameplay", False) for sample in samples),
                "excluded_sample_count": sum(bool(sample.get("excluded_from_gameplay")) for sample in samples),
                "excluded_from_gameplay": excluded_window,
                "exclusion_reason": (
                    "pregame_or_loading_before_gameplay_hud"
                    if pregame_window
                    else ("window_overlaps_level_up_pause" if excluded_window else "")
                ),
                "gameplay_segment_id": gameplay_segment_id,
                "sample_frame_path": sample_frame_path,
            }
            output_rows.append(row)
            candidate_rows.append(
                {
                    "record_type": "hud_ocr_second",
                    "video_second": second,
                    "samples": samples,
                    "timer_fallback": timer_fallback,
                    "selection": {
                        "timer": {
                            "observed": timer_selection.observed,
                            "raw_text": timer_selection.raw_text,
                            "normalized_text": timer_selection.normalized_text,
                            "confidence": timer_selection.confidence,
                            "status": timer_selection.status,
                            "valid_candidate_count": timer_selection.valid_candidate_count,
                            "preprocessing_variant": timer_selection.preprocessing_variant,
                            "normalization_policy": timer_selection.normalization_policy,
                        },
                        "kill": {
                            "observed": kill_selection.observed,
                            "raw_text": kill_selection.raw_text,
                            "observed_value": kill_selection.observed_value,
                            "confidence": kill_selection.confidence,
                            "state_value": kill_selection.state_value,
                            "state_source": kill_selection.state_source,
                            "status": kill_selection.status,
                            "valid_candidate_count": kill_selection.valid_candidate_count,
                        },
                    },
                    "evidence_frame": sample_frame_path or None,
                }
            )
            if second % 30 == 0 or second + 1 == processed_seconds:
                print(
                    "Processed %d/%d video seconds" % (second + 1, processed_seconds),
                    file=sys.stderr,
                )
    finally:
        capture.release()

    changed_kill_seconds = reconcile_kill_output_rows(
        output_rows,
        candidate_rows,
        initial_state=args.initial_kill_state,
        min_confidence=args.min_ocr_confidence,
    )
    refresh_reconciled_kill_evidence(
        video_path=video_path,
        output_dir=output_dir,
        output_rows=output_rows,
        candidate_rows=candidate_rows,
        evidence_rows=evidence_rows,
        changed_seconds=changed_kill_seconds,
        timer_roi=timer_roi,
        kill_roi=kill_roi,
        max_evidence_frames=args.max_evidence_frames,
        cv2=cv2,
    )

    observations_path = output_dir / "hud_observations.csv"
    candidates_path = output_dir / "ocr_candidates.jsonl"
    evidence_index_path = output_dir / "evidence_index.csv"
    qc_path = output_dir / "qc.json"
    manifest_path = output_dir / "manifest.json"
    observation_count = write_csv_atomic(observations_path, output_rows, CSV_FIELDS)
    candidate_count = write_jsonl_atomic(candidates_path, candidate_rows)
    evidence_count = write_csv_atomic(
        evidence_index_path, evidence_rows, EVIDENCE_FIELDS
    )
    qc = _build_qc(
        output_rows,
        candidate_rows,
        processed_seconds,
        decoded_sample_count,
        args.min_timer_observed_rate,
        args.min_kill_observed_rate,
    )
    write_json_atomic(qc_path, qc)
    completed_at = utc_now()
    runtime_seconds = round(time.perf_counter() - started_perf, 6)
    manifest = {
        "prepared_by": PREPARED_BY,
        "artifact_type": "hud_detector_run_manifest",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "failed" if qc["status"] == "failed" else "completed",
        "worker": {
            "name": WORKER_NAME,
            "version": WORKER_VERSION,
            "code_sha256": sha256_file(SCRIPT_PATH),
            "gameplay_state_code_sha256": sha256_file(COMPONENT_DIR / "gameplay_state.py"),
            "gameplay_hud_code_sha256": sha256_file(SCRIPT_PATH.with_name("gems.py")),
            "level_up_geometry_code_sha256": sha256_file(SCRIPT_PATH.with_name("inventory.py")),
        },
        "input_video": {
            "file_name": video_path.name,
            "size_bytes": video_path.stat().st_size,
            "sha256": video_sha256,
            "sha256_status": video_hash_status,
            "fps": round(fps, 8),
            "frame_count": frame_count,
            "width": frame_width,
            "height": frame_height,
            "duration_seconds": round(duration_seconds, 6),
        },
        "configuration": {
            "run_mode": "full" if args.full else "max_seconds",
            "requested_max_seconds": args.max_seconds,
            "processed_seconds": processed_seconds,
            "sample_offsets_seconds": list(args.sample_offsets),
            "min_ocr_confidence": args.min_ocr_confidence,
            "min_timer_observed_rate": args.min_timer_observed_rate,
            "min_kill_observed_rate": args.min_kill_observed_rate,
            "initial_kill_state": args.initial_kill_state,
            "initial_kill_state_semantics": (
                "explicit_assumption_at_first_sustained_gameplay_hud"
            ),
            "gameplay_start_second": gameplay_start_second,
            "timer_forward_fill_enabled": False,
            "timer_fallback": {
                "enabled": True,
                "trigger": "no_valid_primary_timer_candidate",
                "source_sample_policy": "decoded_crop_closest_to_offset_0.5",
                "preprocessing_variants": list(
                    TIMER_FALLBACK_PREPROCESSING_VARIANTS
                ),
                "embedded_normalization_policy": (
                    "accept_exactly_one_valid_embedded_dd_colon_dd"
                ),
                "temporal_imputation_enabled": False,
            },
            "kill_state_carry_enabled": True,
            "kill_temporal_overshoot_policy": {
                "enabled": True,
                "lookahead_seconds": KILL_OVERSHOOT_LOOKAHEAD_SECONDS,
                "minimum_contradicting_seconds": (
                    KILL_OVERSHOOT_MIN_SUPPORTING_SECONDS
                ),
                "minimum_valid_candidates_per_contradicting_second": (
                    KILL_OVERSHOOT_MIN_SAMPLES_PER_SECOND
                ),
                "large_jump_min_increase": KILL_LARGE_JUMP_MIN_INCREASE,
                "large_jump_min_fraction": KILL_LARGE_JUMP_MIN_FRACTION,
                "large_jump_lookahead_seconds": (
                    KILL_LARGE_JUMP_LOOKAHEAD_SECONDS
                ),
                "large_jump_min_future_confidence": (
                    KILL_LARGE_JUMP_MIN_FUTURE_CONFIDENCE
                ),
                "interpolation_enabled": False,
            },
            "kill_gap_recovery_policy": {
                "allowance_per_eligible_second": (
                    KILL_GAP_ALLOWANCE_PER_ELIGIBLE_SECOND
                ),
                "allowance_cap": KILL_GAP_ALLOWANCE_CAP,
                "confirmation_seconds": KILL_GAP_JUMP_CONFIRMATION_SECONDS,
                "minimum_future_confidence": (
                    KILL_GAP_JUMP_MIN_FUTURE_CONFIDENCE
                ),
                "required_future_seconds": 2,
            },
            "level_up_pause_policy": "exclude_overlapping_windows_and_reset_kill_carry",
            "evidence_every_seconds": args.evidence_every,
            "review_frames_enabled": not args.no_review_frames,
            "max_evidence_frames": args.max_evidence_frames,
            "gpu_requested": args.gpu,
            "easyocr_download_enabled": False,
        },
        "templates": _safe_template_manifest(),
        "calibration": calibration,
        "dependencies": {
            **resolved_dependency_versions,
            "version_fingerprint_sha256": dependency_version_fingerprint,
        },
        "easyocr_model_weights": easyocr_model_weights,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "actual_command_runtime_seconds": runtime_seconds,
        "outputs": {
            "hud_observations": {
                "path": "hud_observations.csv",
                "sha256": sha256_file(observations_path),
                "row_count": observation_count,
            },
            "ocr_candidates": {
                "path": "ocr_candidates.jsonl",
                "sha256": sha256_file(candidates_path),
                "row_count": candidate_count,
            },
            "evidence_index": {
                "path": "evidence_index.csv",
                "sha256": sha256_file(evidence_index_path),
                "row_count": evidence_count,
            },
            "qc": {
                "path": "qc.json",
                "sha256": sha256_file(qc_path),
                "status": qc["status"],
            },
        },
        "canonical_database_write_performed": False,
    }
    write_json_atomic(manifest_path, manifest)
    if qc["status"] != "failed":
        write_bytes_atomic(output_dir / "_SUCCESS", (WORKER_VERSION + "\n").encode("utf-8"))
    return {
        "status": manifest["status"],
        "qc_status": qc["status"],
        "processed_seconds": processed_seconds,
        "output_dir": str(output_dir),
        "runtime_seconds": runtime_seconds,
    }


def _write_failure_manifest_if_safe(
    args: argparse.Namespace,
    message: str,
    exit_code: int,
    started_at: str,
    started_perf: float,
) -> None:
    try:
        if "already contains managed artifacts" in message:
            return
        output_dir = args.output_dir.expanduser().resolve()
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "prepared_by": PREPARED_BY,
            "artifact_type": "hud_detector_run_manifest",
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "failed",
            "worker": {"name": WORKER_NAME, "version": WORKER_VERSION},
            "input_video": {"file_name": args.video.name},
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "actual_command_runtime_seconds": round(
                time.perf_counter() - started_perf, 6
            ),
            "error": {"exit_code": exit_code, "message": message},
            "canonical_database_write_performed": False,
        }
        write_json_atomic(manifest_path, payload)
    except Exception:
        return


def main(argv: Optional[Sequence[str]] = None) -> int:
    started_at = utc_now()
    started_perf = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_worker(args)
    except WorkerError as error:
        _write_failure_manifest_if_safe(
            args, str(error), error.exit_code, started_at, started_perf
        )
        print("HUD worker failed: %s" % error, file=sys.stderr)
        return error.exit_code
    except Exception as error:
        _write_failure_manifest_if_safe(
            args,
            "%s: %s" % (type(error).__name__, error),
            EXIT_PROCESSING,
            started_at,
            started_perf,
        )
        print("HUD worker failed: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return EXIT_PROCESSING
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_QC_FAILED if result["status"] == "failed" else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
