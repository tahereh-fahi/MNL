"""Build a five-second publication view from canonical events and telemetry."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ..models import CanonicalEvent


EVENT_COUNT_COLUMNS = (
    "level_up_selection_count",
    "inventory_change_count",
    "consumable_reward_count",
)


def _event_counts(events: list[CanonicalEvent], start_ms: int, end_ms: int) -> Counter:
    counts: Counter = Counter()
    for event in events:
        anchor = event.anchor_time_ms if event.anchor_time_ms is not None else event.time_lower_ms
        if not (start_ms <= anchor < end_ms):
            continue
        if event.event_type == "level_up_selection":
            counts["level_up_selection_count"] += 1
        if event.event_family == "inventory":
            counts["inventory_change_count"] += 1
        if event.event_type in {"big_coin_bag", "floor_chicken", "consumable_reward"}:
            counts["consumable_reward_count"] += 1
    return counts


def write_five_second_windows(
    source_path: Path,
    output_path: Path,
    events: list[CanonicalEvent],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {source_path}")
        fieldnames = [*reader.fieldnames, *EVENT_COUNT_COLUMNS, "event_source_status"]
        rows = list(reader)

    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            start_ms = int(row["interval_start_ms"])
            end_ms = int(row["interval_end_ms"])
            counts = _event_counts(events, start_ms, end_ms)
            for column in EVENT_COUNT_COLUMNS:
                row[column] = counts[column]
            row["event_source_status"] = "upstream_release_plus_framework_adapters"
            writer.writerow(row)
    return len(rows)

