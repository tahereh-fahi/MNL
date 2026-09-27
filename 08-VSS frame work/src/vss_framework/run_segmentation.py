"""Split a source video into gameplay runs using direct HUD clock evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .io import write_json


def _clock_seconds(value: str) -> int | None:
    value = value.strip()
    if not value or ":" not in value:
        return None
    minutes, seconds = value.split(":", 1)
    try:
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return None


def segment_runs(hud_csv: Path, *, video_asset_id: str, output_path: Path) -> dict[str, Any]:
    """Detect restarts without treating clock resets as OCR errors.

    A run begins at an observed 00:00/00:01 after either the start of the file or
    at least five seconds without an observed game clock. It ends at the final
    observed clock sample before the next run (or the end of available evidence).
    """
    with hud_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed: list[tuple[int, int]] = []
    for row in rows:
        if row.get("timer_observed") != "True":
            continue
        media_second = int(float(row["Video Second"]))
        clock_second = _clock_seconds(row.get("Time Stamp", ""))
        if clock_second is not None:
            observed.append((media_second, clock_second))
    starts: list[int] = []
    previous_media: int | None = None
    previous_clock: int | None = None
    for media_second, clock_second in observed:
        reset = previous_clock is not None and clock_second + 10 < previous_clock
        after_gap = previous_media is None or media_second - previous_media >= 5
        if clock_second <= 1 and (not starts or reset or after_gap):
            starts.append(media_second)
        previous_media, previous_clock = media_second, clock_second
    segments = []
    for index, start in enumerate(starts):
        next_start = starts[index + 1] if index + 1 < len(starts) else None
        candidates = [media for media, _ in observed if media >= start and (next_start is None or media < next_start)]
        end = max(candidates) + 1 if candidates else start
        segments.append({
            "segmentId": f"{video_asset_id}_run_{index + 1:02d}",
            "runIndex": index + 1,
            "mediaStartMs": start * 1000,
            "mediaEndMs": end * 1000,
            "durationMs": (end - start) * 1000,
            "sessionTimeOriginMs": start * 1000,
            "evidence": "direct_observed_game_clock_reset",
        })
    payload = {
        "schemaVersion": "vss-run-segments-v1",
        "preparedBy": "Tahereh Fahi",
        "videoAssetId": video_asset_id,
        "segments": segments,
        "policies": {"humanCodedGroundTruthUsed": False, "missingClockIsNotGameplay": True},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    return payload

