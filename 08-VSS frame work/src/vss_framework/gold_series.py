"""Build cross-video cumulative Gold publication series from framework runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .counter_repair import corroborate_repeated_counter_points
from .dashboard_projection import build_gold_counter_trajectory
from .io import iter_jsonl, write_json


def read_coin_points(
    runs_root: Path, run_name: str, *, offset_ms: int = 0,
    start_ms: int | None = None, end_ms: int | None = None,
) -> list[dict[str, Any]]:
    source = runs_root / run_name / "telemetry_observations.jsonl"
    observed = []
    for row in iter_jsonl(source):
        if row.get("observable_code") != "coin_counter" or row.get("observed") is not True:
            continue
        value = row.get("numeric_value")
        if not isinstance(value, (int, float)):
            continue
        media_time_ms = int(row["time_lower_ms"]) + offset_ms
        if start_ms is not None and media_time_ms < start_ms:
            continue
        if end_ms is not None and media_time_ms >= end_ms:
            continue
        point = {
            "mediaTimeMs": media_time_ms,
            "frameNumber": row.get("frame_number"),
            "value": value,
            "confidence": (row.get("attributes") or {}).get("confidence"),
            "evidenceGrade": row.get("evidence_grade"),
            "runName": run_name,
        }
        if (row.get("attributes") or {}).get("rejection_reason") == "gold_fever_monotonic_confirmation":
            point["validationMode"] = "gold_fever_monotonic_confirmation"
        observed.append(point)
    # Outside Gold Fever, counters retain the standard exact-repeat rule.
    # Gold Fever refreshes the accurate upper-right total too rapidly for an
    # unchanged neighboring sample, so the scanner supplies a separate
    # two-sample monotonic confirmation.  Preserve only those already-confirmed
    # points; this never turns a one-frame OCR reading into published data.
    fever_confirmed = [
        point for point in observed
        if point.get("validationMode") == "gold_fever_monotonic_confirmation"
    ]
    standard = [
        point for point in observed
        if point.get("validationMode") != "gold_fever_monotonic_confirmation"
    ]
    return sorted(
        corroborate_repeated_counter_points(standard) + fever_confirmed,
        key=lambda row: int(row["mediaTimeMs"]),
    )


def build_gold_series_payload(
    runs_root: Path,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    published = {}
    for dataset, run_specs in datasets.items():
        points = []
        for spec in run_specs:
            points.extend(
                read_coin_points(
                    runs_root,
                    str(spec["runName"]),
                    offset_ms=int(spec.get("offsetMs") or 0),
                    start_ms=(int(spec["startMs"]) if spec.get("startMs") is not None else None),
                    end_ms=(int(spec["endMs"]) if spec.get("endMs") is not None else None),
                )
            )
        points.sort(key=lambda row: int(row["mediaTimeMs"]))
        trajectory = build_gold_counter_trajectory(points)
        published[dataset] = {
            "sourceType": "Automated",
            "status": "available" if points else "unavailable",
            "timingPrecision": "frame",
            "valueMeaning": (
                "Visible cumulative upper-right in-game gold coin counter; standard OCR values "
                "need a repeated neighboring sample, while Gold Fever values need two nearby "
                "high-confidence monotonic confirmations."
            ),
            "missingDataRule": (
                "Missing or rejected OCR samples remain absent and are never converted to zero. "
                "Datasets produced by the superseded wide-ROI detector remain unavailable until reprocessed."
            ),
            "points": points,
            "counterBins": trajectory["counterBins"],
            "fiveSecondDeltaBins": trajectory["fiveSecondDeltaBins"],
            "centered30SecondCounterMean": trajectory["centered30SecondCounterMean"],
        }
    return {
        "schemaVersion": "1.0.0",
        "preparedBy": "Tahereh Fahi",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "datasets": published,
    }


def build_gold_series(*, runs_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = build_gold_series_payload(runs_root, config["datasets"])
    write_json(output_path, payload)
    return payload
