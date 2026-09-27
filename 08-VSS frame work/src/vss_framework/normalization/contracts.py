"""Canonical observation contract shared by the logical batch workers."""

from __future__ import annotations

import hashlib
from typing import Any


CONTRACT_VERSION = "mnl-observation-v1"
SCHEMA_VERSION = 1

OBSERVATION_FIELDS = (
    "observation_id",
    "session_id",
    "video_asset_id",
    "source_type",
    "record_kind",
    "temporal_precision",
    "observable_code",
    "processing_run_id",
    "annotation_set_id",
    "source_record_key",
    "media_start_ms",
    "media_end_ms",
    "session_start_ms",
    "session_end_ms",
    "game_time_ms",
    "frame_number",
    "numeric_value",
    "text_value",
    "item_code",
    "quantity",
    "confidence",
    "evidence_json",
    "attributes_json",
    "supersedes_observation_id",
    "schema_version",
)


def deterministic_id(prefix: str, *parts: str, length: int = 24) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def processing_run_id(service_slug: str, idempotency_key: str) -> str:
    return f"run_{service_slug}_{idempotency_key[:16]}"


def observation(
    *,
    processing_run_id_value: str,
    source_record_key: str,
    session_id: str,
    video_asset_id: str,
    record_kind: str,
    temporal_precision: str,
    observable_code: str,
    media_start_ms: int,
    media_end_ms: int,
    session_start_ms: int,
    session_end_ms: int,
    game_time_ms: int | None = None,
    frame_number: int | None = None,
    numeric_value: int | float | None = None,
    text_value: str | None = None,
    item_code: str | None = None,
    quantity: int | None = None,
    confidence: float | None = None,
    evidence_json: dict[str, Any] | None = None,
    attributes_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observation_id = deterministic_id(
        "obs", processing_run_id_value, source_record_key
    )
    row = {
        "observation_id": observation_id,
        "session_id": session_id,
        "video_asset_id": video_asset_id,
        "source_type": "automated",
        "record_kind": record_kind,
        "temporal_precision": temporal_precision,
        "observable_code": observable_code,
        "processing_run_id": processing_run_id_value,
        "annotation_set_id": None,
        "source_record_key": source_record_key,
        "media_start_ms": media_start_ms,
        "media_end_ms": media_end_ms,
        "session_start_ms": session_start_ms,
        "session_end_ms": session_end_ms,
        "game_time_ms": game_time_ms,
        "frame_number": frame_number,
        "numeric_value": numeric_value,
        "text_value": text_value,
        "item_code": item_code,
        "quantity": quantity,
        "confidence": confidence,
        "evidence_json": evidence_json,
        "attributes_json": attributes_json,
        "supersedes_observation_id": None,
        "schema_version": SCHEMA_VERSION,
    }
    assert tuple(row) == OBSERVATION_FIELDS
    return row
