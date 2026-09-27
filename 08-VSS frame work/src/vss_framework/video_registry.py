"""Audit real source recordings and configuration identities without inference.

Prepared by Tahereh Fahi. Filenames and historical durations never establish
identity. This audit does not infer run boundaries, character, or game settings.
"""
from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def tool_version(executable: str) -> str:
    return subprocess.run([executable, "-version"], check=True, capture_output=True,
                          text=True).stdout.splitlines()[0]


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=index,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames:"
        "format=duration,start_time,size", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    probe = json.loads(result.stdout)
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one selected video stream")
    stream = streams[0]
    duration = number(stream.get("duration")) or number(probe.get("format", {}).get("duration"))
    if duration is None or duration <= 0:
        raise ValueError("Video duration is unavailable")
    return {"stream": stream, "container": probe.get("format", {}),
            "duration_seconds": duration,
            "frame_timing_status": "presentation_timestamps_not_yet_audited",
            "fps_note": "Nominal/average FPS do not establish constant frame timing."}


def configurations(config_dir: Path, workspace: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(config_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = data.get("dataset", {})
        video = dataset.get("video", {})
        if not video.get("path") or not video.get("sha256"):
            continue
        source = (workspace / video["path"]).resolve()
        if not source.is_relative_to(workspace):
            raise ValueError(f"Configured source must be inside workspace: {path.name}")
        records.append({"config": path.relative_to(workspace).as_posix(),
                        "config_sha256": sha256_file(path),
                        "configured_source": source.relative_to(workspace).as_posix(),
                        "expected_sha256": video["sha256"],
                        "video_asset_id": dataset.get("video_asset_id"),
                        "session_id": dataset.get("session_id"),
                        "segment_id": dataset.get("segment_id"),
                        "configured_duration_ms": dataset.get("duration_ms"),
                        "configured_initial_level": data.get("detectors", {}).get("gem_xp", {}).get("initial_level"),
                        "declared_mapping": dataset.get("media_to_session_mapping"),
                        "declared_multi_run": bool(data.get("policies", {}).get("multi_run_video")),
                        "run_boundary_status": "requires_verification"})
    return records


def resolve_configurations(configs: list[dict[str, Any]], videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = []
    for config in configs:
        matches = [v for v in videos if v.get("sha256") == config["expected_sha256"]]
        exact = next((v for v in matches if v["source"] == config["configured_source"]), None)
        named = next((v for v in videos if v["source"] == config["configured_source"]), None)
        chosen = exact or (matches[0] if len(matches) == 1 else None)
        status = ("verified" if exact else "verified_renamed_source" if chosen else
                  "ambiguous_duplicate_sources" if matches else
                  "configured_source_inspection_failed" if named and named.get("inspection_status") == "failed" else
                  "configured_source_hash_mismatch" if named else "source_missing")
        declared_duration = number(config.get("configured_duration_ms"))
        measured_duration = chosen.get("metadata", {}).get("duration_seconds") if chosen else None
        resolved.append({**config, "identity_status": status,
                         "resolved_source": chosen["source"] if chosen else None,
                         "measured_minus_configured_duration_ms":
                             measured_duration * 1000 - declared_duration
                             if measured_duration is not None and declared_duration is not None else None,
                         "matching_sources": [v["source"] for v in matches]})
    return resolved


def build_registry(workspace: Path, config_dir: Path, videos_dir: Path) -> dict[str, Any]:
    configs = configurations(config_dir, workspace)
    paths = {p.resolve() for p in videos_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"}
    paths.update((workspace / c["configured_source"]).resolve() for c in configs
                 if (workspace / c["configured_source"]).is_file())
    records = []
    for index, path in enumerate(sorted(paths), 1):
        if not path.is_relative_to(workspace):
            raise ValueError("Source recording escapes workspace")
        source = path.relative_to(workspace).as_posix()
        print(json.dumps({"stage": "registry", "video": source, "index": index, "total": len(paths)}), flush=True)
        before = path.stat()
        record: dict[str, Any] = {"source": source, "size_bytes": before.st_size,
                                  "audited_mtime_ns": before.st_mtime_ns}
        try:
            record["sha256"] = sha256_file(path)
            record["metadata"] = probe_video(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source changed during identity audit")
            record["inspection_status"] = "complete"
        except (OSError, ValueError, subprocess.SubprocessError):
            record["inspection_status"] = "failed"
            record["error"] = "Source could not be safely hashed/probed; inspect this file."
            record.pop("sha256", None)
        records.append(record)
    resolved = resolve_configurations(configs, records)
    matched = {p for c in resolved for p in c["matching_sources"]}
    return {"schema_version": "foundation-registry-1", "prepared_by": "Tahereh Fahi",
            "execution_status": "complete", "validation_status": "requires_review",
            "publication_ready": False, "videos": records, "configurations": resolved,
            "unconfigured_sources": [v["source"] for v in records if v["source"] not in matched],
            "summary": {"recordings": len(records), "configurations": len(configs),
                        "inspection_failures": sum(v["inspection_status"] != "complete" for v in records),
                        "configuration_identity_counts": dict(Counter(c["identity_status"] for c in resolved))},
            "scope_note": "Identity and metadata audit only; game settings, boundaries and frame timing remain unvalidated."}
