"""Read-only source audit and sparse visual evidence. Prepared by Tahereh Fahi.

This diagnostic workflow never calls analytical release stages. Sparse frames
cannot validate event counts, durations, or absence; they support layout review.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json
from .validation_cache import run_stage, source_hashes
from .video_registry import build_registry, number, tool_version


def sample_times(duration: float, requested: list[float] | None) -> list[float]:
    if requested:
        if any(not math.isfinite(t) or t < 0 or t >= duration for t in requested):
            raise ValueError("Sample times must be finite, nonnegative and before the recording end")
        return sorted(set(requested))
    return sorted({t for t in (0.0, 1.0, 5.0, 12.0, duration / 2,
                              max(0.0, duration - 5), max(0.0, duration - 1)) if t < duration})


def select_videos(records: list[dict[str, Any]], selectors: list[str] | None) -> list[dict[str, Any]]:
    if not selectors:
        return records
    selected = {}
    for selector in selectors:
        matches = [r for r in records if r["source"] == selector or Path(r["source"]).name == selector]
        if len(matches) != 1:
            raise ValueError(f"Video selection is missing or ambiguous: {selector}")
        selected[matches[0]["source"]] = matches[0]
    return list(selected.values())


def extract_frame(source: Path, second: float, stream_start: float) -> tuple[bytes, dict[str, Any]]:
    # Accurate input seeking discards earlier frames. copyts preserves the
    # selected frame's actual PTS; no frame-index / average-FPS conversion.
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-copyts",
        "-seek_timestamp", "1", "-ss", f"{second + stream_start:.9f}",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "showinfo", "-frames:v", "1", "-fps_mode", "passthrough",
        "-c:v", "png", "-f", "image2pipe", "pipe:1",
    ], check=True, capture_output=True)
    log = result.stderr.decode("utf-8", errors="replace")
    frame = re.search(r"\bn:\s*0\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+\d.eE]+)", log)
    time_base = re.search(r"config in time_base:\s*(\d+)/(\d+)", log)
    if not frame or not time_base or not result.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Decoded frame or its original presentation timestamp is unavailable")
    pts = int(frame.group(1))
    numerator, denominator = map(int, time_base.groups())
    if denominator <= 0:
        raise ValueError("Invalid frame time base")
    pts_seconds = pts * numerator / denominator
    actual = pts_seconds - stream_start
    if actual < second - 0.001:
        raise ValueError("Decoded frame precedes the requested source-relative timestamp")
    return result.stdout, {"requested_media_seconds": second, "actual_media_seconds": actual,
                           "pts": pts, "time_base": f"{numerator}/{denominator}",
                           "pts_seconds": pts_seconds, "stream_start_seconds": stream_start,
                           "seek_offset_seconds": actual - second,
                           "frame_number": None, "game_time_seconds": None}


def save_evidence(directory: Path, source: Path, record: dict[str, Any], times: list[float]) -> None:
    start = number(record["metadata"]["stream"].get("start_time"))
    if start is None:
        raise ValueError("Stream start timestamp is unavailable; do not assume zero")
    before = source.stat()
    if (before.st_size, before.st_mtime_ns) != (record["size_bytes"], record["audited_mtime_ns"]):
        raise ValueError("Recording changed after its identity audit")
    samples = []
    for index, second in enumerate(times):
        image, timing = extract_frame(source, second, start)
        filename = f"sample_{index:03d}.png"
        (directory / filename).write_bytes(image)
        samples.append({**timing, "image": filename, "screen_state": "unreviewed",
                        "visible_level": None, "level_reading_status": "not_reviewed"})
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Recording changed while extracting evidence")
    write_json(directory / "samples.json", {
        "prepared_by": "Tahereh Fahi", "source": record["source"], "sha256": record["sha256"],
        "validation_status": "requires_visual_review", "publication_ready": False,
        "scope": "Sparse full-resolution screenshots only; no OCR or event detection performed.",
        "samples": samples,
    })


def validate_foundations(*, workspace: Path, config_dir: Path, videos_dir: Path,
                         output: Path, stage: str = "registry", videos: list[str] | None = None,
                         at_seconds: list[float] | None = None, resume: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    workspace, config_dir, videos_dir, output = (p.resolve() for p in (workspace, config_dir, videos_dir, output))
    if stage not in {"registry", "evidence"}:
        raise ValueError("Unknown foundation diagnostic stage")
    if stage == "registry" and (videos or at_seconds):
        raise ValueError("--video and --at-second apply only to --stage evidence; registry always audits all sources")
    if not config_dir.is_dir() or not videos_dir.is_dir():
        raise ValueError("Configuration and video directories must exist")
    if not config_dir.is_relative_to(workspace) or not videos_dir.is_relative_to(workspace):
        raise ValueError("Configuration and video directories must be within the workspace")
    if output == workspace or output.is_relative_to(config_dir) or output.is_relative_to(videos_dir):
        raise ValueError("Use a separate diagnostics output folder, not source/configuration directories")
    # A fresh source hash audit is required even when cached screenshots exist.
    # Never use file size/mtime alone as proof of unchanged video content.
    probe_version = tool_version("ffprobe")
    registry = build_registry(workspace, config_dir, videos_dir)
    if not registry["videos"] or not registry["configurations"]:
        raise ValueError("No source videos or video configurations were found")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit = output / "audits" / stamp
    audit.mkdir(parents=True, exist_ok=False)
    registry["ffprobe_version"] = probe_version
    registry["implementation"] = source_hashes(["video_registry.py", "foundation_validation.py"])
    write_json(audit / "video_registry.json", registry)
    summary: dict[str, Any] = {
        "prepared_by": "Tahereh Fahi", "stage": stage, "execution_status": "complete",
        "validation_status": "requires_review", "publication_ready": False,
        "registry": "video_registry.json", "registry_summary": registry["summary"],
        "evidence_stages": [], "failures": [],
        "workflow_elapsed_seconds": "not recorded",
    }
    try:
        if stage == "evidence":
            version = tool_version("ffmpeg")
            for record in select_videos(registry["videos"], videos):
                if record["inspection_status"] != "complete":
                    summary["failures"].append({"source": record["source"], "reason": "source_inspection_failed"})
                    continue
                times = sample_times(record["metadata"]["duration_seconds"], at_seconds)
                inputs = {"source": record["source"], "sha256": record["sha256"],
                          "metadata": record["metadata"], "sample_seconds": times,
                          "ffmpeg_version": version, "ffprobe_version": probe_version,
                          "implementation": source_hashes(["foundation_validation.py", "video_registry.py"]),
                          "scope": "raw_screenshots_no_inference_no_configuration_dependency"}
                print(json.dumps({"stage": "evidence", "source": record["source"]}), flush=True)
                try:
                    current = (workspace / record["source"]).stat()
                    if (current.st_size, current.st_mtime_ns) != (record["size_bytes"], record["audited_mtime_ns"]):
                        raise ValueError("Source changed after identity audit")
                    receipt = run_stage(output / "evidence", "screens", inputs,
                                        lambda directory: save_evidence(directory, workspace / record["source"], record, times),
                                        resume=resume)
                    summary["evidence_stages"].append({"source": record["source"],
                        "directory_from_output_root": "evidence/" + receipt["directory"],
                        "identity": receipt["identity"], "reused": receipt["reused"]})
                except (OSError, ValueError, subprocess.SubprocessError):
                    summary["failures"].append({"source": record["source"], "reason": "evidence_extraction_failed"})
    except (OSError, ValueError, subprocess.SubprocessError):
        summary["failures"].append({"reason": "evidence_setup_or_selection_failed"})
    if summary["failures"] or registry["summary"]["inspection_failures"]:
        summary["execution_status"] = "completed_with_errors"
    summary["command_execution_seconds"] = time.monotonic() - started
    write_json(audit / "summary.json", summary)
    print(f"FOUNDATION AUDIT COMPLETE: {audit / 'summary.json'}", flush=True)
    print("Validation: requires review. Publication: disabled.", flush=True)
    return summary
