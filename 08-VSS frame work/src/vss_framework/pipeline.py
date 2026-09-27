"""Top-level orchestration for the first Video 4 framework slice."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import build_video4_events, resolve_and_verify_sources
from .catalog import load_event_catalog
from .exporters import write_five_second_windows
from .hashing import sha256_file
from .io import write_json, write_jsonl


FRAMEWORK_VERSION = "0.1.0"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id(config_path: Path, catalog_path: Path) -> str:
    identity = (
        FRAMEWORK_VERSION + sha256_file(config_path) + sha256_file(catalog_path)
    ).encode("utf-8")
    import hashlib

    return f"run_video4_{hashlib.sha256(identity).hexdigest()[:20]}"


def build_video4_framework_run(
    *,
    workspace_root: Path,
    config_path: Path,
    catalog_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    catalog = load_event_catalog(catalog_path)
    if config["policies"].get("use_human_coded_ground_truth"):
        raise ValueError("Video 4 bootstrap must not use Human-coded data as ground truth")

    run_id = _run_id(config_path, catalog_path)
    paths, verified_sources = resolve_and_verify_sources(workspace_root, config)
    events = build_video4_events(paths, config, run_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "canonical_events.jsonl"
    event_count = write_jsonl(events_path, (event.to_dict() for event in events))
    windows_path = output_dir / "derived_5s_windows.csv"
    window_count = write_five_second_windows(paths["trajectory_5s"], windows_path, events)

    event_types = Counter(event.event_type for event in events)
    publication = Counter(event.publication_status.value for event in events)
    manifest = {
        "artifact_type": "vss_framework_video4_run",
        "framework_version": FRAMEWORK_VERSION,
        "schema_version": "vss-event-v1",
        "catalog_version": catalog["catalog_version"],
        "processing_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "dataset": config["dataset"],
        "policies": config["policies"],
        "source_integrity": verified_sources,
        "counts": {
            "canonical_events": event_count,
            "derived_five_second_windows": window_count,
            "by_event_type": dict(sorted(event_types.items())),
            "by_publication_status": dict(sorted(publication.items())),
        },
        "outputs": {
            "canonical_events": {
                "path": events_path.name,
                "sha256": sha256_file(events_path),
                "row_count": event_count,
            },
            "derived_5s_windows": {
                "path": windows_path.name,
                "sha256": sha256_file(windows_path),
                "row_count": window_count,
            },
        },
        "limitations": [
            "Human-coded annotations are not used as training, test, or ground truth.",
            "Gem quantities are XP-linked estimates, not exact physical pickup counts.",
            "Review flags and unresolved identities are preserved rather than coerced.",
            "This bootstrap reuses existing artifacts; it does not rerun frame extraction.",
            "Five-second windows are derived publication views, not canonical events.",
        ],
        "metadata_verification": {
            "prepared_by": "Tahereh Fahi",
            "human_ground_truth_used": False,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest

