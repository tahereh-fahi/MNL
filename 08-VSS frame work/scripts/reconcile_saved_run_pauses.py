#!/usr/bin/env python3
"""Rebuild pause export from hash-verified saved gem and chest evidence.

Prepared by Tahereh Fahi.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vss_framework.gameplay_pause_export import build_gameplay_pauses
from vss_framework.hashing import sha256_file
from vss_framework.io import write_json


def checked_artifact(run: Path, root: dict, stage: str, relative: str) -> Path:
    record = root["stages"][stage]
    path = run / record["directory"] / relative
    if not path.is_file() or sha256_file(path) != record["outputs"].get(relative):
        raise RuntimeError(f"Missing or hash-mismatched input: {stage}/{relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    run = args.run.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    root_path = run / "run_manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    if root.get("status") != "complete" or root.get("stage_scope") != "all":
        raise RuntimeError("Source run must be a completed all-stage run")
    signal_names = [
        name for name in root["stages"]["gems"]["outputs"]
        if name.startswith("worker/") and name.endswith("_xp_frame_signal.csv")
    ]
    if len(signal_names) != 1:
        raise RuntimeError("Expected exactly one saved gem frame signal")
    signal = checked_artifact(run, root, "gems", signal_names[0])
    chests_path = checked_artifact(run, root, "chests", "chest_events.jsonl")
    runtime_path = run / "runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    fps = float(runtime["dataset"]["fps"])
    chests = [
        json.loads(line) for line in chests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with signal.open(newline="", encoding="utf-8") as handle:
        pauses = build_gameplay_pauses(csv.DictReader(handle), fps, chests)

    output.mkdir(parents=True, exist_ok=True)
    pauses_path = output / "gameplay_pauses.json"
    write_json(pauses_path, pauses)
    counts = Counter(row["eventType"] for row in pauses["intervals"])
    manifest = {
        "artifact_type": "vss_saved_run_pause_reconciliation",
        "prepared_by": "Tahereh Fahi",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": run.name,
        "source_integrity": {
            "run_manifest_sha256": sha256_file(root_path),
            "runtime_config_sha256": sha256_file(runtime_path),
            "gem_frame_signal_sha256": sha256_file(signal),
            "chest_events_sha256": sha256_file(chests_path),
            "pause_export_source_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src/vss_framework/gameplay_pause_export.py"
            ),
        },
        "counts": {
            "intervals": len(pauses["intervals"]),
            "by_event_type": dict(sorted(counts.items())),
            "accepted_chest_events": sum(
                chest.get("publication_status") == "auto_accepted" for chest in chests
            ),
            "chest_linked_treasure_pauses": sum(
                bool(row.get("sourceChestEventId")) for row in pauses["intervals"]
            ),
            "unmatched_treasure_visual_fragments": sum(
                bool(row.get("unmatchedTreasureVisualEvidence")) for row in pauses["intervals"]
            ),
        },
        "outputs": {
            "gameplay_pauses.json": {
                "sha256": sha256_file(pauses_path),
                "interval_count": len(pauses["intervals"]),
            }
        },
        "execution_seconds": time.monotonic() - started,
        "prompt_workflow_elapsed_seconds": None,
        "publication_ready": False,
        "metadata_verification": {"prepared_by": "Tahereh Fahi"},
    }
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
