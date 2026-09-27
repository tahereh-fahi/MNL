#!/usr/bin/env python3
"""Audit a completed VSS run without decoding its source video.

Prepared by Tahereh Fahi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_time(milliseconds: int | float | None) -> str | None:
    if milliseconds is None:
        return None
    milliseconds = int(round(float(milliseconds)))
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def stage_artifact(run: Path, root: dict[str, Any], stage: str, relative: str) -> Path:
    record = root["stages"][stage]
    return run / record["directory"] / relative


def audit_run(run: Path, *, require_full: bool, require_fresh: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        return {"run": run.name, "status": "missing_manifest"}, ["Missing run_manifest.json"]

    root = read_json(manifest_path)
    if root.get("status") != "complete":
        errors.append(f"Run status is {root.get('status')!r}, not 'complete'")
    if root.get("stage_scope") != "all":
        errors.append(f"Stage scope is {root.get('stage_scope')!r}, not 'all'")
    if require_full and root.get("run_scope") != "full_video":
        errors.append(f"Run scope is {root.get('run_scope')!r}, not 'full_video'")
    if root.get("quality_failed_stages"):
        errors.append(f"Quality failures: {root['quality_failed_stages']}")

    hash_mismatches: list[str] = []
    incomplete: list[str] = []
    reused: list[str] = []
    for name, record in root.get("stages", {}).items():
        if record.get("status") != "complete":
            incomplete.append(name)
        if record.get("reused"):
            reused.append(name)
        for relative, expected in record.get("outputs", {}).items():
            path = run / record["directory"] / relative
            if not path.is_file() or sha256_file(path) != expected:
                hash_mismatches.append(f"{name}/{relative}")
    if incomplete:
        errors.append(f"Incomplete stages: {incomplete}")
    if require_fresh and reused:
        errors.append(f"Reused stages: {reused}")
    if hash_mismatches:
        errors.append(f"Missing or hash-mismatched outputs: {hash_mismatches}")
    hud = root.get("stages", {}).get("hud", {})
    if hud and hud.get("quality_status") != "passed":
        errors.append(f"HUD quality status is {hud.get('quality_status')!r}")

    rewards: list[dict[str, Any]] = []
    if "chest_rewards" in root.get("stages", {}):
        reward_path = stage_artifact(run, root, "chest_rewards", "chest_reward_events.jsonl")
        if reward_path.is_file():
            rewards = read_jsonl(reward_path)
    unresolved_rewards = [row for row in rewards if row.get("publication_status") != "auto_accepted"]
    if unresolved_rewards:
        errors.append(f"Unresolved chest rewards: {len(unresolved_rewards)}")

    pauses: list[dict[str, Any]] = []
    if "release" in root.get("stages", {}):
        pause_path = stage_artifact(run, root, "release", "gameplay_pauses.json")
        if pause_path.is_file():
            pauses = read_json(pause_path).get("intervals", [])

    reconciliation: dict[str, Any] = {}
    if "inventory_reconciled" in root.get("stages", {}):
        reconciliation_path = stage_artifact(run, root, "inventory_reconciled", "run_manifest.json")
        if reconciliation_path.is_file():
            reconciliation = read_json(reconciliation_path)
            if reconciliation.get("status") != "complete":
                errors.append("Inventory reconciliation is not complete")

    # These are review diagnostics, not a run-wide pass/fail gate.  In
    # particular, production_output_recommended describes the optional
    # video-only percentage-assist submethod rather than the complete gems
    # stage, so treating it as overall publication readiness would be false.
    gem_review: dict[str, Any] = {}
    if "gems" in root.get("stages", {}):
        gem_summary_path = stage_artifact(run, root, "gems", "worker/summary.json")
        if gem_summary_path.is_file():
            gem_summary = read_json(gem_summary_path)
            gem_review = {
                "events_needing_review": gem_summary.get("events_needing_review"),
                "unresolved_collected_gems": gem_summary.get(
                    "unresolved_collected_gems"
                ),
                "percentage_assist_enabled": gem_summary.get(
                    "video_only_percentage_assist_enabled"
                ),
                "percentage_assisted_events": gem_summary.get(
                    "percentage_assisted_events"
                ),
                "percentage_unresolved_after": gem_summary.get(
                    "percentage_unresolved_after"
                ),
                "percentage_assist_production_output_recommended": (
                    gem_summary.get("production_output_recommended")
                ),
            }

    summary = {
        "prepared_by": "Tahereh Fahi",
        "run": run.name,
        "audit_status": "passed" if not errors else "failed",
        "run_status": root.get("status"),
        "run_scope": root.get("run_scope"),
        "stage_scope": root.get("stage_scope"),
        "stage_count": len(root.get("stages", {})),
        "reused_stages": reused,
        "quality_failed_stages": root.get("quality_failed_stages", []),
        "hash_mismatches": hash_mismatches,
        "pause_counts": dict(Counter(row.get("eventType", "unknown") for row in pauses)),
        "chest_rewards": [
            {
                "time": format_time(row.get("anchor_time_ms")),
                "item": row.get("item_name"),
                "status": row.get("publication_status"),
            }
            for row in rewards
        ],
        "inventory_reconciliation": reconciliation.get("counts", {}),
        "gem_review": gem_review,
        "execution_seconds": root.get("execution_seconds"),
        "prompt_workflow_elapsed_seconds": root.get("prompt_workflow_elapsed_seconds"),
        "metadata_verification": {"prepared_by": root.get("prepared_by")},
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--require-fresh", action="store_true")
    args = parser.parse_args()
    summary, errors = audit_run(args.run.resolve(), require_full=args.require_full, require_fresh=args.require_fresh)
    summary["errors"] = errors
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
