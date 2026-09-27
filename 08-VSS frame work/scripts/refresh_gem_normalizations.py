"""Refresh every completed gem run from its cached detector CSV.

The detector is deliberately not rerun.  This re-applies the current framework
normalization policy to the immutable worker outputs and refreshes their
canonical and signal projections.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> int:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    framework_root = args.framework_root.resolve()
    workspace_root = framework_root.parent
    from vss_framework.normalization.gem_worker import normalize_gems
    from vss_framework.adapters.signals import adapt_automated_signals

    configs = [json.loads(path.read_text(encoding="utf-8")) for path in (framework_root / "configs").glob("*.json")]
    refreshed = []
    for run_dir in sorted((framework_root / "runs").glob("*gem*final*")):
        manifest_path = run_dir / "run_manifest.json"
        event_paths = sorted((run_dir / "worker").glob("collected_gems_*_xp_ab_events.csv"))
        if not manifest_path.is_file() or len(event_paths) != 1:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_sha = manifest.get("inputs", {}).get("video", {}).get("sha256")
        matches = [config for config in configs if config.get("dataset", {}).get("video", {}).get("sha256") == video_sha]
        if len(matches) != 1:
            raise RuntimeError(f"Could not uniquely match config for {run_dir.name}")
        config = matches[0]
        dataset = config["dataset"]
        source_csv = event_paths[0]
        observations, summary, issues = normalize_gems(
            source_csv=source_csv,
            source_csv_relative=source_csv.relative_to(workspace_root).as_posix(),
            processing_run_id=manifest["processing_run_id"],
            session_id=dataset["session_id"],
            video_asset_id=dataset["video_asset_id"],
            duration_ms=int(dataset["duration_ms"]),
            fps=float(dataset["fps"]),
            config={"evidence_policy": "none"},
        )
        canonical_path = run_dir / "canonical_gem_observations.jsonl"
        canonical_count = write_jsonl(canonical_path, observations)
        signals = [row.to_dict() for row in adapt_automated_signals(
            canonical_path,
            source_csv.relative_to(workspace_root).as_posix(),
            start_ms=0,
            end_ms=int(dataset["duration_ms"]),
        )]
        signals_path = run_dir / "signal_observations.jsonl"
        signal_count = write_jsonl(signals_path, signals)
        issues_path = run_dir / "review_issues.jsonl"
        issue_count = write_jsonl(issues_path, issues)
        manifest["normalization_summary"] = summary
        manifest["normalization_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.setdefault("policies", {})["unresolved_closest_color"] = {
            "status": "preserve_unresolved",
            "minimum_confidence": 0.25,
            "quantity_assignment": "tentative_only_when_threshold_met",
        }
        manifest["outputs"]["canonical_gem_observations"].update({"sha256": sha256(canonical_path), "row_count": canonical_count})
        manifest["outputs"]["signal_observations"].update({"sha256": sha256(signals_path), "row_count": signal_count})
        manifest["outputs"]["review_issues"].update({"sha256": sha256(issues_path), "row_count": issue_count})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        assigned = sum(
            int((row.get("attributes_json") or {}).get("closest_color_quantity") or 0)
            for row in observations
            if (row.get("attributes_json") or {}).get("gem_type") == "unresolved"
        )
        refreshed.append({"run": run_dir.name, "observations": canonical_count, "tentatively_assigned_unresolved_quantity": assigned})
    print(json.dumps({"refreshed": refreshed, "runCount": len(refreshed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
