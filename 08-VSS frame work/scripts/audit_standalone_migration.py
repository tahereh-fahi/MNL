"""Check migration parity and every configured source before batch publication.

Prepared by Tahereh Fahi. This command never edits reference results.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

from vss_framework.hashing import sha256_file
from vss_framework.io import write_json
from vss_framework.resources import PACKAGE_ROOT, load_runtime_config, resolve_path


SOURCES = {
    "detectors/gems.py": "02_gems/scripts/detect_collected_gems_from_xp_ab.py",
    "detectors/inventory.py": "03_weapons/scripts/inventory_event_recorder.py",
    "detectors/weapons.py": "03_weapons/scripts/weapon_screen_recorder.py",
    "detectors/hud.py": "01_kill_counter_and_time_stamp/scripts/extract_hud_worker.py",
    **{f"normalization/{name}.py": f"06-Pipeline/mnl_pipeline/{name}.py" for name in ("contracts", "clock_qc", "io_utils", "gem_worker", "hud_worker")},
}


def algorithm_tree(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def audit(workspace: Path, framework: Path, output: Path) -> dict:
    start = time.monotonic()
    parity = []
    for internal, legacy in SOURCES.items():
        old, new = workspace / legacy, PACKAGE_ROOT / internal
        old_tree, new_tree = algorithm_tree(old), algorithm_tree(new)
        parity.append({"module": internal, "reference_source": legacy, "reference_sha256": sha256_file(old),
            "internal_sha256": sha256_file(new), "algorithm_bodies_identical": old_tree == new_tree,
            "changed_or_added_definitions": sorted(name for name in old_tree.keys() | new_tree.keys() if old_tree.get(name) != new_tree.get(name)),
            "migration_note": "Accept explicitly confirmed zero-event XP input and bound inventory initialization to the requested interval; matching and reconciliation are preserved." if internal == "detectors/inventory.py" else None})
    configs = []
    for path in sorted((framework / "configs").glob("*.json")):
        config = json.loads(path.read_text())
        if "dataset" in config and "video" in config["dataset"]:
            configs.append((path, config))
    video_paths = set((workspace / "00-Videos").glob("*.mp4"))
    video_paths.update(resolve_path(c["dataset"]["video"]["path"], workspace) for _, c in configs)
    hashes = {path: sha256_file(path) for path in sorted(video_paths) if path.is_file()}
    checks = []
    for path, config in configs:
        spec = config["dataset"]["video"]
        configured = resolve_path(spec["path"], workspace)
        matches = [p for p, sha in hashes.items() if sha == spec["sha256"]]
        status = "ready" if configured in matches else "renamed_source_verified" if len(matches) == 1 else "missing_or_changed_source"
        checks.append({"config": path.name, "configured_video": configured.name, "status": status,
            "matching_files": [p.relative_to(workspace).as_posix() for p in matches],
            "requires_per_run_configs": bool(config.get("policies", {}).get("multi_run_video")),
            "initial_level_configured": "initial_level" in config.get("detectors", {}).get("gem_xp", {}),
            "expected_sha256": spec["sha256"], "actual_sha256": hashes.get(configured)})
    current_gem_sha = sha256_file(PACKAGE_ROOT / "detectors/gems.py")
    cached = []
    for path in sorted((framework / "runs").glob("*/run_manifest.json")):
        manifest = json.loads(path.read_text())
        if "gem_xp" not in manifest.get("artifact_type", ""):
            continue
        recorded = manifest.get("inputs", {}).get("detector", {}).get("sha256")
        cached.append({"run": path.parent.name, "detector_matches_current_code": recorded == current_gem_sha,
            "run_scope": manifest.get("run_scope"), "video_sha256": manifest.get("inputs", {}).get("video", {}).get("sha256")})
    unconfigured = [p.name for p, sha in hashes.items() if p.parent == workspace / "00-Videos" and not any(c["dataset"]["video"]["sha256"] == sha for _, c in configs)]
    report = {"prepared_by": "Tahereh Fahi", "algorithm_parity": parity, "configured_videos": checks,
        "source_files_without_matching_config": unconfigured, "cached_gem_runs": cached,
        "execution_seconds": time.monotonic() - start, "prompt_workflow_elapsed_seconds": None,
        "metadata_verification": {"prepared_by": "Tahereh Fahi", "absolute_paths_in_report": False}}
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    framework = Path(__file__).resolve().parents[1]
    result = audit(args.workspace_root.resolve(), framework, args.output)
    print(json.dumps({"algorithm_parity": all(r["algorithm_bodies_identical"] for r in result["algorithm_parity"]),
        "configured_videos": result["configured_videos"], "unconfigured_sources": result["source_files_without_matching_config"],
        "cached_current_detector_runs": sum(r["detector_matches_current_code"] for r in result["cached_gem_runs"]),
        "cached_total_runs": len(result["cached_gem_runs"]), "execution_seconds": result["execution_seconds"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
