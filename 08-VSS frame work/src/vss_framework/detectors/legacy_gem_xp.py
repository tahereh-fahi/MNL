"""Run the existing full XP/gem detector unchanged under framework control."""

from __future__ import annotations

from ..resources import load_runtime_config, resolve_path, source_reference, module_command, worker_environment

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..adapters import adapt_automated_signals
from ..hashing import sha256_file
from ..io import write_json, write_jsonl
from ..xp_level_scan import code_hashes


WRAPPER_VERSION = "0.2.0"


def build_legacy_gem_command(
    *,
    python_executable: Path,
    script_path: Path,
    video_path: Path,
    template_dir: Path,
    output_dir: Path,
    template_profile: str,
    initial_level: int,
    inventory_events: Path | None,
    max_seconds: float | None,
    save_previews: bool,
) -> list[str]:
    command = [
        str(python_executable),
        str(script_path),
        "--video",
        str(video_path),
        "--template-dir",
        str(template_dir),
        "--output-dir",
        str(output_dir),
        "--template-profile",
        template_profile,
        "--initial-level",
        str(initial_level),
    ]
    if inventory_events is not None:
        command.extend(["--inventory-events", str(inventory_events)])
    if max_seconds is not None:
        command.extend(["--max-seconds", format(max_seconds, ".12g")])
    if not save_previews:
        command.append("--no-previews")
    return command


def _relative(path: Path, workspace_root: Path) -> str:
    return source_reference(path, workspace_root)


def _hash_directory(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
            "size_bytes": item.stat().st_size,
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def _processing_run_id(
    *,
    script_sha: str,
    video_sha: str,
    max_seconds: float | None,
    inventory_sha: str,
    template_fingerprint: str,
    template_profile: str,
    initial_level: int,
    save_previews: bool,
    code_fingerprint: str,
    config_sha: str,
) -> str:
    key = (
        f"{WRAPPER_VERSION}|{script_sha}|{video_sha}|{max_seconds}|"
        f"{inventory_sha}|{template_fingerprint}|{template_profile}|"
        f"{initial_level}|{save_previews}|{code_fingerprint}|{config_sha}"
    )
    return f"run_gem_xp_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}"


def _load_normalizer(module_dir: Path | None = None) -> Any:
    from ..normalization.gem_worker import normalize_gems
    return normalize_gems


def normalize_completed_gem_worker(*, worker_summary: dict[str, Any], **arguments: Any) -> Any:
    """Accept zero events only when a completed worker explicitly reports zero."""
    if worker_summary.get("xp_jump_events") != 0:
        return _load_normalizer()(**arguments)
    from ..normalization.io_utils import read_csv
    quantity_fields = ("collected_blue_gems", "collected_green_gems", "collected_red_gems",
                       "unresolved_collected_gems", "collected_gems_total")
    if worker_summary.get("human_labels_used") is not False or worker_summary.get("audio_used") is not False:
        raise ValueError("Empty gem output lacks the required worker provenance")
    if any(worker_summary.get(name) != 0 for name in quantity_fields) or read_csv(arguments["source_csv"]):
        raise ValueError("Empty gem output disagrees with the completed worker summary")
    return [], {
        "source_event_rows": 0, "observation_rows": 0,
        "counts_by_observable": {"gem_pickup": 0},
        "quantity_by_gem_type": {name: 0 for name in ("blue", "green", "red", "unresolved")},
        "total_gem_quantity": 0, "needs_review_event_count": 0,
        "needs_review_event_fraction": None, "unresolved_quantity_fraction": None,
        "unresolved_quantity_fraction_denominator": 0,
        "unresolved_quantity_fraction_status": "no_measured_pickups",
        "events_without_preview_evidence": 0,
        "evidence_policy": arguments["config"].get("evidence_policy", "none"),
        "human_labels_used_by_detector": False, "audio_used_by_detector": False,
        "empty_result_verified_against_worker_summary": True,
    }, []


def run_legacy_gem_xp_detector(
    *,
    workspace_root: Path,
    config_path: Path,
    output_dir: Path,
    python_executable: Path | None = None,
    max_seconds: float | None = None,
    save_previews: bool = False,
    reuse_worker: bool = False,
) -> dict[str, Any]:
    """Execute the prior detector and adapt its output without Human coding."""

    config = load_runtime_config(config_path)
    dataset = config["dataset"]
    detector_config = config["detectors"]["gem_xp"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    script_path = resolve_path(detector_config["script"], workspace_root)
    template_dir = resolve_path(detector_config["template_dir"], workspace_root)
    inventory_value = detector_config.get("inventory_events")
    inventory_path = resolve_path(inventory_value, workspace_root) if inventory_value else None
    python_path = Path(python_executable or sys.executable).absolute()

    required_files = (video_path, script_path) + ((inventory_path,) if inventory_path else ())
    if not all(path.is_file() for path in required_files):
        raise FileNotFoundError("Video, detector script, or inventory input is missing")
    if not template_dir.is_dir():
        raise FileNotFoundError("Gem templates or pipeline normalizer module is missing")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not reuse_worker:
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    video_sha = sha256_file(video_path)
    expected_video_sha = str(dataset["video"]["sha256"])
    if video_sha != expected_video_sha:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    script_sha = sha256_file(script_path)
    implementation_hashes = code_hashes()
    implementation_fingerprint = hashlib.sha256(
        json.dumps(implementation_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config_sha = sha256_file(config_path)
    inventory_sha = sha256_file(inventory_path) if inventory_path else "none"
    template_files = _hash_directory(template_dir)
    template_fingerprint = hashlib.sha256(
        json.dumps(template_files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = _processing_run_id(
        script_sha=script_sha,
        video_sha=video_sha,
        max_seconds=max_seconds,
        inventory_sha=inventory_sha,
        template_fingerprint=template_fingerprint,
        template_profile=str(detector_config["template_profile"]),
        initial_level=int(detector_config["initial_level"]),
        save_previews=save_previews,
        code_fingerprint=implementation_fingerprint,
        config_sha=config_sha,
    )

    if reuse_worker:
        prior_path = output_dir / "run_manifest.json"
        if not prior_path.is_file():
            raise ValueError("Cannot reuse a worker without its verified run manifest")
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if (prior.get("processing_run_id") != run_id
                or prior.get("code_sha256") != implementation_hashes
                or prior.get("config_sha256") != config_sha):
            raise ValueError("Cached worker inputs or detector configuration have changed")
        records = prior.get("outputs", {}).get("worker_files", [])
        if not records:
            raise ValueError("Cached worker has no output integrity records")
        for record in records:
            cached = (output_dir / record["path"]).resolve()
            if not cached.is_relative_to(output_dir.resolve()) or not cached.is_file() or sha256_file(cached) != record["sha256"]:
                raise ValueError("Cached worker output failed integrity verification")

    worker_dir = output_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    command = build_legacy_gem_command(
        python_executable=python_path,
        script_path=script_path,
        video_path=video_path,
        template_dir=template_dir,
        output_dir=worker_dir,
        template_profile=str(detector_config["template_profile"]),
        initial_level=int(detector_config["initial_level"]),
        inventory_events=inventory_path,
        max_seconds=max_seconds,
        save_previews=save_previews,
    )
    if not reuse_worker:
        completed = subprocess.run(module_command(command, "gem_xp"), check=False, env=worker_environment())
        if completed.returncode != 0:
            raise RuntimeError(
                f"Legacy XP/gem detector exited with status {completed.returncode}"
            )

    event_paths = sorted(worker_dir.glob("collected_gems_*_xp_ab_events.csv"))
    if len(event_paths) != 1:
        raise RuntimeError("Detector did not produce exactly one XP A/B event CSV")
    event_path = event_paths[0]
    summary_path = worker_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Detector did not produce summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("human_labels_used") is not False:
        raise ValueError("Detector summary did not certify label-free execution")

    effective_duration_ms = (
        min(dataset["duration_ms"], round(max_seconds * 1000))
        if max_seconds is not None
        else dataset["duration_ms"]
    )
    canonical_rows, normalization_summary, review_issues = normalize_completed_gem_worker(
        worker_summary=summary,
        source_csv=event_path,
        source_csv_relative=_relative(event_path, workspace_root),
        processing_run_id=run_id,
        session_id=dataset["session_id"],
        video_asset_id=dataset["video_asset_id"],
        duration_ms=effective_duration_ms,
        fps=float(dataset["fps"]),
        config={"evidence_policy": "none" if not save_previews else "saved"},
    )
    canonical_path = output_dir / "canonical_gem_observations.jsonl"
    canonical_count = write_jsonl(canonical_path, canonical_rows)
    signal_rows = list(
        adapt_automated_signals(
            canonical_path,
            _relative(canonical_path, workspace_root),
            start_ms=0,
            end_ms=effective_duration_ms,
        )
    )
    signals_path = output_dir / "signal_observations.jsonl"
    signal_count = write_jsonl(
        signals_path, (observation.to_dict() for observation in signal_rows)
    )
    issues_path = output_dir / "review_issues.jsonl"
    issue_count = write_jsonl(issues_path, review_issues)
    type_quantities = Counter()
    for signal in signal_rows:
        type_quantities[str(signal.attributes.get("gem_type"))] += int(
            signal.numeric_value or 0
        )

    logical_command = [
        "<configured-python>",
        detector_config["script"],
        "--video",
        dataset["video"]["path"],
        "--template-dir",
        detector_config["template_dir"],
        "--output-dir",
        "worker",
        "--template-profile",
        detector_config["template_profile"],
        "--initial-level",
        str(detector_config["initial_level"]),
    ]
    if inventory_path is not None:
        logical_command.extend(["--inventory-events", detector_config["inventory_events"]])
    if max_seconds is not None:
        logical_command.extend(["--max-seconds", format(max_seconds, ".12g")])
    if not save_previews:
        logical_command.append("--no-previews")

    produced_files = []
    for path in sorted(worker_dir.rglob("*")):
        if path.is_file():
            produced_files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {
        "artifact_type": "vss_framework_legacy_gem_xp_run",
        "framework_version": "0.3.0",
        "wrapper_version": WRAPPER_VERSION,
        "processing_run_id": run_id,
        "code_sha256": implementation_hashes,
        "implementation_fingerprint_sha256": implementation_fingerprint,
        "config_sha256": config_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "run_scope": "full_video" if max_seconds is None else "smoke_prefix",
        "max_seconds": max_seconds,
        "algorithm_policy": (
            "existing_worker_outputs_reused_for_normalization"
            if reuse_worker else "existing_detector_executed_without_algorithm_changes"
        ),
        "logical_command": logical_command,
        "inputs": {
            "video": {
                "path": dataset["video"]["path"],
                "sha256": video_sha,
                "verified": True,
            },
            "detector": {
                "path": detector_config["script"],
                "sha256": script_sha,
            },
            "inventory": {
                "path": detector_config.get("inventory_events"),
                "sha256": inventory_sha,
                "source_type": "automated",
            },
            "templates": {
                "path": detector_config["template_dir"],
                "fingerprint_sha256": template_fingerprint,
                "files": template_files,
            },
        },
        "counts": {
            "xp_events": int(summary["xp_jump_events"]),
            "canonical_gem_observations": canonical_count,
            "signal_observations": signal_count,
            "review_issues": issue_count,
            "gem_quantity_total": sum(type_quantities.values()),
            "gem_quantity_by_type": dict(sorted(type_quantities.items())),
        },
        "normalization_summary": normalization_summary,
        "outputs": {
            "canonical_gem_observations": {
                "path": canonical_path.name,
                "sha256": sha256_file(canonical_path),
                "row_count": canonical_count,
            },
            "signal_observations": {
                "path": signals_path.name,
                "sha256": sha256_file(signals_path),
                "row_count": signal_count,
            },
            "review_issues": {
                "path": issues_path.name,
                "sha256": sha256_file(issues_path),
                "row_count": issue_count,
            },
            "worker_files": produced_files,
        },
        "policies": {
            "human_coded_ground_truth_used": False,
            "human_validation_loaded": False,
            "database_write_performed": False,
            "blank_is_zero": False,
        },
        "metadata_verification": {
            "prepared_by": "Tahereh Fahi",
            "human_ground_truth_used": False,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    if code_hashes() != implementation_hashes or sha256_file(config_path) != config_sha:
        raise RuntimeError("Framework code or configuration changed during the gem run")
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
