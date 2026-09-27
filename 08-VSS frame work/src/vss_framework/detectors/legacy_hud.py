"""Run the existing Kill Counter/Game Clock worker under framework control."""

from __future__ import annotations

from ..resources import load_runtime_config, resolve_path, source_reference, module_command, worker_environment

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..adapters import adapt_automated_signals
from ..hashing import sha256_file
from ..io import write_json, write_jsonl
from ..xp_level_scan import code_hashes


WRAPPER_VERSION = "0.2.0"
HUD_QC_FAILED_EXIT = 5


def build_legacy_hud_command(
    *,
    python_executable: Path,
    script_path: Path,
    video_path: Path,
    output_dir: Path,
    full: bool,
    max_seconds: int | None,
    sample_offsets: list[float],
    min_ocr_confidence: float,
    min_timer_observed_rate: float,
    min_kill_observed_rate: float,
    initial_kill_state: int,
    evidence_every: int,
    max_evidence_frames: int,
) -> list[str]:
    command = [
        str(python_executable),
        str(script_path),
        "--video",
        str(video_path),
        "--output-dir",
        str(output_dir),
    ]
    if full:
        command.append("--full")
    else:
        if max_seconds is None:
            raise ValueError("A partial HUD run requires max_seconds")
        command.extend(["--max-seconds", str(max_seconds)])
    command.extend(
        [
            "--sample-offsets",
            ",".join(format(value, ".12g") for value in sample_offsets),
            "--min-ocr-confidence",
            format(min_ocr_confidence, ".12g"),
            "--min-timer-observed-rate",
            format(min_timer_observed_rate, ".12g"),
            "--min-kill-observed-rate",
            format(min_kill_observed_rate, ".12g"),
            "--initial-kill-state",
            str(initial_kill_state),
            "--evidence-every",
            str(evidence_every),
            "--max-evidence-frames",
            str(max_evidence_frames),
        ]
    )
    return command


def _relative(path: Path, workspace_root: Path) -> str:
    return source_reference(path, workspace_root)


def _load_normalizer(module_dir: Path | None = None) -> Any:
    from ..normalization.hud_worker import normalize_hud
    return normalize_hud


def _run_id(identity: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"run_hud_clock_{digest[:20]}"


def run_legacy_hud_detector(
    *,
    workspace_root: Path,
    config_path: Path,
    output_dir: Path,
    python_executable: Path | None = None,
    max_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute the prior OCR worker without Human validation or database I/O."""

    config = load_runtime_config(config_path)
    dataset = config["dataset"]
    detector_config = config["detectors"]["hud_clock"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    script_path = resolve_path(detector_config["script"], workspace_root)
    template_paths = [
        resolve_path(path, workspace_root)
        for path in detector_config["template_paths"]
    ]
    python_path = Path(python_executable or sys.executable).absolute()
    if not video_path.is_file() or not script_path.is_file():
        raise FileNotFoundError("Video or existing HUD worker is missing")
    if not all(path.is_file() for path in template_paths):
        raise FileNotFoundError("HUD templates or pipeline normalizer module is missing")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    video_sha = sha256_file(video_path)
    if video_sha != str(dataset["video"]["sha256"]):
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    script_sha = sha256_file(script_path)
    implementation_hashes = code_hashes()
    config_sha = sha256_file(config_path)
    template_records = [
        {
            "path": _relative(path, workspace_root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in template_paths
    ]
    full = max_seconds is None
    effective_configuration = {
        "full": full,
        "max_seconds": max_seconds,
        "sample_offsets_seconds": detector_config["sample_offsets_seconds"],
        "min_ocr_confidence": detector_config["min_ocr_confidence"],
        "min_timer_observed_rate": detector_config["min_timer_observed_rate"],
        "min_kill_observed_rate": detector_config["min_kill_observed_rate"],
        "initial_kill_state": detector_config["initial_kill_state"],
        "evidence_every_seconds": detector_config["evidence_every_seconds"],
        "max_evidence_frames": detector_config["max_evidence_frames"],
    }
    run_id = _run_id(
        {
            "wrapper_version": WRAPPER_VERSION,
            "video_sha256": video_sha,
            "worker_sha256": script_sha,
            "code_sha256": implementation_hashes,
            "config_sha256": config_sha,
            "templates": template_records,
            "configuration": effective_configuration,
        }
    )

    worker_dir = output_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    command = build_legacy_hud_command(
        python_executable=python_path,
        script_path=script_path,
        video_path=video_path,
        output_dir=worker_dir,
        full=full,
        max_seconds=max_seconds,
        sample_offsets=[float(value) for value in detector_config["sample_offsets_seconds"]],
        min_ocr_confidence=float(detector_config["min_ocr_confidence"]),
        min_timer_observed_rate=float(detector_config["min_timer_observed_rate"]),
        min_kill_observed_rate=float(detector_config["min_kill_observed_rate"]),
        initial_kill_state=int(detector_config["initial_kill_state"]),
        evidence_every=int(detector_config["evidence_every_seconds"]),
        max_evidence_frames=int(detector_config["max_evidence_frames"]),
    )
    completed = subprocess.run(module_command(command, "hud_clock"), check=False, env=worker_environment())
    if completed.returncode not in {0, HUD_QC_FAILED_EXIT}:
        raise RuntimeError(
            f"Legacy HUD worker exited with status {completed.returncode}"
        )

    worker_manifest_path = worker_dir / "manifest.json"
    observations_csv = worker_dir / "hud_observations.csv"
    if not worker_manifest_path.is_file() or not observations_csv.is_file():
        raise RuntimeError("HUD worker did not produce its required outputs")
    worker_manifest = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
    qc_path = worker_dir / "qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    processed_seconds = int(worker_manifest["configuration"]["processed_seconds"])
    effective_duration_ms = min(dataset["duration_ms"], processed_seconds * 1000)

    normalize_hud = _load_normalizer()
    canonical_rows, normalization_summary, clock_issues = normalize_hud(
        source_csv=observations_csv,
        source_csv_relative=_relative(observations_csv, workspace_root),
        processing_run_id=run_id,
        session_id=dataset["session_id"],
        video_asset_id=dataset["video_asset_id"],
        duration_ms=effective_duration_ms,
        config={
            "source_sampling_offsets_seconds": detector_config[
                "sample_offsets_seconds"
            ],
            "clock_qc": detector_config["clock_qc"],
        },
    )
    canonical_path = output_dir / "canonical_hud_observations.jsonl"
    canonical_count = write_jsonl(canonical_path, canonical_rows)
    signals = list(
        adapt_automated_signals(
            canonical_path,
            _relative(canonical_path, workspace_root),
            start_ms=0,
            end_ms=effective_duration_ms,
        )
    )
    signals_path = output_dir / "signal_observations.jsonl"
    signal_count = write_jsonl(
        signals_path, (observation.to_dict() for observation in signals)
    )
    issues_path = output_dir / "clock_issues.jsonl"
    issue_count = write_jsonl(issues_path, clock_issues)
    observable_counts = Counter(signal.observable_code for signal in signals)
    kill_signals = [signal for signal in signals if signal.observable_code == "kill_counter"]

    logical_command = [
        "<configured-python>",
        detector_config["script"],
        "--video",
        dataset["video"]["path"],
        "--output-dir",
        "worker",
        "--full" if full else "--max-seconds",
    ]
    if not full:
        logical_command.append(str(max_seconds))
    logical_command.extend(
        [
            "--sample-offsets",
            ",".join(str(value) for value in detector_config["sample_offsets_seconds"]),
            "--min-ocr-confidence",
            str(detector_config["min_ocr_confidence"]),
            "--min-timer-observed-rate",
            str(detector_config["min_timer_observed_rate"]),
            "--min-kill-observed-rate",
            str(detector_config["min_kill_observed_rate"]),
            "--initial-kill-state",
            str(detector_config["initial_kill_state"]),
        ]
    )

    produced_files = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(worker_dir.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "artifact_type": "vss_framework_legacy_hud_run",
        "framework_version": "0.4.0",
        "wrapper_version": WRAPPER_VERSION,
        "processing_run_id": run_id,
        "code_sha256": implementation_hashes,
        "config_sha256": config_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "run_scope": "full_video" if full else "smoke_prefix",
        "max_seconds": max_seconds,
        "algorithm_policy": "existing_worker_executed_without_algorithm_changes",
        "worker_exit_code": completed.returncode,
        "worker_qc_status": qc["status"],
        "logical_command": logical_command,
        "inputs": {
            "video": {
                "path": dataset["video"]["path"],
                "sha256": video_sha,
                "verified": True,
            },
            "worker": {
                "path": detector_config["script"],
                "sha256": script_sha,
            },
            "templates": template_records,
        },
        "configuration": effective_configuration,
        "counts": {
            "canonical_hud_observations": canonical_count,
            "signal_observations": signal_count,
            "signals_by_observable": dict(sorted(observable_counts.items())),
            "clock_issues": issue_count,
            "observed_kill_counter_signals": sum(signal.observed for signal in kill_signals),
            "carried_or_assumed_kill_counter_signals": sum(
                signal.attributes.get("source_state_source") in {"carried_forward", "assumed_initial_state"}
                for signal in kill_signals
            ),
            "excluded_level_up_pause_kill_signals": sum(bool(signal.attributes.get("excluded_from_gameplay")) for signal in kill_signals),
            "final_kill_counter": (
                kill_signals[-1].numeric_value if kill_signals else None
            ),
        },
        "worker_qc": qc,
        "normalization_summary": normalization_summary,
        "outputs": {
            "canonical_hud_observations": {
                "path": canonical_path.name,
                "sha256": sha256_file(canonical_path),
                "row_count": canonical_count,
            },
            "signal_observations": {
                "path": signals_path.name,
                "sha256": sha256_file(signals_path),
                "row_count": signal_count,
            },
            "clock_issues": {
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
            "timer_imputation_performed": False,
            "kill_state_carry_preserved": True,
        },
        "metadata_verification": {
            "prepared_by": "Tahereh Fahi",
            "human_ground_truth_used": False,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    if code_hashes() != implementation_hashes or sha256_file(config_path) != config_sha:
        raise RuntimeError("Framework code or configuration changed during the HUD run")
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
