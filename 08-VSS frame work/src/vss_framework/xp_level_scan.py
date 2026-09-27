"""Opt-in video-only XP evidence stage, separate from inventory/gem tracking."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict, fields
from pathlib import Path
from .runtime import distribution_version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_scope(max_seconds: float) -> None:
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("--max-seconds must be finite and positive")


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted(root.rglob("*.py"))}


def extract_xp_levels(config_path: Path, workspace_root: Path, output: Path,
                      max_seconds: float, video_override: Path | None = None) -> dict:
    started = time.perf_counter()
    validate_scope(max_seconds)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new --output directory")
    config_hash = sha256(config_path)
    config = json.loads(config_path.read_text())
    source = config["dataset"]["video"]
    video = (video_override or workspace_root / source["path"]).resolve()
    source_hash = sha256(video)
    if source_hash != source["sha256"]:
        raise ValueError("Video SHA-256 does not match configuration")
    initial_level = config["detectors"]["gem_xp"]["initial_level"]
    if type(initial_level) is not int or initial_level < 1:
        raise ValueError("Configured initial_level must be a positive integer")

    import easyocr
    import numpy as np
    from .detectors import gems
    from .detectors.hud import preflight_easyocr_models

    _, models = preflight_easyocr_models(easyocr)
    fingerprints = code_hashes()
    cfg = gems.Config(initial_level=initial_level)
    manifest = {
        "prepared_by": "Tahereh Fahi", "stage": "xp_levels",
        "execution_status": "running", "publication_ready": False,
        "detector_accuracy": "not_validated", "reuse_allowed": False,
        "source": {"filename": video.name, "sha256": source_hash},
        "config_sha256": config_hash, "code_sha256": fingerprints,
        "effective_detector_config": asdict(cfg), "ocr_models": models,
        "dependencies": {name: distribution_version(name) for name in
                         ("numpy", "pandas", "scipy", "opencv-python", "easyocr", "torch", "torchvision")},
        "python": platform.python_version(),
        "scope": {"start_seconds": 0, "max_seconds": max_seconds},
        "upstream_dataset_inputs": [],
        "assumptions": ["initial_level is configured, not observed",
                        "Other detector parameters use recorded Config defaults",
                        "Times are frame_index / FPS, not game time or verified VFR timestamps",
                        "HUD OCR is sampled at XP candidate frames, not a complete level timeline",
                        "Accepted OCR and reset-inferred fallback remain distinct fields"],
        "prompt_workflow_elapsed": "not recorded",
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    try:
        metadata, arrays = gems.scan_xp_signal(video, cfg, max_seconds)
        resets = gems.find_level_resets(arrays, cfg)
        events = gems.find_xp_events(metadata, arrays, resets, cfg)
        gems.validate_transition_event_invariants(events, arrays, resets)
        events = gems.attach_hud_levels(video, events, cfg, arrays)
        np.savez_compressed(output / "xp_signal.npz", **arrays)
        # Keep detector observations and inferred values; no reward/color estimates.
        excluded = {"xp_required", "growth_multiplier", "estimated_base_xp_gain",
                    "xp_color_hint", "local_single_step_pixels", "jump_step_ratio"}
        columns = [field.name for field in fields(gems.XPEvent) if field.name not in excluded]
        columns += ["reset_inferred_level"]
        with (output / "xp_level_events.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for event in events:
                row = {key: value for key, value in asdict(event).items() if key not in excluded}
                row["reset_inferred_level"] = event.inferred_level
                writer.writerow(row)
        (output / "reset_candidates.json").write_text(json.dumps([
            {"frame": int(frame), "video_time": int(frame) / metadata["fps"],
             "reset_inferred_level": initial_level + index + 1}
            for index, frame in enumerate(resets)], indent=2) + "\n")
        if (sha256(video) != source_hash or sha256(config_path) != config_hash
                or code_hashes() != fingerprints
                or preflight_easyocr_models(easyocr)[1] != models):
            raise RuntimeError("Input, code, configuration or OCR weights changed during extraction")
        manifest.update(execution_status="complete", video_metadata=metadata,
                        xp_candidate_count=len(events), reset_candidate_count=len(resets),
                        outputs={name: sha256(output / name) for name in
                                 ("xp_signal.npz", "xp_level_events.csv", "reset_candidates.json")})
    except BaseException as error:
        manifest.update(execution_status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["command_execution_seconds"] = time.perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {key: manifest[key] for key in ("execution_status", "stage", "xp_candidate_count",
            "reset_candidate_count", "publication_ready", "command_execution_seconds", "prompt_workflow_elapsed")}
