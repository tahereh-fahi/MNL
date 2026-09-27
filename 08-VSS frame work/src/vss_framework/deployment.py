"""Operator setup, preflight, and fresh-run checks. Author: Tahereh Fahi.

Heavy dependencies load only inside the requested operation. Model downloads
are opt-in. This module never changes detector thresholds or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

from . import __version__
from .hashing import sha256_file
from .resources import PACKAGE_ROOT, load_runtime_config, resolve_path

AUTHOR = "Tahereh Fahi"
EXPECTED_STAGES = {"initial_gems", "weapons", "inventory", "hud", "telemetry",
                   "chests", "chest_rewards", "inventory_reconciled", "gold_fever",
                   "status", "menu_actions", "health", "health_attribution", "gems", "release"}
REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
            "opencv-python-headless": "cv2", "easyocr": "easyocr",
            "torch": "torch", "torchvision": "torchvision"}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_scope(seconds):
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("max-seconds must be finite and positive")


def select_model_root(path):
    if "easyocr" in sys.modules:
        raise RuntimeError("Select model storage before importing EasyOCR")
    root = Path(path).expanduser().resolve()
    os.environ["EASYOCR_MODULE_PATH"] = str(root)
    return root


def model_receipt(root, initialize=False, download=False):
    import easyocr
    from easyocr import config
    if Path(config.MODULE_PATH).resolve() != root:
        raise ValueError("EasyOCR resolved a different model directory")
    if download:
        easyocr.Reader(["en"], gpu=False, download_enabled=True, verbose=False)
    specs = [config.detection_models["craft"], config.recognition_models["gen2"]["english_g2"]]
    models = []
    for spec in specs:
        path = root / "model" / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError("Missing OCR model: " + path.name + "; run prepare-models first")
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != spec["md5sum"]:
            raise ValueError("OCR model fails upstream checksum: " + path.name)
        models.append({"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if initialize and not download:
        easyocr.Reader(["en"], gpu=False, download_enabled=False, verbose=False)
    return models


def asset_receipt(config):
    paths = set()
    for detector in config["detectors"].values():
        for key in ("template_dir", "template_paths", "weapon_icon_dir", "passive_icon_dir",
                    "weapon_manifest", "passive_manifest"):
            values = detector.get(key, [])
            for value in values if isinstance(values, list) else [values]:
                if not str(value).startswith("framework:"):
                    raise ValueError("Portable configuration requires package-owned assets: " + key)
                path = resolve_path(value, Path.cwd())
                if not path.exists():
                    raise FileNotFoundError("Missing package resource: " + str(value))
                paths.update(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else paths.add(path)
    if not paths:
        raise ValueError("No detector resources configured")
    return {p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def preflight(config_path, video, output_parent, model_root, min_free_gb=0):
    if not math.isfinite(min_free_gb) or min_free_gb < 0:
        raise ValueError("min-free-gb must be finite and nonnegative")
    checks = {}
    for distribution, module in REQUIRED.items():
        imported = importlib.import_module(module)
        checks[distribution] = {"distribution": metadata.version(distribution),
                                "module": getattr(imported, "__version__", "not exposed")}
    # All OpenCV wheels use cv2; reject ambiguous installations.
    for other in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            metadata.version(other)
        except metadata.PackageNotFoundError:
            continue
        raise ValueError("Multiple OpenCV distributions installed; create a clean environment")
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if not executable:
            raise FileNotFoundError("Required system tool missing: " + name)
        result = subprocess.run([executable, "-version"], capture_output=True, text=True, check=True)
        tools[name] = result.stdout.splitlines()[0]
    config = load_runtime_config(config_path)
    if config.get("policies", {}).get("multi_run_video"):
        raise ValueError("Select a single-run configuration for this recording")
    if config.get("policies", {}).get("use_human_coded_ground_truth"):
        raise ValueError("Human-coded inputs are not allowed for extraction")
    level = config["detectors"]["gem_xp"].get("initial_level")
    if type(level) is not int or level < 1:
        raise ValueError("A positive integer initial_level is required")
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError("Video missing: " + video.name)
    observed = sha256_file(video)
    if observed != config["dataset"]["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the selected configuration")
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video cannot be decoded with a positive frame rate")
        media = {"filename": video.name, "sha256": observed, "bytes": video.stat().st_size,
                 "fps": fps, "width": frame.shape[1], "height": frame.shape[0]}
    finally:
        capture.release()
    output_parent = Path(output_parent).resolve()
    if not output_parent.is_dir():
        raise FileNotFoundError("Create the output parent directory before preflight")
    with tempfile.TemporaryFile(dir=output_parent) as probe:
        probe.write(b"write check")
    free = shutil.disk_usage(output_parent).free
    if free < min_free_gb * 1e9:
        raise ValueError("Available disk is below the requested free-space threshold")
    return {"prepared_by": AUTHOR, "status": "passed", "framework_version": __version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": checks, "tools": tools, "video": media,
            "config_sha256": sha256_file(config_path), "assets": asset_receipt(config),
            "models": model_receipt(model_root, initialize=True), "free_bytes": free,
            "implementation_sha256": package_fingerprint(),
            "capacity_sufficient_for_full_run": "not_established", "execution_device": "cpu"}


def package_fingerprint():
    entries = {p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p)
               for p in sorted(PACKAGE_ROOT.rglob("*")) if p.is_file()
               and "__pycache__" not in p.parts and (p.suffix == ".py" or "assets" in p.parts)}
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_run(directory, seconds=None):
    """Check the promised all-stage scope and every recorded output, without decoding."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "run_manifest.json").read_text())
    errors = []
    if manifest.get("status") != "complete" or manifest.get("stage_scope") != "all":
        errors.append("Run did not complete all stages")
    if manifest.get("run_scope") != ("full_video" if seconds is None else "smoke_prefix"):
        errors.append("Unexpected run scope")
    stages = manifest.get("stages", {})
    if set(stages) != EXPECTED_STAGES:
        errors.append("Unexpected stage set")
    if manifest.get("quality_failed_stages"):
        errors.append("Required quality check failed")
    for name, stage in stages.items():
        base = (directory / stage["directory"]).resolve()
        if not base.is_relative_to(directory):
            raise ValueError("Unsafe stage directory")
        if stage.get("status") != "complete" or stage.get("reused") or not stage.get("outputs"):
            errors.append("Incomplete, reused, or empty stage: " + name)
        if stage.get("quality_status") == "failed":
            errors.append("Failed quality status: " + name)
        for relative, expected in stage.get("outputs", {}).items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base) or not path.is_file() or sha256_file(path) != expected:
                errors.append("Output integrity failure: " + name + "/" + relative)
    if stages.get("hud", {}).get("quality_status") != "passed":
        errors.append("HUD quality did not pass")
    if manifest.get("publication_ready") is not False:
        errors.append("Evaluation run must not claim publication readiness")
    return {"prepared_by": AUTHOR, "status": "failed" if errors else "passed",
            "errors": errors, "analytical_accuracy": "not_established", "publication_ready": False,
            "stage_count": len(stages), "run_execution_seconds": manifest.get("execution_seconds")}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version=__version__)
    sub = result.add_subparsers(dest="command", required=True)
    models = sub.add_parser("prepare-models", help="Verify supplied weights; use --download to provision them")
    models.add_argument("--model-root", type=Path, required=True)
    models.add_argument("--download", action="store_true")
    models.add_argument("--report", type=Path, required=True)
    for verb in ("preflight", "run"):
        cmd = sub.add_parser(verb)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--video", type=Path, required=True)
        cmd.add_argument("--model-root", type=Path, required=True)
        cmd.add_argument("--min-free-gb", type=float, default=0)
        if verb == "run":
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--max-seconds", type=float)
        else:
            cmd.add_argument("--output-parent", type=Path, required=True)
            cmd.add_argument("--report", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--max-seconds", type=float)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.monotonic()
    report_path = getattr(args, "report", None)
    try:
        if report_path and report_path.exists():
            raise FileExistsError("Choose a new report path; existing reports are preserved")
        if args.command == "audit":
            validate_scope(args.max_seconds)
            report = audit_run(args.output, args.max_seconds)
        else:
            root = select_model_root(args.model_root)
            if args.command == "prepare-models":
                report = {"prepared_by": AUTHOR, "status": "passed",
                          "models": model_receipt(root, initialize=True, download=args.download)}
            else:
                if args.command == "run":
                    validate_scope(args.max_seconds)
                    output = args.output.resolve()
                    if output.exists():
                        raise FileExistsError("Choose a new output directory; evaluation runs never resume")
                    parent = output.parent
                else:
                    parent = args.output_parent
                report = preflight(args.config.resolve(), args.video, parent, root, args.min_free_gb)
                if args.command == "run":
                    output.mkdir(exist_ok=False)
                    write_report(output / "preflight.json", report)
                    report_path = output / "operator_report.json"
                    command = [sys.executable, "-m", "vss_framework.cli", "run-video",
                               "--config", str(args.config.resolve()), "--workspace-root", str(Path.cwd()),
                               "--video", str(args.video.resolve()), "--output", str(output),
                               "--stages", "all", "--no-resume"]
                    if args.max_seconds is not None:
                        command += ["--max-seconds", str(args.max_seconds)]
                    execution = time.monotonic()
                    with (output / "terminal.log").open("w") as log:
                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                   text=True, start_new_session=(os.name == "posix"))
                        try:
                            for line in process.stdout:
                                print(line, end="", flush=True)
                                log.write(line)
                            code = process.wait()
                        except BaseException:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGTERM)
                            else:
                                process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                if os.name == "posix":
                                    os.killpg(process.pid, signal.SIGKILL)
                                else:
                                    process.kill()
                                process.wait()
                            raise
                    if code:
                        raise RuntimeError("Pipeline failed; inspect terminal.log and progress.json")
                    report = audit_run(output, args.max_seconds)
                    report["invocation"] = command
                    report["command_execution_seconds"] = time.monotonic() - execution
                    after_models = model_receipt(root)
                    before = json.loads((output / "preflight.json").read_text())
                    if after_models != before["models"] or sha256_file(args.video) != before["video"]["sha256"] or sha256_file(args.config) != before["config_sha256"] or package_fingerprint() != before["implementation_sha256"]:
                        raise RuntimeError("Video, configuration, models, or package changed during execution")
        report["operator_workflow_seconds"] = time.monotonic() - started
        report["prompt_workflow_elapsed_time"] = "not recorded"
        if report_path:
            write_report(report_path, report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report = {"prepared_by": AUTHOR, "status": "failed", "error": str(exc),
                  "operator_workflow_seconds": time.monotonic() - started,
                  "prompt_workflow_elapsed_time": "not recorded"}
        if report_path and not report_path.exists():
            write_report(report_path, report)
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
