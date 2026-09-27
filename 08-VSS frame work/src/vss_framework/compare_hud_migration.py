"""Bounded, fresh external/internal HUD comparison; never publishes or reuses runs.

Prepared by Tahereh Fahi. Raw detector comparison and isolated normalizer
comparison are separate from accuracy validation and full wrapper equivalence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types


PACKAGE = Path(__file__).resolve().parent
PINS = {
    "external": "3e0ea5cd2536e1aba638a68b1518b4e7e6c34c48eb46d16f759f7de890961d77",
    "internal": "e2b4ebfa3b9dc4ddaa0bbaf7041c45a0a91586585af9c74663ae01f40953556c",
}
SETTING_KEYS = (
    "sample_offsets_seconds", "min_ocr_confidence", "min_timer_observed_rate",
    "min_kill_observed_rate", "initial_kill_state", "evidence_every_seconds",
    "max_evidence_frames", "clock_qc",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def matched_settings(external: dict, internal: dict) -> dict:
    if external["dataset"] != internal["dataset"]:
        raise ValueError("Dataset declarations differ; review the comparison conditions")
    left = {k: external["detectors"]["hud_clock"][k] for k in SETTING_KEYS}
    right = {k: internal["detectors"]["hud_clock"][k] for k in SETTING_KEYS}
    if left != right:
        raise ValueError("HUD effective settings differ")
    return left


def compare_csv(left: Path, right: Path) -> dict:
    def read(path):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)
    lh, lr = read(left)
    rh, rr = read(right)
    differences = []
    for index in range(max(len(lr), len(rr))):
        a = lr[index] if index < len(lr) else None
        b = rr[index] if index < len(rr) else None
        if a != b:
            differences.append({"csv_line": index + 2, "external": a, "internal": b})
    return {"equal": lh == rh and not differences, "external_rows": len(lr),
            "internal_rows": len(rr), "headers_equal": lh == rh,
            "differences": differences}


def load_external_normalizer(directory: Path):
    # Isolate only the normalizer and its relative imports. Do not run the
    # external package's unrelated pipeline initializer or modify its files.
    name = "_hud_migration_external_normalization"
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    return importlib.import_module(name + ".hud_worker").normalize_hud


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 20:
        parser.error("Comparison scope must be an integer from 1 to 20 seconds")
    root = args.workspace_root.resolve()
    output = args.output.resolve()
    runs = PACKAGE.parents[1] / "runs"
    if not output.is_relative_to(runs.resolve()) or output == runs.resolve():
        parser.error("Output must be a new subdirectory of the ongoing framework runs folder")
    if output.exists():
        parser.error("Output exists; choose a new directory")
    if not args.execute:
        print("No processing performed. Add --execute for two fresh bounded HUD runs.")
        return 0

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    report = {"prepared_by": "Tahereh Fahi", "execution_status": "running",
              "publication_ready": False, "accuracy": "not_validated",
              "reuse_allowed": False, "prompt_workflow_elapsed": "not recorded",
              "scope": {"start_seconds": 0, "max_seconds": args.max_seconds},
              "comparison_scope": "raw HUD and isolated normalization, not full wrappers"}
    try:
        configs = {"external": root / "08-VSS-external-reconstructed/configs/video4.json",
                   "internal": PACKAGE.parents[1] / "configs/video4.json"}
        declarations = {key: json.loads(path.read_text()) for key, path in configs.items()}
        settings = matched_settings(declarations["external"], declarations["internal"])
        video = (root / declarations["internal"]["dataset"]["video"]["path"]).resolve()
        scripts = {"external": root / "01_kill_counter_and_time_stamp/scripts/extract_hud_worker.py",
                   "internal": PACKAGE / "detectors/hud.py"}
        for side, script in scripts.items():
            if sha(script) != PINS[side]:
                raise ValueError(side + " detector changed since readiness audit; review baseline")
        external_templates = scripts["external"].parents[1] / "templates"
        for name in ("skull_icon.jpg", "skull_anchor.jpg"):
            if sha(external_templates / name) != sha(PACKAGE / "assets/hud" / name):
                raise ValueError("HUD reference mismatch: " + name)

        # Capture all local package Python sources, including initializers and
        # normalizer dependencies, plus installed distribution versions.
        sources = [video, *configs.values(), scripts["external"]]
        sources += sorted(PACKAGE.rglob("*.py"))
        normalizer_dir = root / "06-Pipeline/mnl_pipeline"
        sources += sorted(normalizer_dir.rglob("*.py"))
        sources += [directory / name for directory in (external_templates, PACKAGE / "assets/hud")
                    for name in ("skull_icon.jpg", "skull_anchor.jpg")]
        def fingerprints():
            return {str(p.relative_to(root)): sha(p) for p in sources}
        identity = fingerprints()
        if identity[str(video.relative_to(root))] != declarations["internal"]["dataset"]["video"]["sha256"]:
            raise ValueError("Source video hash mismatch")
        from .detectors.hud import preflight_easyocr_models, dependency_versions
        import easyocr
        _, weights = preflight_easyocr_models(easyocr)
        def runtime():
            return {"python": sys.version, "executable_sha256": sha(Path(sys.executable)),
                    "distributions": sorted((d.metadata["Name"], d.version)
                                            for d in importlib.metadata.distributions())}
        runtime_before = runtime()
        report.update(input_sha256=identity, runtime=runtime_before,
                      easyocr_model_weights=weights, settings=settings, device="cpu",
                      normalization_context="same logical IDs and evidence path on both sides")
        write_json(output / "comparison_manifest.json", report)
        env = dict(os.environ, PYTHONPATH=str(PACKAGE.parent), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        report["controlled_environment"] = {k: env[k] for k in
            ("PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        # Inspect the actual subprocess environment before either OCR job.
        # This resolves/hashes installed models but does not construct a Reader.
        probe = subprocess.run([sys.executable, "-c",
            "import json,easyocr; from vss_framework.detectors.hud import "
            "preflight_easyocr_models,dependency_versions; "
            "print(json.dumps({'models':preflight_easyocr_models(easyocr)[1],"
            "'dependencies':dependency_versions()}))"],
            cwd=root, env=env, text=True, capture_output=True)
        if probe.returncode:
            (output / "runtime_preflight.log").write_text(probe.stderr)
            raise RuntimeError("Subprocess runtime/model preflight failed; inspect runtime_preflight.log")
        probe_identity = json.loads(probe.stdout)
        if probe_identity != {"models": weights, "dependencies": dependency_versions()}:
            raise RuntimeError("Parent and subprocess runtime/model identities differ")
        report["subprocess_preflight"] = probe_identity
        # Both workers use the same executable/environment, no --gpu, and
        # refuse missing OCR models instead of downloading replacements.
        common = ["--video", str(video), "--max-seconds", str(args.max_seconds),
                  "--sample-offsets", ",".join(str(v) for v in settings["sample_offsets_seconds"])]
        for key, flag in (("min_ocr_confidence", "min-ocr-confidence"),
                          ("min_timer_observed_rate", "min-timer-observed-rate"),
                          ("min_kill_observed_rate", "min-kill-observed-rate"),
                          ("initial_kill_state", "initial-kill-state"),
                          ("evidence_every_seconds", "evidence-every"),
                          ("max_evidence_frames", "max-evidence-frames")):
            common += ["--" + flag, str(settings[key])]
        manifests = {}
        report["workers"] = {}
        for side in ("external", "internal"):
            if fingerprints() != identity or preflight_easyocr_models(easyocr)[1] != weights:
                raise RuntimeError("Inputs/code/models changed before " + side)
            command = [sys.executable, str(scripts[side])] if side == "external" else [
                sys.executable, "-m", "vss_framework.detectors.hud"]
            command += common + ["--output-dir", str(output / side)]
            print("Running " + side + " HUD (maximum 20 seconds of video)", flush=True)
            tick = time.perf_counter()
            with (output / (side + ".log")).open("w") as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            report["workers"][side] = {"exit_code": result.returncode,
                "execution_seconds": time.perf_counter() - tick}
            if result.returncode not in (0, 5):
                raise RuntimeError(side + " worker failed; inspect its log")
            manifest = json.loads((output / side / "manifest.json").read_text())
            manifests[side] = manifest
            if manifest["input_video"]["sha256"] != identity[str(video.relative_to(root))]:
                raise RuntimeError(side + " worker source mismatch")
            if manifest["worker"]["code_sha256"] != PINS[side]:
                raise RuntimeError(side + " worker code mismatch")
            if manifest["easyocr_model_weights"] != weights:
                raise RuntimeError(side + " worker OCR weights mismatch")
            expected_deps = dependency_versions()
            if any(manifest["dependencies"].get(k) != v for k, v in expected_deps.items()):
                raise RuntimeError(side + " worker dependency mismatch")
            for artifact in manifest["outputs"].values():
                path = (output / side / artifact["path"]).resolve()
                if not path.is_relative_to(output / side) or sha(path) != artifact["sha256"]:
                    raise RuntimeError(side + " output integrity mismatch")
            qc = json.loads((output / side / "qc.json").read_text())
            report["workers"][side]["qc_status"] = qc["status"]
            if (result.returncode == 5) != (qc["status"] == "failed"):
                raise RuntimeError("Worker exit/QC status disagreement")
        if fingerprints() != identity or runtime() != runtime_before or preflight_easyocr_models(easyocr)[1] != weights:
            raise RuntimeError("Source/code/runtime/models changed during paired run")
        for key in ("input_video", "configuration", "dependencies", "easyocr_model_weights"):
            if manifests["external"][key] != manifests["internal"][key]:
                raise RuntimeError("Workers did not report matching " + key)
        if manifests["internal"]["configuration"]["processed_seconds"] != args.max_seconds:
            raise RuntimeError("Requested prefix was not fully processed")
        raw = compare_csv(output / "external/hud_observations.csv", output / "internal/hud_observations.csv")
        write_json(output / "raw_comparison.json", raw)
        qc_equal = (json.loads((output / "external/qc.json").read_text()) ==
                    json.loads((output / "internal/qc.json").read_text()))
        candidates_equal = ((output / "external/ocr_candidates.jsonl").read_bytes() ==
                            (output / "internal/ocr_candidates.jsonl").read_bytes())
        from .normalization.hud_worker import normalize_hud
        normalizers = {"external": load_external_normalizer(normalizer_dir), "internal": normalize_hud}
        dataset = declarations["internal"]["dataset"]
        normalized = {}
        for side, normalize in normalizers.items():
            normalized[side] = normalize(
                source_csv=output / side / "hud_observations.csv",
                source_csv_relative="worker/hud_observations.csv", processing_run_id="paired_hud",
                session_id=dataset["session_id"], video_asset_id=dataset["video_asset_id"],
                duration_ms=min(dataset["duration_ms"], args.max_seconds * 1000),
                config={"source_sampling_offsets_seconds": settings["sample_offsets_seconds"],
                        "clock_qc": settings["clock_qc"]})
            write_json(output / (side + "_normalized.json"), normalized[side])
        # Isolate normalizer code even if detector observations differ.
        shared = dict(source_csv=output / "external/hud_observations.csv",
                      source_csv_relative="worker/hud_observations.csv", processing_run_id="paired_hud",
                      session_id=dataset["session_id"], video_asset_id=dataset["video_asset_id"],
                      duration_ms=min(dataset["duration_ms"], args.max_seconds * 1000),
                      config={"source_sampling_offsets_seconds": settings["sample_offsets_seconds"],
                              "clock_qc": settings["clock_qc"]})
        report.update(execution_status="complete", raw_observations_equal=raw["equal"],
                      qc_equal=qc_equal, ocr_candidates_byte_equal=candidates_equal,
                      normalized_own_outputs_equal=normalized["external"] == normalized["internal"],
                      normalizers_on_identical_input_equal=normalized["external"] == normalize_hud(**shared))
        if fingerprints() != identity:
            raise RuntimeError("Inputs changed during normalization")
        report["output_sha256"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p.name != "comparison_manifest.json"}
        return 0
    except Exception as error:
        report.update(execution_status="failed", error_type=type(error).__name__, error=str(error))
        return 1
    finally:
        report["command_execution_seconds"] = time.perf_counter() - started
        write_json(output / "comparison_manifest.json", report)
        keys = ("execution_status", "raw_observations_equal", "normalized_own_outputs_equal",
                "normalizers_on_identical_input_equal", "qc_equal", "workers", "publication_ready",
                "error", "command_execution_seconds", "prompt_workflow_elapsed")
        print(json.dumps({k: report[k] for k in keys if k in report}, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
