"""Fresh bounded gem comparison, prepared by Tahereh Fahi.

Same newly generated inventory, video, references and settings for both workers.
No historical-result reuse, manual answer input, detector edits or publication.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from .compare_hud_migration import PACKAGE, compare_csv, sha, write_json
from .compare_inventory_migration import runtime_identity


PINS = {"external": "c50730765ce7dce4d843083bec4621856c2fe54512aad6234f03c789637ef19b",
        "internal": "6abb5eca7f08d55d58b1131a02398e47c5298f00fef6fa2bfdbc8f3a3796cfe7"}
TOTAL_KEYS = ("xp_jump_events", "level_resets_detected", "collected_blue_gems",
              "collected_green_gems", "collected_red_gems", "unresolved_collected_gems",
              "collected_gems_total", "events_needing_review", "hud_level_ocr_events")


def matched_gem_settings(external: dict, internal: dict) -> dict:
    if external["dataset"] != internal["dataset"]:
        raise ValueError("Dataset declarations differ")
    keys = ("template_profile", "initial_level")
    a = {k: external["detectors"]["gem_xp"][k] for k in keys}
    b = {k: internal["detectors"]["gem_xp"][k] for k in keys}
    if a != b or a["template_profile"] != "legacy":
        raise ValueError("Use matching audited legacy-profile settings")
    return a


def config_ast(path: Path) -> str:
    tree = ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config"))


def one_event_csv(directory: Path) -> Path:
    paths = list(directory.glob("*_xp_ab_events.csv"))
    if len(paths) != 1:
        raise RuntimeError("Expected one gem event CSV in " + directory.name)
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.max_seconds <= 20:
        parser.error("Use an integer prefix from 10 to 20 seconds")
    root, output = args.workspace_root.resolve(), args.output.resolve()
    runs = (PACKAGE.parents[1] / "runs").resolve()
    if not output.is_relative_to(runs) or output == runs or output.exists():
        parser.error("Choose a new output directory inside ongoing framework runs")
    if not args.execute:
        print("No processing performed. Add --execute for fresh shared inventory and paired gem runs.")
        return 0
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    report = {"prepared_by": "Tahereh Fahi", "execution_status": "running",
              "publication_ready": False, "accuracy": "not_validated", "reuse_allowed": False,
              "prompt_workflow_elapsed": "not recorded", "workers": {},
              "scope": {"start_seconds": 0, "max_seconds": args.max_seconds},
              "historical_dataset_inputs": [], "manual_answers_loaded": False,
              "comparison_scope": "raw gems and isolated normalization, not full pipeline",
              "limitations": ["Shared inventory is freshly automated but not independently validated",
                  "Short prefix cannot validate later calibration/validation/deployment partitions",
                  "Attractorb absence cannot validate acquisition or upgrade handling",
                  "Historical external HUD traversal may still decode through the full recording",
                  "Tracking context can extend beyond nominal XP prefix; source is not clipped"]}
    try:
        configs = {"external": root / "08-VSS-external-reconstructed/configs/video4.json",
                   "internal": PACKAGE.parents[1] / "configs/video4.json"}
        declared = {s: json.loads(p.read_text()) for s, p in configs.items()}
        settings = matched_gem_settings(declared["external"], declared["internal"])
        scripts = {"external": root / "02_gems/scripts/detect_collected_gems_from_xp_ab.py",
                   "internal": PACKAGE / "detectors/gems.py"}
        for side, path in scripts.items():
            if sha(path) != PINS[side]:
                raise ValueError(side + " gem code changed since audit; review baseline")
        if config_ast(scripts["external"]) != config_ast(scripts["internal"]):
            raise ValueError("Gem Config definitions differ")
        video = (root / declared["internal"]["dataset"]["video"]["path"]).resolve()
        video_hash = sha(video)
        if video_hash != declared["internal"]["dataset"]["video"]["sha256"]:
            raise ValueError("Source video identity mismatch")
        normalizer_dir = root / "06-Pipeline/mnl_pipeline"
        files = [*configs.values(), scripts["external"], *sorted(PACKAGE.rglob("*.py")),
                 *sorted(normalizer_dir.rglob("*.py"))]
        files += sorted(p for p in (PACKAGE / "assets").rglob("*") if p.is_file())
        def fingerprints():
            return {str(p.relative_to(root)): sha(p) for p in files}
        baseline = fingerprints()
        runtime = runtime_identity()
        env = dict(os.environ, PYTHONPATH=str(PACKAGE.parent), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        from .detectors import gems
        from .detectors.hud import preflight_easyocr_models
        import easyocr
        models = preflight_easyocr_models(easyocr)[1]
        report.update(input_sha256=baseline, video_sha256=video_hash, runtime=runtime,
                      ocr_model_weights=models, settings=settings,
                      effective_config=asdict(gems.Config(initial_level=settings["initial_level"])),
                      controlled_environment={k: env[k] for k in ("PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
        # No Reader or video processing in these dependency/model probes.
        probe = ("import json,easyocr,numpy,cv2,scipy,torch,sys; "
                 "from vss_framework.detectors.hud import preflight_easyocr_models; "
                 "print(json.dumps({'python':sys.version,'numpy':numpy.__version__,"
                 "'cv2':cv2.__version__,'scipy':scipy.__version__,'torch':torch.__version__,"
                 "'models':preflight_easyocr_models(easyocr)[1]}))")
        probes = []
        for cwd in (root, scripts["external"].parent):
            p = subprocess.run([sys.executable, "-c", probe], cwd=cwd, env=env,
                               text=True, capture_output=True, check=True)
            probes.append(json.loads(p.stdout))
        if probes[0] != probes[1] or probes[0]["models"] != models:
            raise RuntimeError("Worker runtime/model resolution differs")
        report["worker_preflight"] = probes[0]
        write_json(output / "comparison_manifest.json", report)
        print("Generating fresh shared inventory (no historical input datasets)", flush=True)
        tick = time.perf_counter()
        with (output / "shared_inventory.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "vss_framework.inventory_prefix_check",
                "--workspace-root", str(root), "--config", str(configs["internal"]),
                "--output", str(output / "shared_inventory"), "--max-seconds", str(args.max_seconds)],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["shared_inventory_seconds"] = time.perf_counter() - tick
        if result.returncode:
            raise RuntimeError("Shared inventory failed; inspect shared_inventory.log")
        shared_dir = output / "shared_inventory"
        manifest = json.loads((shared_dir / "manifest.json").read_text())
        if manifest["execution_status"] != "complete" or manifest["max_seconds"] != args.max_seconds:
            raise RuntimeError("Shared inventory scope/status mismatch")
        for name, digest in manifest["outputs"].items():
            if sha(shared_dir / name) != digest:
                raise RuntimeError("Shared inventory output integrity mismatch")
        xp_manifest = json.loads((shared_dir / "xp/manifest.json").read_text())
        if xp_manifest["source"]["sha256"] != video_hash or xp_manifest["ocr_models"] != models:
            raise RuntimeError("Shared XP source/model mismatch")
        inventory = shared_dir / "inventory/inventory_events.csv"
        shared = {str(p.relative_to(output)): sha(p) for p in sorted(shared_dir.rglob("*")) if p.is_file()}
        changes = [asdict(c) for c in gems.load_attractorb_changes(inventory)]
        report.update(shared_input_sha256=shared, attractorb_changes=changes,
                      ignored_config_inputs=["sources", "detectors.gem_xp.inventory_events"],
                      shared_inventory_limitations=manifest["limitations"])
        def check_inputs():
            if fingerprints() != baseline or runtime_identity() != runtime:
                raise RuntimeError("Code/config/assets/runtime changed")
            if any(sha(output / name) != digest for name, digest in shared.items()):
                raise RuntimeError("Fresh shared inventory inputs changed")
            if sha(video) != video_hash or preflight_easyocr_models(easyocr)[1] != models:
                raise RuntimeError("Source video/OCR weights changed")
        common = ["--video", str(video), "--template-dir", str(PACKAGE / "assets/gems"),
                  "--template-profile", settings["template_profile"], "--initial-level", str(settings["initial_level"]),
                  "--inventory-events", str(inventory), "--max-seconds", str(args.max_seconds), "--no-previews"]
        report["effective_worker_arguments"] = [str(Path(v).relative_to(root)) if v.startswith(str(root) + os.sep) else v for v in common]
        summaries = {}
        for side in ("external", "internal"):
            check_inputs()
            command = [sys.executable, str(scripts[side])] if side == "external" else [sys.executable, "-m", "vss_framework.detectors.gems"]
            print("Running " + side + " gem detector with identical shared inventory", flush=True)
            tick = time.perf_counter()
            with (output / (side + ".log")).open("w") as log:
                result = subprocess.run(command + common + ["--output-dir", str(output / side)],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            report["workers"][side] = {"exit_code": result.returncode, "execution_seconds": time.perf_counter() - tick}
            if not result.returncode:
                summaries[side] = json.loads((output / side / "summary.json").read_text())
        check_inputs()
        if any(w["exit_code"] for w in report["workers"].values()):
            raise RuntimeError("At least one gem worker failed; both logs retained")
        for side, summary in summaries.items():
            if summary["attractorb_changes"] != changes or summary["duration_seconds"] != args.max_seconds:
                raise RuntimeError(side + " effective inventory or scope mismatch")
        comparisons = {}
        for pattern in ("*_xp_ab_events.csv", "*_per_second.csv", "*_5sec_intervals.csv", "*_xp_frame_signal.csv"):
            paths = {s: list((output / s).glob(pattern)) for s in summaries}
            if any(len(v) != 1 for v in paths.values()):
                raise RuntimeError("Missing/ambiguous required gem output " + pattern)
            comparisons[pattern] = compare_csv(paths["external"][0], paths["internal"][0])
        write_json(output / "raw_comparison.json", comparisons)
        # Compare isolated normalizers with matching logical administrative IDs.
        namespace = types.ModuleType("_gem_comparison_external")
        namespace.__path__ = [str(normalizer_dir)]
        sys.modules[namespace.__name__] = namespace
        external_normalizer = importlib.import_module(namespace.__name__ + ".gem_worker").normalize_gems
        from .normalization.gem_worker import normalize_gems
        dataset = declared["internal"]["dataset"]
        kwargs = dict(source_csv_relative="worker/gem_events.csv", processing_run_id="paired_gems",
                      session_id=dataset["session_id"], video_asset_id=dataset["video_asset_id"],
                      duration_ms=args.max_seconds * 1000, fps=summaries["internal"]["fps"], config={})
        normalized = {}
        for side, normalize in (("external", external_normalizer), ("internal", normalize_gems)):
            try:
                normalized[side] = normalize(source_csv=one_event_csv(output / side), **kwargs)
                write_json(output / (side + "_normalized.json"), normalized[side])
            except (ValueError, KeyError) as error:
                report.setdefault("normalization_errors", {})[side] = str(error)
        if len(normalized) == 2:
            report["normalized_own_outputs_equal"] = normalized["external"] == normalized["internal"]
            report["normalizers_on_identical_input_equal"] = normalized["external"] == normalize_gems(source_csv=one_event_csv(output / "external"), **kwargs)
        check_inputs()
        report.update(execution_status="complete", raw_events_equal=comparisons["*_xp_ab_events.csv"]["equal"],
                      comparisons_equal={p: c["equal"] for p, c in comparisons.items()},
                      totals={s: {k: v[k] for k in TOTAL_KEYS} for s, v in summaries.items()},
                      interpretation_status="review_required_no_automatic_migration_verdict")
        return 0
    except Exception as error:
        report.update(execution_status="failed", error_type=type(error).__name__, error=str(error))
        return 1
    finally:
        report["output_sha256"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p != output / "comparison_manifest.json"}
        report["command_execution_seconds"] = time.perf_counter() - started
        write_json(output / "comparison_manifest.json", report)
        keys = ("execution_status", "raw_events_equal", "comparisons_equal", "totals",
                "normalized_own_outputs_equal", "normalizers_on_identical_input_equal", "normalization_errors",
                "shared_inventory_seconds", "workers", "publication_ready", "error",
                "command_execution_seconds", "prompt_workflow_elapsed")
        print(json.dumps({k: report[k] for k in keys if k in report}, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
