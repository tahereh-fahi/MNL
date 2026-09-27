"""Fresh matched inventory-stage comparison. Prepared by Tahereh Fahi.

Generate shared XP and weapon timeline inputs once; compare raw workers only.
No historical datasets, manual chest audits, publication or cached-result reuse.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .compare_hud_migration import PACKAGE, compare_csv, sha, write_json


PINS = {
    "external_inventory": "af78c0a81d7d9f1694f4f8e9e49ca41aacb46e1b28f0ced6c32ff60ff96e6cc1",
    "internal_inventory": "27d1148d11928b3b5b5b781b5c349d25ed9e5742fd2eedc411d4dff90abe00e3",
    "external_weapons": "46358f5314c4232e7ec186c6aedb90b863086e85230bf275d324b494802e4f47",
    "internal_weapons": "f4e16ce688d5af5da2dc2b6a1007267cc62341fd24d2fa5efafadc32b6c35021",
}


def matched_inventory_settings(external: dict, internal: dict) -> dict:
    if external["dataset"] != internal["dataset"]:
        raise ValueError("Dataset declarations differ")
    def settings(config):
        d = config["detectors"]["inventory"]
        return {"sample_fps": float(d["sample_fps"]), "video_key": d["video_key"],
                "start_second": float(d.get("start_second", 0))}
    left, right = settings(external), settings(internal)
    if left != right or left["start_second"] != 0:
        raise ValueError("Inventory settings differ or are not a zero-start prefix")
    return left


def event_summary(path: Path, end: int) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = ("event_id", "video_second", "frame_number", "character_level", "event_source",
            "event_type", "item_after", "slot", "level_after", "confidence", "needs_review")
    return {"rows": len(rows), "events": [{k: r.get(k) for k in keys} for r in rows],
            "outside_requested_prefix": [r for r in rows if not 0 <= float(r["video_second"]) < end]}


def runtime_identity() -> dict:
    return {"python": sys.version, "executable_sha256": sha(Path(sys.executable)),
            "distributions": sorted((d.metadata["Name"], d.version)
                                    for d in importlib.metadata.distributions())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 20 <= args.max_seconds <= 60:
        parser.error("Use an integer prefix length from 20 to 60 seconds")
    root, output = args.workspace_root.resolve(), args.output.resolve()
    runs = (PACKAGE.parents[1] / "runs").resolve()
    if not output.is_relative_to(runs) or output == runs or output.exists():
        parser.error("Choose a new output subdirectory inside the ongoing framework runs folder")
    if not args.execute:
        print("No processing performed. Add --execute for fresh shared inputs and two inventory runs.")
        return 0
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    report = {"prepared_by": "Tahereh Fahi", "execution_status": "running",
              "publication_ready": False, "accuracy": "not_validated", "reuse_allowed": False,
              "scope": {"start_seconds": 0, "inventory_end_seconds": args.max_seconds},
              "prompt_workflow_elapsed": "not recorded", "stages": {},
              "comparison_scope": "raw inventory workers and menu audits, not wrappers/release",
              "historical_dataset_inputs": [], "manual_chest_audit_loaded": False,
              "interpretation": "Differences require review; intentional changes are not automatically validated"}
    try:
        configs = {"external": root / "08-VSS-external-reconstructed/configs/video4.json",
                   "internal": PACKAGE.parents[1] / "configs/video4.json"}
        declared = {side: json.loads(p.read_text()) for side, p in configs.items()}
        settings = matched_inventory_settings(declared["external"], declared["internal"])
        scripts = {"external_inventory": root / "03_weapons/scripts/inventory_event_recorder.py",
                   "internal_inventory": PACKAGE / "detectors/inventory.py",
                   "external_weapons": root / "03_weapons/scripts/weapon_screen_recorder.py",
                   "internal_weapons": PACKAGE / "detectors/weapons.py"}
        for name, path in scripts.items():
            if sha(path) != PINS[name]:
                raise ValueError(name + " changed since the comparison audit; review baseline")
        assets = PACKAGE / "assets"
        # Both workers receive the exact same packaged asset paths, not just
        # corresponding directory names. Missing manifests/images fail closed.
        files = list(configs.values()) + list(scripts.values())
        files += sorted(PACKAGE.rglob("*.py"))
        for folder in ("weapon_icons", "passive_icons"):
            references = sorted(p for p in (assets / folder).rglob("*") if p.is_file())
            if not references:
                raise ValueError("Missing packaged references: " + folder)
            files += references
        files += [assets / "manifests" / name for name in
                  ("wiki_weapon_manifest.csv", "wiki_passive_item_manifest.csv")]
        video = (root / declared["internal"]["dataset"]["video"]["path"]).resolve()
        video_sha = sha(video)
        if video_sha != declared["internal"]["dataset"]["video"]["sha256"]:
            raise ValueError("Video identity mismatch")
        def code_assets():
            return {str(p.relative_to(root)): sha(p) for p in files}
        baseline = code_assets()
        runtime = runtime_identity()
        report.update(settings=settings, input_sha256=baseline, video_sha256=video_sha, runtime=runtime)
        write_json(output / "comparison_manifest.json", report)
        from .xp_level_scan import extract_xp_levels
        from .detectors.weapons import record_video_weapons, MatchConfig
        from .detectors.hud import preflight_easyocr_models
        import easyocr
        models = preflight_easyocr_models(easyocr)[1]
        report["ocr_model_weights_for_shared_xp"] = models

        print("Generating fresh shared XP/level inputs", flush=True)
        tick = time.perf_counter()
        extract_xp_levels(configs["internal"], root, output / "shared_xp", args.max_seconds)
        report["stages"]["shared_xp_seconds"] = time.perf_counter() - tick
        xp_manifest = json.loads((output / "shared_xp/manifest.json").read_text())
        if xp_manifest["execution_status"] != "complete" or xp_manifest["source"]["sha256"] != video_sha:
            raise RuntimeError("Fresh XP generation did not complete with the pinned video")
        for name, digest in xp_manifest["outputs"].items():
            if sha(output / "shared_xp" / name) != digest:
                raise RuntimeError("XP output integrity failure")

        print("Generating fresh shared weapon timeline", flush=True)
        tick = time.perf_counter()
        # Default scanner samples through nominal end and can search nearby
        # frames. Record that context honestly; this is not a clipped video.
        timeline_config = {"start_second": 0.0, "end_second": args.max_seconds,
                           "sample_interval": 5.0, "temporal_offsets": (-0.24, 0.0, 0.24),
                           "debug_every_seconds": None, "overlay_search_seconds": 5.0}
        record_video_weapons(video, output / "shared_weapons", assets / "weapon_icons",
            assets / "manifests/wiki_weapon_manifest.csv", **timeline_config)
        report["stages"]["shared_weapons_seconds"] = time.perf_counter() - tick
        report["shared_timeline_settings"] = {**timeline_config, "match_config": asdict(MatchConfig()),
            "scope_note": "Nominal prefix sampling; default gameplay search/temporal context may read beyond the nominal end"}
        xp = output / "shared_xp/xp_level_events.csv"
        timeline = output / "shared_weapons/weapon_timeline.csv"
        shared = {str(p.relative_to(output)): sha(p) for folder in (output / "shared_xp", output / "shared_weapons")
                  for p in sorted(folder.rglob("*")) if p.is_file()}
        report["shared_input_sha256"] = shared
        report["ignored_config_inputs"] = ["sources", "detectors.inventory.xp_events", "detectors.inventory.weapon_timeline"]
        def check_inputs():
            if code_assets() != baseline or runtime_identity() != runtime:
                raise RuntimeError("Local code/assets/config/runtime changed")
            if any(sha(output / name) != digest for name, digest in shared.items()):
                raise RuntimeError("Shared generated inputs changed")
        check_inputs()
        env = dict(os.environ, PYTHONPATH=str(PACKAGE.parent), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        # Probe dependency resolution in both launch contexts without running
        # either inventory detector. Scripts import only the identified helper
        # plus installed libraries; internal package Python files are all hashed.
        probe = "import json,sys,numpy,pandas,cv2; print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'pandas':pandas.__version__,'cv2':cv2.__version__}))"
        import numpy, pandas, cv2
        expected = {"python": sys.version, "numpy": numpy.__version__, "pandas": pandas.__version__, "cv2": cv2.__version__}
        for context in (root, scripts["external_inventory"].parent):
            result = subprocess.run([sys.executable, "-c", probe], cwd=context, env=env,
                                    text=True, capture_output=True, check=True)
            if json.loads(result.stdout) != expected:
                raise RuntimeError("Inventory subprocess dependency mismatch")
        report["worker_runtime_preflight"] = expected
        report["workers"] = {}
        common = ["--video", str(video), "--xp-events", str(xp), "--weapon-timeline", str(timeline),
                  "--weapon-icon-dir", str(assets / "weapon_icons"), "--passive-icon-dir", str(assets / "passive_icons"),
                  "--weapon-manifest", str(assets / "manifests/wiki_weapon_manifest.csv"),
                  "--passive-manifest", str(assets / "manifests/wiki_passive_item_manifest.csv"),
                  "--video-key", settings["video_key"], "--sample-fps", str(settings["sample_fps"]),
                  "--start", "0", "--end", str(args.max_seconds)]
        report["effective_worker_arguments"] = [str(Path(v).relative_to(root)) if v.startswith(str(root) + os.sep) else v for v in common]
        for side in ("external", "internal"):
            check_inputs()
            if sha(video) != video_sha:
                raise RuntimeError("Source video changed before " + side)
            command = [sys.executable, str(scripts["external_inventory"])] if side == "external" else [
                sys.executable, "-m", "vss_framework.detectors.inventory"]
            command += common + ["--output-dir", str(output / side)]
            print("Running " + side + " inventory with identical shared inputs", flush=True)
            tick = time.perf_counter()
            with (output / (side + ".log")).open("w") as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            report["workers"][side] = {"exit_code": result.returncode, "execution_seconds": time.perf_counter() - tick}
            if result.returncode:
                raise RuntimeError(side + " worker failed; inspect " + side + ".log")
            for name in ("inventory_events.csv", "level_up_menu_audit.csv"):
                if not (output / side / name).is_file():
                    raise RuntimeError(side + " missing " + name)
        check_inputs()
        if sha(video) != video_sha or preflight_easyocr_models(easyocr)[1] != models:
            raise RuntimeError("Source video or XP OCR models changed during comparison")
        comparisons = {}
        for name in ("inventory_events.csv", "level_up_menu_audit.csv"):
            comparisons[name] = compare_csv(output / "external" / name, output / "internal" / name)
        write_json(output / "raw_comparison.json", comparisons)
        summaries = {side: event_summary(output / side / "inventory_events.csv", args.max_seconds)
                     for side in ("external", "internal")}
        write_json(output / "event_summaries.json", summaries)
        report.update(execution_status="complete", inventory_rows_equal=comparisons["inventory_events.csv"]["equal"],
                      menu_audit_rows_equal=comparisons["level_up_menu_audit.csv"]["equal"],
                      row_counts={side: s["rows"] for side, s in summaries.items()},
                      outside_scope_counts={side: len(s["outside_requested_prefix"]) for side, s in summaries.items()},
                      interpretation_status="review_required_no_automatic_migration_verdict")
        report["output_sha256"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p != output / "comparison_manifest.json"}
        return 0
    except Exception as error:
        report.update(execution_status="failed", error_type=type(error).__name__, error=str(error))
        return 1
    finally:
        report["command_execution_seconds"] = time.perf_counter() - started
        write_json(output / "comparison_manifest.json", report)
        keys = ("execution_status", "inventory_rows_equal", "menu_audit_rows_equal", "row_counts",
                "outside_scope_counts", "stages", "workers", "publication_ready", "error",
                "command_execution_seconds", "prompt_workflow_elapsed")
        print(json.dumps({key: report[key] for key in keys if key in report}, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
