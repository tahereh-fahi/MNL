"""Fresh bounded inventory → health → attribution → export integration check.

Prepared by Tahereh Fahi. No historical datasets, cached runs or publication.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .hashing import sha256_file
from .health_provenance import implementation_receipt, verified_outputs
from .io import write_json
from .resources import PACKAGE_ROOT, asset_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=60, choices=range(20, 61))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    runs = PACKAGE_ROOT.parents[1] / "runs"
    if output == runs or not output.is_relative_to(runs) or output.exists():
        parser.error("Choose a new directory inside the ongoing framework runs folder")
    if not args.execute:
        print("No processing performed. Add --execute to generate fresh inputs and health outputs.")
        return 0
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    report = {"prepared_by": "Tahereh Fahi", "execution_status": "running",
              "publication_ready": False, "accuracy": "not_validated", "reuse_allowed": False,
              "max_seconds": args.max_seconds, "stages": {},
              "historical_dataset_inputs": [], "prompt_workflow_elapsed": "not recorded",
              "limitations": ["Bounded integration, not full-run or accuracy validation",
                  "Weapon scan context may extend beyond nominal prefix",
                  "No chest reconciliation or gem tracking; attribution uses fresh base inventory",
                  "Prefix-specific health calibration; no dashboard replacement"]}
    try:
        from .run_video import prepare_config, resource_fingerprints
        from .xp_level_scan import extract_xp_levels
        from .detectors.weapons import record_video_weapons
        from .detectors.legacy_inventory import run_legacy_inventory_detector
        from .detectors.hud import preflight_easyocr_models
        from .health_calibration import calibrate_video4_health
        from .health_attribution import attribute_video4_health
        from .health_export import build_health_payload
        import easyocr

        root = args.workspace_root.resolve()
        config = prepare_config(args.config.resolve(), root)
        baseline = implementation_receipt()
        assets = resource_fingerprints(config, root)
        model_receipt = preflight_easyocr_models(easyocr)[1]
        config_hash = sha256_file(args.config)
        video = Path(config["dataset"]["video"]["path"])
        report.update(provenance=baseline, resources=assets, ocr_models=model_receipt,
                      video_sha256=config["dataset"]["video"]["sha256"], config_sha256=config_hash)
        tracked = {}

        def check_inputs():
            if (implementation_receipt() != baseline or resource_fingerprints(config, root) != assets
                    or preflight_easyocr_models(easyocr)[1] != model_receipt
                    or sha256_file(args.config) != config_hash
                    or sha256_file(video) != report["video_sha256"]):
                raise RuntimeError("Source/code/configuration/assets/runtime/models changed")
            if any(sha256_file(output / name) != digest for name, digest in tracked.items()):
                raise RuntimeError("Generated upstream artifact changed")

        def stage(name, action):
            check_inputs()
            print("Running fresh " + name, flush=True)
            tick = time.perf_counter()
            directory = output / name
            result = action(directory)
            report["stages"][name] = {"execution_seconds": time.perf_counter() - tick}
            check_inputs()
            files = [p for p in directory.rglob("*") if p.is_file()]
            if not files:
                raise RuntimeError("Stage produced no files: " + name)
            tracked.update({str(p.relative_to(output)): sha256_file(p) for p in files})
            return directory, result

        xp, _ = stage("xp", lambda out: extract_xp_levels(args.config.resolve(), root, out, args.max_seconds))
        xm = json.loads((xp / "manifest.json").read_text())
        if xm["execution_status"] != "complete" or xm["source"]["sha256"] != report["video_sha256"]:
            raise RuntimeError("Fresh XP manifest does not match source")
        for name, digest in xm["outputs"].items():
            if sha256_file(xp / name) != digest:
                raise RuntimeError("Fresh XP output integrity mismatch")
        weapons, _ = stage("weapons", lambda out: record_video_weapons(
            video, out, asset_path("weapon_icons"), asset_path("manifests/wiki_weapon_manifest.csv"),
            end_second=args.max_seconds, debug_every_seconds=None))
        config["detectors"]["inventory"]["xp_events"] = str(xp / "xp_level_events.csv")
        config["detectors"]["inventory"]["weapon_timeline"] = str(weapons / "weapon_timeline.csv")
        runtime = output / "runtime_config.json"
        write_json(runtime, config)
        tracked[runtime.name] = sha256_file(runtime)
        inventory, _ = stage("inventory", lambda out: run_legacy_inventory_detector(
            workspace_root=root, config_path=runtime, output_dir=out, max_seconds=args.max_seconds))
        _, inventory_outputs = verified_outputs(inventory)
        health, hm = stage("health", lambda out: calibrate_video4_health(
            workspace_root=root, config_path=runtime, output_dir=out,
            sample_fps=2., max_seconds=args.max_seconds))
        attribution, am = stage("attribution", lambda out: attribute_video4_health(
            health_run_dir=health, inventory_events_path=inventory_outputs["canonical_inventory_events"], output_dir=out))
        payload = build_health_payload(health, attribution)
        export = output / "health.json"
        write_json(export, payload)
        tracked[export.name] = sha256_file(export)
        check_inputs()
        report.update(execution_status="complete", health_counts=hm["counts"],
                      recovery_attributions=am["counts"]["recovery_attributions"],
                      export="health.json", output_sha256=tracked)
        return 0
    except Exception as error:
        report.update(execution_status="failed", error_type=type(error).__name__, error=str(error))
        return 1
    finally:
        report["command_execution_seconds"] = time.perf_counter() - started
        write_json(output / "integration_manifest.json", report)
        print(json.dumps({k: report[k] for k in ("execution_status", "health_counts",
            "recovery_attributions", "export", "error", "publication_ready",
            "command_execution_seconds", "prompt_workflow_elapsed") if k in report}, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
