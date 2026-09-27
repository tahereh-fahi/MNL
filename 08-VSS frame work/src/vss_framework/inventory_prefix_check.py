"""Fresh XP-to-inventory diagnostic; no historical datasets or publication."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from .xp_level_scan import code_hashes, extract_xp_levels, sha256, validate_scope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=20)
    args = parser.parse_args()
    validate_scope(args.max_seconds)
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(prepared_by="Tahereh Fahi", execution_status="running",
                    publication_ready=False, detector_accuracy="not_validated",
                    prompt_workflow_elapsed="not recorded", reuse_allowed=False,
                    max_seconds=args.max_seconds, sample_fps=10.0,
                    limitations=["No weapon-timeline/evolution reconciliation",
                                 "No historical datasets or manual chest audit loaded",
                                 "Not validation of all in-game menu classifications"])
    try:
        xp_dir = args.output / "xp"
        extract_xp_levels(args.config, args.workspace_root, xp_dir, args.max_seconds)
        upstream = json.loads((xp_dir / "manifest.json").read_text())
        config = json.loads(args.config.read_text())
        from .detectors.inventory import record_inventory_events
        assets = Path(__file__).resolve().parent / "assets"
        video = args.workspace_root / config["dataset"]["video"]["path"]
        xp_csv = xp_dir / "xp_level_events.csv"
        inputs = {"config": args.config, "video": video, "xp_csv": xp_csv,
                  "xp_manifest": xp_dir / "manifest.json"}
        for folder in ("weapon_icons", "passive_icons"):
            inputs.update({str(p.relative_to(assets)): p for p in
                           (assets / folder).rglob("*") if p.is_file()})
        for name in ("wiki_weapon_manifest.csv", "wiki_passive_item_manifest.csv"):
            inputs[name] = assets / "manifests" / name
        hashes = {name: sha256(path) for name, path in inputs.items()}
        if (hashes["video"] != upstream["source"]["sha256"]
                or hashes["config"] != upstream["config_sha256"]
                or hashes["xp_csv"] != upstream["outputs"][xp_csv.name]
                or code_hashes() != upstream["code_sha256"]):
            raise RuntimeError("Upstream evidence changed before inventory extraction")
        manifest["input_sha256"] = hashes
        manifest["code_sha256"] = upstream["code_sha256"]
        inventory_dir = args.output / "inventory"
        inventory_dir.mkdir()
        events, audit = record_inventory_events(
            video_path=video, xp_events_path=xp_csv, output_dir=inventory_dir,
            weapon_icon_dir=assets / "weapon_icons", passive_icon_dir=assets / "passive_icons",
            weapon_manifest_path=assets / "manifests/wiki_weapon_manifest.csv",
            passive_manifest_path=assets / "manifests/wiki_passive_item_manifest.csv",
            video_key=config["dataset"]["video_asset_id"], end_second=args.max_seconds,
            sample_fps=10.0,
        )
        if (code_hashes() != upstream["code_sha256"] or
                any(sha256(path) != hashes[name] for name, path in inputs.items())):
            raise RuntimeError("Inputs changed during inventory extraction")
        manifest["outputs"] = {str(p.relative_to(args.output)): sha256(p) for p in (events, audit)}
        manifest["execution_status"] = "complete"
    except BaseException as error:
        manifest.update(execution_status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["command_execution_seconds"] = time.perf_counter() - started
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({key: manifest[key] for key in (
            "execution_status", "publication_ready", "command_execution_seconds", "prompt_workflow_elapsed")}, indent=2))


if __name__ == "__main__":
    main()
