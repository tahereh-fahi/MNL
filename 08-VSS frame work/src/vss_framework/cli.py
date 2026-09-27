"""Command-line entry point for VSS framework runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import load_event_catalog
from .direct_scan import scan_video4_interval
from .detectors import (
    run_legacy_gem_xp_detector,
    run_legacy_hud_detector,
    run_legacy_inventory_detector,
)
from .pipeline import build_video4_framework_run
from .telemetry_scan import scan_video4_telemetry
from .health_calibration import calibrate_video4_health
from .health_attribution import attribute_video4_health
from .world_pickups import calibrate_video4_world_pickups, scan_video4_world_pickups
from .world_pickup_effects import resolve_video4_world_pickup_effects
from .freeze_effect import scan_video4_freeze_effect
from .vacuum_flow import scan_video4_vacuum_flow
from .gold_fever import scan_video4_gold_fever
from .chest_lifecycle import scan_video4_chests
from .chest_rewards import identify_chest_rewards
from .menu_actions import scan_video4_menu_actions
from .status_events import scan_video4_status_events
from .release import compile_video4_release
from .gold_series import build_gold_series
from .resources import asset_path


def _project_root() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "pyproject.toml").is_file() else Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    project_root = _project_root()
    parser = argparse.ArgumentParser(prog="vss-framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    xp = subparsers.add_parser("extract-xp-levels", help="Fresh video-only XP/level evidence; no gem tracking or publication")
    xp.add_argument("--config", type=Path, required=True)
    xp.add_argument("--workspace-root", type=Path, default=project_root.parent)
    xp.add_argument("--video", type=Path)
    xp.add_argument("--output", type=Path, required=True)
    xp.add_argument("--max-seconds", type=float, required=True)

    foundations = subparsers.add_parser("validate-foundations", help="Isolated source identity audit and sparse screenshots; never publishes")
    foundations.add_argument("--workspace-root", type=Path, default=project_root.parent)
    foundations.add_argument("--configs", type=Path, default=project_root / "configs")
    foundations.add_argument("--videos-dir", type=Path, help="Default: WORKSPACE/00-Videos")
    foundations.add_argument("--output", type=Path, required=True)
    foundations.add_argument("--stage", choices=("registry", "evidence"), default="registry")
    foundations.add_argument("--video", action="append", help="Evidence only: exact source-relative path or unambiguous basename; repeatable")
    foundations.add_argument("--at-second", type=float, action="append", help="Evidence only: source-video seconds, not game-clock time; repeatable")
    foundations.add_argument("--no-resume", action="store_true", help="Regenerate evidence even when its content hashes match")

    run = subparsers.add_parser("run-video", help="Extract events from video using only framework-owned detectors and assets")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--workspace-root", type=Path, default=project_root.parent)
    run.add_argument("--video", type=Path, help="Explicit renamed source file; must match the configured SHA-256")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-seconds", type=float)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--stages", choices=("all", "inventory-gems"), default="all")

    catalog = subparsers.add_parser("check-catalog", help="Validate Event Catalog v1")
    catalog.add_argument(
        "--catalog", type=Path, default=asset_path("event_catalog.json")
    )

    video4 = subparsers.add_parser(
        "bootstrap-video4", help="Build the first framework run from existing Video 4 artifacts"
    )
    video4.add_argument("--workspace-root", type=Path, default=project_root.parent)
    video4.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    video4.add_argument(
        "--catalog", type=Path, default=asset_path("event_catalog.json")
    )
    video4.add_argument(
        "--output", type=Path, default=project_root / "runs/video4_bootstrap"
    )

    scan = subparsers.add_parser(
        "scan-video4", aliases=["scan-frames"], help="Run internal frame-level detectors on a configured video interval"
    )
    scan.add_argument("--workspace-root", type=Path, default=project_root.parent)
    scan.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    scan.add_argument("--output", type=Path, default=project_root / "runs/video4_direct_scan")
    scan.add_argument("--start-second", type=float, default=0.0)
    scan.add_argument("--end-second", type=float)
    scan.add_argument("--sample-fps", type=float, default=2.0)
    scan.add_argument("--include-cached-signals", action="store_true", help="Explicitly import pinned historical signal artifacts as well as scanning frames")
    scan.add_argument(
        "--skip-video-hash",
        action="store_true",
        help="Skip the expensive video hash check and record that verification was not performed",
    )

    gem_xp = subparsers.add_parser(
        "run-video4-gem-xp",
        help="Run the existing full XP/gem detector unchanged under framework control",
    )
    gem_xp.add_argument("--workspace-root", type=Path, default=project_root.parent)
    gem_xp.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    gem_xp.add_argument(
        "--output", type=Path, default=project_root / "runs/video4_gem_xp"
    )
    gem_xp.add_argument(
        "--python", type=Path, help="Python environment containing the existing detector dependencies"
    )
    gem_xp.add_argument(
        "--max-seconds",
        type=float,
        help="Development prefix; omit to run the complete video",
    )
    gem_xp.add_argument("--save-previews", action="store_true")
    gem_xp.add_argument(
        "--reuse-worker",
        action="store_true",
        help="Reuse completed detector files in the output worker directory and rerun normalization only",
    )

    hud = subparsers.add_parser(
        "run-video4-hud",
        help="Run the existing Kill Counter/Game Clock worker unchanged under framework control",
    )
    hud.add_argument("--workspace-root", type=Path, default=project_root.parent)
    hud.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    hud.add_argument(
        "--output", type=Path, default=project_root / "runs/video4_hud"
    )
    hud.add_argument(
        "--python", type=Path, help="Python environment containing the existing OCR dependencies"
    )
    hud.add_argument(
        "--max-seconds",
        type=int,
        help="Development prefix; omit to process the complete video",
    )

    inventory = subparsers.add_parser(
        "run-video4-inventory",
        help="Run the existing automated Level-Up/inventory recorder under framework control",
    )
    inventory.add_argument("--workspace-root", type=Path, default=project_root.parent)
    inventory.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    inventory.add_argument(
        "--output", type=Path, default=project_root / "runs/video4_inventory"
    )
    inventory.add_argument("--python", type=Path)
    inventory.add_argument(
        "--max-seconds", type=float, help="Development prefix; omit for the complete video"
    )

    telemetry = subparsers.add_parser(
        "run-video4-telemetry",
        help="Measure Coin Counter and raw player health-bar width directly from Video 4",
    )
    telemetry.add_argument("--workspace-root", type=Path, default=project_root.parent)
    telemetry.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    telemetry.add_argument("--output", type=Path, default=project_root / "runs/video4_telemetry")
    telemetry.add_argument("--start-second", type=float, default=0.0)
    telemetry.add_argument("--end-second", "--max-seconds", dest="end_second", type=float)
    telemetry.add_argument("--sample-fps", type=float, default=1.0)

    health = subparsers.add_parser(
        "calibrate-video4-health",
        help="Self-calibrate player health percentage from the complete Video 4",
    )
    health.add_argument("--workspace-root", type=Path, default=project_root.parent)
    health.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    health.add_argument("--output", type=Path, default=project_root / "runs/video4_health_calibration")
    health.add_argument("--sample-fps", type=float, default=2.0)
    health.add_argument("--max-seconds", type=float, help="Bounded diagnostic; calibration is prefix-specific")

    attribution = subparsers.add_parser(
        "attribute-video4-health",
        help="Attribute Video 4 recovery events using automated reward and inventory evidence",
    )
    attribution.add_argument("--health-run", type=Path, default=project_root / "runs/video4_health_events_v3")
    attribution.add_argument("--inventory-events", type=Path, required=True, help="Explicit canonical inventory output with an intact producer manifest")
    attribution.add_argument("--output", type=Path, default=project_root / "runs/video4_health_attribution")

    world = subparsers.add_parser(
        "run-video4-world-pickups",
        help="Track visible world pickups and emit conservative pickup candidates",
    )
    world.add_argument("--workspace-root", type=Path, default=project_root.parent)
    world.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    world.add_argument("--assets", type=Path, default=asset_path("world_pickups"))
    world.add_argument("--output", type=Path, default=project_root / "runs/video4_world_pickups")
    world.add_argument("--start-second", type=float, default=0.0)
    world.add_argument("--end-second", type=float)
    world.add_argument("--sample-fps", type=float, default=2.0)
    world.add_argument("--threshold", type=float, default=0.86)
    world.add_argument("--calibration", type=Path, default=asset_path("world_pickups/calibration_video4.json"))

    world_calibration = subparsers.add_parser(
        "calibrate-video4-world-pickups",
        help="Self-calibrate sprite thresholds without Human-coded labels",
    )
    world_calibration.add_argument("--workspace-root", type=Path, default=project_root.parent)
    world_calibration.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    world_calibration.add_argument("--assets", type=Path, default=asset_path("world_pickups"))
    world_calibration.add_argument("--output", type=Path, default=project_root / "runs/world_pickup_calibration.json")

    effects = subparsers.add_parser("resolve-video4-world-pickup-effects",
        help="Corroborate pickup candidates with direct Health and Coin Counter effects")
    effects.add_argument("--health-events", type=Path, default=project_root / "runs/video4_health_events_v3/health_events.jsonl")
    effects.add_argument("--sprite-candidates", type=Path, default=project_root / "runs/video4_world_pickups_calibrated_v2/world_pickup_candidates.jsonl")
    effects.add_argument("--telemetry", type=Path)
    effects.add_argument("--output", type=Path, default=project_root / "runs/video4_world_pickup_effects")

    freeze = subparsers.add_parser("run-video4-freeze-effect",
        help="Detect freeze-like intervals compatible with Orologion")
    freeze.add_argument("--workspace-root", type=Path, default=project_root.parent)
    freeze.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    freeze.add_argument("--output", type=Path, default=project_root / "runs/video4_freeze_effect")
    freeze.add_argument("--sample-fps", type=float, default=4.0)
    freeze.add_argument("--baseline-seconds", type=float, default=8.0)

    vacuum = subparsers.add_parser("run-video4-vacuum-flow",
        help="Detect mass radial gem flow compatible with Vacuum")
    vacuum.add_argument("--workspace-root", type=Path, default=project_root.parent)
    vacuum.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    vacuum.add_argument("--output", type=Path, default=project_root / "runs/video4_vacuum_flow")
    vacuum.add_argument("--sample-fps", type=float, default=5.0)
    vacuum.add_argument("--start-second", type=float, default=0.0)
    vacuum.add_argument("--end-second", type=float)

    gold_fever = subparsers.add_parser("run-video4-gold-fever",
        help="Detect persistent Gold Fever HUD intervals")
    gold_fever.add_argument("--workspace-root", type=Path, default=project_root.parent)
    gold_fever.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    gold_fever.add_argument("--output", type=Path, default=project_root / "runs/video4_gold_fever")
    gold_fever.add_argument("--sample-fps", type=float, default=4.0)
    gold_fever.add_argument("--start-second", type=float, default=0.0)
    gold_fever.add_argument("--end-second", type=float)

    chests = subparsers.add_parser("run-video4-chests",
        help="Detect Treasure Chest animation intervals and visible reward tiers")
    chests.add_argument("--workspace-root", type=Path, default=project_root.parent)
    chests.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    chests.add_argument("--output", type=Path, default=project_root / "runs/video4_chests")
    chests.add_argument("--sample-fps", type=float, default=4.0)
    chests.add_argument("--start-second", type=float, default=0.0)
    chests.add_argument("--end-second", type=float)

    rewards = subparsers.add_parser("identify-chest-rewards",
        aliases=["identify-video4-chest-rewards"],
        help="Identify chest reward sprites using prior Automated inventory state")
    rewards.add_argument("--workspace-root", type=Path, default=project_root.parent)
    rewards.add_argument("--config", type=Path, default=project_root / "configs/video4.json")
    rewards.add_argument("--chest-events", type=Path, default=project_root / "runs/video4_chests_v2/chest_events.jsonl")
    rewards.add_argument("--automated-inventory", type=Path, default=project_root / "runs/video4_levelup_transactions_complete/canonical_inventory_events.jsonl")
    rewards.add_argument("--output", type=Path, default=project_root / "runs/video4_chest_rewards")

    actions=subparsers.add_parser("run-video4-menu-actions",help="Detect Reroll, Skip, and Banish counter decrements")
    actions.add_argument("--workspace-root",type=Path,default=project_root.parent)
    actions.add_argument("--config",type=Path,default=project_root/"configs/video4.json")
    actions.add_argument("--menu-audit",type=Path,default=project_root/"runs/video4_levelup_transactions_complete/worker/level_up_menu_audit.csv")
    actions.add_argument("--output",type=Path,default=project_root/"runs/video4_menu_actions")

    status=subparsers.add_parser("run-video4-status-events",help="Detect pre-game, death, results, and achievement screens")
    status.add_argument("--workspace-root",type=Path,default=project_root.parent)
    status.add_argument("--config",type=Path,default=project_root/"configs/video4.json")
    status.add_argument("--output",type=Path,default=project_root/"runs/video4_status_events")
    status.add_argument("--sample-fps",type=float,default=2.0)

    release=subparsers.add_parser("compile-video4-release",help="Compile every completed Automated stage into one release")
    release.add_argument("--config",type=Path,default=project_root/"configs/video4.json")
    release.add_argument("--catalog",type=Path,default=project_root/"configs/event_catalog.json")
    release.add_argument("--output",type=Path,default=project_root/"runs/video4_complete_release")
    release.add_argument("--dashboard-output",type=Path)

    gold_series = subparsers.add_parser(
        "build-gold-series",
        help="Publish corroborated cumulative Gold observations for configured datasets",
    )
    gold_series.add_argument("--runs-root", type=Path, default=project_root / "runs")
    gold_series.add_argument(
        "--config", type=Path, default=project_root / "configs/dashboard_gold_series.json"
    )
    gold_series.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract-xp-levels":
        from .xp_level_scan import extract_xp_levels

        result = extract_xp_levels(args.config, args.workspace_root, args.output,
                                   args.max_seconds, args.video)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "validate-foundations":
        from .foundation_validation import validate_foundations

        result = validate_foundations(
            workspace=args.workspace_root, config_dir=args.configs,
            videos_dir=args.videos_dir or args.workspace_root / "00-Videos",
            output=args.output, stage=args.stage, videos=args.video,
            at_seconds=args.at_second, resume=not args.no_resume,
        )
        print(json.dumps(result["registry_summary"], indent=2))
        return 0 if result["execution_status"] == "complete" else 1
    if args.command == "run-video":
        from .run_video import run_video
        manifest = run_video(config_path=args.config.resolve(), workspace_root=args.workspace_root.resolve(),
            output_dir=args.output.resolve(), video_override=args.video, max_seconds=args.max_seconds,
            resume=not args.no_resume, stages=args.stages)
        print(json.dumps({"status": manifest["status"], "scope": manifest["run_scope"],
            "execution_seconds": manifest["execution_seconds"], "stages": list(manifest["stages"])}, indent=2))
        return 0
    if args.command == "check-catalog":
        catalog = load_event_catalog(args.catalog.resolve())
        print(json.dumps({"status": "ok", "event_count": len(catalog["events"])}, indent=2))
        return 0

    if args.command == "build-gold-series":
        payload = build_gold_series(
            runs_root=args.runs_root.resolve(),
            config_path=args.config.resolve(),
            output_path=args.output.resolve(),
        )
        print(json.dumps({"status": "ok", "datasets": len(payload["datasets"]), "output": str(args.output.resolve())}, indent=2))
        return 0

    if args.command in {"scan-video4", "scan-frames"}:
        manifest = scan_video4_interval(
            workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
            start_ms=round(args.start_second * 1000),
            end_ms=None if args.end_second is None else round(args.end_second * 1000),
            sample_fps=args.sample_fps,
            verify_video_hash=not args.skip_video_hash,
            include_cached_signals=args.include_cached_signals,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "processing_run_id": manifest["processing_run_id"],
                    "frame_observations": manifest["counts"]["frame_observations"],
                    "signal_observations": manifest["counts"]["signal_observations"],
                    "by_state": manifest["counts"]["by_state"],
                    "signals_by_observable": manifest["counts"]["signals_by_observable"],
                    "manifest": manifest["manifest_path"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-video4-gem-xp":
        manifest = run_legacy_gem_xp_detector(
            workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
            python_executable=(
                args.python.absolute() if args.python is not None else None
            ),
            max_seconds=args.max_seconds,
            save_previews=args.save_previews,
            reuse_worker=args.reuse_worker,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "processing_run_id": manifest["processing_run_id"],
                    "xp_events": manifest["counts"]["xp_events"],
                    "gem_quantity_total": manifest["counts"]["gem_quantity_total"],
                    "gem_quantity_by_type": manifest["counts"]["gem_quantity_by_type"],
                    "manifest": manifest["manifest_path"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-video4-hud":
        manifest = run_legacy_hud_detector(
            workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
            python_executable=(
                args.python.absolute() if args.python is not None else None
            ),
            max_seconds=args.max_seconds,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "processing_run_id": manifest["processing_run_id"],
                    "worker_qc_status": manifest["worker_qc_status"],
                    "signals_by_observable": manifest["counts"]["signals_by_observable"],
                    "observed_kill_counter_signals": manifest["counts"][
                        "observed_kill_counter_signals"
                    ],
                    "carried_or_assumed_kill_counter_signals": manifest["counts"][
                        "carried_or_assumed_kill_counter_signals"
                    ],
                    "final_kill_counter": manifest["counts"]["final_kill_counter"],
                    "clock_issues": manifest["counts"]["clock_issues"],
                    "manifest": manifest["manifest_path"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-video4-inventory":
        manifest = run_legacy_inventory_detector(
            workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
            python_executable=args.python.absolute() if args.python is not None else None,
            max_seconds=args.max_seconds,
        )
        print(json.dumps({
            "status": "ok",
            "processing_run_id": manifest["processing_run_id"],
            "canonical_events": manifest["counts"]["canonical_events"],
            "by_event_type": manifest["counts"]["by_event_type"],
            "by_publication_status": manifest["counts"]["by_publication_status"],
            "level_up_transactions": manifest["counts"]["level_up_transactions"],
            "resolved_level_up_transactions": manifest["counts"][
                "resolved_level_up_transactions"
            ],
            "unresolved_level_up_transactions": manifest["counts"][
                "unresolved_level_up_transactions"
            ],
            "instant_reward_events": manifest["counts"]["instant_reward_events"],
            "manifest": manifest["manifest_path"],
        }, indent=2))
        return 0

    if args.command == "run-video4-telemetry":
        manifest = scan_video4_telemetry(
            workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
            start_second=args.start_second,
            max_seconds=args.end_second,
            sample_fps=args.sample_fps,
        )
        print(json.dumps({
            "status": "ok",
            "processing_run_id": manifest["processing_run_id"],
            "observations": manifest["counts"]["observations"],
            "observed_by_code": manifest["counts"]["observed_by_code"],
            "manifest": manifest["manifest_path"],
        }, indent=2))
        return 0

    if args.command == "calibrate-video4-health":
        manifest = calibrate_video4_health(
            workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
            sample_fps=args.sample_fps,
            max_seconds=args.max_seconds,
        )
        print(json.dumps({
            "status": "ok",
            "processing_run_id": manifest["processing_run_id"],
            "full_bar_reference_width_1440p_px": manifest["calibration"]["full_bar_reference_width_1440p_px"],
            "counts": manifest["counts"],
            "manifest": manifest["manifest_path"],
            "publication_ready": False,
            "execution_seconds": manifest["execution_seconds"],
            "prompt_workflow_elapsed_seconds": None,
        }, indent=2))
        return 0

    if args.command == "attribute-video4-health":
        manifest = attribute_video4_health(
            health_run_dir=args.health_run.resolve(),
            inventory_events_path=args.inventory_events.resolve(),
            output_dir=args.output.resolve(),
        )
        print(json.dumps({"status": "ok", "counts": manifest["counts"], "manifest": manifest["manifest_path"]}, indent=2))
        return 0

    if args.command == "run-video4-world-pickups":
        manifest = scan_video4_world_pickups(
            workspace_root=args.workspace_root.resolve(), config_path=args.config.resolve(),
            asset_dir=args.assets.resolve(), output_dir=args.output.resolve(),
            start_second=args.start_second, end_second=args.end_second,
            sample_fps=args.sample_fps, threshold=args.threshold,
            calibration_path=args.calibration.resolve() if args.calibration is not None else None,
        )
        print(json.dumps({"status": "ok", "counts": manifest["counts"], "manifest": manifest["manifest_path"]}, indent=2))
        return 0

    if args.command == "calibrate-video4-world-pickups":
        calibration = calibrate_video4_world_pickups(
            workspace_root=args.workspace_root.resolve(), config_path=args.config.resolve(),
            asset_dir=args.assets.resolve(), output_path=args.output.resolve(),
        )
        print(json.dumps({"status": "ok", "counts": calibration["counts"], "output": str(args.output.resolve())}, indent=2))
        return 0

    if args.command == "resolve-video4-world-pickup-effects":
        manifest = resolve_video4_world_pickup_effects(
            health_events_path=args.health_events.resolve(), sprite_candidates_path=args.sprite_candidates.resolve(),
            telemetry_path=args.telemetry.resolve() if args.telemetry is not None else None,
            output_dir=args.output.resolve())
        print(json.dumps({"status": "ok", "counts": manifest["counts"], "manifest": manifest["manifest_path"]}, indent=2))
        return 0

    if args.command == "run-video4-freeze-effect":
        manifest = scan_video4_freeze_effect(workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(), output_dir=args.output.resolve(),
            sample_fps=args.sample_fps, baseline_seconds=args.baseline_seconds)
        print(json.dumps({"status": "ok", "counts": manifest["counts"], "manifest": manifest["manifest_path"]}, indent=2))
        return 0

    if args.command == "run-video4-vacuum-flow":
        manifest = scan_video4_vacuum_flow(workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(), output_dir=args.output.resolve(), sample_fps=args.sample_fps,
            start_second=args.start_second, end_second=args.end_second)
        print(json.dumps({"status": "ok", "counts": manifest["counts"], "manifest": manifest["manifest_path"]}, indent=2))
        return 0

    if args.command == "run-video4-gold-fever":
        manifest = scan_video4_gold_fever(workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(), output_dir=args.output.resolve(), sample_fps=args.sample_fps,
            start_second=args.start_second, end_second=args.end_second)
        print(json.dumps({"status": "ok", "counts": manifest["counts"], "manifest": manifest["manifest_path"]}, indent=2))
        return 0

    if args.command == "run-video4-chests":
        manifest = scan_video4_chests(workspace_root=args.workspace_root.resolve(),
            config_path=args.config.resolve(), output_dir=args.output.resolve(), sample_fps=args.sample_fps,
            start_second=args.start_second, end_second=args.end_second)
        print(json.dumps({"status": "ok", "counts": manifest["counts"], "manifest": manifest["manifest_path"]}, indent=2))
        return 0

    if args.command in {"identify-chest-rewards", "identify-video4-chest-rewards"}:
        manifest=identify_chest_rewards(workspace_root=args.workspace_root.resolve(),config_path=args.config.resolve(),
            chest_events_path=args.chest_events.resolve(),automated_inventory_path=args.automated_inventory.resolve(),output_dir=args.output.resolve())
        print(json.dumps({"status":"ok","counts":manifest["counts"],"manifest":manifest["manifest_path"]},indent=2)); return 0

    if args.command == "run-video4-menu-actions":
        manifest=scan_video4_menu_actions(workspace_root=args.workspace_root.resolve(),config_path=args.config.resolve(),
            menu_audit_path=args.menu_audit.resolve(),output_dir=args.output.resolve())
        print(json.dumps({"status":"ok","counts":manifest["counts"],"manifest":manifest["manifest_path"]},indent=2)); return 0

    if args.command == "run-video4-status-events":
        manifest=scan_video4_status_events(workspace_root=args.workspace_root.resolve(),config_path=args.config.resolve(),
            output_dir=args.output.resolve(),sample_fps=args.sample_fps)
        print(json.dumps({"status":"ok","counts":manifest["counts"],"manifest":manifest["manifest_path"]},indent=2)); return 0

    if args.command == "compile-video4-release":
        manifest=compile_video4_release(project_root=_project_root(),config_path=args.config.resolve(),catalog_path=args.catalog.resolve(),
            output_dir=args.output.resolve(),dashboard_output=args.dashboard_output.resolve() if args.dashboard_output else None)
        print(json.dumps({"status":"ok","counts":manifest["counts"],"manifest":manifest["manifest_path"]},indent=2)); return 0

    manifest = build_video4_framework_run(
        workspace_root=args.workspace_root.resolve(),
        config_path=args.config.resolve(),
        catalog_path=args.catalog.resolve(),
        output_dir=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "processing_run_id": manifest["processing_run_id"],
                "canonical_events": manifest["counts"]["canonical_events"],
                "derived_five_second_windows": manifest["counts"]["derived_five_second_windows"],
                "manifest": manifest["manifest_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
