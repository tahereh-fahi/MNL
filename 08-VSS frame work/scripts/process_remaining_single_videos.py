#!/usr/bin/env python3
"""Compatibility entry point for the framework-owned video pipeline.

Prepared by Tahereh Fahi.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from vss_framework.run_video import run_video

VIDEOS = {
    "video5_imelda_0_part1": "video5_Imelda_0_part1.mp4",
    "video5_imelda_0_part2": "video5_Imelda_0_part2.mp4",
    "imelda_powerups_highluck": "Imelda_PowerUps_HighLuck.mp4",
    "sigma_bossrash_powerups_only": "Sigma_BossRash_PowerupsOnly.mp4",
    "sigma_bossrash_powerups_highluck": "Sigma_BossRash_Powerups_HighLuck.mp4",
    "video6_imelda": "video6_Imelda.mp4",
}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*", choices=sorted(VIDEOS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    framework = Path(__file__).resolve().parents[1]
    root = framework.parent
    for key in args.videos or VIDEOS:
        run_video(config_path=framework / "configs" / (key + ".json"), workspace_root=root,
            video_override=root / "00-Videos" / VIDEOS[key],
            output_dir=(args.output or framework / "runs" / "internal_pipeline") / key,
            max_seconds=args.max_seconds)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
