#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vss_framework.run_segmentation import segment_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hud-csv", type=Path, required=True)
    parser.add_argument("--video-asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = segment_runs(args.hud_csv, video_asset_id=args.video_asset_id, output_path=args.output)
    print(json.dumps({"status": "ok", "segments": result["segments"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
