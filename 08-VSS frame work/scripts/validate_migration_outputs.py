"""Compare real Video 4 migration outputs with the retained smoke reference."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from vss_framework.io import write_json

FIELDS = ("frame_a", "frame_b", "video_time_a", "video_time_b", "collected_blue_gems",
          "collected_green_gems", "collected_red_gems", "unresolved_collected_gems",
          "collected_gems_total", "needs_review")

def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: row.get(key) for key in FIELDS} for row in csv.DictReader(handle)]

def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    new_csv = next((args.run / manifest["stages"]["gems"]["directory"] / "worker").glob("*xp_ab_events.csv"))
    old_csv = next((args.reference / "worker").glob("*xp_ab_events.csv"))
    old, new = rows(old_csv), rows(new_csv)
    result = {"prepared_by": "Tahereh Fahi", "scope": "real_video4_first_20_seconds",
        "reference_event_count": len(old), "migrated_event_count": len(new), "compared_fields": list(FIELDS),
        "timing_quantities_and_unresolved_flags_identical": old == new,
        "execution_seconds": time.monotonic() - started, "prompt_workflow_elapsed_time": "not recorded",
        "metadata_verification": {"prepared_by": "Tahereh Fahi", "absolute_paths_in_report": False}}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if old == new else 1

if __name__ == "__main__":
    raise SystemExit(main())
