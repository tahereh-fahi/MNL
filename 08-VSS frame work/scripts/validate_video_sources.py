"""Run internal frame detectors on real samples from each available source."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from vss_framework.detectors import ScreenStateDetector, XPBarDetector
from vss_framework.io import write_json
from vss_framework.video import OpenCVVideoReader

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    audit = json.loads(args.audit.read_text())
    sources = sorted({name for check in audit["configured_videos"] for name in check["matching_files"]})
    detectors = [ScreenStateDetector(), XPBarDetector()]
    results = []
    for source in sources:
        observations = []
        with OpenCVVideoReader(args.workspace_root / source, source) as reader:
            duration = reader.metadata.duration_ms
            for ms in sorted({min(10000, max(0, duration - 1000)), duration // 2, max(0, duration - 2000)}):
                for packet in reader.iter_packets(start_ms=ms, end_ms=min(duration, ms + 500), sample_fps=2):
                    for detector in detectors:
                        observations.extend(o.to_dict() for o in detector.observe(packet))
        results.append({"source": source, "observations": len(observations), "states": [o["value"] for o in observations if o["observation_type"] == "screen_state"]})
    result = {"prepared_by": "Tahereh Fahi", "scope": "three_real_frame_samples_per_available_source_not_full_extraction",
              "sources": results, "source_count": len(results), "execution_seconds": time.monotonic() - start,
              "prompt_workflow_elapsed_time": "not recorded", "metadata_verification": {"prepared_by": "Tahereh Fahi"}}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
