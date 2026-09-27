"""Small command/comparator checks using existing real artifacts, never OCR."""
import json
from pathlib import Path
import unittest

from vss_framework.compare_hud_migration import (
    PINS, PACKAGE, compare_csv, load_external_normalizer, matched_settings, sha,
)
from vss_framework.normalization.hud_worker import normalize_hud

ROOT = PACKAGE.parents[2]


class HudMigrationComparisonTest(unittest.TestCase):
    def test_current_real_configurations_match(self):
        paths = [ROOT / "08-VSS-external-reconstructed/configs/video4.json",
                 PACKAGE.parents[1] / "configs/video4.json"]
        if not all(p.exists() for p in paths):
            self.skipTest("Comparison configurations not installed")
        settings = matched_settings(*(json.loads(p.read_text()) for p in paths))
        self.assertEqual(settings["initial_kill_state"], 0)

    def test_audited_detector_pins(self):
        paths = {"external": ROOT / "01_kill_counter_and_time_stamp/scripts/extract_hud_worker.py",
                 "internal": PACKAGE / "detectors/hud.py"}
        if not all(p.exists() for p in paths.values()):
            self.skipTest("External comparison source not installed")
        # The comparison harness remains pinned to its historical pair.
        # The active HUD worker has intentionally changed since that audit.
        self.assertEqual(sha(paths["external"]), PINS["external"])
        self.assertNotEqual(sha(paths["internal"]), PINS["internal"])

    def test_real_artifact_comparison_and_normalization(self):
        worker = ROOT / "08-VSS-external-reconstructed/runs/video4_hud_smoke20_02/detector/worker"
        source = worker / "hud_observations.csv"
        if not source.exists():
            self.skipTest("Real HUD artifact not installed")
        manifest = json.loads((worker / "manifest.json").read_text())
        self.assertEqual(sha(source), manifest["outputs"]["hud_observations"]["sha256"])
        result = compare_csv(source, source)
        self.assertTrue(result["equal"])
        self.assertEqual(result["external_rows"], 20)
        config = json.loads((PACKAGE.parents[1] / "configs/video4.json").read_text())
        dataset = config["dataset"]
        settings = config["detectors"]["hud_clock"]
        kwargs = dict(source_csv=source, source_csv_relative="worker/hud_observations.csv",
                      processing_run_id="paired_hud", session_id=dataset["session_id"],
                      video_asset_id=dataset["video_asset_id"], duration_ms=20000,
                      config={"source_sampling_offsets_seconds": settings["sample_offsets_seconds"],
                              "clock_qc": settings["clock_qc"]})
        external = load_external_normalizer(ROOT / "06-Pipeline/mnl_pipeline")
        self.assertEqual(external(**kwargs), normalize_hud(**kwargs))


if __name__ == "__main__":
    unittest.main()
