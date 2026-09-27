"""Small checks on actual source/config/data; no OCR or gem tracking runs."""
import json
import unittest

from vss_framework.compare_gem_migration import (
    PACKAGE, PINS, config_ast, matched_gem_settings, one_event_csv,
)
from vss_framework.compare_hud_migration import compare_csv, sha

ROOT = PACKAGE.parents[2]


class GemMigrationComparisonTest(unittest.TestCase):
    def test_actual_settings_and_config_defaults(self):
        configs = [ROOT / "08-VSS-external-reconstructed/configs/video4.json",
                   PACKAGE.parents[1] / "configs/video4.json"]
        if not all(p.exists() for p in configs):
            self.skipTest("Comparison configurations not installed")
        settings = matched_gem_settings(*(json.loads(p.read_text()) for p in configs))
        self.assertEqual(settings, {"template_profile": "legacy", "initial_level": 1})
        a = ROOT / "02_gems/scripts/detect_collected_gems_from_xp_ab.py"
        b = PACKAGE / "detectors/gems.py"
        self.assertEqual(config_ast(a), config_ast(b))

    def test_detector_pins(self):
        paths = {"external": ROOT / "02_gems/scripts/detect_collected_gems_from_xp_ab.py",
                 "internal": PACKAGE / "detectors/gems.py"}
        if not all(p.exists() for p in paths.values()):
            self.skipTest("Comparison source files not installed")
        self.assertEqual({k: sha(p) for k, p in paths.items()}, PINS)

    def test_real_gem_csv_comparator(self):
        detector = ROOT / "08-VSS-external-reconstructed/runs/video4_gems_smoke20_01/detector"
        if not detector.exists():
            self.skipTest("Real gem artifact not installed")
        source = one_event_csv(detector / "worker")
        # The actual manifest is checked rather than treating the old filename
        # as evidence of integrity. This tests comparison only, not extraction.
        manifest = json.loads((detector / "run_manifest.json").read_text())
        candidates = [v for v in manifest["outputs"]["worker_files"]
                      if v.get("path") == str(source.relative_to(detector))]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(sha(source), candidates[0]["sha256"])
        result = compare_csv(source, source)
        self.assertTrue(result["equal"])
        self.assertGreater(result["external_rows"], 0)


if __name__ == "__main__":
    unittest.main()
