"""Read-only harness checks on actual code/config/artifacts; no video job."""
import ast
import json
from pathlib import Path
import unittest

from vss_framework.compare_inventory_migration import (
    PACKAGE, PINS, event_summary, matched_inventory_settings,
)
from vss_framework.compare_hud_migration import compare_csv, sha

ROOT = PACKAGE.parents[2]


class InventoryMigrationComparisonTest(unittest.TestCase):
    def test_actual_settings(self):
        paths = [ROOT / "08-VSS-external-reconstructed/configs/video4.json",
                 PACKAGE.parents[1] / "configs/video4.json"]
        if not all(p.is_file() for p in paths):
            self.skipTest("Real comparison configurations are not installed")
        settings = matched_inventory_settings(*(json.loads(p.read_text()) for p in paths))
        self.assertEqual(settings["sample_fps"], 10)
        self.assertEqual(settings["start_second"], 0)

    def test_code_pins_and_matching_reference_parameters(self):
        paths = {"external_inventory": ROOT / "03_weapons/scripts/inventory_event_recorder.py",
                 "internal_inventory": PACKAGE / "detectors/inventory.py",
                 "external_weapons": ROOT / "03_weapons/scripts/weapon_screen_recorder.py",
                 "internal_weapons": PACKAGE / "detectors/weapons.py"}
        if not all(p.is_file() for p in paths.values()):
            self.skipTest("External comparison code is not installed")
        # The comparison harness remains pinned to its historical code pair.
        # The ongoing internal detectors have since changed intentionally;
        # this test must not present the old pairing as today's equivalence.
        for key in ("external_inventory", "external_weapons"):
            self.assertEqual(sha(paths[key]), PINS[key])
        for key in ("internal_inventory", "internal_weapons"):
            self.assertNotEqual(sha(paths[key]), PINS[key])
        def match_config(path):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body
                                 if isinstance(n, ast.ClassDef) and n.name == "MatchConfig"))
        self.assertEqual(match_config(paths["external_weapons"]), match_config(paths["internal_weapons"]))

    def test_real_raw_artifact_summary_preserves_levels(self):
        worker = ROOT / "08-VSS-external-reconstructed/runs/video4_inventory_full_20260912_162323/detector"
        source = worker / "worker/inventory_events.csv"
        if not source.is_file():
            self.skipTest("Real inventory regression artifact not installed")
        manifest = json.loads((worker / "run_manifest.json").read_text())
        self.assertEqual(sha(source), manifest["outputs"]["inventory_events"]["sha256"])
        result = compare_csv(source, source)
        self.assertTrue(result["equal"])
        summary = event_summary(source, 60)
        self.assertEqual(summary["rows"], result["external_rows"])
        self.assertTrue(summary["outside_requested_prefix"])
        self.assertTrue(all("character_level" in row for row in summary["events"]))


if __name__ == "__main__":
    unittest.main()
