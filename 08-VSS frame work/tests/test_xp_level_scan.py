"""Small guard/CLI checks using the real project configuration; no video job."""
import json
import unittest
from pathlib import Path

from vss_framework.cli import build_parser
from vss_framework.xp_level_scan import extract_xp_levels, sha256, validate_scope

ROOT = Path(__file__).resolve().parents[1]


class XPLevelScanTests(unittest.TestCase):
    def test_cli(self):
        args = build_parser().parse_args([
            "extract-xp-levels", "--config", str(ROOT / "configs/video4.json"),
            "--output", "unused", "--max-seconds", "20"])
        self.assertEqual(args.max_seconds, 20)
        self.assertEqual(args.command, "extract-xp-levels")

    def test_bad_scope(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_scope(value)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            extract_xp_levels(ROOT / "configs/video4.json", ROOT.parent, ROOT, 20)

    def test_real_config_identity(self):
        path = ROOT / "configs/video4.json"
        config = json.loads(path.read_text())
        self.assertEqual(len(sha256(path)), 64)
        self.assertEqual(len(config["dataset"]["video"]["sha256"]), 64)
        self.assertEqual(config["detectors"]["gem_xp"]["initial_level"], 1)


if __name__ == "__main__":
    unittest.main()
