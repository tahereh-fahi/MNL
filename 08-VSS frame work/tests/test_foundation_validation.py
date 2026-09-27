"""Focused diagnostic checks using existing project configurations.

Prepared by Tahereh Fahi. These tests do not establish video detection accuracy.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from vss_framework.cli import build_parser
from vss_framework.foundation_validation import sample_times, select_videos
from vss_framework.hashing import sha256_file
from vss_framework.validation_cache import run_stage, verified_receipt
from vss_framework.video_registry import configurations


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CONFIG = PROJECT / "configs/video2_sigma_full.json"
OTHER_CONFIG = PROJECT / "configs/video4.json"


class FoundationValidationTests(unittest.TestCase):
    def test_cli_defaults_to_identity_only(self):
        args = build_parser().parse_args(["validate-foundations", "--output", "audit"])
        self.assertEqual(args.stage, "registry")
        self.assertIsNone(args.video)
        self.assertFalse(args.no_resume)

    def test_configurations_preserve_different_video2_sources(self):
        rows = configurations(PROJECT / "configs", WORKSPACE)
        old = next(row for row in rows if row["config"].endswith("/video2.json"))
        full = next(row for row in rows if row["config"].endswith("/video2_sigma_full.json"))
        self.assertNotEqual(old["expected_sha256"], full["expected_sha256"])
        self.assertEqual(full["run_boundary_status"], "requires_verification")
        self.assertFalse(Path(full["configured_source"]).is_absolute())
        self.assertEqual(full["config_sha256"], sha256_file(CONFIG))

    def test_samples_use_real_config_duration_without_end_overrun(self):
        duration = json.loads(CONFIG.read_text())["dataset"]["duration_ms"] / 1000
        times = sample_times(duration, None)
        self.assertEqual(times[0], 0)
        self.assertLess(max(times), duration)
        self.assertEqual(times, sorted(set(times)))
        for invalid in (-1, duration, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sample_times(duration, [invalid])

    def test_selection_requires_unambiguous_source(self):
        records = [{"source": r["configured_source"]}
                   for r in configurations(PROJECT / "configs", WORKSPACE)]
        selector = json.loads(CONFIG.read_text())["dataset"]["video"]["path"]
        result = select_videos(records, [selector])
        self.assertEqual(result, [{"source": selector}])
        with self.assertRaises(ValueError):
            select_videos(result + result, [selector])

    def test_cache_requires_same_input_and_intact_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = {"config_sha256": sha256_file(CONFIG)}

            def copy_config(directory):
                shutil.copyfile(CONFIG, directory / "config.json")

            first = run_stage(root, "config_check", inputs, copy_config)
            reused = run_stage(root, "config_check", inputs, copy_config)
            self.assertTrue(reused["reused"])
            self.assertFalse(first["publication_ready"])
            self.assertEqual(first["validation_status"], "not_validated")
            directory = root / first["directory"]
            shutil.copyfile(OTHER_CONFIG, directory / "config.json")
            self.assertIsNone(verified_receipt(directory, first["identity"]))
            rebuilt = run_stage(root, "config_check", inputs, copy_config)
            self.assertFalse(rebuilt["reused"])
            self.assertNotEqual(rebuilt["directory"], first["directory"])
            changed = run_stage(root, "config_check", {"config_sha256": sha256_file(OTHER_CONFIG)}, copy_config)
            self.assertFalse(changed["reused"])
            forced = run_stage(root, "config_check", inputs, copy_config, resume=False)
            self.assertFalse(forced["reused"])

    def test_unexpected_nested_receipt_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = run_stage(root, "config_check", {"config_sha256": sha256_file(CONFIG)},
                                lambda directory: shutil.copyfile(CONFIG, directory / "config.json"))
            directory = root / receipt["directory"]
            (directory / "extra").mkdir()
            shutil.copyfile(CONFIG, directory / "extra/stage_receipt.json")
            self.assertIsNone(verified_receipt(directory, receipt["identity"]))


if __name__ == "__main__":
    unittest.main()
