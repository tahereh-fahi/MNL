"""Portability and cache-integrity regression checks for internal detectors."""
from __future__ import annotations

import json
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vss_framework.hashing import sha256_file
from vss_framework.resources import PACKAGE_ROOT, asset_path, load_runtime_config, resolve_path
from vss_framework.run_video import verified_stage
from vss_framework.detectors.legacy_gem_xp import normalize_completed_gem_worker
from vss_framework.detectors.inventory import read_inventory_xp_events
from vss_framework.detectors.inventory import EVENT_COLUMNS
from vss_framework.detectors.gems import Config as GemConfig, find_xp_events
from vss_framework.inventory_reconciliation import reconcile_inventory_with_chest_rewards


class StandaloneTests(unittest.TestCase):
    def test_gem_event_after_bar_reappearance_is_rejected(self):
        arrays = {
            "pixels": [0, 5, 7, 7],
            "delta_pixels": [0, 5, 2, 0],
            "progress": [0.0, 0.05, 0.07, 0.07],
            "quality": [1.0, 1.0, 1.0, 1.0],
            "overlay": [0.0, 0.0, 0.0, 0.0],
            "hud": [1.0, 1.0, 1.0, 1.0],
            "health_bar_detected": [True, True, True, True],
            "level_up_transition_guard": [False, False, False, False],
            "xp_bar_reappearance": [False, True, False, False],
        }
        events = find_xp_events(
            {"fps": 30.0, "bar_width": 100.0},
            arrays,
            resets=[],
            cfg=GemConfig(),
        )
        self.assertEqual(events, [])

    def test_automatic_chest_rewards_replace_hud_chest_guesses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.csv"
            rows = [
                {"event_id":"1","video":"video","video_second":0,"frame_number":0,"character_level":1,"event_source":"initial_state","event_type":"initial_state","item_type":"weapon","slot":1,"item_before":"","item_after":"Magic Wand","level_before":"","level_after":1,"normal_max_level":8,"suggested_item":"Magic Wand","confidence":"high","needs_review":False},
                {"event_id":"2","video":"video","video_second":10,"frame_number":300,"character_level":2,"event_source":"level_up","event_type":"upgrade","item_type":"weapon","slot":1,"item_before":"Magic Wand","item_after":"Magic Wand","level_before":1,"level_after":2,"normal_max_level":8,"suggested_item":"Magic Wand","confidence":"high","needs_review":False},
                {"event_id":"4","video":"video","video_second":15,"frame_number":450,"character_level":3,"event_source":"level_up","event_type":"new","item_type":"weapon","slot":2,"item_before":"","item_after":"Axe","level_before":"","level_after":1,"normal_max_level":8,"suggested_item":"Axe","confidence":"high","needs_review":False},
                {"event_id":"3","video":"video","video_second":20,"frame_number":600,"character_level":2,"event_source":"treasure_chest","event_type":"evolution","item_type":"weapon","slot":1,"item_before":"Knife","item_after":"Thousand Edge","level_before":"","level_after":1,"normal_max_level":"","suggested_item":"Thousand Edge","confidence":"low","needs_review":True},
            ]
            with base.open("w", newline="", encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=EVENT_COLUMNS); writer.writeheader(); writer.writerows(rows)
            rewards=root / "rewards.jsonl"
            reward={"anchor_time_ms":20000,"publication_status":"auto_accepted","item_name":"Holy Wand","item_type":"weapon","action":"evolution","attributes":{"base_item":"Magic Wand"}}
            rewards.write_text(json.dumps(reward)+"\n",encoding="utf-8")
            result=reconcile_inventory_with_chest_rewards(base_inventory_path=base,chest_rewards_path=rewards,output_dir=root/"out",fps=30)
            self.assertEqual(result["counts"]["removed_hud_chest_inferences"],1)
            self.assertEqual(result["counts"]["inventory_rows"],4)
            output_path=root/"out/inventory_events.csv"
            output=output_path.read_text()
            self.assertIn("Holy Wand",output)
            self.assertNotIn("Thousand Edge",output)
            with output_path.open(encoding="utf-8") as handle:
                reconciled=list(csv.DictReader(handle))
            chest=next(row for row in reconciled if row["event_source"] == "treasure_chest")
            self.assertEqual(chest["character_level"], "3")

    def test_real_zero_event_worker_output_is_not_missing_data(self):
        fixture = Path(__file__).parent / "fixtures/video4_intro_no_gems"
        summary = json.loads((fixture / "summary.json").read_text())
        observations, normalized, issues = normalize_completed_gem_worker(
            worker_summary=summary, source_csv=fixture / "events.csv", config={})
        self.assertEqual(observations, [])
        self.assertEqual(issues, [])
        self.assertTrue(normalized["empty_result_verified_against_worker_summary"])
        self.assertIsNone(normalized["unresolved_quantity_fraction"])
        with self.assertRaises(FileNotFoundError):
            normalize_completed_gem_worker(worker_summary=summary, source_csv=fixture / "missing.csv", config={})

    def test_empty_worker_output_requires_explicit_zero_confirmation(self):
        fixture = Path(__file__).parent / "fixtures/video4_intro_no_gems"
        summary = json.loads((fixture / "summary.json").read_text())
        del summary["collected_gems_total"]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            normalize_completed_gem_worker(worker_summary=summary, source_csv=fixture / "events.csv", config={})

    def test_inventory_accepts_confirmed_empty_xp_without_fabricating_events(self):
        fixture = Path(__file__).parent / "fixtures/video4_intro_no_gems"
        events = read_inventory_xp_events(fixture / "events.csv")
        self.assertTrue(events.empty)
        self.assertIn("reset_inferred_level", events.columns)
        self.assertIn("video_time_b", events.columns)

    def test_relocated_package_loads_detectors_and_assets_without_legacy_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PACKAGE_ROOT, root / "vss_framework", ignore=shutil.ignore_patterns("__pycache__"))
            code = '''
import sys
class DenyLegacy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mnl_pipeline', 'inventory_event_recorder', 'weapon_screen_recorder', 'detect_collected_gems_from_xp_ab'}:
            raise AssertionError('Legacy import: ' + fullname)
sys.meta_path.insert(0, DenyLegacy())
from vss_framework.detectors import gems, weapons, inventory, hud
from vss_framework.detectors.screen_state import ScreenStateDetector
from vss_framework.detectors.xp_bar import XPBarDetector
from vss_framework.normalization.gem_worker import normalize_gems
from vss_framework.normalization.hud_worker import normalize_hud
from vss_framework.resources import asset_path
from vss_framework.catalog import load_event_catalog
assert load_event_catalog(asset_path('event_catalog.json'))['events']
assert all((hud.TEMPLATE_DIR / name).is_file() for name in hud.SKULL_TEMPLATE_NAMES)
assert gems.load_template_bank(asset_path('gems'), 1440, profile='legacy')
assert weapons.load_weapon_manifest(asset_path('manifests/wiki_weapon_manifest.csv')).shape[0] > 0
assert ScreenStateDetector()._load_legacy()[0] is inventory
assert XPBarDetector()._load_legacy()[0] is gems
print('isolated imports and assets passed')
'''
            result = subprocess.run([sys.executable, "-c", code], cwd=root,
                env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cache_rejects_changed_outputs_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "catalog.json"
            shutil.copyfile(asset_path("event_catalog.json"), output)
            receipt = {"status": "complete", "identity": "original", "outputs": {output.name: sha256_file(output)}}
            (root / "stage_receipt.json").write_text(json.dumps(receipt))
            self.assertTrue(verified_stage(root, "original"))
            self.assertFalse(verified_stage(root, "changed"))
            output.write_bytes(output.read_bytes() + b"\n")
            self.assertFalse(verified_stage(root, "original"))

    def test_resource_cannot_escape_package(self):
        with self.assertRaises(ValueError):
            resolve_path("framework:../outside", Path.cwd())

    def test_configs_resolve_to_packaged_detectors(self):
        project = Path(__file__).resolve().parents[1]
        for path in (project / "configs").glob("*.json"):
            if "detectors" not in json.loads(path.read_text()):
                continue
            config = load_runtime_config(path)
            for name, detector in config["detectors"].items():
                if "script" in detector:
                    source = resolve_path(detector["script"], project.parent)
                    self.assertTrue(source.is_relative_to(PACKAGE_ROOT), (path.name, name))
                    self.assertTrue(source.is_file())
                self.assertNotIn("pipeline_module_dir", detector)
