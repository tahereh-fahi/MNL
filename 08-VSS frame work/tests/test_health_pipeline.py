"""Targeted checks using real saved artifacts; no video/OCR or synthetic data."""
import ast
import json
import unittest
from pathlib import Path

from vss_framework.health_calibration import estimate_full_health_width
from vss_framework.health_export import build_health_payload
from vss_framework.health_provenance import verified_outputs, verify_inventory, implementation_receipt
from vss_framework.dashboard_release import project_canonical_event

ROOT = Path(__file__).resolve().parents[1]


class HealthPipelineTest(unittest.TestCase):
    def setUp(self):
        self.health = ROOT / "runs/video4_health_events_v3"
        self.attribution = ROOT / "runs/video4_health_attribution_v1"
        if not (self.health / "run_manifest.json").exists():
            self.skipTest("Real historical health artifacts are not installed")

    def test_saved_health_hashes_and_real_calibration(self):
        manifest, outputs = verified_outputs(self.health)
        rows = [json.loads(x) for x in outputs["health_observations"].read_text().splitlines()]
        widths = [r["attributes"]["raw_red_fill_width_1440p_px"] for r in rows if r["observed"]]
        reference, _ = estimate_full_health_width(widths)
        self.assertEqual(reference, manifest["calibration"]["full_bar_reference_width_1440p_px"])

    def test_changed_real_inventory_is_rejected(self):
        path = ROOT / "runs/video4_levelup_transactions_complete/canonical_inventory_events.jsonl"
        if not path.exists():
            self.skipTest("Historical inventory artifact not installed")
        # Known mismatched real artifact, not a fabricated corruption fixture.
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            verify_inventory(path)

    def test_export_preserves_real_values_and_uncertainty(self):
        payload = build_health_payload(self.health, self.attribution)
        existing = json.loads((ROOT.parent / "07-Dashboard/public/data/video4-health.json").read_text())
        self.assertEqual(payload["observations"], existing["observations"])
        self.assertEqual(payload["calibration"], existing["calibration"])
        self.assertFalse(payload["publicationReady"])
        self.assertEqual(len(payload["events"]), len(existing["events"]))
        for old, new in zip(existing["events"], payload["events"]):
            self.assertEqual(old, {key: new[key] for key in old})
            self.assertIn("publicationStatus", new)
        event = json.loads((self.attribution / "attributed_health_events.jsonl").read_text().splitlines()[0])
        self.assertEqual(project_canonical_event(event)["evidence"], event["evidence"])

    def test_receipt_covers_transitive_health_code(self):
        receipt = implementation_receipt()
        for name in ("health_calibration.py", "health_attribution.py", "telemetry_scan.py", "detectors/gems.py", "resources.py"):
            self.assertIn(name, receipt["code_sha256"])
        self.assertFalse(receipt["reuse_allowed"])

    def test_runner_wires_fresh_health_inputs(self):
        tree = ast.parse((ROOT / "src/vss_framework/run_video.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        stages = [n.args[0].value for n in calls if isinstance(n.func, ast.Name)
                  and n.func.id == "stage" and n.args and isinstance(n.args[0], ast.Constant)]
        self.assertIn("health", stages)
        self.assertIn("health_attribution", stages)
        attribution = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "attribute_video4_health")
        inventory = next(k.value for k in attribution.keywords if k.arg == "inventory_events_path")
        self.assertIsInstance(inventory, ast.BinOp)
        self.assertEqual(inventory.left.id, "base_inventory")


if __name__ == "__main__":
    unittest.main()
