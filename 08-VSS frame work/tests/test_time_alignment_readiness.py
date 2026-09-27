import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("readiness", Path(__file__).resolve().parents[1] / "scripts/audit_time_alignment_readiness.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SpanDiagnosticsTest(unittest.TestCase):
    def test_declared_gap(self):
        gaps = [{"startMs": 320533, "endMs": 632000}]
        self.assertEqual(module.media_span_status(967700, gaps, 0, 300000), "span-only")
        self.assertEqual(module.media_span_status(967700, gaps, 300000, 600000), "gap")
        self.assertEqual(module.media_span_status(967700, gaps, 600000, 900000), "gap")

    def test_short_video_is_not_zero(self):
        self.assertEqual(module.media_span_status(200000, [], 0, 300000), "partial")
        self.assertEqual(module.media_span_status(200000, [], 300000, 600000), "absent")

    def test_boundary_touch_does_not_overlap(self):
        self.assertEqual(module.media_span_status(600000, [{"startMs": 300000, "endMs": 310000}], 0, 300000), "span-only")

    def test_missing_duration_is_absent(self):
        self.assertEqual(module.media_span_status(None, [], 0, 300000), "absent")


if __name__ == "__main__":
    unittest.main()
