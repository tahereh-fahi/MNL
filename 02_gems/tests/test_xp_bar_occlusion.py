from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/detect_collected_gems_from_xp_ab.py"
SPEC = importlib.util.spec_from_file_location("xp_occlusion_detector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
detector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = detector
SPEC.loader.exec_module(detector)


class XPBarOcclusionTests(unittest.TestCase):
    @staticmethod
    def _config() -> detector.Config:
        return detector.Config(
            xp_bar_x0_fraction=0.0,
            xp_bar_x1_fraction=1.0,
            xp_bar_y0_fraction=0.0,
            xp_bar_y1_fraction=1.0,
        )

    @staticmethod
    def _frame(fill_endpoint: int, orange_start: int, orange_end: int) -> np.ndarray:
        hsv = np.zeros((10, 100, 3), dtype=np.uint8)
        hsv[:, :fill_endpoint] = (110, 220, 220)
        hsv[:7, orange_start:orange_end] = (15, 240, 240)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_orange_overlay_does_not_interrupt_blue_fill_beneath_it(self) -> None:
        progress, quality = detector.measure_xp_bar_progress(
            self._frame(fill_endpoint=70, orange_start=40, orange_end=55),
            self._config(),
        )

        self.assertAlmostEqual(progress, 0.70)
        self.assertAlmostEqual(quality, 1.0)

    def test_orange_overlay_beyond_fill_does_not_extend_progress(self) -> None:
        progress, quality = detector.measure_xp_bar_progress(
            self._frame(fill_endpoint=40, orange_start=40, orange_end=55),
            self._config(),
        )

        self.assertAlmostEqual(progress, 0.40)
        self.assertAlmostEqual(quality, 1.0)


if __name__ == "__main__":
    unittest.main()
