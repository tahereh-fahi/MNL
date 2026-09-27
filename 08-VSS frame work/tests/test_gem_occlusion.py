from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


from vss_framework.detectors import gems as detector
from vss_framework.detectors.xp_bar import XPBarDetector
from vss_framework.video import FramePacket


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

    @staticmethod
    def _gameplay_frame(
        *,
        with_border: bool,
        height: int = 768,
        width: int = 1366,
        border_row: int = 22,
        with_blue_fill: bool = False,
    ) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x0 = int(round(frame.shape[1] * 0.04))
        x1 = int(round(frame.shape[1] * 0.76))
        if with_border:
            hsv[border_row, x0:x1] = (20, 220, 220)
        if with_blue_fill:
            cfg = detector.Config()
            bar_x0 = int(round(width * cfg.xp_bar_x0_fraction))
            bar_x1 = int(round(width * cfg.xp_bar_x1_fraction))
            bar_y0 = int(round(height * cfg.xp_bar_y0_fraction))
            bar_y1 = int(round(height * cfg.xp_bar_y1_fraction))
            fill_x1 = bar_x0 + int(round((bar_x1 - bar_x0) * 0.37))
            hsv[bar_y0:bar_y1, bar_x0:fill_x1] = (110, 220, 220)
        # Model the bright orange rectangle seen over the center of the XP bar.
        orange_x0 = int(round(width * 0.46))
        orange_x1 = int(round(width * 0.54))
        hsv[0 : max(1, int(round(height * 0.016))), orange_x0:orange_x1] = (
            15,
            240,
            240,
        )
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_gameplay_hud_score_keeps_shifted_border_under_orange_overlay(self) -> None:
        cases = (
            (768, 1366, 20),
            (768, 1366, 21),
            (768, 1366, 22),
            (1080, 1920, 29),
            (1080, 1920, 30),
            (1080, 1920, 31),
        )
        for height, width, border_row in cases:
            with self.subTest(
                height=height, width=width, border_row=border_row
            ):
                score = detector.gameplay_hud_score(
                    self._gameplay_frame(
                        with_border=True,
                        height=height,
                        width=width,
                        border_row=border_row,
                    )
                )
                self.assertGreaterEqual(score, 0.90)

    def test_orange_overlay_alone_does_not_claim_gameplay_hud(self) -> None:
        score = detector.gameplay_hud_score(
            self._gameplay_frame(with_border=False)
        )

        self.assertLess(score, 0.90)

    def test_scattered_short_gold_regions_do_not_claim_gameplay_hud(self) -> None:
        frame = self._gameplay_frame(with_border=False)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[22, 100:220] = (20, 220, 220)
        hsv[18, 500:610] = (20, 220, 220)
        hsv[20, 900:970] = (20, 220, 220)

        score = detector.gameplay_hud_score(
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        )

        self.assertLess(score, 0.90)

    def test_xp_detector_accepts_shifted_border_with_orange_overlay(self) -> None:
        frame = self._gameplay_frame(with_border=True, with_blue_fill=True)
        packet = FramePacket("synthetic", 1, 33, frame)
        xp_detector = XPBarDetector()

        with patch(
            "vss_framework.detectors.xp_bar.gameplay_pause_evidence",
            return_value={"blocked": False, "phase": "gameplay_unblocked"},
        ):
            observation = xp_detector.observe(packet)[0]

        self.assertTrue(observation.attributes["accepted"])
        self.assertAlmostEqual(observation.value, 0.37, places=2)


if __name__ == "__main__":
    unittest.main()
