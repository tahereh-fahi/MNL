"""Targeted real-frame geometry checks; no OCR or full video extraction."""
from pathlib import Path
import unittest
import cv2
import numpy as np
from vss_framework.detectors.gems import observed_level_crop, prepare_level_ocr_crop
from vss_framework.detectors.inventory import _find_initial_gameplay_frame
from vss_framework.detectors.weapons import _largest_gold_bar_component

VIDEOS = Path(__file__).resolve().parents[2] / "00-Videos"


class GameplayLayoutRegression(unittest.TestCase):
    def capture(self, filename):
        path = VIDEOS / filename
        if not path.is_file():
            self.skipTest("Real source video unavailable")
        capture = cv2.VideoCapture(str(path))
        self.addCleanup(capture.release)
        return capture

    def test_loading_has_no_observed_bar_or_level_crop(self):
        capture = self.capture("Video1_Imelda.mp4")
        capture.set(cv2.CAP_PROP_POS_MSEC, 7000)
        ok, frame = capture.read()
        self.assertTrue(ok)
        with self.assertRaises(ValueError):
            _largest_gold_bar_component(frame, allow_fallback=False)
        self.assertEqual(observed_level_crop(frame).size, 0)

    def test_loading_is_not_initial_inventory(self):
        capture = self.capture("Video1_Imelda.mp4")
        _, second, frame_number = _find_initial_gameplay_frame(capture, second=5, duration=13)
        self.assertGreater(second, 7.0)
        self.assertGreater(frame_number, 144)

    def test_no_gameplay_fails_closed(self):
        capture = self.capture("Video1_Imelda.mp4")
        with self.assertRaises(RuntimeError):
            _find_initial_gameplay_frame(capture, second=7.0, duration=7.2)

    def test_level_crops_contain_bright_text_in_both_layouts(self):
        for filename, number in (("Video1_Imelda.mp4", 535),
                                 ("Video1_Imelda.mp4", 824),
                                 ("video4_Imelda_100.mp4", 461)):
            with self.subTest(video=filename, frame=number):
                capture = self.capture(filename)
                capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                ok, frame = capture.read()
                self.assertTrue(ok)
                crop = observed_level_crop(frame)
                self.assertGreater(crop.size, 0)
                # White text evidence, not a claim of successful OCR.
                self.assertGreater(int(np.all(crop > 180, axis=2).sum()), 50)
                prepared = prepare_level_ocr_crop(crop)
                self.assertGreater(prepared.size, 0)
                self.assertTrue(np.all(prepared[0] == 255))
                self.assertTrue(np.all(prepared[:, -1] == 255))
                _, _, stats, _ = cv2.connectedComponentsWithStats(
                    (prepared == 0).astype(np.uint8), 8)
                # These real frames contain LV1/LV2: three substantial glyphs,
                # not a fourth stroke from the right-edge highlight.
                self.assertEqual(sum(int(s[cv2.CC_STAT_AREA]) > 10 for s in stats[1:]), 3)


if __name__ == "__main__":
    unittest.main()
