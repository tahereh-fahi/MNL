"""Small regression checks using decoded source-video frames, without OCR.

Frame numbers are zero-based source indices. Video 5's 487--494 boundary was
reviewed by the user; the other frames exercise existing menus and chest UI.
These checks cover specific real examples, not whole-video accuracy.
"""

from pathlib import Path
import unittest

import cv2
import numpy as np

from vss_framework.detectors.inventory import (
    level_up_option_rectangles,
    stable_level_up_option_rectangles,
)
from vss_framework.detectors.screen_state import ScreenStateDetector
from vss_framework.detectors.weapons import (
    has_large_menu_overlay,
    read_temporal_median,
)
from vss_framework.gameplay_state import gameplay_pause_evidence, level_up_pause_evidence
from vss_framework.video import FramePacket


VIDEOS = Path(__file__).resolve().parents[2] / "00-Videos"


class RealPauseInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images = {}
        samples = {
            "video5_Imelda_0_part1.mp4": list(range(487, 495)) + [1131],
            "Video1_Imelda.mp4": [144, 535, 690, 824],
            "video4_Imelda_100.mp4": [
                60, 125, 280, 804, 2760, 2865, 2960,
                22410, 22416, 22417, 22620, 22740, 22770,
                32778,
            ],
            "FlameGuy_LowLuck50_MadForest_Hurry.mp4": [
                2311, 2315, 2319, 2331, 2347,
                2804, 2808, 2835,
                3028, 3035, 3041, 3045,
                3105, 3111, 3127, 3136,
            ],
        }
        for name, indices in samples.items():
            source = VIDEOS / name
            if not source.is_file():
                raise unittest.SkipTest(f"Real video is not installed: {name}")
            capture = cv2.VideoCapture(str(source))
            try:
                for index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if not ok:
                        raise AssertionError(f"Cannot decode {name} frame {index}")
                    cls.images[name, index] = frame
            finally:
                capture.release()

    def test_video5_transition_blocks_gameplay_before_menu_is_readable(self):
        name = "video5_Imelda_0_part1.mp4"
        self.assertFalse(has_large_menu_overlay(self.images[name, 487]))
        detector = ScreenStateDetector()
        for index in range(488, 494):
            with self.subTest(frame=index):
                image = self.images[name, index]
                self.assertTrue(has_large_menu_overlay(image))
                self.assertEqual(level_up_pause_evidence(image)["phase"], "level_up_transition")
                self.assertEqual(stable_level_up_option_rectangles(image), [])
                observation = detector.observe(FramePacket(name, index, round(index * 1000 / 30), image))[0]
                self.assertIn(observation.value, {"interruption_transition", "level_up_transition"})
                self.assertFalse(observation.attributes["gameplay_measurements_allowed"])
        # The old rectangle detector sees 3 choices while the last card is
        # still sliding into place. They must not enter the menu audit yet.
        self.assertEqual(len(level_up_option_rectangles(self.images[name, 493])), 3)
        self.assertEqual(len(stable_level_up_option_rectangles(self.images[name, 494])), 3)

    def test_temporal_median_does_not_mix_transition_with_gameplay(self):
        name = "video5_Imelda_0_part1.mp4"
        capture = cv2.VideoCapture(str(VIDEOS / name))
        try:
            median = read_temporal_median(capture, 487 / 30, (0, 1 / 30, 2 / 30))
            self.assertTrue(np.array_equal(median, self.images[name, 487]))
            with self.assertRaisesRegex(RuntimeError, "Could not read"):
                read_temporal_median(capture, 489 / 30, (0,))
        finally:
            capture.release()

    def test_pregame_geometry_is_not_a_level_up_choice(self):
        for key in [("Video1_Imelda.mp4", 144), ("video4_Imelda_100.mp4", 60)]:
            with self.subTest(source_frame=key):
                image = self.images[key]
                self.assertFalse(level_up_pause_evidence(image)["blocked"])
                self.assertEqual(stable_level_up_option_rectangles(image), [])

    def test_four_three_and_two_choice_menus_remain_readable(self):
        for name, index, count in [
            ("video5_Imelda_0_part1.mp4", 1131, 3),
            ("Video1_Imelda.mp4", 690, 3),
            ("video4_Imelda_100.mp4", 280, 3),
            ("video4_Imelda_100.mp4", 804, 4),
            ("video4_Imelda_100.mp4", 32778, 2),
        ]:
            with self.subTest(video=name, frame=index):
                image = self.images[name, index]
                self.assertEqual(level_up_pause_evidence(image)["phase"], "level_up_menu")
                self.assertEqual(len(stable_level_up_option_rectangles(image)), count)

    def test_chest_panel_and_bottom_reward_card_are_not_level_up_choices(self):
        for index in [2760, 2865, 2960]:
            with self.subTest(frame=index):
                image = self.images["video4_Imelda_100.mp4", index]
                self.assertFalse(level_up_pause_evidence(image)["blocked"])
                pause = gameplay_pause_evidence(image)
                self.assertTrue(pause["blocked"])
                self.assertEqual(pause["interruption_type"], "treasure")
                self.assertEqual(pause["phase"], "treasure_menu")
                self.assertEqual(stable_level_up_option_rectangles(image), [])

    def test_clear_gameplay_stays_available(self):
        for key in [("Video1_Imelda.mp4", 535), ("Video1_Imelda.mp4", 824),
                    ("video4_Imelda_100.mp4", 125)]:
            with self.subTest(source_frame=key):
                self.assertFalse(has_large_menu_overlay(self.images[key]))

    def test_reviewed_spellbinder_treasure_sequence_is_not_level_up(self):
        name = "video4_Imelda_100.mp4"
        self.assertFalse(gameplay_pause_evidence(self.images[name, 22410])["blocked"])
        opening = gameplay_pause_evidence(self.images[name, 22416])
        self.assertTrue(opening["blocked"])
        self.assertEqual(opening["phase"], "interruption_transition")
        for index in (22417, 22620, 22740):
            with self.subTest(frame=index):
                pause = gameplay_pause_evidence(self.images[name, index])
                self.assertTrue(pause["blocked"])
                self.assertEqual(pause["interruption_type"], "treasure")
                self.assertFalse(level_up_pause_evidence(self.images[name, index])["blocked"])
        self.assertFalse(gameplay_pause_evidence(self.images[name, 22770])["blocked"])

    def test_flameguy_compressed_level_up_cards_keep_all_four_menus_blocked(self):
        name = "FlameGuy_LowLuck50_MadForest_Hurry.mp4"
        groups = [
            [2311, 2315, 2319, 2331, 2347],
            [2804, 2808, 2835],
            [3028, 3035, 3041, 3045],
            [3105, 3111, 3127, 3136],
        ]
        for group in groups:
            phases = []
            for index in group:
                pause = gameplay_pause_evidence(self.images[name, index])
                self.assertTrue(pause["blocked"], f"frame {index}")
                self.assertEqual(pause["interruption_type"], "level_up")
                phases.append(pause["phase"])
            self.assertIn("level_up_menu", phases)


if __name__ == "__main__":
    unittest.main()
