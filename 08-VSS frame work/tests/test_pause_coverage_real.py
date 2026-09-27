"""Small real-video checks for consumers of the shared level-up pause gate.

The Video 5 observations were reviewed at frames 487–494. Video 4 frames
2760–2960 cover the first saved chest interval; these must retain chest evidence.
These checks decode only selected source frames and do not run OCR.
"""

from pathlib import Path
import unittest

import cv2

from vss_framework.chest_lifecycle import chest_overlay_features, mark_chest_rows
from vss_framework.chest_rewards import (
    _candidate_crop_for_references,
    _orb_crops,
    resolve_assignments,
)
from vss_framework.detectors import inventory as inventory_detector
from vss_framework.detectors import weapons as weapon_detector
from vss_framework.detectors.xp_bar import XPBarDetector
from vss_framework.gameplay_state import gameplay_pause_evidence, level_up_pause_evidence
from vss_framework.gold_fever import gold_fever_overlay_features, mark_gold_fever_rows
from vss_framework.status_events import classify_status_frame
from vss_framework.video import FramePacket
from vss_framework.world_pickups import is_large_level_up_overlay
from vss_framework.resources import asset_path


VIDEOS = Path(__file__).resolve().parents[2] / "00-Videos"


class PauseCoverageRealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = {}
        for name, indexes in (
            ("video5_Imelda_0_part1.mp4", (487, 488, 489, 493, 494)),
            ("video4_Imelda_100.mp4", (2760, 2865, 2960, 30804)),
            ("Video1_Imelda.mp4", (10496, 21683)),
            ("Video2_Sigma.mp4", (7,)),
        ):
            path = VIDEOS / name
            if not path.is_file():
                raise unittest.SkipTest("Required real source videos unavailable")
            capture = cv2.VideoCapture(str(path))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                for index in indexes:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = capture.read()
                    if not ok:
                        raise AssertionError(f"Cannot decode {name} frame {index}")
                    cls.frames[name, index] = FramePacket(name, index, round(index * 1000 / fps), image)
            finally:
                capture.release()

    def test_animation_excludes_xp_pickup_and_status_misclassification(self):
        detector = XPBarDetector()
        for index in (488, 489, 493, 494):
            with self.subTest(frame=index):
                packet = self.frames["video5_Imelda_0_part1.mp4", index]
                observation, = detector.observe(packet)
                self.assertIsNone(observation.value)
                self.assertFalse(observation.attributes["accepted"])
                self.assertTrue(is_large_level_up_overlay(packet.image))
                self.assertEqual(_orb_crops(packet.image), [])
                status, _ = classify_status_frame(packet.image)
                self.assertIn(status, {"interruption_transition", "level_up_transition", "level_up_menu"})
        before = self.frames["video5_Imelda_0_part1.mp4", 487]
        self.assertFalse(is_large_level_up_overlay(before.image))
        self.assertTrue(detector.observe(before)[0].attributes["accepted"])

    def test_chest_frames_are_not_level_up_menus(self):
        for index in (2760, 2865, 2960):
            with self.subTest(frame=index):
                packet = self.frames["video4_Imelda_100.mp4", index]
                self.assertFalse(level_up_pause_evidence(packet.image)["blocked"])
                pause = gameplay_pause_evidence(packet.image)
                self.assertTrue(pause["blocked"])
                self.assertEqual(pause["interruption_type"], "treasure")
        features = chest_overlay_features(self.frames["video4_Imelda_100.mp4", 2960].image)
        mark_chest_rows([features])
        self.assertEqual(features["reward_orb_count"], 1)
        self.assertTrue(features["chest_overlay_like"])
        self.assertEqual(len(_orb_crops(self.frames["video4_Imelda_100.mp4", 2960].image)), 1)

    def test_frozen_gold_fever_gauge_is_raw_evidence_only(self):
        frame = self.frames["video4_Imelda_100.mp4", 30804].image
        features = gold_fever_overlay_features(frame)
        mark_gold_fever_rows([features])
        self.assertTrue(features["gameplay_paused"])
        self.assertTrue(features["gold_fever_visible_raw"])
        self.assertFalse(features["gold_fever_like"])

    def test_character_selection_gold_decoration_is_not_gold_fever(self):
        frame = self.frames["Video2_Sigma.mp4", 7].image
        features = gold_fever_overlay_features(frame)
        mark_gold_fever_rows([features])
        self.assertTrue(features["gold_fever_visible_raw"])
        self.assertLess(features["gameplay_hud_score"], 0.90)
        self.assertFalse(features["gold_fever_like"])

    def test_video1_red_reward_icons_survive_orb_background_removal(self):
        weapons, passives, _ = inventory_detector.load_item_manifests(
            asset_path("manifests/wiki_weapon_manifest.csv"),
            asset_path("manifests/wiki_passive_item_manifest.csv"),
        )
        match_config = weapon_detector.MatchConfig(
            occupied_luma_std=5,
            occupied_saturation_std=5,
            occupied_laplacian_var=5,
            high_score=.55,
            medium_score=.40,
            unknown_score=.25,
        )
        references = inventory_detector.prepare_combined_references(
            asset_path("weapon_icons"),
            asset_path("passive_icons"),
            weapons,
            passives,
            28,
            match_config,
        )
        allowed = [
            "Magic Wand", "Fire Wand", "King Bible", "Santa Water",
            "Victory Sword", "Pummarola", "Spellbinder",
        ]

        def assignments(frame_index):
            crops = _orb_crops(self.frames["Video1_Imelda.mp4", frame_index].image)
            score_rows = []
            for crop in crops:
                score_rows.append({
                    name: float(weapon_detector.match_weapon_slot(
                        _candidate_crop_for_references(
                            crop,
                            [reference for reference in references if reference.name == name],
                        ),
                        [reference for reference in references if reference.name == name],
                        match_config,
                    )["match_score"])
                    for name in allowed
                })
            return resolve_assignments(score_rows, allowed)

        first = assignments(10496)
        self.assertEqual(first[0][0], "Fire Wand")
        self.assertGreater(first[0][1], .55)
        triple = assignments(21683)
        self.assertEqual([row[0] for row in triple], [
            "Santa Water", "Pummarola", "Spellbinder",
        ])
        self.assertTrue(all(row[1] >= .49 for row in triple))


if __name__ == "__main__":
    unittest.main()
