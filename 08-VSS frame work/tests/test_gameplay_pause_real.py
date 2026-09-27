"""Regressions from the user's real Video 5 transition; never detector inputs."""
import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vss_framework.detectors import gems
from vss_framework.gameplay_pause_export import build_gameplay_pauses
from vss_framework.gameplay_state import level_up_pause_evidence

ROOT = Path(__file__).resolve().parents[1]


class RealPauseXPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT.parent / "00-Videos/video5_Imelda_0_part1.mp4"
        if not source.exists():
            raise unittest.SkipTest("Real Video 5 source required")
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 480)
        cls.cfg = gems.Config()
        cls.frames = []
        for _ in range(101):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Cannot decode bounded real transition fixture")
            cls.frames.append(frame)
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        height, width = cls.frames[0].shape[:2]
        cls.metadata = {"fps": fps, "bar_width": round(width * (cls.cfg.xp_bar_x1_fraction - cls.cfg.xp_bar_x0_fraction)),
                        "frame_count": len(cls.frames), "height": height, "width": width}
        values, qualities = zip(*(gems.measure_xp_bar_progress(f, cls.cfg) for f in cls.frames))
        phases = [level_up_pause_evidence(f) for f in cls.frames]
        cls.anchors = [gems.detect_player_anchor(f, cls.cfg) for f in cls.frames]
        cls.arrays = {
            "progress": np.asarray(values), "quality": np.asarray(qualities),
            "overlay": np.asarray([gems.level_up_overlay_score(f) for f in cls.frames]),
            "hud": np.asarray([gems.gameplay_hud_score(f) for f in cls.frames]),
            "gameplay_phase": np.asarray([p["phase"] for p in phases]),
            "gameplay_paused": np.asarray([p["blocked"] for p in phases]),
            "health_bar_detected": np.asarray([a.health_bar_detected for a in cls.anchors]),
        }
        for key, attr in (("player_anchor_x", "x"), ("player_anchor_y", "y"),
                          ("health_bar_x", "health_bar_x"), ("health_bar_y", "health_bar_y"),
                          ("health_bar_confidence", "confidence")):
            cls.arrays[key] = np.asarray([getattr(a, attr) for a in cls.anchors])
        cls.arrays["pixels"] = np.rint(cls.arrays["progress"] * cls.metadata["bar_width"]).astype(np.int32)
        cls.arrays["delta_pixels"] = np.diff(cls.arrays["pixels"], prepend=cls.arrays["pixels"][0])
        cls.arrays["level_up_transition_guard"] = cls.arrays["gameplay_paused"].copy()
        cls.arrays["xp_bar_reappearance"] = gems.find_xp_bar_reappearances(cls.arrays["pixels"], cls.cfg)
        cls.resets = gems.find_level_resets(cls.arrays, cls.cfg)
        cls.events = gems.find_xp_events(cls.metadata, cls.arrays, cls.resets, cls.cfg)

    def test_real_opening_and_stable_menu(self):
        self.assertFalse(self.arrays["gameplay_paused"][7])  # original frame 487
        self.assertEqual(list(self.arrays["gameplay_phase"][8:14]), ["level_up_transition"] * 6)
        self.assertEqual(self.arrays["gameplay_phase"][14], "level_up_menu")
        self.assertFalse(self.arrays["gameplay_paused"][94])  # original frame 574

    def test_level_boundary_retained_without_inventing_one_gem(self):
        self.assertEqual(self.resets, [8])  # anchored at original frame 488
        gems.validate_transition_event_invariants(self.events, self.arrays, self.resets)
        event = next(e for e in self.events if e.level_up_boundary_candidate)
        model = gems.MagnetModel(enabled=False, attractorb_changes=())
        state = gems.magnet_state_at(event.video_time_b, model, self.cfg)
        row, _ = gems.summarize_pair(event, self.frames[0].shape, [], [], [], [],
                                    self.anchors[event.frame_a], self.anchors[event.frame_b], state, self.cfg)
        self.assertEqual(row["collected_gems_total"], 0)
        self.assertEqual(row["unresolved_collected_gems"], 0)
        self.assertEqual(row["pickup_count_status"], "unobserved_at_level_boundary")
        self.assertTrue(row["level_up_boundary_candidate"])
        self.assertFalse(any(self.arrays["gameplay_paused"][e.frame_b] for e in self.events if not e.level_up_boundary_candidate))

    def test_export_keeps_missing_pause_measurements_and_resume_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signal.csv"
            gems.write_signal_diagnostics(path, self.metadata, self.arrays, self.events, self.resets,
                                          gems.MagnetModel(enabled=False, attractorb_changes=()), self.cfg)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        for row in rows:
            if row["gameplay_paused"] == "1":
                self.assertEqual(row["xp_progress"], "")
                self.assertEqual(row["xp_delta_pixels"], "")
        for i in range(1, len(rows)):
            if rows[i-1]["gameplay_paused"] == "1" and rows[i]["gameplay_paused"] == "0":
                self.assertEqual(rows[i]["xp_delta_pixels"], "")
        pauses = build_gameplay_pauses(rows, self.metadata["fps"])
        self.assertEqual(pauses["intervals"][0]["startFrame"], 8)
        self.assertIn("level_up_menu", pauses["intervals"][0]["phases"])


if __name__ == "__main__":
    unittest.main()
