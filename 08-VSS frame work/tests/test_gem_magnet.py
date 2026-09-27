from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


from vss_framework.detectors import gems as detector


class MagnetModelTests(unittest.TestCase):
    @staticmethod
    def _anchor() -> detector.PlayerAnchor:
        return detector.PlayerAnchor(
            x=0.0,
            y=0.0,
            health_bar_detected=True,
            health_bar_x=0.0,
            health_bar_y=0.0,
            confidence=1.0,
        )

    @staticmethod
    def _state(radius: float = 320.0) -> detector.MagnetState:
        return detector.MagnetState(
            model_enabled=True,
            magnet_value=30.0,
            multiplier_from_base=1.0,
            attractorb_level=0,
            attractorb_multiplier=1.0,
            search_radius_1440p=radius,
            source_event_id="base_magnet_state",
            source_confidence="configured",
            source_needs_review=False,
        )

    @staticmethod
    def _entry(
        *,
        predicted_arrival_frame: int = 15,
    ) -> detector.MagnetEntryEvidence:
        return detector.MagnetEntryEvidence(
            track_id=1,
            color="blue",
            entry_frame=2,
            confirmation_frame=6,
            predicted_arrival_frame=predicted_arrival_frame,
            entry_reason="magnet_boundary_crossing",
            entry_x=300.0,
            entry_y=0.0,
            entry_distance=300.0,
            magnet_radius=320.0,
            radial_speed_pixels_per_frame=20.0,
            mean_template_score=0.95,
            monotonic_fraction=1.0,
            confidence=0.96,
            attractorb_level=0,
            magnet_value=30.0,
            magnet_state_source_event_id="base_magnet_state",
            magnet_state_needs_review=False,
        )

    def test_attractorb_total_multipliers_match_wiki_table(self) -> None:
        expected = (1.0, 1.5, 1.995, 2.49375, 2.9925, 3.980025)
        actual = tuple(detector.attractorb_multiplier(level) for level in range(6))
        self.assertEqual(actual, expected)

    def test_magnet_state_changes_at_attractorb_timestamp(self) -> None:
        cfg = detector.Config()
        change = detector.AttractorbChange(
            video_second=10.0,
            level=2,
            source_event_id="event_1",
            confidence="medium",
            needs_review=True,
        )
        model = detector.MagnetModel(
            enabled=True,
            attractorb_changes=(change,),
            powerup_rank=2,
        )

        before = detector.magnet_state_at(9.999, model, cfg)
        after = detector.magnet_state_at(10.0, model, cfg)

        self.assertEqual(before.attractorb_level, 0)
        self.assertAlmostEqual(before.magnet_value, 30.0 * 1.5625)
        self.assertEqual(after.attractorb_level, 2)
        self.assertAlmostEqual(after.magnet_value, 30.0 * 1.5625 * 1.995)
        self.assertEqual(after.source_event_id, "event_1")
        self.assertTrue(after.source_needs_review)

    def test_disabled_model_preserves_fixed_search_radius(self) -> None:
        cfg = detector.Config()
        change = detector.AttractorbChange(1.0, 5, "event_5", "high", False)
        model = detector.MagnetModel(False, (change,))

        state = detector.magnet_state_at(2.0, model, cfg)

        self.assertAlmostEqual(state.magnet_value, 30.0 * 3.980025)
        self.assertEqual(state.search_radius_1440p, cfg.search_radius_1440p)

    def test_magnet_search_radius_is_capped(self) -> None:
        cfg = detector.Config(magnet_search_radius_cap_1440p=1000.0)
        change = detector.AttractorbChange(1.0, 5, "event_5", "high", False)
        model = detector.MagnetModel(
            enabled=True,
            attractorb_changes=(change,),
            powerup_rank=2,
            character_multiplier=1.25,
        )

        state = detector.magnet_state_at(2.0, model, cfg)

        self.assertEqual(state.search_radius_1440p, 1000.0)

    def test_inventory_reader_retains_or_excludes_review_rows(self) -> None:
        fieldnames = [
            "event_id",
            "video_second",
            "item_after",
            "level_after",
            "confidence",
            "needs_review",
        ]
        rows = [
            {
                "event_id": "a",
                "video_second": "10.0",
                "item_after": "Attractorb",
                "level_after": "1",
                "confidence": "medium",
                "needs_review": "True",
            },
            {
                "event_id": "b",
                "video_second": "20.0",
                "item_after": "Attractorb",
                "level_after": "2",
                "confidence": "high",
                "needs_review": "False",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            all_changes = detector.load_attractorb_changes(path)
            reviewed_only = detector.load_attractorb_changes(
                path,
                include_review_required=False,
            )

        self.assertEqual([change.level for change in all_changes], [1, 2])
        self.assertEqual([change.level for change in reviewed_only], [2])

    def test_tracker_commits_stable_outside_to_inside_crossing(self) -> None:
        cfg = detector.Config()
        tracker = detector.MagnetEntryTracker(cfg, fps=30.0, frame_height=1440)
        anchor = self._anchor()
        state = self._state()

        for frame_index, distance in ((0, 350.0), (2, 300.0), (4, 260.0)):
            tracker.update(
                frame_index,
                [detector.GemDetection("blue", distance, 0.0, 0.95)],
                anchor,
                state,
            )

        self.assertEqual(len(tracker.entries), 1)
        entry = tracker.entries[0]
        self.assertEqual(entry.color, "blue")
        self.assertEqual(entry.entry_frame, 2)
        self.assertEqual(entry.confirmation_frame, 4)
        self.assertEqual(entry.entry_reason, "magnet_boundary_crossing")
        self.assertGreater(entry.predicted_arrival_frame, entry.confirmation_frame)

    def test_tracker_rejects_stationary_gem_first_seen_inside_radius(self) -> None:
        cfg = detector.Config()
        tracker = detector.MagnetEntryTracker(cfg, fps=30.0, frame_height=1440)
        anchor = self._anchor()
        state = self._state()

        for frame_index in (0, 2, 4, 6):
            tracker.update(
                frame_index,
                [detector.GemDetection("green", 250.0, 0.0, 0.95)],
                anchor,
                state,
            )

        self.assertEqual(tracker.entries, [])

    def test_entry_is_assigned_to_one_unique_xp_event(self) -> None:
        rows = [
            {"event_id": 1, "frame_b": 15},
            {"event_id": 2, "frame_b": 35},
        ]

        assignments, diagnostics = detector.pair_magnet_entries_to_events(
            rows,
            [self._entry()],
            detector.Config(),
        )

        self.assertEqual(list(assignments), [1])
        self.assertEqual(assignments[1][0].track_id, 1)
        self.assertEqual(diagnostics["entry_assigned_to_xp_event"], 1)

    def test_entry_is_rejected_when_two_xp_events_are_equally_plausible(self) -> None:
        rows = [
            {"event_id": 1, "frame_b": 12},
            {"event_id": 2, "frame_b": 18},
        ]

        assignments, diagnostics = detector.pair_magnet_entries_to_events(
            rows,
            [self._entry()],
            detector.Config(),
        )

        self.assertEqual(assignments, {})
        self.assertEqual(diagnostics["entry_with_ambiguous_xp_event"], 1)

    def test_color_candidate_requires_unanimous_entry_and_xp_agreement(self) -> None:
        row = {
            "xp_color_hint": "blue",
            "xp_bar_saturated": 0,
            "xp_increase_censored": 0,
            "hud_level_ocr_accepted": 1,
            "previous_xp_event_gap_frames": 10,
            "next_xp_event_gap_frames": 10,
        }
        blue = self._entry()
        cfg = detector.Config()

        self.assertEqual(
            detector.magnet_entry_color_candidate(row, [blue, blue], cfg),
            "blue",
        )
        self.assertEqual(
            detector.magnet_entry_color_candidate(
                row,
                [blue, detector.replace(blue, color="green")],
                cfg,
            ),
            "",
        )
        self.assertEqual(
            detector.magnet_entry_color_candidate(
                {**row, "xp_color_hint": "green"},
                [blue],
                cfg,
            ),
            "",
        )
        self.assertEqual(
            detector.magnet_entry_color_candidate(
                {**row, "next_xp_event_gap_frames": 3},
                [blue],
                cfg,
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
