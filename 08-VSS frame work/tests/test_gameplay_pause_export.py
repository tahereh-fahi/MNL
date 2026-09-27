"""Pause-export policy tests; saved signals are replay evidence, not truth."""

import csv
import hashlib
import json
from pathlib import Path
import unittest

from vss_framework.gameplay_pause_export import build_gameplay_pauses


FRAMEWORK = Path(__file__).resolve().parents[1]
VIDEO5_RUN = (
    FRAMEWORK / "runs"
    / "video5_imelda_0_part1_pause_validation_full_20260918_022656_89992"
)


def row(frame: int, paused: bool, phase: str = "no_level_up_overlay") -> dict:
    return {
        "frame_index": frame,
        "gameplay_paused": int(paused),
        "gameplay_phase": phase,
        "gameplay_state_version": "test-v1",
    }


class GameplayPauseExportTests(unittest.TestCase):
    def test_merges_one_frame_orphan_transition_before_menu(self):
        rows = [
            row(10, True, "level_up_transition"),
            row(11, False),
            row(12, True, "level_up_transition"),
            row(13, True, "level_up_menu"),
            row(14, False),
        ]
        intervals = build_gameplay_pauses(rows, 30.0)["intervals"]
        self.assertEqual(len(intervals), 1)
        interval = intervals[0]
        self.assertEqual((interval["startFrame"], interval["endFrameExclusive"]), (10, 14))
        self.assertEqual(interval["mergedGapFrames"], 1)
        self.assertEqual(len(interval["mergedComponents"]), 2)
        self.assertEqual(
            interval["mergeReason"],
            "short_transition_fragment_across_one_frame_gap",
        )

    def test_does_not_merge_two_substantive_adjacent_menus(self):
        rows = [
            row(10, True, "level_up_transition"),
            row(11, True, "level_up_menu"),
            row(12, True, "level_up_menu"),
            row(13, False),
            row(14, True, "level_up_transition"),
            row(15, True, "level_up_menu"),
            row(16, True, "level_up_menu"),
        ]
        intervals = build_gameplay_pauses(rows, 30.0)["intervals"]
        self.assertEqual(len(intervals), 2)
        self.assertNotIn("mergedComponents", intervals[0])

    def test_merges_treasure_components_across_brief_animation_dropouts(self):
        rows = []
        rows.extend(row(frame, True, "treasure_menu") for frame in range(100, 106))
        rows.extend(row(frame, False) for frame in range(106, 112))
        rows.extend(row(frame, True, "treasure_menu") for frame in range(112, 115))
        rows.extend(row(frame, False) for frame in range(115, 120))
        rows.extend(row(frame, True, "treasure_transition") for frame in range(120, 125))

        intervals = build_gameplay_pauses(rows, 30.0)["intervals"]

        self.assertEqual(len(intervals), 1)
        self.assertEqual(
            (intervals[0]["startFrame"], intervals[0]["endFrameExclusive"]),
            (100, 125),
        )
        self.assertEqual(intervals[0]["eventType"], "treasure_pause")
        self.assertEqual(intervals[0]["mergedGapFrames"], 11)
        self.assertEqual(len(intervals[0]["mergedComponents"]), 3)
        self.assertEqual(
            intervals[0]["mergeReason"],
            "treasure_components_across_brief_visual_dropout",
        )

    def test_does_not_merge_treasure_components_across_long_gap(self):
        rows = [row(10, True, "treasure_menu")]
        rows.extend(row(frame, False) for frame in range(11, 20))
        rows.append(row(20, True, "treasure_menu"))
        self.assertEqual(len(build_gameplay_pauses(rows, 30.0)["intervals"]), 2)

    def test_treasure_menu_is_not_exported_as_level_up_pause(self):
        rows = [
            row(20, True, "interruption_transition"),
            row(21, True, "treasure_transition"),
            row(22, True, "treasure_menu"),
            row(23, False),
        ]
        interval = build_gameplay_pauses(rows, 30.0)["intervals"][0]
        self.assertEqual(interval["eventType"], "treasure_pause")
        self.assertEqual(interval["eventId"], "treasure_pause_20")

    def test_transition_without_menu_stays_unresolved(self):
        interval = build_gameplay_pauses(
            [row(30, True, "interruption_transition")], 30.0
        )["intervals"][0]
        self.assertEqual(interval["eventType"], "unresolved_interruption_pause")

    def test_merges_multiframe_leading_transition_before_level_up(self):
        rows = [row(frame, True, "level_up_transition") for frame in range(10, 15)]
        rows.extend(row(frame, False) for frame in range(15, 18))
        rows.extend(row(frame, True, "level_up_menu") for frame in range(18, 22))
        interval = build_gameplay_pauses(rows, 30.0)["intervals"][0]
        self.assertEqual(interval["eventType"], "level_up_pause")
        self.assertEqual((interval["startFrame"], interval["endFrameExclusive"]), (10, 22))
        self.assertEqual(interval["mergeReason"], "leading_transition_fragment_before_level_up_menu")

    def test_merges_six_frame_transition_across_four_frame_gap(self):
        rows = [row(frame, True, "level_up_transition") for frame in range(10, 16)]
        rows.extend(row(frame, False) for frame in range(16, 20))
        rows.extend(row(frame, True, "level_up_menu") for frame in range(20, 24))
        intervals = build_gameplay_pauses(rows, 30.0)["intervals"]
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["eventType"], "level_up_pause")
        self.assertEqual(
            (intervals[0]["startFrame"], intervals[0]["endFrameExclusive"]),
            (10, 24),
        )
        self.assertEqual(intervals[0]["mergedGapFrames"], 4)

    def test_merges_trailing_transition_fragment_after_level_up_menu(self):
        rows = [row(frame, True, "level_up_menu") for frame in range(10, 20)]
        rows.append(row(20, False))
        rows.extend(row(frame, True, "level_up_transition") for frame in range(21, 26))
        intervals = build_gameplay_pauses(rows, 30.0)["intervals"]
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["eventType"], "level_up_pause")
        self.assertEqual(
            (intervals[0]["startFrame"], intervals[0]["endFrameExclusive"]),
            (10, 26),
        )
        self.assertEqual(
            intervals[0]["mergeReason"],
            "trailing_transition_fragment_after_level_up_menu",
        )

    def test_merges_transition_dominant_onset_with_stable_level_up_menu(self):
        rows = [row(frame, True, "level_up_transition") for frame in range(10, 15)]
        rows.append(row(15, True, "level_up_menu"))
        rows.append(row(16, False))
        rows.extend(row(frame, True, "level_up_menu") for frame in range(17, 30))
        intervals = build_gameplay_pauses(rows, 30.0)["intervals"]
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["eventType"], "level_up_pause")
        self.assertEqual(
            intervals[0]["mergeReason"],
            "transition_dominant_level_up_onset_before_stable_menu",
        )

    def test_accepted_chest_bridges_long_animation_occlusion(self):
        rows = [row(90, True, "treasure_transition")]
        rows.extend(row(frame, True, "treasure_menu") for frame in range(100, 111))
        rows.extend(row(frame, False) for frame in range(111, 240))
        rows.extend(row(frame, True, "treasure_menu") for frame in range(240, 251))
        chest = {
            "event_id": "chest-1", "publication_status": "auto_accepted",
            "time_lower_ms": 3_000, "time_upper_ms": 8_500,
        }
        intervals = build_gameplay_pauses(rows, 30.0, [chest])["intervals"]
        treasures = [row for row in intervals if row["eventType"] == "treasure_pause"]
        self.assertEqual(len(treasures), 1)
        self.assertEqual((treasures[0]["startMs"], treasures[0]["endMs"]), (3_000, 8_500))
        self.assertEqual(treasures[0]["sourceChestEventId"], "chest-1")
        self.assertEqual(len(treasures[0]["mergedComponents"]), 3)

    def test_unmatched_treasure_visual_fragment_is_conservative(self):
        rows = [row(10, True, "treasure_transition")]
        rows.extend(row(frame, False) for frame in range(11, 100))
        rows.extend(row(frame, True, "treasure_menu") for frame in range(100, 110))
        chest = {
            "event_id": "chest-1", "publication_status": "auto_accepted",
            "time_lower_ms": 3_300, "time_upper_ms": 3_700,
        }
        intervals = build_gameplay_pauses(rows, 30.0, [chest])["intervals"]
        self.assertEqual(intervals[0]["eventType"], "unresolved_interruption_pause")
        self.assertTrue(intervals[0]["unmatchedTreasureVisualEvidence"])
        self.assertEqual(intervals[1]["eventType"], "treasure_pause")

    def test_inventory_event_joins_one_level_up_and_its_fragmented_collapse(self):
        rows = []
        rows.extend(row(frame, True, "level_up_menu") for frame in range(100, 111))
        rows.extend(row(frame, False) for frame in range(111, 120))
        rows.extend(row(frame, True, "level_up_transition") for frame in range(120, 124))
        rows.extend(row(frame, False) for frame in range(124, 134))
        rows.extend(row(frame, True, "level_up_transition") for frame in range(134, 137))
        event = {
            "event_id": "inventory-1",
            "event_source": "level_up",
            "video_second": 110 / 30,
        }
        intervals = build_gameplay_pauses(
            rows, 30.0, level_up_events=[event]
        )["intervals"]
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["eventType"], "level_up_pause")
        self.assertEqual(
            (intervals[0]["startFrame"], intervals[0]["endFrameExclusive"]),
            (100, 137),
        )
        self.assertEqual(intervals[0]["sourceInventoryEventId"], "inventory-1")
        self.assertEqual(intervals[0]["sourceInventoryEventIds"], ["inventory-1"])

    def test_inventory_event_does_not_absorb_later_stable_level_up_menu(self):
        rows = [row(frame, True, "level_up_menu") for frame in range(100, 111)]
        rows.extend(row(frame, False) for frame in range(111, 140))
        rows.extend(row(frame, True, "level_up_menu") for frame in range(140, 151))
        event = {
            "event_id": "inventory-1",
            "event_source": "level_up",
            "video_second": 110 / 30,
        }
        intervals = build_gameplay_pauses(
            rows, 30.0, level_up_events=[event]
        )["intervals"]
        self.assertEqual(len(intervals), 2)
        self.assertNotIn("sourceInventoryEventId", intervals[1])

    def test_hash_verified_video5_replay_yields_thirteen_intervals(self):
        manifest_path = VIDEO5_RUN / "run_manifest.json"
        if not manifest_path.is_file():
            self.skipTest("Saved full Video 5 run is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage = manifest["stages"]["initial_gems"]
        relative = next(
            name for name in stage["outputs"] if name.endswith("_xp_frame_signal.csv")
        )
        signal = VIDEO5_RUN / stage["directory"] / relative
        self.assertEqual(
            hashlib.sha256(signal.read_bytes()).hexdigest(), stage["outputs"][relative]
        )
        with signal.open(newline="", encoding="utf-8") as handle:
            payload = build_gameplay_pauses(csv.DictReader(handle), 30.0)
        self.assertEqual(len(payload["intervals"]), 13)
        interval = next(
            item for item in payload["intervals"] if item["startFrame"] == 2194
        )
        self.assertEqual(interval["endFrameExclusive"], 2232)
        self.assertEqual(interval["mergedGapFrames"], 1)
        self.assertEqual(
            [(part["startFrame"], part["endFrameExclusive"])
             for part in interval["mergedComponents"]],
            [(2194, 2195), (2196, 2232)],
        )


if __name__ == "__main__":
    unittest.main()
