from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


from vss_framework.detectors.inventory import (  # noqa: E402
    apply_initial_weapon_timeline,
    InventorySnapshotEntry,
    infer_retrospective_inventory_change,
    infer_unique_retrospective_upgrade_from_owned_slots,
    inventory_level_pip_counts,
    inventory_selection_exclusion_reason,
    level_up_option_rectangles,
    level_transition_windows,
    level_up_pause_intervals_from_signal,
    menu_content_changed,
    rebuild_inventory_progression,
    stabilize_inventory_snapshots,
)
from vss_framework.inventory_reconciliation import (  # noqa: E402
    apply_initial_level_observations,
)
from vss_framework.detectors.weapons import HUDLayout


def synthetic_level_up_menu(option_count: int) -> np.ndarray:
    frame = np.zeros((1_440, 2_560, 3), dtype=np.uint8)
    gold = (20, 150, 220)
    outer = (680, 120, 1_000, 1_000)
    cv2.rectangle(
        frame,
        (outer[0], outer[1]),
        (outer[0] + outer[2], outer[1] + outer[3]),
        gold,
        6,
    )
    for index in range(option_count):
        top = 300 + index * 210
        cv2.rectangle(frame, (720, top), (1_640, top + 150), gold, 6)
    return frame


class LevelUpOptionRectangleTests(unittest.TestCase):
    def test_detects_two_choice_late_game_menu(self) -> None:
        rows = level_up_option_rectangles(synthetic_level_up_menu(2))
        self.assertEqual(len(rows), 2)

    def test_keeps_four_choice_menu_support(self) -> None:
        rows = level_up_option_rectangles(synthetic_level_up_menu(4))
        self.assertEqual(len(rows), 4)

    def test_detects_single_choice_late_game_menu(self) -> None:
        rows = level_up_option_rectangles(synthetic_level_up_menu(1))
        self.assertEqual(len(rows), 1)


class LevelUpSearchWindowTests(unittest.TestCase):
    def test_final_pause_is_searchable_without_a_later_xp_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            signal = Path(directory) / "signal.csv"
            signal.write_text(
                "frame_index,gameplay_paused\n"
                + "".join(
                    f"{frame},{int(3 <= frame < 6 or 9 <= frame < 14)}\n"
                    for frame in range(15)
                ),
                encoding="utf-8",
            )
            intervals = level_up_pause_intervals_from_signal(signal, 1.0)
        self.assertEqual(intervals, [(3.0, 6.0), (9.0, 14.0)])
        # The XP data has no event following the second pause.
        xp_events = pd.DataFrame([
            {"video_time_b": 3.0, "reset_inferred_level": 1},
            {"video_time_b": 7.0, "reset_inferred_level": 2},
        ])
        windows = level_transition_windows(
            xp_events, start_second=0.0, end_second=15.0,
            pause_intervals=intervals,
        )
        self.assertEqual(len(windows), 2)
        self.assertTrue(windows[1][0] < 9.0 < 13.0 < windows[1][1])

    def test_boundary_event_fallback_does_not_need_later_xp(self) -> None:
        xp_events = pd.DataFrame([
            {"video_time_b": 10.0, "reset_inferred_level": 1,
             "level_up_boundary_candidate": 1},
        ])
        windows = level_transition_windows(
            xp_events, start_second=0.0, end_second=20.0,
        )
        self.assertEqual(len(windows), 1)
        self.assertLessEqual(windows[0][0], 10.0)
        self.assertGreaterEqual(windows[0][1], 15.0)


class InventorySelectionGuardTests(unittest.TestCase):
    def test_rejects_match_for_item_already_at_normal_max(self) -> None:
        reason = inventory_selection_exclusion_reason(
            "Pummarola",
            "passive_item",
            {"weapon": [], "passive_item": ["Pummarola"]},
            {"Pummarola": 5},
            cursor_score=100.0,
            needs_review=False,
        )
        self.assertEqual(reason, "item_already_at_normal_max")

    def test_rejects_uncertain_new_weapon_after_six_slots(self) -> None:
        reason = inventory_selection_exclusion_reason(
            "Garlic",
            "weapon",
            {"weapon": ["a", "b", "c", "d", "e", "f"], "passive_item": []},
            {},
            cursor_score=100.0,
            needs_review=True,
        )
        self.assertEqual(reason, "low_confidence_beyond_standard_slots")

    def test_keeps_valid_late_game_upgrade(self) -> None:
        reason = inventory_selection_exclusion_reason(
            "Attractorb",
            "passive_item",
            {"weapon": [], "passive_item": ["Attractorb"]},
            {"Attractorb": 4},
            cursor_score=100.0,
            needs_review=False,
        )
        self.assertIsNone(reason)

    def test_rejects_menu_without_confirmed_selection_pointer(self) -> None:
        reason = inventory_selection_exclusion_reason(
            "Peachone",
            "weapon",
            {"weapon": ["Peachone"], "passive_item": []},
            {"Peachone": 3},
            cursor_score=0.0,
            needs_review=True,
        )
        self.assertEqual(reason, "selection_pointer_unconfirmed")


class StackedMenuBoundaryTests(unittest.TestCase):
    def test_detects_abrupt_option_icon_change(self) -> None:
        previous = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        current = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
        self.assertTrue(menu_content_changed(previous, current))

    def test_keeps_same_menu_continuous(self) -> None:
        previous = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        current = [np.array([0.98, 0.02]), np.array([0.01, 0.99])]
        self.assertFalse(menu_content_changed(previous, current))


class RetrospectiveInventoryTests(unittest.TestCase):
    def test_stable_timeline_corrects_single_frame_opening_weapon_identity(self) -> None:
        opening = pd.DataFrame([
            {"slot": 1, "occupied": True, "weapon": "Magic Wand",
             "suggested_weapon": "Magic Wand", "confidence": "high",
             "needs_review": False},
            {"slot": 4, "occupied": True, "weapon": "Party Popper",
             "suggested_weapon": "Party Popper", "confidence": "medium",
             "needs_review": False},
        ])
        timeline = pd.DataFrame([
            {"video_second": 5.0, "screen_state": "gameplay",
             "weapon_1": "Magic Wand", "confidence_1": "high",
             "weapon_4": "Fire Wand", "confidence_4": "high",
             "weapon_2": "", "confidence_2": "empty"},
        ])
        corrected = apply_initial_weapon_timeline(
            opening, timeline, initial_second=5.0
        )
        self.assertEqual(
            corrected.loc[corrected.slot.eq(4), "weapon"].iloc[0], "Fire Wand"
        )

    def test_level_pips_back_out_prior_chest_upgrade(self) -> None:
        events = pd.DataFrame([
            {"video_second": 5.0, "event_source": "initial_state",
             "frame_number": 150,
             "event_type": "initial_state", "item_type": "weapon",
             "item_after": "Magic Wand", "level_after": 1,
             "inference_method": "first_gameplay_inventory"},
            {"video_second": 5.0, "event_source": "initial_state",
             "frame_number": 150,
             "event_type": "initial_state", "item_type": "weapon",
             "item_after": "Fire Wand", "level_after": 1,
             "inference_method": "first_gameplay_inventory"},
            {"video_second": 16.0, "event_source": "treasure_chest",
             "frame_number": 480,
             "event_type": "upgrade", "item_type": "weapon",
             "item_after": "Fire Wand", "level_after": "",
             "inference_method": ""},
            {"video_second": 24.9, "event_source": "level_up",
             "frame_number": 747,
             "event_type": "upgrade", "item_type": "weapon",
             "item_after": "Fire Wand", "level_after": 2,
             "inference_method": "final_selected_option"},
        ])
        events["video"] = "video5_part2"
        events["item_before"] = ""
        events["needs_review"] = False
        observations = pd.DataFrame([
            {"video_second": 23.2, "item_type": "weapon", "slot": 1,
             "item_name": "Magic Wand", "observed_level": 5},
            {"video_second": 23.2, "item_type": "weapon", "slot": 4,
             "item_name": "Fire Wand", "observed_level": 2},
        ])
        adjusted = apply_initial_level_observations(events, observations)
        initial = adjusted.loc[adjusted.event_source.eq("initial_state")]
        self.assertEqual(
            int(initial.loc[initial.item_after.eq("Magic Wand"), "level_after"].iloc[0]),
            5,
        )
        self.assertEqual(
            int(initial.loc[initial.item_after.eq("Fire Wand"), "level_after"].iloc[0]),
            1,
        )
        rebuilt = rebuild_inventory_progression(adjusted)
        fire = rebuilt.loc[rebuilt.item_after.eq("Fire Wand")]
        self.assertEqual(fire.level_after.astype(int).tolist(), [1, 2, 3])

    def test_rebuilt_ids_use_the_actual_video_name(self) -> None:
        events = pd.DataFrame([
            {"event_id": "old", "video": "video_3_run01", "video_second": 1,
             "frame_number": 30, "event_source": "initial_state",
             "event_type": "initial_state", "item_type": "weapon",
             "item_before": "", "item_after": "Axe", "needs_review": False},
        ])
        rebuilt = rebuild_inventory_progression(events)
        self.assertEqual(
            rebuilt.loc[0, "event_id"], "video_3_run01_inventory_0001"
        )

    def test_drops_retrospective_upgrade_that_conflicts_with_evolution(self) -> None:
        events = pd.DataFrame([
            {"event_id": "1", "video_second": 10, "frame_number": 300, "event_source": "initial_state", "event_type": "initial_state", "item_type": "weapon", "item_before": "", "item_after": "Pentagram", "needs_review": False},
            {"event_id": "2", "video_second": 20, "frame_number": 600, "event_source": "treasure_chest", "event_type": "evolution", "item_type": "weapon", "item_before": "Pentagram", "item_after": "Gorgeous Moon", "needs_review": False},
            {"event_id": "3", "video_second": 30, "frame_number": 900, "event_source": "level_up_retrospective", "event_type": "upgrade", "item_type": "weapon", "item_before": "Pentagram", "item_after": "Pentagram", "needs_review": False},
        ])
        rebuilt = rebuild_inventory_progression(events)
        self.assertNotIn("3", set(rebuilt["event_id"].astype(str)))
        self.assertNotIn("Pentagram", set(rebuilt.loc[rebuilt["video_second"].eq(30), "item_after"]))

    def test_infers_unique_upgrade_when_selected_icon_is_unreadable(self) -> None:
        before = [
            InventorySnapshotEntry("weapon", 1, "", 6, "pip_geometry"),
            InventorySnapshotEntry("weapon", 2, "", 4, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 1, "", 3, "pip_geometry"),
        ]
        after = [
            InventorySnapshotEntry("weapon", 1, "", 6, "pip_geometry"),
            InventorySnapshotEntry("weapon", 2, "", 5, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 1, "", 3, "pip_geometry"),
        ]
        change = infer_unique_retrospective_upgrade_from_owned_slots(
            before,
            after,
            {"weapon": ["Magic Wand", "King Bible"], "passive_item": ["Crown"]},
            after,
        )
        self.assertIsNotNone(change)
        self.assertEqual(change.item_name, "King Bible")
        self.assertEqual(change.level_before, 4)
        self.assertEqual(change.level_after, 5)

    def test_rejects_multiple_slot_increases_without_selected_identity(self) -> None:
        before = [
            InventorySnapshotEntry("weapon", 1, "", 4, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 1, "", 3, "pip_geometry"),
        ]
        after = [
            InventorySnapshotEntry("weapon", 1, "", 5, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 1, "", 4, "pip_geometry"),
        ]
        self.assertIsNone(
            infer_unique_retrospective_upgrade_from_owned_slots(
                before,
                after,
                {"weapon": ["Magic Wand"], "passive_item": ["Crown"]},
                after,
            )
        )

    def test_rejects_one_snapshot_pip_spike_in_reward_only_menu(self) -> None:
        before = [InventorySnapshotEntry("passive_item", 4, "", 1, "pip_geometry")]
        after = [InventorySnapshotEntry("passive_item", 4, "", 2, "pip_geometry")]
        later = [InventorySnapshotEntry("passive_item", 4, "", 1, "pip_geometry")]
        self.assertIsNone(
            infer_unique_retrospective_upgrade_from_owned_slots(
                before,
                after,
                {"weapon": [], "passive_item": ["a", "b", "c", "Armor"]},
                later,
            )
        )

    def test_infers_one_level_increase_for_the_visible_candidate(self) -> None:
        before = [
            InventorySnapshotEntry("passive_item", 6, "Attractorb", 4, "high"),
            InventorySnapshotEntry("passive_item", 5, "Spinach", 5, "high"),
        ]
        after = [
            InventorySnapshotEntry("passive_item", 6, "Attractorb", 5, "high"),
            InventorySnapshotEntry("passive_item", 5, "Spinach", 5, "high"),
        ]
        change = infer_retrospective_inventory_change(
            before, after, expected_item="Attractorb"
        )
        self.assertIsNotNone(change)
        self.assertEqual(change.event_type, "upgrade")
        self.assertEqual(change.level_before, 4)
        self.assertEqual(change.level_after, 5)

    def test_rejects_multiple_inventory_changes(self) -> None:
        before = [
            InventorySnapshotEntry("passive_item", 5, "Spinach", 4, "high"),
            InventorySnapshotEntry("passive_item", 6, "Attractorb", 4, "high"),
        ]
        after = [
            InventorySnapshotEntry("passive_item", 5, "Spinach", 5, "high"),
            InventorySnapshotEntry("passive_item", 6, "Attractorb", 5, "high"),
        ]
        self.assertIsNone(
            infer_retrospective_inventory_change(
                before, after, expected_item="Attractorb"
            )
        )

    def test_ignores_noise_in_the_other_inventory_row(self) -> None:
        before = [
            InventorySnapshotEntry("weapon", 1, "", 6, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 6, "", 3, "pip_geometry"),
        ]
        after = [
            InventorySnapshotEntry("weapon", 1, "", 8, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 6, "", 4, "pip_geometry"),
        ]
        change = infer_retrospective_inventory_change(
            before,
            after,
            expected_item="Attractorb",
            expected_item_type="passive_item",
            expected_slot=6,
        )
        self.assertIsNotNone(change)
        self.assertEqual(change.item_name, "Attractorb")
        self.assertEqual(change.level_after, 4)

    def test_ignores_missing_pips_but_not_a_second_positive_change(self) -> None:
        before = [
            InventorySnapshotEntry("passive_item", 2, "", 4, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 4, "", 4, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 6, "", 3, "pip_geometry"),
        ]
        after = [
            InventorySnapshotEntry("passive_item", 2, "", 3, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 4, "", 5, "pip_geometry"),
            InventorySnapshotEntry("passive_item", 6, "", 4, "pip_geometry"),
        ]
        self.assertIsNone(
            infer_retrospective_inventory_change(
                before,
                after,
                expected_item="Attractorb",
                expected_item_type="passive_item",
                expected_slot=6,
            )
        )

    def test_does_not_infer_a_new_item_from_a_previously_missing_pip(self) -> None:
        self.assertIsNone(
            infer_retrospective_inventory_change(
                [],
                [InventorySnapshotEntry("weapon", 6, "", 1, "pip_geometry")],
                expected_item="Garlic",
                expected_item_type="weapon",
                expected_slot=6,
            )
        )

    def test_infers_from_one_changed_passive_slot_when_identity_match_is_low(self) -> None:
        before = [
            InventorySnapshotEntry("passive_item", 6, "", 3, "pip_geometry")
        ]
        after = [
            InventorySnapshotEntry("passive_item", 6, "", 4, "pip_geometry")
        ]
        change = infer_retrospective_inventory_change(
            before,
            after,
            expected_item="Attractorb",
            expected_item_type="passive_item",
            expected_slot=None,
        )
        self.assertIsNotNone(change)
        self.assertEqual(change.item_name, "Attractorb")
        self.assertEqual(change.slot, 6)

    def test_counts_level_pips_by_inventory_row_and_slot(self) -> None:
        frame = np.zeros((420, 640, 3), dtype=np.uint8)
        layout = HUDLayout(40, 30, 40, 44, 6, 36, 0, 560, 32)
        gold = (20, 150, 220)
        for band_y, levels in ((100, [2, 3, 1, 0, 0, 0]), (220, [1, 4, 0, 0, 0, 0])):
            for slot, level in enumerate(levels):
                center_x = layout.x + layout.slot_size // 2 + slot * layout.step_x
                for pip in range(level):
                    row = pip // 3
                    column = pip % 3
                    x = center_x - 10 + column * 10
                    y = band_y + row * 12
                    cv2.rectangle(frame, (x, y), (x + 7, y + 7), gold, -1)
        self.assertEqual(
            inventory_level_pip_counts(frame, layout),
            [[2, 3, 1, 0, 0, 0], [1, 4, 0, 0, 0, 0]],
        )

    def test_stabilizes_pip_count_across_noisy_menu_frames(self) -> None:
        snapshots = [
            [InventorySnapshotEntry("passive_item", 6, "", level, "pip_geometry")]
            for level in (4, 4, 3, 4, 5)
        ]
        stable = stabilize_inventory_snapshots(snapshots)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].level, 4)


if __name__ == "__main__":
    unittest.main()
