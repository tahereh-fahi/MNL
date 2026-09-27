from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from vss_framework.catalog import load_event_catalog
from vss_framework.adapters.signals import adapt_automated_signals
from vss_framework.exporters.five_second import write_five_second_windows
from vss_framework.models import (
    CanonicalEvent,
    EvidenceGrade,
    FrameObservation,
    PublicationStatus,
    TemporalPrecision,
    Visibility,
)
from vss_framework.detectors.screen_state import ScreenStateDetector
from vss_framework.detectors.xp_bar import XPBarDetector
from vss_framework.detectors.legacy_gem_xp import build_legacy_gem_command
from vss_framework.detectors.legacy_hud import build_legacy_hud_command
from vss_framework.detectors.legacy_inventory import build_legacy_inventory_command
from vss_framework.detectors.instant_reward import classify_instant_reward_icon
from vss_framework.video import FramePacket
from vss_framework.level_up_transactions import resolve_level_up_transactions
from vss_framework.telemetry_scan import has_attached_digit_ocr_artifact, is_gold_fever_monotonic_coin_candidate, is_leading_digit_ocr_substitution, is_plausible_coin_increase, select_coin_candidate
from vss_framework.health_calibration import estimate_full_health_width, extract_health_events, health_percent, reject_isolated_short_widths
from vss_framework.models import SignalObservation
from vss_framework.health_attribution import attribute_recovery_event
from vss_framework.world_pickups import PickupDetection, associate_tracks, non_maximum_suppression, resolve_pickup_events
from vss_framework.world_pickup_effects import extract_abrupt_healing_candidates, extract_currency_gains
from vss_framework.freeze_effect import mark_freeze_like_rows, resolve_freeze_intervals
from vss_framework.vacuum_flow import mark_vacuum_like_rows, radial_flow_features, resolve_vacuum_intervals
from vss_framework.gold_fever import mark_gold_fever_rows, resolve_gold_fever_intervals
from vss_framework.chest_lifecycle import mark_chest_rows, resolve_chest_intervals
from vss_framework.chest_rewards import (
    _reward_detail_icon_crop,
    advance_reconciled_level,
    orb_identity_acceptance,
    resolve_assignments,
    resolve_consensus_assignments,
)
from vss_framework.menu_actions import parse_action_counter_text, resolve_action_counter_drops
from vss_framework.status_events import resolve_status_intervals
from vss_framework.game_level import (
    apply_gameplay_interruptions,
    build_gameplay_interruption_intervals,
    build_xp_progress,
    merge_xp_progress,
    reconcile_game_levels_from_completed_selections,
)
from vss_framework.counter_repair import (
    corroborate_repeated_counter_points,
    repair_cumulative_counter_values,
    repair_persistent_leading_place_shift,
    restore_truncated_counter_from_raw,
)
from vss_framework.dashboard_projection import (
    build_five_second_counter_windows,
    build_gold_counter_trajectory,
    build_reward_trajectory,
    centered_window_mean,
    project_gem_events,
    project_inventory_events,
)
from vss_framework.dashboard_release import build_dashboard_release, canonicalize_lucky_level_ups, canonicalize_weapon_evolutions, merge_dashboard_releases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChestRewardStateTests(unittest.TestCase):
    def test_reported_level_is_lower_bound_after_chest_upgrade(self) -> None:
        self.assertEqual(advance_reconciled_level(2, 2), 3)

    def test_reported_level_can_initialize_late_stream(self) -> None:
        self.assertEqual(advance_reconciled_level(0, 7), 7)


class GameLevelProgressTests(unittest.TestCase):
    def _row(self, time: float, level: int, before: int, after: int, boundary: bool = True) -> dict:
        return {
            "video_time_b": str(time), "frame_b": str(round(time * 30)),
            "hud_level": str(level), "inferred_level": str(level),
            "xp_quality_b": "0.99", "xp_bar_progress_b_effective_percent": "10",
            "hud_level_before": str(before), "hud_level_after": str(after),
            "level_up_boundary_candidate": "1" if boundary else "0",
            "needs_review": "0",
        }

    def test_game_level_is_consecutive_and_rejects_regressions_and_jumps(self) -> None:
        rows = [
            self._row(1, 1, 1, 2),
            self._row(2, 2, 2, 3),
            self._row(3, 8, 8, 9),
            self._row(4, 3, 3, 4),
            self._row(5, 2, 2, 3),
        ]
        result = build_xp_progress(rows, initial_level=1)
        self.assertEqual([event["toLevel"] for event in result["levelChanges"]], [2, 3, 4])
        self.assertEqual(len(result["unresolvedLevelChanges"]), 2)
        self.assertIn("weapon and passive-item levels are excluded", result["levelMeaning"])

    def test_same_level_gap_can_use_lower_reliability_measured_bridge(self) -> None:
        first = self._row(1, 2, 2, 2, boundary=False)
        second = self._row(5, 2, 2, 2, boundary=False)
        first["xp_bar_progress_b_effective_percent"] = "10"
        second["xp_bar_progress_b_effective_percent"] = "60"
        frame_rows = [{
            "video_time": "3", "frame_index": "90", "xp_measurement_valid": "1",
            "xp_progress_percent": "35", "xp_quality": ".7",
        }]
        result = build_xp_progress([first, second], frame_rows=frame_rows)
        self.assertEqual(len(result["lessReliableSegments"]), 1)
        self.assertEqual(result["lessReliableSegments"][0]["points"][1]["reliability"], "lower")

    def test_xp_interval_keeps_both_measured_endpoints(self) -> None:
        row = self._row(1, 2, 2, 2, boundary=False)
        row.update({
            "video_time_a": ".5", "frame_a": "15", "xp_quality_a": ".98",
            "xp_bar_progress_a_percent": "12", "reset_inferred_level": "2",
        })
        result = build_xp_progress([row])
        self.assertEqual(
            [
                (point["mediaTimeMs"], point["progressPercent"])
                for point in result["points"]
                if point.get("endpoint")
            ],
            [(500, 12.0), (1000, 10.0)],
        )

    def test_reset_sequence_wins_over_disagreeing_hud_ocr(self) -> None:
        row = self._row(1, 19, 15, 15, boundary=False)
        row.update({
            "inferred_level": "15", "reset_inferred_level": "15",
            "hud_level_ocr_accepted": "1", "level_agrees_with_reset_inference": "0",
            "video_time_a": ".5", "frame_a": "15", "xp_quality_a": ".98",
            "xp_bar_progress_a_percent": "8",
        })
        result = build_xp_progress([row])
        self.assertEqual({point["hudLevel"] for point in result["points"]}, {15})
        self.assertEqual(
            {point["levelSource"] for point in result["points"] if point.get("endpoint")},
            {"reset_inference"},
        )

    def test_completed_level_gets_100_and_next_level_gets_zero_boundary_anchors(self) -> None:
        row = self._row(2, 1, 1, 2)
        row.update({
            "video_time_a": "1.9", "frame_a": "57", "xp_quality_a": "1",
            "xp_bar_progress_a_percent": "100", "reset_inferred_level": "1",
        })
        result = build_xp_progress([row], initial_level=1)
        anchors = [
            point for point in result["points"]
            if point.get("boundaryAnchor") and point["mediaTimeMs"] == 2_000
        ]
        self.assertEqual(
            [(point["hudLevel"], point["progressPercent"], point["boundaryAnchor"]) for point in anchors],
            [(1, 100.0, "level_end"), (2, 0.0, "level_start")],
        )
        self.assertTrue(anchors[1]["breakBefore"])

    def test_reset_sequence_corrects_misread_boundary_levels(self) -> None:
        row = self._row(2, 5, 5, 6)
        row.update({
            "inferred_level": "4", "reset_inferred_level": "4",
            "video_time_a": "1.9", "frame_a": "57", "xp_quality_a": "1",
            "xp_bar_progress_a_percent": "100",
        })
        result = build_xp_progress([row], initial_level=4)
        self.assertEqual(
            [(event["fromLevel"], event["toLevel"]) for event in result["levelChanges"]],
            [(4, 5)],
        )
        self.assertEqual({point["hudLevel"] for point in result["points"]}, {4, 5})

    def test_initial_completed_level_has_explicit_zero_anchor(self) -> None:
        row = self._row(2, 1, 1, 2)
        row.update({
            "video_time_a": "1", "frame_a": "30", "xp_quality_a": ".98",
            "xp_bar_progress_a_percent": "45", "reset_inferred_level": "1",
        })
        result = build_xp_progress([row], initial_level=1)
        level_one = [point["progressPercent"] for point in result["points"] if point["hudLevel"] == 1]
        self.assertEqual((min(level_one), max(level_one)), (0.0, 100.0))

    def test_consistent_reset_inference_preserves_xp_when_hud_ocr_is_missing(self) -> None:
        row = self._row(1, 1, 1, 1, boundary=False)
        row.update({
            "hud_level_ocr_accepted": "0",
            "hud_level_source": "reset_inference_fallback",
            "reset_inferred_level": "1",
            "level_agrees_with_reset_inference": "1",
        })
        result = build_xp_progress([row])
        measured = [point for point in result["points"] if point.get("endpoint")]
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["hudLevel"], 1)

    def test_inconsistent_reset_inference_does_not_replace_missing_hud_ocr(self) -> None:
        row = self._row(1, 1, 1, 1, boundary=False)
        row.update({
            "hud_level_ocr_accepted": "0",
            "hud_level_source": "reset_inference_fallback",
            "reset_inferred_level": "2",
            "level_agrees_with_reset_inference": "0",
        })
        result = build_xp_progress([row])
        self.assertEqual(result["points"], [])

    def test_reliable_next_hud_level_recovers_a_missing_boundary_as_upper_bound(self) -> None:
        rows = [
            self._row(1, 1, 1, 2),
            self._row(2, 2, 2, 2, boundary=False),
            self._row(3, 3, 3, 3, boundary=False),
        ]
        result = build_xp_progress(rows, initial_level=1)
        self.assertEqual([event["toLevel"] for event in result["levelChanges"]], [2, 3])
        self.assertEqual(result["levelChanges"][1]["timingPrecision"], "upper_bound")
        self.assertTrue(result["levelChanges"][1]["needsReview"])

    def test_exact_local_transition_can_resume_after_an_explicit_gap(self) -> None:
        rows = [
            self._row(1, 1, 1, 2),
            self._row(2, 4, 4, 5),
            self._row(3, 5, 5, 6),
        ]
        result = build_xp_progress(rows, initial_level=1)
        self.assertEqual([event["toLevel"] for event in result["levelChanges"]], [2, 5, 6])
        self.assertEqual(result["levelChanges"][1]["missingPrecedingLevels"], [3, 4])
        self.assertTrue(result["levelChanges"][1]["gapBefore"])

    def test_xp_segments_merge_time_levels_bridges_and_unresolved_evidence(self) -> None:
        progress = build_xp_progress([
            self._row(1, 1, 1, 2),
            self._row(2, 4, 4, 5),
            self._row(3, 8, 8, 10),
        ], initial_level=1)
        merged = merge_xp_progress([
            {"progress": progress},
            {"progress": progress, "offsetMs": 10_000, "levelOffset": 5},
        ])
        self.assertEqual(merged["levelChanges"][-1]["toLevel"], 10)
        self.assertEqual(merged["levelChanges"][-1]["mediaTimeMs"], 12_000)
        self.assertTrue(any(row["mediaTimeMs"] == 13_000 for row in merged["unresolvedLevelChanges"]))

    def test_gameplay_interruptions_merge_overlapping_blocking_events(self) -> None:
        intervals = build_gameplay_interruption_intervals([
            {"eventId": "menu", "eventType": "level_up_transaction", "startMs": 1_000, "endMs": 3_000, "anchorMs": 3_000},
            {"eventId": "lucky", "eventType": "lucky_level_up", "startMs": 2_500, "endMs": 3_500, "anchorMs": 3_500},
            {"eventId": "gem", "eventType": "blue_gem_pickup", "startMs": 2_000, "endMs": 2_010, "anchorMs": 2_010},
        ])
        self.assertEqual(len(intervals), 1)
        self.assertEqual((intervals[0]["startMs"], intervals[0]["endMs"]), (1_000, 3_500))
        self.assertEqual(intervals[0]["eventTypes"], ["level_up_transaction", "lucky_level_up"])

    def test_xp_is_flat_and_measurements_are_suppressed_during_interruption(self) -> None:
        progress = {
            "points": [
                {"mediaTimeMs": 500, "frameNumber": 15, "hudLevel": 2, "progressPercent": 25.0, "quality": .99, "breakBefore": True, "sourceType": "Automated"},
                {"mediaTimeMs": 2_000, "frameNumber": 60, "hudLevel": 2, "progressPercent": 55.0, "quality": .99, "breakBefore": False, "sourceType": "Automated"},
                {"mediaTimeMs": 4_000, "frameNumber": 120, "hudLevel": 2, "progressPercent": 70.0, "quality": .99, "breakBefore": False, "sourceType": "Automated"},
            ],
            "lessReliableSegments": [{"hudLevel": 2, "startMs": 500, "endMs": 4_000, "points": []}],
            "levelChanges": [],
            "unresolvedLevelChanges": [],
        }
        result = apply_gameplay_interruptions(progress, [
            {"eventId": "chest", "eventType": "loot_box_tier_1", "startMs": 1_000, "endMs": 3_000, "anchorMs": 1_000},
        ])
        hold_points = [point for point in result["points"] if point.get("interruptionHold")]
        self.assertEqual(
            [(point["mediaTimeMs"], point["progressPercent"]) for point in hold_points],
            [(1_000, 25.0), (3_000, 25.0)],
        )
        self.assertFalse(any(point["mediaTimeMs"] == 2_000 for point in result["points"]))
        self.assertEqual(result["lessReliableSegments"], [])
        self.assertEqual(result["interruptionHolds"][0]["policy"], "last_observation_carried_forward_no_xp_gain")

    def test_confirmed_level_reset_is_preserved_inside_interruption(self) -> None:
        progress = build_xp_progress([self._row(2, 1, 1, 2)], initial_level=1)
        result = apply_gameplay_interruptions(progress, [
            {"eventId": "menu", "eventType": "level_up_transaction", "startMs": 1_500, "endMs": 3_000, "anchorMs": 3_000},
        ])
        anchors = [
            point for point in result["points"]
            if point.get("boundaryAnchor") and point["mediaTimeMs"] == 2_000
        ]
        self.assertEqual(
            [(point["hudLevel"], point["progressPercent"]) for point in anchors],
            [(1, 100.0), (2, 0.0)],
        )
        self.assertTrue(any(hold["hudLevel"] == 1 for hold in result["interruptionHolds"]))
        self.assertTrue(any(hold["hudLevel"] == 2 for hold in result["interruptionHolds"]))

    def test_completed_selection_sequence_corrects_queued_level_undercount(self) -> None:
        progress = build_xp_progress([
            self._row(2, 1, 1, 2),
            self._row(8, 2, 2, 3),
        ], initial_level=1)
        events = [
            {"eventId": f"menu-{index}", "eventType": "level_up_transaction", "action": "select_reward", "startMs": end - 500, "endMs": end, "publicationStatus": "auto_accepted"}
            for index, end in enumerate((2_500, 4_000, 5_500), start=1)
        ]
        result = reconcile_game_levels_from_completed_selections(progress, events)
        self.assertTrue(result["gameLevelReconciliationApplied"])
        self.assertEqual([change["toLevel"] for change in result["levelChanges"]], [2, 3, 4])
        self.assertEqual(
            [(point["hudLevel"], point["progressPercent"]) for point in result["points"] if point.get("boundaryAnchor") and point["mediaTimeMs"] == 5_500],
            [(3, 100.0), (4, 0.0)],
        )

    def test_completed_selection_correction_ignores_reroll_and_requires_more_coverage(self) -> None:
        progress = build_xp_progress([
            self._row(2, 1, 1, 2),
            self._row(4, 2, 2, 3),
        ], initial_level=1)
        events = [
            {"eventId": "selection", "eventType": "level_up_transaction", "action": "select_reward", "startMs": 1_000, "endMs": 2_500},
            {"eventId": "reroll", "eventType": "level_up_transaction", "action": "reroll", "startMs": 3_000, "endMs": 3_500},
        ]
        result = reconcile_game_levels_from_completed_selections(progress, events)
        self.assertFalse(result["gameLevelReconciliationApplied"])
        self.assertEqual([change["toLevel"] for change in result["levelChanges"]], [2, 3])


class CounterRepairTests(unittest.TestCase):
    def test_repeated_counter_confirmation_rejects_isolated_spike(self) -> None:
        points = [
            {"mediaTimeMs": 0, "value": 100}, {"mediaTimeMs": 1000, "value": 100},
            {"mediaTimeMs": 2000, "value": 9100},
            {"mediaTimeMs": 3000, "value": 105}, {"mediaTimeMs": 4000, "value": 105},
        ]
        self.assertEqual(
            [row["value"] for row in corroborate_repeated_counter_points(points)],
            [100, 100, 105, 105],
        )

    def test_restores_truncated_leading_digits_from_raw_ocr(self) -> None:
        self.assertEqual(restore_truncated_counter_from_raw(7, "2,497"), (2497, "full_raw_ocr_restored"))
        self.assertEqual(restore_truncated_counter_from_raw(2497, "2497"), (2497, None))

    def test_repairs_persistent_thousands_place_shift_without_video_constants(self) -> None:
        repaired, reasons = repair_persistent_leading_place_shift([1328, 4333, 4345, 4350])
        self.assertEqual(repaired, [1328, 1333, 1345, 1350])
        self.assertEqual(reasons[1:], ["persistent_leading_place_ocr_shift_corrected"] * 3)

    def test_rejects_isolated_counter_spikes_without_flattening_next_value(self) -> None:
        repaired, reasons = repair_cumulative_counter_values([3000, 3030, 73029, 3035, 3040])
        self.assertEqual(repaired, [3000, 3030, 3030, 3035, 3040])
        self.assertEqual(reasons[2], "isolated_cumulative_counter_ocr_discontinuity_rejected")


class DashboardProjectionTests(unittest.TestCase):
    def test_inventory_item_level_is_not_published_as_game_level(self) -> None:
        rows = [{
            "event_id": "i1", "event_type": "upgrade", "video_second": "2",
            "frame_number": "60", "character_level": "20", "item_type": "weapon",
            "slot": "1", "item_after": "Axe", "item_before": "Axe",
            "level_before": "2", "level_after": "3", "confidence": "high",
            "needs_review": "0",
        }]
        event = project_inventory_events(rows)[0]
        self.assertEqual(event["observedCharacterLevel"], 20)
        self.assertEqual(event["itemLevelAfter"], 3)
        self.assertEqual(event["resultingLevel"], 20)

    def test_gem_projection_preserves_unresolved_quantity(self) -> None:
        rows = [{
            "event_key": "e1", "video_time_b": "1.5", "frame_b": "45",
            "collected_blue_gems": "2", "collected_green_gems": "0",
            "collected_red_gems": "0", "unresolved_collected_gems": "1",
            "confidence": ".8", "needs_review": "1", "review_reason": "color",
        }]
        event = project_gem_events(rows, 500, "segment")[0]
        self.assertEqual(event["videoTimeMs"], 2000)
        self.assertEqual(event["quantities"]["unresolved"], 1)

    def test_unresolved_gem_keeps_status_but_exposes_qualified_closest_color(self) -> None:
        rows = [{
            "event_key": "e1", "video_time_b": "1.5", "frame_b": "45",
            "collected_blue_gems": "0", "collected_green_gems": "0",
            "collected_red_gems": "0", "unresolved_collected_gems": "2",
            "confidence": ".25", "needs_review": "1", "review_reason": "color",
            "percentage_color_candidate": "blue",
        }]
        event = project_gem_events(rows, 0, "segment")[0]
        self.assertEqual(event["quantities"]["unresolved"], 2)
        self.assertEqual(event["unresolvedClosestColor"]["closestColorCandidate"], "blue")
        self.assertEqual(event["unresolvedClosestColor"]["closestColorQuantity"], 2)

    def test_unresolved_gem_does_not_assign_below_confidence_threshold(self) -> None:
        rows = [{
            "event_key": "e1", "video_time_b": "1.5", "frame_b": "45",
            "collected_blue_gems": "0", "collected_green_gems": "0",
            "collected_red_gems": "0", "unresolved_collected_gems": "1",
            "confidence": ".249", "needs_review": "1", "review_reason": "color",
            "percentage_color_candidate": "blue",
        }]
        event = project_gem_events(rows, 0, "segment")[0]
        self.assertEqual(event["quantities"]["unresolved"], 1)
        self.assertFalse(event["unresolvedClosestColor"]["closestColorAssigned"])
        self.assertEqual(event["unresolvedClosestColor"]["closestColorQuantity"], 0)

    def test_five_second_counter_windows_and_centered_mean(self) -> None:
        telemetry = [
            {"mediaStartMs": 0, "killCounter": 10},
            {"mediaStartMs": 5000, "killCounter": 15},
            {"mediaStartMs": 10000, "killCounter": 21},
        ]
        windows = build_five_second_counter_windows(telemetry, 10000)
        self.assertEqual([row["delta"] for row in windows], [5, 6])
        means = centered_window_mean(
            [{"time": 0, "value": 1}, {"time": 5000, "value": 3}],
            time_key="time", value_key="value",
        )
        self.assertEqual([row["value"] for row in means], [2, 2])

    def test_release_contains_exact_gems_without_client_side_join(self) -> None:
        release = build_dashboard_release(
            video_asset_id="video", session_id="session", duration_ms=5000,
            event_files=[],
            gem_events=[{
                "eventKey": "gem1", "videoTimeMs": 1000,
                "quantities": {"blue": 2, "green": 0, "red": 0, "unresolved": 1},
                "needsReview": False,
            }],
        )
        self.assertEqual(release["summary"]["byEventType"], {"blue_gem_pickup": 1, "unresolved_gem_pickup": 1})
        self.assertEqual(sum(event["quantity"] for event in release["events"]), 3)

    def test_release_can_project_inventory_selections(self) -> None:
        release = build_dashboard_release(
            video_asset_id="video", session_id="session", duration_ms=5000,
            event_files=[],
            inventory_events=[{
                "eventId": "selection-1", "mediaTimeMs": 1000,
                "itemName": "Axe", "action": "upgrade",
                "confidenceLabel": "high", "needsReview": False,
            }],
        )
        self.assertEqual(
            release["summary"]["byEventType"],
            {"level_up_selection": 1, "level_up_transaction": 1},
        )

    def test_real_releases_do_not_double_count_inventory_level_ups(self) -> None:
        root = Path(__file__).resolve().parents[1] / "runs"
        cases = (
            ("video4_full_pipeline_check40_20260917_173051", "inventory_reconciled", "inventory_events.csv", 2),
            ("video3_run02_cross40_20260918_010122", "inventory", "worker/inventory_events.csv", 2),
        )
        if any(not (root / run_name / "run_manifest.json").is_file() for run_name, _, _, _ in cases):
            self.skipTest("Real Video 4 and Video 3 run 2 artifacts are not installed")
        for run_name, csv_stage, csv_name, expected_count in cases:
            with self.subTest(run=run_name):
                run = root / run_name
                manifest = json.loads((run / "run_manifest.json").read_text())
                inputs = {}
                for stage, name in (("inventory", "canonical_inventory_events.jsonl"),
                                    (csv_stage, csv_name)):
                    receipt = manifest["stages"][stage]
                    path = run / receipt["directory"] / name
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), receipt["outputs"][name])
                    inputs[stage, name] = path
                with inputs[csv_stage, csv_name].open(newline="", encoding="utf-8") as handle:
                    rows = [row for row in csv.DictReader(handle)
                            if row.get("event_source") in {"level_up", "level_up_retrospective"}]
                release = build_dashboard_release(
                    video_asset_id="real_video", session_id="real_session", duration_ms=40_000,
                    event_files=[inputs["inventory", "canonical_inventory_events.jsonl"]],
                    inventory_events=project_inventory_events(rows),
                )
                level_ups = [event for event in release["events"]
                             if event["eventType"] in {"level_up_selection", "level_up_transaction"}]
                self.assertEqual(release["summary"]["byEventType"]["level_up_selection"], expected_count)
                self.assertEqual(release["summary"]["byEventType"]["level_up_transaction"], expected_count)
                self.assertTrue(all(event["evidenceGrade"] == "A" for event in level_ups))

    def test_duplicate_evolution_observations_become_one_canonical_event(self) -> None:
        events = [
            {"eventId": "chest-1", "eventType": "loot_box_tier_1", "family": "chest", "startMs": 747267, "endMs": 758517, "anchorMs": 758517, "quantity": 1, "itemName": None, "action": "opened", "evidenceGrade": "A", "publicationStatus": "auto_accepted"},
            {"eventId": "vss_chest_reward_757017_1", "eventType": "weapon_evolution", "family": "inventory", "startMs": 757017, "endMs": 757017, "anchorMs": 757017, "quantity": 1, "itemName": "Holy Wand", "action": "evolution", "evidenceGrade": "B", "publicationStatus": "auto_accepted"},
            {"eventId": "vss_inventory_outcome_1", "eventType": "weapon_evolution", "family": "inventory", "startMs": 758000, "endMs": 758000, "anchorMs": 758000, "quantity": 1, "itemName": "Holy Wand", "action": "evolution", "evidenceGrade": "A", "publicationStatus": "auto_accepted"},
            {"eventId": "vss_inventory_outcome_2", "eventType": "weapon_evolution", "family": "inventory", "startMs": 758500, "endMs": 758500, "anchorMs": 758500, "quantity": 1, "itemName": "Holy Wand", "action": "evolution", "evidenceGrade": "B", "publicationStatus": "auto_accepted"},
        ]
        result = canonicalize_weapon_evolutions(events)
        self.assertEqual(len(result), 2)
        evolution = next(event for event in result if event["eventType"] == "weapon_evolution")
        self.assertEqual(evolution["eventId"], "vss_chest_reward_757017_1")
        self.assertEqual(evolution["anchorMs"], 757017)
        self.assertEqual(evolution["evidenceGrade"], "A")
        self.assertEqual(evolution["evidenceCount"], 3)
        self.assertEqual(evolution["parentChestEventId"], "chest-1")

    def test_split_lucky_screen_fragments_become_one_transaction_backed_interval(self) -> None:
        events = [
            {"eventId": "tx-l56", "eventType": "level_up_transaction", "family": "progression", "startMs": 973667, "endMs": 973967, "anchorMs": 973967, "quantity": 1, "itemName": "Spinach", "action": "select_reward", "evidenceGrade": "A", "publicationStatus": "auto_accepted"},
            {"eventId": "lucky-primary", "eventType": "lucky_level_up", "family": "progression", "startMs": 973667, "endMs": 973967, "anchorMs": 973967, "quantity": 1, "itemName": None, "action": "four_options_visible", "evidenceGrade": "A", "publicationStatus": "auto_accepted"},
            {"eventId": "lucky-orphan-fragment", "eventType": "lucky_level_up", "family": "progression", "startMs": 974167, "endMs": 975167, "anchorMs": 975167, "quantity": 1, "itemName": None, "action": "four_options_visible", "evidenceGrade": "A", "publicationStatus": "auto_accepted"},
        ]
        result = canonicalize_lucky_level_ups(events)
        lucky = [event for event in result if event["eventType"] == "lucky_level_up"]
        self.assertEqual(len(lucky), 1)
        self.assertEqual((lucky[0]["startMs"], lucky[0]["endMs"], lucky[0]["anchorMs"]), (973667, 975167, 973967))
        self.assertEqual(lucky[0]["sourceTransactionEventId"], "tx-l56")
        self.assertEqual(lucky[0]["supportingEventIds"], ["lucky-orphan-fragment", "lucky-primary"])
        self.assertTrue(lucky[0]["canonicalized"])

    def test_adjacent_transaction_backed_lucky_level_ups_stay_separate(self) -> None:
        events = []
        for index, (start, end) in enumerate(((897600, 898800), (899200, 900800), (901100, 904100)), start=1):
            events.extend([
                {"eventId": f"tx-{index}", "eventType": "level_up_transaction", "family": "progression", "startMs": start, "endMs": end, "anchorMs": end, "quantity": 1, "itemName": "Item", "action": "select_reward", "evidenceGrade": "A", "publicationStatus": "auto_accepted"},
                {"eventId": f"lucky-{index}", "eventType": "lucky_level_up", "family": "progression", "startMs": start, "endMs": end, "anchorMs": end, "quantity": 1, "itemName": None, "action": "four_options_visible", "evidenceGrade": "A", "publicationStatus": "auto_accepted"},
            ])
        result = canonicalize_lucky_level_ups(events)
        self.assertEqual(len([event for event in result if event["eventType"] == "lucky_level_up"]), 3)

    def test_separate_releases_merge_with_offsets_and_unique_ids(self) -> None:
        part = build_dashboard_release(
            video_asset_id="part", session_id="part", duration_ms=5000,
            event_files=[], gem_events=[{
                "eventKey": "g1", "videoTimeMs": 1000,
                "quantities": {"blue": 1}, "needsReview": False,
            }],
        )
        merged = merge_dashboard_releases(
            video_asset_id="combined", session_id="combined", duration_ms=15000,
            segments=[
                {"release": part, "eventIdPrefix": "p1_"},
                {"release": part, "eventIdPrefix": "p2_", "offsetMs": 10000},
            ],
        )
        self.assertEqual(merged["summary"]["eventCount"], 2)
        self.assertEqual([row["anchorMs"] for row in merged["events"]], [1000, 11000])

    def test_reward_trajectory_keeps_kill_and_gem_units_separate(self) -> None:
        telemetry = [
            {"mediaStartMs": 0, "killCounter": 10, "gameClockMs": 0},
            {"mediaStartMs": 4000, "killCounter": 15, "gameClockMs": 4000},
        ]
        gems = [{"videoTimeMs": 2000, "quantities": {"blue": 2}, "needsReview": False}]
        window = build_reward_trajectory(telemetry, gems, 5000)[0]
        self.assertEqual(window["kills"], 5)
        self.assertEqual(window["gems"]["blue"], 2)

    def test_reward_trajectory_can_use_counter_boundary_snapshots(self) -> None:
        telemetry = [
            {"mediaStartMs": 0, "killCounter": 10, "gameClockMs": 0},
            {"mediaStartMs": 4_000, "killCounter": 15, "gameClockMs": 4_000},
            {"mediaStartMs": 5_000, "killCounter": 18, "gameClockMs": 5_000},
        ]
        window = build_reward_trajectory(
            telemetry, [], 5_000, counter_semantics="boundary_snapshot"
        )[0]
        self.assertEqual(window["killCounterStart"], 10)
        self.assertEqual(window["killCounterEnd"], 18)
        self.assertEqual(window["kills"], 8)

    def test_gold_bars_are_counter_deltas_and_mean_is_of_counter(self) -> None:
        points = [
            {"mediaTimeMs": 1_000, "value": 100},
            {"mediaTimeMs": 6_000, "value": 110},
            {"mediaTimeMs": 11_000, "value": 130},
            {"mediaTimeMs": 31_000, "value": 140},
        ]
        result = build_gold_counter_trajectory(points)
        self.assertEqual([row["goldDelta"] for row in result["fiveSecondDeltaBins"]], [10, 20])
        self.assertEqual(result["centered30SecondCounterMean"][0]["value"], 340 / 3)
        self.assertTrue(result["centered30SecondCounterMean"][-1]["breakBefore"])


class HealthCalibrationTests(unittest.TestCase):
    def test_uses_repeated_upper_cluster_not_lone_maximum(self) -> None:
        widths = [74.0] * 5 + [88.0] * 12 + [93.0, 94.0, 94.0, 95.0] * 20 + [130.0]
        reference, details = estimate_full_health_width(widths)
        self.assertEqual(reference, 94.0)
        self.assertGreater(details["cluster_support"], 50)

    def test_health_percent_is_bounded(self) -> None:
        self.assertAlmostEqual(health_percent(47, 94), 50.0)
        self.assertEqual(health_percent(100, 94), 100.0)

    def _health(self, frame: int, value: float | None) -> SignalObservation:
        return SignalObservation(
            observation_id=str(frame), video_asset_id="video", session_id="session",
            detector_name="health", detector_version="1", observable_code="player_health_percent",
            time_lower_ms=frame * 1000, time_upper_ms=frame * 1000,
            temporal_precision=TemporalPrecision.FRAME,
            visibility=Visibility.VISIBLE if value is not None else Visibility.UNKNOWN,
            evidence_grade=EvidenceGrade.B if value is not None else EvidenceGrade.UNRESOLVED,
            observed=value is not None, source_artifact="health.jsonl", source_record_key=str(frame),
            numeric_value=value, unit="percent", frame_number=frame,
        )

    def test_persistent_drop_and_recovery_are_events(self) -> None:
        rows = [self._health(0, 100), self._health(1, 90), self._health(2, 90), self._health(3, 100), self._health(4, 100)]
        events = extract_health_events(rows, processing_run_id="run")
        self.assertEqual([event.event_type for event in events], ["hp_loss", "hp_recovery"])

    def test_one_frame_change_and_missing_gap_are_not_events(self) -> None:
        rows = [self._health(0, 100), self._health(1, 80), self._health(2, 100), self._health(3, None), self._health(4, 80)]
        self.assertEqual(extract_health_events(rows, processing_run_id="run"), [])

    def test_short_width_requires_adjacent_confirmation(self) -> None:
        samples = [
            {"width": 94.0, "confidence": 1.0},
            {"width": 16.0, "confidence": 0.8},
            {"width": 94.0, "confidence": 1.0},
            {"width": 28.0, "confidence": 0.8},
            {"width": 30.0, "confidence": 0.8},
        ]
        self.assertEqual(reject_isolated_short_widths(samples), 1)
        self.assertIsNone(samples[1]["width"])
        self.assertEqual(samples[3]["width"], 28.0)


class HealthAttributionTests(unittest.TestCase):
    def test_floor_chicken_is_preferred_when_temporally_close(self) -> None:
        event = {"anchor_time_ms": 10_000, "quantity": 20}
        inventory = [{"event_id": "chicken", "event_type": "floor_chicken", "anchor_time_ms": 8_000}]
        result = attribute_recovery_event(event, inventory)
        self.assertEqual(result["cause_code"], "floor_chicken_temporal_match")
        self.assertEqual(result["latency_ms"], 2_000)

    def test_pummarola_is_compatible_not_causal(self) -> None:
        event = {"anchor_time_ms": 20_000, "quantity": 4}
        inventory = [{"event_id": "pum", "event_type": "new_passive_item", "item_name": "Pummarola", "anchor_time_ms": 10_000}]
        result = attribute_recovery_event(event, inventory)
        self.assertEqual(result["cause_code"], "pummarola_compatible_regeneration")
        self.assertEqual(result["causal_status"], "compatible_not_confirmed")


class WorldPickupTests(unittest.TestCase):
    def _d(self, time_ms: int, x: float, y: float, score: float = 0.97) -> PickupDetection:
        return PickupDetection("floor_chicken", time_ms // 10, time_ms, x, y, score, 0.5)

    def test_non_maximum_suppression_keeps_best_local_match(self) -> None:
        kept = non_maximum_suppression([self._d(0, 10, 10, .95), self._d(0, 12, 11, .99), self._d(0, 80, 80, .96)])
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].score, .99)

    def test_persistent_track_disappearing_near_player_is_review_candidate(self) -> None:
        tracks = associate_tracks([[self._d(0, 80, 50)], [self._d(500, 95, 50)]])
        events = resolve_pickup_events(tracks, processing_run_id="run", video_asset_id="video",
                                       session_id="session", player_center=(100, 50), final_sample_ms=1500,
                                       collection_radius=20, sample_period_ms=500)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].publication_status, PublicationStatus.NEEDS_REVIEW)
        self.assertTrue(events[0].attributes["candidate_only"])

    def test_single_frame_and_far_disappearance_are_not_pickups(self) -> None:
        tracks = associate_tracks([[self._d(0, 10, 10)], [self._d(500, 11, 10)]])
        self.assertEqual(resolve_pickup_events(tracks, processing_run_id="run", video_asset_id="video",
                         session_id="session", player_center=(100, 100), final_sample_ms=1000,
                         collection_radius=20), [])
        single = associate_tracks([[self._d(0, 100, 100)]])
        self.assertEqual(resolve_pickup_events(single, processing_run_id="run", video_asset_id="video",
                         session_id="session", player_center=(100, 100), final_sample_ms=500), [])


class WorldPickupEffectTests(unittest.TestCase):
    def test_large_recovery_is_candidate_not_confirmed_chicken(self) -> None:
        source = event_at(1000, family="recovery", event_type="hp_recovery").to_dict()
        source["quantity"] = 17
        events = extract_abrupt_healing_candidates([source], processing_run_id="run")
        self.assertEqual(events[0].event_type, "abrupt_healing_effect")
        self.assertEqual(events[0].publication_status, PublicationStatus.NEEDS_REVIEW)
        self.assertFalse(events[0].attributes["floor_chicken_confirmed"])

    def test_coin_delta_requires_consecutive_observed_values(self) -> None:
        def row(time: int, value: int, observed: bool = True) -> dict:
            return {"observable_code": "coin_counter", "observed": observed,
                    "time_lower_ms": time, "numeric_value": value,
                    "video_asset_id": "video", "session_id": "session",
                    "frame_number": time // 10, "source_artifact": "coin.jsonl",
                    "source_record_key": str(time), "attributes": {"confidence": .99}}
        events = extract_currency_gains([row(0, 10), row(1000, 20), row(1500, 20), row(5000, 30)], processing_run_id="run")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].quantity, 10)
        self.assertIsNone(events[0].attributes["physical_pickup_identity"])


class FreezeEffectTests(unittest.TestCase):
    def test_persistent_relative_motion_drop_becomes_review_candidate(self) -> None:
        rows = []
        for index in range(12):
            rows.append({"frame_number": index, "media_time_ms": index * 500, "observable": True,
                         "residual_motion_fraction": .10 if index < 6 else .02,
                         "pale_cyan_fraction": .02 if index < 6 else .06,
                         "alignment_response": .9})
        mark_freeze_like_rows(rows, baseline_seconds=3, sample_fps=2)
        events = resolve_freeze_intervals(rows, processing_run_id="run", video_asset_id="video",
                                          session_id="session", sample_period_ms=500,
                                          minimum_duration_ms=1000)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].publication_status, PublicationStatus.NEEDS_REVIEW)
        self.assertFalse(events[0].attributes["orologion_confirmed"])

    def test_unobservable_row_breaks_freeze_interval(self) -> None:
        rows = [{"frame_number": 1, "media_time_ms": 0, "freeze_like": True, "motion_ratio": .1, "tint_lift": .1},
                {"frame_number": 2, "media_time_ms": 500, "freeze_like": False},
                {"frame_number": 3, "media_time_ms": 1000, "freeze_like": True, "motion_ratio": .1, "tint_lift": .1}]
        self.assertEqual(resolve_freeze_intervals(rows, processing_run_id="run", video_asset_id="video",
                         session_id="session", sample_period_ms=500, minimum_duration_ms=1000), [])


class VacuumFlowTests(unittest.TestCase):
    def test_radial_flow_distinguishes_inward_motion(self) -> None:
        previous = [(100, 190), (540, 190), (320, 40), (320, 340)]
        current = [(120, 190), (520, 190), (320, 60), (320, 320)]
        result = radial_flow_features(previous, current, dt_seconds=.2)
        self.assertEqual(result["outer_matches"], 4)
        self.assertEqual(result["inward_fraction"], 1)
        self.assertGreater(result["median_inward_speed"], 35)

    def test_three_persistent_samples_form_review_candidate(self) -> None:
        rows = []
        for index in range(12):
            rows.append({"frame_number": index, "media_time_ms": index * 200, "observable": True,
                         "outer_matches": 5 if index < 6 else 14,
                         "inward_fraction": .2 if index < 6 else .8,
                         "median_inward_speed": 5 if index < 6 else 60,
                         "previous_components": 30 if index >= 6 else 10,
                         "current_components": max(10, 30 - (index - 5) * 5) if index >= 6 else 10})
        mark_vacuum_like_rows(rows, sample_fps=5, baseline_seconds=1)
        events = resolve_vacuum_intervals(rows, processing_run_id="run", video_asset_id="video",
                                          session_id="session", sample_period_ms=200)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].publication_status, PublicationStatus.NEEDS_REVIEW)
        self.assertFalse(events[0].attributes["vacuum_confirmed"])


class GoldFeverTests(unittest.TestCase):
    def test_persistent_bottom_gauge_becomes_interval(self) -> None:
        rows = [{"frame_number": i, "media_time_ms": i * 250,
                 "peak_gold_row_coverage": .8, "supported_gold_row_coverage": .7,
                 "bottom_gold_fraction": .12, "peak_row_y_360": 354,
                 "gold_fever_label_gold_fraction": .20}
                for i in range(5)]
        mark_gold_fever_rows(rows)
        events = resolve_gold_fever_intervals(rows, processing_run_id="run", video_asset_id="video",
            session_id="session", sample_period_ms=250)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "gold_fever_effect_candidate")
        self.assertFalse(events[0].attributes["gilded_clover_confirmed"])

    def test_label_keeps_interval_alive_after_meter_depletes(self) -> None:
        rows = [
            {"frame_number": 1, "media_time_ms": 0, "peak_gold_row_coverage": .8,
             "supported_gold_row_coverage": .7, "bottom_gold_fraction": .12,
             "peak_row_y_360": 354, "gold_fever_label_gold_fraction": .20},
            {"frame_number": 2, "media_time_ms": 250, "peak_gold_row_coverage": .08,
             "supported_gold_row_coverage": .05, "bottom_gold_fraction": .02,
             "peak_row_y_360": 354, "gold_fever_label_gold_fraction": .03},
            {"frame_number": 3, "media_time_ms": 500, "peak_gold_row_coverage": .05,
             "supported_gold_row_coverage": .03, "bottom_gold_fraction": .01,
             "peak_row_y_360": 354, "gold_fever_label_gold_fraction": .03},
            {"frame_number": 4, "media_time_ms": 750, "peak_gold_row_coverage": .05,
             "supported_gold_row_coverage": .03, "bottom_gold_fraction": .01,
             "peak_row_y_360": 354, "gold_fever_label_gold_fraction": .03},
            {"frame_number": 5, "media_time_ms": 1000, "peak_gold_row_coverage": .01,
             "supported_gold_row_coverage": .01, "bottom_gold_fraction": .001,
             "peak_row_y_360": 330, "gold_fever_label_gold_fraction": 0.0},
        ]
        mark_gold_fever_rows(rows)
        events = resolve_gold_fever_intervals(rows, processing_run_id="run", video_asset_id="video",
            session_id="session", sample_period_ms=250)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].time_upper_ms, 1000)

    def test_meter_only_false_positive_is_rejected_without_label(self) -> None:
        rows = [{"frame_number": i, "media_time_ms": i * 250,
                 "peak_gold_row_coverage": .8, "supported_gold_row_coverage": .7,
                 "bottom_gold_fraction": .12, "peak_row_y_360": 354,
                 "gold_fever_label_gold_fraction": 0.0}
                for i in range(5)]
        mark_gold_fever_rows(rows)
        self.assertEqual(resolve_gold_fever_intervals(rows, processing_run_id="run",
            video_asset_id="video", session_id="session", sample_period_ms=250), [])

    def test_short_gold_flash_is_rejected(self) -> None:
        rows = [{"frame_number": 1, "media_time_ms": 0, "gold_fever_like": True,
                 "peak_gold_row_coverage": .8}]
        self.assertEqual(resolve_gold_fever_intervals(rows, processing_run_id="run",
            video_asset_id="video", session_id="session", sample_period_ms=250), [])


class ChestLifecycleTests(unittest.TestCase):
    def test_final_reward_detail_icon_is_found_by_geometry(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (500, 750), (560, 810), (0, 210, 255), 8)
        cv2.rectangle(frame, (515, 765), (545, 795), (255, 120, 10), -1)
        crop = _reward_detail_icon_crop(frame)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (28, 28, 3))

    def test_three_reward_reveal_is_tier_two(self) -> None:
        rows = [{"frame_number": i, "media_time_ms": i * 250,
                 "center_beam_fraction": .7, "purple_panel_fraction": .8,
                 "reward_orb_count": 3 if i == 5 else 0} for i in range(8)]
        mark_chest_rows(rows)
        events = resolve_chest_intervals(rows, processing_run_id="run", video_asset_id="video",
            session_id="session", sample_period_ms=250)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "loot_box_tier_2")
        self.assertEqual(events[0].quantity, 3)

    def test_short_panel_flash_is_not_a_chest(self) -> None:
        rows = [{"frame_number": 1, "media_time_ms": 0, "chest_overlay_like": True,
                 "reward_orb_count": 1, "center_beam_fraction": .7}]
        self.assertEqual(resolve_chest_intervals(rows, processing_run_id="run", video_asset_id="video",
            session_id="session", sample_period_ms=250), [])

    def test_reward_assignment_is_one_to_one(self) -> None:
        rows=[{"Peachone":.9,"Lightning Ring":.4},{"Peachone":.8,"Lightning Ring":.7}]
        result=resolve_assignments(rows,["Peachone","Lightning Ring"])
        self.assertEqual([x[0] for x in result],["Peachone","Lightning Ring"])

    def test_reward_identity_uses_cross_frame_consensus(self) -> None:
        frames=[
            (100,[("Peachone",.76,.01),("Crown",.70,.20),("Magic Wand",.72,.20)]),
            (200,[("Pentagram",.62,.08),("Crown",.68,.18),("Magic Wand",.70,.18)]),
            (300,[("Pentagram",.61,.07),("Crown",.69,.19),("Magic Wand",.71,.19)]),
        ]
        _,result,votes=resolve_consensus_assignments(frames,3)
        self.assertEqual([x[0] for x in result],["Pentagram","Crown","Magic Wand"])
        self.assertEqual(votes,[2,3,3])

    def test_stable_multi_reward_identity_can_clear_orb_occlusion(self) -> None:
        accepted,stable,vote_fraction=orb_identity_acceptance(
            score=.36,margin=-.04,consensus_votes=5,
            complete_frame_count=6,reward_count=5)
        self.assertTrue(accepted)
        self.assertTrue(stable)
        self.assertAlmostEqual(vote_fraction,5/6)

    def test_weak_or_sparse_multi_reward_identity_remains_unresolved(self) -> None:
        self.assertFalse(orb_identity_acceptance(
            score=.34,margin=.10,consensus_votes=6,
            complete_frame_count=6,reward_count=5)[0])
        self.assertFalse(orb_identity_acceptance(
            score=.36,margin=-.04,consensus_votes=2,
            complete_frame_count=6,reward_count=5)[0])

    def test_single_reward_does_not_use_lower_multi_reward_threshold(self) -> None:
        accepted,stable,_=orb_identity_acceptance(
            score=.36,margin=.10,consensus_votes=6,
            complete_frame_count=6,reward_count=1)
        self.assertFalse(accepted)
        self.assertFalse(stable)


class MenuActionTests(unittest.TestCase):
    def test_counter_text_keeps_unavailable_skip_missing(self) -> None:
        self.assertEqual(parse_action_counter_text(["Reroll", "+1", "Skip", "-", "Banish", "+4"]),
                         {"reroll":1,"skip":None,"banish":4})

    def test_counter_text_repairs_plus_misread_as_leading_four(self) -> None:
        self.assertEqual(parse_action_counter_text(["Reroll", "4108", "Skip", "+108", "Banish", "4107"]),
                         {"reroll":108,"skip":108,"banish":107})

    def test_counter_drop_emits_bounded_action(self) -> None:
        rows=[{"frame_number":1,"media_time_ms":1000,"reroll_remaining":1,"skip_remaining":None,"banish_remaining":4},
              {"frame_number":2,"media_time_ms":2000,"reroll_remaining":0,"skip_remaining":None,"banish_remaining":4}]
        events=resolve_action_counter_drops(rows,processing_run_id="run",video_asset_id="video",session_id="session")
        self.assertEqual(len(events),1); self.assertEqual(events[0].action,"reroll")
        self.assertEqual(events[0].temporal_precision,TemporalPrecision.BOUNDED)

    def test_isolated_counter_drop_that_rebounds_is_rejected(self) -> None:
        rows=[{"frame_number":1,"media_time_ms":1000,"reroll_remaining":108,"skip_remaining":108,"banish_remaining":108},
              {"frame_number":2,"media_time_ms":2000,"reroll_remaining":108,"skip_remaining":100,"banish_remaining":108},
              {"frame_number":3,"media_time_ms":3000,"reroll_remaining":108,"skip_remaining":108,"banish_remaining":108}]
        events=resolve_action_counter_drops(rows,processing_run_id="run",video_asset_id="video",session_id="session")
        self.assertEqual(events,[])

    def test_repeated_counter_drop_is_accepted_once(self) -> None:
        rows=[{"frame_number":1,"media_time_ms":1000,"reroll_remaining":108,"skip_remaining":108,"banish_remaining":108},
              {"frame_number":2,"media_time_ms":2000,"reroll_remaining":106,"skip_remaining":108,"banish_remaining":108},
              {"frame_number":3,"media_time_ms":3000,"reroll_remaining":106,"skip_remaining":108,"banish_remaining":108}]
        events=resolve_action_counter_drops(rows,processing_run_id="run",video_asset_id="video",session_id="session")
        self.assertEqual(len(events),2)
        self.assertTrue(all(event.action=="reroll" for event in events))

    def test_persistent_panel_without_reward_reveal_is_not_published(self) -> None:
        rows = [{"frame_number": i, "media_time_ms": i * 250, "chest_overlay_like": True,
                 "reward_orb_count": 0, "center_beam_fraction": .7} for i in range(8)]
        self.assertEqual(resolve_chest_intervals(rows, processing_run_id="run", video_asset_id="video",
            session_id="session", sample_period_ms=250), [])


class StatusEventTests(unittest.TestCase):
    def test_persistent_game_over_and_results_emit_death_transition_and_achievement(self) -> None:
        rows=[]
        for i,state in enumerate(["gameplay_or_overlay","game_over","game_over","results","results"]):
            rows.append({"frame_number":i,"media_time_ms":i*500,"state":state,"features":{}})
        events=resolve_status_intervals(rows,processing_run_id="run",video_asset_id="video",
            session_id="session",sample_period_ms=500)
        self.assertEqual([e.event_type for e in events],["death","match_transition","achievement"])


def event_at(time_ms: int, *, family: str, event_type: str) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=f"event_{event_type}_{time_ms}",
        video_asset_id="video",
        session_id="session",
        event_family=family,
        event_type=event_type,
        time_lower_ms=time_ms,
        time_upper_ms=time_ms,
        anchor_time_ms=time_ms,
        temporal_precision=TemporalPrecision.FRAME,
        evidence_grade=EvidenceGrade.A,
        publication_status=PublicationStatus.AUTO_ACCEPTED,
        inference_method="fixture",
        processing_run_id="test",
    )


class ModelTests(unittest.TestCase):
    def test_event_rejects_inverted_time_bounds(self) -> None:
        with self.assertRaises(ValueError):
            CanonicalEvent(
                event_id="bad",
                video_asset_id="video",
                session_id="session",
                event_family="test",
                event_type="test",
                time_lower_ms=10,
                time_upper_ms=9,
                anchor_time_ms=None,
                temporal_precision=TemporalPrecision.BOUNDED,
                evidence_grade=EvidenceGrade.C,
                publication_status=PublicationStatus.NEEDS_REVIEW,
                inference_method="fixture",
                processing_run_id="test",
            )

    def test_event_rejects_anchor_outside_bounds(self) -> None:
        with self.assertRaises(ValueError):
            event = event_at(5_000, family="test", event_type="test")
            object.__setattr__(event, "anchor_time_ms", 6_000)
            event.__post_init__()

    def test_observation_rejects_negative_frame(self) -> None:
        with self.assertRaises(ValueError):
            FrameObservation(
                observation_id="bad",
                video_asset_id="video",
                detector_name="fixture",
                detector_version="1",
                observation_type="screen_state",
                frame_number=-1,
                media_time_ms=0,
                visibility=Visibility.UNKNOWN,
                evidence_grade=EvidenceGrade.C,
                value=None,
                source_artifact="fixture.py",
            )


class ScreenStateDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        boundary = patch(
            "vss_framework.detectors.screen_state.gameplay_pause_evidence",
            return_value={"blocked": False, "phase": "gameplay_unblocked"},
        )
        self.pause_evidence = boundary.start()
        self.addCleanup(boundary.stop)

    def test_reused_level_up_geometry_emits_menu_observation(self) -> None:
        self.pause_evidence.return_value = {"blocked": True, "phase": "level_up_menu"}
        detector = ScreenStateDetector(".", "legacy/inventory_event_recorder.py")
        detector._inventory_module = SimpleNamespace(
            level_up_option_rectangles=lambda _frame: [(10, 20, 30, 40)]
        )
        detector._weapon_module = SimpleNamespace(locate_weapon_hud=lambda _frame: None)
        packet = FramePacket("video", 300, 10_000, object())
        observation = detector.observe(packet)[0]
        self.assertEqual(observation.value, "level_up_menu")
        self.assertEqual(observation.attributes["option_count"], 1)
        self.assertEqual(observation.evidence_grade, EvidenceGrade.A)

    def test_hud_visibility_does_not_claim_gameplay(self) -> None:
        layout = SimpleNamespace(x=10, y=20, step_x=46, slots=6, slot_size=44)
        detector = ScreenStateDetector(".", "legacy/inventory_event_recorder.py")
        detector._inventory_module = SimpleNamespace(
            level_up_option_rectangles=lambda _frame: []
        )
        detector._weapon_module = SimpleNamespace(
            locate_weapon_hud=lambda _frame: layout
        )
        packet = FramePacket("video", 302, 10_067, object())
        observation = detector.observe(packet)[0]
        self.assertEqual(observation.value, "hud_visible_unclassified")
        self.assertEqual(observation.visibility, Visibility.VISIBLE)
        self.assertEqual(observation.evidence_grade, EvidenceGrade.B)

    def test_unreadable_hud_remains_unknown(self) -> None:
        def fail(_frame: object) -> object:
            raise RuntimeError("not found")

        detector = ScreenStateDetector(".", "legacy/inventory_event_recorder.py")
        detector._inventory_module = SimpleNamespace(
            level_up_option_rectangles=lambda _frame: []
        )
        detector._weapon_module = SimpleNamespace(locate_weapon_hud=fail)
        packet = FramePacket("video", 301, 10_033, object())
        observation = detector.observe(packet)[0]
        self.assertEqual(observation.value, "unknown")
        self.assertEqual(observation.visibility, Visibility.UNKNOWN)


class XPBarDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        boundary = patch(
            "vss_framework.detectors.xp_bar.gameplay_pause_evidence",
            return_value={"blocked": False, "phase": "gameplay_unblocked"},
        )
        boundary.start()
        self.addCleanup(boundary.stop)

    def detector(self, *, progress: float, quality: float, hud: float) -> XPBarDetector:
        detector = XPBarDetector(".", "legacy/gem_detector.py")
        detector._config = SimpleNamespace(
            xp_min_quality=0.78,
            hud_score_threshold=0.90,
            level_up_strong_overlay_threshold=0.30,
        )
        detector._module = SimpleNamespace(
            measure_xp_bar_progress=lambda _frame, _cfg: (progress, quality),
            gameplay_hud_score=lambda _frame: hud,
            level_up_overlay_score=lambda _frame: 0.0,
        )
        return detector

    def test_reliable_xp_measurement_is_accepted(self) -> None:
        packet = FramePacket("video", 30, 1_000, object())
        observation = self.detector(
            progress=0.42, quality=0.98, hud=0.95
        ).observe(packet)[0]
        self.assertAlmostEqual(observation.value, 0.42)
        self.assertTrue(observation.attributes["accepted"])
        self.assertEqual(observation.evidence_grade, EvidenceGrade.A)

    def test_unconfirmed_hud_keeps_raw_xp_but_not_accepted_value(self) -> None:
        packet = FramePacket("video", 31, 1_033, object())
        observation = self.detector(
            progress=0.42, quality=0.98, hud=0.40
        ).observe(packet)[0]
        self.assertIsNone(observation.value)
        self.assertAlmostEqual(observation.attributes["raw_progress_fraction"], 0.42)
        self.assertEqual(observation.visibility, Visibility.OCCLUDED)

    def test_level_up_overlay_rejects_otherwise_clean_xp_measurement(self) -> None:
        detector = self.detector(progress=1.0, quality=0.99, hud=0.98)
        detector._module.level_up_overlay_score = lambda _frame: 0.76
        packet = FramePacket("video", 32, 1_067, object())
        observation = detector.observe(packet)[0]
        self.assertIsNone(observation.value)
        self.assertFalse(observation.attributes["accepted"])
        self.assertEqual(
            observation.attributes["reliability_reason"], "level_up_overlay_present"
        )


class LegacyGemXPDetectorTests(unittest.TestCase):
    def test_command_runs_existing_detector_without_human_inputs(self) -> None:
        command = build_legacy_gem_command(
            python_executable=Path("env/python"),
            script_path=Path("detector.py"),
            video_path=Path("video.mp4"),
            template_dir=Path("templates"),
            output_dir=Path("output"),
            template_profile="legacy",
            initial_level=1,
            inventory_events=Path("inventory.csv"),
            max_seconds=20.0,
            save_previews=False,
        )
        self.assertEqual(command[1], "detector.py")
        self.assertIn("--inventory-events", command)
        self.assertIn("--max-seconds", command)
        self.assertIn("--no-previews", command)
        self.assertNotIn("human", " ".join(command).lower())


class LegacyHUDDetectorTests(unittest.TestCase):
    def command(self, *, full: bool, max_seconds: int | None) -> list[str]:
        return build_legacy_hud_command(
            python_executable=Path("env/python"),
            script_path=Path("extract_hud_worker.py"),
            video_path=Path("video.mp4"),
            output_dir=Path("output"),
            full=full,
            max_seconds=max_seconds,
            sample_offsets=[0.3, 0.5, 0.7],
            min_ocr_confidence=0.0,
            min_timer_observed_rate=0.5,
            min_kill_observed_rate=0.4,
            initial_kill_state=0,
            evidence_every=60,
            max_evidence_frames=250,
        )

    def test_partial_command_has_prefix_and_no_human_input(self) -> None:
        command = self.command(full=False, max_seconds=20)
        self.assertIn("--max-seconds", command)
        self.assertNotIn("--full", command)
        self.assertEqual(command[command.index("--max-seconds") + 1], "20")
        self.assertNotIn("human", " ".join(command).lower())

    def test_full_command_uses_full_switch(self) -> None:
        command = self.command(full=True, max_seconds=None)
        self.assertIn("--full", command)
        self.assertNotIn("--max-seconds", command)


class LegacyInventoryDetectorTests(unittest.TestCase):
    def test_command_uses_automated_sources_without_human_or_review_audit(self) -> None:
        command = build_legacy_inventory_command(
            python_executable=Path("env/python"),
            script_path=Path("inventory_event_recorder.py"),
            video_path=Path("video.mp4"),
            xp_events_path=Path("automated_xp_events.csv"),
            output_dir=Path("output"),
            weapon_icon_dir=Path("weapon_icons"),
            passive_icon_dir=Path("passive_icons"),
            weapon_manifest_path=Path("weapon_manifest.csv"),
            passive_manifest_path=Path("passive_manifest.csv"),
            weapon_timeline_path=Path("automated_weapon_timeline.csv"),
            video_key="video_4",
            sample_fps=10.0,
            end_second=60.0,
        )
        joined = " ".join(command).lower()
        self.assertIn("--end 60", joined)
        self.assertNotIn("human", joined)
        self.assertNotIn("treasure-chest-audit", joined)


class LevelUpTransactionTests(unittest.TestCase):
    def test_split_menu_uses_its_final_completion_time_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            menus = root / "menus.csv"
            inventory = root / "inventory.csv"
            menus.write_text(
                "menu_start_second,menu_end_second,frame_number,video_second,selected_index,cursor_score,item_name,suggested_item,item_type,confidence,match_score,score_margin,needs_review,option_count\n"
                "10.0,10.3,309,10.3,0,100,Spinach,Spinach,passive_item,high,0.8,0.3,False,4\n"
                "10.5,11.5,345,11.5,0,100,Spinach,Spinach,passive_item,high,0.8,0.3,False,4\n",
                encoding="utf-8",
            )
            inventory.write_text(
                "event_id,video_second,character_level,event_source,event_type,item_after\n"
                "event_1,10.3,56,level_up,upgrade,Spinach\n",
                encoding="utf-8",
            )
            events = resolve_level_up_transactions(
                menu_audit_path=menus, inventory_events_path=inventory,
                video_asset_id="video", session_id="session", processing_run_id="run",
                source_artifact="menus.csv",
            )
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].time_lower_ms, events[0].time_upper_ms, events[0].anchor_time_ms),
                         (10_000, 11_500, 11_500))
        self.assertEqual(events[0].attributes["collapsed_menu_record_count"], 2)

    def test_inventory_match_resolves_selection_and_missing_match_stays_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            menus = root / "menus.csv"
            inventory = root / "inventory.csv"
            menus.write_text(
                "menu_start_second,menu_end_second,frame_number,video_second,selected_index,cursor_score,item_name,suggested_item,item_type,confidence,match_score,score_margin,needs_review\n"
                "9.0,10.0,300,10.0,0,100,King Bible,King Bible,weapon,high,0.8,0.3,False\n"
                "19.0,20.0,600,20.0,1,80,Attractorb,Attractorb,passive_item,medium,0.6,0.1,False\n",
                encoding="utf-8",
            )
            inventory.write_text(
                "event_id,video_second,character_level,event_source,event_type,item_after\n"
                "event_1,10.0,4,level_up,new,King Bible\n",
                encoding="utf-8",
            )
            events = resolve_level_up_transactions(
                menu_audit_path=menus,
                inventory_events_path=inventory,
                video_asset_id="video",
                session_id="session",
                processing_run_id="run",
                source_artifact="menus.csv",
            )
        self.assertEqual(events[0].action, "select_reward")
        self.assertEqual(events[0].publication_status, PublicationStatus.AUTO_ACCEPTED)
        self.assertEqual(events[1].action, "unresolved_no_inventory_transition")
        self.assertEqual(events[1].publication_status, PublicationStatus.UNRESOLVED)
        self.assertEqual(
            events[1].attributes["possible_actions"],
            ["banish", "skip", "instant_reward"],
        )

    def test_unclear_pointer_is_not_promoted_to_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            menus = root / "menus.csv"
            inventory = root / "inventory.csv"
            menus.write_text(
                "menu_start_second,menu_end_second,frame_number,video_second,selected_index,cursor_score,item_name,suggested_item,item_type,confidence,match_score,score_margin,needs_review\n"
                "1,2,60,2,0,0,UNKNOWN,UNKNOWN,unknown,low,0,0,True\n",
                encoding="utf-8",
            )
            inventory.write_text(
                "event_id,video_second,character_level,event_source,event_type,item_after\n",
                encoding="utf-8",
            )
            events = resolve_level_up_transactions(
                menu_audit_path=menus,
                inventory_events_path=inventory,
                video_asset_id="video",
                session_id="session",
                processing_run_id="run",
                source_artifact="menus.csv",
            )
        self.assertEqual(events, [])

    def test_owned_item_at_normal_max_resolves_as_banish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            menus = root / "menus.csv"
            inventory = root / "inventory.csv"
            menus.write_text(
                "menu_start_second,menu_end_second,frame_number,video_second,selected_index,cursor_score,item_name,suggested_item,item_type,confidence,match_score,score_margin,needs_review,option_count\n"
                "99,100,3000,100,0,100,Attractorb,Attractorb,passive_item,high,0.8,0.3,False,1\n",
                encoding="utf-8",
            )
            inventory.write_text(
                "event_id,video_second,character_level,event_source,event_type,item_after,level_after,normal_max_level\n"
                "event_1,90,66,level_up,upgrade,Attractorb,5,5\n",
                encoding="utf-8",
            )
            event = resolve_level_up_transactions(
                menu_audit_path=menus,
                inventory_events_path=inventory,
                video_asset_id="video",
                session_id="session",
                processing_run_id="run",
                source_artifact="menus.csv",
            )[0]
        self.assertEqual(event.action, "banish")
        self.assertEqual(event.evidence_grade, EvidenceGrade.B)
        self.assertEqual(event.publication_status, PublicationStatus.AUTO_ACCEPTED)


class InstantRewardDetectorTests(unittest.TestCase):
    def test_green_and_orange_icons_are_separated(self) -> None:
        import numpy as np

        green = np.full((20, 20, 3), (40, 180, 40), dtype=np.uint8)
        orange = np.full((20, 20, 3), (20, 120, 240), dtype=np.uint8)
        self.assertEqual(classify_instant_reward_icon(green)[0], "Big Coin Bag")
        self.assertEqual(classify_instant_reward_icon(orange)[0], "Floor Chicken")


class DirectTelemetryTests(unittest.TestCase):
    def test_coin_counter_rejects_single_leading_digit_substitution(self) -> None:
        self.assertTrue(is_leading_digit_ocr_substitution(7963, 1963))
        self.assertFalse(is_leading_digit_ocr_substitution(2328, 1963))

    def test_coin_counter_rejects_implausible_single_sample_jump(self) -> None:
        self.assertTrue(is_plausible_coin_increase(3736, 2858))
        self.assertFalse(is_plausible_coin_increase(27715, 2715))

    def test_gold_fever_accepts_only_high_confidence_bounded_monotonic_steps(self) -> None:
        self.assertTrue(is_gold_fever_monotonic_coin_candidate(2085, 2083, 0.99))
        self.assertTrue(is_gold_fever_monotonic_coin_candidate(2111, 2085, 0.97))
        self.assertFalse(is_gold_fever_monotonic_coin_candidate(3093, 2085, 0.76))
        self.assertFalse(is_gold_fever_monotonic_coin_candidate(2686, 2085, 0.99))
        self.assertFalse(is_gold_fever_monotonic_coin_candidate(2085, 2085, 0.99))

    def test_coin_counter_rejects_digit_attached_to_either_side(self) -> None:
        self.assertTrue(has_attached_digit_ocr_artifact(2534, 253))
        self.assertTrue(has_attached_digit_ocr_artifact(7253, 253))
        self.assertFalse(has_attached_digit_ocr_artifact(378, 253))

    def test_coin_candidate_uses_counter_row_not_top_noise(self) -> None:
        results = [
            ([[0, 0], [20, 0], [20, 10], [0, 10]], "14", 0.99),
            ([[5, 40], [75, 40], [75, 70], [5, 70]], "2,565", 0.95),
        ]
        value, confidence, raw = select_coin_candidate(results, crop_height=100)
        self.assertEqual(value, 2565)
        self.assertEqual(confidence, 0.95)
        self.assertEqual(raw, "2,565")

    def test_gold_crop_stays_left_of_kill_counter_region(self) -> None:
        width = 1920
        self.assertEqual(int(0.890 * width), 1708)
        self.assertEqual(int(0.985 * width), 1891)


class AutomatedSignalAdapterTests(unittest.TestCase):
    def test_kill_state_preserves_observed_vs_carried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "observations.jsonl"
            rows = []
            for index, observed in enumerate((True, False)):
                rows.append(
                    {
                        "observation_id": f"kill_{index}",
                        "video_asset_id": "video",
                        "session_id": "session",
                        "source_type": "automated",
                        "source_record_key": f"row:{index}",
                        "observable_code": "kill_counter",
                        "media_start_ms": index * 1000,
                        "media_end_ms": (index + 1) * 1000,
                        "temporal_precision": "window",
                        "numeric_value": 12,
                        "text_value": None,
                        "frame_number": None,
                        "evidence_json": {"modalities": ["ocr", "hud"]},
                        "attributes_json": {
                            "unit": "kills",
                            "source_value_observed": observed,
                            "source_state_source": (
                                "observed" if observed else "carried_forward"
                            ),
                            "needs_review": not observed,
                        },
                    }
                )
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            signals = list(
                adapt_automated_signals(
                    source, "sealed/observations.jsonl", start_ms=0, end_ms=2_000
                )
            )
        self.assertTrue(signals[0].observed)
        self.assertFalse(signals[1].observed)
        self.assertEqual(signals[1].numeric_value, 12)
        self.assertEqual(signals[1].evidence_grade, EvidenceGrade.C)
        self.assertEqual(
            signals[1].attributes["source_state_source"], "carried_forward"
        )

    def test_gem_quantity_comes_from_quantity_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "observations.jsonl"
            row = {
                "observation_id": "gem_1",
                "video_asset_id": "video",
                "session_id": "session",
                "source_type": "automated",
                "source_record_key": "gem:1",
                "observable_code": "gem_pickup",
                "media_start_ms": 100,
                "media_end_ms": 133,
                "temporal_precision": "window",
                "numeric_value": None,
                "quantity": 3,
                "text_value": None,
                "frame_number": 4,
                "evidence_json": {
                    "modalities": ["visual", "experience_bar_change"]
                },
                "attributes_json": {
                    "gem_type": "green",
                    "needs_review": False,
                },
            }
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            signal = next(
                iter(
                    adapt_automated_signals(
                        source,
                        "sealed/observations.jsonl",
                        start_ms=0,
                        end_ms=1_000,
                    )
                )
            )
        self.assertEqual(signal.numeric_value, 3)
        self.assertEqual(signal.unit, "estimated_gems")


class CatalogTests(unittest.TestCase):
    def test_catalog_has_unique_complete_scope(self) -> None:
        catalog = load_event_catalog(PROJECT_ROOT / "configs/event_catalog.json")
        codes = {row["code"] for row in catalog["events"]}
        self.assertGreaterEqual(len(codes), 40)
        self.assertIn("enemy_kill_counter", codes)
        self.assertIn("blue_gem_pickup", codes)
        self.assertIn("weapon_evolution", codes)
        self.assertIn("gold_fever", codes)
        self.assertFalse(catalog["principles"]["human_coded_ground_truth_used"])
        self.assertFalse(catalog["principles"]["blank_is_zero"])


class FiveSecondExportTests(unittest.TestCase):
    def test_half_open_window_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "output.csv"
            source.write_text(
                "interval_start_ms,interval_end_ms,kills_in_window\n"
                "0,5000,2\n"
                "5000,10000,3\n",
                encoding="utf-8",
            )
            events = [
                event_at(4_999, family="progression", event_type="level_up_selection"),
                event_at(5_000, family="progression", event_type="level_up_selection"),
                event_at(5_000, family="inventory", event_type="new_weapon"),
                event_at(9_999, family="consumable", event_type="floor_chicken"),
            ]
            self.assertEqual(write_five_second_windows(source, output, events), 2)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["level_up_selection_count"], "1")
            self.assertEqual(rows[1]["level_up_selection_count"], "1")
            self.assertEqual(rows[1]["inventory_change_count"], "1")
            self.assertEqual(rows[1]["consumable_reward_count"], "1")


if __name__ == "__main__":
    unittest.main()
