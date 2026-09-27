from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path



from vss_framework.normalization.gem_worker import normalize_gems
from vss_framework.normalization.hud_worker import normalize_hud


class WorkerTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_bad_clock_does_not_drop_kill_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hud.csv"
            self._write_csv(
                source,
                [
                    {"Video Second": 0, "Time Stamp": "00:00", "Kill Counter Quantity": 0, "review_flag": "False"},
                    {"Video Second": 1, "Time Stamp": "13:83", "Kill Counter Quantity": 10, "review_flag": "False"},
                    {"Video Second": 2, "Time Stamp": "00:02", "Kill Counter Quantity": 20, "review_flag": "False"},
                ],
            )
            observations, summary, qc = normalize_hud(
                source_csv=source,
                source_csv_relative="fixtures/hud.csv",
                processing_run_id="run_hud_test",
                session_id="session_test",
                video_asset_id="video_test",
                duration_ms=3000,
                config={"source_sampling_offsets_seconds": [0.3, 0.5, 0.7]},
            )
        kills = [row for row in observations if row["observable_code"] == "kill_counter"]
        clocks = [row for row in observations if row["observable_code"] == "game_clock"]
        self.assertEqual(len(kills), 3)
        self.assertEqual(len(clocks), 2)
        self.assertIsNone(kills[1]["game_time_ms"])
        self.assertEqual(kills[1]["attributes_json"]["source_ocr_time_stamp"], "13:83")
        self.assertEqual(kills[1]["attributes_json"]["game_clock_qc_status"], "rejected")
        self.assertEqual(clocks[-1]["numeric_value"], 2000)
        self.assertEqual(clocks[-1]["text_value"], "00:02")
        self.assertEqual(summary["clock_status_counts"]["rejected"], 1)
        self.assertTrue(any(row["raw_text"] == "13:83" for row in qc))

    def test_kill_decrease_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hud.csv"
            self._write_csv(
                source,
                [
                    {"Video Second": 0, "Time Stamp": "00:00", "Kill Counter Quantity": 10, "review_flag": "False"},
                    {"Video Second": 1, "Time Stamp": "00:01", "Kill Counter Quantity": 9, "review_flag": "False"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "decreases"):
                normalize_hud(
                    source_csv=source,
                    source_csv_relative="fixtures/hud.csv",
                    processing_run_id="run_hud_test",
                    session_id="session_test",
                    video_asset_id="video_test",
                    duration_ms=2000,
                    config={},
                )

    def test_enriched_hud_preserves_observation_provenance_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hud_observations.csv"
            self._write_csv(
                source,
                [
                    {
                        "Video Second": 0,
                        "Time Stamp": "00:00",
                        "Kill Counter Quantity": 7,
                        "review_flag": "False",
                        "timer_observed": "True",
                        "timer_raw_text": "00:00",
                        "timer_confidence": "0.91",
                        "timer_status": "observed",
                        "timer_primary_candidate_count": 0,
                        "timer_fallback_candidate_count": 2,
                        "timer_fallback_attempted": "True",
                        "timer_preprocessing_variant": "grayscale_upscaled_2x",
                        "timer_normalization_policy": "unique_embedded_mm_ss",
                        "kill_observed": "True",
                        "kill_confidence": "0.87",
                        "kill_state_value": 7,
                        "kill_state_source": "observed",
                        "kill_status": "observed",
                        "sample_frame_path": "evidence/hud_second_000000.jpg",
                    },
                    {
                        "Video Second": 1,
                        "Time Stamp": "00:01",
                        "Kill Counter Quantity": 7,
                        "review_flag": "True",
                        "timer_observed": "True",
                        "timer_raw_text": "00:01",
                        "timer_confidence": "0.93",
                        "timer_status": "observed",
                        "timer_primary_candidate_count": 3,
                        "timer_fallback_candidate_count": 0,
                        "timer_fallback_attempted": "False",
                        "timer_preprocessing_variant": "fixed_threshold_200_inv",
                        "timer_normalization_policy": "exact_cleaned_mm_ss",
                        "kill_observed": "False",
                        "kill_confidence": "",
                        "kill_state_value": 7,
                        "kill_state_source": "carried_forward",
                        "kill_status": "missing",
                        "sample_frame_path": "",
                    },
                ],
            )
            observations, summary, _ = normalize_hud(
                source_csv=source,
                source_csv_relative="hud-job/raw/hud_observations.csv",
                processing_run_id="run_hud_active",
                session_id="session_test",
                video_asset_id="video_test",
                duration_ms=2000,
                config={"source_sampling_offsets_seconds": [0.3, 0.5, 0.7]},
            )

        kills = [row for row in observations if row["observable_code"] == "kill_counter"]
        clocks = [row for row in observations if row["observable_code"] == "game_clock"]
        self.assertEqual(kills[0]["confidence"], 0.87)
        self.assertIsNone(kills[1]["confidence"])
        self.assertFalse(kills[0]["attributes_json"]["needs_review"])
        self.assertTrue(kills[1]["attributes_json"]["needs_review"])
        self.assertEqual(
            kills[1]["attributes_json"]["source_state_source"],
            "carried_forward",
        )
        self.assertEqual(clocks[0]["confidence"], 0.91)
        self.assertFalse(
            clocks[0]["attributes_json"]["source_may_forward_fill_value"]
        )
        self.assertTrue(
            clocks[0]["attributes_json"]["source_timer_fallback_attempted"]
        )
        self.assertEqual(
            clocks[0]["attributes_json"]["source_timer_preprocessing_variant"],
            "grayscale_upscaled_2x",
        )
        self.assertEqual(
            clocks[0]["attributes_json"]["source_timer_normalization_policy"],
            "unique_embedded_mm_ss",
        )
        self.assertEqual(
            kills[0]["evidence_json"]["sample_frame_path"],
            "hud-job/raw/evidence/hud_second_000000.jpg",
        )
        self.assertTrue(summary["source_confidence_available"])
        self.assertEqual(summary["kill_observed_count"], 1)
        self.assertEqual(summary["timer_observed_count"], 2)
        self.assertEqual(summary["timer_fallback_attempted_count"], 1)
        self.assertEqual(summary["timer_fallback_recovered_count"], 1)
        self.assertEqual(summary["timer_unique_embedded_normalization_count"], 1)
        self.assertEqual(
            summary["kill_state_source_counts"],
            {"carried_forward": 1, "observed": 1},
        )

    def _gem_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "event_id": 1,
            "event_key": "frame_000030",
            "frame_a": 29,
            "frame_b": 30,
            "video_time_a": "0.9667",
            "video_time_b": "1.0000",
            "collected_blue_gems": 1,
            "collected_green_gems": 2,
            "collected_red_gems": 0,
            "unresolved_collected_gems": 0,
            "collected_gems_total": 3,
            "color_evidence": "visual_trajectory",
            "confidence": "0.9",
            "needs_review": 0,
            "review_reason": "",
        }
        row.update(overrides)
        return row

    def test_multicolor_gem_event_preserves_quantity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "gems.csv"
            self._write_csv(source, [self._gem_row()])
            observations, summary, _ = normalize_gems(
                source_csv=source,
                source_csv_relative="fixtures/gems.csv",
                processing_run_id="run_gem_test",
                session_id="session_test",
                video_asset_id="video_test",
                duration_ms=2000,
                fps=30.0,
                config={},
            )
        self.assertEqual(len(observations), 2)
        self.assertEqual(sum(row["quantity"] for row in observations), 3)
        self.assertEqual(
            {row["attributes_json"]["gem_type"] for row in observations},
            {"blue", "green"},
        )
        self.assertTrue(all(row["temporal_precision"] == "window" for row in observations))
        self.assertTrue(all(row["media_start_ms"] == 967 for row in observations))
        self.assertTrue(all(row["media_end_ms"] == 1000 for row in observations))
        self.assertEqual(summary["total_gem_quantity"], 3)

    def test_unbalanced_gem_event_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "gems.csv"
            self._write_csv(source, [self._gem_row(collected_gems_total=4)])
            with self.assertRaisesRegex(ValueError, "do not balance"):
                normalize_gems(
                    source_csv=source,
                    source_csv_relative="fixtures/gems.csv",
                    processing_run_id="run_gem_test",
                    session_id="session_test",
                    video_asset_id="video_test",
                    duration_ms=2000,
                    fps=30.0,
                    config={},
                )


if __name__ == "__main__":
    unittest.main()
