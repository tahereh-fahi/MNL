"""Consumer regression replay from real frames and saved OCR evidence; no OCR job."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from vss_framework.adapters import adapt_automated_signals
from vss_framework.detectors.gems import Config
from vss_framework.detectors import gems
from vss_framework.detectors.legacy_gem_xp import normalize_completed_gem_worker
from vss_framework.detectors.hud import (
    CSV_FIELDS, KillSelection, TimerSelection, exclude_paused_hud_window,
)
from vss_framework.gameplay_state import level_up_pause_evidence
from vss_framework.hashing import sha256_file
from vss_framework.health_calibration import reject_isolated_short_widths
from vss_framework.io import write_jsonl
from vss_framework.normalization.hud_worker import normalize_hud
from vss_framework.telemetry_scan import measure_health_bar_width


FRAMEWORK = Path(__file__).resolve().parents[1]
WORKSPACE = FRAMEWORK.parent
HUD_WORKER = FRAMEWORK / "runs/video4_full_pipeline_check40_20260917_173051/hud_5d1f9e334ee43948/worker"
XP_FIXTURE = FRAMEWORK / "runs/video4_health_integration60_20260917_171042/xp"


class PauseMeasurementConsumersTest(unittest.TestCase):
    def video(self, filename):
        path = WORKSPACE / "00-Videos" / filename
        if not path.exists():
            self.skipTest("Real source video is unavailable")
        capture = cv2.VideoCapture(str(path))
        self.addCleanup(capture.release)
        return capture

    def frame(self, capture, index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        self.assertTrue(ok)
        return frame

    def test_real_transition_health_exclusion_survives_component_filter(self):
        capture = self.video("video5_Imelda_0_part1.mp4")
        samples = []
        for index in range(487, 495):
            frame = self.frame(capture, index)
            pause = level_up_pause_evidence(frame)
            width, confidence = (None, 0.0) if pause["blocked"] else measure_health_bar_width(frame, Config())
            samples.append({"frame_number": index, "width": width, "confidence": confidence,
                            "excluded_from_gameplay": pause["blocked"],
                            "rejection_reason": pause["reason"] if pause["blocked"] else None})
        self.assertFalse(samples[0]["excluded_from_gameplay"])
        self.assertTrue(all(row["excluded_from_gameplay"] for row in samples[1:]))
        reject_isolated_short_widths(samples)
        for row in samples[1:]:
            self.assertIsNone(row["width"])
            self.assertIsNone(row["raw_width_candidate"])
            self.assertIn(row["rejection_reason"], {
                "level_up_transition_candidate_excludes_gameplay_measurements",
                "level_up_panel_excludes_gameplay_measurements",
            })

    def test_real_paused_hud_window_stays_missing_through_normalization(self):
        if not (HUD_WORKER / "manifest.json").exists():
            self.skipTest("Saved real HUD evidence is unavailable")
        manifest = json.loads((HUD_WORKER / "manifest.json").read_text())
        for key in ("hud_observations", "ocr_candidates"):
            record = manifest["outputs"][key]
            self.assertEqual(sha256_file(HUD_WORKER / record["path"]), record["sha256"])
        ledger = [json.loads(line) for line in (HUD_WORKER / "ocr_candidates.jsonl").read_text().splitlines()]
        window = next(row for row in ledger if row["video_second"] == 9)
        capture = self.video("video4_Imelda_100.mp4")
        samples = []
        for source in window["samples"]:
            pause = level_up_pause_evidence(self.frame(capture, source["frame_index"]))
            samples.append({**source, "excluded_from_gameplay": pause["blocked"]})
        self.assertTrue(all(sample["excluded_from_gameplay"] for sample in samples))
        timer, kill = exclude_paused_hud_window(
            samples, TimerSelection(**window["selection"]["timer"]),
            KillSelection(**window["selection"]["kill"]),
        )
        self.assertFalse(timer.observed)
        self.assertIsNone(timer.normalized_text)
        self.assertFalse(kill.observed)
        self.assertIsNone(kill.state_value)
        with (HUD_WORKER / "hud_observations.csv").open(newline="") as handle:
            row = next(row for row in csv.DictReader(handle) if int(row["Video Second"]) == 9)
        # Re-project the saved row according to its independently decoded pause
        # evidence. These derived records are test inputs, never detector output.
        row.update({"Time Stamp": "", "Kill Counter Quantity": "", "timer_observed": False,
                    "timer_status": timer.status, "kill_observed": False, "kill_state_value": "",
                    "kill_state_source": kill.state_source, "kill_status": kill.status,
                    "excluded_from_gameplay": True, "exclusion_reason": "window_overlaps_level_up_pause"})
        config = json.loads((FRAMEWORK / "configs/video4.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source_csv = Path(directory) / "hud_observations.csv"
            with source_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            rows, summary, _ = normalize_hud(
                source_csv=source_csv, source_csv_relative="worker/hud_observations.csv",
                processing_run_id=manifest["worker"]["name"],
                session_id=config["dataset"]["session_id"],
                video_asset_id=config["dataset"]["video_asset_id"], duration_ms=40_000,
                config={"source_sampling_offsets_seconds": manifest["configuration"]["sample_offsets_seconds"],
                        "clock_qc": config["detectors"]["hud_clock"]["clock_qc"]},
            )
            self.assertEqual(summary["level_up_pause_excluded_seconds"], 1)
            self.assertEqual({row["observable_code"] for row in rows}, {"kill_counter", "game_clock"})
            canonical = Path(directory) / "canonical.jsonl"
            write_jsonl(canonical, rows)
            signals = list(adapt_automated_signals(canonical, canonical.name, start_ms=0, end_ms=40_000))
        for signal in signals:
            self.assertFalse(signal.observed)
            self.assertIsNone(signal.numeric_value)
            self.assertTrue(signal.attributes["excluded_from_gameplay"])
            self.assertEqual(signal.attributes["exclusion_reason"], "window_overlaps_level_up_pause")

    def test_real_boundary_only_replay_has_no_pickup_denominator(self):
        if not (XP_FIXTURE / "manifest.json").exists():
            self.skipTest("Saved real XP evidence is unavailable")
        manifest = json.loads((XP_FIXTURE / "manifest.json").read_text())
        self.assertEqual(sha256_file(XP_FIXTURE / "xp_signal.npz"),
                         manifest["outputs"]["xp_signal.npz"])
        # Replay recorded measurements as a regression fixture, not as proof
        # that changed extraction code has produced a new verified video run.
        with np.load(XP_FIXTURE / "xp_signal.npz") as saved:
            arrays = {name: saved[name] for name in saved.files}
        cfg = Config()
        events = gems.find_xp_events(manifest["video_metadata"], arrays,
                                     gems.find_level_resets(arrays, cfg), cfg)
        boundary = next(event for event in events if event.level_up_boundary_candidate)
        capture = self.video(manifest["source"]["filename"])
        frame_a = self.frame(capture, boundary.frame_a)
        frame_b = self.frame(capture, boundary.frame_b)
        summary_row, _ = gems.summarize_pair(
            boundary, frame_a.shape, [], [], [], [],
            gems.detect_player_anchor(frame_a, cfg), gems.detect_player_anchor(frame_b, cfg),
            gems.magnet_state_at(boundary.video_time_b, gems.MagnetModel(False, ()), cfg), cfg,
        )
        self.assertEqual(summary_row["collected_gems_total"], 0)
        config = json.loads((FRAMEWORK / "configs/video4.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source_csv = Path(directory) / "boundary_events.csv"
            gems.write_csv(source_csv, [summary_row])
            rows, summary, issues = normalize_completed_gem_worker(
                worker_summary={"xp_jump_events": len([summary_row])},
                source_csv=source_csv, source_csv_relative=source_csv.name,
                processing_run_id="real_boundary_fixture_replay",
                session_id=config["dataset"]["session_id"],
                video_asset_id=config["dataset"]["video_asset_id"], duration_ms=60_000,
                fps=manifest["video_metadata"]["fps"], config={"evidence_policy": "none"},
            )
        self.assertEqual(rows, [])
        self.assertEqual(summary["total_gem_quantity"], 0)
        self.assertIsNone(summary["unresolved_quantity_fraction"])
        self.assertEqual(summary["unresolved_quantity_fraction_denominator"], 0)
        self.assertEqual(summary["unresolved_quantity_fraction_status"], "no_measured_pickups")
        self.assertEqual(len(issues), 1)


if __name__ == "__main__":
    unittest.main()
