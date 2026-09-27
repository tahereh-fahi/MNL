"""Replay recorded OCR candidates without rerunning OCR or claiming accuracy."""

import json
from pathlib import Path
import unittest

from vss_framework.detectors.hud import (
    TimerSelection,
    _build_qc,
    advance_kill_state,
    calibration_candidate_seconds,
    exclude_paused_hud_window,
    reconcile_kill_output_rows,
    select_kill,
    select_kill_sequence,
)
from vss_framework.hashing import sha256_file


WORKER = (
    Path(__file__).resolve().parents[1]
    / "runs/video4_pause_validation_full_20260918_022656_89992"
    / "hud_af13a58801cf77f3/worker"
)


class HudPauseStateReplayTests(unittest.TestCase):
    @staticmethod
    def _record(second, values, *, excluded=False):
        samples = []
        for sample_index, value in enumerate(values):
            candidates = [] if value is None else [{
                "text": str(value),
                "confidence": 0.95,
                "frame_index": second * 10 + sample_index,
                "offset_seconds": 0.3 + 0.2 * sample_index,
            }]
            samples.append({
                "decoded": True,
                "excluded_from_gameplay": excluded,
                "exclusion_reason": (
                    "window_overlaps_level_up_pause" if excluded else None
                ),
                "kill_candidates": candidates,
            })
        return {"video_second": second, "samples": samples}

    def test_temporal_sequence_rejects_supported_overshoot_and_recovers(self):
        records = [
            self._record(0, [36, 36, 36]),
            self._record(1, [46, 46, 46]),
            self._record(2, [40, 40, 40]),
            self._record(3, [40, 40, 40]),
            self._record(4, [42, 42, 42]),
        ]

        selected = select_kill_sequence(records, initial_state=36)

        self.assertEqual(selected[1].status, "rejected_temporal_overshoot")
        self.assertEqual(selected[1].state_value, 36)
        self.assertTrue(selected[2].observed)
        self.assertEqual(selected[2].state_value, 40)
        self.assertEqual(selected[4].state_value, 42)

    def test_temporal_sequence_keeps_a_confirmed_real_increase(self):
        records = [
            self._record(0, [36, 36, 36]),
            self._record(1, [46, 46, 46]),
            self._record(2, [46, 47, 47]),
            self._record(3, [48, 48, 48]),
        ]

        selected = select_kill_sequence(records, initial_state=36)

        self.assertTrue(selected[1].observed)
        self.assertEqual(selected[1].state_value, 46)

    def test_one_lower_second_is_not_enough_to_reject_an_increase(self):
        records = [
            self._record(0, [36, 36, 36]),
            self._record(1, [46, 46, 46]),
            self._record(2, [40, 40, 40]),
            self._record(3, [47, 47, 47]),
        ]

        selected = select_kill_sequence(records, initial_state=36)

        self.assertTrue(selected[1].observed)
        self.assertEqual(selected[1].state_value, 46)

    def test_large_unsupported_jump_is_rejected_after_sparse_later_evidence(self):
        records = [self._record(0, [344, 344, 344])]
        records.append(self._record(1, [391, 395, 391]))
        records.extend(self._record(second, []) for second in range(2, 17))
        records.extend([
            self._record(17, [361]),
            self._record(18, [364]),
        ])

        selected = select_kill_sequence(records, initial_state=344)

        self.assertEqual(selected[1].status, "rejected_temporal_overshoot")
        self.assertEqual(selected[17].state_value, 361)
        self.assertEqual(selected[18].state_value, 364)

    def test_sparse_low_confidence_future_does_not_veto_large_jump(self):
        records = [self._record(0, [344, 344, 344])]
        records.append(self._record(1, [391, 391, 391]))
        records.extend(self._record(second, []) for second in range(2, 17))
        for second, value in ((17, 361), (18, 364)):
            record = self._record(second, [value])
            record["samples"][0]["kill_candidates"][0]["confidence"] = 0.2
            records.append(record)

        selected = select_kill_sequence(records, initial_state=344)

        self.assertTrue(selected[1].observed)
        self.assertEqual(selected[1].state_value, 391)

    def test_large_absolute_but_small_relative_change_is_retained(self):
        records = [self._record(0, [2298, 2298, 2298])]
        records.append(self._record(1, [2354, 2354, 2354]))
        records.extend(self._record(second, []) for second in range(2, 17))
        records.extend([
            self._record(17, [2302]),
            self._record(18, [2330]),
        ])

        selected = select_kill_sequence(records, initial_state=2298)

        self.assertTrue(selected[1].observed)
        self.assertEqual(selected[1].state_value, 2354)

    def test_confirmed_jump_recovers_after_unobserved_gameplay_gap(self):
        records = [self._record(0, [367, 367, 367])]
        records.extend(self._record(second, []) for second in range(1, 15))
        records.extend([
            self._record(15, [452, 452, 452]),
            self._record(16, [455, 455, 455]),
            self._record(17, [465, 465, 465]),
        ])

        selected = select_kill_sequence(records, initial_state=367)

        self.assertEqual(selected[15].state_value, 452)
        self.assertEqual(selected[15].status, "observed")

    def test_unconfirmed_gap_jump_remains_unobserved(self):
        records = [self._record(0, [367, 367, 367])]
        records.extend(self._record(second, []) for second in range(1, 15))
        records.append(self._record(15, [452, 452, 452]))
        records.extend(self._record(second, []) for second in range(16, 19))

        selected = select_kill_sequence(records, initial_state=367)

        self.assertEqual(selected[15].status, "unconfirmed_gap_jump")
        self.assertEqual(selected[15].state_value, 367)

    def test_excluded_pause_does_not_supply_overshoot_contradiction(self):
        records = [
            self._record(0, [36, 36, 36]),
            self._record(1, [46, 46, 46]),
            self._record(2, [40, 40, 40], excluded=True),
            self._record(3, [41, 41, 41], excluded=True),
            self._record(4, [47, 47, 47]),
        ]

        selected = select_kill_sequence(records, initial_state=36)

        self.assertTrue(selected[1].observed)
        self.assertEqual(selected[1].state_value, 46)

    def test_reconciliation_updates_public_rows_and_candidate_provenance(self):
        records = [
            self._record(0, [36, 36, 36]),
            self._record(1, [46, 46, 46]),
            self._record(2, [40, 40, 40]),
            self._record(3, [40, 40, 40]),
        ]
        rows = []
        for second, state in enumerate((36, 46, 46, 46)):
            rows.append({
                "Video Second": second,
                "timer_observed": True,
                "timer_fallback_attempted": False,
                "decoded_sample_count": 3,
                "Kill Counter Quantity": state,
                "kill_observed": True,
                "kill_raw_text": str(state),
                "kill_observed_value": state,
                "kill_confidence": 0.95,
                "kill_state_value": state,
                "kill_state_source": "observed",
                "kill_status": "observed",
                "review_flag": False,
            })
        for record, row in zip(records, rows):
            record["selection"] = {"kill": {
                "observed": True,
                "raw_text": row["kill_raw_text"],
                "observed_value": row["kill_observed_value"],
                "confidence": row["kill_confidence"],
                "state_value": row["kill_state_value"],
                "state_source": row["kill_state_source"],
                "status": row["kill_status"],
                "valid_candidate_count": 3,
            }}

        changed = reconcile_kill_output_rows(
            rows, records, initial_state=36, min_confidence=0.0
        )

        self.assertEqual(changed, [1, 2, 3])
        self.assertFalse(rows[1]["kill_observed"])
        self.assertEqual(rows[1]["kill_status"], "rejected_temporal_overshoot")
        self.assertEqual(rows[1]["kill_state_value"], 36)
        self.assertTrue(rows[1]["review_flag"])
        self.assertEqual(
            records[1]["selection"]["kill"]["status"],
            "rejected_temporal_overshoot",
        )
        self.assertEqual(rows[2]["kill_state_value"], 40)

    def test_calibration_candidates_follow_late_gameplay_onset(self):
        candidates = calibration_candidate_seconds(66, 120)
        self.assertIn(66.5, candidates)
        self.assertIn(67.0, candidates)
        self.assertIn(90.0, candidates)
        self.assertTrue(all(66 <= second < 120 for second in candidates))

    def test_qc_exempts_only_initial_blank_zero_from_kill_ocr_denominator(self):
        rows = []
        for second in range(7):
            observed = second in {3, 4, 5}
            rows.append({
                "Video Second": second,
                "timer_observed": True,
                "timer_fallback_attempted": False,
                "timer_preprocessing_variant": "",
                "timer_normalization_policy": "",
                "timer_status": "observed",
                "kill_observed": observed,
                "kill_status": "observed" if observed else "missing",
                "kill_state_value": 5 if second >= 3 else 0,
                "kill_state_source": (
                    "observed" if observed else "carried_forward"
                ),
                "excluded_from_gameplay": False,
                "exclusion_reason": "",
                "review_flag": not observed,
            })
        qc = _build_qc(
            rows, [{} for _ in rows], len(rows), 21, 0.5, 0.5
        )
        self.assertEqual(qc["metrics"]["initial_blank_zero_exempt_seconds"], 3)
        self.assertEqual(qc["metrics"]["kill_quality_eligible_seconds"], 4)
        self.assertEqual(qc["metrics"]["kill_observed_rate"], 0.75)
        self.assertEqual(qc["status"], "passed")

    def test_observed_zero_does_not_end_blank_zero_exemption(self):
        rows = []
        for second, observed, state in [
            (0, False, 0),
            (1, True, 0),
            (2, False, 0),
            (3, True, 5),
            (4, False, 5),
        ]:
            rows.append({
                "Video Second": second,
                "timer_observed": True,
                "timer_fallback_attempted": False,
                "timer_preprocessing_variant": "",
                "timer_normalization_policy": "",
                "timer_status": "observed",
                "kill_observed": observed,
                "kill_status": "observed" if observed else "missing",
                "kill_state_value": state,
                "kill_state_source": "observed" if observed else "carried_forward",
                "excluded_from_gameplay": False,
                "exclusion_reason": "",
                "review_flag": not observed,
            })
        qc = _build_qc(rows, [{} for _ in rows], 5, 15, 0.5, 0.5)
        self.assertEqual(qc["metrics"]["initial_blank_zero_exempt_seconds"], 2)
        self.assertEqual(qc["metrics"]["kill_quality_eligible_seconds"], 3)
        self.assertEqual(qc["metrics"]["kill_observed_rate"], 0.66666667)
        self.assertEqual(qc["status"], "passed")

    def test_trailing_skull_digit_is_removed_only_for_plausible_transition(self):
        self.assertEqual(
            select_kill([{"text": "140", "confidence": 1.0}], 13).state_value,
            14,
        )
        self.assertEqual(
            select_kill([{"text": "266", "confidence": 1.0}], 25).state_value,
            26,
        )
        # A genuine three-digit counter remains intact when the prior state
        # makes the full value, rather than a shortened prefix, plausible.
        self.assertEqual(
            select_kill([{"text": "140", "confidence": 1.0}], 139).state_value,
            140,
        )

    def test_pause_output_is_missing_but_internal_count_survives(self):
        prior = 1711
        samples = [{"excluded_from_gameplay": True}]
        selected = select_kill([{"text": "17", "confidence": 1.0}], prior)
        _, excluded = exclude_paused_hud_window(
            samples, TimerSelection(False, None, None, None, "missing", 0),
            selected,
        )
        self.assertIsNone(excluded.state_value)
        self.assertEqual(excluded.state_source, "excluded_level_up_pause")
        preserved = advance_kill_state(prior, excluded, excluded_window=True)
        self.assertEqual(preserved, prior)
        partial = select_kill([{"text": "17", "confidence": 1.0}], preserved)
        self.assertFalse(partial.observed)
        self.assertEqual(partial.state_value, prior)
        resumed = select_kill([{"text": "1743", "confidence": 1.0}], prior)
        self.assertEqual(resumed.state_value, 1743)

    def test_full_video4_saved_ocr_replay_has_no_count_drop(self):
        manifest_path = WORKER / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest("The saved Video 4 OCR ledger is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["outputs"]["ocr_candidates"]
        source = WORKER / record["path"]
        self.assertEqual(sha256_file(source), record["sha256"])

        previous = 0
        published_states = []
        states_at = {}
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                samples = row["samples"]
                candidates = [
                    candidate for sample in samples
                    for candidate in sample.get("kill_candidates", [])
                ]
                selection = select_kill(candidates, previous)
                excluded_window = any(
                    sample.get("excluded_from_gameplay") for sample in samples
                )
                if excluded_window:
                    _, selection = exclude_paused_hud_window(
                        samples,
                        TimerSelection(False, None, None, None, "missing", 0),
                        selection,
                    )
                previous = advance_kill_state(
                    previous, selection, excluded_window=excluded_window
                )
                states_at[row["video_second"]] = selection
                if selection.state_value is not None:
                    published_states.append(selection.state_value)

        self.assertEqual(len(states_at), 1196)
        self.assertTrue(all(
            later >= earlier
            for earlier, later in zip(published_states, published_states[1:])
        ))
        self.assertEqual(states_at[296].state_value, 1711)
        self.assertFalse(states_at[296].observed)
        self.assertEqual(states_at[297].state_value, 1743)
        self.assertEqual(states_at[915].state_value, 6989)
        self.assertEqual(states_at[944].state_value, 8087)
        self.assertEqual(states_at[971].state_value, 9162)


if __name__ == "__main__":
    unittest.main()
