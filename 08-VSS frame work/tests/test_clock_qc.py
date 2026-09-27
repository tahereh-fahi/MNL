from __future__ import annotations

import sys
import unittest
from pathlib import Path



from vss_framework.normalization.clock_qc import evaluate_clock_series, format_clock, parse_clock


class ClockQCTests(unittest.TestCase):
    def test_parse_valid_clock_formats(self) -> None:
        self.assertEqual(parse_clock("00:00"), (0, None))
        self.assertEqual(parse_clock("16:17"), (977, None))
        self.assertEqual(parse_clock("01:02:03"), (3723, None))
        self.assertEqual(format_clock(977), "16:17")
        self.assertEqual(format_clock(3723), "01:02:03")

    def test_invalid_clock_components_are_rejected(self) -> None:
        self.assertEqual(parse_clock("13:83"), (None, "seconds_out_of_range"))
        self.assertEqual(parse_clock("-1:02"), (None, "invalid_characters"))
        self.assertEqual(parse_clock("NaN"), (None, "invalid_characters"))
        self.assertEqual(parse_clock(""), (None, "missing_ocr"))

    def test_pause_is_valid(self) -> None:
        result = evaluate_clock_series(
            ["05:14", "05:14", "05:14"], video_duration_ms=1_000_000
        )
        self.assertEqual([sample.status for sample in result], ["accepted"] * 3)

    def test_isolated_forward_spike_does_not_cascade(self) -> None:
        result = evaluate_clock_series(
            ["03:06", "13:07", "03:08"], video_duration_ms=1_000_000
        )
        self.assertEqual([sample.status for sample in result], ["accepted", "rejected", "accepted"])
        self.assertIn("isolated_order_violation", result[1].reasons)

    def test_isolated_regression_does_not_cascade(self) -> None:
        result = evaluate_clock_series(
            ["10:37", "10:33", "10:39"], video_duration_ms=1_000_000
        )
        self.assertEqual([sample.status for sample in result], ["accepted", "rejected", "accepted"])
        self.assertEqual(result[2].accepted_seconds, 639)

    def test_plausible_catchup_is_reviewed_not_dropped(self) -> None:
        result = evaluate_clock_series(
            ["04:41", "04:49", "04:50"], video_duration_ms=1_000_000
        )
        self.assertEqual(result[1].status, "review")
        self.assertEqual(result[1].accepted_seconds, 289)
        self.assertIn("large_positive_clock_discontinuity", result[1].reasons)


if __name__ == "__main__":
    unittest.main()
