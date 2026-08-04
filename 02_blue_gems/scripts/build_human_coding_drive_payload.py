#!/usr/bin/env python3
"""Prepare bounded Google Sheets payloads for Video 4 XP-event human coding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
EVENTS = (
    PROJECT / "results" / "video4_Imelda_100" / "collected_gems_cv_final"
    / "collected_gems_video4_imelda_100_xp_ab_events.csv"
)


def stamp(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def coding_queue() -> pd.DataFrame:
    events = pd.read_csv(EVENTS)
    second = np.floor(pd.to_numeric(events["video_time_b"], errors="coerce")).astype(int)
    queue = pd.DataFrame(
        {
            "event_id": events["event_id"].astype(int),
            "video_time": events["video_time_stamp"],
            "video_second": second,
            "second_interval": [f"{stamp(value)}-{stamp(value + 1)}" for value in second],
            "frame_a": events["frame_a"].astype(int),
            "frame_b": events["frame_b"].astype(int),
            "context_start_frame": (events["frame_a"] - 7).clip(lower=0).astype(int),
            "context_end_frame": (events["frame_b"] + 7).astype(int),
            "hud_level_before": pd.to_numeric(events["hud_level"], errors="coerce"),
            "inferred_level_after": pd.to_numeric(events["inferred_level"], errors="coerce"),
            "level_up_event": events["xp_bar_saturated"].eq(1),
            "pickup_confirmed": "",
            "blue_gems_collected": "",
            "green_gems_collected": "",
            "red_gems_collected": "",
            "unknown_color_gems": "",
            "total_gems_collected": "",
            "simultaneous_pickup": "",
            "color_confidence": "",
            "quantity_confidence": "",
            "needs_review": False,
            "review_reason": "",
            "coder_notes": "",
            "coder_id": "",
            "coded_at": "",
        }
    )
    return queue


def json_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    queue = coding_queue()
    chunk = queue.iloc[args.start : args.start + args.count]
    print(
        json.dumps(
            {
                "columns": list(queue.columns),
                "total_rows": len(queue),
                "start": args.start,
                "rows": [
                    [json_value(value) for value in row]
                    for row in chunk.itertuples(index=False, name=None)
                ],
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
