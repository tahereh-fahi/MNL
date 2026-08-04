"""Prepare bounded JSON table payloads for the XP weak visual workbook."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


MODEL_OUTPUTS = (
    Path(__file__).resolve().parents[1]
    / "XP and Trajectory Gem Classification Model"
    / "model_outputs"
)

DATASETS = {
    "validation": {
        "file": "xp_weak_visual_validation_predictions.csv",
        "columns": [
            "event_id",
            "video_time_stamp",
            "hud_level",
            "crown_level_before",
            "crown_growth_percent",
            "total_growth_percent",
            "xp_bar_increase_percent",
            "estimated_base_xp_gain_crown_adjusted",
            "level_normalized_xp_gain",
            "color_evidence",
            "fold",
            "visual_reference_color",
            "model_prediction",
            "model_confidence",
            "probability_blue",
            "probability_green",
            "probability_red",
            "correct",
        ],
    },
    "five_second": {
        "file": "xp_weak_visual_5sec_intervals.csv",
        "columns": [
            "time_stamp",
            "interval_start_second",
            "interval_end_second",
            "xp_jump_events",
            "model_collected_blue_gems",
            "model_collected_green_gems",
            "model_collected_red_gems",
            "model_unresolved_collected_gems",
            "model_collected_gems_total",
            "model_assigned_events",
            "model_mean_confidence",
            "likely_collected_blue_gems",
            "likely_collected_green_gems",
            "likely_collected_red_gems",
            "likely_color_events",
            "event_ids",
        ],
    },
    "unresolved": {
        "file": "xp_weak_visual_unresolved_predictions.csv",
        "columns": [
            "event_id",
            "video_time_stamp",
            "frame_a",
            "frame_b",
            "hud_level",
            "reset_inferred_level",
            "inferred_level",
            "crown_level_before",
            "crown_growth_percent",
            "total_growth_percent",
            "xp_bar_increase_percent",
            "estimated_base_xp_gain_crown_adjusted",
            "level_normalized_xp_gain",
            "count_trajectory_colors",
            "count_trajectory_track_count",
            "count_trajectory_max_score",
            "count_trajectory_scores",
            "count_trajectory_disappearance_frames",
            "model_prediction",
            "model_confidence",
            "probability_blue",
            "probability_green",
            "probability_red",
            "level_up_visual_rule_applied",
            "model_status",
            "likely_color",
            "likely_color_confidence",
            "likely_color_reason",
            "pair_image",
            "percentage_assist_gate_reason",
        ],
    },
    "review": {
        "file": "xp_weak_visual_unresolved_predictions.csv",
        "columns": [
            "event_id",
            "video_time_stamp",
            "frame_a",
            "frame_b",
            "hud_level",
            "reset_inferred_level",
            "inferred_level",
            "crown_level_before",
            "crown_growth_percent",
            "total_growth_percent",
            "xp_bar_increase_percent",
            "estimated_base_xp_gain_crown_adjusted",
            "level_normalized_xp_gain",
            "count_trajectory_colors",
            "count_trajectory_track_count",
            "count_trajectory_max_score",
            "count_trajectory_scores",
            "count_trajectory_disappearance_frames",
            "model_prediction",
            "model_confidence",
            "probability_blue",
            "probability_green",
            "probability_red",
            "level_up_visual_rule_applied",
            "model_status",
            "likely_color",
            "likely_color_confidence",
            "likely_color_reason",
            "pair_image",
            "percentage_assist_gate_reason",
        ],
        "filter": "model_status != 'candidate_assignment'",
    },
    "human_evaluation": {
        "file": "xp_weak_visual_human_evaluation.csv",
        "columns": [
            "time_stamp", "interval_start_video_second", "interval_end_video_second",
            "actual_collected_blue_gems", "model_collected_blue_gems",
            "likely_collected_blue_gems", "exact_error", "likely_error",
            "actual_count_bin", "likely_count_bin", "source",
        ],
    },
    "human_metrics": {
        "file": "xp_weak_visual_human_metrics.csv",
        "columns": [
            "Estimate", "Intervals", "Exact-count accuracy", "Within-one accuracy",
            "MAE", "RMSE", "Bias", "Human total", "Predicted total", "Total error",
        ],
    },
    "human_confusion": {
        "file": "xp_weak_visual_human_count_confusion_matrix.csv",
        "columns": [
            "Unnamed: 0", "Predicted 0", "Predicted 1-5", "Predicted 6-10",
            "Predicted 11-20", "Predicted 21+",
        ],
    },
    "human_color_confusion": {
        "file": "xp_weak_visual_human_dominant_color_confusion.csv",
        "columns": [
            "Unnamed: 0", "Predicted Blue", "Predicted Green", "Predicted Red",
        ],
    },
    "human_color_presence": {
        "file": "xp_weak_visual_human_color_presence_metrics.csv",
        "columns": [
            "Color", "TN", "FP", "FN", "TP", "Presence accuracy",
            "Presence precision", "Presence recall",
        ],
    },
    "three_color_metrics": {
        "file": "xp_weak_visual_three_color_count_metrics.csv",
        "columns": [
            "Color", "Intervals", "Human total", "Predicted total", "Total error",
            "Bias", "MAE", "RMSE", "Exact-count accuracy",
            "Within-one accuracy", "Correlation",
        ],
    },
    "three_color_overall": {
        "file": "xp_weak_visual_three_color_overall_metrics.csv",
        "columns": ["Metric", "Value"],
    },
}


def json_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    frame = pd.read_csv(MODEL_OUTPUTS / config["file"])
    if "filter" in config:
        frame = frame.query(config["filter"])
    columns = config["columns"]
    frame = frame.loc[:, columns].reset_index(drop=True)

    chunk = frame.iloc[args.start : args.start + args.count]
    rows = [[json_value(value) for value in row] for row in chunk.itertuples(index=False, name=None)]
    print(
        json.dumps(
            {
                "columns": columns,
                "total_rows": len(frame),
                "start": args.start,
                "rows": rows,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
