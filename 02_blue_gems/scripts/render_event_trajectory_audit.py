#!/usr/bin/env python3
"""Recreate sensitive trajectory contact sheets for selected detector events."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import cv2
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
DEFAULT_OUTPUT = (
    PROJECT
    / "results"
    / "video4_Imelda_100"
    / "collected_gems_cv_final"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_ids", nargs="+", type=int)
    parser.add_argument(
        "--video",
        type=Path,
        default=WORKSPACE / "MNL" / "videos" / "video4_Imelda_100.mp4",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_OUTPUT / "collected_gems_video4_imelda_100_xp_ab_events.csv",
    )
    parser.add_argument(
        "--signal",
        type=Path,
        default=DEFAULT_OUTPUT / "collected_gems_video4_imelda_100_xp_frame_signal.csv",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=PROJECT / "templates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "event_reviews",
    )
    return parser.parse_args()


def load_detector():
    path = PROJECT / "scripts" / "detect_collected_gems_from_xp_ab.py"
    spec = importlib.util.spec_from_file_location("collected_gem_detector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import detector: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nullable_integer(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def event_from_row(detector, row: pd.Series):
    return detector.XPEvent(
        event_id=int(row["event_id"]),
        frame_a=int(row["frame_a"]),
        frame_b=int(row["frame_b"]),
        video_time_a=float(row["video_time_a"]),
        video_time_b=float(row["video_time_b"]),
        progress_a=float(row["xp_progress_a"]),
        progress_b=float(row["xp_progress_b"]),
        delta_pixels=int(row["xp_delta_pixels"]),
        delta_fraction=float(row["xp_delta_fraction"]),
        quality_a=float(row["xp_quality_a"]),
        quality_b=float(row["xp_quality_b"]),
        inferred_level=int(row["inferred_level"]),
        hud_level=int(row["hud_level"]),
        hud_level_ocr_raw=nullable_integer(row["hud_level_ocr_raw"]),
        hud_level_ocr_text="" if pd.isna(row["hud_level_ocr_text"]) else str(row["hud_level_ocr_text"]),
        hud_level_ocr_confidence=float(row["hud_level_ocr_confidence"]),
        hud_level_ocr_accepted=bool(int(row["hud_level_ocr_accepted"])),
        hud_level_ocr_validation=str(row["hud_level_ocr_validation"]),
        hud_level_source=str(row["hud_level_source"]),
        xp_required=float(row["xp_required_for_level"]),
        growth_multiplier=float(row["growth_multiplier"]),
        estimated_base_xp_gain=float(row["estimated_base_xp_gain"]),
        xp_color_hint=str(row["xp_color_hint"]),
        bar_saturated=bool(int(row["xp_bar_saturated"])),
        local_single_step_pixels=float(row["local_single_step_pixels"]),
        jump_step_ratio=float(row["xp_jump_step_ratio"]),
    )


def read_frames(video: Path, first: int, last: int) -> tuple[dict[int, object], float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = {}
    for frame_index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index}")
        frames[frame_index] = frame
    capture.release()
    return frames, fps


def main() -> None:
    args = arguments()
    detector = load_detector()
    cfg = detector.Config()
    count_cfg = detector.count_trajectory_config(cfg)
    table = pd.read_csv(args.events)
    requested = set(args.event_ids)
    target_rows = table.loc[table["event_id"].isin(requested)].copy()
    missing = requested - set(target_rows["event_id"].astype(int))
    if missing:
        raise ValueError(f"Unknown event IDs: {sorted(missing)}")

    target_first = int(target_rows["frame_b"].min()) - cfg.trajectory_history_frames
    target_last = int(target_rows["frame_b"].max()) + cfg.trajectory_post_frames
    context_rows = table.loc[
        table["frame_b"].le(target_rows["frame_b"].max())
        & (table["frame_b"] + cfg.trajectory_post_frames).ge(target_first)
        & (table["frame_b"] - cfg.trajectory_history_frames).le(target_last)
    ].sort_values(["frame_b", "event_id"])
    context_events = [event_from_row(detector, row) for _, row in context_rows.iterrows()]
    first_frame = min(event.frame_b - cfg.trajectory_history_frames for event in context_events)
    last_frame = max(event.frame_b + cfg.trajectory_post_frames for event in context_events)
    frames, fps = read_frames(args.video, first_frame, last_frame)

    signal = pd.read_csv(args.signal).set_index("frame_index")
    anchors = {}
    for frame_index in frames:
        row = signal.loc[frame_index]
        anchors[frame_index] = detector.PlayerAnchor(
            x=float(row["player_anchor_x"]),
            y=float(row["player_anchor_y"]),
            health_bar_detected=bool(int(row["health_bar_detected"])),
            health_bar_x=float(row["health_bar_x"]),
            health_bar_y=float(row["health_bar_y"]),
            confidence=float(row["health_bar_confidence"]),
        )

    sample = frames[first_frame]
    templates = detector.load_template_bank(
        args.templates,
        sample.shape[0],
        profile="legacy",
    )
    detections = {
        frame_index: detector.detect_gems_near_player(
            frame,
            templates,
            anchors[frame_index],
            count_cfg,
        )
        for frame_index, frame in frames.items()
    }

    claimed = []
    audit_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for event in context_events:
        candidates = detector.deduplicate_count_trajectories(
            detector.find_trajectory_evidence(
                event,
                frames[event.frame_b].shape,
                detections,
                anchors,
                count_cfg,
            ),
            frames[event.frame_b].shape,
            cfg,
        )
        claim_distance = (
            cfg.count_trajectory_dedup_distance_1440p
            * frames[event.frame_b].shape[0]
            / 1440.0
        )
        trajectories = []
        for candidate in candidates:
            candidate_end = candidate.points[-1][1]
            already_claimed = any(
                abs(candidate.disappearance_frame - earlier.disappearance_frame) <= 1
                and math.hypot(
                    candidate_end.cx - earlier.points[-1][1].cx,
                    candidate_end.cy - earlier.points[-1][1].cy,
                )
                <= claim_distance
                for earlier in claimed
            )
            if not already_claimed:
                trajectories.append(candidate)
        claimed.extend(trajectories)
        if event.event_id not in requested:
            continue
        path = args.output_dir / (
            f"event_{event.event_id:05d}_sensitive_trajectory_"
            f"frames_{event.frame_b - cfg.trajectory_history_frames}_"
            f"{event.frame_b + cfg.trajectory_post_frames}.jpg"
        )
        detector.write_trajectory_preview(
            path,
            event,
            frames,
            detections,
            anchors,
            trajectories,
            fps,
            count_cfg,
        )
        for trajectory_index, trajectory in enumerate(trajectories, start=1):
            audit_rows.append(
                {
                    "event_id": event.event_id,
                    "trajectory": trajectory_index,
                    "color": trajectory.color,
                    "start_frame": trajectory.points[0][0],
                    "end_frame": trajectory.points[-1][0],
                    "disappearance_frame": trajectory.disappearance_frame,
                    "approach_pixels": round(trajectory.approach_pixels, 2),
                    "monotonic_fraction": round(trajectory.monotonic_fraction, 4),
                    "mean_template_score": round(trajectory.mean_template_score, 4),
                    "evidence_score": round(trajectory.evidence_score, 4),
                    "audit_image": str(path),
                }
            )
        print(f"Event {event.event_id}: {len(trajectories)} trajectories -> {path}")
    pd.DataFrame(audit_rows).to_csv(
        args.output_dir / "sensitive_trajectory_audit.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
