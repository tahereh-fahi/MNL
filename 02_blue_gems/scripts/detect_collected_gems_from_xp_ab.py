#!/usr/bin/env python3
"""Detect collected blue, green, and red gems around XP-bar jumps.

This pipeline is intentionally label-free. It does not read the Gaming Content
spreadsheet or any human-coded count. The XP bar is measured on every video
frame to flag pickup moments. Gem recognition is then run only in a short
temporal window around each flag:

* frame A: the final frame before the XP bar increases
* frame B: the first frame with the increased XP bar
* history frames: the frames immediately before A, used to track gems moving
  toward the player and to absorb a short XP-HUD rendering delay

A frame-specific player anchor is located immediately above Imelda's red health
bar. Screen center is used only when the bar is hidden. XP-bar restorations during
level-up UI transitions are rejected before pickup events are created.

A colored sprite that approaches the player for several frames and disappears
at or shortly before B is the strongest color evidence. The original A/B
disappearance remains as a fallback. When the sprite is hidden under the
player, XP-jump magnitude supplies a lower-confidence color hint. Saturated
level-up events are left unresolved unless visual evidence supplies the color.
Each color can use several reference sprites; duplicate detections from those
references are merged before trajectory analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Config:
    xp_bar_x0_fraction: float = 0.042
    xp_bar_x1_fraction: float = 0.948
    xp_bar_y0_fraction: float = 0.006
    xp_bar_y1_fraction: float = 0.019
    xp_hue_low: int = 100
    xp_hue_high: int = 125
    xp_saturation_low: int = 80
    xp_value_low: int = 50
    xp_column_fraction: float = 0.55
    xp_min_quality: float = 0.78
    xp_min_jump_pixels: int = 1
    xp_persistence_tolerance_pixels: int = 1
    xp_overlay_threshold: float = 0.05
    hud_score_threshold: float = 0.90
    reset_drop_fraction: float = 0.20
    reset_collapse_frames: int = 15
    base_growth_multiplier: float = 1.15
    template_threshold: float = 0.78
    visual_strong_threshold: float = 0.88
    search_radius_1440p: float = 320.0
    pickup_radius_1440p: float = 150.0
    match_distance_1440p: float = 105.0
    trajectory_history_frames: int = 8
    trajectory_post_frames: int = 1
    trajectory_max_hud_lag_frames: int = 2
    trajectory_min_frames: int = 4
    trajectory_link_distance_1440p: float = 70.0
    trajectory_min_approach_1440p: float = 40.0
    trajectory_monotonic_tolerance_1440p: float = 8.0
    trajectory_min_monotonic_fraction: float = 0.65
    trajectory_min_template_score: float = 0.86
    trajectory_min_strong_history_frames: int = 3
    trajectory_max_occluded_endpoint_frames: int = 2
    trajectory_dedup_distance_1440p: float = 70.0
    count_template_threshold: float = 0.72
    count_pickup_radius_1440p: float = 175.0
    count_trajectory_min_frames: int = 3
    count_trajectory_min_approach_1440p: float = 25.0
    count_trajectory_min_monotonic_fraction: float = 0.60
    count_trajectory_min_template_score: float = 0.76
    count_trajectory_min_strong_history_frames: int = 2
    count_trajectory_max_occluded_endpoint_frames: int = 3
    count_trajectory_dedup_distance_1440p: float = 80.0
    percentage_calibration_end_seconds: float = 210.0
    percentage_validation_end_seconds: float = 300.0
    percentage_target_precision: float = 0.95
    percentage_min_validation_predictions: int = 20
    percentage_blue_max_base_xp: float = 2.25
    percentage_green_max_base_xp: float = 9.50
    percentage_isolation_frames: int = 1
    percentage_growth_rolling_samples: int = 40
    percentage_growth_min_rolling_samples: int = 12
    nms_radius_1440p: float = 24.0
    component_pad_1440p: int = 12
    preview_jpeg_quality: int = 84
    health_bar_search_x_1440p: float = 400.0
    health_bar_search_y_1440p: float = 260.0
    health_bar_min_width_1440p: float = 40.0
    health_bar_max_width_1440p: float = 135.0
    health_bar_min_height_1440p: float = 5.0
    health_bar_max_height_1440p: float = 24.0
    health_bar_min_aspect_ratio: float = 3.5
    health_bar_expected_y_offset_1440p: float = 12.5
    health_bar_max_x_error_1440p: float = 120.0
    health_bar_max_y_error_1440p: float = 35.0
    player_anchor_above_health_bar_1440p: float = 12.5
    level_up_strong_overlay_threshold: float = 0.30
    level_up_min_strong_frames: int = 3
    level_up_guard_before_frames: int = 8
    level_up_guard_after_frames: int = 8
    xp_reappearance_lookback_frames: int = 12
    level_crop_x0_fraction: float = 0.906
    level_crop_x1_fraction: float = 0.948
    level_crop_y0_fraction: float = 0.004
    level_crop_y1_fraction: float = 0.026
    level_ocr_scale: float = 5.0
    level_ocr_min_confidence: float = 0.45
    level_ocr_max_reset_lead: int = 8
    level_ocr_max_reset_lag: int = 2
    level_ocr_spike_lookahead_events: int = 6


@dataclass(frozen=True)
class ColorSpec:
    name: str
    hue_ranges: tuple[tuple[int, int], ...]
    draw_color: tuple[int, int, int]


@dataclass(frozen=True)
class GemDetection:
    color: str
    cx: float
    cy: float
    score: float


@dataclass(frozen=True)
class PlayerAnchor:
    x: float
    y: float
    health_bar_detected: bool
    health_bar_x: float
    health_bar_y: float
    confidence: float


@dataclass(frozen=True)
class XPEvent:
    event_id: int
    frame_a: int
    frame_b: int
    video_time_a: float
    video_time_b: float
    progress_a: float
    progress_b: float
    delta_pixels: int
    delta_fraction: float
    quality_a: float
    quality_b: float
    inferred_level: int
    hud_level: int
    hud_level_ocr_raw: int | None
    hud_level_ocr_text: str
    hud_level_ocr_confidence: float
    hud_level_ocr_accepted: bool
    hud_level_ocr_validation: str
    hud_level_source: str
    xp_required: float
    growth_multiplier: float
    estimated_base_xp_gain: float
    xp_color_hint: str
    bar_saturated: bool
    local_single_step_pixels: float
    jump_step_ratio: float


@dataclass(frozen=True)
class TrajectoryEvidence:
    color: str
    points: tuple[tuple[int, GemDetection], ...]
    disappearance_frame: int
    start_distance: float
    end_distance: float
    approach_pixels: float
    monotonic_fraction: float
    mean_template_score: float
    evidence_score: float
    strong_template_frames: int
    weak_endpoint_frames: int
    occlusion_accepted: bool


COLORS = (
    ColorSpec("blue", ((88, 106),), (255, 140, 0)),
    ColorSpec("green", ((72, 88),), (0, 220, 0)),
    ColorSpec("red", ((0, 12), (170, 179)), (0, 0, 255)),
)

GemTemplate = tuple[np.ndarray, np.ndarray]
GemTemplateBank = dict[str, tuple[GemTemplate, ...]]


def seconds_to_stamp(seconds: float, include_milliseconds: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    if include_milliseconds:
        return f"{minutes}:{remainder:06.3f}"
    return f"{minutes}:{int(remainder):02d}"


def interval_stamp(start: int, end: int) -> str:
    return f"{seconds_to_stamp(start)}-{seconds_to_stamp(end)}"


def clean_video_label(path: Path) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in path.stem.lower())
    return "_".join(cleaned.split("_")).strip("_") or "video"


def measure_xp_bar_progress(frame_bgr: np.ndarray, cfg: Config) -> tuple[float, float]:
    height, width = frame_bgr.shape[:2]
    x0 = int(round(width * cfg.xp_bar_x0_fraction))
    x1 = int(round(width * cfg.xp_bar_x1_fraction))
    y0 = int(round(height * cfg.xp_bar_y0_fraction))
    y1 = int(round(height * cfg.xp_bar_y1_fraction))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return math.nan, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = (
        (hsv[:, :, 0] >= cfg.xp_hue_low)
        & (hsv[:, :, 0] <= cfg.xp_hue_high)
        & (hsv[:, :, 1] >= cfg.xp_saturation_low)
        & (hsv[:, :, 2] >= cfg.xp_value_low)
    )
    columns = (np.mean(blue, axis=0) >= cfg.xp_column_fraction).astype(np.uint8)
    columns = cv2.morphologyEx(
        columns.reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, 5), dtype=np.uint8),
    ).reshape(-1)
    indexes = np.flatnonzero(columns)
    if not len(indexes):
        return 0.0, 1.0
    endpoint = int(indexes[-1])
    quality = float(np.mean(columns[: endpoint + 1]))
    return float((endpoint + 1) / len(columns)), quality


def gameplay_hud_score(frame_bgr: np.ndarray) -> float:
    """Measure the long gold XP-bar border that exists only during gameplay."""
    height, width = frame_bgr.shape[:2]
    x0 = int(round(width * 0.04))
    x1 = int(round(width * 0.76))
    roi = frame_bgr[: max(8, int(round(height * 0.03))) : 3, x0:x1:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gold = (
        (hsv[:, :, 0] >= 8)
        & (hsv[:, :, 0] <= 35)
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 80)
    )
    return float(np.mean(np.any(gold, axis=0)))


def level_up_overlay_score(frame_bgr: np.ndarray) -> float:
    height, width = frame_bgr.shape[:2]
    roi = frame_bgr[
        int(height * 0.09) : int(height * 0.20) : 4,
        int(width * 0.36) : int(width * 0.64) : 4,
    ]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    panel = (
        (hsv[:, :, 0] >= 105)
        & (hsv[:, :, 0] <= 145)
        & (hsv[:, :, 1] >= 25)
        & (hsv[:, :, 1] <= 100)
        & (hsv[:, :, 2] >= 50)
        & (hsv[:, :, 2] <= 180)
    )
    return float(np.mean(panel))


def detect_player_anchor(frame_bgr: np.ndarray, cfg: Config) -> PlayerAnchor:
    """Locate Imelda from the long red health bar drawn directly below her."""
    height, width = frame_bgr.shape[:2]
    scale = height / 1440.0
    center_x = width / 2.0
    center_y = height / 2.0
    search_x = int(round(cfg.health_bar_search_x_1440p * scale))
    search_y = int(round(cfg.health_bar_search_y_1440p * scale))
    x0 = max(0, int(round(center_x)) - search_x)
    x1 = min(width, int(round(center_x)) + search_x)
    y0 = max(0, int(round(center_y)) - search_y)
    y1 = min(height, int(round(center_y)) + search_y)
    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    red = (
        ((hue <= 8) | (hue >= 172))
        & (hsv[:, :, 1] >= 150)
        & (hsv[:, :, 2] >= 70)
    ).astype(np.uint8) * 255
    count, _, stats, centers = cv2.connectedComponentsWithStats(red, connectivity=8)

    expected_bar_y = center_y + cfg.health_bar_expected_y_offset_1440p * scale
    candidates: list[tuple[float, float, float, float]] = []
    for component in range(1, count):
        local_x = int(stats[component, cv2.CC_STAT_LEFT])
        local_y = int(stats[component, cv2.CC_STAT_TOP])
        component_width = float(stats[component, cv2.CC_STAT_WIDTH])
        component_height = float(stats[component, cv2.CC_STAT_HEIGHT])
        area = float(stats[component, cv2.CC_STAT_AREA])
        bar_x = x0 + float(centers[component, 0])
        bar_y = y0 + float(centers[component, 1])
        if not (
            cfg.health_bar_min_width_1440p * scale
            <= component_width
            <= cfg.health_bar_max_width_1440p * scale
        ):
            continue
        if not (
            cfg.health_bar_min_height_1440p * scale
            <= component_height
            <= cfg.health_bar_max_height_1440p * scale
        ):
            continue
        if component_width / max(1.0, component_height) < cfg.health_bar_min_aspect_ratio:
            continue
        if abs(bar_x - center_x) > cfg.health_bar_max_x_error_1440p * scale:
            continue
        if abs(bar_y - expected_bar_y) > cfg.health_bar_max_y_error_1440p * scale:
            continue

        rectangularity = area / max(1.0, component_width * component_height)
        position_error = (
            abs(bar_x - center_x)
            + 2.0 * abs(bar_y - expected_bar_y)
        ) / max(1.0, scale)
        rank = position_error - 12.0 * rectangularity
        confidence = float(
            np.clip(
                0.55
                + 0.35 * rectangularity
                - 0.002 * position_error,
                0.0,
                1.0,
            )
        )
        candidates.append((rank, bar_x, bar_y, confidence))

    if not candidates:
        return PlayerAnchor(
            x=center_x,
            y=center_y,
            health_bar_detected=False,
            health_bar_x=math.nan,
            health_bar_y=math.nan,
            confidence=0.0,
        )

    _, bar_x, bar_y, confidence = min(candidates, key=lambda item: item[0])
    return PlayerAnchor(
        x=bar_x,
        y=bar_y - cfg.player_anchor_above_health_bar_1440p * scale,
        health_bar_detected=True,
        health_bar_x=bar_x,
        health_bar_y=bar_y,
        confidence=confidence,
    )


def build_level_up_transition_guard(
    overlay: np.ndarray,
    health_bar_detected: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Mark sustained level-up panels plus their opening and closing frames."""
    strong = (
        (overlay >= cfg.level_up_strong_overlay_threshold)
        & ~health_bar_detected.astype(bool)
    )
    guard = np.zeros(len(strong), dtype=bool)
    start = 0
    while start < len(strong):
        if not strong[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(strong) and strong[end + 1]:
            end += 1
        if end - start + 1 >= cfg.level_up_min_strong_frames:
            guard[
                max(0, start - cfg.level_up_guard_before_frames) :
                min(len(guard), end + cfg.level_up_guard_after_frames + 1)
            ] = True
        start = end + 1
    return guard


def find_xp_bar_reappearances(pixels: np.ndarray, cfg: Config) -> np.ndarray:
    """Flag a bar that returns after a short unreadable zero-valued gap."""
    recovered = np.zeros(len(pixels), dtype=bool)
    tolerance = cfg.xp_persistence_tolerance_pixels
    for frame_b in range(1, len(pixels)):
        if pixels[frame_b] <= tolerance or pixels[frame_b - 1] > tolerance:
            continue
        first = max(-1, frame_b - cfg.xp_reappearance_lookback_frames - 1)
        previous_visible = next(
            (
                int(pixels[index])
                for index in range(frame_b - 2, first, -1)
                if pixels[index] > tolerance
            ),
            None,
        )
        if previous_visible is not None and previous_visible >= pixels[frame_b] - tolerance:
            recovered[frame_b] = True
    return recovered


def experience_required_for_level(level: int) -> float:
    if level <= 20:
        required = 5.0 + 10.0 * (level - 1)
    elif level <= 40:
        required = 195.0 + 13.0 * (level - 20)
    else:
        required = 455.0 + 16.0 * (level - 40)
    if level == 20:
        required += 600.0
    if level == 40:
        required += 2400.0
    return required


def imelda_growth_multiplier(level: int, cfg: Config) -> float:
    character_bonus = 0.10 * min(level // 5, 3)
    level_spike_bonus = 1.0 if level in (20, 40) else 0.0
    return cfg.base_growth_multiplier + character_bonus + level_spike_bonus


def read_hud_level(
    frame_bgr: np.ndarray,
    reader: object,
    cfg: Config,
) -> tuple[int | None, float, str]:
    """Read the fixed `LV n` label at the right end of the XP bar."""
    height, width = frame_bgr.shape[:2]
    x0 = int(round(width * cfg.level_crop_x0_fraction))
    x1 = int(round(width * cfg.level_crop_x1_fraction))
    y0 = int(round(height * cfg.level_crop_y0_fraction))
    y1 = int(round(height * cfg.level_crop_y1_fraction))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0, ""
    enlarged = cv2.resize(
        crop,
        None,
        fx=cfg.level_ocr_scale,
        fy=cfg.level_ocr_scale,
        interpolation=cv2.INTER_CUBIC,
    )
    results = reader.recognize(
        enlarged,
        detail=1,
        allowlist="LV0123456789",
        contrast_ths=0.05,
        adjust_contrast=0.70,
    )
    candidates: list[tuple[float, int, str]] = []
    for result in results:
        if len(result) < 3:
            continue
        text = re.sub(r"[^LV0-9]", "", str(result[1]).upper())
        match = re.fullmatch(r"(?:LV)?([0-9]{1,3})", text)
        if not match:
            continue
        level = int(match.group(1))
        confidence = float(result[2])
        if level >= 1:
            candidates.append((confidence, level, text))
    if not candidates:
        return None, 0.0, ""
    confidence, level, text = max(candidates)
    return level, confidence, text


def attach_hud_levels(
    video_path: Path,
    events: list[XPEvent],
    cfg: Config,
) -> list[XPEvent]:
    """OCR the visible level on every flagged XP frame and recompute XP diagnostics."""
    if not events:
        return []
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "easyocr is required to read the visible LV value on XP events"
        ) from exc

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
        download_enabled=False,
    )
    events_by_frame = {event.frame_b: event for event in events}
    observations: dict[int, tuple[int | None, float, str]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for level OCR: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    next_report = 500
    observed_count = 0
    for frame_index in range(frame_count):
        ok = cap.grab()
        if not ok:
            break
        if frame_index not in events_by_frame:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        observations[frame_index] = read_hud_level(frame, reader, cfg)
        observed_count += 1
        if observed_count >= next_report:
            print(f"Level OCR: {observed_count}/{len(events)} events", flush=True)
            next_report += 500
    cap.release()

    raw_observations = [
        observations.get(event.frame_b, (None, 0.0, "")) for event in events
    ]
    base_valid: list[bool] = []
    validation: list[str] = []
    for event, (raw_level, confidence, raw_text) in zip(events, raw_observations):
        if raw_level is None:
            base_valid.append(False)
            validation.append("ocr_missing")
        elif confidence < cfg.level_ocr_min_confidence:
            base_valid.append(False)
            validation.append("ocr_low_confidence")
        elif not raw_text.startswith("LV"):
            base_valid.append(False)
            validation.append("ocr_missing_lv_prefix")
        elif raw_level > event.inferred_level + cfg.level_ocr_max_reset_lead:
            base_valid.append(False)
            validation.append("ocr_too_far_ahead_of_reset_count")
        elif raw_level < max(1, event.inferred_level - cfg.level_ocr_max_reset_lag):
            base_valid.append(False)
            validation.append("ocr_too_far_behind_reset_count")
        else:
            base_valid.append(True)
            validation.append("accepted")

    for index, (event, observation) in enumerate(zip(events, raw_observations)):
        if not base_valid[index]:
            continue
        raw_level = int(observation[0])
        future_levels = [
            int(raw_observations[future][0])
            for future in range(
                index + 1,
                min(len(events), index + cfg.level_ocr_spike_lookahead_events + 1),
            )
            if base_valid[future]
        ]
        if sum(level < raw_level for level in future_levels) >= 2:
            base_valid[index] = False
            validation[index] = "ocr_temporal_spike"

    enriched: list[XPEvent] = []
    previous_ocr_level: int | None = None
    for index, event in enumerate(events):
        raw_level, confidence, raw_text = raw_observations[index]
        accepted = base_valid[index]
        if accepted and previous_ocr_level is not None and raw_level < previous_ocr_level:
            accepted = False
            validation[index] = "ocr_temporal_decrease"

        if accepted:
            level = int(raw_level)
            source = "hud_ocr"
            previous_ocr_level = level
        elif previous_ocr_level is not None:
            level = previous_ocr_level
            source = "previous_hud_ocr_fallback"
        else:
            level = event.inferred_level
            source = "reset_inference_fallback"

        required = experience_required_for_level(level)
        growth = imelda_growth_multiplier(level, cfg)
        estimated_base_xp = event.delta_fraction * required / growth
        hint = color_hint_from_xp(estimated_base_xp, event.bar_saturated)
        enriched.append(
            replace(
                event,
                hud_level=level,
                hud_level_ocr_raw=raw_level,
                hud_level_ocr_text=raw_text,
                hud_level_ocr_confidence=confidence,
                hud_level_ocr_accepted=accepted,
                hud_level_ocr_validation=validation[index],
                hud_level_source=source,
                xp_required=required,
                growth_multiplier=growth,
                estimated_base_xp_gain=estimated_base_xp,
                xp_color_hint=hint,
            )
        )
    return enriched


def scan_xp_signal(
    video_path: Path,
    cfg: Config,
    max_seconds: float | None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    if max_seconds is not None:
        frame_count = min(frame_count, int(math.ceil(max_seconds * fps)))

    progress: list[float] = []
    quality: list[float] = []
    overlay: list[float] = []
    hud: list[float] = []
    player_anchor_x: list[float] = []
    player_anchor_y: list[float] = []
    health_bar_x: list[float] = []
    health_bar_y: list[float] = []
    health_bar_confidence: list[float] = []
    health_bar_detected: list[bool] = []
    next_report = 120.0
    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        value, frame_quality = measure_xp_bar_progress(frame, cfg)
        progress.append(value)
        quality.append(frame_quality)
        overlay.append(level_up_overlay_score(frame))
        hud.append(gameplay_hud_score(frame))
        anchor = detect_player_anchor(frame, cfg)
        player_anchor_x.append(anchor.x)
        player_anchor_y.append(anchor.y)
        health_bar_x.append(anchor.health_bar_x)
        health_bar_y.append(anchor.health_bar_y)
        health_bar_confidence.append(anchor.confidence)
        health_bar_detected.append(anchor.health_bar_detected)
        video_time = frame_index / fps
        if video_time >= next_report:
            print(f"XP scan: {seconds_to_stamp(video_time)}", flush=True)
            next_report += 120.0
    cap.release()

    actual_count = len(progress)
    bar_width = int(round(width * (cfg.xp_bar_x1_fraction - cfg.xp_bar_x0_fraction)))
    arrays = {
        "progress": np.asarray(progress, dtype=np.float32),
        "quality": np.asarray(quality, dtype=np.float32),
        "overlay": np.asarray(overlay, dtype=np.float32),
        "hud": np.asarray(hud, dtype=np.float32),
        "player_anchor_x": np.asarray(player_anchor_x, dtype=np.float32),
        "player_anchor_y": np.asarray(player_anchor_y, dtype=np.float32),
        "health_bar_x": np.asarray(health_bar_x, dtype=np.float32),
        "health_bar_y": np.asarray(health_bar_y, dtype=np.float32),
        "health_bar_confidence": np.asarray(health_bar_confidence, dtype=np.float32),
        "health_bar_detected": np.asarray(health_bar_detected, dtype=bool),
    }
    arrays["pixels"] = np.rint(arrays["progress"] * bar_width).astype(np.int32)
    arrays["delta_pixels"] = np.diff(arrays["pixels"], prepend=arrays["pixels"][0])
    arrays["level_up_transition_guard"] = build_level_up_transition_guard(
        arrays["overlay"],
        arrays["health_bar_detected"],
        cfg,
    )
    arrays["xp_bar_reappearance"] = find_xp_bar_reappearances(arrays["pixels"], cfg)
    metadata = {
        "fps": fps,
        "frame_count": float(actual_count),
        "width": float(width),
        "height": float(height),
        "duration": actual_count / fps,
        "bar_width": float(bar_width),
    }
    return metadata, arrays


def find_level_resets(arrays: dict[str, np.ndarray], cfg: Config) -> list[int]:
    pixels = arrays["pixels"]
    delta = arrays["delta_pixels"]
    hud = arrays["hud"]
    reappearance = arrays["xp_bar_reappearance"]
    width = max(1, int(np.max(pixels)))
    candidates = np.flatnonzero(
        (delta <= -cfg.reset_drop_fraction * width)
        & (hud >= cfg.hud_score_threshold)
        & (np.r_[hud[0], hud[:-1]] >= cfg.hud_score_threshold)
    )
    valid_positive = np.zeros(len(pixels), dtype=bool)
    for frame_index in range(1, len(pixels)):
        frame_a = frame_index - 1
        valid_positive[frame_index] = (
            delta[frame_index] >= cfg.xp_min_jump_pixels
            and hud[frame_a] >= cfg.hud_score_threshold
            and hud[frame_index] >= cfg.hud_score_threshold
            and arrays["quality"][frame_a] >= cfg.xp_min_quality
            and arrays["quality"][frame_index] >= cfg.xp_min_quality
            and arrays["overlay"][frame_a] < cfg.xp_overlay_threshold
            and arrays["overlay"][frame_index] < cfg.xp_overlay_threshold
            and not reappearance[frame_index]
        )
    resets: list[int] = []
    for frame_index in candidates.tolist():
        if resets and frame_index - resets[-1] <= cfg.reset_collapse_frames:
            continue
        # During one level-up animation, decorative gems repeatedly make the
        # bar measurement jump to full and fall back to zero. A later drop is
        # a new level only after normal gameplay XP activity has resumed.
        if resets and not np.any(valid_positive[resets[-1] + 1 : frame_index]):
            continue
        before = pixels[max(0, frame_index - 15) : frame_index]
        after = pixels[frame_index : min(len(pixels), frame_index + 5)]
        if not len(before) or not len(after):
            continue
        if np.max(before) < 0.90 * width:
            continue
        if np.median(after) > 0.80 * width:
            continue
        resets.append(frame_index)
    return resets


def color_hint_from_xp(estimated_base_xp: float, saturated: bool) -> str:
    if saturated or not math.isfinite(estimated_base_xp) or estimated_base_xp <= 0:
        return "unknown"
    # The small tolerance absorbs endpoint quantization and the textured bar cap.
    if estimated_base_xp <= 3.0:
        return "blue"
    if estimated_base_xp <= 11.0:
        return "green"
    return "red"


def find_xp_events(
    metadata: dict[str, float],
    arrays: dict[str, np.ndarray],
    resets: list[int],
    cfg: Config,
) -> list[XPEvent]:
    fps = metadata["fps"]
    bar_width = int(metadata["bar_width"])
    pixels = arrays["pixels"]
    delta = arrays["delta_pixels"]
    progress = arrays["progress"]
    quality = arrays["quality"]
    overlay = arrays["overlay"]
    hud = arrays["hud"]
    health_bar_detected = arrays["health_bar_detected"]
    level_guard = arrays["level_up_transition_guard"]
    reappearance = arrays["xp_bar_reappearance"]
    events: list[XPEvent] = []

    reset_cursor = 0
    current_level = 1
    for frame_b in range(1, len(pixels)):
        while reset_cursor < len(resets) and resets[reset_cursor] < frame_b:
            current_level += 1
            reset_cursor += 1

        if delta[frame_b] < cfg.xp_min_jump_pixels:
            continue
        frame_a = frame_b - 1
        if reappearance[frame_b]:
            continue
        b_saturated = pixels[frame_b] >= bar_width - 2
        saturated_gameplay_fill = (
            b_saturated
            and health_bar_detected[frame_a]
            and health_bar_detected[frame_b]
        )
        if (
            (level_guard[frame_a] or level_guard[frame_b])
            and not health_bar_detected[frame_a]
            and not health_bar_detected[frame_b]
            and not saturated_gameplay_fill
        ):
            continue
        a_valid = (
            hud[frame_a] >= cfg.hud_score_threshold
            and quality[frame_a] >= cfg.xp_min_quality
            and (
                overlay[frame_a] < cfg.xp_overlay_threshold
                or saturated_gameplay_fill
            )
        )
        b_valid = (
            hud[frame_b] >= cfg.hud_score_threshold
            and quality[frame_b] >= cfg.xp_min_quality
            and (
                overlay[frame_b] < cfg.xp_overlay_threshold
                or saturated_gameplay_fill
            )
        )
        if not (a_valid and b_valid):
            continue

        # A one-frame endpoint flicker is not an XP event. A later increase is
        # allowed, since separate pickups can be only a few frames apart.
        next_valid_pixel: int | None = None
        for future in range(frame_b + 1, min(len(pixels), frame_b + 4)):
            if (
                hud[future] >= cfg.hud_score_threshold
                and quality[future] >= cfg.xp_min_quality
                and overlay[future] < cfg.xp_overlay_threshold
                and not reappearance[future]
            ):
                next_valid_pixel = int(pixels[future])
                break
        if (
            next_valid_pixel is not None
            and next_valid_pixel
            < int(pixels[frame_b]) - cfg.xp_persistence_tolerance_pixels
        ):
            continue

        level = current_level
        required = experience_required_for_level(level)
        growth = imelda_growth_multiplier(level, cfg)
        delta_fraction = float(delta[frame_b] / bar_width)
        estimated_base_xp = delta_fraction * required / growth
        hint = color_hint_from_xp(estimated_base_xp, b_saturated)
        events.append(
            XPEvent(
                event_id=len(events) + 1,
                frame_a=frame_a,
                frame_b=frame_b,
                video_time_a=frame_a / fps,
                video_time_b=frame_b / fps,
                progress_a=float(progress[frame_a]),
                progress_b=float(progress[frame_b]),
                delta_pixels=int(delta[frame_b]),
                delta_fraction=delta_fraction,
                quality_a=float(quality[frame_a]),
                quality_b=float(quality[frame_b]),
                inferred_level=level,
                hud_level=level,
                hud_level_ocr_raw=None,
                hud_level_ocr_text="",
                hud_level_ocr_confidence=0.0,
                hud_level_ocr_accepted=False,
                hud_level_ocr_validation="pending_ocr",
                hud_level_source="reset_inference_pending_ocr",
                xp_required=required,
                growth_multiplier=growth,
                estimated_base_xp_gain=estimated_base_xp,
                xp_color_hint=hint,
                bar_saturated=b_saturated,
                local_single_step_pixels=math.nan,
                jump_step_ratio=math.nan,
            )
        )
    return events


def attach_step_estimates(
    events: list[XPEvent],
    bar_width: int,
) -> list[XPEvent]:
    """Estimate the local size of one ordinary XP-bar step without labels.

    The lower tail within each reset-delimited level segment usually contains
    isolated blue pickups. A theoretical one-XP step keeps burst-only segments
    from treating a large merged update as a single gem.
    """
    by_level: dict[int, list[int]] = {}
    for event in events:
        if not event.bar_saturated:
            by_level.setdefault(event.hud_level, []).append(event.delta_pixels)

    baselines: dict[int, float] = {}
    first_event_by_level = {event.hud_level: event for event in events}
    for level, values in by_level.items():
        empirical = float(np.quantile(values, 0.20)) if len(values) >= 5 else math.inf
        theoretical = (
            bar_width
            * first_event_by_level[level].growth_multiplier
            / experience_required_for_level(level)
            * 1.25
        )
        baseline = min(empirical, theoretical * 1.60)
        if not math.isfinite(baseline):
            baseline = theoretical
        baselines[level] = max(1.0, baseline)

    enriched: list[XPEvent] = []
    for event in events:
        baseline = baselines.get(event.hud_level)
        if baseline is None:
            theoretical = (
                bar_width
                * event.growth_multiplier
                / max(1.0, event.xp_required)
                * 1.25
            )
            baseline = max(1.0, theoretical)
        enriched.append(
            replace(
                event,
                local_single_step_pixels=baseline,
                jump_step_ratio=event.delta_pixels / baseline,
            )
        )
    return enriched


def load_template(path: Path, frame_height: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError(f"Expected a four-channel PNG template: {path}")
    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if not len(xs):
        raise RuntimeError(f"Template has no visible pixels: {path}")
    image = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = frame_height / 1440.0
    target_width = max(12, int(round(27 * scale)))
    target_height = max(18, int(round(40 * scale)))
    interpolation = cv2.INTER_AREA if target_width < image.shape[1] else cv2.INTER_CUBIC
    image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return image[:, :, :3], image[:, :, 3]


def load_template_bank(
    template_dir: Path,
    frame_height: int,
    profile: str = "legacy",
) -> GemTemplateBank:
    manifest_path = template_dir / "gem_template_manifest.json"
    if profile == "wiki-multi":
        if not manifest_path.exists():
            raise RuntimeError(f"Missing template manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text())
        color_files = payload.get("colors")
        if not isinstance(color_files, dict):
            raise RuntimeError(f"Template manifest has no colors mapping: {manifest_path}")
    elif profile == "legacy":
        color_files = {
            color: [f"{color}_gem_icon_true.png"]
            for color in ("blue", "green", "red")
        }
    else:
        raise RuntimeError(f"Unknown template profile: {profile}")

    bank: GemTemplateBank = {}
    for color in ("blue", "green", "red"):
        filenames = color_files.get(color)
        if not isinstance(filenames, list) or not filenames:
            raise RuntimeError(f"Template manifest has no references for {color}: {manifest_path}")
        references: list[GemTemplate] = []
        for filename in filenames:
            if not isinstance(filename, str):
                raise RuntimeError(f"Invalid {color} template filename in {manifest_path}")
            references.append(load_template(template_dir / filename, frame_height))
        bank[color] = tuple(references)
    return bank


def local_maxima(score_map: np.ndarray, threshold: float, radius: int) -> list[tuple[int, int, float]]:
    finite = np.nan_to_num(score_map, nan=-1.0, posinf=-1.0, neginf=-1.0)
    size = max(3, 2 * radius + 1)
    dilated = cv2.dilate(finite, np.ones((size, size), dtype=np.uint8))
    mask = (finite >= threshold) & (finite >= dilated - 1e-6)
    ys, xs = np.where(mask)
    return [(int(x), int(y), float(finite[y, x])) for x, y in zip(xs, ys)]


def hue_mask(hsv: np.ndarray, spec: ColorSpec) -> np.ndarray:
    hue = hsv[:, :, 0]
    selected = np.zeros(hue.shape, dtype=bool)
    for low, high in spec.hue_ranges:
        selected |= (hue >= low) & (hue <= high)
    selected &= hsv[:, :, 1] >= 100
    selected &= hsv[:, :, 2] >= 50
    return selected.astype(np.uint8) * 255


def detect_gems_near_player(
    frame_bgr: np.ndarray,
    templates: GemTemplateBank,
    player_anchor: PlayerAnchor,
    cfg: Config,
) -> list[GemDetection]:
    height, width = frame_bgr.shape[:2]
    scale = height / 1440.0
    radius = int(round(cfg.search_radius_1440p * scale))
    player_x = int(round(player_anchor.x))
    player_y = int(round(player_anchor.y))
    crop_x0 = max(0, player_x - radius)
    crop_y0 = max(0, player_y - radius)
    crop_x1 = min(width, player_x + radius)
    crop_y1 = min(height, player_y + radius)
    crop = frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    detections: list[GemDetection] = []

    min_area = 18.0 * scale * scale
    max_area = 5000.0 * scale * scale
    max_component_width = 160.0 * scale
    max_component_height = 140.0 * scale
    pad = max(7, int(round(cfg.component_pad_1440p * scale)))

    for spec in COLORS:
        color_templates = templates[spec.name]
        template_height = max(template[0].shape[0] for template in color_templates)
        template_width = max(template[0].shape[1] for template in color_templates)
        color = hue_mask(hsv, spec)
        color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(color, connectivity=8)
        candidates: list[GemDetection] = []
        for component in range(1, count):
            x = int(stats[component, cv2.CC_STAT_LEFT])
            y = int(stats[component, cv2.CC_STAT_TOP])
            component_width = int(stats[component, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
            area = float(stats[component, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            if component_width > max_component_width or component_height > max_component_height:
                continue
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(crop.shape[1], x + component_width + pad)
            y1 = min(crop.shape[0], y + component_height + pad)
            roi = crop[y0:y1, x0:x1]
            if roi.shape[0] < template_height or roi.shape[1] < template_width:
                continue
            for template_bgr, template_alpha in color_templates:
                current_height, current_width = template_bgr.shape[:2]
                scores = cv2.matchTemplate(
                    roi,
                    template_bgr,
                    cv2.TM_CCORR_NORMED,
                    mask=template_alpha,
                )
                for local_x, local_y, score in local_maxima(
                    scores,
                    cfg.template_threshold,
                    max(2, int(round(current_width * 0.35))),
                ):
                    candidates.append(
                        GemDetection(
                            color=spec.name,
                            cx=crop_x0 + x0 + local_x + current_width / 2.0,
                            cy=crop_y0 + y0 + local_y + current_height / 2.0,
                            score=score,
                        )
                    )

        nms_radius = cfg.nms_radius_1440p * scale
        selected: list[GemDetection] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if all(
                math.hypot(candidate.cx - kept.cx, candidate.cy - kept.cy) > nms_radius
                for kept in selected
            ):
                selected.append(candidate)
        detections.extend(selected)
    return detections


def match_persistent_gems(
    detections_a: list[GemDetection],
    detections_b: list[GemDetection],
    max_distance: float,
) -> tuple[set[int], set[int]]:
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for color in (spec.name for spec in COLORS):
        indexes_a = [index for index, item in enumerate(detections_a) if item.color == color]
        indexes_b = [index for index, item in enumerate(detections_b) if item.color == color]
        if not indexes_a or not indexes_b:
            continue
        impossible = 1e6
        costs = np.full((len(indexes_a), len(indexes_b)), impossible, dtype=float)
        for row, index_a in enumerate(indexes_a):
            a = detections_a[index_a]
            for column, index_b in enumerate(indexes_b):
                b = detections_b[index_b]
                distance = math.hypot(a.cx - b.cx, a.cy - b.cy)
                if distance <= max_distance:
                    costs[row, column] = distance
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if costs[row, column] >= impossible:
                continue
            matched_a.add(indexes_a[row])
            matched_b.add(indexes_b[column])
    return matched_a, matched_b


def detection_distance_to_player(
    detection: GemDetection,
    player_anchor: PlayerAnchor,
) -> float:
    return math.hypot(
        detection.cx - player_anchor.x,
        detection.cy - player_anchor.y,
    )


def trace_detection_backwards(
    end_frame: int,
    end_detection: GemDetection,
    first_frame: int,
    detections_by_frame: dict[int, list[GemDetection]],
    max_link_distance: float,
) -> tuple[tuple[int, GemDetection], ...]:
    """Follow one same-color detection backward using conservative nearest links."""
    reversed_points: list[tuple[int, GemDetection]] = [(end_frame, end_detection)]
    current = end_detection
    for frame_index in range(end_frame - 1, first_frame - 1, -1):
        options = [
            detection
            for detection in detections_by_frame.get(frame_index, [])
            if detection.color == current.color
            and math.hypot(detection.cx - current.cx, detection.cy - current.cy)
            <= max_link_distance
        ]
        if not options:
            break
        current = min(
            options,
            key=lambda detection: math.hypot(
                detection.cx - current.cx,
                detection.cy - current.cy,
            ),
        )
        reversed_points.append((frame_index, current))
    return tuple(reversed(reversed_points))


def find_trajectory_evidence(
    event: XPEvent,
    frame_shape: tuple[int, ...],
    detections_by_frame: dict[int, list[GemDetection]],
    anchors_by_frame: dict[int, PlayerAnchor],
    cfg: Config,
) -> list[TrajectoryEvidence]:
    """Find gems that approach Imelda and vanish just before an XP jump.

    The XP bar can update one or two frames after the sprite is collected.
    Looking only at A=B-1 therefore misses a valid disappearance. A qualifying
    trajectory must persist for several frames, move radially toward the
    player, end within pickup range, and disappear no more than the configured
    HUD-lag allowance before B.
    """
    height = frame_shape[0]
    scale = height / 1440.0
    first_frame = max(0, event.frame_b - cfg.trajectory_history_frames)
    first_disappearance = max(
        first_frame + 1,
        event.frame_b - cfg.trajectory_max_hud_lag_frames,
    )
    max_match_distance = cfg.match_distance_1440p * scale
    max_link_distance = cfg.trajectory_link_distance_1440p * scale
    pickup_radius = cfg.pickup_radius_1440p * scale
    min_approach = cfg.trajectory_min_approach_1440p * scale
    monotonic_tolerance = cfg.trajectory_monotonic_tolerance_1440p * scale
    candidates: list[TrajectoryEvidence] = []

    for disappearance_frame in range(first_disappearance, event.frame_b + 1):
        frame_before = disappearance_frame - 1
        detections_before = detections_by_frame.get(frame_before, [])
        detections_after = detections_by_frame.get(disappearance_frame, [])
        matched_before, _ = match_persistent_gems(
            detections_before,
            detections_after,
            max_match_distance,
        )
        vanished = [
            detection
            for index, detection in enumerate(detections_before)
            if index not in matched_before
            and detection.score >= cfg.template_threshold
            and detection_distance_to_player(
                detection,
                anchors_by_frame[frame_before],
            )
            <= pickup_radius
        ]
        for detection in vanished:
            points = trace_detection_backwards(
                frame_before,
                detection,
                first_frame,
                detections_by_frame,
                max_link_distance,
            )
            if len(points) < cfg.trajectory_min_frames:
                continue
            template_scores = [item.score for _, item in points]
            strong_template_frames = sum(
                score >= cfg.trajectory_min_template_score
                for score in template_scores
            )
            weak_endpoint_frames = 0
            for score in reversed(template_scores):
                if score >= cfg.trajectory_min_template_score:
                    break
                weak_endpoint_frames += 1
            occlusion_accepted = weak_endpoint_frames > 0
            if occlusion_accepted and (
                strong_template_frames < cfg.trajectory_min_strong_history_frames
                or weak_endpoint_frames > cfg.trajectory_max_occluded_endpoint_frames
            ):
                continue
            distances = np.asarray(
                [
                    detection_distance_to_player(item, anchors_by_frame[frame_index])
                    for frame_index, item in points
                ],
                dtype=float,
            )
            approach = float(distances[0] - distances[-1])
            monotonic_fraction = float(
                np.mean(np.diff(distances) <= monotonic_tolerance)
            )
            if approach < min_approach:
                continue
            if monotonic_fraction < cfg.trajectory_min_monotonic_fraction:
                continue
            mean_score = float(np.mean(template_scores))
            approach_component = min(1.0, approach / max(1.0, 2.0 * min_approach))
            evidence_score = (
                0.40 * mean_score
                + 0.35 * monotonic_fraction
                + 0.25 * approach_component
            )
            candidates.append(
                TrajectoryEvidence(
                    color=detection.color,
                    points=points,
                    disappearance_frame=disappearance_frame,
                    start_distance=float(distances[0]),
                    end_distance=float(distances[-1]),
                    approach_pixels=approach,
                    monotonic_fraction=monotonic_fraction,
                    mean_template_score=mean_score,
                    evidence_score=evidence_score,
                    strong_template_frames=strong_template_frames,
                    weak_endpoint_frames=weak_endpoint_frames,
                    occlusion_accepted=occlusion_accepted,
                )
            )

    selected: list[TrajectoryEvidence] = []
    dedup_distance = cfg.trajectory_dedup_distance_1440p * scale
    for candidate in sorted(candidates, key=lambda item: item.evidence_score, reverse=True):
        end_detection = candidate.points[-1][1]
        duplicate = False
        for kept in selected:
            kept_end = kept.points[-1][1]
            if (
                candidate.color == kept.color
                and abs(candidate.disappearance_frame - kept.disappearance_frame) <= 1
                and math.hypot(
                    end_detection.cx - kept_end.cx,
                    end_detection.cy - kept_end.cy,
                )
                <= dedup_distance
            ):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item.disappearance_frame, item.color))


def deduplicate_count_trajectories(
    trajectories: list[TrajectoryEvidence],
    frame_shape: tuple[int, ...],
    cfg: Config,
) -> list[TrajectoryEvidence]:
    """Merge weak count tracks that land on the same physical sprite.

    The count-only pass intentionally lowers the color-template threshold. A
    partially occluded sprite can therefore be proposed by more than one color
    reference. Count evidence must treat those proposals as one pickup even
    though the color is still uncertain.
    """
    scale = frame_shape[0] / 1440.0
    dedup_distance = cfg.count_trajectory_dedup_distance_1440p * scale
    selected: list[TrajectoryEvidence] = []
    for candidate in sorted(
        trajectories,
        key=lambda item: item.evidence_score,
        reverse=True,
    ):
        candidate_end = candidate.points[-1][1]
        duplicate = any(
            abs(candidate.disappearance_frame - kept.disappearance_frame) <= 1
            and math.hypot(
                candidate_end.cx - kept.points[-1][1].cx,
                candidate_end.cy - kept.points[-1][1].cy,
            )
            <= dedup_distance
            for kept in selected
        )
        if not duplicate:
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.disappearance_frame)


def count_trajectory_config(cfg: Config) -> Config:
    """Create a more sensitive trajectory profile used only for pickup count."""
    return replace(
        cfg,
        template_threshold=cfg.count_template_threshold,
        pickup_radius_1440p=cfg.count_pickup_radius_1440p,
        trajectory_min_frames=cfg.count_trajectory_min_frames,
        trajectory_min_approach_1440p=cfg.count_trajectory_min_approach_1440p,
        trajectory_min_monotonic_fraction=cfg.count_trajectory_min_monotonic_fraction,
        trajectory_min_template_score=cfg.count_trajectory_min_template_score,
        trajectory_min_strong_history_frames=(
            cfg.count_trajectory_min_strong_history_frames
        ),
        trajectory_max_occluded_endpoint_frames=(
            cfg.count_trajectory_max_occluded_endpoint_frames
        ),
        trajectory_dedup_distance_1440p=(
            cfg.count_trajectory_dedup_distance_1440p
        ),
    )


def summarize_pair(
    event: XPEvent,
    frame_shape: tuple[int, ...],
    detections_a: list[GemDetection],
    detections_b: list[GemDetection],
    trajectories: list[TrajectoryEvidence],
    count_trajectories: list[TrajectoryEvidence],
    anchor_a: PlayerAnchor,
    anchor_b: PlayerAnchor,
    cfg: Config,
) -> tuple[dict[str, object], list[GemDetection]]:
    height = frame_shape[0]
    scale = height / 1440.0
    matched_a, _ = match_persistent_gems(
        detections_a,
        detections_b,
        cfg.match_distance_1440p * scale,
    )
    pickup_radius = cfg.pickup_radius_1440p * scale
    unmatched_candidates = [
        detection
        for index, detection in enumerate(detections_a)
        if index not in matched_a
        and detection_distance_to_player(detection, anchor_a) <= pickup_radius
    ]
    strong_candidates = [
        detection for detection in unmatched_candidates if detection.score >= cfg.visual_strong_threshold
    ]
    weak_accepted = False
    if strong_candidates:
        disappeared = strong_candidates
    elif (
        len(unmatched_candidates) == 1
        and unmatched_candidates[0].color == "blue"
        and event.jump_step_ratio <= 1.80
        and not event.bar_saturated
    ):
        # A partially occluded gem can score below the strict visual threshold.
        # Accept it only when it is the sole unmatched candidate and the
        # independent XP magnitude agrees with its color.
        disappeared = unmatched_candidates
        weak_accepted = True
    else:
        disappeared = []
    visual_counts = Counter(item.color for item in disappeared)
    visual_total = sum(visual_counts.values())
    trajectory_counts = Counter(item.color for item in trajectories)
    trajectory_total = sum(trajectory_counts.values())
    count_trajectory_scores = [item.evidence_score for item in count_trajectories]
    review_reasons: list[str] = []

    counts = Counter({"blue": 0, "green": 0, "red": 0})
    unresolved = 0
    if trajectory_total:
        for color in counts:
            # The same pickup can appear in both routes. Taking the per-color
            # maximum preserves additional direct disappearances without
            # double-counting a trajectory that ends in A.
            counts[color] = max(trajectory_counts[color], visual_counts[color])
        visible_colors = {color for color, value in counts.items() if value}
        color_conflict = event.xp_color_hint != "unknown" and event.xp_color_hint not in visible_colors
        evidence = "visual_trajectory"
        confidence = min(
            0.98,
            0.78
            + 0.10 * float(np.mean([item.evidence_score for item in trajectories]))
            + 0.03 * sum(counts.values()),
        )
        if sum(counts.values()) > 1:
            review_reasons.append("multiple_trajectory_pickups_in_one_xp_update")
        if color_conflict:
            review_reasons.append("visual_color_differs_from_merged_xp_value_band")
            confidence = min(confidence, 0.72)
    elif visual_total:
        visible_colors = set(visual_counts)
        color_conflict = event.xp_color_hint != "unknown" and event.xp_color_hint not in visible_colors
        counts.update(visual_counts)
        evidence = "visual_disappearance_weak" if weak_accepted else "visual_disappearance"
        confidence = min(
            0.98,
            0.72 + 0.08 * visual_total + 0.12 * np.mean([d.score for d in disappeared]),
        )
        if weak_accepted:
            confidence = min(confidence, 0.68)
            review_reasons.append("weak_visual_candidate_confirmed_by_single_xp_step")
        if visual_total > 1:
            review_reasons.append("multiple_sprites_disappeared_in_one_xp_update")
        if color_conflict:
            # Magnitude is retained as a diagnostic band, but cannot overrule
            # direct color evidence because several lower-value gems can merge.
            review_reasons.append("visual_color_differs_from_merged_xp_value_band")
            confidence = min(confidence, 0.72)
    elif not event.bar_saturated and event.jump_step_ratio <= 1.60:
        counts["blue"] = 1
        evidence = "xp_single_step_blue_fallback"
        confidence = 0.48
        review_reasons.append("blue_sized_xp_step_with_hidden_sprite")
    else:
        unresolved = 1
        evidence = "xp_confirmed_color_ambiguous"
        confidence = 0.20
        if event.bar_saturated:
            review_reasons.append("bar_saturated_and_no_disappearing_sprite")
        else:
            review_reasons.append("merged_or_high_value_xp_jump_without_direct_color")

    total = int(sum(counts.values()) + unresolved)
    if any(detection.score < cfg.template_threshold + 0.04 for detection in disappeared):
        review_reasons.append("weak_template_match")

    row = {
        "event_id": event.event_id,
        "event_key": f"frame_{event.frame_b:06d}",
        "frame_a": event.frame_a,
        "frame_b": event.frame_b,
        "video_time_a": round(event.video_time_a, 4),
        "video_time_b": round(event.video_time_b, 4),
        "video_time_stamp": seconds_to_stamp(event.video_time_b, include_milliseconds=True),
        "xp_progress_a": round(event.progress_a, 6),
        "xp_progress_b": round(event.progress_b, 6),
        "xp_bar_progress_a_percent": round(100.0 * event.progress_a, 4),
        "xp_bar_progress_b_percent": round(100.0 * event.progress_b, 4),
        "xp_delta_pixels": event.delta_pixels,
        "xp_delta_fraction": round(event.delta_fraction, 7),
        "xp_bar_increase_percent": round(100.0 * event.delta_fraction, 4),
        "xp_quality_a": round(event.quality_a, 4),
        "xp_quality_b": round(event.quality_b, 4),
        "hud_level": event.hud_level,
        "hud_level_ocr_raw": event.hud_level_ocr_raw or "",
        "hud_level_ocr_text": event.hud_level_ocr_text,
        "hud_level_ocr_confidence": round(event.hud_level_ocr_confidence, 4),
        "hud_level_ocr_accepted": int(event.hud_level_ocr_accepted),
        "hud_level_ocr_validation": event.hud_level_ocr_validation,
        "hud_level_source": event.hud_level_source,
        "reset_inferred_level": event.inferred_level,
        "level_agrees_with_reset_inference": int(
            event.hud_level == event.inferred_level
        ),
        "inferred_level": event.inferred_level,
        "xp_required_for_level": round(event.xp_required, 3),
        "growth_multiplier": round(event.growth_multiplier, 3),
        "estimated_base_xp_gain": round(event.estimated_base_xp_gain, 3),
        "level_segment_estimate": event.inferred_level,
        "local_single_step_pixels": round(event.local_single_step_pixels, 3),
        "xp_jump_step_ratio": round(event.jump_step_ratio, 3),
        "xp_value_band": event.xp_color_hint,
        "xp_color_hint": event.xp_color_hint,
        "xp_bar_saturated": int(event.bar_saturated),
        "player_anchor_a_x": round(anchor_a.x, 2),
        "player_anchor_a_y": round(anchor_a.y, 2),
        "player_anchor_b_x": round(anchor_b.x, 2),
        "player_anchor_b_y": round(anchor_b.y, 2),
        "health_bar_detected_a": int(anchor_a.health_bar_detected),
        "health_bar_detected_b": int(anchor_b.health_bar_detected),
        "health_bar_confidence_a": round(anchor_a.confidence, 4),
        "health_bar_confidence_b": round(anchor_b.confidence, 4),
        "pickup_radius_pixels": round(pickup_radius, 2),
        "detections_a_blue": sum(d.color == "blue" for d in detections_a),
        "detections_a_green": sum(d.color == "green" for d in detections_a),
        "detections_a_red": sum(d.color == "red" for d in detections_a),
        "detections_b_blue": sum(d.color == "blue" for d in detections_b),
        "detections_b_green": sum(d.color == "green" for d in detections_b),
        "detections_b_red": sum(d.color == "red" for d in detections_b),
        "unmatched_candidate_blue": sum(d.color == "blue" for d in unmatched_candidates),
        "unmatched_candidate_green": sum(d.color == "green" for d in unmatched_candidates),
        "unmatched_candidate_red": sum(d.color == "red" for d in unmatched_candidates),
        "disappeared_blue": visual_counts["blue"],
        "disappeared_green": visual_counts["green"],
        "disappeared_red": visual_counts["red"],
        "trajectory_blue": trajectory_counts["blue"],
        "trajectory_green": trajectory_counts["green"],
        "trajectory_red": trajectory_counts["red"],
        "trajectory_track_count": trajectory_total,
        "trajectory_start_frames": "|".join(str(item.points[0][0]) for item in trajectories),
        "trajectory_end_frames": "|".join(str(item.points[-1][0]) for item in trajectories),
        "trajectory_disappearance_frames": "|".join(
            str(item.disappearance_frame) for item in trajectories
        ),
        "trajectory_start_distances": "|".join(
            f"{item.start_distance:.1f}" for item in trajectories
        ),
        "trajectory_end_distances": "|".join(
            f"{item.end_distance:.1f}" for item in trajectories
        ),
        "trajectory_approach_pixels": "|".join(
            f"{item.approach_pixels:.1f}" for item in trajectories
        ),
        "trajectory_monotonic_fraction": "|".join(
            f"{item.monotonic_fraction:.3f}" for item in trajectories
        ),
        "trajectory_score": "|".join(
            f"{item.evidence_score:.3f}" for item in trajectories
        ),
        "trajectory_strong_template_frames": "|".join(
            str(item.strong_template_frames) for item in trajectories
        ),
        "trajectory_weak_endpoint_frames": "|".join(
            str(item.weak_endpoint_frames) for item in trajectories
        ),
        "trajectory_occlusion_accepted": int(
            any(item.occlusion_accepted for item in trajectories)
        ),
        "count_trajectory_track_count": len(count_trajectories),
        "count_trajectory_max_score": (
            round(max(count_trajectory_scores), 4) if count_trajectory_scores else 0.0
        ),
        "count_trajectory_mean_score": (
            round(float(np.mean(count_trajectory_scores)), 4)
            if count_trajectory_scores
            else 0.0
        ),
        "count_trajectory_colors": "|".join(
            item.color for item in count_trajectories
        ),
        "count_trajectory_start_frames": "|".join(
            str(item.points[0][0]) for item in count_trajectories
        ),
        "count_trajectory_end_frames": "|".join(
            str(item.points[-1][0]) for item in count_trajectories
        ),
        "count_trajectory_disappearance_frames": "|".join(
            str(item.disappearance_frame) for item in count_trajectories
        ),
        "count_trajectory_scores": "|".join(
            f"{item.evidence_score:.3f}" for item in count_trajectories
        ),
        "collected_blue_gems": int(counts["blue"]),
        "collected_green_gems": int(counts["green"]),
        "collected_red_gems": int(counts["red"]),
        "unresolved_collected_gems": unresolved,
        "collected_gems_total": total,
        "color_evidence": evidence,
        "confidence": round(float(confidence), 4),
        "needs_review": int(bool(review_reasons)),
        "review_reason": "|".join(review_reasons),
        "pair_image": "",
        "trajectory_image": "",
        "count_trajectory_image": "",
    }
    return row, disappeared


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[GemDetection],
    disappeared: Iterable[GemDetection],
) -> np.ndarray:
    annotated = frame.copy()
    disappeared_ids = {id(item) for item in disappeared}
    colors = {spec.name: spec.draw_color for spec in COLORS}
    for detection in detections:
        center = (int(round(detection.cx)), int(round(detection.cy)))
        color = colors[detection.color]
        cv2.circle(annotated, center, 25, color, 3, cv2.LINE_AA)
        label = f"{detection.color[0].upper()} {detection.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        if id(detection) in disappeared_ids:
            cv2.line(annotated, (center[0] - 20, center[1] - 20), (center[0] + 20, center[1] + 20), color, 4)
            cv2.line(annotated, (center[0] + 20, center[1] - 20), (center[0] - 20, center[1] + 20), color, 4)
    return annotated


def draw_player_anchor(
    frame: np.ndarray,
    anchor: PlayerAnchor,
    cfg: Config,
) -> np.ndarray:
    annotated = frame.copy()
    scale = frame.shape[0] / 1440.0
    center = (int(round(anchor.x)), int(round(anchor.y)))
    radius = int(round(cfg.pickup_radius_1440p * scale))
    color = (0, 255, 255) if anchor.health_bar_detected else (0, 165, 255)
    cv2.circle(annotated, center, radius, color, 3, cv2.LINE_AA)
    cv2.drawMarker(
        annotated,
        center,
        color,
        cv2.MARKER_CROSS,
        max(18, int(round(28 * scale))),
        3,
        cv2.LINE_AA,
    )
    return annotated


def write_pair_preview(
    path: Path,
    event: XPEvent,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    detections_a: list[GemDetection],
    detections_b: list[GemDetection],
    disappeared: list[GemDetection],
    anchor_a: PlayerAnchor,
    anchor_b: PlayerAnchor,
    cfg: Config,
) -> None:
    annotated_a = draw_player_anchor(
        draw_detections(frame_a, detections_a, disappeared),
        anchor_a,
        cfg,
    )
    annotated_b = draw_player_anchor(
        draw_detections(frame_b, detections_b, []),
        anchor_b,
        cfg,
    )
    height, width = frame_a.shape[:2]
    radius_x = int(round(380 * height / 1440.0))
    radius_y = int(round(280 * height / 1440.0))
    crops = []
    for frame, anchor in ((annotated_a, anchor_a), (annotated_b, anchor_b)):
        center_x = int(round(anchor.x))
        center_y = int(round(anchor.y))
        crops.append(
            frame[
                max(0, center_y - radius_y) : min(height, center_y + radius_y),
                max(0, center_x - radius_x) : min(width, center_x + radius_x),
            ]
        )
    top_height = max(45, int(round(height * 0.05)))
    tops = [frame[:top_height] for frame in (frame_a, frame_b)]
    crop_width = 760
    crop_height = 560
    resized_crops = [cv2.resize(crop, (crop_width, crop_height), interpolation=cv2.INTER_AREA) for crop in crops]
    resized_tops = [cv2.resize(top, (crop_width, 48), interpolation=cv2.INTER_AREA) for top in tops]
    canvas = np.zeros((660, crop_width * 2, 3), dtype=np.uint8)
    canvas[52:100, :crop_width] = resized_tops[0]
    canvas[52:100, crop_width:] = resized_tops[1]
    canvas[100:, :crop_width] = resized_crops[0]
    canvas[100:, crop_width:] = resized_crops[1]
    labels = (
        f"A frame {event.frame_a}  {seconds_to_stamp(event.video_time_a, True)}",
        f"B frame {event.frame_b}  {seconds_to_stamp(event.video_time_b, True)}  "
        f"LV {event.hud_level}  XP +{100.0 * event.delta_fraction:.3f}%",
    )
    for index, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (18 + index * crop_width, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, cfg.preview_jpeg_quality])


def write_trajectory_preview(
    path: Path,
    event: XPEvent,
    frames_by_index: dict[int, np.ndarray],
    detections_by_frame: dict[int, list[GemDetection]],
    anchors_by_frame: dict[int, PlayerAnchor],
    trajectories: list[TrajectoryEvidence],
    fps: float,
    cfg: Config,
) -> None:
    """Write a compact contact sheet that makes temporal evidence auditable."""
    if not trajectories:
        return
    first = max(0, event.frame_b - cfg.trajectory_history_frames)
    last = event.frame_b + cfg.trajectory_post_frames
    indexes = [index for index in range(first, last + 1) if index in frames_by_index]
    if not indexes:
        return

    sample = frames_by_index[indexes[0]]
    height, width = sample.shape[:2]
    scale = height / 1440.0
    radius_x = int(round(380 * scale))
    radius_y = int(round(280 * scale))
    columns = 5
    rows = int(math.ceil(len(indexes) / columns))
    tile_width = 304
    tile_image_height = 224
    tile_height = 258
    title_height = 42
    canvas = np.zeros((title_height + rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    colors = {spec.name: spec.draw_color for spec in COLORS}

    title = (
        f"Event {event.event_id}  XP frame {event.frame_b}  "
        f"{seconds_to_stamp(event.video_time_b, True)}  LV {event.hud_level}  "
        f"+{100.0 * event.delta_fraction:.3f}%"
    )
    cv2.putText(
        canvas,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for tile_index, frame_index in enumerate(indexes):
        annotated = draw_detections(
            frames_by_index[frame_index],
            detections_by_frame.get(frame_index, []),
            [],
        )
        anchor = anchors_by_frame[frame_index]
        annotated = draw_player_anchor(annotated, anchor, cfg)
        for trajectory in trajectories:
            path_points = [
                (int(round(item.cx)), int(round(item.cy)))
                for point_frame, item in trajectory.points
                if point_frame <= frame_index
            ]
            if len(path_points) >= 2:
                cv2.polylines(
                    annotated,
                    [np.asarray(path_points, dtype=np.int32)],
                    False,
                    colors[trajectory.color],
                    4,
                    cv2.LINE_AA,
                )
            for point_frame, item in trajectory.points:
                if point_frame == frame_index:
                    cv2.circle(
                        annotated,
                        (int(round(item.cx)), int(round(item.cy))),
                        34,
                        colors[trajectory.color],
                        5,
                        cv2.LINE_AA,
                    )

        center_x = int(round(anchor.x))
        center_y = int(round(anchor.y))
        crop = annotated[
            max(0, center_y - radius_y) : min(height, center_y + radius_y),
            max(0, center_x - radius_x) : min(width, center_x + radius_x),
        ]
        resized = cv2.resize(crop, (tile_width, tile_image_height), interpolation=cv2.INTER_AREA)
        row = tile_index // columns
        column = tile_index % columns
        x0 = column * tile_width
        y0 = title_height + row * tile_height
        canvas[y0 + 34 : y0 + 34 + tile_image_height, x0 : x0 + tile_width] = resized
        suffix = "  XP" if frame_index == event.frame_b else ""
        label = f"{frame_index}  {frame_index / fps:.3f}s{suffix}"
        cv2.putText(
            canvas,
            label,
            (x0 + 8, y0 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, cfg.preview_jpeg_quality])


def analyze_flagged_windows(
    video_path: Path,
    events: list[XPEvent],
    templates: GemTemplateBank,
    signal_arrays: dict[str, np.ndarray],
    output_dir: Path,
    cfg: Config,
    save_previews: bool,
) -> list[dict[str, object]]:
    if not events:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count_cfg = count_trajectory_config(cfg)

    def window_indexes(event: XPEvent) -> range:
        start = max(0, event.frame_b - cfg.trajectory_history_frames)
        end = min(frame_count - 1, event.frame_b + cfg.trajectory_post_frames)
        return range(start, end + 1)

    events_by_end: dict[int, list[XPEvent]] = {}
    for event in events:
        events_by_end.setdefault(window_indexes(event).stop - 1, []).append(event)
    use_counts = Counter(index for event in events for index in window_indexes(event))
    needed = set(use_counts)
    frames: dict[int, np.ndarray] = {}
    detection_cache: dict[int, list[GemDetection]] = {}
    count_detection_cache: dict[int, list[GemDetection]] = {}
    anchor_cache: dict[int, PlayerAnchor] = {}
    claimed_trajectories: list[TrajectoryEvidence] = []
    claimed_count_trajectories: list[TrajectoryEvidence] = []
    rows: list[dict[str, object]] = []
    next_report = 100

    for frame_index in range(frame_count):
        ok = cap.grab()
        if not ok:
            break
        if frame_index not in needed:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        frames[frame_index] = frame
        anchor = PlayerAnchor(
            x=float(signal_arrays["player_anchor_x"][frame_index]),
            y=float(signal_arrays["player_anchor_y"][frame_index]),
            health_bar_detected=bool(
                signal_arrays["health_bar_detected"][frame_index]
            ),
            health_bar_x=float(signal_arrays["health_bar_x"][frame_index]),
            health_bar_y=float(signal_arrays["health_bar_y"][frame_index]),
            confidence=float(signal_arrays["health_bar_confidence"][frame_index]),
        )
        anchor_cache[frame_index] = anchor
        detection_cache[frame_index] = detect_gems_near_player(
            frame,
            templates,
            anchor,
            cfg,
        )
        count_detection_cache[frame_index] = detect_gems_near_player(
            frame,
            templates,
            anchor,
            count_cfg,
        )

        ending_events = events_by_end.get(frame_index, [])
        if not ending_events:
            continue
        for event in ending_events:
            if event.frame_a not in frames or event.frame_b not in frames:
                continue
            frame_a = frames[event.frame_a]
            frame_b = frames[event.frame_b]
            detections_a = detection_cache[event.frame_a]
            detections_b = detection_cache[event.frame_b]
            trajectory_candidates = find_trajectory_evidence(
                event,
                frame_b.shape,
                detection_cache,
                anchor_cache,
                cfg,
            )
            scale = frame_b.shape[0] / 1440.0
            claim_distance = cfg.trajectory_dedup_distance_1440p * scale
            trajectories = []
            for candidate in trajectory_candidates:
                candidate_end = candidate.points[-1][1]
                already_claimed = any(
                    candidate.color == claimed.color
                    and candidate.disappearance_frame == claimed.disappearance_frame
                    and math.hypot(
                        candidate_end.cx - claimed.points[-1][1].cx,
                        candidate_end.cy - claimed.points[-1][1].cy,
                    )
                    <= claim_distance
                    for claimed in claimed_trajectories
                )
                if not already_claimed:
                    trajectories.append(candidate)
            claimed_trajectories.extend(trajectories)
            count_candidates = deduplicate_count_trajectories(
                find_trajectory_evidence(
                    event,
                    frame_b.shape,
                    count_detection_cache,
                    anchor_cache,
                    count_cfg,
                ),
                frame_b.shape,
                cfg,
            )
            count_claim_distance = (
                cfg.count_trajectory_dedup_distance_1440p * scale
            )
            count_trajectories: list[TrajectoryEvidence] = []
            for candidate in count_candidates:
                candidate_end = candidate.points[-1][1]
                already_claimed = any(
                    abs(
                        candidate.disappearance_frame
                        - claimed.disappearance_frame
                    )
                    <= 1
                    and math.hypot(
                        candidate_end.cx - claimed.points[-1][1].cx,
                        candidate_end.cy - claimed.points[-1][1].cy,
                    )
                    <= count_claim_distance
                    for claimed in claimed_count_trajectories
                )
                if not already_claimed:
                    count_trajectories.append(candidate)
            claimed_count_trajectories.extend(count_trajectories)
            row, disappeared = summarize_pair(
                event,
                frame_b.shape,
                detections_a,
                detections_b,
                trajectories,
                count_trajectories,
                anchor_cache[event.frame_a],
                anchor_cache[event.frame_b],
                cfg,
            )
            if save_previews:
                filename = f"event_{event.event_id:05d}_frames_{event.frame_a}_{event.frame_b}.jpg"
                preview_path = output_dir / "ab_pairs" / filename
                write_pair_preview(
                    preview_path,
                    event,
                    frame_a,
                    frame_b,
                    detections_a,
                    detections_b,
                    disappeared,
                    anchor_cache[event.frame_a],
                    anchor_cache[event.frame_b],
                    cfg,
                )
                row["pair_image"] = str(Path("ab_pairs") / filename)
                if trajectories:
                    first = max(0, event.frame_b - cfg.trajectory_history_frames)
                    last = min(frame_count - 1, event.frame_b + cfg.trajectory_post_frames)
                    trajectory_filename = (
                        f"event_{event.event_id:05d}_frames_{first}_{last}.jpg"
                    )
                    trajectory_path = output_dir / "trajectory_windows" / trajectory_filename
                    write_trajectory_preview(
                        trajectory_path,
                        event,
                        frames,
                        detection_cache,
                        anchor_cache,
                        trajectories,
                        fps,
                        cfg,
                    )
                    row["trajectory_image"] = str(
                        Path("trajectory_windows") / trajectory_filename
                    )
                if count_trajectories:
                    first = max(0, event.frame_b - cfg.trajectory_history_frames)
                    last = min(
                        frame_count - 1,
                        event.frame_b + cfg.trajectory_post_frames,
                    )
                    count_filename = (
                        f"event_{event.event_id:05d}_frames_{first}_{last}.jpg"
                    )
                    count_path = (
                        output_dir / "count_trajectory_windows" / count_filename
                    )
                    write_trajectory_preview(
                        count_path,
                        event,
                        frames,
                        count_detection_cache,
                        anchor_cache,
                        count_trajectories,
                        fps,
                        count_cfg,
                    )
                    row["count_trajectory_image"] = str(
                        Path("count_trajectory_windows") / count_filename
                    )
            rows.append(row)

            for used_index in window_indexes(event):
                use_counts[used_index] -= 1
                if use_counts[used_index] <= 0:
                    frames.pop(used_index, None)
                    detection_cache.pop(used_index, None)
                    count_detection_cache.pop(used_index, None)
                    anchor_cache.pop(used_index, None)
        if len(rows) >= next_report:
            print(f"Temporal analysis: {len(rows)}/{len(events)} events", flush=True)
            next_report += 100
        if len(rows) == len(events):
            break
    cap.release()
    if len(rows) != len(events):
        raise RuntimeError(f"Only analyzed {len(rows)} of {len(events)} flagged windows")
    return rows


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return float((center - margin) / denominator)


def strict_visual_single_color(row: dict[str, object]) -> str | None:
    """Return a high-confidence video-derived color label for validation."""
    if int(row["xp_bar_saturated"]):
        return None
    if not str(row["color_evidence"]).startswith("visual_"):
        return None
    if int(row["unresolved_collected_gems"]) or int(row["collected_gems_total"]) != 1:
        return None
    if int(row["trajectory_track_count"]) != 1:
        return None
    if not int(row["health_bar_detected_a"]) or not int(row["health_bar_detected_b"]):
        return None
    if not int(row["hud_level_ocr_accepted"]):
        return None
    if float(row["confidence"]) < 0.75:
        return None
    colors = [
        color
        for color in ("blue", "green", "red")
        if int(row[f"collected_{color}_gems"]) == 1
    ]
    return colors[0] if len(colors) == 1 else None


def strict_visual_pickup_count(row: dict[str, object]) -> int | None:
    """Return a strong visual pickup count used to validate the weaker pass."""
    if int(row["xp_bar_saturated"]):
        return None
    if not str(row["color_evidence"]).startswith("visual_"):
        return None
    total = int(row["collected_gems_total"])
    if total < 1 or int(row["unresolved_collected_gems"]):
        return None
    if int(row["trajectory_track_count"]) != total:
        return None
    if not int(row["health_bar_detected_a"]) or not int(row["health_bar_detected_b"]):
        return None
    return total


def split_name(row: dict[str, object], cfg: Config) -> str:
    video_time = float(row["video_time_b"])
    if video_time < cfg.percentage_calibration_end_seconds:
        return "calibration"
    if video_time < cfg.percentage_validation_end_seconds:
        return "validation"
    return "deployment"


def add_event_isolation(rows: list[dict[str, object]], cfg: Config) -> None:
    ordered = sorted(rows, key=lambda row: int(row["frame_b"]))
    for index, row in enumerate(ordered):
        frame = int(row["frame_b"])
        previous_gap = (
            frame - int(ordered[index - 1]["frame_b"])
            if index > 0
            else 1_000_000
        )
        next_gap = (
            int(ordered[index + 1]["frame_b"]) - frame
            if index + 1 < len(ordered)
            else 1_000_000
        )
        row["previous_xp_event_gap_frames"] = previous_gap
        row["next_xp_event_gap_frames"] = next_gap
        row["xp_event_temporally_isolated"] = int(
            previous_gap > cfg.percentage_isolation_frames
            and next_gap > cfg.percentage_isolation_frames
        )


def metric_record(
    task: str,
    split: str,
    truth: list[bool],
    predicted: list[bool],
    threshold: float | str,
) -> dict[str, object]:
    true_positive = sum(actual and guess for actual, guess in zip(truth, predicted))
    false_positive = sum(not actual and guess for actual, guess in zip(truth, predicted))
    false_negative = sum(actual and not guess for actual, guess in zip(truth, predicted))
    predicted_count = true_positive + false_positive
    actual_count = true_positive + false_negative
    precision = safe_rate(true_positive, predicted_count)
    recall = safe_rate(true_positive, actual_count)
    return {
        "task": task,
        "split": split,
        "reference_events": len(truth),
        "actual_positive_events": actual_count,
        "predicted_positive_events": predicted_count,
        "true_positive_events": true_positive,
        "false_positive_events": false_positive,
        "false_negative_events": false_negative,
        "precision": round(precision, 4),
        "precision_wilson_95_lower": round(
            wilson_lower_bound(true_positive, predicted_count),
            4,
        ),
        "recall": round(recall, 4),
        "threshold": threshold,
    }


def select_count_score_threshold(
    calibration_rows: list[dict[str, object]],
    cfg: Config,
) -> tuple[float, dict[str, object]]:
    references = [
        (row, strict_visual_pickup_count(row))
        for row in calibration_rows
    ]
    references = [(row, count) for row, count in references if count is not None]
    best: tuple[int, float, dict[str, object]] | None = None
    fallback: tuple[float, int, float, dict[str, object]] | None = None
    for threshold in np.arange(0.60, 0.961, 0.01):
        truth = [count == 1 for _, count in references]
        predicted = [
            int(row["count_trajectory_track_count"]) == 1
            and float(row["count_trajectory_max_score"]) >= float(threshold)
            for row, _ in references
        ]
        metric = metric_record(
            "single_pickup_count",
            "calibration",
            truth,
            predicted,
            round(float(threshold), 2),
        )
        predicted_count = int(metric["predicted_positive_events"])
        precision = float(metric["precision"])
        candidate = (predicted_count, -float(threshold), metric)
        if (
            predicted_count >= cfg.percentage_min_validation_predictions
            and precision >= cfg.percentage_target_precision
            and (best is None or candidate[:2] > best[:2])
        ):
            best = candidate
        fallback_candidate = (
            precision,
            predicted_count,
            -float(threshold),
            metric,
        )
        if fallback is None or fallback_candidate[:3] > fallback[:3]:
            fallback = fallback_candidate

    if best is not None:
        metric = best[2]
        return float(metric["threshold"]), metric
    if fallback is None:
        metric = metric_record(
            "single_pickup_count",
            "calibration",
            [],
            [],
            0.96,
        )
        return 0.96, metric
    metric = fallback[3]
    return float(metric["threshold"]), metric


def xp_color_from_calibrated_gain(gain: float, cfg: Config) -> str:
    if not math.isfinite(gain) or gain <= 0.0:
        return "unknown"
    if gain <= cfg.percentage_blue_max_base_xp:
        return "blue"
    if gain <= cfg.percentage_green_max_base_xp:
        return "green"
    return "red"


def isolated_visual_blue_growth_ratio(
    row: dict[str, object],
    reference_factor: float = 1.0,
) -> float | None:
    """Estimate unmodelled Growth from a certain, isolated blue pickup."""
    if strict_visual_single_color(row) != "blue":
        return None
    if not int(row["xp_event_temporally_isolated"]):
        return None
    raw_gain = float(row["estimated_base_xp_gain"])
    normalized_gain = raw_gain / max(reference_factor, 1e-6)
    expected_gain = 1.0 if normalized_gain < 1.5 else 2.0
    ratio = raw_gain / expected_gain
    return ratio if 0.65 <= ratio <= 1.40 else None


def apply_video_only_percentage_calibration(
    rows: list[dict[str, object]],
    cfg: Config,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Calibrate XP percentage on clear video events and gate new assignments.

    The calibration and validation labels come only from strong visual tracks in
    this run. The Gaming Content spreadsheet is neither read nor required.
    """
    baseline_totals = {
        color: sum(int(row[f"collected_{color}_gems"]) for row in rows)
        for color in ("blue", "green", "red")
    }
    baseline_unresolved = sum(int(row["unresolved_collected_gems"]) for row in rows)
    for row in rows:
        for name in (
            "collected_blue_gems",
            "collected_green_gems",
            "collected_red_gems",
            "unresolved_collected_gems",
            "collected_gems_total",
            "color_evidence",
            "confidence",
        ):
            row[f"baseline_{name}"] = row[name]
        row["percentage_calibration_split"] = split_name(row, cfg)
        row["percentage_assist_applied"] = 0
        row["percentage_assist_color"] = ""
        row["percentage_assist_gate_reason"] = ""
    add_event_isolation(rows, cfg)

    calibration_rows = [
        row for row in rows if row["percentage_calibration_split"] == "calibration"
    ]
    validation_rows = [
        row for row in rows if row["percentage_calibration_split"] == "validation"
    ]
    deployment_rows = [
        row for row in rows if row["percentage_calibration_split"] == "deployment"
    ]

    correction_ratios = [
        ratio
        for row in calibration_rows
        if (ratio := isolated_visual_blue_growth_ratio(row)) is not None
    ]
    correction_factor = (
        float(np.median(correction_ratios))
        if len(correction_ratios) >= 10
        else 1.0
    )
    correction_mad = (
        float(
            np.median(
                np.abs(np.asarray(correction_ratios) - correction_factor)
            )
        )
        if correction_ratios
        else 0.0
    )

    count_threshold, count_calibration_metric = select_count_score_threshold(
        calibration_rows,
        cfg,
    )
    metric_rows: list[dict[str, object]] = [count_calibration_metric]

    rolling_ratios: list[float] = []
    deployment_factors: list[float] = []
    growth_updates = Counter()
    for row in sorted(rows, key=lambda item: int(item["frame_b"])):
        split = str(row["percentage_calibration_split"])
        recent_ratios = rolling_ratios[-cfg.percentage_growth_rolling_samples :]
        adaptive_ready = (
            split == "deployment"
            and len(recent_ratios) >= cfg.percentage_growth_min_rolling_samples
        )
        row_correction_factor = (
            float(np.median(recent_ratios))
            if adaptive_ready
            else correction_factor
        )
        raw_gain = float(row["estimated_base_xp_gain"])
        calibrated_gain = raw_gain / row_correction_factor
        candidate_color = (
            "unknown"
            if int(row["xp_bar_saturated"])
            else xp_color_from_calibrated_gain(calibrated_gain, cfg)
        )
        single_supported = (
            int(row["count_trajectory_track_count"]) == 1
            and float(row["count_trajectory_max_score"]) >= count_threshold
        )
        trajectory_color_supported = (
            candidate_color != "unknown"
            and str(row["count_trajectory_colors"]) == candidate_color
        )
        same_color_candidates = (
            int(row[f"unmatched_candidate_{candidate_color}"])
            if candidate_color in ("blue", "green", "red")
            else 0
        )
        multiplicity_supported = same_color_candidates <= 1
        row["xp_growth_correction_factor"] = round(row_correction_factor, 5)
        row["xp_growth_correction_source"] = (
            "rolling_past_visual_blue"
            if adaptive_ready
            else (
                "fixed_calibration_frozen_validation"
                if split == "validation"
                else (
                    "fixed_calibration_fallback"
                    if split == "deployment"
                    else "fixed_calibration"
                )
            )
        )
        row["xp_growth_history_samples"] = len(rolling_ratios)
        row["xp_growth_recent_samples"] = len(recent_ratios)
        row["level_normalized_xp_gain"] = round(calibrated_gain, 4)
        row["percentage_color_candidate"] = candidate_color
        row["count_single_pickup_supported"] = int(single_supported)
        row["percentage_trajectory_color_supported"] = int(
            trajectory_color_supported
        )
        row["percentage_same_color_candidates"] = same_color_candidates
        row["percentage_multiplicity_supported"] = int(multiplicity_supported)
        row["percentage_assist_candidate"] = int(
            single_supported
            and trajectory_color_supported
            and multiplicity_supported
            and int(row["xp_event_temporally_isolated"])
            and not int(row["xp_bar_saturated"])
            and int(row["hud_level_ocr_accepted"])
            and candidate_color != "unknown"
        )
        if split == "deployment":
            deployment_factors.append(row_correction_factor)
        update_ratio = isolated_visual_blue_growth_ratio(
            row,
            reference_factor=row_correction_factor,
        )
        if update_ratio is not None:
            rolling_ratios.append(update_ratio)
            growth_updates[split] += 1

    def count_metric(split: str, split_rows: list[dict[str, object]]) -> dict[str, object]:
        references = [
            (row, strict_visual_pickup_count(row))
            for row in split_rows
        ]
        references = [(row, count) for row, count in references if count is not None]
        return metric_record(
            "single_pickup_count",
            split,
            [count == 1 for _, count in references],
            [
                int(row["count_trajectory_track_count"]) == 1
                and float(row["count_trajectory_max_score"]) >= count_threshold
                for row, _ in references
            ],
            round(count_threshold, 2),
        )

    count_validation_metric = count_metric("validation", validation_rows)
    metric_rows.append(count_validation_metric)
    count_deployment_metric = count_metric("deployment", deployment_rows)
    if deployment_rows:
        metric_rows.append(count_deployment_metric)
    count_gate_enabled = (
        int(count_calibration_metric["predicted_positive_events"])
        >= cfg.percentage_min_validation_predictions
        and float(count_calibration_metric["precision"])
        >= cfg.percentage_target_precision
        and int(count_validation_metric["predicted_positive_events"])
        >= cfg.percentage_min_validation_predictions
        and float(count_validation_metric["precision"])
        >= cfg.percentage_target_precision
    )

    def color_metric(
        split: str,
        split_rows: list[dict[str, object]],
        color: str,
        require_assist_conditions: bool,
    ) -> dict[str, object]:
        references = [
            (row, strict_visual_pickup_count(row))
            for row in split_rows
        ]
        references = [(row, count) for row, count in references if count is not None]
        predicted = []
        for row, _ in references:
            guess = row["percentage_color_candidate"] == color
            if require_assist_conditions:
                guess = guess and bool(int(row["percentage_assist_candidate"]))
            predicted.append(guess)
        return metric_record(
            f"{'assisted_' if require_assist_conditions else ''}color_{color}",
            split,
            [
                count == 1 and int(row[f"collected_{color}_gems"]) == 1
                for row, count in references
            ],
            predicted,
            (
                f"<={cfg.percentage_blue_max_base_xp:.2f} XP"
                if color == "blue"
                else (
                    f"<={cfg.percentage_green_max_base_xp:.2f} XP"
                    if color == "green"
                    else f">{cfg.percentage_green_max_base_xp:.2f} XP"
                )
            ),
        )

    color_gates: dict[str, bool] = {}
    color_metrics: dict[str, dict[str, object]] = {}
    for color in ("blue", "green", "red"):
        calibration_metric = color_metric(
            "calibration",
            calibration_rows,
            color,
            require_assist_conditions=True,
        )
        validation_metric = color_metric(
            "validation",
            validation_rows,
            color,
            require_assist_conditions=True,
        )
        deployment_metric = color_metric(
            "deployment",
            deployment_rows,
            color,
            require_assist_conditions=True,
        )
        metric_rows.extend([calibration_metric, validation_metric])
        if deployment_rows:
            metric_rows.append(deployment_metric)
        enabled = (
            int(calibration_metric["predicted_positive_events"])
            >= cfg.percentage_min_validation_predictions
            and float(calibration_metric["precision"])
            >= cfg.percentage_target_precision
            and int(validation_metric["predicted_positive_events"])
            >= cfg.percentage_min_validation_predictions
            and float(validation_metric["precision"])
            >= cfg.percentage_target_precision
        )
        color_gates[color] = enabled
        color_metrics[color] = {
            "calibration": calibration_metric,
            "validation": validation_metric,
            "deployment": deployment_metric,
            "enabled": enabled,
        }

    deployment_audit_passed = {
        color: bool(
            color_gates[color]
            and int(color_metrics[color]["deployment"]["predicted_positive_events"])
            >= cfg.percentage_min_validation_predictions
            and float(color_metrics[color]["deployment"]["precision"])
            >= cfg.percentage_target_precision
        )
        for color in ("blue", "green", "red")
    }

    reclassified = Counter()
    for row in rows:
        if not int(row["baseline_unresolved_collected_gems"]):
            row["percentage_assist_gate_reason"] = "baseline_already_classified"
            continue
        if not int(row["percentage_assist_candidate"]):
            reasons = []
            if int(row["count_trajectory_track_count"]) != 1:
                reasons.append("pickup_count_not_exactly_one")
            elif float(row["count_trajectory_max_score"]) < count_threshold:
                reasons.append("pickup_count_score_below_gate")
            if not int(row["xp_event_temporally_isolated"]):
                reasons.append("adjacent_xp_event")
            if not int(row["percentage_trajectory_color_supported"]):
                reasons.append("trajectory_color_disagrees_with_xp")
            if not int(row["percentage_multiplicity_supported"]):
                reasons.append("multiple_same_color_pickup_candidates")
            if int(row["xp_bar_saturated"]):
                reasons.append("xp_bar_saturated")
            if not int(row["hud_level_ocr_accepted"]):
                reasons.append("hud_level_not_directly_confirmed")
            row["percentage_assist_gate_reason"] = "|".join(reasons)
            continue
        color = str(row["percentage_color_candidate"])
        if not color_gates.get(color, False):
            row["percentage_assist_gate_reason"] = (
                f"heldout_{color}_precision_or_support_gate_failed"
            )
            continue
        if (
            row["percentage_calibration_split"] == "deployment"
            and not deployment_audit_passed.get(color, False)
        ):
            row["percentage_assist_gate_reason"] = (
                f"deployment_{color}_precision_or_support_gate_failed"
            )
            continue
        row[f"collected_{color}_gems"] = int(row[f"collected_{color}_gems"]) + 1
        row["unresolved_collected_gems"] = int(row["unresolved_collected_gems"]) - 1
        row["color_evidence"] = "video_only_level_normalized_xp"
        validation_precision = float(
            color_metrics[color]["validation"]["precision"]
        )
        count_precision = float(count_validation_metric["precision"])
        row["confidence"] = round(min(validation_precision, count_precision), 4)
        row["percentage_assist_applied"] = 1
        row["percentage_assist_color"] = color
        row["percentage_assist_gate_reason"] = "passed_heldout_precision_gate"
        prior_reason = str(row["review_reason"])
        added_reason = "video_only_percentage_assisted_assignment"
        row["review_reason"] = (
            f"{prior_reason}|{added_reason}" if prior_reason else added_reason
        )
        row["needs_review"] = 1
        reclassified[color] += 1

    final_totals = {
        color: sum(int(row[f"collected_{color}_gems"]) for row in rows)
        for color in ("blue", "green", "red")
    }
    final_unresolved = sum(int(row["unresolved_collected_gems"]) for row in rows)
    visual_reference_counts = {
        split: Counter(
            label
            for row in split_rows
            if (label := strict_visual_single_color(row)) is not None
        )
        for split, split_rows in (
            ("calibration", calibration_rows),
            ("validation", validation_rows),
            ("deployment", deployment_rows),
        )
    }
    report = {
        "method": "video_only_level_normalized_xp_percentage_assist",
        "human_labels_used": False,
        "gaming_content_sheet_read": False,
        "calibration_interval_seconds": [0.0, cfg.percentage_calibration_end_seconds],
        "validation_interval_seconds": [
            cfg.percentage_calibration_end_seconds,
            cfg.percentage_validation_end_seconds,
        ],
        "target_empirical_precision": cfg.percentage_target_precision,
        "minimum_validation_predictions": cfg.percentage_min_validation_predictions,
        "validation_includes_strong_visual_multi_pickup_events": True,
        "visual_reference_counts": {
            split: dict(counts) for split, counts in visual_reference_counts.items()
        },
        "growth_correction": {
            "factor": round(correction_factor, 5),
            "calibration_blue_samples": len(correction_ratios),
            "median_absolute_deviation": round(correction_mad, 5),
            "deployment_adaptation": {
                "enabled": True,
                "past_only": True,
                "rolling_sample_limit": cfg.percentage_growth_rolling_samples,
                "minimum_samples": cfg.percentage_growth_min_rolling_samples,
                "visual_blue_updates_by_split": dict(growth_updates),
                "deployment_factor_min": (
                    round(min(deployment_factors), 5)
                    if deployment_factors
                    else None
                ),
                "deployment_factor_max": (
                    round(max(deployment_factors), 5)
                    if deployment_factors
                    else None
                ),
                "deployment_factor_final": (
                    round(deployment_factors[-1], 5)
                    if deployment_factors
                    else None
                ),
            },
        },
        "single_pickup_count_gate": {
            "score_threshold": round(count_threshold, 2),
            "calibration": count_calibration_metric,
            "validation": count_validation_metric,
            "deployment": count_deployment_metric,
            "enabled": count_gate_enabled,
        },
        "color_gates": color_metrics,
        "deployment_audit_passed": deployment_audit_passed,
        "baseline": {
            **{f"collected_{color}_gems": value for color, value in baseline_totals.items()},
            "unresolved_collected_gems": baseline_unresolved,
        },
        "percentage_assisted": {
            **{f"{color}_gems": int(reclassified[color]) for color in ("blue", "green", "red")},
            "events": int(sum(reclassified.values())),
        },
        "final": {
            **{f"collected_{color}_gems": value for color, value in final_totals.items()},
            "unresolved_collected_gems": final_unresolved,
        },
        "full_video_run_recommended": any(color_gates.values()),
        "production_output_recommended": bool(deployment_rows)
        and any(deployment_audit_passed.values()),
    }
    return report, metric_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate_rows(
    event_rows: list[dict[str, object]],
    duration: float,
    bin_seconds: int,
) -> list[dict[str, object]]:
    bin_count = int(math.ceil(duration / bin_seconds))
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in event_rows:
        index = int(float(row["video_time_b"]) // bin_seconds)
        grouped.setdefault(index, []).append(row)
    result: list[dict[str, object]] = []
    for index in range(bin_count):
        start = index * bin_seconds
        end = min(int(math.ceil(duration)), start + bin_seconds)
        rows = grouped.get(index, [])
        levels = [int(row["hud_level"]) for row in rows]
        increases = [float(row["xp_bar_increase_percent"]) for row in rows]
        result.append(
            {
                "time_stamp": interval_stamp(start, end),
                "interval_start_second": start,
                "interval_end_second": end,
                "xp_jump_events": len(rows),
                "hud_level_min": min(levels) if levels else "",
                "hud_level_max": max(levels) if levels else "",
                "xp_bar_increase_percent_sum": (
                    round(sum(increases), 4) if increases else 0.0
                ),
                "xp_bar_increase_percent_mean": (
                    round(float(np.mean(increases)), 4) if increases else 0.0
                ),
                "xp_bar_increase_percent_max": (
                    round(max(increases), 4) if increases else 0.0
                ),
                "collected_blue_gems": sum(int(row["collected_blue_gems"]) for row in rows),
                "collected_green_gems": sum(int(row["collected_green_gems"]) for row in rows),
                "collected_red_gems": sum(int(row["collected_red_gems"]) for row in rows),
                "unresolved_collected_gems": sum(int(row["unresolved_collected_gems"]) for row in rows),
                "collected_gems_total": sum(int(row["collected_gems_total"]) for row in rows),
                "percentage_assisted_blue_gems": sum(
                    int(row.get("percentage_assist_applied", 0))
                    and row.get("percentage_assist_color") == "blue"
                    for row in rows
                ),
                "percentage_assisted_green_gems": sum(
                    int(row.get("percentage_assist_applied", 0))
                    and row.get("percentage_assist_color") == "green"
                    for row in rows
                ),
                "percentage_assisted_red_gems": sum(
                    int(row.get("percentage_assist_applied", 0))
                    and row.get("percentage_assist_color") == "red"
                    for row in rows
                ),
                "percentage_assisted_events": sum(
                    int(row.get("percentage_assist_applied", 0)) for row in rows
                ),
                "visual_color_events": sum(str(row["color_evidence"]).startswith("visual_") for row in rows),
                "trajectory_color_events": sum(row["color_evidence"] == "visual_trajectory" for row in rows),
                "occlusion_aware_trajectory_events": sum(
                    int(row["trajectory_occlusion_accepted"]) for row in rows
                ),
                "xp_single_step_blue_events": sum(row["color_evidence"] == "xp_single_step_blue_fallback" for row in rows),
                "color_ambiguous_events": sum(row["color_evidence"] == "xp_confirmed_color_ambiguous" for row in rows),
                "events_needing_review": sum(int(row["needs_review"]) for row in rows),
                "event_ids": "|".join(str(row["event_id"]) for row in rows),
                "event_keys": "|".join(str(row["event_key"]) for row in rows),
            }
        )
    return result


def build_level_pattern_rows(
    event_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize non-truncated XP-bar increases by observed HUD level."""
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in event_rows:
        grouped.setdefault(int(row["hud_level"]), []).append(row)

    def percent_values(rows: list[dict[str, object]]) -> np.ndarray:
        return np.asarray(
            [float(row["xp_bar_increase_percent"]) for row in rows],
            dtype=np.float64,
        )

    def median_or_blank(rows: list[dict[str, object]]) -> float | str:
        values = percent_values(rows)
        return round(float(np.median(values)), 4) if len(values) else ""

    result: list[dict[str, object]] = []
    for level in sorted(grouped):
        rows = grouped[level]
        complete = [row for row in rows if not int(row["xp_bar_saturated"])]
        values = percent_values(complete)

        def single_visual_color(color: str) -> list[dict[str, object]]:
            return [
                row
                for row in complete
                if str(row["color_evidence"]).startswith("visual_")
                and int(row[f"collected_{color}_gems"]) == 1
                and int(row["collected_gems_total"]) == 1
                and int(row["unresolved_collected_gems"]) == 0
            ]

        blue = single_visual_color("blue")
        green = single_visual_color("green")
        red = single_visual_color("red")
        confidences = np.asarray(
            [float(row["hud_level_ocr_confidence"]) for row in rows],
            dtype=np.float64,
        )
        result.append(
            {
                "hud_level": level,
                "first_video_time": min(float(row["video_time_b"]) for row in rows),
                "last_video_time": max(float(row["video_time_b"]) for row in rows),
                "xp_jump_events": len(rows),
                "non_saturated_events": len(complete),
                "saturated_events": len(rows) - len(complete),
                "xp_increase_percent_min": (
                    round(float(np.min(values)), 4) if len(values) else ""
                ),
                "xp_increase_percent_q1": (
                    round(float(np.quantile(values, 0.25)), 4) if len(values) else ""
                ),
                "xp_increase_percent_median": (
                    round(float(np.median(values)), 4) if len(values) else ""
                ),
                "xp_increase_percent_mean": (
                    round(float(np.mean(values)), 4) if len(values) else ""
                ),
                "xp_increase_percent_q3": (
                    round(float(np.quantile(values, 0.75)), 4) if len(values) else ""
                ),
                "xp_increase_percent_max": (
                    round(float(np.max(values)), 4) if len(values) else ""
                ),
                "single_visual_blue_events": len(blue),
                "single_visual_blue_median_percent": median_or_blank(blue),
                "single_visual_green_events": len(green),
                "single_visual_green_median_percent": median_or_blank(green),
                "single_visual_red_events": len(red),
                "single_visual_red_median_percent": median_or_blank(red),
                "unresolved_events": sum(
                    int(row["unresolved_collected_gems"]) > 0 for row in rows
                ),
                "mean_level_ocr_confidence": round(float(np.mean(confidences)), 4),
                "reset_inference_agreement_rate": round(
                    float(
                        np.mean(
                            [
                                int(row["level_agrees_with_reset_inference"])
                                for row in rows
                            ]
                        )
                    ),
                    4,
                ),
            }
        )
    return result


def write_signal_diagnostics(
    path: Path,
    metadata: dict[str, float],
    arrays: dict[str, np.ndarray],
    events: list[XPEvent],
    resets: list[int],
) -> None:
    events_by_frame = {event.frame_b: event for event in events}
    reset_set = set(resets)
    fps = metadata["fps"]
    bar_width = float(metadata["bar_width"])
    fieldnames = [
        "frame_index",
        "video_time",
        "xp_progress",
        "xp_progress_percent",
        "xp_endpoint_pixels",
        "xp_delta_pixels",
        "xp_delta_percent_of_full_bar",
        "xp_quality",
        "overlay_score",
        "gameplay_hud_score",
        "player_anchor_x",
        "player_anchor_y",
        "health_bar_x",
        "health_bar_y",
        "health_bar_detected",
        "health_bar_confidence",
        "level_up_transition_guard",
        "xp_bar_reappearance",
        "flagged_event_id",
        "flagged_event_hud_level",
        "flagged_event_level_ocr_confidence",
        "flagged_event_level_ocr_accepted",
        "flagged_event_level_ocr_validation",
        "flagged_event_level_source",
        "level_reset",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in range(len(arrays["pixels"])):
            event = events_by_frame.get(frame_index)
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "video_time": round(frame_index / fps, 4),
                    "xp_progress": round(float(arrays["progress"][frame_index]), 7),
                    "xp_progress_percent": round(
                        100.0 * float(arrays["progress"][frame_index]),
                        4,
                    ),
                    "xp_endpoint_pixels": int(arrays["pixels"][frame_index]),
                    "xp_delta_pixels": int(arrays["delta_pixels"][frame_index]),
                    "xp_delta_percent_of_full_bar": round(
                        100.0
                        * float(arrays["delta_pixels"][frame_index])
                        / bar_width,
                        4,
                    ),
                    "xp_quality": round(float(arrays["quality"][frame_index]), 4),
                    "overlay_score": round(float(arrays["overlay"][frame_index]), 4),
                    "gameplay_hud_score": round(float(arrays["hud"][frame_index]), 4),
                    "player_anchor_x": round(
                        float(arrays["player_anchor_x"][frame_index]), 2
                    ),
                    "player_anchor_y": round(
                        float(arrays["player_anchor_y"][frame_index]), 2
                    ),
                    "health_bar_x": (
                        round(float(arrays["health_bar_x"][frame_index]), 2)
                        if math.isfinite(float(arrays["health_bar_x"][frame_index]))
                        else ""
                    ),
                    "health_bar_y": (
                        round(float(arrays["health_bar_y"][frame_index]), 2)
                        if math.isfinite(float(arrays["health_bar_y"][frame_index]))
                        else ""
                    ),
                    "health_bar_detected": int(
                        arrays["health_bar_detected"][frame_index]
                    ),
                    "health_bar_confidence": round(
                        float(arrays["health_bar_confidence"][frame_index]), 4
                    ),
                    "level_up_transition_guard": int(
                        arrays["level_up_transition_guard"][frame_index]
                    ),
                    "xp_bar_reappearance": int(
                        arrays["xp_bar_reappearance"][frame_index]
                    ),
                    "flagged_event_id": event.event_id if event else "",
                    "flagged_event_hud_level": event.hud_level if event else "",
                    "flagged_event_level_ocr_confidence": (
                        round(event.hud_level_ocr_confidence, 4) if event else ""
                    ),
                    "flagged_event_level_ocr_accepted": (
                        int(event.hud_level_ocr_accepted) if event else ""
                    ),
                    "flagged_event_level_ocr_validation": (
                        event.hud_level_ocr_validation if event else ""
                    ),
                    "flagged_event_level_source": (
                        event.hud_level_source if event else ""
                    ),
                    "level_reset": int(frame_index in reset_set),
                }
            )


def write_summary(
    path: Path,
    video_path: Path,
    metadata: dict[str, float],
    rows: list[dict[str, object]],
    resets: list[int],
    template_profile: str,
    arrays: dict[str, np.ndarray],
    percentage_report: dict[str, object] | None = None,
) -> dict[str, object]:
    visual_rows = [row for row in rows if str(row["color_evidence"]).startswith("visual_")]
    ocr_rows = [row for row in rows if row["hud_level_source"] == "hud_ocr"]
    summary = {
        "video": video_path.name,
        "template_profile": template_profile,
        "duration_seconds": round(metadata["duration"], 3),
        "fps": round(metadata["fps"], 4),
        "xp_bar_width_pixels": int(metadata["bar_width"]),
        "xp_jump_events": len(rows),
        "hud_level_min": min((int(row["hud_level"]) for row in rows), default=None),
        "hud_level_max": max((int(row["hud_level"]) for row in rows), default=None),
        "hud_level_raw_ocr_events": sum(
            bool(row["hud_level_ocr_text"]) for row in rows
        ),
        "hud_level_ocr_events": len(ocr_rows),
        "hud_level_rejected_ocr_events": sum(
            not int(row["hud_level_ocr_accepted"]) for row in rows
        ),
        "hud_level_fallback_events": len(rows) - len(ocr_rows),
        "hud_level_ocr_mean_confidence": round(
            float(
                np.mean(
                    [float(row["hud_level_ocr_confidence"]) for row in ocr_rows]
                )
            )
            if ocr_rows
            else 0.0,
            4,
        ),
        "reset_inference_level_agreement_rate": round(
            float(
                np.mean(
                    [int(row["level_agrees_with_reset_inference"]) for row in rows]
                )
            )
            if rows
            else 0.0,
            4,
        ),
        "mean_xp_bar_increase_percent": round(
            float(np.mean([float(row["xp_bar_increase_percent"]) for row in rows]))
            if rows
            else 0.0,
            4,
        ),
        "collected_blue_gems": sum(int(row["collected_blue_gems"]) for row in rows),
        "collected_green_gems": sum(int(row["collected_green_gems"]) for row in rows),
        "collected_red_gems": sum(int(row["collected_red_gems"]) for row in rows),
        "unresolved_collected_gems": sum(int(row["unresolved_collected_gems"]) for row in rows),
        "collected_gems_total": sum(int(row["collected_gems_total"]) for row in rows),
        "visual_color_events": sum(str(row["color_evidence"]).startswith("visual_") for row in rows),
        "trajectory_color_events": sum(row["color_evidence"] == "visual_trajectory" for row in rows),
        "occlusion_aware_trajectory_events": sum(
            int(row["trajectory_occlusion_accepted"]) for row in rows
        ),
        "visually_classified_blue_gems": sum(int(row["collected_blue_gems"]) for row in visual_rows),
        "visually_classified_green_gems": sum(int(row["collected_green_gems"]) for row in visual_rows),
        "visually_classified_red_gems": sum(int(row["collected_red_gems"]) for row in visual_rows),
        "xp_single_step_blue_fallback_events": sum(row["color_evidence"] == "xp_single_step_blue_fallback" for row in rows),
        "color_ambiguous_events": sum(row["color_evidence"] == "xp_confirmed_color_ambiguous" for row in rows),
        "events_needing_review": sum(int(row["needs_review"]) for row in rows),
        "level_resets_detected": len(resets),
        "health_bar_detection_rate": round(
            float(np.mean(arrays["health_bar_detected"])),
            4,
        ),
        "level_up_transition_guard_frames": int(
            np.sum(arrays["level_up_transition_guard"])
        ),
        "xp_bar_reappearance_frames": int(
            np.sum(arrays["xp_bar_reappearance"])
        ),
        "pickup_radius_1440p": Config().pickup_radius_1440p,
        "trajectory_min_strong_history_frames": Config().trajectory_min_strong_history_frames,
        "trajectory_max_occluded_endpoint_frames": Config().trajectory_max_occluded_endpoint_frames,
        "level_ocr_min_confidence": Config().level_ocr_min_confidence,
        "level_ocr_max_reset_lead": Config().level_ocr_max_reset_lead,
        "level_ocr_max_reset_lag": Config().level_ocr_max_reset_lag,
        "human_labels_used": False,
        "audio_used": False,
    }
    if percentage_report is not None:
        assisted = percentage_report["percentage_assisted"]
        count_gate = percentage_report["single_pickup_count_gate"]
        color_gates = percentage_report["color_gates"]
        growth_adaptation = percentage_report["growth_correction"][
            "deployment_adaptation"
        ]
        summary.update(
            {
                "video_only_percentage_assist_enabled": True,
                "percentage_assisted_events": int(assisted["events"]),
                "percentage_assisted_blue_gems": int(assisted["blue_gems"]),
                "percentage_assisted_green_gems": int(assisted["green_gems"]),
                "percentage_assisted_red_gems": int(assisted["red_gems"]),
                "percentage_unresolved_before": int(
                    percentage_report["baseline"]["unresolved_collected_gems"]
                ),
                "percentage_unresolved_after": int(
                    percentage_report["final"]["unresolved_collected_gems"]
                ),
                "count_single_pickup_validation_precision": float(
                    count_gate["validation"]["precision"]
                ),
                "xp_growth_correction_factor": float(
                    percentage_report["growth_correction"]["factor"]
                ),
                "xp_growth_adaptation_past_only": bool(
                    growth_adaptation["past_only"]
                ),
                "xp_growth_rolling_sample_limit": int(
                    growth_adaptation["rolling_sample_limit"]
                ),
                "xp_growth_deployment_factor_min": growth_adaptation[
                    "deployment_factor_min"
                ],
                "xp_growth_deployment_factor_max": growth_adaptation[
                    "deployment_factor_max"
                ],
                "xp_growth_deployment_factor_final": growth_adaptation[
                    "deployment_factor_final"
                ],
                "assisted_blue_validation_precision": float(
                    color_gates["blue"]["validation"]["precision"]
                ),
                "assisted_blue_validation_predictions": int(
                    color_gates["blue"]["validation"][
                        "predicted_positive_events"
                    ]
                ),
                "assisted_blue_gate_enabled": bool(
                    color_gates["blue"]["enabled"]
                ),
                "assisted_blue_deployment_precision": float(
                    color_gates["blue"]["deployment"]["precision"]
                ),
                "assisted_blue_deployment_predictions": int(
                    color_gates["blue"]["deployment"][
                        "predicted_positive_events"
                    ]
                ),
                "assisted_blue_deployment_audit_passed": bool(
                    percentage_report["deployment_audit_passed"]["blue"]
                ),
                "assisted_green_gate_enabled": bool(
                    color_gates["green"]["enabled"]
                ),
                "assisted_red_gate_enabled": bool(
                    color_gates["red"]["enabled"]
                ),
                "full_video_run_recommended": bool(
                    percentage_report["full_video_run_recommended"]
                ),
                "production_output_recommended": bool(
                    percentage_report["production_output_recommended"]
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_method_note(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# XP-jump temporal collected-gem detection",
        "",
        "## Method",
        "",
        "The run is computer-vision only. No Gaming Content spreadsheet or human-coded count is read.",
        f"Template profile: {summary['template_profile']}.",
        "Imelda is located from the red health bar below her; screen center is only a fallback when",
        "the bar is temporarily hidden. Pickup distance is measured from that frame-specific anchor.",
        "Level-up transitions are excluded, and a bar that reappears after a short unreadable gap is",
        "treated as restoration rather than a new XP increase.",
        "The XP crop spans the complete visible bar. Every video frame is checked for its endpoint, and",
        "each persistent positive XP-bar change flags",
        "A (the last unchanged frame) and B (the first changed frame). Blue, green, and red sprite",
        "matching is then performed only in a short window around each flagged jump.",
        "At every flagged B frame, OCR reads the fixed `LV n` label at the right end of the bar. The event",
        "table retains the raw OCR value, confidence, source, and the older reset-count estimate for audit.",
        "A confirmed level must include the `LV` prefix, remain close to the reset timeline, and preserve",
        "the nondecreasing level sequence. Rejected raw OCR is retained but cannot set the confirmed level.",
        "Progress before, progress after, and increase are reported as percentages of the full XP bar.",
        "A bar-filling increase is marked saturated because its visible percentage is right-censored.",
        "",
        "The strongest evidence is a same-color sprite tracked for several frames toward Imelda that",
        "disappears at B or within the allowed two-frame XP-HUD delay. This catches pickups whose sprite",
        "vanishes just before the bar visibly changes and rejects stationary or outward-moving effects.",
        "When the final template score falls during overlap with Imelda or her health bar, the trajectory",
        "can remain valid only if it contains at least three earlier strong same-color detections and no",
        "more than two weak endpoint frames. Color is therefore established from the track history rather",
        "than from the partially occluded final image.",
        "The original A-to-B disappearance remains a fallback. Jump magnitude is retained",
        "as a diagnostic XP-value band: blue up to 2 XP, green up to 9 XP, and red above 9 XP after an",
        "estimated Growth adjustment. Because several lower-value gems can merge into one frame, magnitude",
        "alone is used only for an isolated blue-sized step; it never claims a green or red pickup. A jump",
        "that fills the bar is not assigned a color from magnitude because its true XP is truncated.",
        "",
        "## Results",
        "",
        f"- XP jump events: {summary['xp_jump_events']}",
        f"- Full XP-bar width measured: {summary['xp_bar_width_pixels']} pixels",
        f"- Observed HUD level range: {summary['hud_level_min']}-{summary['hud_level_max']}",
        f"- Events with direct HUD-level OCR: {summary['hud_level_ocr_events']}",
        f"- Events with rejected or missing level OCR: {summary['hud_level_rejected_ocr_events']}",
        f"- Events using a level fallback: {summary['hud_level_fallback_events']}",
        f"- Mean accepted level-OCR confidence: {summary['hud_level_ocr_mean_confidence']:.1%}",
        f"- Reset-count estimate agreement with visible level: {summary['reset_inference_level_agreement_rate']:.1%}",
        f"- Blue pickups: {summary['collected_blue_gems']}",
        f"- Green pickups: {summary['collected_green_gems']}",
        f"- Red pickups: {summary['collected_red_gems']}",
        f"- Unresolved-color pickups: {summary['unresolved_collected_gems']}",
        f"- Total collected gems (including unresolved): {summary['collected_gems_total']}",
        f"- Events with direct visual color evidence: {summary['visual_color_events']}",
        f"- Events classified by multi-frame trajectory: {summary['trajectory_color_events']}",
        f"- Trajectory events accepted through the occlusion-aware endpoint rule: {summary['occlusion_aware_trajectory_events']}",
        f"- Hidden-sprite events classified only as a single blue-sized XP step: {summary['xp_single_step_blue_fallback_events']}",
        f"- XP-confirmed events with ambiguous color: {summary['color_ambiguous_events']}",
        f"- Health-bar detection rate across all frames: {summary['health_bar_detection_rate']:.1%}",
        f"- XP-bar restoration frames rejected: {summary['xp_bar_reappearance_frames']}",
        "",
        "## Interpretation limits",
        "",
        "The XP bar confirms collection but does not always reveal how many gems arrived in exactly the",
        "same video frame. A large jump can be one high-value gem or several lower-value gems, so magnitude",
        "alone is never used to claim green or red. When no sprite disappearance is visible, the output",
        "records one conservative lower-bound pickup for the XP update. Those rows are marked for review.",
        "Level resets are diagnostics and are not counted as new pickups. The CSVs are model output, not an",
        "accuracy evaluation, because this run uses no ground truth.",
        "",
    ]
    if summary.get("video_only_percentage_assist_enabled"):
        interpretation_index = lines.index("## Interpretation limits")
        percentage_lines = [
            "## Video-only percentage assistance",
            "",
            "The first 3:30 is used to calibrate the conversion from level-normalized XP-bar growth",
            "to estimated base XP. The following 1:30 is frozen held-out validation. Reference labels",
            "come only from strong visual trajectories, including visually confirmed multi-pickup",
            "events; no Gaming Content sheet or human-coded count is read.",
            "After 5:00, Growth correction uses a rolling median of only earlier isolated, visually",
            "certain blue pickups. The current event and future frames cannot tune their own XP value.",
            "This past-only update can follow Crown or another later Growth change.",
            "",
            "A second, more sensitive trajectory pass proposes pickup count and a visual color candidate.",
            "The final classifier requires exactly one such track, agreement between its color and the",
            "level-normalized XP band, no evidence of a second unmatched candidate of that color, an",
            "isolated XP update, a direct HUD level, and a non-saturated bar. Each color must have at",
            "least 20 held-out predictions and at least 95% empirical precision before it can change an",
            "unresolved event. Post-5:00 assignments additionally require the full deployment audit to",
            "meet the same support and precision requirements.",
            "",
            f"- Learned XP Growth correction factor: {summary['xp_growth_correction_factor']:.4f}",
            f"- Standalone count-pass held-out precision: {summary['count_single_pickup_validation_precision']:.2%}",
            f"- End-to-end blue held-out precision: {summary['assisted_blue_validation_precision']:.2%} "
            f"({summary['assisted_blue_validation_predictions']} predictions)",
            f"- Blue percentage gate enabled: {summary['assisted_blue_gate_enabled']}",
            f"- Deployment blue audit precision: {summary['assisted_blue_deployment_precision']:.2%} "
            f"({summary['assisted_blue_deployment_predictions']} predictions)",
            f"- Deployment blue audit passed: {summary['assisted_blue_deployment_audit_passed']}",
            f"- Green percentage gate enabled: {summary['assisted_green_gate_enabled']}",
            f"- Red percentage gate enabled: {summary['assisted_red_gate_enabled']}",
            f"- Percentage-assisted events: {summary['percentage_assisted_events']}",
            f"- Unresolved pickups before/after assistance: "
            f"{summary['percentage_unresolved_before']}/{summary['percentage_unresolved_after']}",
            "",
        ]
        lines[interpretation_index:interpretation_index] = percentage_lines
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template-profile",
        choices=("legacy", "wiki-multi"),
        default="legacy",
        help="Use the stable single-reference profile or the experimental wiki reference bank.",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--percentage-calibration-end-seconds",
        type=float,
        default=Config().percentage_calibration_end_seconds,
    )
    parser.add_argument(
        "--percentage-validation-end-seconds",
        type=float,
        default=Config().percentage_validation_end_seconds,
    )
    parser.add_argument("--disable-percentage-assist", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        Config(),
        percentage_calibration_end_seconds=(
            args.percentage_calibration_end_seconds
        ),
        percentage_validation_end_seconds=(
            args.percentage_validation_end_seconds
        ),
    )
    if (
        cfg.percentage_calibration_end_seconds
        >= cfg.percentage_validation_end_seconds
    ):
        raise ValueError(
            "Percentage calibration must end before percentage validation."
        )
    video_path = args.video.expanduser().resolve()
    template_dir = args.template_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, arrays = scan_xp_signal(video_path, cfg, args.max_seconds)
    resets = find_level_resets(arrays, cfg)
    events = find_xp_events(metadata, arrays, resets, cfg)
    events = attach_hud_levels(video_path, events, cfg)
    events = attach_step_estimates(events, int(metadata["bar_width"]))
    print(f"Flagged {len(events)} XP A/B events; detected {len(resets)} level resets", flush=True)

    templates = load_template_bank(
        template_dir,
        int(metadata["height"]),
        profile=args.template_profile,
    )
    template_counts = ", ".join(
        f"{color}={len(references)}" for color, references in templates.items()
    )
    print(
        f"Loaded {args.template_profile} gem references: {template_counts}",
        flush=True,
    )
    rows = analyze_flagged_windows(
        video_path,
        events,
        templates,
        arrays,
        output_dir,
        cfg,
        save_previews=not args.no_previews,
    )
    percentage_report: dict[str, object] | None = None
    percentage_metrics: list[dict[str, object]] = []
    if not args.disable_percentage_assist:
        percentage_report, percentage_metrics = (
            apply_video_only_percentage_calibration(rows, cfg)
        )

    label = clean_video_label(video_path)
    event_path = output_dir / f"collected_gems_{label}_xp_ab_events.csv"
    per_second_path = output_dir / f"collected_gems_{label}_per_second.csv"
    five_second_path = output_dir / f"collected_gems_{label}_5sec_intervals.csv"
    diagnostics_path = output_dir / f"collected_gems_{label}_xp_frame_signal.csv"
    level_pattern_path = output_dir / f"collected_gems_{label}_xp_level_patterns.csv"
    percentage_metrics_path = (
        output_dir / f"collected_gems_{label}_video_only_validation.csv"
    )
    write_csv(event_path, rows)
    write_csv(per_second_path, build_aggregate_rows(rows, metadata["duration"], 1))
    write_csv(five_second_path, build_aggregate_rows(rows, metadata["duration"], 5))
    write_csv(level_pattern_path, build_level_pattern_rows(rows))
    if percentage_report is not None:
        write_csv(percentage_metrics_path, percentage_metrics)
        (output_dir / "video_only_xp_calibration_report.json").write_text(
            json.dumps(percentage_report, indent=2) + "\n"
        )
    write_signal_diagnostics(diagnostics_path, metadata, arrays, events, resets)
    summary = write_summary(
        output_dir / "summary.json",
        video_path,
        metadata,
        rows,
        resets,
        args.template_profile,
        arrays,
        percentage_report,
    )
    write_method_note(output_dir / "METHOD_AND_RESULTS.md", summary)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Event CSV: {event_path}", flush=True)
    print(f"Level-pattern CSV: {level_pattern_path}", flush=True)
    print(f"5-second CSV: {five_second_path}", flush=True)
    if percentage_report is not None:
        print(
            "Video-only calibration report: "
            f"{output_dir / 'video_only_xp_calibration_report.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
