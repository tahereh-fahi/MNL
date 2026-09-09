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
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Config:
    initial_level: int = 1
    xp_bar_x0_fraction: float = 0.042
    xp_bar_x1_fraction: float = 0.948
    xp_bar_y0_fraction: float = 0.006
    xp_bar_y1_fraction: float = 0.019
    xp_hue_low: int = 100
    xp_hue_high: int = 125
    xp_saturation_low: int = 80
    xp_value_low: int = 50
    xp_column_fraction: float = 0.55
    xp_occluder_hue_low: int = 3
    xp_occluder_hue_high: int = 28
    xp_occluder_saturation_low: int = 120
    xp_occluder_value_low: int = 100
    xp_occluder_min_visible_fraction: float = 0.20
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
    magnet_base_value: float = 30.0
    magnet_search_radius_cap_1440p: float = 1280.0
    magnet_entry_stride_frames: int = 2
    magnet_entry_outer_margin_1440p: float = 100.0
    magnet_entry_template_threshold: float = 0.84
    magnet_entry_link_distance_1440p: float = 95.0
    magnet_entry_max_track_gap_frames: int = 4
    magnet_entry_min_observations: int = 3
    magnet_entry_inside_confirmation_observations: int = 2
    magnet_entry_boundary_tolerance_1440p: float = 10.0
    magnet_entry_min_approach_1440p: float = 12.0
    magnet_entry_min_monotonic_fraction: float = 0.67
    magnet_entry_min_confidence: float = 0.72
    magnet_entry_max_assignment_lag_frames: int = 60
    magnet_entry_arrival_tolerance_frames: int = 8
    magnet_entry_assignment_margin_frames: int = 3
    magnet_entry_event_isolation_frames: int = 3
    magnet_entry_validation_min_predictions: int = 20
    magnet_entry_target_precision: float = 0.95
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
    post_level_up_review: bool
    event_detection_reason: str
    level_up_boundary_candidate: bool
    xp_increase_censored: bool
    xp_full_inferred_from_level_change: bool
    hud_level_before: int
    hud_level_after: int


@dataclass(frozen=True)
class AttractorbChange:
    video_second: float
    level: int
    source_event_id: str
    confidence: str
    needs_review: bool


@dataclass(frozen=True)
class MagnetModel:
    enabled: bool
    attractorb_changes: tuple[AttractorbChange, ...]
    powerup_rank: int = 0
    character_multiplier: float = 1.0
    golden_egg_bonus: float = 0.0
    timeline_source: str = ""


@dataclass(frozen=True)
class MagnetState:
    model_enabled: bool
    magnet_value: float
    multiplier_from_base: float
    attractorb_level: int
    attractorb_multiplier: float
    search_radius_1440p: float
    source_event_id: str
    source_confidence: str
    source_needs_review: bool


@dataclass(frozen=True)
class TrajectoryEvidence:
    color: str
    points: tuple[tuple[int, GemDetection], ...]
    disappearance_frame: int
    start_distance: float
    start_distance_magnet_ratio: float
    end_distance: float
    approach_pixels: float
    monotonic_fraction: float
    mean_template_score: float
    evidence_score: float
    strong_template_frames: int
    weak_endpoint_frames: int
    occlusion_accepted: bool


@dataclass(frozen=True)
class MagnetTrackObservation:
    frame_index: int
    detection: GemDetection
    distance: float
    magnet_radius: float
    magnet_state: MagnetState


@dataclass(frozen=True)
class MagnetEntryEvidence:
    track_id: int
    color: str
    entry_frame: int
    confirmation_frame: int
    predicted_arrival_frame: int
    entry_reason: str
    entry_x: float
    entry_y: float
    entry_distance: float
    magnet_radius: float
    radial_speed_pixels_per_frame: float
    mean_template_score: float
    monotonic_fraction: float
    confidence: float
    attractorb_level: int
    magnet_value: float
    magnet_state_source_event_id: str
    magnet_state_needs_review: bool


@dataclass
class ActiveMagnetTrack:
    track_id: int
    color: str
    observations: list[MagnetTrackObservation] = field(default_factory=list)
    committed: bool = False


COLORS = (
    ColorSpec("blue", ((88, 106),), (255, 140, 0)),
    ColorSpec("green", ((72, 88),), (0, 220, 0)),
    ColorSpec("red", ((0, 12), (170, 179)), (0, 0, 255)),
)

GemTemplate = tuple[np.ndarray, np.ndarray]
GemTemplateBank = dict[str, tuple[GemTemplate, ...]]


# Exact total multipliers reported by the Vampire Survivors Wiki for
# Attractorb levels 0 through 5.
ATTRACTORB_TOTAL_MULTIPLIERS = (
    1.0,
    1.5,
    1.995,
    2.49375,
    2.9925,
    3.980025,
)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def attractorb_multiplier(level: int) -> float:
    if not 0 <= level < len(ATTRACTORB_TOTAL_MULTIPLIERS):
        raise ValueError(f"Attractorb level must be between 0 and 5, got {level}")
    return ATTRACTORB_TOTAL_MULTIPLIERS[level]


def load_attractorb_changes(
    path: Path | None,
    include_review_required: bool = True,
) -> tuple[AttractorbChange, ...]:
    """Read Attractorb acquisitions/upgrades from an inventory-event CSV.

    A compact manually curated file with ``video_second`` and
    ``attractorb_level`` is also accepted. Inventory-model uncertainty is
    retained on every change and later copied into the gem-event audit table.
    """
    if path is None:
        return ()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    changes: list[AttractorbChange] = []
    for index, row in enumerate(rows, start=2):
        dedicated_level = str(row.get("attractorb_level", "")).strip()
        item_names = {
            str(row.get(key, "")).strip().lower()
            for key in ("item_before", "item_after", "suggested_item")
        }
        if not dedicated_level and "attractorb" not in item_names:
            continue
        level_text = dedicated_level or str(row.get("level_after", "")).strip()
        second_text = str(row.get("video_second", "")).strip()
        if not level_text or not second_text:
            raise ValueError(
                f"Missing Attractorb level or video_second in {path} row {index}"
            )
        level = int(float(level_text))
        attractorb_multiplier(level)
        needs_review = parse_bool(row.get("needs_review", False))
        if needs_review and not include_review_required:
            continue
        changes.append(
            AttractorbChange(
                video_second=float(second_text),
                level=level,
                source_event_id=str(
                    row.get("event_id", f"{path.stem}_row_{index}")
                ),
                confidence=str(row.get("confidence", "")).strip(),
                needs_review=needs_review,
            )
        )
    changes.sort(key=lambda item: (item.video_second, item.level))
    previous_level = 0
    for change in changes:
        if change.level < previous_level:
            raise ValueError(
                "Attractorb levels must not decrease: "
                f"{previous_level} -> {change.level} at {change.video_second:.3f}s"
            )
        previous_level = change.level
    return tuple(changes)


def magnet_state_at(
    video_second: float,
    model: MagnetModel,
    cfg: Config,
) -> MagnetState:
    change_times = [change.video_second for change in model.attractorb_changes]
    change_index = bisect_right(change_times, video_second) - 1
    change = model.attractorb_changes[change_index] if change_index >= 0 else None
    level = change.level if change else 0
    item_multiplier = attractorb_multiplier(level)
    powerup_multiplier = 1.25 ** model.powerup_rank
    magnet_value = (
        (cfg.magnet_base_value + model.golden_egg_bonus)
        * model.character_multiplier
        * powerup_multiplier
        * item_multiplier
    )
    multiplier_from_base = magnet_value / cfg.magnet_base_value
    search_radius = cfg.search_radius_1440p
    if model.enabled:
        search_radius = min(
            cfg.magnet_search_radius_cap_1440p,
            cfg.search_radius_1440p * multiplier_from_base,
        )
    return MagnetState(
        model_enabled=model.enabled,
        magnet_value=magnet_value,
        multiplier_from_base=multiplier_from_base,
        attractorb_level=level,
        attractorb_multiplier=item_multiplier,
        search_radius_1440p=search_radius,
        source_event_id=change.source_event_id if change else "base_magnet_state",
        source_confidence=change.confidence if change else "configured",
        source_needs_review=change.needs_review if change else False,
    )


def seconds_to_stamp(seconds: float, include_milliseconds: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    if include_milliseconds:
        return f"{minutes}:{remainder:06.3f}"
    return f"{minutes}:{int(remainder):02d}"


def seconds_to_duration(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remainder = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


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
    # Some runs place a bright orange rectangular HUD element over the upper
    # part of the XP bar. Treat those pixels as unavailable rather than as
    # evidence that the blue fill stopped. The still-visible pixels beneath
    # the rectangle remain sufficient to measure the endpoint. Requiring a
    # minimum visible fraction prevents a fully covered column from being
    # guessed, and an orange rectangle beyond the fill cannot extend it.
    occluder = (
        (hsv[:, :, 0] >= cfg.xp_occluder_hue_low)
        & (hsv[:, :, 0] <= cfg.xp_occluder_hue_high)
        & (hsv[:, :, 1] >= cfg.xp_occluder_saturation_low)
        & (hsv[:, :, 2] >= cfg.xp_occluder_value_low)
    )
    visible = ~occluder
    visible_count = np.sum(visible, axis=0)
    visible_fraction = visible_count / max(1, blue.shape[0])
    blue_fraction = np.divide(
        np.sum(blue & visible, axis=0),
        visible_count,
        out=np.zeros_like(visible_count, dtype=np.float64),
        where=visible_count > 0,
    )
    columns = (
        (visible_fraction >= cfg.xp_occluder_min_visible_fraction)
        & (blue_fraction >= cfg.xp_column_fraction)
    ).astype(np.uint8)
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
    target_frames = {event.frame_b for event in events}
    target_frames.update(
        event.frame_a for event in events if event.level_up_boundary_candidate
    )
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
        if frame_index not in target_frames:
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
        if event.level_up_boundary_candidate:
            before_level, before_confidence, before_text = observations.get(
                event.frame_a, (None, 0.0, "")
            )
            after_level, after_confidence, after_text = observations.get(
                event.frame_b, (None, 0.0, "")
            )
            boundary_accepted = (
                before_level is not None
                and after_level is not None
                and before_confidence >= cfg.level_ocr_min_confidence
                and after_confidence >= cfg.level_ocr_min_confidence
                and before_text.startswith("LV")
                and after_text.startswith("LV")
                and int(after_level) == int(before_level) + 1
            )
            if boundary_accepted:
                level_before = int(before_level)
                level_after = int(after_level)
                boundary_validation = "accepted_level_up_boundary"
                boundary_source = "hud_ocr_level_change"
                previous_ocr_level = level_after
            else:
                level_before = event.hud_level_before
                level_after = event.hud_level_after
                boundary_validation = "level_up_boundary_needs_visual_review"
                boundary_source = "boundary_pattern_pending_review"
            required = experience_required_for_level(level_before)
            growth = imelda_growth_multiplier(level_before, cfg)
            effective_delta_fraction = max(0.0, 1.0 - event.progress_a)
            enriched.append(
                replace(
                    event,
                    hud_level=level_before,
                    hud_level_ocr_raw=after_level,
                    hud_level_ocr_text=after_text,
                    hud_level_ocr_confidence=min(
                        before_confidence, after_confidence
                    ),
                    hud_level_ocr_accepted=boundary_accepted,
                    hud_level_ocr_validation=boundary_validation,
                    hud_level_source=boundary_source,
                    xp_required=required,
                    growth_multiplier=growth,
                    delta_fraction=(
                        effective_delta_fraction
                        if boundary_accepted
                        else event.delta_fraction
                    ),
                    estimated_base_xp_gain=(
                        effective_delta_fraction * required / growth
                        if boundary_accepted
                        else math.nan
                    ),
                    xp_color_hint="unknown",
                    xp_increase_censored=not boundary_accepted,
                    xp_full_inferred_from_level_change=boundary_accepted,
                    hud_level_before=level_before,
                    hud_level_after=level_after,
                )
            )
            continue
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
    level_guard = arrays["level_up_transition_guard"]
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
            and not level_guard[frame_a]
            and not level_guard[frame_index]
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
    reset_set = set(resets)
    boundary_frame_a: dict[int, int] = {}
    for frame_b in resets:
        candidates = np.arange(max(0, frame_b - 15), frame_b, dtype=np.int32)
        reliable = candidates[
            (quality[candidates] >= cfg.xp_min_quality)
            & (hud[candidates] >= cfg.hud_score_threshold)
        ]
        pool = reliable if len(reliable) else candidates
        if len(pool):
            # Prefer the latest frame when several frames share the same
            # maximum, keeping A as close as possible to the canonical reset.
            maximum = int(np.max(pixels[pool]))
            boundary_frame_a[frame_b] = int(pool[pixels[pool] == maximum][-1])

    reset_cursor = 0
    current_level = cfg.initial_level
    for frame_b in range(1, len(pixels)):
        while reset_cursor < len(resets) and resets[reset_cursor] < frame_b:
            current_level += 1
            reset_cursor += 1

        if frame_b in reset_set and frame_b in boundary_frame_a:
            frame_a = boundary_frame_a[frame_b]
            effective_delta_pixels = max(0, bar_width - int(pixels[frame_a]))
            effective_delta_fraction = max(0.0, 1.0 - float(progress[frame_a]))
            events.append(
                XPEvent(
                    event_id=len(events) + 1,
                    frame_a=frame_a,
                    frame_b=frame_b,
                    video_time_a=frame_a / fps,
                    video_time_b=frame_b / fps,
                    progress_a=float(progress[frame_a]),
                    progress_b=float(progress[frame_b]),
                    delta_pixels=effective_delta_pixels,
                    delta_fraction=effective_delta_fraction,
                    quality_a=float(quality[frame_a]),
                    quality_b=float(quality[frame_b]),
                    inferred_level=current_level,
                    hud_level=current_level,
                    hud_level_ocr_raw=None,
                    hud_level_ocr_text="",
                    hud_level_ocr_confidence=0.0,
                    hud_level_ocr_accepted=False,
                    hud_level_ocr_validation="pending_boundary_ocr",
                    hud_level_source="level_up_boundary_pending_ocr",
                    xp_required=experience_required_for_level(current_level),
                    growth_multiplier=imelda_growth_multiplier(current_level, cfg),
                    estimated_base_xp_gain=math.nan,
                    xp_color_hint="unknown",
                    bar_saturated=True,
                    local_single_step_pixels=math.nan,
                    jump_step_ratio=math.nan,
                    post_level_up_review=True,
                    event_detection_reason="level_up_boundary_candidate",
                    level_up_boundary_candidate=True,
                    xp_increase_censored=True,
                    xp_full_inferred_from_level_change=False,
                    hud_level_before=current_level,
                    hud_level_after=current_level + 1,
                )
            )
            continue

        frame_a = frame_b - 1
        if delta[frame_b] < cfg.xp_min_jump_pixels:
            continue
        # A hidden/empty bar followed by an apparent full bar is a transition
        # artifact, not a measurable pickup. The actual level-up boundary is
        # represented by the preceding positive-to-zero pair instead.
        if (
            pixels[frame_a] <= cfg.xp_persistence_tolerance_pixels
            and pixels[frame_b] >= bar_width - 2
        ):
            continue
        if reappearance[frame_b]:
            continue
        post_level_up_review = bool(level_guard[frame_a] or level_guard[frame_b])
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
        # During the level-up panel, the bar detector can alternate between
        # empty and full. Keep the single positive-to-zero boundary event
        # above, but reject every positive event that touches this guarded
        # transition window before gem classification or aggregation.
        if post_level_up_review:
            continue
        ordinary_a_valid = (
            hud[frame_a] >= cfg.hud_score_threshold
            and quality[frame_a] >= cfg.xp_min_quality
            and (
                overlay[frame_a] < cfg.xp_overlay_threshold
                or saturated_gameplay_fill
            )
        )
        ordinary_b_valid = (
            hud[frame_b] >= cfg.hud_score_threshold
            and quality[frame_b] >= cfg.xp_min_quality
            and (
                overlay[frame_b] < cfg.xp_overlay_threshold
                or saturated_gameplay_fill
            )
        )
        low_fill_pair_valid = (
            progress[frame_a] < 0.03
            and health_bar_detected[frame_a]
            and health_bar_detected[frame_b]
            and hud[frame_a] >= cfg.hud_score_threshold
            and hud[frame_b] >= cfg.hud_score_threshold
            and quality[frame_b] >= cfg.xp_min_quality
            and overlay[frame_a] < cfg.level_up_strong_overlay_threshold
            and overlay[frame_b] < cfg.level_up_strong_overlay_threshold
        )
        if not (
            (ordinary_a_valid and ordinary_b_valid)
            or low_fill_pair_valid
        ):
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
                post_level_up_review=False,
                event_detection_reason=(
                    "low_fill_xp_increase"
                    if low_fill_pair_valid
                    and not (ordinary_a_valid and ordinary_b_valid)
                    else "ordinary_persistent_xp_increase"
                ),
                level_up_boundary_candidate=False,
                xp_increase_censored=False,
                xp_full_inferred_from_level_change=False,
                hud_level_before=current_level,
                hud_level_after=current_level,
            )
        )
    return events


def validate_transition_event_invariants(
    events: list[XPEvent],
    arrays: dict[str, np.ndarray],
    resets: list[int],
) -> None:
    """Fail fast if transition artifacts can reach downstream outputs."""
    guard = arrays["level_up_transition_guard"]
    reappearance = arrays["xp_bar_reappearance"]
    reset_set = set(resets)
    boundary_frames: list[int] = []
    invalid_events: list[int] = []

    for event in events:
        if event.level_up_boundary_candidate:
            boundary_frames.append(event.frame_b)
            if event.frame_b not in reset_set:
                invalid_events.append(event.event_id)
            continue
        if (
            guard[event.frame_a]
            or guard[event.frame_b]
            or reappearance[event.frame_a]
            or reappearance[event.frame_b]
        ):
            invalid_events.append(event.event_id)

    boundary_set = set(boundary_frames)
    duplicate_boundaries = len(boundary_frames) != len(boundary_set)
    missing_boundaries = sorted(reset_set - boundary_set)
    unexpected_boundaries = sorted(boundary_set - reset_set)
    if (
        invalid_events
        or duplicate_boundaries
        or missing_boundaries
        or unexpected_boundaries
    ):
        details = ",".join(str(event_id) for event_id in invalid_events[:20])
        raise RuntimeError(
            "Transition-event invariant failed: "
            f"invalid_event_ids=[{details}], "
            f"duplicate_boundary_frames={duplicate_boundaries}, "
            f"missing_boundary_frames={missing_boundaries[:20]}, "
            f"unexpected_boundary_frames={unexpected_boundaries[:20]}"
        )


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
    search_radius_1440p: float | None = None,
    enforce_search_circle: bool = False,
) -> list[GemDetection]:
    height, width = frame_bgr.shape[:2]
    scale = height / 1440.0
    radius = int(
        round(
            (
                cfg.search_radius_1440p
                if search_radius_1440p is None
                else search_radius_1440p
            )
            * scale
        )
    )
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
            if enforce_search_circle and math.hypot(
                candidate.cx - player_anchor.x,
                candidate.cy - player_anchor.y,
            ) > radius:
                continue
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


class MagnetEntryTracker:
    """Track gems only until their attraction to the player is committed.

    The tracker samples the video at a configurable stride. A track is committed
    after a stable outside-to-inside Magnet-boundary crossing, or after a gem
    first seen inside the radius shows sustained inward motion. Once committed,
    the color is locked and expensive endpoint tracking is no longer required.
    """

    def __init__(self, cfg: Config, fps: float, frame_height: int) -> None:
        self.cfg = cfg
        self.fps = fps
        self.scale = frame_height / 1440.0
        self.active: dict[int, ActiveMagnetTrack] = {}
        self.entries: list[MagnetEntryEvidence] = []
        self.next_track_id = 1

    def _observation(
        self,
        frame_index: int,
        detection: GemDetection,
        anchor: PlayerAnchor,
        magnet_state: MagnetState,
    ) -> MagnetTrackObservation:
        return MagnetTrackObservation(
            frame_index=frame_index,
            detection=detection,
            distance=math.hypot(
                detection.cx - anchor.x,
                detection.cy - anchor.y,
            ),
            magnet_radius=magnet_state.search_radius_1440p * self.scale,
            magnet_state=magnet_state,
        )

    def _start_track(
        self,
        observation: MagnetTrackObservation,
    ) -> None:
        track = ActiveMagnetTrack(
            track_id=self.next_track_id,
            color=observation.detection.color,
            observations=[observation],
        )
        self.active[track.track_id] = track
        self.next_track_id += 1

    def _recent_duplicate(
        self,
        color: str,
        entry_frame: int,
        x: float,
        y: float,
    ) -> bool:
        frame_tolerance = max(
            self.cfg.magnet_entry_stride_frames,
            self.cfg.magnet_entry_max_track_gap_frames,
        )
        distance_tolerance = self.cfg.nms_radius_1440p * self.scale
        return any(
            entry.color == color
            and abs(entry.entry_frame - entry_frame) <= frame_tolerance
            and math.hypot(entry.entry_x - x, entry.entry_y - y)
            <= distance_tolerance
            for entry in self.entries[-20:]
        )

    def _commit_if_ready(self, track: ActiveMagnetTrack) -> None:
        if track.committed:
            return
        observations = track.observations
        if len(observations) < self.cfg.magnet_entry_min_observations:
            return

        tolerance = self.cfg.magnet_entry_boundary_tolerance_1440p * self.scale
        inside = [
            item.distance <= item.magnet_radius - tolerance
            for item in observations
        ]
        outside = [
            item.distance >= item.magnet_radius + tolerance
            for item in observations
        ]
        inside_required = self.cfg.magnet_entry_inside_confirmation_observations
        if sum(inside[-inside_required:]) < inside_required:
            return

        entry_index: int | None = None
        entry_reason = ""
        for index in range(1, len(observations)):
            if inside[index] and any(outside[:index]):
                entry_index = index
                previous = observations[index - 1]
                current = observations[index]
                entry_reason = (
                    "radius_expansion_crossing"
                    if current.magnet_radius > previous.magnet_radius + tolerance
                    else "magnet_boundary_crossing"
                )
                break

        distances = np.asarray([item.distance for item in observations], dtype=float)
        frame_indexes = np.asarray(
            [item.frame_index for item in observations],
            dtype=float,
        )
        changes = np.diff(distances)
        monotonic_fraction = float(
            np.mean(changes <= self.cfg.trajectory_monotonic_tolerance_1440p * self.scale)
        )
        approach = float(distances[0] - distances[-1])
        minimum_approach = self.cfg.magnet_entry_min_approach_1440p * self.scale

        if entry_index is None:
            # A gem can be created while already inside the current Magnet radius.
            # Accept that case only after several strong inward observations.
            if not inside[0] or approach < minimum_approach:
                return
            entry_index = 0
            entry_reason = "first_seen_moving_inside_magnet"
        elif (
            entry_reason != "radius_expansion_crossing"
            and approach < minimum_approach
        ):
            return

        if monotonic_fraction < self.cfg.magnet_entry_min_monotonic_fraction:
            return
        scores = np.asarray(
            [item.detection.score for item in observations],
            dtype=float,
        )
        mean_score = float(np.mean(scores))
        if mean_score < self.cfg.magnet_entry_template_threshold:
            return

        radial_speeds = []
        for index in range(1, len(observations)):
            frame_gap = frame_indexes[index] - frame_indexes[index - 1]
            if frame_gap <= 0:
                continue
            radial_speeds.append(
                max(0.0, (distances[index - 1] - distances[index]) / frame_gap)
            )
        positive_speeds = [speed for speed in radial_speeds if speed > 0.25]
        radial_speed = (
            float(np.median(positive_speeds)) if positive_speeds else 0.0
        )
        if radial_speed <= 0.0 and entry_reason != "radius_expansion_crossing":
            return

        current = observations[-1]
        pickup_radius = self.cfg.pickup_radius_1440p * self.scale
        remaining_distance = max(0.0, current.distance - pickup_radius)
        predicted_travel = (
            remaining_distance / radial_speed
            if radial_speed > 0.0
            else self.cfg.magnet_entry_max_assignment_lag_frames / 2.0
        )
        predicted_travel = min(
            float(self.cfg.magnet_entry_max_assignment_lag_frames),
            max(1.0, predicted_travel),
        )
        predicted_arrival = int(round(current.frame_index + predicted_travel))

        approach_component = min(
            1.0,
            approach / max(1.0, 3.0 * minimum_approach),
        )
        crossing_component = 1.0 if entry_reason != "first_seen_moving_inside_magnet" else 0.65
        confidence = float(
            0.45 * mean_score
            + 0.25 * monotonic_fraction
            + 0.20 * approach_component
            + 0.10 * crossing_component
        )
        if confidence < self.cfg.magnet_entry_min_confidence:
            return

        entry_observation = observations[entry_index]
        entry_detection = entry_observation.detection
        if self._recent_duplicate(
            track.color,
            entry_observation.frame_index,
            entry_detection.cx,
            entry_detection.cy,
        ):
            track.committed = True
            return

        self.entries.append(
            MagnetEntryEvidence(
                track_id=track.track_id,
                color=track.color,
                entry_frame=entry_observation.frame_index,
                confirmation_frame=current.frame_index,
                predicted_arrival_frame=predicted_arrival,
                entry_reason=entry_reason,
                entry_x=entry_detection.cx,
                entry_y=entry_detection.cy,
                entry_distance=entry_observation.distance,
                magnet_radius=entry_observation.magnet_radius,
                radial_speed_pixels_per_frame=radial_speed,
                mean_template_score=mean_score,
                monotonic_fraction=monotonic_fraction,
                confidence=confidence,
                attractorb_level=entry_observation.magnet_state.attractorb_level,
                magnet_value=entry_observation.magnet_state.magnet_value,
                magnet_state_source_event_id=(
                    entry_observation.magnet_state.source_event_id
                ),
                magnet_state_needs_review=(
                    entry_observation.magnet_state.source_needs_review
                ),
            )
        )
        track.committed = True

    def update(
        self,
        frame_index: int,
        detections: list[GemDetection],
        anchor: PlayerAnchor,
        magnet_state: MagnetState,
    ) -> None:
        maximum_gap = self.cfg.magnet_entry_max_track_gap_frames
        self.active = {
            track_id: track
            for track_id, track in self.active.items()
            if frame_index - track.observations[-1].frame_index <= maximum_gap
        }

        matched_detection_indexes: set[int] = set()
        for color in (spec.name for spec in COLORS):
            tracks = [
                track
                for track in self.active.values()
                if track.color == color
            ]
            detection_indexes = [
                index
                for index, detection in enumerate(detections)
                if detection.color == color
            ]
            if not tracks or not detection_indexes:
                continue
            impossible = 1e6
            costs = np.full(
                (len(tracks), len(detection_indexes)),
                impossible,
                dtype=float,
            )
            for row_index, track in enumerate(tracks):
                previous = track.observations[-1]
                frame_gap = frame_index - previous.frame_index
                allowed = (
                    self.cfg.magnet_entry_link_distance_1440p
                    * self.scale
                    * max(1.0, frame_gap / self.cfg.magnet_entry_stride_frames)
                )
                for column_index, detection_index in enumerate(detection_indexes):
                    detection = detections[detection_index]
                    distance = math.hypot(
                        previous.detection.cx - detection.cx,
                        previous.detection.cy - detection.cy,
                    )
                    if distance <= allowed:
                        costs[row_index, column_index] = distance
            rows, columns = linear_sum_assignment(costs)
            for row_index, column_index in zip(rows.tolist(), columns.tolist()):
                if costs[row_index, column_index] >= impossible:
                    continue
                track = tracks[row_index]
                detection_index = detection_indexes[column_index]
                detection = detections[detection_index]
                track.observations.append(
                    self._observation(
                        frame_index,
                        detection,
                        anchor,
                        magnet_state,
                    )
                )
                # Keep enough history to establish a crossing while bounding memory.
                track.observations = track.observations[-12:]
                self._commit_if_ready(track)
                matched_detection_indexes.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue
            self._start_track(
                self._observation(
                    frame_index,
                    detection,
                    anchor,
                    magnet_state,
                )
            )


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
    magnet_search_radii_by_frame: dict[int, float],
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
            start_frame = points[0][0]
            magnet_radius = magnet_search_radii_by_frame[start_frame]
            start_distance_magnet_ratio = float(
                distances[0] / max(1.0, magnet_radius)
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
                    start_distance_magnet_ratio=start_distance_magnet_ratio,
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
    magnet_state: MagnetState,
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
    if event.post_level_up_review:
        review_reasons.append("post_level_up_xp_increase")
    if event.level_up_boundary_candidate:
        review_reasons.append("level_up_boundary_event")
        if event.xp_full_inferred_from_level_change:
            review_reasons.append("xp_bar_full_inferred_from_level_change")
        else:
            review_reasons.append("level_up_boundary_needs_visual_review")

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
        "xp_bar_progress_b_observed_percent": round(100.0 * event.progress_b, 4),
        "xp_bar_progress_b_effective_percent": (
            100.0
            if event.xp_full_inferred_from_level_change
            else round(100.0 * event.progress_b, 4)
        ),
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
        "post_level_up_review": int(event.post_level_up_review),
        "event_detection_reason": event.event_detection_reason,
        "level_up_boundary_candidate": int(event.level_up_boundary_candidate),
        "xp_increase_censored": int(event.xp_increase_censored),
        "xp_full_inferred_from_level_change": int(
            event.xp_full_inferred_from_level_change
        ),
        "hud_level_before": event.hud_level_before,
        "hud_level_after": event.hud_level_after,
        "player_anchor_a_x": round(anchor_a.x, 2),
        "player_anchor_a_y": round(anchor_a.y, 2),
        "player_anchor_b_x": round(anchor_b.x, 2),
        "player_anchor_b_y": round(anchor_b.y, 2),
        "health_bar_detected_a": int(anchor_a.health_bar_detected),
        "health_bar_detected_b": int(anchor_b.health_bar_detected),
        "health_bar_confidence_a": round(anchor_a.confidence, 4),
        "health_bar_confidence_b": round(anchor_b.confidence, 4),
        "magnet_model_enabled": int(magnet_state.model_enabled),
        "magnet_value": round(magnet_state.magnet_value, 6),
        "magnet_multiplier_from_base": round(
            magnet_state.multiplier_from_base,
            6,
        ),
        "attractorb_level": magnet_state.attractorb_level,
        "attractorb_multiplier": round(
            magnet_state.attractorb_multiplier,
            6,
        ),
        "magnet_search_radius_pixels": round(
            magnet_state.search_radius_1440p * scale,
            2,
        ),
        "magnet_state_source_event_id": magnet_state.source_event_id,
        "magnet_state_source_confidence": magnet_state.source_confidence,
        "magnet_state_needs_review": int(magnet_state.source_needs_review),
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
        "trajectory_start_magnet_ratios": "|".join(
            f"{item.start_distance_magnet_ratio:.3f}" for item in trajectories
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
    magnet_model: MagnetModel,
    save_previews: bool,
    track_magnet_entries: bool = True,
) -> tuple[list[dict[str, object]], list[MagnetEntryEvidence]]:
    if not events:
        return [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")
    frame_count = min(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(signal_arrays["pixels"]),
    )
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count_cfg = count_trajectory_config(cfg)
    entry_cfg = replace(
        cfg,
        template_threshold=cfg.magnet_entry_template_threshold,
    )
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    entry_tracker = MagnetEntryTracker(cfg, fps, frame_height)

    # A final signal sample can round onto the half-open frame-count boundary.
    # Clamp only that terminal reference to the last addressable video frame so
    # the event remains reviewable instead of failing the complete run.
    last_reliable_frame = max(0, frame_count - 1)
    events = [
        replace(
            event,
            frame_a=min(event.frame_a, last_reliable_frame),
            frame_b=min(event.frame_b, last_reliable_frame),
        )
        for event in events
    ]

    def window_indexes(event: XPEvent) -> range:
        # The A/B pair can span farther than the trajectory-history window at
        # EOF. Always retain frame A because pair classification requires it.
        start = max(0, min(event.frame_a, event.frame_b - cfg.trajectory_history_frames))
        end = min(last_reliable_frame, event.frame_b + cfg.trajectory_post_frames)
        start = min(start, end)
        return range(start, end + 1)

    events_by_end: dict[int, list[XPEvent]] = {}
    for event in events:
        events_by_end.setdefault(window_indexes(event).stop - 1, []).append(event)
    use_counts = Counter(index for event in events for index in window_indexes(event))
    analysis_needed = set(use_counts)
    entry_needed = (
        set(range(0, frame_count, cfg.magnet_entry_stride_frames))
        if track_magnet_entries and magnet_model.enabled
        else set()
    )
    needed = analysis_needed | entry_needed
    frames: dict[int, np.ndarray] = {}
    detection_cache: dict[int, list[GemDetection]] = {}
    count_detection_cache: dict[int, list[GemDetection]] = {}
    magnet_detection_cache: dict[int, list[GemDetection]] = {}
    magnet_count_detection_cache: dict[int, list[GemDetection]] = {}
    anchor_cache: dict[int, PlayerAnchor] = {}
    magnet_state_cache: dict[int, MagnetState] = {}
    magnet_search_radius_cache: dict[int, float] = {}
    claimed_trajectories: list[TrajectoryEvidence] = []
    claimed_count_trajectories: list[TrajectoryEvidence] = []
    claimed_magnet_trajectories: list[TrajectoryEvidence] = []
    claimed_magnet_count_trajectories: list[TrajectoryEvidence] = []
    rows: list[dict[str, object]] = []
    next_report = 100
    next_entry_report_frame = max(1, int(round(120.0 * fps)))

    for frame_index in range(frame_count):
        ok = cap.grab()
        if not ok:
            break
        if frame_index not in needed:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
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
        magnet_state = magnet_state_at(frame_index / fps, magnet_model, cfg)
        if frame_index in entry_needed:
            entry_detections = detect_gems_near_player(
                frame,
                templates,
                anchor,
                entry_cfg,
                magnet_state.search_radius_1440p
                + cfg.magnet_entry_outer_margin_1440p,
                enforce_search_circle=True,
            )
            entry_tracker.update(
                frame_index,
                entry_detections,
                anchor,
                magnet_state,
            )
            if frame_index >= next_entry_report_frame:
                print(
                    "Magnet-entry scan: "
                    f"{seconds_to_stamp(frame_index / fps)} / "
                    f"{seconds_to_stamp(frame_count / fps)}, "
                    f"{len(entry_tracker.entries)} committed tracks",
                    flush=True,
                )
                next_entry_report_frame += max(1, int(round(120.0 * fps)))
        if frame_index not in analysis_needed:
            continue
        frames[frame_index] = frame
        anchor_cache[frame_index] = anchor
        magnet_state_cache[frame_index] = magnet_state
        magnet_search_radius_cache[frame_index] = (
            magnet_state.search_radius_1440p * frame.shape[0] / 1440.0
        )
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
        if magnet_state.search_radius_1440p > cfg.search_radius_1440p:
            magnet_detection_cache[frame_index] = detect_gems_near_player(
                frame,
                templates,
                anchor,
                cfg,
                magnet_state.search_radius_1440p,
                enforce_search_circle=True,
            )
            magnet_count_detection_cache[frame_index] = detect_gems_near_player(
                frame,
                templates,
                anchor,
                count_cfg,
                magnet_state.search_radius_1440p,
                enforce_search_circle=True,
            )
        else:
            magnet_detection_cache[frame_index] = detection_cache[frame_index]
            magnet_count_detection_cache[frame_index] = count_detection_cache[
                frame_index
            ]

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
                magnet_search_radius_cache,
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
                    magnet_search_radius_cache,
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
                magnet_state_cache[event.frame_b],
                cfg,
            )
            rescue_attempted = (
                magnet_state_cache[event.frame_b].search_radius_1440p
                > cfg.search_radius_1440p
                and row["color_evidence"]
                in {
                    "xp_single_step_blue_fallback",
                    "xp_confirmed_color_ambiguous",
                }
            )
            rescue_applied = False
            expanded_trajectories: list[TrajectoryEvidence] = []
            expanded_count_trajectories: list[TrajectoryEvidence] = []
            preview_detection_cache = detection_cache
            preview_count_detection_cache = count_detection_cache
            if rescue_attempted:
                base_search_radius = cfg.search_radius_1440p * scale

                def starts_outside_base_search(
                    candidate: TrajectoryEvidence,
                ) -> bool:
                    start_frame, start_detection = candidate.points[0]
                    start_anchor = anchor_cache[start_frame]
                    return (
                        abs(start_detection.cx - start_anchor.x)
                        > base_search_radius
                        or abs(start_detection.cy - start_anchor.y)
                        > base_search_radius
                    )

                expanded_candidates = find_trajectory_evidence(
                    event,
                    frame_b.shape,
                    magnet_detection_cache,
                    anchor_cache,
                    magnet_search_radius_cache,
                    cfg,
                )
                for candidate in expanded_candidates:
                    if not starts_outside_base_search(candidate):
                        continue
                    candidate_end = candidate.points[-1][1]
                    already_claimed = any(
                        candidate.color == claimed.color
                        and candidate.disappearance_frame
                        == claimed.disappearance_frame
                        and math.hypot(
                            candidate_end.cx - claimed.points[-1][1].cx,
                            candidate_end.cy - claimed.points[-1][1].cy,
                        )
                        <= claim_distance
                        for claimed in (
                            claimed_trajectories
                            + claimed_magnet_trajectories
                        )
                    )
                    if not already_claimed:
                        expanded_trajectories.append(candidate)

                expanded_count_candidates = deduplicate_count_trajectories(
                    find_trajectory_evidence(
                        event,
                        frame_b.shape,
                        magnet_count_detection_cache,
                        anchor_cache,
                        magnet_search_radius_cache,
                        count_cfg,
                    ),
                    frame_b.shape,
                    cfg,
                )
                for candidate in expanded_count_candidates:
                    if not starts_outside_base_search(candidate):
                        continue
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
                        for claimed in (
                            claimed_count_trajectories
                            + claimed_magnet_count_trajectories
                        )
                    )
                    if not already_claimed:
                        expanded_count_trajectories.append(candidate)

                if expanded_trajectories or expanded_count_trajectories:
                    candidate_row, candidate_disappeared = summarize_pair(
                        event,
                        frame_b.shape,
                        detections_a,
                        detections_b,
                        expanded_trajectories,
                        expanded_count_trajectories,
                        anchor_cache[event.frame_a],
                        anchor_cache[event.frame_b],
                        magnet_state_cache[event.frame_b],
                        cfg,
                    )
                    visual_rescue = str(candidate_row["color_evidence"]).startswith(
                        "visual_"
                    )
                    stronger_count_evidence = len(expanded_count_trajectories) > len(
                        count_trajectories
                    )
                    if visual_rescue or stronger_count_evidence:
                        row = candidate_row
                        disappeared = candidate_disappeared
                        trajectories = expanded_trajectories
                        count_trajectories = expanded_count_trajectories
                        claimed_magnet_trajectories.extend(expanded_trajectories)
                        claimed_magnet_count_trajectories.extend(
                            expanded_count_trajectories
                        )
                        preview_detection_cache = magnet_detection_cache
                        preview_count_detection_cache = magnet_count_detection_cache
                        rescue_applied = True
            row["magnet_rescue_attempted"] = int(rescue_attempted)
            row["magnet_rescue_applied"] = int(rescue_applied)
            row["magnet_expanded_trajectory_count"] = len(
                expanded_trajectories
            )
            row["magnet_expanded_count_trajectory_count"] = len(
                expanded_count_trajectories
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
                        preview_detection_cache,
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
                        preview_count_detection_cache,
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
                    magnet_detection_cache.pop(used_index, None)
                    magnet_count_detection_cache.pop(used_index, None)
                    anchor_cache.pop(used_index, None)
                    magnet_state_cache.pop(used_index, None)
                    magnet_search_radius_cache.pop(used_index, None)
        if len(rows) >= next_report:
            print(f"Temporal analysis: {len(rows)}/{len(events)} events", flush=True)
            next_report += 100
        if len(rows) == len(events):
            break
    cap.release()
    if len(rows) != len(events):
        raise RuntimeError(f"Only analyzed {len(rows)} of {len(events)} flagged windows")
    return rows, entry_tracker.entries


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


def build_magnet_entry_rows(
    entries: list[MagnetEntryEvidence],
    fps: float,
) -> list[dict[str, object]]:
    return [
        {
            "track_id": entry.track_id,
            "color": entry.color,
            "entry_frame": entry.entry_frame,
            "entry_video_time": round(entry.entry_frame / fps, 4),
            "entry_time_stamp": seconds_to_stamp(
                entry.entry_frame / fps,
                include_milliseconds=True,
            ),
            "confirmation_frame": entry.confirmation_frame,
            "predicted_arrival_frame": entry.predicted_arrival_frame,
            "predicted_arrival_video_time": round(
                entry.predicted_arrival_frame / fps,
                4,
            ),
            "entry_reason": entry.entry_reason,
            "entry_x": round(entry.entry_x, 2),
            "entry_y": round(entry.entry_y, 2),
            "entry_distance": round(entry.entry_distance, 2),
            "magnet_radius": round(entry.magnet_radius, 2),
            "radial_speed_pixels_per_frame": round(
                entry.radial_speed_pixels_per_frame,
                3,
            ),
            "mean_template_score": round(entry.mean_template_score, 4),
            "monotonic_fraction": round(entry.monotonic_fraction, 4),
            "confidence": round(entry.confidence, 4),
            "attractorb_level": entry.attractorb_level,
            "magnet_value": round(entry.magnet_value, 6),
            "magnet_state_source_event_id": (
                entry.magnet_state_source_event_id
            ),
            "magnet_state_needs_review": int(
                entry.magnet_state_needs_review
            ),
        }
        for entry in entries
    ]


def build_magnet_likely_color_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build a compact audit table without implying exact gem quantity."""
    fields = (
        "event_id",
        "event_key",
        "video_time_stamp",
        "video_time_b",
        "frame_a",
        "frame_b",
        "hud_level",
        "xp_bar_increase_percent",
        "estimated_base_xp_gain",
        "xp_color_hint",
        "magnet_entry_candidate_count",
        "magnet_entry_candidate_blue",
        "magnet_entry_candidate_green",
        "magnet_entry_candidate_red",
        "magnet_entry_likely_color",
        "magnet_entry_likely_color_confidence",
        "magnet_entry_likely_color_reason",
        "magnet_entry_track_ids",
        "magnet_entry_frames",
        "magnet_entry_predicted_arrival_frames",
        "magnet_entry_arrival_residual_frames",
        "magnet_entry_min_confidence",
        "previous_xp_event_gap_frames",
        "next_xp_event_gap_frames",
        "magnet_value",
        "attractorb_level",
        "magnet_state_source_event_id",
        "magnet_state_needs_review",
        "unresolved_collected_gems",
    )
    result = []
    for row in rows:
        if not int(row.get("magnet_entry_likely_color_applied", 0)):
            continue
        audit_row = {field: row[field] for field in fields}
        audit_row["quantity_status"] = "unresolved"
        result.append(audit_row)
    return result


def pair_magnet_entries_to_events(
    rows: list[dict[str, object]],
    entries: list[MagnetEntryEvidence],
    cfg: Config,
) -> tuple[dict[int, list[MagnetEntryEvidence]], dict[str, int]]:
    """Assign each committed entry to at most one temporally compatible XP event."""
    ordered_rows = sorted(rows, key=lambda row: int(row["frame_b"]))
    assignments: dict[int, list[MagnetEntryEvidence]] = {}
    diagnostics = Counter()
    for entry in entries:
        if entry.confidence < cfg.magnet_entry_min_confidence:
            diagnostics["entry_below_confidence_gate"] += 1
            continue
        candidates: list[tuple[int, int, dict[str, object]]] = []
        for row in ordered_rows:
            event_frame = int(row["frame_b"])
            if event_frame < entry.confirmation_frame:
                continue
            lag = event_frame - entry.entry_frame
            if lag > cfg.magnet_entry_max_assignment_lag_frames:
                if event_frame > entry.predicted_arrival_frame:
                    break
                continue
            residual = abs(event_frame - entry.predicted_arrival_frame)
            if residual <= cfg.magnet_entry_arrival_tolerance_frames:
                candidates.append((residual, event_frame, row))
        if not candidates:
            diagnostics["entry_without_compatible_xp_event"] += 1
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        best = candidates[0]
        if (
            len(candidates) > 1
            and candidates[1][0] - best[0]
            < cfg.magnet_entry_assignment_margin_frames
        ):
            diagnostics["entry_with_ambiguous_xp_event"] += 1
            continue
        event_id = int(best[2]["event_id"])
        assignments.setdefault(event_id, []).append(entry)
        diagnostics["entry_assigned_to_xp_event"] += 1
    return assignments, dict(diagnostics)


def magnet_entry_validation_metric(
    rows: list[dict[str, object]],
    assignments: dict[int, list[MagnetEntryEvidence]],
    split: str,
    cfg: Config,
) -> dict[str, object]:
    reference_rows = [
        row
        for row in rows
        if split_name(row, cfg) == split
        and strict_visual_pickup_count(row) is not None
    ]
    predicted_events = 0
    predicted_gems = 0
    correct_color_gems = 0
    exact_composition_events = 0
    for row in reference_rows:
        entries = assignments.get(int(row["event_id"]), [])
        if not entries:
            continue
        predicted_events += 1
        predicted = Counter(entry.color for entry in entries)
        actual = Counter(
            {
                color: int(row[f"collected_{color}_gems"])
                for color in ("blue", "green", "red")
            }
        )
        predicted_gems += sum(predicted.values())
        correct_color_gems += sum(
            min(predicted[color], actual[color])
            for color in ("blue", "green", "red")
        )
        exact_composition_events += int(
            all(
                predicted[color] == actual[color]
                for color in ("blue", "green", "red")
            )
        )
    return {
        "split": split,
        "reference_events": len(reference_rows),
        "predicted_events": predicted_events,
        "predicted_gems": predicted_gems,
        "correct_color_gems": correct_color_gems,
        "color_precision": round(safe_rate(correct_color_gems, predicted_gems), 4),
        "exact_composition_events": exact_composition_events,
        "exact_composition_accuracy": round(
            safe_rate(exact_composition_events, predicted_events),
            4,
        ),
        "reference_event_coverage": round(
            safe_rate(predicted_events, len(reference_rows)),
            4,
        ),
    }


def magnet_entry_color_candidate(
    row: dict[str, object],
    entries: list[MagnetEntryEvidence],
    cfg: Config,
) -> str:
    """Return a unanimous Magnet-entry color corroborated by XP physics."""
    colors = {entry.color for entry in entries}
    if len(colors) != 1:
        return ""
    color = next(iter(colors))
    if str(row["xp_color_hint"]) != color:
        return ""
    if int(row["xp_bar_saturated"]) or int(row["xp_increase_censored"]):
        return ""
    if not int(row["hud_level_ocr_accepted"]):
        return ""
    if (
        int(row["previous_xp_event_gap_frames"])
        <= cfg.magnet_entry_event_isolation_frames
        or int(row["next_xp_event_gap_frames"])
        <= cfg.magnet_entry_event_isolation_frames
    ):
        return ""
    return color


def magnet_entry_color_validation_metric(
    rows: list[dict[str, object]],
    assignments: dict[int, list[MagnetEntryEvidence]],
    split: str,
    cfg: Config,
) -> dict[str, object]:
    """Evaluate event-level color without treating entry count as ground truth."""
    reference_rows = [
        row
        for row in rows
        if split_name(row, cfg) == split
        and strict_visual_pickup_count(row) is not None
    ]
    predicted_events = 0
    correct_color_events = 0
    for row in reference_rows:
        candidate = magnet_entry_color_candidate(
            row,
            assignments.get(int(row["event_id"]), []),
            cfg,
        )
        if not candidate:
            continue
        predicted_events += 1
        actual_colors = {
            color
            for color in ("blue", "green", "red")
            if int(row[f"collected_{color}_gems"])
        }
        correct_color_events += int(actual_colors == {candidate})
    return {
        "split": split,
        "reference_events": len(reference_rows),
        "predicted_events": predicted_events,
        "correct_color_events": correct_color_events,
        "color_precision": round(
            safe_rate(correct_color_events, predicted_events),
            4,
        ),
        "reference_event_coverage": round(
            safe_rate(predicted_events, len(reference_rows)),
            4,
        ),
    }


def apply_magnet_entry_assist(
    rows: list[dict[str, object]],
    entries: list[MagnetEntryEvidence],
    cfg: Config,
) -> dict[str, object]:
    """Resolve ambiguous XP events only after internal precision validation."""
    baseline_unresolved = sum(
        int(row["unresolved_collected_gems"])
        for row in rows
    )
    add_event_isolation(rows, cfg)
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
            row[f"pre_magnet_entry_{name}"] = row[name]
        row["magnet_entry_candidate_count"] = 0
        row["magnet_entry_candidate_blue"] = 0
        row["magnet_entry_candidate_green"] = 0
        row["magnet_entry_candidate_red"] = 0
        row["magnet_entry_track_ids"] = ""
        row["magnet_entry_frames"] = ""
        row["magnet_entry_predicted_arrival_frames"] = ""
        row["magnet_entry_arrival_residual_frames"] = ""
        row["magnet_entry_min_confidence"] = 0.0
        row["magnet_entry_resolution_applied"] = 0
        row["magnet_entry_gate_reason"] = ""
        row["magnet_entry_color_candidate"] = ""
        row["magnet_entry_likely_color"] = ""
        row["magnet_entry_likely_color_confidence"] = 0.0
        row["magnet_entry_likely_color_applied"] = 0
        row["magnet_entry_likely_color_reason"] = ""

    assignments, pairing_diagnostics = pair_magnet_entries_to_events(
        rows,
        entries,
        cfg,
    )
    metrics = {
        split: magnet_entry_validation_metric(rows, assignments, split, cfg)
        for split in ("calibration", "validation", "deployment")
    }
    color_metrics = {
        split: magnet_entry_color_validation_metric(
            rows,
            assignments,
            split,
            cfg,
        )
        for split in ("calibration", "validation", "deployment")
    }
    calibration = metrics["calibration"]
    validation = metrics["validation"]
    deployment = metrics["deployment"]
    gate_enabled = bool(
        int(calibration["predicted_events"])
        >= cfg.magnet_entry_validation_min_predictions
        and float(calibration["exact_composition_accuracy"])
        >= cfg.magnet_entry_target_precision
        and int(validation["predicted_events"])
        >= cfg.magnet_entry_validation_min_predictions
        and float(validation["exact_composition_accuracy"])
        >= cfg.magnet_entry_target_precision
    )
    deployment_gate_enabled = bool(
        gate_enabled
        and int(deployment["predicted_events"])
        >= cfg.magnet_entry_validation_min_predictions
        and float(deployment["exact_composition_accuracy"])
        >= cfg.magnet_entry_target_precision
    )
    color_calibration = color_metrics["calibration"]
    color_validation = color_metrics["validation"]
    color_deployment = color_metrics["deployment"]
    color_gate_enabled = bool(
        int(color_calibration["predicted_events"])
        >= cfg.magnet_entry_validation_min_predictions
        and float(color_calibration["color_precision"])
        >= cfg.magnet_entry_target_precision
        and int(color_validation["predicted_events"])
        >= cfg.magnet_entry_validation_min_predictions
        and float(color_validation["color_precision"])
        >= cfg.magnet_entry_target_precision
    )
    color_deployment_gate_enabled = bool(
        color_gate_enabled
        and int(color_deployment["predicted_events"])
        >= cfg.magnet_entry_validation_min_predictions
        and float(color_deployment["color_precision"])
        >= cfg.magnet_entry_target_precision
    )

    reclassified = Counter()
    likely_colors = Counter()
    for row in rows:
        event_entries = assignments.get(int(row["event_id"]), [])
        if not event_entries:
            row["magnet_entry_gate_reason"] = "no_unique_entry_assignment"
            continue
        counts = Counter(entry.color for entry in event_entries)
        row["magnet_entry_candidate_count"] = len(event_entries)
        for color in ("blue", "green", "red"):
            row[f"magnet_entry_candidate_{color}"] = counts[color]
        row["magnet_entry_track_ids"] = "|".join(
            str(entry.track_id) for entry in event_entries
        )
        row["magnet_entry_frames"] = "|".join(
            str(entry.entry_frame) for entry in event_entries
        )
        row["magnet_entry_predicted_arrival_frames"] = "|".join(
            str(entry.predicted_arrival_frame) for entry in event_entries
        )
        row["magnet_entry_arrival_residual_frames"] = "|".join(
            str(abs(int(row["frame_b"]) - entry.predicted_arrival_frame))
            for entry in event_entries
        )
        row["magnet_entry_min_confidence"] = round(
            min(entry.confidence for entry in event_entries),
            4,
        )
        color_candidate = magnet_entry_color_candidate(
            row,
            event_entries,
            cfg,
        )
        row["magnet_entry_color_candidate"] = color_candidate
        if not int(row["pre_magnet_entry_unresolved_collected_gems"]):
            row["magnet_entry_likely_color_reason"] = (
                "baseline_already_classified"
            )
        elif not color_candidate:
            row["magnet_entry_likely_color_reason"] = (
                "entry_color_not_unanimous_or_xp_disagrees"
            )
        elif not color_gate_enabled:
            row["magnet_entry_likely_color_reason"] = (
                "calibration_or_validation_color_gate_failed"
            )
        elif (
            split_name(row, cfg) == "deployment"
            and not color_deployment_gate_enabled
        ):
            row["magnet_entry_likely_color_reason"] = (
                "deployment_color_audit_gate_failed"
            )
        elif any(entry.magnet_state_needs_review for entry in event_entries):
            row["magnet_entry_likely_color_reason"] = (
                "magnet_state_requires_review"
            )
        else:
            row["magnet_entry_likely_color"] = color_candidate
            row["magnet_entry_likely_color_confidence"] = round(
                min(
                    float(color_validation["color_precision"]),
                    min(entry.confidence for entry in event_entries),
                ),
                4,
            )
            row["magnet_entry_likely_color_applied"] = 1
            row["magnet_entry_likely_color_reason"] = (
                "unanimous_entry_color_and_xp_agreement"
            )
            likely_colors[color_candidate] += 1
        if not int(row["pre_magnet_entry_unresolved_collected_gems"]):
            row["magnet_entry_gate_reason"] = "baseline_already_classified"
            continue
        if not gate_enabled:
            row["magnet_entry_gate_reason"] = "calibration_or_validation_gate_failed"
            continue
        if split_name(row, cfg) == "deployment" and not deployment_gate_enabled:
            row["magnet_entry_gate_reason"] = "deployment_audit_gate_failed"
            continue

        for color in ("blue", "green", "red"):
            row[f"collected_{color}_gems"] = counts[color]
        row["unresolved_collected_gems"] = 0
        row["collected_gems_total"] = len(event_entries)
        row["color_evidence"] = "magnet_entry_queue"
        row["confidence"] = round(
            min(
                float(validation["color_precision"]),
                min(entry.confidence for entry in event_entries),
            ),
            4,
        )
        row["magnet_entry_resolution_applied"] = 1
        row["magnet_entry_gate_reason"] = "passed_internal_precision_gate"
        prior_reason = str(row["review_reason"])
        added_reason = "magnet_entry_queue_assignment"
        row["review_reason"] = (
            f"{prior_reason}|{added_reason}" if prior_reason else added_reason
        )
        row["needs_review"] = int(
            any(entry.magnet_state_needs_review for entry in event_entries)
        )
        reclassified.update(counts)

    final_unresolved = sum(
        int(row["unresolved_collected_gems"])
        for row in rows
    )
    return {
        "method": "stateful_magnet_entry_queue",
        "human_labels_used": False,
        "entry_stride_frames": cfg.magnet_entry_stride_frames,
        "color_event_isolation_frames": (
            cfg.magnet_entry_event_isolation_frames
        ),
        "committed_entry_tracks": len(entries),
        "pairing_diagnostics": pairing_diagnostics,
        "composition_validation_metrics": metrics,
        "validation_metrics": metrics,
        "color_validation_metrics": color_metrics,
        "target_precision": cfg.magnet_entry_target_precision,
        "minimum_validation_predictions": (
            cfg.magnet_entry_validation_min_predictions
        ),
        "gate_enabled": gate_enabled,
        "deployment_gate_enabled": deployment_gate_enabled,
        "color_gate_enabled": color_gate_enabled,
        "color_deployment_gate_enabled": color_deployment_gate_enabled,
        "baseline_unresolved_events": baseline_unresolved,
        "resolved_events": sum(
            int(row["magnet_entry_resolution_applied"])
            for row in rows
        ),
        "resolved_gems_by_color": dict(reclassified),
        "likely_color_events": int(sum(likely_colors.values())),
        "likely_color_events_by_color": dict(likely_colors),
        "final_unresolved_events": final_unresolved,
    }


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
        attractorb_levels = [int(row["attractorb_level"]) for row in rows]
        magnet_values = [float(row["magnet_value"]) for row in rows]
        increases = [float(row["xp_bar_increase_percent"]) for row in rows]
        result.append(
            {
                "time_stamp": interval_stamp(start, end),
                "interval_start_second": start,
                "interval_end_second": end,
                "xp_jump_events": len(rows),
                "hud_level_min": min(levels) if levels else "",
                "hud_level_max": max(levels) if levels else "",
                "attractorb_level_min": (
                    min(attractorb_levels) if attractorb_levels else ""
                ),
                "attractorb_level_max": (
                    max(attractorb_levels) if attractorb_levels else ""
                ),
                "magnet_value_min": (
                    round(min(magnet_values), 6) if magnet_values else ""
                ),
                "magnet_value_max": (
                    round(max(magnet_values), 6) if magnet_values else ""
                ),
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
                "events_with_review_required_magnet_state": sum(
                    int(row["magnet_state_needs_review"]) for row in rows
                ),
                "magnet_rescue_attempted_events": sum(
                    int(row["magnet_rescue_attempted"]) for row in rows
                ),
                "magnet_rescue_applied_events": sum(
                    int(row["magnet_rescue_applied"]) for row in rows
                ),
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
    magnet_model: MagnetModel,
    cfg: Config,
) -> None:
    events_by_frame = {event.frame_b: event for event in events}
    reset_set = set(resets)
    fps = metadata["fps"]
    bar_width = float(metadata["bar_width"])
    fieldnames = [
        "frame_index",
        "video_time",
        "xp_measurement_valid",
        "xp_measurement_rejection_reason",
        "xp_progress",
        "xp_progress_percent",
        "xp_endpoint_pixels",
        "xp_delta_pixels",
        "xp_delta_percent_of_full_bar",
        "xp_progress_raw",
        "xp_progress_raw_percent",
        "xp_endpoint_pixels_raw",
        "xp_delta_pixels_raw",
        "xp_delta_raw_percent_of_full_bar",
        "xp_quality",
        "overlay_score",
        "gameplay_hud_score",
        "player_anchor_x",
        "player_anchor_y",
        "health_bar_x",
        "health_bar_y",
        "health_bar_detected",
        "health_bar_confidence",
        "magnet_value",
        "magnet_multiplier_from_base",
        "attractorb_level",
        "magnet_search_radius_pixels",
        "magnet_state_source_event_id",
        "magnet_state_needs_review",
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
            magnet_state = magnet_state_at(frame_index / fps, magnet_model, cfg)
            frame_height = float(metadata["height"])
            transition_guarded = bool(
                arrays["level_up_transition_guard"][frame_index]
            )
            bar_reappearance = bool(arrays["xp_bar_reappearance"][frame_index])
            measurement_valid = not transition_guarded and not bar_reappearance
            rejection_reason = (
                "level_up_transition_guard"
                if transition_guarded
                else ("xp_bar_reappearance" if bar_reappearance else "")
            )
            raw_progress = float(arrays["progress"][frame_index])
            raw_pixels = int(arrays["pixels"][frame_index])
            raw_delta = int(arrays["delta_pixels"][frame_index])
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "video_time": round(frame_index / fps, 4),
                    "xp_measurement_valid": int(measurement_valid),
                    "xp_measurement_rejection_reason": rejection_reason,
                    "xp_progress": round(raw_progress, 7) if measurement_valid else "",
                    "xp_progress_percent": (
                        round(100.0 * raw_progress, 4) if measurement_valid else ""
                    ),
                    "xp_endpoint_pixels": raw_pixels if measurement_valid else "",
                    "xp_delta_pixels": raw_delta if measurement_valid else "",
                    "xp_delta_percent_of_full_bar": (
                        round(100.0 * raw_delta / bar_width, 4)
                        if measurement_valid
                        else ""
                    ),
                    "xp_progress_raw": round(raw_progress, 7),
                    "xp_progress_raw_percent": round(100.0 * raw_progress, 4),
                    "xp_endpoint_pixels_raw": raw_pixels,
                    "xp_delta_pixels_raw": raw_delta,
                    "xp_delta_raw_percent_of_full_bar": round(
                        100.0 * raw_delta / bar_width,
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
                    "magnet_value": round(magnet_state.magnet_value, 6),
                    "magnet_multiplier_from_base": round(
                        magnet_state.multiplier_from_base,
                        6,
                    ),
                    "attractorb_level": magnet_state.attractorb_level,
                    "magnet_search_radius_pixels": round(
                        magnet_state.search_radius_1440p * frame_height / 1440.0,
                        2,
                    ),
                    "magnet_state_source_event_id": magnet_state.source_event_id,
                    "magnet_state_needs_review": int(
                        magnet_state.source_needs_review
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
    cfg: Config,
    magnet_model: MagnetModel,
    run_execution_seconds: float,
    magnet_entry_report: dict[str, object] | None = None,
    percentage_report: dict[str, object] | None = None,
) -> dict[str, object]:
    visual_rows = [row for row in rows if str(row["color_evidence"]).startswith("visual_")]
    ocr_rows = [row for row in rows if row["hud_level_source"] == "hud_ocr"]
    summary = {
        "author": "Tahereh Fahi",
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
        "pickup_radius_1440p": cfg.pickup_radius_1440p,
        "trajectory_min_strong_history_frames": cfg.trajectory_min_strong_history_frames,
        "trajectory_max_occluded_endpoint_frames": cfg.trajectory_max_occluded_endpoint_frames,
        "level_ocr_min_confidence": cfg.level_ocr_min_confidence,
        "level_ocr_max_reset_lead": cfg.level_ocr_max_reset_lead,
        "level_ocr_max_reset_lag": cfg.level_ocr_max_reset_lag,
        "magnet_model_enabled": magnet_model.enabled,
        "magnet_base_value": cfg.magnet_base_value,
        "magnet_powerup_rank": magnet_model.powerup_rank,
        "magnet_character_multiplier": magnet_model.character_multiplier,
        "magnet_golden_egg_bonus": magnet_model.golden_egg_bonus,
        "magnet_timeline_source": magnet_model.timeline_source,
        "magnet_search_radius_base_1440p": cfg.search_radius_1440p,
        "magnet_search_radius_cap_1440p": cfg.magnet_search_radius_cap_1440p,
        "attractorb_changes": [
            {
                "video_second": change.video_second,
                "level": change.level,
                "source_event_id": change.source_event_id,
                "confidence": change.confidence,
                "needs_review": change.needs_review,
            }
            for change in magnet_model.attractorb_changes
        ],
        "review_required_attractorb_changes": sum(
            change.needs_review for change in magnet_model.attractorb_changes
        ),
        "magnet_rescue_attempted_events": sum(
            int(row["magnet_rescue_attempted"]) for row in rows
        ),
        "magnet_rescue_applied_events": sum(
            int(row["magnet_rescue_applied"]) for row in rows
        ),
        "magnet_entry_tracking_enabled": magnet_entry_report is not None,
        "magnet_entry_committed_tracks": (
            int(magnet_entry_report["committed_entry_tracks"])
            if magnet_entry_report is not None
            else 0
        ),
        "magnet_entry_gate_enabled": (
            bool(magnet_entry_report["gate_enabled"])
            if magnet_entry_report is not None
            else False
        ),
        "magnet_entry_deployment_gate_enabled": (
            bool(magnet_entry_report["deployment_gate_enabled"])
            if magnet_entry_report is not None
            else False
        ),
        "magnet_entry_color_gate_enabled": (
            bool(magnet_entry_report["color_gate_enabled"])
            if magnet_entry_report is not None
            else False
        ),
        "magnet_entry_color_deployment_gate_enabled": (
            bool(magnet_entry_report["color_deployment_gate_enabled"])
            if magnet_entry_report is not None
            else False
        ),
        "magnet_entry_resolved_events": (
            int(magnet_entry_report["resolved_events"])
            if magnet_entry_report is not None
            else 0
        ),
        "magnet_entry_likely_color_events": (
            int(magnet_entry_report["likely_color_events"])
            if magnet_entry_report is not None
            else 0
        ),
        "magnet_entry_likely_color_events_by_color": (
            magnet_entry_report["likely_color_events_by_color"]
            if magnet_entry_report is not None
            else {}
        ),
        "magnet_entry_unresolved_before": (
            int(magnet_entry_report["baseline_unresolved_events"])
            if magnet_entry_report is not None
            else None
        ),
        "magnet_entry_unresolved_after": (
            int(magnet_entry_report["final_unresolved_events"])
            if magnet_entry_report is not None
            else None
        ),
        "human_labels_used": False,
        "audio_used": False,
        "timing_summary": {
            "prompt_workflow_elapsed_time": "not recorded",
            "notebook_run_execution_time": seconds_to_duration(
                run_execution_seconds
            ),
        },
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
        "**Author:** Tahereh Fahi",
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
        "The Magnet-aware rescue pass uses the exact multiplicative Attractorb progression documented",
        "by the Vampire Survivors Wiki. It preserves the original fixed-radius decision and examines an",
        "expanded circular search region only when the event remains visually unresolved or relies on an",
        "XP-only fallback. A rescue is valid only when its trajectory begins outside the original search",
        "square, preventing distant detections from rearranging already-resolved nearby associations.",
        "Sources: [Magnet](https://vampire.survivors.wiki/w/Magnet) and",
        "[Attractorb](https://vampire.survivors.wiki/w/Attractorb).",
        "",
        "A separate stateful Magnet-entry pass samples the video every two frames. It locks a gem's",
        "color only after a high-score track crosses the estimated Magnet boundary and remains inside",
        "for two observations, or after a gem first seen inside shows sustained inward motion. The track",
        "then becomes a pending pickup; its radial speed predicts an arrival window, and it can be linked",
        "to at most one XP event. It is rejected when two nearby XP events are similarly plausible.",
        "A likely-color decision additionally requires more than three frames of separation from both",
        "the preceding and following XP events, so a pending entry is not attached inside a dense burst.",
        "Because video pixels are only an estimate of the game's Magnet statistic, two validation gates",
        "are kept separate. A unanimous entry color corroborated by the XP band can become a validated",
        "likely color after calibration and validation each reach at least 20 predictions and 95% event-",
        "level color precision. Exact counts require a separate 95% composition gate. A likely color does",
        "not overwrite the official color counts or remove quantity uncertainty. Deployment repeats both",
        "audits before either output can be used after the first five minutes.",
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
        f"- Magnet-aware trajectory search enabled: {summary['magnet_model_enabled']}",
        f"- Magnet timeline source: {summary['magnet_timeline_source'] or 'configured base state only'}",
        f"- Attractorb state changes: {len(summary['attractorb_changes'])}",
        f"- Attractorb changes requiring review: {summary['review_required_attractorb_changes']}",
        f"- Magnet rescue attempts: {summary['magnet_rescue_attempted_events']}",
        f"- Magnet rescues applied: {summary['magnet_rescue_applied_events']}",
        f"- Stateful Magnet-entry tracks committed: {summary['magnet_entry_committed_tracks']}",
        f"- Magnet-entry exact-composition gate enabled: {summary['magnet_entry_gate_enabled']}",
        f"- Magnet-entry exact-composition deployment gate enabled: {summary['magnet_entry_deployment_gate_enabled']}",
        f"- Magnet-entry likely-color gate enabled: {summary['magnet_entry_color_gate_enabled']}",
        f"- Magnet-entry likely-color deployment gate enabled: {summary['magnet_entry_color_deployment_gate_enabled']}",
        f"- Unresolved events receiving a validated likely color: {summary['magnet_entry_likely_color_events']}",
        f"- Validated likely colors by class: {summary['magnet_entry_likely_color_events_by_color']}",
        f"- Events resolved by the Magnet-entry queue: {summary['magnet_entry_resolved_events']}",
        f"- Unresolved pickups before/after Magnet-entry assistance: "
        f"{summary['magnet_entry_unresolved_before']}/{summary['magnet_entry_unresolved_after']}",
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
        "## Timing Summary",
        "",
        "| Timing measure | Duration |",
        "|---|---:|",
        f"| Prompt/workflow elapsed time | {summary['timing_summary']['prompt_workflow_elapsed_time']} |",
        f"| Notebook/run execution time | {summary['timing_summary']['notebook_run_execution_time']} |",
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
        "--initial-level",
        type=int,
        default=1,
        help="HUD level at the beginning of this video segment (default: 1).",
    )
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
    parser.add_argument(
        "--inventory-events",
        type=Path,
        default=None,
        help=(
            "Inventory-event CSV providing Attractorb acquisition and upgrade "
            "times. A curated video_second/attractorb_level CSV is also accepted."
        ),
    )
    parser.add_argument(
        "--magnet-powerup-rank",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Permanent Magnet PowerUp rank active for the run.",
    )
    parser.add_argument(
        "--magnet-character-multiplier",
        type=float,
        default=1.0,
        help="Character-specific multiplicative Magnet modifier.",
    )
    parser.add_argument(
        "--magnet-golden-egg-bonus",
        type=float,
        default=0.0,
        help="Absolute Magnet points added by Golden Eggs before multipliers.",
    )
    parser.add_argument(
        "--exclude-review-required-magnet-events",
        action="store_true",
        help="Ignore Attractorb timeline rows whose needs_review field is true.",
    )
    parser.add_argument(
        "--disable-magnet-aware-search",
        action="store_true",
        help="Record Magnet state but retain the original fixed search radius.",
    )
    parser.add_argument(
        "--disable-magnet-entry-assist",
        action="store_true",
        help=(
            "Disable stateful Magnet-boundary tracking and pending-entry "
            "association with XP events."
        ),
    )
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    cfg = replace(
        Config(),
        initial_level=args.initial_level,
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
    if cfg.initial_level < 1:
        raise ValueError("Initial level must be at least 1.")
    if args.magnet_character_multiplier <= 0:
        raise ValueError("Magnet character multiplier must be positive.")
    if args.magnet_golden_egg_bonus < 0:
        raise ValueError("Magnet Golden Egg bonus cannot be negative.")
    video_path = args.video.expanduser().resolve()
    template_dir = args.template_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = (
        args.inventory_events.expanduser().resolve()
        if args.inventory_events is not None
        else None
    )
    attractorb_changes = load_attractorb_changes(
        inventory_path,
        include_review_required=(
            not args.exclude_review_required_magnet_events
        ),
    )
    magnet_model = MagnetModel(
        enabled=not args.disable_magnet_aware_search,
        attractorb_changes=attractorb_changes,
        powerup_rank=args.magnet_powerup_rank,
        character_multiplier=args.magnet_character_multiplier,
        golden_egg_bonus=args.magnet_golden_egg_bonus,
        timeline_source=inventory_path.name if inventory_path else "",
    )
    print(
        "Magnet model: "
        f"enabled={magnet_model.enabled}, "
        f"PowerUp rank={magnet_model.powerup_rank}, "
        f"Attractorb changes={len(attractorb_changes)}",
        flush=True,
    )

    metadata, arrays = scan_xp_signal(video_path, cfg, args.max_seconds)
    resets = find_level_resets(arrays, cfg)
    events = find_xp_events(metadata, arrays, resets, cfg)
    validate_transition_event_invariants(events, arrays, resets)
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
    rows, magnet_entries = analyze_flagged_windows(
        video_path,
        events,
        templates,
        arrays,
        output_dir,
        cfg,
        magnet_model,
        save_previews=not args.no_previews,
        track_magnet_entries=not args.disable_magnet_entry_assist,
    )
    percentage_report: dict[str, object] | None = None
    percentage_metrics: list[dict[str, object]] = []
    if not args.disable_percentage_assist:
        percentage_report, percentage_metrics = (
            apply_video_only_percentage_calibration(rows, cfg)
        )
    magnet_entry_report: dict[str, object] | None = None
    if not args.disable_magnet_entry_assist:
        magnet_entry_report = apply_magnet_entry_assist(
            rows,
            magnet_entries,
            cfg,
        )

    label = clean_video_label(video_path)
    event_path = output_dir / f"collected_gems_{label}_xp_ab_events.csv"
    per_second_path = output_dir / f"collected_gems_{label}_per_second.csv"
    five_second_path = output_dir / f"collected_gems_{label}_5sec_intervals.csv"
    diagnostics_path = output_dir / f"collected_gems_{label}_xp_frame_signal.csv"
    level_pattern_path = output_dir / f"collected_gems_{label}_xp_level_patterns.csv"
    magnet_entry_path = output_dir / f"collected_gems_{label}_magnet_entries.csv"
    magnet_likely_color_path = (
        output_dir / f"collected_gems_{label}_magnet_likely_colors.csv"
    )
    percentage_metrics_path = (
        output_dir / f"collected_gems_{label}_video_only_validation.csv"
    )
    write_csv(event_path, rows)
    write_csv(per_second_path, build_aggregate_rows(rows, metadata["duration"], 1))
    write_csv(five_second_path, build_aggregate_rows(rows, metadata["duration"], 5))
    write_csv(level_pattern_path, build_level_pattern_rows(rows))
    if magnet_entry_report is not None:
        write_csv(
            magnet_entry_path,
            build_magnet_entry_rows(magnet_entries, metadata["fps"]),
        )
        write_csv(
            magnet_likely_color_path,
            build_magnet_likely_color_rows(rows),
        )
        (output_dir / "magnet_entry_validation_report.json").write_text(
            json.dumps(magnet_entry_report, indent=2) + "\n"
        )
    if percentage_report is not None:
        write_csv(percentage_metrics_path, percentage_metrics)
        (output_dir / "video_only_xp_calibration_report.json").write_text(
            json.dumps(percentage_report, indent=2) + "\n"
        )
    write_signal_diagnostics(
        diagnostics_path,
        metadata,
        arrays,
        events,
        resets,
        magnet_model,
        cfg,
    )
    run_execution_seconds = time.perf_counter() - run_started
    summary = write_summary(
        output_dir / "summary.json",
        video_path,
        metadata,
        rows,
        resets,
        args.template_profile,
        arrays,
        cfg,
        magnet_model,
        run_execution_seconds,
        magnet_entry_report,
        percentage_report,
    )
    write_method_note(output_dir / "METHOD_AND_RESULTS.md", summary)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Event CSV: {event_path}", flush=True)
    print(f"Level-pattern CSV: {level_pattern_path}", flush=True)
    if magnet_entry_report is not None:
        print(f"Magnet-entry CSV: {magnet_entry_path}", flush=True)
        print(
            f"Magnet likely-color CSV: {magnet_likely_color_path}",
            flush=True,
        )
    print(f"5-second CSV: {five_second_path}", flush=True)
    if percentage_report is not None:
        print(
            "Video-only calibration report: "
            f"{output_dir / 'video_only_xp_calibration_report.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
