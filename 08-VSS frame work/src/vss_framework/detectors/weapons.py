"""Detect and record Vampire Survivors weapons shown in the top-left HUD.

The detector uses the weapon-icon catalogue from the Vampire Survivors Wiki.
It is intentionally conservative: every prediction includes a score, a margin
over the runner-up, a confidence label, and a ``needs_review`` flag.

The main outputs are:

* ``weapon_timeline.csv`` -- one row per sampled video time.
* ``weapon_events.csv`` -- initial/acquired/changed/removed slot events.
* ``debug_frames`` -- optional frames annotated with the detected HUD slots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from ..gameplay_state import gameplay_pause_evidence


WIKI_CATEGORY_URL = (
    "https://vampire-survivors.fandom.com/wiki/Category:Weapon_icons"
)
WIKI_API_URL = "https://vampire-survivors.fandom.com/api.php"
USER_AGENT = "UCSB-Vampire-Survivors-Weapon-Research/1.0"


@dataclass(frozen=True)
class HUDLayout:
    """Pixel geometry for the six weapon slots in the first HUD row."""

    x: int
    y: int
    slot_size: int
    step_x: int
    slots: int = 6
    bar_x: int = 0
    bar_y: int = 0
    bar_width: int = 0
    bar_height: int = 0


@dataclass(frozen=True)
class MatchConfig:
    """Matching and confidence thresholds."""

    template_inner_margin: float = 0.18
    template_background_distance: float = 32.0
    color_histogram_weight: float = 0.22
    search_padding_fraction: float = 0.055
    occupied_luma_std: float = 24.0
    occupied_saturation_std: float = 28.0
    occupied_laplacian_var: float = 800.0
    high_score: float = 0.60
    high_margin: float = 0.07
    medium_score: float = 0.48
    medium_margin: float = 0.025
    unknown_score: float = 0.38


@dataclass(frozen=True)
class PreparedReference:
    name: str
    filename: str
    image: np.ndarray
    mask: np.ndarray
    histogram: np.ndarray


def clean_video_label(path: str | Path) -> str:
    """Return a filesystem-friendly video label."""

    stem = Path(path).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def _safe_icon_filename(name: str) -> str:
    clean = re.sub(r"[\\/:*?\"<>|]+", "-", name).strip()
    return clean.replace(" ", "_") + ".png"


def _wiki_catalogue_rows() -> list[dict[str, str]]:
    """Read every file in the wiki's ``Weapon icons`` category."""

    rows: list[dict[str, str]] = []
    continuation: str | None = None

    while True:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": "Category:Weapon_icons",
            "gcmtype": "file",
            "gcmlimit": "max",
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
        if continuation:
            params["gcmcontinue"] = continuation

        request = urllib.request.Request(
            WIKI_API_URL + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)

        for page in payload.get("query", {}).get("pages", {}).values():
            title = page["title"]
            name = re.sub(r"^File:Icon-", "", title, flags=re.IGNORECASE)
            name = re.sub(r"\.png$", "", name, flags=re.IGNORECASE)
            rows.append(
                {
                    "weapon_name": name,
                    "filename": _safe_icon_filename(name),
                    "wiki_image_url": page["imageinfo"][0]["url"],
                }
            )

        continuation = payload.get("continue", {}).get("gcmcontinue")
        if not continuation:
            break

    return sorted(rows, key=lambda row: row["weapon_name"].casefold())


def download_wiki_weapon_icons(
    icon_dir: str | Path,
    manifest_path: str | Path,
    *,
    force: bool = False,
    delay_seconds: float = 0.05,
) -> pd.DataFrame:
    """Download all wiki weapon icons and write a reproducible manifest.

    Fandom may return WebP bytes for a URL ending in ``.png``. OpenCV decodes
    those bytes and rewrites a genuine PNG so downstream programs do not have
    to guess the file format.
    """

    icon_dir = Path(icon_dir)
    manifest_path = Path(manifest_path)
    icon_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _wiki_catalogue_rows()
    for index, row in enumerate(rows, start=1):
        output_path = icon_dir / row["filename"]
        if output_path.exists() and not force:
            continue

        request = urllib.request.Request(
            row["wiki_image_url"], headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = np.frombuffer(response.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not decode wiki icon {row['weapon_name']}")
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not write {output_path}")

        if delay_seconds and index < len(rows):
            time.sleep(delay_seconds)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["weapon_name", "filename", "wiki_image_url"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return pd.DataFrame(rows)


def load_weapon_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"weapon_name", "filename", "wiki_image_url"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return manifest.sort_values("weapon_name", key=lambda x: x.str.casefold())


def _largest_gold_bar_component(frame: np.ndarray, *, allow_fallback: bool = True) -> tuple[int, int, int, int]:
    """Locate the gold outline surrounding the long bar at screen top."""

    height, width = frame.shape[:2]
    top = frame[: max(100, int(round(height * 0.08)))]
    blue, green, red = cv2.split(top)
    mask = (
        (red > 130)
        & (green > 65)
        & (red > green.astype(np.float32) * 1.05)
        & (green > blue.astype(np.float32) * 1.20)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3)),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[tuple[int, int, int, int, int]] = []
    for component in range(1, count):
        x, y, w, h, area = map(int, stats[component])
        if w >= width * 0.45 and 18 <= h <= height * 0.08:
            candidates.append((x, y, w, h, area))

    if not candidates:
        if not allow_fallback:
            raise ValueError("No observed XP-bar border")
        # The empty XP bar can leave only disconnected one-pixel gold edges,
        # especially in pillar-boxed 16:9 captures. The game's viewport in
        # these recordings occupies the centered 90% of the frame and the top
        # bar is 2.5% of frame height. Keep this as a deterministic fallback;
        # downstream icon confidence still decides whether inventory is usable.
        margin_x = int(round(width * 0.05))
        return (
            margin_x,
            0,
            max(1, width - 2 * margin_x),
            max(18, int(round(height * 0.025))),
        )
    x, y, w, h, _ = max(candidates, key=lambda item: (item[2], item[4]))
    return x, y, w, h


def locate_weapon_hud(frame: np.ndarray) -> HUDLayout:
    """Infer the six-slot weapon row from the top gold bar."""

    bar_x, bar_y, bar_width, bar_height = _largest_gold_bar_component(frame)
    # A level-up/status overlay touches the top bar and can make its connected
    # component look artificially tall. Video height supplies an independent
    # upper bound for the game's UI scale. The 0.96 factor is calibrated from
    # the 1080p, 1440p, and 1800p project recordings.
    scale = min(bar_height / 35.0, 0.96 * frame.shape[0] / 1080.0)
    inferred_bar_height = int(round(35 * scale))
    return HUDLayout(
        x=max(0, int(round(bar_x + 4 * scale))),
        y=max(0, int(round(bar_y + inferred_bar_height + 2 * scale))),
        slot_size=max(24, int(round(44 * scale))),
        step_x=max(25, int(round(46 * scale))),
        slots=6,
        bar_x=bar_x,
        bar_y=bar_y,
        bar_width=bar_width,
        bar_height=bar_height,
    )


def crop_weapon_slots(frame: np.ndarray, layout: HUDLayout) -> list[np.ndarray]:
    """Crop all six first-row HUD slots."""

    height, width = frame.shape[:2]
    crops: list[np.ndarray] = []
    for slot_index in range(layout.slots):
        x1 = layout.x + slot_index * layout.step_x
        y1 = layout.y
        x2 = min(width, x1 + layout.slot_size)
        y2 = min(height, y1 + layout.slot_size)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[:2] != (layout.slot_size, layout.slot_size):
            raise RuntimeError(
                f"Weapon slot {slot_index + 1} lies outside the video frame."
            )
        crops.append(crop)
    return crops


def _slot_visual_features(slot: np.ndarray) -> dict[str, float]:
    size = slot.shape[0]
    margin = max(2, int(round(size * 0.15)))
    center = slot[margin:-margin, margin:-margin]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    return {
        "luma_std": float(gray.std()),
        "saturation_std": float(hsv[:, :, 1].std()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }


def occupied_weapon_slots(
    slot_crops: Sequence[np.ndarray], config: MatchConfig
) -> tuple[list[bool], list[dict[str, float]]]:
    """Return contiguous occupied slots and their diagnostic features."""

    features = [_slot_visual_features(slot) for slot in slot_crops]
    visually_active = [
        (
            item["luma_std"] >= config.occupied_luma_std
            or item["saturation_std"] >= config.occupied_saturation_std
            or item["laplacian_var"] >= config.occupied_laplacian_var
        )
        for item in features
    ]

    # Weapons are filled from left to right. Taking every slot through the last
    # active one is more robust than independently dropping a dark, low-contrast
    # weapon in the middle of the row.
    occupied_count = 0
    for index, active in enumerate(visually_active):
        if active:
            occupied_count = index + 1
    occupied = [index < occupied_count for index in range(len(slot_crops))]
    return occupied, features


def _mode_color(image: np.ndarray) -> np.ndarray:
    pixels = image.reshape(-1, image.shape[-1])[:, :3]
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return colors[int(np.argmax(counts))].astype(np.float32)


def _color_histogram(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], mask, [12, 4, 4], [0, 180, 0, 256, 0, 256]
    ).reshape(-1)
    histogram /= np.linalg.norm(histogram) + 1e-9
    return histogram


def _prepare_reference(
    name: str,
    filename: str,
    icon_path: Path,
    slot_size: int,
    config: MatchConfig,
) -> PreparedReference:
    icon = cv2.imread(str(icon_path), cv2.IMREAD_COLOR)
    if icon is None:
        raise FileNotFoundError(icon_path)

    height, width = icon.shape[:2]
    margin = config.template_inner_margin
    y1, y2 = int(margin * height), int((1 - margin) * height)
    x1, x2 = int(margin * width), int((1 - margin) * width)
    inner = icon[y1:y2, x1:x2]
    background = _mode_color(inner)
    distance = np.linalg.norm(inner.astype(np.float32) - background, axis=2)
    mask = (distance > config.template_background_distance).astype(np.uint8) * 255

    interpolation = cv2.INTER_NEAREST
    resized = cv2.resize(inner, (slot_size, slot_size), interpolation=interpolation)
    resized_mask = cv2.resize(
        mask, (slot_size, slot_size), interpolation=interpolation
    )
    return PreparedReference(
        name=name,
        filename=filename,
        image=resized,
        mask=resized_mask,
        histogram=_color_histogram(resized, resized_mask),
    )


def prepare_reference_library(
    icon_dir: str | Path,
    manifest: pd.DataFrame,
    slot_size: int,
    config: MatchConfig,
) -> list[PreparedReference]:
    icon_dir = Path(icon_dir)
    references: list[PreparedReference] = []
    for row in manifest.itertuples(index=False):
        icon_path = icon_dir / row.filename
        if not icon_path.exists():
            continue
        references.append(
            _prepare_reference(
                row.weapon_name,
                row.filename,
                icon_path,
                slot_size,
                config,
            )
        )
    if not references:
        raise FileNotFoundError(f"No readable weapon icons found in {icon_dir}")
    return references


def _patch_foreground_mask(slot: np.ndarray) -> np.ndarray:
    height, width = slot.shape[:2]
    depth = max(2, height // 8)
    ring = np.concatenate(
        [
            slot[:depth].reshape(-1, 3),
            slot[-depth:].reshape(-1, 3),
            slot[:, :depth].reshape(-1, 3),
            slot[:, -depth:].reshape(-1, 3),
        ]
    )
    quantized = (ring // 24).astype(np.int16)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    background = (colors[int(np.argmax(counts))] * 24 + 12).astype(np.float32)
    distance = np.linalg.norm(slot.astype(np.float32) - background, axis=2)
    mask = (distance > 45).astype(np.uint8) * 255
    border = max(2, int(round(height * 0.08)))
    mask[:border] = 0
    mask[-border:] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def _aligned_pixel_score(
    patch: np.ndarray, template: np.ndarray, mask: np.ndarray
) -> float:
    selected = mask > 0
    if int(selected.sum()) < 5:
        return -1.0
    observed = patch[selected].astype(np.float32)
    expected = template[selected].astype(np.float32)

    mae = float(np.mean(np.abs(observed - expected)) / 255.0)
    observed_centered = observed - observed.mean(axis=0)
    expected_centered = expected - expected.mean(axis=0)
    correlation = float(
        np.sum(observed_centered * expected_centered)
        / (
            math.sqrt(
                float(np.sum(observed_centered**2))
                * float(np.sum(expected_centered**2))
            )
            + 1e-9
        )
    )
    cosine = float(
        np.sum(observed * expected)
        / (
            math.sqrt(float(np.sum(observed**2)) * float(np.sum(expected**2)))
            + 1e-9
        )
    )
    return 0.40 * correlation + 0.30 * cosine + 0.30 * (1.0 - mae)


def _score_reference(
    slot: np.ndarray,
    reference: PreparedReference,
    slot_histogram: np.ndarray,
    config: MatchConfig,
) -> float:
    padding = max(2, int(round(slot.shape[0] * config.search_padding_fraction)))
    padded = cv2.copyMakeBorder(
        slot, padding, padding, padding, padding, cv2.BORDER_REFLECT
    )
    correlation_map = cv2.matchTemplate(
        padded,
        reference.image,
        cv2.TM_CCORR_NORMED,
        mask=reference.mask,
    )
    _, _, _, best_location = cv2.minMaxLoc(correlation_map)
    x, y = best_location
    aligned = padded[
        y : y + reference.image.shape[0],
        x : x + reference.image.shape[1],
    ]
    pixel_score = _aligned_pixel_score(aligned, reference.image, reference.mask)
    histogram_score = float(np.dot(slot_histogram, reference.histogram))
    hist_weight = config.color_histogram_weight
    return (1.0 - hist_weight) * pixel_score + hist_weight * histogram_score


def _confidence_label(score: float, margin: float, config: MatchConfig) -> str:
    if score >= config.high_score and margin >= config.high_margin:
        return "high"
    if score >= config.medium_score and margin >= config.medium_margin:
        return "medium"
    return "low"


def match_weapon_slot(
    slot: np.ndarray,
    references: Sequence[PreparedReference],
    config: MatchConfig,
) -> dict[str, object]:
    foreground_mask = _patch_foreground_mask(slot)
    slot_histogram = _color_histogram(slot, foreground_mask)
    ranked = sorted(
        (
            (_score_reference(slot, reference, slot_histogram, config), reference.name)
            for reference in references
        ),
        reverse=True,
    )
    best_score, best_name = ranked[0]
    second_score, second_name = ranked[1] if len(ranked) > 1 else (-1.0, "")
    margin = best_score - second_score
    confidence = _confidence_label(best_score, margin, config)
    display_name = best_name if best_score >= config.unknown_score else "UNKNOWN"
    return {
        "weapon": display_name,
        "suggested_weapon": best_name,
        "match_score": float(best_score),
        "runner_up": second_name,
        "runner_up_score": float(second_score),
        "score_margin": float(margin),
        "confidence": confidence,
        "needs_review": confidence == "low",
    }


def analyze_weapon_frame(
    frame: np.ndarray,
    icon_dir: str | Path,
    manifest: pd.DataFrame,
    *,
    layout: HUDLayout | None = None,
    config: MatchConfig | None = None,
    references: Sequence[PreparedReference] | None = None,
) -> tuple[pd.DataFrame, HUDLayout, list[PreparedReference]]:
    """Identify every occupied weapon slot in one BGR frame."""

    config = config or MatchConfig()
    layout = layout or locate_weapon_hud(frame)
    slot_crops = crop_weapon_slots(frame, layout)
    occupied, features = occupied_weapon_slots(slot_crops, config)
    prepared = list(references) if references is not None else prepare_reference_library(
        icon_dir, manifest, layout.slot_size, config
    )

    rows: list[dict[str, object]] = []
    for index, (slot, is_occupied, diagnostics) in enumerate(
        zip(slot_crops, occupied, features), start=1
    ):
        if is_occupied:
            result = match_weapon_slot(slot, prepared, config)
        else:
            result = {
                "weapon": "",
                "suggested_weapon": "",
                "match_score": np.nan,
                "runner_up": "",
                "runner_up_score": np.nan,
                "score_margin": np.nan,
                "confidence": "empty",
                "needs_review": False,
            }
        rows.append(
            {
                "slot": index,
                "occupied": is_occupied,
                **result,
                **diagnostics,
            }
        )
    return pd.DataFrame(rows), layout, prepared


def annotate_weapon_frame(
    frame: np.ndarray,
    detections: pd.DataFrame,
    layout: HUDLayout,
    timestamp_seconds: float | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    for row in detections.itertuples(index=False):
        x = layout.x + (int(row.slot) - 1) * layout.step_x
        y = layout.y
        if not row.occupied:
            color = (130, 130, 130)
            label = f"{row.slot}: empty"
        elif row.confidence == "high":
            color = (80, 220, 80)
            label = f"{row.slot}: {row.weapon}"
        elif row.confidence == "medium":
            color = (0, 200, 255)
            label = f"{row.slot}: {row.weapon}"
        else:
            color = (0, 80, 255)
            label = f"{row.slot}: {row.suggested_weapon}?"
        cv2.rectangle(
            annotated,
            (x, y),
            (x + layout.slot_size, y + layout.slot_size),
            color,
            max(2, layout.slot_size // 20),
        )
        cv2.putText(
            annotated,
            label,
            (x, y + layout.slot_size + max(18, layout.slot_size // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.38, layout.slot_size / 105.0),
            color,
            max(1, layout.slot_size // 28),
            cv2.LINE_AA,
        )

    if timestamp_seconds is not None:
        cv2.putText(
            annotated,
            f"video second {timestamp_seconds:.1f}",
            (layout.x, layout.y + layout.slot_size * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, layout.slot_size / 90.0),
            (255, 255, 255),
            max(1, layout.slot_size // 25),
            cv2.LINE_AA,
        )
    return annotated


def read_temporal_median(
    capture: cv2.VideoCapture,
    second: float,
    offsets: Sequence[float] = (-0.24, 0.0, 0.24),
    *,
    start_second: float = 0.0,
    end_second: float | None = None,
) -> np.ndarray:
    """Median readable gameplay frames without admitting paused HUD panels."""

    frames: list[np.ndarray] = []
    for offset in offsets:
        target_second = second + float(offset)
        if target_second < start_second or (
            end_second is not None and target_second >= end_second
        ):
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, target_second * 1000.0)
        ok, frame = capture.read()
        if ok and not has_large_menu_overlay(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"Could not read frames around video second {second}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def has_large_menu_overlay(frame: np.ndarray) -> bool:
    """Detect the gold-framed level-up/status panels that obscure gameplay.

    Empty HUD cells are translucent. On a level-up frame, the frozen scene and
    status-panel background can therefore look like extra icons. Large gold
    rectangles away from the top bar provide a reliable overlay signal without
    OCR or language-specific text detection.
    """

    # The central panel starts small and rotates before its final layout. Its
    # shared pause evidence must gate the HUD before large rectangles exist.
    if gameplay_pause_evidence(frame)["blocked"]:
        return True

    height, width = frame.shape[:2]
    blue, green, red = cv2.split(frame)
    gold = (
        (red > 130)
        & (green > 65)
        & (red > green.astype(np.float32) * 1.05)
        & (green > blue.astype(np.float32) * 1.20)
    ).astype(np.uint8) * 255
    gold = cv2.morphologyEx(
        gold,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    contours, _ = cv2.findContours(gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            y >= height * 0.04
            and w >= width * 0.12
            and h >= height * 0.25
        ):
            return True
    return False


def read_nearest_gameplay_frame(
    capture: cv2.VideoCapture,
    second: float,
    *,
    duration: float,
    max_search_seconds: float = 5.0,
    search_step: float = 0.5,
) -> tuple[np.ndarray, float, str]:
    """Read the nearest clean gameplay frame, preferring post-overlay frames."""

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second) * 1000.0)
    exact_ok, exact_frame = capture.read()
    if exact_ok and not has_large_menu_overlay(exact_frame):
        return exact_frame, second, "gameplay"

    # After a modal closes, wait long enough for its left status panel and the
    # frozen scene to clear. Otherwise translucent empty slots can inherit menu
    # graphics and look occupied.
    offsets: list[float] = []
    distance = 2.0
    while distance <= max_search_seconds + 1e-9:
        offsets.extend([distance, -distance])
        distance += search_step

    fallback: tuple[np.ndarray, float] | None = (
        (exact_frame, second) if exact_ok else None
    )
    for offset in offsets:
        candidate_second = min(max(0.0, second + offset), max(0.0, duration - 1e-3))
        capture.set(cv2.CAP_PROP_POS_MSEC, candidate_second * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        if fallback is None:
            fallback = (frame, candidate_second)
        if not has_large_menu_overlay(frame):
            return frame, candidate_second, "gameplay_near_overlay"

    if fallback is None:
        raise RuntimeError(f"Could not read a frame near video second {second}")
    return fallback[0], fallback[1], "overlay_unresolved"


def _format_timestamp(seconds: float) -> str:
    whole = int(round(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def _wide_timeline_row(
    video_name: str,
    second: float,
    detections: pd.DataFrame,
    *,
    frame_second_used: float | None = None,
    screen_state: str = "gameplay",
) -> dict[str, object]:
    row: dict[str, object] = {
        "video": video_name,
        "video_second": float(second),
        "time_stamp": _format_timestamp(second),
        "frame_second_used": float(
            second if frame_second_used is None else frame_second_used
        ),
        "screen_state": screen_state,
        "weapon_count": int(detections["occupied"].sum()),
        "review_count": int(detections["needs_review"].sum()),
    }
    for detection in detections.itertuples(index=False):
        slot = int(detection.slot)
        row[f"weapon_{slot}"] = detection.weapon
        row[f"suggested_weapon_{slot}"] = detection.suggested_weapon
        row[f"confidence_{slot}"] = detection.confidence
        row[f"match_score_{slot}"] = detection.match_score
        row[f"score_margin_{slot}"] = detection.score_margin
        row[f"needs_review_{slot}"] = bool(detection.needs_review)
    return row


def build_weapon_events(timeline: pd.DataFrame, slots: int = 6) -> pd.DataFrame:
    """Convert the wide timeline into a compact slot-change table."""

    events: list[dict[str, object]] = []
    previous = [""] * slots
    for sample_index, row in enumerate(timeline.itertuples(index=False)):
        for slot in range(1, slots + 1):
            current = str(getattr(row, f"weapon_{slot}") or "")
            old = previous[slot - 1]
            if current == old:
                continue
            if not old and current:
                event = "initial" if sample_index == 0 else "acquired"
            elif old and not current:
                event = "removed"
            else:
                event = "changed_or_evolved"
            events.append(
                {
                    "video": row.video,
                    "video_second": row.video_second,
                    "time_stamp": row.time_stamp,
                    "slot": slot,
                    "event": event,
                    "previous_weapon": old,
                    "weapon": current,
                    "suggested_weapon": getattr(row, f"suggested_weapon_{slot}"),
                    "confidence": getattr(row, f"confidence_{slot}"),
                    "match_score": getattr(row, f"match_score_{slot}"),
                    "score_margin": getattr(row, f"score_margin_{slot}"),
                    "needs_review": getattr(row, f"needs_review_{slot}"),
                }
            )
            previous[slot - 1] = current
    return pd.DataFrame(events)


def stabilize_weapon_timeline(
    timeline: pd.DataFrame,
    *,
    slots: int = 6,
    lookahead_samples: int = 3,
    required_support: int = 2,
) -> pd.DataFrame:
    """Remove isolated translucent-slot false positives.

    A real weapon persists in its slot until it evolves, while enemies, coins,
    and effects showing through an empty cell usually produce inconsistent
    one-sample guesses. The raw columns are retained for audit and manual review.
    """

    stable = timeline.copy()
    if stable.empty:
        return stable

    metric_names = [
        "weapon",
        "suggested_weapon",
        "confidence",
        "match_score",
        "score_margin",
        "needs_review",
    ]
    for slot in range(1, slots + 1):
        for metric in metric_names:
            source = f"{metric}_{slot}"
            stable[f"raw_{source}"] = stable[source]

        raw_names = stable[f"raw_weapon_{slot}"].fillna("").astype(str).tolist()
        raw_suggestions = (
            stable[f"raw_suggested_weapon_{slot}"].fillna("").astype(str).tolist()
        )
        sample_count = len(raw_names)

        def support_at(index: int, name: str) -> int:
            stop = min(sample_count, index + lookahead_samples)
            return sum(candidate == name for candidate in raw_names[index:stop])

        stable_start: int | None = None
        for index, name in enumerate(raw_names):
            if name and support_at(index, name) >= required_support:
                stable_start = index
                break

        if stable_start is None and sample_count == 1:
            if raw_names[0] and stable.at[0, f"raw_confidence_{slot}"] == "high":
                stable_start = 0

        current_name = ""
        current_suggestion = ""
        current_metrics: dict[str, object] = {}
        carried_flags: list[bool] = []

        for index in range(sample_count):
            candidate = raw_names[index]
            candidate_suggestion = raw_suggestions[index]
            accepted_here = False

            if stable_start is not None and index >= stable_start:
                if candidate and (
                    candidate == current_name
                    or support_at(index, candidate) >= required_support
                ):
                    current_name = candidate
                    current_suggestion = candidate_suggestion or candidate
                    current_metrics = {
                        metric: stable.at[index, f"raw_{metric}_{slot}"]
                        for metric in metric_names[2:]
                    }
                    accepted_here = True

            if not current_name:
                stable.at[index, f"weapon_{slot}"] = ""
                stable.at[index, f"suggested_weapon_{slot}"] = ""
                stable.at[index, f"confidence_{slot}"] = "empty"
                stable.at[index, f"match_score_{slot}"] = np.nan
                stable.at[index, f"score_margin_{slot}"] = np.nan
                stable.at[index, f"needs_review_{slot}"] = False
                carried_flags.append(False)
                continue

            stable.at[index, f"weapon_{slot}"] = current_name
            stable.at[index, f"suggested_weapon_{slot}"] = current_suggestion
            for metric, value in current_metrics.items():
                stable.at[index, f"{metric}_{slot}"] = value
            carried_flags.append(not accepted_here)

        stable[f"stabilized_carried_{slot}"] = carried_flags

    stable["weapon_count"] = sum(
        stable[f"weapon_{slot}"].fillna("").ne("").astype(int)
        for slot in range(1, slots + 1)
    )
    stable["review_count"] = sum(
        stable[f"needs_review_{slot}"].fillna(False).astype(bool).astype(int)
        for slot in range(1, slots + 1)
    )
    return stable


def record_video_weapons(
    video_path: str | Path,
    output_dir: str | Path,
    icon_dir: str | Path,
    manifest_path: str | Path,
    *,
    start_second: float = 0.0,
    end_second: float | None = None,
    sample_interval: float = 5.0,
    temporal_offsets: Sequence[float] = (-0.24, 0.0, 0.24),
    debug_every_seconds: float | None = 30.0,
    overlay_search_seconds: float = 5.0,
    manual_layout: HUDLayout | None = None,
    config: MatchConfig | None = None,
) -> tuple[Path, Path, Path]:
    """Record all visible HUD weapons across a video."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    if debug_every_seconds is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_weapon_manifest(manifest_path)
    config = config or MatchConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    stop = min(duration, end_second if end_second is not None else duration)
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    timeline_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    layout = manual_layout
    references: list[PreparedReference] | None = None
    next_debug = float(start_second)

    sample_times = np.arange(start_second, stop + 1e-9, sample_interval)
    try:
        for second in sample_times:
            if second >= stop:
                continue
            capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            pause = gameplay_pause_evidence(frame)
            if has_large_menu_overlay(frame):
                excluded_rows.append({
                    "video": video_path.name,
                    "video_second": float(second),
                    "frame_number": max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1),
                    "screen_state": pause["phase"] if pause["blocked"] else "overlay_unresolved",
                    "reason": pause["reason"] if pause["blocked"] else "large_gold_panel",
                })
                continue
            # A post-menu HUD belongs to its actual resumed video time. Never
            # substitute that observation into an earlier paused sample.
            frame_second_used = float(second)
            screen_state = "gameplay"
            if tuple(temporal_offsets) != (0.0,):
                try:
                    frame = read_temporal_median(
                        capture, frame_second_used, temporal_offsets,
                        start_second=start_second, end_second=stop,
                    )
                except RuntimeError:
                    # The exact frame above is already a clean observation.
                    pass
            detections, layout, references = analyze_weapon_frame(
                frame,
                icon_dir,
                manifest,
                layout=layout,
                config=config,
                references=references,
            )
            timeline_rows.append(
                _wide_timeline_row(
                    video_path.name,
                    float(second),
                    detections,
                    frame_second_used=frame_second_used,
                    screen_state=screen_state,
                )
            )

            if debug_every_seconds is not None and second + 1e-9 >= next_debug:
                annotated = annotate_weapon_frame(frame, detections, layout, float(second))
                debug_path = debug_dir / f"debug_{int(round(second)):05d}s.jpg"
                cv2.imwrite(str(debug_path), annotated)
                next_debug += debug_every_seconds
    finally:
        capture.release()

    timeline = stabilize_weapon_timeline(pd.DataFrame(timeline_rows))
    events = build_weapon_events(timeline)
    timeline_path = output_dir / "weapon_timeline.csv"
    events_path = output_dir / "weapon_events.csv"
    pd.DataFrame(excluded_rows, columns=[
        "video", "video_second", "frame_number", "screen_state", "reason",
    ]).to_csv(output_dir / "weapon_excluded_samples.csv", index=False)
    if timeline.empty:
        raise RuntimeError("No readable gameplay weapon samples in requested scope")
    timeline.to_csv(timeline_path, index=False)
    events.to_csv(events_path, index=False)
    return timeline_path, events_path, debug_dir


def _parse_layout(text: str | None) -> HUDLayout | None:
    if not text:
        return None
    values = [int(value.strip()) for value in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Layout must be x,y,slot_size,step_x")
    return HUDLayout(x=values[0], y=values[1], slot_size=values[2], step_x=values[3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--icon-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--debug-every", type=float, default=30.0)
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help="Optional manual x,y,slot_size,step_x override",
    )
    args = parser.parse_args()

    timeline, events, debug_dir = record_video_weapons(
        video_path=args.video,
        output_dir=args.output_dir,
        icon_dir=args.icon_dir,
        manifest_path=args.manifest,
        start_second=args.start,
        end_second=args.end,
        sample_interval=args.interval,
        debug_every_seconds=args.debug_every,
        manual_layout=_parse_layout(args.layout),
    )
    print("Timeline:", timeline)
    print("Events:", events)
    print("Debug frames:", debug_dir)


if __name__ == "__main__":
    main()
