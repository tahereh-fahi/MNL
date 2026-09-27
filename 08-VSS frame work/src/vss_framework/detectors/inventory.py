#!/usr/bin/env python3
"""Record Vampire Survivors weapon and passive-item inventory events.

The recorder combines four video-only signals:

* the first readable gameplay inventory supplies the initial state;
* the blue-gem XP output supplies bounded windows containing level-up menus;
* the final highlighted option in each menu supplies the selected item.
* an optional verified chest audit supplies ordinary chest rewards and
  evolutions that do not appear in Level Up menus.

The output uses the approved Google Sheet event schema.  Low-confidence icon
matches are retained as suggestions and explicitly flagged for review.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from ..gameplay_state import level_up_pause_evidence

from .weapons import (
    HUDLayout,
    MatchConfig,
    PreparedReference,
    analyze_weapon_frame,
    locate_weapon_hud,
    match_weapon_slot,
    prepare_reference_library,
    read_nearest_gameplay_frame,
)


EVENT_COLUMNS = [
    "event_id",
    "video",
    "video_second",
    "frame_number",
    "character_level",
    "event_source",
    "event_type",
    "item_type",
    "slot",
    "item_before",
    "item_after",
    "level_before",
    "level_after",
    "normal_max_level",
    "suggested_item",
    "confidence",
    "needs_review",
    "observed_at_level",
    "inference_method",
]


def _assign_inventory_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    """Assign session-specific IDs instead of leaking a video-specific prefix."""

    output = events.drop(columns=["event_id"], errors="ignore").copy()
    video_names = output.get("video", pd.Series(dtype=object)).dropna().astype(str)
    video_name = next((name.strip() for name in video_names if name.strip()), "video")
    safe_video_name = re.sub(r"[^A-Za-z0-9_]+", "_", video_name).strip("_") or "video"
    output.insert(
        0,
        "event_id",
        [
            f"{safe_video_name}_inventory_{index:04d}"
            for index in range(1, len(output) + 1)
        ],
    )
    return output

CURSOR_MARGIN_THRESHOLD = 20.0
SUPPORTED_MENU_OPTION_COUNTS = (4, 3, 2, 1)
MENU_ICON_CONTINUITY_THRESHOLD = 0.45

EVOLUTION_BASE_BY_NAME = {
    "Bloody Tear": "Whip",
    "Holy Wand": "Magic Wand",
    "Thousand Edge": "Knife",
    "Death Spiral": "Axe",
    "Heaven Sword": "Cross",
    "Unholy Vespers": "King Bible",
    "Hellfire": "Fire Wand",
    "Soul Eater": "Garlic",
    "La Borra": "Santa Water",
    "NO FUTURE": "Runetracer",
    "Thunder Loop": "Lightning Ring",
    "Gorgeous Moon": "Pentagram",
    "Mannajja": "Song of Mana",
    "Infinite Corridor": "Clock Lancet",
    "Crimson Shroud": "Laurel",
}

NORMAL_MAX_LEVEL_BY_ITEM = {
    "Magic Wand": 8,
    "Holy Wand": 1,
    "King Bible": 8,
    "Peachone": 8,
    "Lightning Ring": 8,
    "Pentagram": 8,
    "Gorgeous Moon": 1,
    "Ebony Wings": 8,
    "Duplicator": 2,
    "Crown": 5,
    "Empty Tome": 5,
    "Armor": 5,
    "Spinach": 5,
    "Pummarola": 5,
    "Attractorb": 5,
}


@dataclass(frozen=True)
class MenuObservation:
    """One sampled frame containing a one- to four-choice menu."""

    frame_number: int
    video_second: float
    selected_index: int
    cursor_score: float
    item_name: str
    suggested_item: str
    item_type: str
    confidence: str
    match_score: float
    score_margin: float
    needs_review: bool
    option_count: int


@dataclass(frozen=True)
class InventorySnapshotEntry:
    """One item and its visible level in a level-up-menu inventory panel."""

    item_type: str
    slot: int
    item_name: str
    level: int
    confidence: str


@dataclass(frozen=True)
class RetrospectiveInventoryChange:
    """A unique inventory change revealed by the following level-up menu."""

    item_type: str
    slot: int
    item_name: str
    level_before: int | None
    level_after: int
    event_type: str


@dataclass(frozen=True)
class MenuSegment:
    """The final observation from one contiguous level-up menu."""

    start_second: float
    end_second: float
    observation: MenuObservation
    inventory_snapshot: tuple[InventorySnapshotEntry, ...] = ()


def _gold_mask(frame: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    blue, green, red = cv2.split(frame)
    mask = (
        (red > 130)
        & (green > 65)
        & (red > green.astype(np.float32) * 1.05)
        & (green > blue.astype(np.float32) * 1.20)
    ).astype(np.uint8) * 255
    if kernel_size > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_size, kernel_size)
            ),
        )
    return mask


def _inside(inner: Sequence[int], outer: Sequence[int]) -> bool:
    ix, iy, iw, ih = map(int, inner)
    ox, oy, ow, oh = map(int, outer)
    return (
        ix >= ox
        and iy >= oy
        and ix + iw <= ox + ow
        and iy + ih <= oy + oh
    )


def _deduplicate_rectangles(
    rectangles: Iterable[Sequence[int]], tolerance: int = 12
) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for rectangle in sorted(
        (tuple(map(int, item)) for item in rectangles),
        key=lambda item: (item[1], item[0], -item[2] * item[3]),
    ):
        if any(
            abs(rectangle[0] - other[0]) <= tolerance
            and abs(rectangle[1] - other[1]) <= tolerance
            and abs(rectangle[2] - other[2]) <= tolerance
            and abs(rectangle[3] - other[3]) <= tolerance
            for other in result
        ):
            continue
        result.append(rectangle)
    return result


def level_up_option_rectangles(
    frame: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Return the one to four gold-framed rows in a level-up menu.

    Late-game menus can contain only one or two remaining item choices.
    Treating three choices as the minimum silently truncated Video 4 at level
    58 even though valid smaller menus continued afterward.
    """

    height, width = frame.shape[:2]
    mask = _gold_mask(frame)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [cv2.boundingRect(contour) for contour in contours]
    outer_candidates = [
        box
        for box in boxes
        if width * 0.25 <= box[2] <= width * 0.55
        and box[3] >= height * 0.55
        and box[1] >= height * 0.03
        and abs((box[0] + box[2] / 2.0) - width / 2.0) <= width * 0.15
    ]
    for outer in sorted(outer_candidates, key=lambda box: box[2] * box[3], reverse=True):
        ox, oy, ow, oh = outer
        rows = [
            box
            for box in boxes
            if _inside(box, outer)
            and box != outer
            and box[2] >= ow * 0.72
            and oh * 0.11 <= box[3] <= oh * 0.23
            and box[2] / max(1, box[3]) >= 3.0
            and box[1] >= oy + oh * 0.12
        ]
        rows = _deduplicate_rectangles(rows)
        if len(rows) < 1:
            continue
        rows = sorted(rows, key=lambda box: box[1])
        for option_count in SUPPORTED_MENU_OPTION_COUNTS:
            if len(rows) < option_count:
                continue
            best_group: list[tuple[int, int, int, int]] | None = None
            best_error = float("inf")
            for start in range(len(rows) - option_count + 1):
                group = rows[start : start + option_count]
                gaps = [
                    group[index + 1][1] - group[index][1]
                    for index in range(option_count - 1)
                ]
                heights = [box[3] for box in group]
                gap_error = float(np.std(gaps)) if gaps else 0.0
                error = gap_error + float(np.std(heights))
                if error < best_error:
                    best_error = error
                    best_group = group
            if best_group is not None and best_error <= height * 0.025:
                return best_group
    return []


def selected_option_index(
    frame: np.ndarray, option_rows: Sequence[Sequence[int]]
) -> tuple[int, float]:
    """Identify the option row bracketed by the two pale-gold pointers."""

    blue, green, red = cv2.split(frame)
    cursor_color = (
        (red >= 220)
        & (green >= 190)
        & (blue >= 120)
        & (blue <= 230)
        & ((red.astype(np.int16) - blue.astype(np.int16)) >= 25)
    ).astype(np.uint8)
    scores: list[float] = []
    for x, y, width, height in option_rows:
        inset = max(4, int(round(width * 0.02)))
        reach = max(inset + 1, int(round(width * 0.12)))
        y0 = max(0, y + int(round(height * 0.12)))
        y1 = min(frame.shape[0], y + int(round(height * 0.88)))
        left = cursor_color[
            y0:y1,
            max(0, x - reach) : max(1, x - inset),
        ]
        right = cursor_color[
            y0:y1,
            min(frame.shape[1] - 1, x + width + inset) : min(
                frame.shape[1], x + width + reach
            ),
        ]
        scores.append(float(min(left.sum(), right.sum())))
    selected = int(np.argmax(scores))
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    return selected, margin


def stable_level_up_option_rectangles(
    frame: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Read choices only after the shared pause classifier sees a stable menu."""
    if level_up_pause_evidence(frame)["phase"] != "level_up_menu":
        return []
    return level_up_option_rectangles(frame)


def crop_option_icon(frame: np.ndarray, option_row: Sequence[int]) -> np.ndarray:
    """Crop the black icon interior from a level-up option row."""

    x, y, width, height = map(int, option_row)
    side = max(24, int(round(height * 0.34)))
    x0 = x + int(round(width * 0.025))
    y0 = y + int(round(height * 0.075))
    x1 = min(frame.shape[1], x0 + side)
    y1 = min(frame.shape[0], y0 + side)
    crop = frame[y0:y1, x0:x1]
    if crop.shape[:2] != (side, side):
        raise RuntimeError("Level-up option icon lies outside the frame")
    return crop


def menu_icon_signatures(
    frame: np.ndarray, option_rows: Sequence[Sequence[int]]
) -> list[np.ndarray]:
    """Build compact signatures that expose a stacked-menu content change."""

    signatures: list[np.ndarray] = []
    for option_row in option_rows:
        icon = crop_option_icon(frame, option_row)
        mask = (np.max(icon, axis=2) > 16).astype(np.uint8) * 255
        signatures.append(_sprite_histogram(icon, mask))
    return signatures


def menu_content_changed(
    previous: Sequence[np.ndarray], current: Sequence[np.ndarray]
) -> bool:
    """Return true when directly stacked menus change their option icons."""

    if len(previous) != len(current):
        return True
    similarities = [float(left @ right) for left, right in zip(previous, current)]
    return bool(similarities) and min(similarities) < MENU_ICON_CONTINUITY_THRESHOLD


def load_item_manifests(
    weapon_manifest_path: Path, passive_manifest_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    weapons = pd.read_csv(weapon_manifest_path)
    passives = pd.read_csv(passive_manifest_path).rename(
        columns={"item_name": "weapon_name"}
    )
    item_types = {
        **{str(name): "weapon" for name in weapons["weapon_name"]},
        **{str(name): "passive_item" for name in passives["weapon_name"]},
    }
    return weapons, passives, item_types


def inventory_level_pip_counts(
    frame: np.ndarray, layout: HUDLayout
) -> list[list[int]]:
    """Count the small gold level pips under the weapon and passive rows.

    The expanded level-up/status panel lays these pips out in horizontal rows.
    We detect their repeated component geometry rather than relying on a fixed
    pixel crop, so the same rule scales with the HUD.
    """

    mask = _gold_mask(frame, kernel_size=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    scale = float(layout.slot_size)
    x_min = layout.x
    x_max = layout.x + (layout.slots - 1) * layout.step_x + layout.slot_size
    y_min = layout.y + int(round(layout.slot_size * 1.25))
    y_max = min(frame.shape[0], layout.y + int(round(layout.slot_size * 6.0)))
    components: list[tuple[float, float]] = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        if not (x_min <= center_x <= x_max and y_min <= center_y <= y_max):
            continue
        if not (scale * 0.14 <= width <= scale * 0.36):
            continue
        if not (scale * 0.11 <= height <= scale * 0.32):
            continue
        if area < scale * scale * 0.014:
            continue
        components.append((center_x, center_y))

    y_rows: list[list[tuple[float, float]]] = []
    for point in sorted(components, key=lambda value: value[1]):
        if not y_rows or abs(point[1] - np.mean([item[1] for item in y_rows[-1]])) > scale * 0.10:
            y_rows.append([point])
        else:
            y_rows[-1].append(point)
    bands: list[list[list[tuple[float, float]]]] = []
    for row in y_rows:
        row_y = float(np.mean([item[1] for item in row]))
        if not bands:
            bands.append([row])
            continue
        previous_y = float(np.mean([item[1] for item in bands[-1][-1]]))
        if row_y - previous_y <= scale * 0.45:
            bands[-1].append(row)
        else:
            bands.append([row])

    bands = [
        band
        for band in bands
        if sum(len(row) for row in band) >= 3
        and any(len(row) >= 2 for row in band)
    ]
    selected_bands: list[list[list[tuple[float, float]]]] = []
    for target_y in (
        layout.y + layout.slot_size * 2.1,
        layout.y + layout.slot_size * 4.25,
    ):
        remaining = [band for band in bands if band not in selected_bands]
        if not remaining:
            break
        candidate = min(
            remaining,
            key=lambda band: abs(
                float(np.mean([item[1] for row in band for item in row])) - target_y
            ),
        )
        candidate_y = float(np.mean([item[1] for row in candidate for item in row]))
        if abs(candidate_y - target_y) <= layout.slot_size * 0.8:
            selected_bands.append(candidate)
    bands = selected_bands
    result: list[list[int]] = []
    for band in bands[:2]:
        levels = [0] * layout.slots
        for row in band:
            for center_x, _ in row:
                slot = int(round((center_x - (layout.x + layout.slot_size / 2.0)) / layout.step_x))
                if 0 <= slot < layout.slots:
                    levels[slot] += 1
        result.append(levels)
    return result


def _menu_inventory_snapshot(
    frame: np.ndarray,
) -> tuple[InventorySnapshotEntry, ...]:
    """Read per-slot level-pip counts from a level-up menu's side panel."""

    try:
        layout = locate_weapon_hud(frame)
        pip_bands = inventory_level_pip_counts(frame, layout)
        if len(pip_bands) < 2:
            return ()
    except (RuntimeError, ValueError):
        return ()

    entries: list[InventorySnapshotEntry] = []
    for item_type, levels in (("weapon", pip_bands[0]), ("passive_item", pip_bands[1])):
        for slot, level in enumerate(levels, start=1):
            if level < 1:
                continue
            entries.append(
                InventorySnapshotEntry(
                    item_type=item_type,
                    slot=slot,
                    item_name="",
                    level=level,
                    confidence="pip_geometry",
                )
            )
    return tuple(entries)


def stabilize_inventory_snapshots(
    snapshots: Sequence[Sequence[InventorySnapshotEntry]],
) -> tuple[InventorySnapshotEntry, ...]:
    """Keep per-slot pip counts that are stable across one menu segment."""

    if len(snapshots) < 2:
        return ()
    entries: list[InventorySnapshotEntry] = []
    for item_type in ("weapon", "passive_item"):
        for slot in range(1, 7):
            values = []
            for snapshot in snapshots:
                by_key = {(entry.item_type, entry.slot): entry.level for entry in snapshot}
                values.append(by_key.get((item_type, slot), 0))
            level, support = Counter(values).most_common(1)[0]
            if level < 1 or support / len(values) < 0.6:
                continue
            entries.append(
                InventorySnapshotEntry(
                    item_type=item_type,
                    slot=slot,
                    item_name="",
                    level=int(level),
                    confidence="pip_geometry",
                )
            )
    return tuple(entries)


def infer_retrospective_inventory_change(
    before: Sequence[InventorySnapshotEntry],
    after: Sequence[InventorySnapshotEntry],
    *,
    expected_item: str,
    expected_item_type: str | None = None,
    expected_slot: int | None = None,
) -> RetrospectiveInventoryChange | None:
    """Accept a prior selection when the next menu reveals one matching increase.

    Only positive changes in the expected inventory row are evidence.  Missing
    pips and changes in the other row are ignored because animation and visual
    effects can temporarily hide them.  A second positive change in the same
    row remains ambiguous and therefore prevents automatic backfilling.
    """

    if not before or not after or not expected_item or expected_item == "UNKNOWN":
        return None
    if expected_item_type is None:
        expected_entries = [entry for entry in before if entry.item_name == expected_item]
        if len(expected_entries) != 1:
            return None
        expected_item_type = expected_entries[0].item_type
        expected_slot = expected_entries[0].slot
    before_by_slot = {
        (entry.item_type, entry.slot): entry
        for entry in before
        if entry.item_type == expected_item_type
    }
    after_by_slot = {
        (entry.item_type, entry.slot): entry
        for entry in after
        if entry.item_type == expected_item_type
    }

    changes: list[RetrospectiveInventoryChange] = []
    for key, after_entry in after_by_slot.items():
        before_entry = before_by_slot.get(key)
        if before_entry is not None and after_entry.level == before_entry.level + 1:
            changes.append(
                RetrospectiveInventoryChange(
                    item_type=after_entry.item_type,
                    slot=after_entry.slot,
                    item_name=(
                        expected_item
                        if key == (expected_item_type, expected_slot)
                        else after_entry.item_name
                    ),
                    level_before=before_entry.level,
                    level_after=after_entry.level,
                    event_type="upgrade",
                )
            )
        # Decreases are detector dropouts, not valid inventory changes.
    if (
        len(changes) != 1
        or changes[0].item_type != expected_item_type
        or (expected_slot is not None and changes[0].slot != expected_slot)
    ):
        return None
    changes[0] = RetrospectiveInventoryChange(
        item_type=changes[0].item_type,
        slot=changes[0].slot,
        item_name=expected_item,
        level_before=changes[0].level_before,
        level_after=changes[0].level_after,
        event_type=changes[0].event_type,
    )
    return changes[0]


def infer_unique_retrospective_upgrade_from_owned_slots(
    before: Sequence[InventorySnapshotEntry],
    after: Sequence[InventorySnapshotEntry],
    owned: dict[str, list[str]],
    confirmation: Sequence[InventorySnapshotEntry] = (),
) -> RetrospectiveInventoryChange | None:
    """Infer one existing-item upgrade without trusting the selected icon.

    This is the late-game fallback used when Banish, animation, or cursor
    occlusion makes the selected option unreadable.  It accepts only a single
    +1 pip change across the complete weapon and passive inventory. The new
    level must still be visible in a later menu snapshot; this prevents a
    single unstable pip reading in a reward-only menu from becoming a false
    inventory upgrade. Missing pips, decreases, jumps larger than one, new
    slots, and multiple positive changes are never promoted automatically.
    """

    if not before or not after:
        return None
    before_by_slot = {(entry.item_type, entry.slot): entry for entry in before}
    after_by_slot = {(entry.item_type, entry.slot): entry for entry in after}
    changes: list[RetrospectiveInventoryChange] = []
    for key, after_entry in after_by_slot.items():
        before_entry = before_by_slot.get(key)
        item_type, slot = key
        owned_row = owned.get(item_type, [])
        if (
            before_entry is None
            or after_entry.level != before_entry.level + 1
            or not (1 <= slot <= len(owned_row))
        ):
            continue
        changes.append(
            RetrospectiveInventoryChange(
                item_type=item_type,
                slot=slot,
                item_name=owned_row[slot - 1],
                level_before=before_entry.level,
                level_after=after_entry.level,
                event_type="upgrade",
            )
        )
    if len(changes) != 1 or not confirmation:
        return None
    change = changes[0]
    confirmed_level = next(
        (
            entry.level
            for entry in confirmation
            if entry.item_type == change.item_type and entry.slot == change.slot
        ),
        None,
    )
    return change if confirmed_level is not None and confirmed_level >= change.level_after else None


def _sprite_histogram(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], mask, [12, 4, 4], [0, 180, 0, 256, 0, 256]
    ).reshape(-1)
    histogram /= np.linalg.norm(histogram) + 1e-9
    return histogram


def _prepare_sprite_reference(
    name: str, filename: str, path: Path, slot_size: int
) -> PreparedReference:
    """Place one raw in-game sprite on a menu-style black square."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        occupied = alpha > 8
    else:
        bgr = image[:, :, :3]
        occupied = np.max(bgr, axis=2) > 8
        alpha = occupied.astype(np.uint8) * 255
    coordinates = cv2.findNonZero(occupied.astype(np.uint8))
    if coordinates is None:
        raise RuntimeError(f"Sprite has no visible pixels: {path}")
    x, y, width, height = cv2.boundingRect(coordinates)
    bgr = bgr[y : y + height, x : x + width]
    alpha = alpha[y : y + height, x : x + width]
    target_extent = max(8, int(round(slot_size * 0.86)))
    scale = min(target_extent / width, target_extent / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
    bgr = cv2.resize(bgr, (new_width, new_height), interpolation=interpolation)
    alpha = cv2.resize(
        alpha, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((slot_size, slot_size, 3), dtype=np.uint8)
    mask = np.zeros((slot_size, slot_size), dtype=np.uint8)
    x0 = (slot_size - new_width) // 2
    y0 = (slot_size - new_height) // 2
    selected = alpha > 8
    destination = canvas[y0 : y0 + new_height, x0 : x0 + new_width]
    destination[selected] = bgr[selected]
    mask[y0 : y0 + new_height, x0 : x0 + new_width][selected] = 255
    return PreparedReference(
        name=name,
        filename=filename,
        image=canvas,
        mask=mask,
        histogram=_sprite_histogram(canvas, mask),
    )


def prepare_combined_references(
    weapon_icon_dir: Path,
    passive_icon_dir: Path,
    weapons: pd.DataFrame,
    passives: pd.DataFrame,
    slot_size: int,
    config: MatchConfig,
) -> list[PreparedReference]:
    # Level-up choices use the in-game ``Sprite-*`` artwork. Prefer those raw
    # wiki assets when available, then fall back to the processed HUD catalogue
    # for weapons whose sprite filename is absent.
    raw_weapon_rows: list[tuple[str, str]] = []
    fallback_weapon_rows: list[dict[str, str]] = []
    for row in weapons.itertuples(index=False):
        sprite_filename = f"Sprite-{row.weapon_name}.png"
        if (passive_icon_dir / sprite_filename).exists():
            raw_weapon_rows.append((str(row.weapon_name), sprite_filename))
        else:
            fallback_weapon_rows.append(
                {
                    "weapon_name": str(row.weapon_name),
                    "filename": str(row.filename),
                }
            )
    fallback_weapon_manifest = pd.DataFrame(fallback_weapon_rows)
    references = [
        _prepare_sprite_reference(
            name,
            filename,
            passive_icon_dir / filename,
            slot_size,
        )
        for name, filename in raw_weapon_rows
    ]
    if not fallback_weapon_manifest.empty:
        references.extend(
            prepare_reference_library(
                weapon_icon_dir, fallback_weapon_manifest, slot_size, config
            )
        )
    references.extend(
        _prepare_sprite_reference(
            str(row.weapon_name),
            str(row.filename),
            passive_icon_dir / str(row.filename),
            slot_size,
        )
        for row in passives.itertuples(index=False)
    )
    return references


def _menu_observation(
    frame: np.ndarray,
    frame_number: int,
    fps: float,
    option_rows: Sequence[Sequence[int]],
    references: Sequence[PreparedReference],
    item_types: dict[str, str],
    config: MatchConfig,
) -> MenuObservation:
    selected, cursor_score = selected_option_index(frame, option_rows)
    icon = crop_option_icon(frame, option_rows[selected])
    result = match_weapon_slot(icon, references, config)
    display_name = str(result["weapon"])
    suggestion = str(result["suggested_weapon"])
    resolved_name = display_name if display_name != "UNKNOWN" else suggestion
    item_type = item_types.get(resolved_name, "unknown")
    cursor_is_clear = cursor_score >= CURSOR_MARGIN_THRESHOLD
    needs_review = (
        bool(result["needs_review"])
        or item_type == "unknown"
        or not cursor_is_clear
    )
    confidence = str(result["confidence"])
    if not cursor_is_clear:
        confidence = "low"
    return MenuObservation(
        frame_number=frame_number,
        video_second=frame_number / fps,
        selected_index=selected,
        cursor_score=cursor_score,
        item_name=display_name,
        suggested_item=suggestion,
        item_type=item_type,
        confidence=confidence,
        match_score=float(result["match_score"]),
        score_margin=float(result["score_margin"]),
        needs_review=needs_review,
        option_count=len(option_rows),
    )


def level_transition_windows(
    xp_events: pd.DataFrame,
    *,
    start_second: float,
    end_second: float,
    lookback_seconds: float = 15.0,
    lookahead_seconds: float = 0.75,
    pause_intervals: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Build and merge bounded menu-search windows from XP level resets."""

    if pause_intervals:
        # The upstream video scan identifies the whole animated menu, including
        # a final boundary with no later XP event. Small margins let the menu
        # parser observe both opening and disappearance before finalizing a pick.
        raw_windows = [
            (max(start_second, begin - 0.2), min(end_second, end + 0.2))
            for begin, end in pause_intervals
            if end > start_second and begin < end_second
        ]
    else:
        # Historical XP inputs may not include the companion frame signal.
        # Prefer explicit boundaries; a level difference appears only at the
        # *next* XP event and can omit the final level-up in a bounded prefix.
        if "level_up_boundary_candidate" in xp_events:
            boundary = xp_events["level_up_boundary_candidate"].astype(str).str.lower()
            transitions = xp_events.loc[boundary.isin(("1", "true", "1.0"))].copy()
        else:
            reset = pd.to_numeric(
                xp_events["reset_inferred_level"], errors="coerce"
            ).ffill()
            transitions = xp_events.loc[reset.diff().fillna(0).gt(0)].copy()
        times = pd.to_numeric(transitions["video_time_b"], errors="coerce").dropna()
        raw_windows = [
            (max(start_second, float(second) - lookback_seconds),
             min(end_second, float(second) + max(lookahead_seconds, 5.0)))
            for second in times
            if start_second <= float(second) <= end_second + lookback_seconds
        ]
    merged: list[list[float]] = []
    for start, stop in sorted(raw_windows):
        if not merged or start > merged[-1][1] + 0.25:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def level_up_pause_intervals_from_signal(path: Path, fps: float) -> list[tuple[float, float]]:
    """Read the automated per-frame pause gate beside this run's XP events."""
    intervals: list[tuple[float, float]] = []
    start: int | None = None
    previous: int | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"frame_index", "gameplay_paused"}.issubset(reader.fieldnames or ()):
            raise ValueError("Companion XP frame signal lacks pause provenance columns")
        for row in reader:
            frame = int(row["frame_index"])
            if previous is not None and frame != previous + 1:
                raise ValueError("Companion XP frame signal is not frame-contiguous")
            blocked = str(row["gameplay_paused"]).strip().lower() in {"1", "true"}
            if blocked and start is None:
                start = frame
            elif not blocked and start is not None:
                intervals.append((start / fps, frame / fps))
                start = None
            previous = frame
    if start is not None and previous is not None:
        intervals.append((start / fps, (previous + 1) / fps))
    return intervals


def inventory_selection_exclusion_reason(
    item_name: str,
    item_type: str,
    owned: dict[str, list[str]],
    levels: dict[str, int],
    *,
    cursor_score: float,
    needs_review: bool,
) -> str | None:
    """Reject impossible low-confidence matches from reward-only menus."""

    if cursor_score < CURSOR_MARGIN_THRESHOLD:
        return "selection_pointer_unconfirmed"
    normal_max = NORMAL_MAX_LEVEL_BY_ITEM.get(item_name)
    if (
        item_name in owned.get(item_type, [])
        and normal_max is not None
        and levels.get(item_name, 1) >= normal_max
    ):
        return "item_already_at_normal_max"
    if (
        item_name not in owned.get(item_type, [])
        and len(owned.get(item_type, [])) >= 6
        and needs_review
    ):
        return "low_confidence_beyond_standard_slots"
    return None


def scan_level_up_menus(
    video_path: Path,
    xp_events: pd.DataFrame,
    weapon_icon_dir: Path,
    passive_icon_dir: Path,
    weapons: pd.DataFrame,
    passives: pd.DataFrame,
    item_types: dict[str, str],
    *,
    start_second: float,
    end_second: float,
    sample_fps: float = 10.0,
    config: MatchConfig | None = None,
    pause_signal_path: Path | None = None,
) -> list[MenuSegment]:
    """Find level-up menu segments and classify their final selections."""

    config = config or MatchConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    step_frames = max(1, int(round(fps / sample_fps)))
    pause_intervals = (
        level_up_pause_intervals_from_signal(pause_signal_path, fps)
        if pause_signal_path is not None else None
    )
    windows = level_transition_windows(
        xp_events,
        start_second=start_second,
        end_second=end_second,
        pause_intervals=pause_intervals,
    )
    references_by_size: dict[int, list[PreparedReference]] = {}
    segments: list[MenuSegment] = []
    try:
        for window_start, window_end in windows:
            start_frame = max(0, int(np.floor(window_start * fps)))
            end_frame = int(np.ceil(window_end * fps))
            frame_number = start_frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            active_start: float | None = None
            last_frame: np.ndarray | None = None
            last_frame_number: int | None = None
            last_option_rows: list[tuple[int, int, int, int]] | None = None
            last_reliable_frame: np.ndarray | None = None
            last_reliable_frame_number: int | None = None
            last_reliable_option_rows: list[
                tuple[int, int, int, int]
            ] | None = None
            last_icon_signatures: list[np.ndarray] | None = None
            active_initial_option_count: int | None = None
            active_inventory_snapshots: list[tuple[InventorySnapshotEntry, ...]] = []
            misses = 0
            while frame_number <= end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                # Menu selection is valid during the pause once the card
                # layout is stable. Sliding cards must not look like a
                # different one-/two-choice menu or a completed selection.
                option_rows = stable_level_up_option_rectangles(frame)
                current_icon_signatures: list[np.ndarray] | None = None
                if len(option_rows) in SUPPORTED_MENU_OPTION_COUNTS:
                    current_icon_signatures = menu_icon_signatures(frame, option_rows)
                    if (
                        active_start is not None
                        and last_icon_signatures is not None
                        and active_initial_option_count != len(option_rows)
                        and last_frame_number is not None
                        and (last_frame_number / fps - active_start) >= 0.3
                    ):
                        # A stacked level can change the number of choices
                        # without blank sampled frames. Finalize the prior menu
                        # state; the following sample begins the new state.
                        option_rows = []
                        current_icon_signatures = None
                        misses = 1
                if len(option_rows) in SUPPORTED_MENU_OPTION_COUNTS:
                    if active_start is None:
                        active_start = frame_number / fps
                        active_initial_option_count = len(option_rows)
                    last_frame = frame.copy()
                    last_frame_number = frame_number
                    last_option_rows = option_rows
                    last_icon_signatures = current_icon_signatures
                    active_inventory_snapshots.append(_menu_inventory_snapshot(frame))
                    _, cursor_score = selected_option_index(frame, option_rows)
                    if cursor_score >= CURSOR_MARGIN_THRESHOLD:
                        last_reliable_frame = frame.copy()
                        last_reliable_frame_number = frame_number
                        last_reliable_option_rows = option_rows
                    misses = 0
                elif active_start is not None:
                    misses += 1
                    if (
                        misses >= 2
                        and last_frame is not None
                        and last_frame_number is not None
                        and last_option_rows is not None
                    ):
                        selected_frame = (
                            last_reliable_frame
                            if last_reliable_frame is not None
                            else last_frame
                        )
                        selected_frame_number = (
                            last_reliable_frame_number
                            if last_reliable_frame_number is not None
                            else last_frame_number
                        )
                        selected_option_rows = (
                            last_reliable_option_rows
                            if last_reliable_option_rows is not None
                            else last_option_rows
                        )
                        side = max(
                            24,
                            int(round(selected_option_rows[0][3] * 0.34)),
                        )
                        if side not in references_by_size:
                            references_by_size[side] = prepare_combined_references(
                                weapon_icon_dir,
                                passive_icon_dir,
                                weapons,
                                passives,
                                side,
                                config,
                            )
                        observation = _menu_observation(
                            selected_frame,
                            selected_frame_number,
                            fps,
                            selected_option_rows,
                            references_by_size[side],
                            item_types,
                            config,
                        )
                        segments.append(
                            MenuSegment(
                                start_second=active_start,
                                end_second=last_frame_number / fps,
                                observation=observation,
                                inventory_snapshot=stabilize_inventory_snapshots(
                                    active_inventory_snapshots
                                ),
                            )
                        )
                        active_start = None
                        last_frame = None
                        last_frame_number = None
                        last_option_rows = None
                        last_reliable_frame = None
                        last_reliable_frame_number = None
                        last_reliable_option_rows = None
                        last_icon_signatures = None
                        active_initial_option_count = None
                        active_inventory_snapshots = []
                        misses = 0
                for _ in range(step_frames - 1):
                    if not capture.grab():
                        break
                frame_number += step_frames
            if (
                active_start is not None
                and last_frame is not None
                and last_frame_number is not None
                and last_option_rows is not None
            ):
                selected_frame = (
                    last_reliable_frame
                    if last_reliable_frame is not None
                    else last_frame
                )
                selected_frame_number = (
                    last_reliable_frame_number
                    if last_reliable_frame_number is not None
                    else last_frame_number
                )
                selected_option_rows = (
                    last_reliable_option_rows
                    if last_reliable_option_rows is not None
                    else last_option_rows
                )
                side = max(
                    24, int(round(selected_option_rows[0][3] * 0.34))
                )
                if side not in references_by_size:
                    references_by_size[side] = prepare_combined_references(
                        weapon_icon_dir,
                        passive_icon_dir,
                        weapons,
                        passives,
                        side,
                        config,
                    )
                observation = _menu_observation(
                    selected_frame,
                    selected_frame_number,
                    fps,
                    selected_option_rows,
                    references_by_size[side],
                    item_types,
                    config,
                )
                segments.append(
                    MenuSegment(
                        start_second=active_start,
                        end_second=last_frame_number / fps,
                        observation=observation,
                        inventory_snapshot=stabilize_inventory_snapshots(
                            active_inventory_snapshots
                        ),
                    )
                )
    finally:
        capture.release()

    deduplicated: list[MenuSegment] = []
    for segment in sorted(segments, key=lambda item: item.start_second):
        if (
            deduplicated
            and abs(segment.start_second - deduplicated[-1].start_second) < 0.5
            and abs(segment.end_second - deduplicated[-1].end_second) < 0.5
            and segment.observation.suggested_item
            == deduplicated[-1].observation.suggested_item
            and segment.observation.selected_index
            == deduplicated[-1].observation.selected_index
        ):
            if segment.end_second > deduplicated[-1].end_second:
                deduplicated[-1] = segment
            continue
        deduplicated.append(segment)
    return deduplicated


def _find_initial_gameplay_frame(capture, *, second: float, duration: float):
    """Require consecutive positive HUD observations; never accept loading fallback."""
    from .gems import gameplay_hud_score
    from .weapons import _largest_gold_bar_component, has_large_menu_overlay

    candidate = None
    for probe in np.arange(max(0.0, second), duration, 0.25):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(probe) * 1000)
        ok, frame = capture.read()
        if not ok:
            break
        valid = (
            not level_up_pause_evidence(frame)["blocked"]
            and gameplay_hud_score(frame) >= 0.90
            and not has_large_menu_overlay(frame)
        )
        if valid:
            try:
                _largest_gold_bar_component(frame, allow_fallback=False)
            except ValueError:
                valid = False
        if not valid:
            candidate = None
            continue
        frame_number = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
        if candidate is not None:
            return candidate
        candidate = (frame, float(probe), frame_number)
    raise RuntimeError("No confirmed initial gameplay HUD in requested scope; inventory unresolved")


def _first_gameplay_inventory(
    video_path: Path,
    weapon_icon_dir: Path,
    passive_icon_dir: Path,
    weapons: pd.DataFrame,
    passives: pd.DataFrame,
    *,
    second: float,
    config: MatchConfig,
    end_second: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    if end_second is not None:
        duration = min(duration, end_second)
    try:
        frame, used_second, used_frame = _find_initial_gameplay_frame(
            capture, second=second, duration=duration)
    finally:
        capture.release()
    layout = locate_weapon_hud(frame)
    weapon_rows, _, _ = analyze_weapon_frame(
        frame,
        weapon_icon_dir,
        weapons,
        layout=layout,
        config=config,
    )
    passive_layout = HUDLayout(
        x=layout.x,
        y=layout.y + layout.slot_size,
        slot_size=layout.slot_size,
        step_x=layout.step_x,
        slots=layout.slots,
        bar_x=layout.bar_x,
        bar_y=layout.bar_y,
        bar_width=layout.bar_width,
        bar_height=layout.bar_height,
    )
    passive_rows, _, _ = analyze_weapon_frame(
        frame,
        passive_icon_dir,
        passives,
        layout=passive_layout,
        config=config,
    )
    return weapon_rows, passive_rows, used_second, used_frame


def character_level_after(
    xp_events: pd.DataFrame, second: float
) -> int | str:
    times = pd.to_numeric(xp_events["video_time_b"], errors="coerce")
    after = xp_events.loc[times.ge(second)].head(1)
    if after.empty:
        return "unresolved"
    level = pd.to_numeric(after.iloc[0].get("hud_level"), errors="coerce")
    return int(level) if pd.notna(level) else "unresolved"


def _timeline_weapon_names(row: pd.Series) -> list[str]:
    names: list[str] = []
    for slot in range(1, 7):
        name = str(row.get(f"weapon_{slot}", "") or "").strip()
        if name and name != "nan" and name != "UNKNOWN":
            names.append(name)
    return names


def _gameplay_weapon_timeline(timeline: pd.DataFrame) -> pd.DataFrame:
    """Never use explicitly paused/unresolved HUD rows as inventory evidence."""
    if "screen_state" not in timeline.columns:
        return timeline
    return timeline.loc[timeline["screen_state"].isin(
        {"gameplay", "gameplay_near_overlay"}
    )]


def add_hud_evolution_events(
    events: pd.DataFrame,
    weapon_timeline: pd.DataFrame,
    xp_events: pd.DataFrame,
    *,
    video_name: str,
    fps: float,
) -> pd.DataFrame:
    """Add known evolution pairs visible as persistent HUD replacements."""

    additions: list[dict[str, object]] = []
    previous_names: list[str] | None = None
    for _, timeline_row in _gameplay_weapon_timeline(weapon_timeline).sort_values("video_second").iterrows():
        current_names = _timeline_weapon_names(timeline_row)
        if previous_names is None:
            previous_names = current_names
            continue
        removed = set(previous_names) - set(current_names)
        added = set(current_names) - set(previous_names)
        for evolved_name in sorted(added):
            base_name = EVOLUTION_BASE_BY_NAME.get(evolved_name)
            if not base_name or base_name not in removed:
                continue
            already_recorded = events.loc[
                events["event_type"].eq("evolution")
                & events["item_after"].eq(evolved_name)
            ]
            if not already_recorded.empty:
                continue
            second = pd.to_numeric(
                timeline_row.get("frame_second_used"), errors="coerce"
            )
            if pd.isna(second):
                second = float(timeline_row["video_second"])
            matching_levels = pd.to_numeric(
                events.loc[
                    events["item_after"].eq(base_name), "level_after"
                ],
                errors="coerce",
            ).dropna()
            level_before: int | str = (
                int(matching_levels.max()) if not matching_levels.empty else ""
            )
            slot = current_names.index(evolved_name) + 1
            slot_confidence = str(
                timeline_row.get(f"confidence_{slot}", "medium")
            )
            additions.append(
                {
                    "video": video_name,
                    "video_second": round(float(second), 4),
                    "frame_number": int(round(float(second) * fps)),
                    "character_level": character_level_after(
                        xp_events, float(second)
                    ),
                    "event_source": "treasure_chest",
                    "event_type": "evolution",
                    "item_type": "weapon",
                    "slot": slot,
                    "item_before": base_name,
                    "item_after": evolved_name,
                    "level_before": level_before,
                    "level_after": 1,
                    "normal_max_level": "",
                    "suggested_item": evolved_name,
                    "confidence": (
                        slot_confidence
                        if slot_confidence in {"high", "medium", "low"}
                        else "medium"
                    ),
                    "needs_review": slot_confidence == "low",
                }
            )
        previous_names = current_names

    if additions:
        events = pd.concat(
            [events.drop(columns=["event_id"], errors="ignore"), pd.DataFrame(additions)],
            ignore_index=True,
        )
    else:
        events = events.drop(columns=["event_id"], errors="ignore")
    events = events.sort_values(
        ["video_second", "frame_number"], kind="stable"
    ).reset_index(drop=True)
    events = _assign_inventory_event_ids(events)
    return events.reindex(columns=EVENT_COLUMNS)


def apply_weapon_slots_from_timeline(
    events: pd.DataFrame, weapon_timeline: pd.DataFrame
) -> pd.DataFrame:
    """Use the first subsequent readable HUD state for weapon-slot position."""

    output = events.copy()
    weapon_timeline = _gameplay_weapon_timeline(weapon_timeline)
    times = pd.to_numeric(weapon_timeline["video_second"], errors="coerce")
    for index, event in output.loc[output["item_type"].eq("weapon")].iterrows():
        second = float(event["video_second"])
        candidates = weapon_timeline.loc[
            times.between(second, second + 10.0, inclusive="both")
        ]
        for _, timeline_row in candidates.iterrows():
            names = _timeline_weapon_names(timeline_row)
            if str(event["item_after"]) in names:
                output.at[index, "slot"] = names.index(str(event["item_after"])) + 1
                break
    return output


def flag_out_of_range_slots(events: pd.DataFrame) -> pd.DataFrame:
    """Flag, rather than accept, automatically inferred slots above six."""

    output = events.copy()
    output["slot"] = output["slot"].astype(object)
    numeric_slots = pd.to_numeric(output["slot"], errors="coerce")
    invalid = numeric_slots.lt(1) | numeric_slots.gt(6) | numeric_slots.isna()
    output.loc[invalid, "slot"] = "unresolved"
    output.loc[invalid, "needs_review"] = True
    return output


def add_verified_treasure_chest_events(
    events: pd.DataFrame,
    chest_audit: pd.DataFrame,
    *,
    video_name: str,
) -> pd.DataFrame:
    """Merge frame-verified chest rewards into the inventory event stream."""

    required = {
        "reward_second",
        "reward_frame",
        "character_level",
        "event_type",
        "item_type",
        "slot",
        "item_before",
        "item_after",
        "confidence",
        "needs_review",
    }
    missing = sorted(required - set(chest_audit.columns))
    if missing:
        raise ValueError(
            "Treasure chest audit is missing required columns: "
            + ", ".join(missing)
        )

    additions: list[dict[str, object]] = []
    for audit_row in chest_audit.itertuples(index=False):
        item_after = str(audit_row.item_after)
        additions.append(
            {
                "video": video_name,
                "video_second": round(float(audit_row.reward_second), 4),
                "frame_number": int(audit_row.reward_frame),
                "character_level": int(audit_row.character_level),
                "event_source": "treasure_chest",
                "event_type": str(audit_row.event_type),
                "item_type": str(audit_row.item_type),
                "slot": audit_row.slot,
                "item_before": str(audit_row.item_before),
                "item_after": item_after,
                "level_before": "",
                "level_after": "",
                "normal_max_level": NORMAL_MAX_LEVEL_BY_ITEM.get(item_after, ""),
                "suggested_item": item_after,
                "confidence": str(audit_row.confidence),
                "needs_review": bool(audit_row.needs_review),
            }
        )

    combined = pd.concat(
        [events.drop(columns=["event_id"], errors="ignore"), pd.DataFrame(additions)],
        ignore_index=True,
    )
    combined = combined.sort_values(
        ["video_second", "frame_number"], kind="stable"
    ).reset_index(drop=True)
    combined = _assign_inventory_event_ids(combined)
    return combined.reindex(columns=EVENT_COLUMNS)


def rebuild_inventory_progression(events: pd.DataFrame) -> pd.DataFrame:
    """Recompute levels in chronological order after all event sources merge."""

    output = events.sort_values(
        ["video_second", "frame_number"], kind="stable"
    ).reset_index(drop=True)
    owned: dict[str, list[str]] = {"weapon": [], "passive_item": []}
    levels: dict[str, int] = {}
    retired_items: set[str] = set()
    contradictory_retrospective_rows: list[int] = []

    for index, event in output.iterrows():
        item_type = str(event["item_type"])
        item_before = str(event["item_before"] or "").strip()
        item_after = str(event["item_after"] or "").strip()
        event_type = str(event["event_type"])
        if not item_after or item_after == "nan":
            continue
        if (
            str(event.get("event_source", "")) == "level_up_retrospective"
            and item_after in retired_items
        ):
            contradictory_retrospective_rows.append(index)
            continue
        owned.setdefault(item_type, [])

        if event_type == "evolution":
            before_level = levels.get(item_before, "")
            output.at[index, "level_before"] = before_level
            output.at[index, "level_after"] = 1
            if item_before in owned[item_type]:
                position = owned[item_type].index(item_before)
                owned[item_type][position] = item_after
            elif item_after not in owned[item_type]:
                owned[item_type].append(item_after)
            levels.pop(item_before, None)
            retired_items.add(item_before)
            levels[item_after] = 1
        elif item_after in owned[item_type]:
            before_level = levels.get(item_after, 1)
            after_level = before_level + 1
            output.at[index, "event_type"] = "upgrade"
            output.at[index, "item_before"] = item_after
            output.at[index, "level_before"] = before_level
            output.at[index, "level_after"] = after_level
            levels[item_after] = after_level
        else:
            owned[item_type].append(item_after)
            configured_level = pd.to_numeric(
                event.get("level_after", ""), errors="coerce"
            )
            initial_level = (
                max(1, int(configured_level))
                if event_type == "initial_state" and pd.notna(configured_level)
                else 1
            )
            levels[item_after] = initial_level
            if event_type != "initial_state":
                output.at[index, "event_type"] = "new"
            output.at[index, "item_before"] = ""
            output.at[index, "level_before"] = ""
            output.at[index, "level_after"] = initial_level

        normal_max = NORMAL_MAX_LEVEL_BY_ITEM.get(item_after, "")
        output.at[index, "normal_max_level"] = normal_max
        after_numeric = pd.to_numeric(output.at[index, "level_after"], errors="coerce")
        if normal_max and pd.notna(after_numeric) and int(after_numeric) > normal_max:
            output.at[index, "needs_review"] = True

    if contradictory_retrospective_rows:
        output = output.drop(index=contradictory_retrospective_rows).reset_index(drop=True)
    output = _assign_inventory_event_ids(output)
    return output.reindex(columns=EVENT_COLUMNS)


def build_inventory_events(
    video_path: Path,
    xp_events: pd.DataFrame,
    menu_segments: Sequence[MenuSegment],
    weapon_rows: pd.DataFrame,
    passive_rows: pd.DataFrame,
    *,
    initial_second: float,
    initial_frame: int,
) -> pd.DataFrame:
    """Convert initial inventory and selected options to sheet-ready rows."""

    # The gameplay snapshot is the baseline, not another menu transaction.
    # Never replay pre-baseline menus (including pregame selections), even if
    # their item match is rejected later. Filter before retrospective lookahead
    # and before any character/item-level counters are advanced.
    menu_segments = tuple(
        segment for segment in menu_segments
        if segment.start_second > initial_second
    )

    owned: dict[str, list[str]] = {"weapon": [], "passive_item": []}
    levels: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for item_type, detections in (
        ("weapon", weapon_rows),
        ("passive_item", passive_rows),
    ):
        for detection in detections.itertuples(index=False):
            name = str(detection.weapon or "")
            if not detection.occupied or not name or name == "UNKNOWN":
                continue
            owned[item_type].append(name)
            levels[name] = 1
            rows.append(
                {
                    "video": video_path.name,
                    "video_second": round(initial_second, 4),
                    "frame_number": initial_frame,
                    "character_level": 1,
                    "event_source": "initial_state",
                    "event_type": "initial_state",
                    "item_type": item_type,
                    "slot": int(detection.slot),
                    "item_before": "",
                    "item_after": name,
                    "level_before": "",
                    "level_after": 1,
                    "normal_max_level": "",
                    "suggested_item": str(detection.suggested_weapon or name),
                    "confidence": str(detection.confidence),
                    "needs_review": bool(detection.needs_review),
                    "observed_at_level": "",
                    "inference_method": "first_gameplay_inventory",
                }
            )

    initial_character_level = character_level_after(xp_events, initial_second)
    if isinstance(initial_character_level, int):
        for row in rows:
            row["character_level"] = initial_character_level
        next_character_level = initial_character_level + 1
    else:
        next_character_level = None

    for segment_index, segment in enumerate(menu_segments):
        observation = segment.observation
        event_level = (
            next_character_level
            if next_character_level is not None
            else character_level_after(xp_events, segment.end_second)
        )
        item_name = (
            observation.item_name
            if observation.item_name != "UNKNOWN"
            else observation.suggested_item
        )
        item_type = observation.item_type
        if item_type not in owned:
            item_type = "weapon"
        exclusion_reason = inventory_selection_exclusion_reason(
            item_name,
            item_type,
            owned,
            levels,
            cursor_score=observation.cursor_score,
            needs_review=observation.needs_review,
        )
        retrospective: RetrospectiveInventoryChange | None = None
        if (
            observation.needs_review
            and item_name in owned[item_type]
            and segment_index + 1 < len(menu_segments)
        ):
            expected_slot_candidate = (
                owned[item_type].index(item_name) + 1
                if item_name in owned[item_type]
                else len(owned[item_type]) + 1
            )
            expected_slot = (
                expected_slot_candidate
                if 1 <= expected_slot_candidate <= 6
                else None
            )
            retrospective = infer_retrospective_inventory_change(
                segment.inventory_snapshot,
                menu_segments[segment_index + 1].inventory_snapshot,
                expected_item=item_name,
                expected_item_type=item_type,
                expected_slot=expected_slot,
            )
        if (
            retrospective is None
            and observation.needs_review
            and exclusion_reason not in {
                "item_already_at_normal_max",
                "low_confidence_beyond_standard_slots",
            }
            and segment_index + 1 < len(menu_segments)
        ):
            retrospective = infer_unique_retrospective_upgrade_from_owned_slots(
                segment.inventory_snapshot,
                menu_segments[segment_index + 1].inventory_snapshot,
                owned,
                menu_segments[segment_index + 2].inventory_snapshot
                if segment_index + 2 < len(menu_segments)
                else (),
            )
        if retrospective is not None:
            item_name = retrospective.item_name
            item_type = retrospective.item_type
            event_type = retrospective.event_type
            slot = retrospective.slot
            before = (
                levels.get(item_name, retrospective.level_before)
                if retrospective.event_type == "upgrade"
                else ""
            )
            after = int(before) + 1 if retrospective.event_type == "upgrade" else 1
            item_before = item_name if event_type == "upgrade" else ""
            if item_name not in owned[item_type]:
                owned[item_type].append(item_name)
            levels[item_name] = int(after)
            exclusion_reason = None
        if exclusion_reason is not None:
            if next_character_level is not None:
                next_character_level += 1
            continue
        if retrospective is None and item_name in owned[item_type]:
            event_type = "upgrade"
            slot = owned[item_type].index(item_name) + 1
            before = levels.get(item_name, 1)
            after = before + 1
            item_before = item_name
        elif retrospective is None:
            event_type = "new"
            owned[item_type].append(item_name)
            slot = len(owned[item_type])
            before = ""
            after = 1
            item_before = ""
        levels[item_name] = int(after)
        rows.append(
            {
                "video": video_path.name,
                "video_second": round(observation.video_second, 4),
                "frame_number": observation.frame_number,
                "character_level": event_level,
                "event_source": (
                    "level_up_retrospective"
                    if retrospective is not None
                    else "level_up"
                ),
                "event_type": event_type,
                "item_type": item_type,
                "slot": slot,
                "item_before": item_before,
                "item_after": item_name,
                "level_before": before,
                "level_after": after,
                "normal_max_level": "",
                "suggested_item": observation.suggested_item,
                "confidence": "medium" if retrospective is not None else observation.confidence,
                "needs_review": False if retrospective is not None else observation.needs_review,
                "observed_at_level": (
                    event_level + 1
                    if retrospective is not None and isinstance(event_level, int)
                    else event_level
                ),
                "inference_method": (
                    "next_menu_inventory_difference"
                    if retrospective is not None
                    else "final_selected_option"
                ),
            }
        )
        if next_character_level is not None:
            next_character_level += 1

    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    output = _assign_inventory_event_ids(output)
    return output.reindex(columns=EVENT_COLUMNS)


def apply_initial_weapon_timeline(
    weapon_rows: pd.DataFrame,
    weapon_timeline: pd.DataFrame,
    *,
    initial_second: float,
) -> pd.DataFrame:
    """Replace a one-frame opening guess with the nearest stable HUD timeline."""

    gameplay = _gameplay_weapon_timeline(weapon_timeline).copy()
    if gameplay.empty:
        return weapon_rows
    times = pd.to_numeric(gameplay["video_second"], errors="coerce")
    valid = gameplay.loc[times.notna()].copy()
    if valid.empty:
        return weapon_rows
    valid["_distance"] = (
        pd.to_numeric(valid["video_second"], errors="coerce") - initial_second
    ).abs()
    timeline_row = valid.sort_values(
        ["_distance", "video_second"], kind="stable"
    ).iloc[0]
    output = weapon_rows.copy()
    for slot in range(1, 7):
        name = str(timeline_row.get(f"weapon_{slot}", "") or "").strip()
        mask = pd.to_numeric(output["slot"], errors="coerce").eq(slot)
        if not mask.any():
            continue
        if not name or name in {"nan", "UNKNOWN"}:
            output.loc[mask, "occupied"] = False
            output.loc[mask, "weapon"] = ""
            output.loc[mask, "suggested_weapon"] = ""
            output.loc[mask, "confidence"] = "empty"
            output.loc[mask, "needs_review"] = False
            continue
        confidence = str(
            timeline_row.get(f"confidence_{slot}", "medium") or "medium"
        )
        output.loc[mask, "occupied"] = True
        output.loc[mask, "weapon"] = name
        output.loc[mask, "suggested_weapon"] = name
        output.loc[mask, "confidence"] = confidence
        output.loc[mask, "needs_review"] = confidence == "low"
    return output


def initial_level_observations(
    menu_segments: Sequence[MenuSegment],
    weapon_rows: pd.DataFrame,
    passive_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Map the first stable menu's HUD pips to opening item identities."""

    columns = [
        "video_second", "frame_number", "item_type", "slot", "item_name",
        "observed_level", "confidence",
    ]
    first = next(
        (segment for segment in menu_segments if segment.inventory_snapshot), None
    )
    if first is None:
        return pd.DataFrame(columns=columns)
    names: dict[tuple[str, int], str] = {}
    for item_type, detections in (
        ("weapon", weapon_rows), ("passive_item", passive_rows)
    ):
        for detection in detections.itertuples(index=False):
            name = str(detection.weapon or "").strip()
            if bool(detection.occupied) and name and name != "UNKNOWN":
                names[(item_type, int(detection.slot))] = name
    rows = []
    for entry in first.inventory_snapshot:
        name = names.get((entry.item_type, int(entry.slot)), "")
        if not name or int(entry.level) < 1:
            continue
        # The highlighted item's side-panel pips can preview the offered level
        # rather than the level immediately before selection. Its ordinary
        # transaction progression remains the safer evidence source.
        if name == first.observation.item_name:
            continue
        rows.append({
            "video_second": float(first.start_second),
            "frame_number": int(first.observation.frame_number),
            "item_type": entry.item_type,
            "slot": int(entry.slot),
            "item_name": name,
            "observed_level": int(entry.level),
            "confidence": entry.confidence,
        })
    return pd.DataFrame(rows, columns=columns)


def read_inventory_xp_events(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # The gem worker writes a blank CSV when no XP events were observed.
        # Accept that only with its matching explicit zero-event summary.
        from .legacy_gem_xp import normalize_completed_gem_worker
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        if summary.get("xp_jump_events") != 0:
            raise ValueError("Empty XP input is not a confirmed zero-event extraction")
        normalize_completed_gem_worker(worker_summary=summary, source_csv=path, config={})
        return pd.DataFrame(columns=["reset_inferred_level", "video_time_b", "hud_level"])


def record_inventory_events(
    video_path: Path,
    xp_events_path: Path,
    output_dir: Path,
    weapon_icon_dir: Path,
    passive_icon_dir: Path,
    weapon_manifest_path: Path,
    passive_manifest_path: Path,
    weapon_timeline_path: Path | None = None,
    treasure_chest_audit_path: Path | None = None,
    video_key: str = "video_4",
    *,
    start_second: float = 0.0,
    end_second: float | None = None,
    sample_fps: float = 10.0,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    xp_events = read_inventory_xp_events(xp_events_path)
    signal_name = xp_events_path.name.replace("_xp_ab_events.csv", "_xp_frame_signal.csv")
    pause_signal_path = xp_events_path.with_name(signal_name) if signal_name != xp_events_path.name else None
    if pause_signal_path is not None and not pause_signal_path.is_file():
        pause_signal_path = None
    weapons, passives, item_types = load_item_manifests(
        weapon_manifest_path, passive_manifest_path
    )
    config = MatchConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    capture.release()
    stop = min(duration, end_second if end_second is not None else duration)
    initial_target = min(max(start_second, 5.0), max(start_second, stop - 1.0 / fps))
    weapon_rows, passive_rows, initial_used, initial_frame = _first_gameplay_inventory(
        video_path,
        weapon_icon_dir,
        passive_icon_dir,
        weapons,
        passives,
        second=initial_target,
        config=config,
        end_second=stop,
    )
    weapon_timeline = None
    if weapon_timeline_path is not None:
        weapon_timeline = pd.read_csv(weapon_timeline_path)
        weapon_rows = apply_initial_weapon_timeline(
            weapon_rows, weapon_timeline, initial_second=initial_used
        )
    menus = scan_level_up_menus(
        video_path,
        xp_events,
        weapon_icon_dir,
        passive_icon_dir,
        weapons,
        passives,
        item_types,
        # Only search after the actual gameplay snapshot, not a fixed video
        # timestamp. Pregame UI must not enter the menu audit/downstream stages.
        start_second=max(start_second, initial_used + 1.0 / fps),
        end_second=stop,
        sample_fps=sample_fps,
        config=config,
        pause_signal_path=pause_signal_path,
    )
    events = build_inventory_events(
        video_path,
        xp_events,
        menus,
        weapon_rows,
        passive_rows,
        initial_second=initial_used,
        initial_frame=initial_frame,
    )
    level_observations = initial_level_observations(
        menus, weapon_rows, passive_rows
    )
    events["video"] = video_key
    if treasure_chest_audit_path is not None:
        chest_audit = pd.read_csv(treasure_chest_audit_path)
        events = add_verified_treasure_chest_events(
            events,
            chest_audit,
            video_name=video_key,
        )
    if weapon_timeline is not None:
        events = add_hud_evolution_events(
            events,
            weapon_timeline,
            xp_events,
            video_name=video_key,
            fps=fps,
        )
        events = apply_weapon_slots_from_timeline(events, weapon_timeline)
    events = rebuild_inventory_progression(events)
    events = flag_out_of_range_slots(events)
    event_prefix = re.sub(r"[^a-z0-9]+", "_", video_key.lower()).strip("_")
    events = events.drop(columns=["event_id"], errors="ignore")
    events.insert(
        0,
        "event_id",
        [f"{event_prefix}_inventory_{index:04d}" for index in range(1, len(events) + 1)],
    )
    events_path = output_dir / "inventory_events.csv"
    menu_path = output_dir / "level_up_menu_audit.csv"
    level_observations_path = output_dir / "initial_level_observations.csv"
    events.to_csv(events_path, index=False)
    level_observations.to_csv(level_observations_path, index=False)
    pd.DataFrame(
        [
            {
                "menu_start_second": segment.start_second,
                "menu_end_second": segment.end_second,
                **segment.observation.__dict__,
            }
            for segment in menus
        ]
    ).to_csv(menu_path, index=False)
    return events_path, menu_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--xp-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weapon-icon-dir", type=Path, required=True)
    parser.add_argument("--passive-icon-dir", type=Path, required=True)
    parser.add_argument("--weapon-manifest", type=Path, required=True)
    parser.add_argument("--passive-manifest", type=Path, required=True)
    parser.add_argument("--weapon-timeline", type=Path, default=None)
    parser.add_argument("--treasure-chest-audit", type=Path, default=None)
    parser.add_argument("--video-key", default="video_4")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    args = parser.parse_args()
    events, audit = record_inventory_events(
        video_path=args.video,
        xp_events_path=args.xp_events,
        output_dir=args.output_dir,
        weapon_icon_dir=args.weapon_icon_dir,
        passive_icon_dir=args.passive_icon_dir,
        weapon_manifest_path=args.weapon_manifest,
        passive_manifest_path=args.passive_manifest,
        weapon_timeline_path=args.weapon_timeline,
        treasure_chest_audit_path=args.treasure_chest_audit,
        video_key=args.video_key,
        start_second=args.start,
        end_second=args.end,
        sample_fps=args.sample_fps,
    )
    print("Inventory events:", events)
    print("Level-up audit:", audit)


if __name__ == "__main__":
    main()
