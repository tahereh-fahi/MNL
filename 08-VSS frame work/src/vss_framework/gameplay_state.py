"""Shared visual exclusion for gameplay-pausing overlays.

This is a measurement gate, not proof that every unblocked frame is gameplay.
Consumers must retain their pregame/results/occlusion checks. It deliberately
uses panel geometry rather than text, choice names, video IDs, or frame numbers.
"""

from __future__ import annotations

import cv2
import numpy as np

GAMEPLAY_STATE_VERSION = "interruption-panel-v3"


def _readable_level_up_option_rows(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Reuse inventory's 1--4 option geometry as independent menu evidence.

    The import is intentionally local: inventory's stable-menu wrapper uses
    this module, while its raw rectangle detector has no pause dependency.
    Rows confined to the lower reward-card area are excluded so a treasure
    result cannot be relabelled as a Level Up menu.
    """
    from .detectors.inventory import level_up_option_rectangles

    height = frame_bgr.shape[0]
    return [
        tuple(map(int, row))
        for row in level_up_option_rectangles(frame_bgr)
        if int(row[1]) + int(row[3]) <= height * 0.78
    ]


def gameplay_pause_evidence(frame_bgr: np.ndarray) -> dict:
    """Classify purple, gold-bordered level-up and treasure interruptions.

    A small rotating panel is already a pause. A large axis-aligned panel whose
    choice cards have stopped protruding is a stable-menu candidate. Neither
    phase is a valid source for gameplay deltas. Menu consumers may still read
    stable choices, and progression consumers may retain the level-up boundary.
    """
    result = {"blocked": False, "phase": "gameplay_unblocked",
              "interruption_type": None,
              "panel_area_fraction": 0.0, "reason": "",
              "classifier_version": GAMEPLAY_STATE_VERSION}
    if frame_bgr is None or frame_bgr.size == 0:
        return result
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, 640.0 / max(1, width))
    frame = cv2.resize(frame_bgr, (max(1, round(width * scale)),
                                  max(1, round(height * scale))))
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(hsv, (105, 25, 50), (145, 110, 180))
    gold = cv2.inRange(hsv, (10, 65, 110), (35, 255, 255))
    # Pregame character/stage selection uses the same palette but has no
    # persistent XP-bar border. Do not label those panels as level-up menus.
    top_border = gold[:max(1, round(h * .045)), round(w * .10):round(w * .90)]
    if top_border.size == 0 or np.max(np.mean(top_border > 0, axis=1)) < .60:
        return result
    # The rendered XP fill ranges from cyan/blue to purple at later levels.
    xp_fill = cv2.inRange(hsv, (80, 120, 100), (165, 255, 255))
    top_fill = xp_fill[:max(1, round(h * .045)), round(w * .10):round(w * .90)]
    filled_xp_at_boundary = bool(np.max(np.mean(top_fill > 0, axis=1)) >= .90)
    # Run the reusable card geometry on the already downscaled classifier
    # frame; only presence/count is needed here, and this keeps the shared
    # per-frame gate inexpensive for every downstream detector.
    structured_option_rows = _readable_level_up_option_rows(frame)
    gray = cv2.inRange(hsv, (0, 0, 85), (179, 30, 185))
    n, _, stats, _ = cv2.connectedComponentsWithStats(gray)
    choice_cards = []
    for i in range(1, n):
        gx, gy, gw, gh, ga = stats[i]
        if not (gw > w * .16 and gh > h * .06
                and 2.5 <= gw / max(1, gh) <= 6.0
                and ga / max(1, gw * gh) >= .70
                and h * .12 <= gy <= h * .70 and gx + gw > w * .40):
            continue
        card_border = np.zeros((h, w), np.uint8)
        cv2.rectangle(card_border, (gx, gy), (gx + gw - 1, gy + gh - 1), 255, 5)
        card_gold = cv2.countNonZero(cv2.bitwise_and(card_border, gold))
        if card_gold / max(1, cv2.countNonZero(card_border)) >= .08:
            choice_cards.append((gx, gy, gw, gh))
    contours, _ = cv2.findContours(purple, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    area = float(h * w)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        contour_area = cv2.contourArea(contour)
        if contour_area < area * 0.0025:
            break
        x, y, pw, ph = cv2.boundingRect(contour)
        cx, cy = x + pw / 2, y + ph / 2
        if not (w * .25 < cx < w * .75 and h * .18 < cy < h * .85):
            continue
        if pw < w * .035 or ph < h * .035:
            continue
        hull = cv2.convexHull(contour)
        rect = cv2.minAreaRect(hull)
        rw, rh = rect[1]
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > 5:
            continue
        # A flat panel has a largely rectangular hull, including when rotated.
        if cv2.contourArea(hull) / max(1.0, rw * rh) < .82:
            continue
        boundary = np.zeros((h, w), np.uint8)
        cv2.drawContours(boundary, [hull], -1, 255, 5)
        support = cv2.countNonZero(cv2.bitwise_and(boundary, gold))
        small = contour_area / area < .02
        if small and not (abs(cx - w / 2) < w * .12 and abs(cy - h / 2) < h * .15):
            continue
        # Subpixel borders on the first rotating frames occupy very few pixels
        # after resizing; require central rectangular geometry in that case.
        if support / max(1, cv2.countNonZero(boundary)) < (.01 if small else .055):
            continue
        hull_fill = cv2.contourArea(hull) / max(1, pw * ph)
        # Both interruptions animate the same purple/gold rectangle. Choice
        # cards are positive level-up evidence. Without them, a filled XP bar
        # identifies the opening level-up animation; otherwise the panel is a
        # treasure animation/menu. This prevents chest transitions from being
        # published as level-up transactions while still excluding both from
        # gameplay measurements.
        if choice_cards or structured_option_rows:
            interruption_type = "level_up"
        elif (small or hull_fill < .93) and filled_xp_at_boundary:
            # Before choice cards settle, a bright animation can temporarily
            # cover the XP bar in either event. Keep this frame blocked but do
            # not assign it to level-up or treasure until later sequence
            # evidence supplies the event type.
            interruption_type = "interruption"
        else:
            interruption_type = "treasure"
        stable = ph >= h * .50 and pw >= w * .22 and hull_fill >= .93
        if stable:
            # Sliding option cards can extend outside an already upright panel.
            # Uniform gray card surfaces, not changing option text, expose this.
            for gx, gy, gw, gh in choice_cards:
                if gx < x - 5 or gx + gw > x + pw + 5:
                    stable = False
                    break
        phase_suffix = "menu" if stable else "transition"
        result.update(blocked=True,
                      phase=f"{interruption_type}_{phase_suffix}",
                      interruption_type=interruption_type,
                      panel_area_fraction=round(contour_area / area, 6),
                      reason=f"{interruption_type}_panel_excludes_gameplay_measurements")
        return result
    if structured_option_rows:
        result.update(
            blocked=True,
            phase="level_up_menu",
            interruption_type="level_up",
            panel_area_fraction=0.0,
            reason="level_up_option_geometry_excludes_gameplay_measurements",
        )
    return result


def level_up_pause_evidence(frame_bgr: np.ndarray) -> dict:
    """Compatibility view containing level-up evidence only.

    Inventory and chest-reward readers use this strict view so a treasure
    panel remains observable to the chest pipeline. Gameplay measurement
    consumers must use :func:`gameplay_pause_evidence` instead.
    """
    result = gameplay_pause_evidence(frame_bgr)
    if result.get("interruption_type") in {"level_up", "interruption"}:
        if result.get("interruption_type") == "interruption":
            result = dict(result)
            result["phase"] = "level_up_transition"
            result["reason"] = "level_up_transition_candidate_excludes_gameplay_measurements"
        return result
    return {
        "blocked": False,
        "phase": "no_level_up_overlay",
        "interruption_type": result.get("interruption_type"),
        "panel_area_fraction": result.get("panel_area_fraction", 0.0),
        "reason": "",
        "classifier_version": GAMEPLAY_STATE_VERSION,
    }
