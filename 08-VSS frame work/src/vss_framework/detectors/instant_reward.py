"""Classify late-game Level-Up instant rewards from their selected icons."""

from __future__ import annotations

from ..resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

from dataclasses import replace
from pathlib import Path

from ..models import CanonicalEvent, EvidenceGrade, PublicationStatus
from ..gameplay_state import level_up_pause_evidence


DETECTOR_VERSION = "0.2.0"


def classify_instant_reward_icon(icon_bgr: object) -> tuple[str | None, dict[str, float]]:
    """Distinguish the green Big Coin Bag and orange Floor Chicken sprites."""

    import cv2
    import numpy as np

    hsv = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 70)
    hues = hsv[:, :, 0][saturated]
    if not len(hues):
        return None, {"green_fraction": 0.0, "orange_fraction": 0.0}
    green = float(np.mean((hues >= 35) & (hues <= 95)))
    orange = float(np.mean((hues >= 3) & (hues <= 25)))
    metrics = {"green_fraction": green, "orange_fraction": orange}
    if green >= 0.5 and green >= orange * 2:
        return "Big Coin Bag", metrics
    if orange >= 0.5 and orange >= green * 2:
        return "Floor Chicken", metrics
    return None, metrics


def resolve_instant_reward_transactions(
    *,
    video_path: Path,
    inventory_script_path: Path,
    transactions: list[CanonicalEvent],
) -> tuple[list[CanonicalEvent], list[CanonicalEvent]]:
    """Resolve two-choice instant-reward menus and emit their reward outcomes."""

    import cv2

    from . import inventory

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    resolved: list[CanonicalEvent] = []
    outcomes: list[CanonicalEvent] = []
    try:
        for transaction in transactions:
            if (
                transaction.attributes.get("option_count") != 2
                or transaction.frame_number is None
            ):
                resolved.append(transaction)
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, transaction.frame_number)
            ok, frame = capture.read()
            if not ok:
                resolved.append(transaction)
                continue
            if level_up_pause_evidence(frame)["phase"] != "level_up_menu":
                resolved.append(transaction)
                continue
            rows = inventory.level_up_option_rectangles(frame)
            selected_index = int(transaction.evidence[0].details["selected_index"])
            if len(rows) != 2 or selected_index >= len(rows):
                resolved.append(transaction)
                continue
            classified = [
                classify_instant_reward_icon(inventory.crop_option_icon(frame, row))
                for row in rows
            ]
            labels = [item[0] for item in classified]
            if set(labels) != {"Big Coin Bag", "Floor Chicken"}:
                resolved.append(transaction)
                continue
            reward, metrics = classified[selected_index]
            event_type = "big_coin_bag" if reward == "Big Coin Bag" else "floor_chicken"
            family = "currency" if reward == "Big Coin Bag" else "consumable"
            attributes = {
                **transaction.attributes,
                "resolved_action": "select_instant_reward",
                "possible_actions": ["select_instant_reward"],
                "reward_outcome": reward,
                "icon_color_metrics": metrics,
                "instant_reward_detector_version": DETECTOR_VERSION,
                "menu_option_labels": labels,
            }
            updated = replace(
                transaction,
                evidence_grade=EvidenceGrade.A,
                publication_status=PublicationStatus.AUTO_ACCEPTED,
                inference_method="selected_icon_color_in_two_choice_reward_menu",
                item_name=reward,
                item_type="instant_reward",
                action="select_instant_reward",
                attributes=attributes,
            )
            resolved.append(updated)
            outcomes.append(replace(
                updated,
                event_id=f"{updated.event_id}_{event_type}",
                event_family=family,
                event_type=event_type,
                action="acquired",
                quantity=1,
                unit="reward",
                attributes={
                    "transaction_event_id": updated.event_id,
                    "reward_outcome": reward,
                    "icon_color_metrics": metrics,
                    "quantity_semantics": "one_selected_reward_option",
                },
            ))
    finally:
        capture.release()
    return resolved, outcomes
