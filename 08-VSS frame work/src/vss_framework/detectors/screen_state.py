"""Screen-state observations backed by the existing inventory detector."""

from __future__ import annotations

from ..resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

import hashlib
from pathlib import Path
from types import ModuleType

from ..models import EvidenceGrade, FrameObservation, Visibility
from ..gameplay_state import gameplay_pause_evidence
from ..video import FramePacket


def _observation_id(packet: FramePacket, detector: str, kind: str) -> str:
    key = (
        f"{packet.video_asset_id}|{packet.frame_number}|{detector}|{kind}"
    ).encode("utf-8")
    return f"obs_{hashlib.sha256(key).hexdigest()[:24]}"


class ScreenStateDetector:
    """Classify Level-Up menus and ordinary HUD-readable gameplay frames.

    The Level-Up geometry and HUD locator are imported from the existing
    Video 4 inventory pipeline.  This wrapper standardizes their output; it
    does not fork or retrain those algorithms.
    """

    name = "screen_state"
    version = "0.2.0"

    def __init__(self, legacy_scripts_dir: str | Path | None = None, source_artifact: str = "framework:detectors/inventory.py") -> None:
        self.source_artifact = source_artifact
        self._inventory_module: ModuleType | None = None
        self._weapon_module: ModuleType | None = None

    def _load_legacy(self) -> tuple[ModuleType, ModuleType]:
        if self._inventory_module is None or self._weapon_module is None:
            from . import inventory, weapons
            self._inventory_module = inventory
            self._weapon_module = weapons
        return self._inventory_module, self._weapon_module

    def observe(self, packet: FramePacket) -> list[FrameObservation]:
        inventory, weapons = self._load_legacy()
        pause = gameplay_pause_evidence(packet.image)
        option_rows = inventory.level_up_option_rectangles(packet.image)
        if option_rows and pause["phase"] == "level_up_menu":
            return [
                FrameObservation(
                    observation_id=_observation_id(packet, self.name, "level_up_menu"),
                    video_asset_id=packet.video_asset_id,
                    detector_name=self.name,
                    detector_version=self.version,
                    observation_type="screen_state",
                    frame_number=packet.frame_number,
                    media_time_ms=packet.media_time_ms,
                    visibility=Visibility.VISIBLE,
                    evidence_grade=EvidenceGrade.A,
                    value="level_up_menu",
                    source_artifact=self.source_artifact,
                    attributes={
                        "option_count": len(option_rows),
                        "option_rectangles_xywh": [list(map(int, row)) for row in option_rows],
                        "gameplay_measurements_allowed": False,
                        "pause_evidence": pause,
                    },
                )
            ]

        if pause["blocked"]:
            state = str(pause["phase"])
            return [FrameObservation(
                observation_id=_observation_id(packet, self.name, state),
                video_asset_id=packet.video_asset_id,
                detector_name=self.name,
                detector_version=self.version,
                observation_type="screen_state",
                frame_number=packet.frame_number,
                media_time_ms=packet.media_time_ms,
                visibility=Visibility.VISIBLE,
                evidence_grade=EvidenceGrade.B,
                value=state,
                source_artifact=self.source_artifact,
                attributes={"gameplay_measurements_allowed": False,
                            "pause_evidence": pause},
            )]

        try:
            layout = weapons.locate_weapon_hud(packet.image)
        except RuntimeError:
            state = "unknown"
            visibility = Visibility.UNKNOWN
            grade = EvidenceGrade.C
            attributes = {"reason": "level_up_geometry_absent_and_weapon_hud_unreadable"}
        else:
            # HUD visibility alone does not prove that the entire screen is in
            # ordinary gameplay; overlays may leave the HUD readable.
            state = "hud_visible_unclassified"
            visibility = Visibility.VISIBLE
            grade = EvidenceGrade.B
            attributes = {
                "weapon_hud_xywh": [
                    int(layout.x),
                    int(layout.y),
                    int(layout.step_x * layout.slots),
                    int(layout.slot_size),
                ]
            }

        return [
            FrameObservation(
                observation_id=_observation_id(packet, self.name, state),
                video_asset_id=packet.video_asset_id,
                detector_name=self.name,
                detector_version=self.version,
                observation_type="screen_state",
                frame_number=packet.frame_number,
                media_time_ms=packet.media_time_ms,
                visibility=visibility,
                evidence_grade=grade,
                value=state,
                source_artifact=self.source_artifact,
                attributes=attributes,
            )
        ]
