"""Frame-level XP-bar wrapper around the existing gem detector functions."""

from __future__ import annotations

from ..resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

import hashlib
import math
from pathlib import Path
from types import ModuleType

from ..models import EvidenceGrade, FrameObservation, Visibility
from ..gameplay_state import gameplay_pause_evidence
from ..video import FramePacket


def _observation_id(packet: FramePacket) -> str:
    key = f"{packet.video_asset_id}|{packet.frame_number}|xp_bar_progress".encode(
        "utf-8"
    )
    return f"obs_{hashlib.sha256(key).hexdigest()[:24]}"


class XPBarDetector:
    """Measure XP-bar progress while preserving unreliable measurements."""

    name = "xp_bar"
    version = "0.2.0"

    def __init__(self, legacy_scripts_dir: str | Path | None = None, source_artifact: str = "framework:detectors/gems.py") -> None:
        self.source_artifact = source_artifact
        self._module: ModuleType | None = None
        self._config: object | None = None

    def _load_legacy(self) -> tuple[ModuleType, object]:
        if self._module is None:
            from . import gems
            self._module = gems
            self._config = gems.Config()
        assert self._config is not None
        return self._module, self._config

    def observe(self, packet: FramePacket) -> list[FrameObservation]:
        module, config = self._load_legacy()
        raw_progress, quality = module.measure_xp_bar_progress(packet.image, config)
        hud_score = module.gameplay_hud_score(packet.image)
        overlay_score = module.level_up_overlay_score(packet.image)
        pause = gameplay_pause_evidence(packet.image)
        finite = math.isfinite(float(raw_progress))
        accepted = bool(
            finite
            and not pause["blocked"]
            and quality >= config.xp_min_quality
            and hud_score >= config.hud_score_threshold
            and overlay_score < config.level_up_strong_overlay_threshold
        )
        if accepted:
            visibility = Visibility.VISIBLE
            grade = EvidenceGrade.A
            value = float(raw_progress)
            reason = "quality_and_gameplay_hud_passed"
        elif pause["blocked"]:
            visibility = Visibility.OCCLUDED
            grade = EvidenceGrade.C
            value = None
            reason = str(pause["reason"])
        elif overlay_score >= config.level_up_strong_overlay_threshold:
            visibility = Visibility.OCCLUDED
            grade = EvidenceGrade.C
            value = None
            reason = "level_up_overlay_present"
        elif finite and quality >= config.xp_min_quality:
            visibility = Visibility.OCCLUDED
            grade = EvidenceGrade.C
            value = None
            reason = "gameplay_hud_not_confirmed"
        else:
            visibility = Visibility.UNKNOWN
            grade = EvidenceGrade.UNRESOLVED
            value = None
            reason = "xp_bar_quality_below_threshold"

        return [
            FrameObservation(
                observation_id=_observation_id(packet),
                video_asset_id=packet.video_asset_id,
                detector_name=self.name,
                detector_version=self.version,
                observation_type="xp_bar_progress",
                frame_number=packet.frame_number,
                media_time_ms=packet.media_time_ms,
                visibility=visibility,
                evidence_grade=grade,
                value=value,
                source_artifact=self.source_artifact,
                attributes={
                    "unit": "fraction",
                    "accepted": accepted,
                    "reliability_reason": reason,
                    "gameplay_pause": pause,
                    "raw_progress_fraction": (
                        float(raw_progress) if finite else None
                    ),
                    "bar_quality": float(quality),
                    "gameplay_hud_score": float(hud_score),
                    "level_up_overlay_score": float(overlay_score),
                    "minimum_bar_quality": float(config.xp_min_quality),
                    "minimum_gameplay_hud_score": float(
                        config.hud_score_threshold
                    ),
                    "maximum_level_up_overlay_score": float(
                        config.level_up_strong_overlay_threshold
                    ),
                },
            )
        ]
