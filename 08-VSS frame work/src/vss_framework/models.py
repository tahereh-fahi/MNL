"""Core, source-independent data contracts for the VSS framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:  # Python 3.9 compatibility for the existing OpenCV environment
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any


class EvidenceGrade(StrEnum):
    """Strength of available evidence; not a calibrated probability."""

    A = "A"
    B = "B"
    C = "C"
    UNRESOLVED = "unresolved"


class PublicationStatus(StrEnum):
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_REVIEW = "needs_review"
    UNRESOLVED = "unresolved"
    UNOBSERVABLE = "unobservable"


class TemporalPrecision(StrEnum):
    FRAME = "frame"
    WINDOW = "window"
    BOUNDED = "bounded"
    INTERVAL = "interval"


class Visibility(StrEnum):
    """Whether the visual source needed for an observation was readable."""

    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceReference:
    source_artifact: str
    source_record_key: str
    modalities: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalEvent:
    event_id: str
    video_asset_id: str
    session_id: str
    event_family: str
    event_type: str
    time_lower_ms: int
    time_upper_ms: int
    anchor_time_ms: int | None
    temporal_precision: TemporalPrecision
    evidence_grade: EvidenceGrade
    publication_status: PublicationStatus
    inference_method: str
    processing_run_id: str
    frame_number: int | None = None
    game_time_ms: int | None = None
    character_level: int | None = None
    item_name: str | None = None
    item_type: str | None = None
    action: str | None = None
    acquisition_source: str | None = None
    quantity: float | None = None
    quantity_min: float | None = None
    quantity_max: float | None = None
    unit: str | None = None
    evidence: tuple[EvidenceReference, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_lower_ms < 0:
            raise ValueError("time_lower_ms cannot be negative")
        if self.time_upper_ms < self.time_lower_ms:
            raise ValueError("time_upper_ms cannot precede time_lower_ms")
        if self.anchor_time_ms is not None and not (
            self.time_lower_ms <= self.anchor_time_ms <= self.time_upper_ms
        ):
            raise ValueError("anchor_time_ms must lie inside the event bounds")
        if self.quantity is not None and self.quantity < 0:
            raise ValueError("quantity cannot be negative")
        if self.quantity_min is not None and self.quantity_max is not None:
            if self.quantity_max < self.quantity_min:
                raise ValueError("quantity_max cannot be smaller than quantity_min")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameObservation:
    """One detector result tied to an exact sampled video frame.

    Observations are detector evidence, not canonical events.  A later
    temporal resolver may combine several observations into one candidate or
    event without losing the frame-level provenance recorded here.
    """

    observation_id: str
    video_asset_id: str
    detector_name: str
    detector_version: str
    observation_type: str
    frame_number: int
    media_time_ms: int
    visibility: Visibility
    evidence_grade: EvidenceGrade
    value: str | int | float | bool | None
    source_artifact: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_number < 0:
            raise ValueError("frame_number cannot be negative")
        if self.media_time_ms < 0:
            raise ValueError("media_time_ms cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalObservation:
    """A numeric or textual detector signal over a bounded media interval."""

    observation_id: str
    video_asset_id: str
    session_id: str
    detector_name: str
    detector_version: str
    observable_code: str
    time_lower_ms: int
    time_upper_ms: int
    temporal_precision: TemporalPrecision
    visibility: Visibility
    evidence_grade: EvidenceGrade
    observed: bool
    source_artifact: str
    source_record_key: str
    numeric_value: float | int | None = None
    text_value: str | None = None
    unit: str | None = None
    frame_number: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_lower_ms < 0:
            raise ValueError("time_lower_ms cannot be negative")
        if self.time_upper_ms < self.time_lower_ms:
            raise ValueError("time_upper_ms cannot precede time_lower_ms")
        if self.frame_number is not None and self.frame_number < 0:
            raise ValueError("frame_number cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
