"""Common contract implemented by all frame-level detectors."""

from __future__ import annotations

from typing import Protocol

from ..models import FrameObservation
from ..video import FramePacket


class FrameDetector(Protocol):
    name: str
    version: str

    def observe(self, packet: FramePacket) -> list[FrameObservation]:
        """Return zero or more evidence observations for one decoded frame."""

