"""Deterministic OpenCV video reading with explicit frame/media clocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_ms(self) -> int:
        return round(self.frame_count * 1000.0 / self.fps)


@dataclass(frozen=True)
class FramePacket:
    """A decoded BGR image with stable source timing."""

    video_asset_id: str
    frame_number: int
    media_time_ms: int
    image: Any


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenCV is required for direct video scans. Install opencv-python "
            "or run with the workspace .venv interpreter."
        ) from error
    return cv2


class OpenCVVideoReader:
    """Read a half-open media interval and sample it without clock drift."""

    def __init__(self, path: str | Path, video_asset_id: str) -> None:
        self.path = Path(path).resolve()
        self.video_asset_id = video_asset_id
        cv2 = _cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise RuntimeError(f"Video reports an invalid frame rate: {fps}")
        self._cv2_module = cv2
        self._capture = capture
        self.metadata = VideoMetadata(
            path=self.path,
            fps=fps,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "OpenCVVideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_packets(
        self,
        *,
        start_ms: int = 0,
        end_ms: int | None = None,
        sample_fps: float = 2.0,
    ) -> Iterator[FramePacket]:
        """Yield sampled frames from ``[start_ms, end_ms)``.

        Target timestamps are converted independently to native frame numbers,
        which avoids cumulative rounding drift when native FPS is not evenly
        divisible by the requested sample rate.
        """

        if start_ms < 0:
            raise ValueError("start_ms cannot be negative")
        if sample_fps <= 0 or sample_fps > self.metadata.fps:
            raise ValueError("sample_fps must be positive and no greater than native FPS")
        stop_ms = self.metadata.duration_ms if end_ms is None else end_ms
        if stop_ms <= start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        stop_ms = min(stop_ms, self.metadata.duration_ms)

        start_frame = int(start_ms * self.metadata.fps / 1000.0)
        self._capture.set(self._cv2_module.CAP_PROP_POS_FRAMES, start_frame)
        next_sample_ms = float(start_ms)
        sample_period_ms = 1000.0 / sample_fps
        frame_number = start_frame

        while frame_number < self.metadata.frame_count:
            media_time_ms = round(frame_number * 1000.0 / self.metadata.fps)
            if media_time_ms >= stop_ms:
                break
            ok, image = self._capture.read()
            if not ok:
                break
            if media_time_ms + 0.5 >= next_sample_ms:
                yield FramePacket(
                    video_asset_id=self.video_asset_id,
                    frame_number=frame_number,
                    media_time_ms=media_time_ms,
                    image=image,
                )
                next_sample_ms += sample_period_ms
            frame_number += 1
