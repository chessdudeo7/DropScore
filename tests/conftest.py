"""Shared fixtures.

Tests generate their own video rather than checking a fixture file into the repo,
so the suite stays self-contained and the clip's exact properties (frame count,
size, fps) are known rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

# (fourcc, extension) pairs to try. Availability varies by OpenCV build, so the
# fixture falls back rather than assuming any one encoder is present.
_ENCODERS = [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]

CLIP_FRAMES = 90
CLIP_FPS = 30.0
CLIP_WIDTH = 640
CLIP_HEIGHT = 360


def _render_frame(index: int, width: int, height: int) -> np.ndarray:
    """A crude falling-tile frame: dark background, keybed strip, moving tiles."""
    frame = np.full((height, width, 3), 12, dtype=np.uint8)

    # keybed occupying the bottom fifth
    keybed_top = int(height * 0.8)
    frame[keybed_top:, :] = 235
    for i in range(0, width, 24):
        cv2.rectangle(frame, (i, keybed_top), (i + 10, keybed_top + 30), (20, 20, 20), -1)

    # tiles falling toward it
    for lane in range(6):
        x = 40 + lane * 95
        y = (index * 6 + lane * 55) % keybed_top
        cv2.rectangle(frame, (x, y), (x + 26, y + 60), (110, 215, 245), -1)

    return frame


def _write_clip(path: Path, frames: int, fps: float, width: int, height: int) -> Path:
    for fourcc_name, suffix in _ENCODERS:
        target = path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            continue

        for i in range(frames):
            writer.write(_render_frame(i, width, height))
        writer.release()

        if target.exists() and target.stat().st_size > 0:
            return target

    pytest.skip("No usable OpenCV video encoder in this build")


@pytest.fixture(scope="session")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short synthetic falling-tile clip with known properties."""
    directory = tmp_path_factory.mktemp("clips")
    return _write_clip(
        directory / "clip", CLIP_FRAMES, CLIP_FPS, CLIP_WIDTH, CLIP_HEIGHT
    )
