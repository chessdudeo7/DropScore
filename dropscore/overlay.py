"""Stage 6: draw what the pipeline thinks it sees.

For computer-vision work these overlays are worth more than the test suite. Nearly
every bug in stages 3-5 is obvious at a glance in an annotated frame and invisible
in a stack trace: a key grid off by half a key, a bloom halo swallowing a
neighbour, a merged repeated note, a strike line a few pixels high. Tests tell you
*that* something regressed; an overlay tells you *what*.

The grid overlay is the important one. Drawing the fitted key boundaries straight
onto the real keybed makes a misalignment immediately visible — and misalignment
is the failure that silently transposes an entire transcription.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .calibrate import Calibration
from .config import DEFAULT, Config
from .keyboard import is_black
from .notes import pitch_name
from .tiles import Palette, Tile, detect_in_frame
from .tracking import SpeedEstimate
from .video import Frame, open_writer

log = logging.getLogger(__name__)

# Distinct, colour-blind-safe-ish track colours, in BGR.
TRACK_COLORS = (
    (80, 200, 255),  # amber
    (255, 170, 80),  # blue
    (120, 240, 140),  # green
    (200, 120, 255),  # pink
)

GRID_COLOR = (90, 90, 90)
ANCHOR_COLOR = (60, 220, 255)
STRIKE_COLOR = (60, 60, 255)
BLACK_KEY_COLOR = (200, 120, 60)
HUD_TEXT = (240, 240, 240)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def track_color(index: int) -> tuple[int, int, int]:
    return TRACK_COLORS[index % len(TRACK_COLORS)]


def _label(image: np.ndarray, text: str, origin: tuple[int, int], color, scale: float = 0.35) -> None:
    """Text with a dark outline, so it stays readable on any background."""
    cv2.putText(image, text, origin, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, _FONT, scale, color, 1, cv2.LINE_AA)


def draw_grid(image: np.ndarray, calibration: Calibration, labels: bool = True) -> np.ndarray:
    """Draw the fitted key grid over the keybed.

    Every white boundary is drawn faintly; every C is drawn brightly and named.
    If the bright lines do not land on the real C keys, the fit is wrong and the
    transcription will be transposed.
    """
    layout = calibration.layout
    height = image.shape[0]
    top = calibration.strike_y
    bottom = min(calibration.keybed_bottom, height)

    for pitch in layout.pitches:
        if is_black(pitch):
            continue
        _, right = layout.white_span(pitch)
        cv2.line(image, (int(right), top), (int(right), bottom), GRID_COLOR, 1)

    for pitch in layout.pitches:
        if is_black(pitch):
            center = int(layout.key_center(pitch))
            cv2.line(image, (center, top), (center, top + (bottom - top) // 3), BLACK_KEY_COLOR, 1)
        elif pitch % 12 == 0:
            left, right = layout.white_span(pitch)
            cv2.line(image, (int(left), top), (int(left), bottom), ANCHOR_COLOR, 1)
            if labels:
                _label(image, pitch_name(pitch), (int(left) + 2, bottom - 4), ANCHOR_COLOR)

    cv2.line(image, (0, top), (image.shape[1], top), STRIKE_COLOR, 1)
    return image


def draw_tiles(image: np.ndarray, tiles: Sequence[Tile], labels: bool = True) -> np.ndarray:
    """Box every detected tile and name the key it was assigned to."""
    for tile in tiles:
        color = track_color(tile.track)
        cv2.rectangle(
            image,
            (int(tile.left), int(tile.top)),
            (int(tile.right), int(tile.bottom)),
            color,
            1,
        )
        if labels and tile.height >= 10:
            _label(image, pitch_name(tile.pitch), (int(tile.left) + 1, int(tile.bottom) - 3), color)
    return image


def draw_hud(
    image: np.ndarray,
    frame: Frame,
    calibration: Calibration,
    tiles: Sequence[Tile] | None = None,
    speed: SpeedEstimate | float | None = None,
    palette: Palette | None = None,
) -> np.ndarray:
    """Corner panel with the numbers behind the drawing."""
    lines = [
        f"frame {frame.index}  t={frame.time:.3f}s",
        f"keys {calibration.layout.first_pitch}-{calibration.layout.last_pitch}"
        f"  width {calibration.white_width:.2f}px",
        f"strike y={calibration.strike_y}  fit {calibration.confidence:.2f}",
    ]
    if speed is not None:
        if isinstance(speed, SpeedEstimate):
            lines.append(f"speed {speed.value:.1f}px/s  conf {speed.confidence:.2f}")
        else:
            lines.append(f"speed {float(speed):.1f}px/s")
    if palette is not None:
        lines.append(f"tracks {palette.track_count}")
    if tiles is not None:
        lines.append(f"tiles {len(tiles)}")

    # Dim behind the text with numpy: a slice is a non-contiguous view, which
    # OpenCV cannot write into as a destination.
    panel_height = min(image.shape[0], 14 * len(lines) + 10)
    panel_width = min(image.shape[1], 230)
    image[0:panel_height, 0:panel_width] = (
        image[0:panel_height, 0:panel_width] * 0.35
    ).astype(np.uint8)

    for row, text in enumerate(lines):
        _label(image, text, (8, 18 + row * 14), HUD_TEXT, scale=0.38)

    return image


def annotate(
    frame: Frame,
    calibration: Calibration,
    tiles: Sequence[Tile] | None = None,
    speed: SpeedEstimate | float | None = None,
    palette: Palette | None = None,
    labels: bool = True,
) -> np.ndarray:
    """One fully annotated frame. Never mutates the input image."""
    image = frame.image.copy()
    draw_grid(image, calibration, labels=labels)
    if tiles is not None:
        draw_tiles(image, tiles, labels=labels)
    draw_hud(image, frame, calibration, tiles, speed, palette)
    return image


def dump_frames(
    frames: Iterable[Frame],
    calibration: Calibration,
    out_dir: str | Path,
    palette: Palette | None = None,
    speed: SpeedEstimate | float | None = None,
    prefix: str = "frame",
    config: Config | None = None,
) -> list[Path]:
    """Write annotated PNGs, one per frame."""

    config = config or DEFAULT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for frame in frames:
        tiles = (
            detect_in_frame(frame, palette, calibration, config)
            if palette is not None
            else None
        )
        path = out_dir / f"{prefix}_{frame.index:06d}.png"
        cv2.imwrite(str(path), annotate(frame, calibration, tiles, speed, palette))
        written.append(path)

    log.info("wrote %d annotated frames to %s", len(written), out_dir)
    return written


def dump_video(
    frames: Iterable[Frame],
    calibration: Calibration,
    path: str | Path,
    fps: float,
    size: tuple[int, int],
    palette: Palette | None = None,
    speed: SpeedEstimate | float | None = None,
    config: Config | None = None,
) -> Path:
    """Write an annotated copy of the video.

    Far easier to judge than stills: a grid error that looks plausible in one
    frame is obvious when tiles slide along it.
    """

    config = config or DEFAULT
    writer, target = open_writer(path, fps, size)
    count = 0
    try:
        for frame in frames:
            tiles = (
                detect_in_frame(frame, palette, calibration, config)
                if palette is not None
                else None
            )
            writer.write(annotate(frame, calibration, tiles, speed, palette))
            count += 1
    finally:
        writer.release()

    log.info("wrote %d annotated frames to %s", count, target)
    return target
