"""Stage 3: fit a KeyboardLayout to a video.

Everything downstream is a function of "which pixel column is which MIDI note",
so an error of one key transposes the entire transcription. The fit is therefore
built from three independent signals, each checked against the next:

1. **Where the keybed is** — from temporal activity. Above the strike line tiles
   move constantly; below it only struck keys change. That discriminator works
   regardless of palette, unlike brightness.
2. **How wide a white key is** — from the dominant spatial frequency of the
   key-separator edges, taken below the black keys where only white keys exist.
   FFT gives period and phase together, so boundaries come out sub-pixel.
3. **Which key is which** — from the black-key pattern. Black keys sit on white
   boundaries in a 2-3 grouping, so the boundary occupancy sequence repeats as
   [1,1,0,1,1,1,0] from C. Finding that rotation pins C, which is the only way to
   turn "white key #17" into a pitch without assuming the keyboard's range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .config import Config, DEFAULT
from .keyboard import COMMON_RANGES, KeyboardLayout, WHITE_PITCH_CLASSES, white_pitches
from .video import Frame

log = logging.getLogger(__name__)

# Boundary occupancy over one octave of white keys, starting at C:
# C|D C#, D|E D#, E|F none, F|G F#, G|A G#, A|B A#, B|C none.
BLACK_PATTERN = (1, 1, 0, 1, 1, 1, 0)


class CalibrationError(RuntimeError):
    """Raised when no plausible keyboard can be fitted."""


@dataclass(frozen=True)
class Calibration:
    """A fitted keyboard, plus how much to trust it."""

    layout: KeyboardLayout
    strike_y: int  # top of the keybed; tiles are read above this
    keybed_bottom: int
    white_width: float
    confidence: float  # 0-1, the black-key pattern match rate
    diagnostics: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.layout.white_count} white keys, "
            f"{self.layout.first_pitch}-{self.layout.last_pitch}, "
            f"key width {self.white_width:.2f}px, strike line y={self.strike_y}, "
            f"confidence {self.confidence:.2f}"
        )


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def median_frame(frames: Sequence[Frame]) -> np.ndarray:
    """Temporal median — the static background with the tiles removed."""
    stack = np.stack([f.image for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def row_activity(frames: Sequence[Frame], background: np.ndarray) -> np.ndarray:
    """Mean absolute deviation from the background, per row."""
    reference = _gray(background).astype(np.int16)
    total = np.zeros(reference.shape[0], dtype=np.float64)
    for frame in frames:
        diff = np.abs(_gray(frame.image).astype(np.int16) - reference)
        total += diff.mean(axis=1)
    return total / len(frames)


def find_keybed(frames: Sequence[Frame], background: np.ndarray, config: Config) -> tuple[int, int]:
    """Locate the keybed as the quiet band at the bottom of the frame."""
    cfg = config.calibration
    activity = row_activity(frames, background)
    height = activity.shape[0]

    peak = float(activity.max())
    if peak <= 1e-6:
        raise CalibrationError("nothing moves in this video; no tiles to track")

    # Split the frame where activity drops most sharply, rather than at a fixed
    # threshold: struck keys make the keybed move too, so its absolute activity
    # is not reliably small — only reliably smaller than the fall area's.
    lo = int(height * cfg.split_search[0])
    hi = int(height * cfg.split_search[1])
    above = np.cumsum(activity)
    contrasts = np.array(
        [above[y - 1] / y - (above[-1] - above[y - 1]) / (height - y) for y in range(lo, hi)]
    )
    top = lo + int(np.argmax(contrasts))
    contrast = float(contrasts.max())

    if contrast < peak * cfg.activity_ratio:
        raise CalibrationError(
            "no clear boundary between a moving area and a still keyboard; "
            "this may not be a falling-tile piano video"
        )

    # Trim a uniform letterbox below the keys, if any.
    gray = _gray(background)
    edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=1)
    bottom = height
    while bottom > top and edges[bottom - 1] < cfg.min_edge_energy:
        bottom -= 1

    depth = bottom - top
    if depth < cfg.min_keybed_px:
        raise CalibrationError(
            f"keybed band is only {depth}px tall; expected at least {cfg.min_keybed_px}"
        )
    if depth > height * cfg.max_keybed_ratio:
        raise CalibrationError(
            f"keybed band fills {depth / height:.0%} of the frame; the video is "
            f"probably not a falling-tile piano video"
        )
    return top, bottom


def _band(image: np.ndarray, top: int, bottom: int, lo: float, hi: float) -> np.ndarray:
    """Slice a fractional depth range out of the keybed."""
    height = bottom - top
    y0 = top + int(height * lo)
    y1 = max(y0 + 1, top + int(height * hi))
    return image[y0:y1]


def estimate_period(profile: np.ndarray, min_px: float, max_px: float) -> tuple[float, float]:
    """Dominant period and first peak offset of a near-periodic 1-D profile.

    Uses the FFT rather than peak-finding: separators can be missed or doubled by
    compression, but the dominant frequency survives that, and the phase gives a
    sub-pixel offset for free.
    """
    length = profile.shape[0]
    centred = profile - profile.mean()
    if not np.any(centred):
        raise CalibrationError("keybed has no vertical structure to measure")

    spectrum = np.fft.rfft(centred * np.hanning(length))
    k_lo = max(1, int(np.ceil(length / max_px)))
    k_hi = min(len(spectrum) - 1, int(np.floor(length / min_px)))
    if k_hi <= k_lo:
        raise CalibrationError("frame is too small to resolve individual keys")

    magnitudes = np.abs(spectrum[k_lo : k_hi + 1])
    k = k_lo + int(np.argmax(magnitudes))

    period = length / k
    # cos(2*pi*k*x/L + phase) peaks where the argument is a multiple of 2*pi.
    phase = float(np.angle(spectrum[k]))
    offset = (-phase / (2 * np.pi)) * period
    return period, offset % period


def _boundary_positions(offset: float, period: float, x_lo: float, x_hi: float) -> np.ndarray:
    first = offset + np.ceil((x_lo - offset) / period) * period
    count = int(np.floor((x_hi - first) / period)) + 1
    return first + np.arange(max(count, 0)) * period


def _keyboard_extent(profile: np.ndarray, config: Config) -> tuple[int, int]:
    """Columns spanned by the keyboard, ignoring quiet margins."""
    threshold = profile.max() * config.calibration.extent_ratio
    active = np.flatnonzero(profile > threshold)
    if active.size < 2:
        raise CalibrationError("could not find the horizontal extent of the keyboard")
    return int(active[0]), int(active[-1])


def _classify_black(values: np.ndarray, white_reference: float) -> np.ndarray:
    """Split boundary samples into black/not-black by distance to white."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        raise CalibrationError("keybed shows no black keys; cannot anchor pitch")

    # Two-means on a 1-D signal converges in a couple of passes.
    centres = np.array([lo, hi])
    for _ in range(12):
        labels = np.abs(values[:, None] - centres[None, :]).argmin(axis=1)
        for i in (0, 1):
            if np.any(labels == i):
                centres[i] = values[labels == i].mean()

    labels = np.abs(values[:, None] - centres[None, :]).argmin(axis=1)
    # The cluster nearer the known white brightness is the gap between keys.
    white_cluster = int(np.argmin(np.abs(centres - white_reference)))
    return labels != white_cluster


def anchor_pitch(black: np.ndarray) -> tuple[int, float]:
    """Find which white key is C, from the black-key pattern.

    Returns the index into WHITE_PITCH_CLASSES for the first white key, and the
    fraction of boundaries the best rotation explains.
    """
    if black.size < len(BLACK_PATTERN):
        raise CalibrationError(
            f"only {black.size} key boundaries visible; need at least "
            f"{len(BLACK_PATTERN)} to identify the black-key pattern"
        )

    best_rotation, best_score = 0, -1.0
    for rotation in range(7):
        expected = np.array(
            [BLACK_PATTERN[(i + rotation) % 7] for i in range(black.size)], dtype=bool
        )
        score = float((expected == black).mean())
        if score > best_score:
            best_rotation, best_score = rotation, score

    return best_rotation, best_score


def _first_pitch(white_index_of_first: int, white_count: int) -> int:
    """Choose the octave for a white key of known pitch class."""
    pitch_class = WHITE_PITCH_CLASSES[white_index_of_first]
    candidates = [p for p in range(12, 109) if p % 12 == pitch_class]

    # Prefer an exact match against a range these videos actually use.
    for first, last in COMMON_RANGES.values():
        if first in candidates and len(white_pitches(first, last)) == white_count:
            return first

    # Otherwise put the keyboard's centre as near middle C as possible.
    def centre_error(first: int) -> float:
        whites = [p for p in range(first, 109) if p % 12 in WHITE_PITCH_CLASSES]
        if len(whites) < white_count:
            return float("inf")
        return abs((first + whites[white_count - 1]) / 2 - 60)

    return min(candidates, key=centre_error)


def calibrate(frames: Sequence[Frame], config: Config = DEFAULT) -> Calibration:
    """Fit a keyboard to sampled frames."""
    if len(frames) < 2:
        raise CalibrationError("calibration needs at least two frames")

    cfg = config.calibration
    background = median_frame(frames)
    top, bottom = find_keybed(frames, background, config)

    gray = _gray(background).astype(np.float32)
    lower = _band(gray, top, bottom, *cfg.white_band)
    edges = np.abs(cv2.Sobel(lower, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)

    x_lo, x_hi = _keyboard_extent(edges, config)
    period, offset = estimate_period(edges, cfg.min_key_px, cfg.max_key_px)
    boundaries = _boundary_positions(offset, period, x_lo, x_hi)
    if boundaries.size < 2:
        raise CalibrationError("not enough key boundaries to fit a keyboard")

    # Sample the upper keybed at each interior boundary: dark means a black key.
    upper = _band(gray, top, bottom, *cfg.black_band)
    half = max(1, int(period * cfg.sample_ratio / 2))
    samples = np.array(
        [upper[:, max(0, int(x) - half) : int(x) + half + 1].mean() for x in boundaries]
    )
    white_reference = float(np.median(lower))
    black = _classify_black(samples, white_reference)

    # The keyboard's outer edges are boundaries too, and never black keys.
    interior = black[1:-1]
    rotation, confidence = anchor_pitch(interior)

    white_count = boundaries.size - 1
    first_pitch = _first_pitch(rotation, white_count)
    whites = [p for p in range(first_pitch, 109) if p % 12 in WHITE_PITCH_CLASSES]
    if len(whites) < white_count:
        raise CalibrationError(
            f"fitted {white_count} white keys starting at {first_pitch}, which "
            f"runs past the top of the piano"
        )
    last_pitch = whites[white_count - 1]

    x0 = float(boundaries[0])
    layout = KeyboardLayout(
        first_pitch=first_pitch,
        last_pitch=last_pitch,
        x0=x0,
        width=float(boundaries[-1] - boundaries[0]),
    )

    if confidence < cfg.min_confidence:
        raise CalibrationError(
            f"black-key pattern only matches {confidence:.0%} of boundaries; "
            f"the key grid is probably misaligned"
        )

    log.debug(
        "keybed y=%d..%d, period %.3fpx, %d white keys from %d, confidence %.2f",
        top, bottom, period, white_count, first_pitch, confidence,
    )

    return Calibration(
        layout=layout,
        strike_y=top,
        keybed_bottom=bottom,
        white_width=period,
        confidence=confidence,
        diagnostics={
            "boundaries": boundaries.tolist(),
            "black_boundaries": black.tolist(),
            "rotation": rotation,
            "extent": [x_lo, x_hi],
            "white_reference": white_reference,
        },
    )
