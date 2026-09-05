"""Stage 3: fit a KeyboardLayout to a video.

Everything downstream is a function of "which pixel column is which MIDI note",
so an error of one key transposes the entire transcription. The fit is therefore
built from three independent signals, each checked against the next:

1. **Where the keybed is** — from how structured each row of the static
   background is. A keybed row crosses white and black keys and so varies
   hugely; a fall-area row is near-uniform. Motion looks like the obvious
   signal and is not: struck keys light up over the keybed's full height, so
   it is nearly as busy as the falling area.
2. **How wide a white key is** — from the spatial frequency of the separator
   edges, taken below the black keys where only white keys exist. The FFT bin
   is only a starting point; real key widths do not divide the frame exactly,
   so period and offset are then refined against how well the gridlines sit on
   the edges.
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


def _row_coverage(gray: np.ndarray) -> np.ndarray:
    """Share of the frame's width over which each row carries structure."""
    edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    strong = edges > max(1.0, float(np.percentile(edges, 90)) * 0.25)

    # Judge in coarse blocks so a key boundary counts once rather than as the
    # two or three columns its edge happens to span. Coarse enough, too, that
    # the measure does not depend on how many keys are on screen: a 49-key
    # board has 29 white keys, which over 64 blocks leaves coverage at 0.45
    # and gates out the very keybed it is meant to find.
    width = gray.shape[1]
    block = max(1, width // 16)
    usable = (width // block) * block
    blocks = strong[:, :usable].reshape(len(gray), -1, block).any(axis=2)
    return blocks.mean(axis=1)


def find_keybed(frames: Sequence[Frame], background: np.ndarray, config: Config) -> tuple[int, int]:
    """Locate the keybed as the quiet band at the bottom of the frame."""
    cfg = config.calibration

    if float(row_activity(frames, background).max()) <= 1e-6:
        raise CalibrationError("nothing moves in this video; no tiles to track")

    # Split on how *structured* each row of the static background is, not on how
    # much it moves. Motion was the obvious choice and is wrong: struck keys
    # light up over the keybed's full height, so it is nearly as busy as the
    # fall area (measured: 3.9 against 4.5), and the sharpest drop in activity
    # falls where the black keys end rather than at the strike line.
    #
    # Structure separates them cleanly instead. A keybed row crosses white and
    # black keys, so its spread is large; a fall-area row is near-uniform
    # background. Across every theme that is a spread of ~73 against ~2-6.
    gray = _gray(background).astype(np.float32)
    structure = gray.std(axis=1)
    height = structure.shape[0]

    # Structure alone is not enough once the music slows down. A piece of long
    # held chords in a narrow register keeps the same few columns lit for most
    # of the video, so the temporal median keeps those tiles: they *become*
    # background, and their rows read as structured. Measured on such a clip
    # the keybed band ran from row 0 to row 492 — 68% of the frame — and
    # calibration gave up.
    #
    # What still separates them is how far that structure reaches. A keybed row
    # crosses every key on the board, so it is busy across essentially the whole
    # width; a few static bars are busy in a narrow strip. Measured on the same
    # clip: 0.98 of the width against 0.21.
    coverage = _row_coverage(gray)
    structure = structure * (coverage >= cfg.min_keybed_coverage)

    # Find the keybed as a *band*, not as a split with everything below it.
    # Real videos put the keyboard across the middle of the frame, with the
    # player's hands and a title card underneath — so "everything below the
    # strike line is keyboard" is simply not the shape of the picture, and the
    # split formulation could not describe it at all.
    baseline = float(np.percentile(structure, 20))
    peak = float(structure.max())
    if peak - baseline < cfg.min_keybed_contrast:
        raise CalibrationError(
            "no row of keys under a plain falling area; this may not be a "
            "falling-tile piano video"
        )

    threshold = baseline + (peak - baseline) * cfg.keybed_band_ratio
    runs = _runs_above(structure, threshold, cfg.min_keybed_px)
    if not runs:
        raise CalibrationError("could not find a band of keys in this video")

    # Rank by total structure — mean times depth — not by mean alone. Hands
    # score 17 against a keybed's 43 and lose either way, but a caption burned
    # into the corner is white text on black and denser per row than a keyboard
    # is: measured on a real video, 53.7 against 43.9 over fourteen rows. What
    # separates them is that a keybed is deep as well as busy, so the totals are
    # 4527 against 752.
    top, band_end = max(
        runs, key=lambda run: float(structure[run[0] : run[1]].sum())
    )

    # Several renderers shade the first few rows of the keybed, which weakens
    # their structure enough to start the band just below the true edge. Walk
    # back up while the rows still look like keys rather than background.
    floor = float(structure[top:band_end].mean()) * cfg.keybed_edge_ratio
    limit = max(0, top - max(2, int(height * cfg.keybed_edge_max)))
    while top > limit and structure[top - 1] > floor:
        top -= 1

    # The band that stands out is the part crossed by black keys; below them
    # the keybed is near-uniform white and barely varies, so it does not show
    # up as structure at all. Black keys cover a known fraction of a keyboard's
    # depth, which is enough to recover the rest.
    bottom = min(height, top + int(round((band_end - top) / cfg.black_height_ratio)))

    # That ratio only holds while the band really is just the black keys. Where
    # the white key region is busy enough to join the band — a sustained piece
    # keeps far more keys lit — scaling it up again overshoots, and on one clip
    # ran the keybed 91px past the bottom of the keyboard into the dark strip
    # below, where the white reference sampled 4 instead of 106.
    #
    # Below the keyboard there are no keys, so the structure stops reaching
    # across the frame. Walk down to where that happens and take the closer of
    # the two, which also recovers the bottom edge exactly when the ratio is
    # right rather than merely approximately.
    edge = band_end
    while edge < height and coverage[edge] >= cfg.min_keybed_coverage:
        edge += 1
    if edge > band_end:
        bottom = min(bottom, edge)

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


def _runs_above(profile: np.ndarray, threshold: float, minimum: int) -> list[tuple[int, int]]:
    """Contiguous stretches of ``profile`` above ``threshold``, as [start, end)."""
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(profile):
        if value > threshold and start is None:
            start = index
        elif value <= threshold and start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None

    if start is not None and len(profile) - start >= minimum:
        runs.append((start, len(profile)))
    return runs


def _band(image: np.ndarray, top: int, bottom: int, lo: float, hi: float) -> np.ndarray:
    """Slice a fractional depth range out of the keybed."""
    height = bottom - top
    y0 = top + int(height * lo)
    y1 = max(y0 + 1, top + int(height * hi))
    return image[y0:y1]


def estimate_period(
    profile: np.ndarray,
    min_px: float,
    max_px: float,
    harmonic_tolerance: float = 0.75,
) -> tuple[float, float]:
    """Dominant period and first peak offset of a near-periodic 1-D profile.

    Uses the FFT rather than peak-finding: separators can be missed or doubled by
    compression, but the dominant frequency survives that, and the phase gives a
    sub-pixel offset for free.

    ``harmonic_tolerance`` is the share of the peak magnitude a lower frequency
    must reach to be preferred over it — see the note below on impulse trains.
    """
    cfg_harmonic = harmonic_tolerance
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

    # Take the *lowest* frequency that explains the profile, not the strongest.
    # A row of sharp separator lines is close to an impulse train, whose
    # harmonics all carry similar energy, so argmax picks among them
    # arbitrarily — and picking the third harmonic would fit a key grid three
    # times too fine, mismapping every column to the wrong pitch.
    strong = np.flatnonzero(magnitudes >= magnitudes.max() * cfg_harmonic)
    k = k_lo + int(strong[0])

    # The FFT only resolves periods that divide the width exactly, and real key
    # widths do not: a true 18.00px grid in a 960px row lands between bins 53
    # and 54, giving 18.11. That is 0.11px of error per key, which accumulates
    # to most of a key across 52 of them — enough to mismap pitches at the far
    # end. The bin is only a starting point; both period and offset are then
    # refined against the thing actually wanted, which is how well the
    # gridlines sit on the edges.
    return _refine_grid(profile, length / k)


def _refine_grid(
    profile: np.ndarray, coarse: np.ndarray | float, span: float = 0.04, steps: int = 48
) -> tuple[float, float]:
    """Search near ``coarse`` for the (period, offset) best aligned to the peaks."""
    length = profile.shape[0]
    weights = np.clip(profile - float(np.median(profile)), 0, None)

    best_score, best = -np.inf, (float(coarse), 0.0)
    for period in np.linspace(coarse * (1 - span), coarse * (1 + span), steps):
        positions = np.arange(0.0, length, period)
        for offset in np.linspace(0.0, period, steps, endpoint=False):
            index = np.round(positions + offset).astype(int)
            index = index[(index >= 0) & (index < length)]
            score = float(weights[index].sum()) / max(len(index), 1)
            if score > best_score:
                best_score, best = score, (float(period), float(offset))

    period, offset = best
    return period, offset % period


def _boundary_positions(offset: float, period: float, x_lo: float, x_hi: float) -> np.ndarray:
    """Grid lines bracketing ``[x_lo, x_hi]``, not merely those inside it.

    The keyboard's two outer edges are key boundaries as much as any separator,
    but nothing is drawn there — the board simply ends. Starting at the first
    gridline *inside* the detected extent therefore discarded the outermost key
    at each end, and every pitch shifted with them.
    """
    first = offset + np.floor((x_lo - offset) / period) * period
    count = int(np.ceil((x_hi - first) / period)) + 1
    return first + np.arange(max(count, 0)) * period


def _keyboard_extent(
    keybed: np.ndarray, background_level: float, config: Config
) -> tuple[int, int]:
    """Columns the keyboard actually occupies.

    Measured as difference from the fall area's background, not as edge energy.
    Edges were the wrong signal: the outermost key has no separator line on its
    far side, so both ends of the board registered as empty and were cut from
    the grid — losing two keys at each end and shifting every pitch with them.
    """
    difference = np.abs(keybed.mean(axis=0) - background_level)
    threshold = float(difference.max()) * config.calibration.extent_ratio
    inside = np.flatnonzero(difference > threshold)
    if inside.size < 2:
        raise CalibrationError("could not find the horizontal extent of the keyboard")
    return int(inside[0]), int(inside[-1])


def _classify_black(values: np.ndarray, white_reference: float) -> np.ndarray:
    """Split boundary samples into black/not-black by distance to white."""
    # Clip at the known white brightness first. A key lit by the player, or a
    # hand crossing the board, samples far brighter than an unplayed white key
    # — measured at 209 against 85 on a real recording — and two-means on the
    # raw values split those outliers off as their own cluster, leaving every
    # real black key on the "white" side of the line. Nothing is brighter than
    # white for this purpose, so folding them down costs no information and
    # makes the split depend on the black keys rather than on the highlights.
    values = np.minimum(values, white_reference)

    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        raise CalibrationError("keybed shows no black keys; cannot anchor pitch")

    # Two-means on a 1-D signal converges in a couple of passes.
    centres = np.array([lo, hi], dtype=np.float64)
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
    # A 1px separator gives an antisymmetric Sobel response, so its magnitude
    # peaks on *both* sides with a dip on the line itself. Left as a doublet the
    # grid latches onto one half and sits a pixel or two off true, which is
    # enough to drop a key at the end of the board. Smoothing merges each pair
    # back into a single peak centred on the line.
    edges = np.convolve(edges, np.ones(3, dtype=np.float32) / 3.0, mode="same")

    x_lo, x_hi = _keyboard_extent(
        gray[top:bottom], float(np.median(gray[:top])), config
    )
    period, offset = estimate_period(edges, cfg.min_key_px, cfg.max_key_px)
    boundaries = _boundary_positions(offset, period, x_lo, x_hi)
    if boundaries.size < 2:
        raise CalibrationError("not enough key boundaries to fit a keyboard")

    # Sample the upper keybed at each interior boundary: dark means a black key.
    upper = _band(gray, top, bottom, *cfg.black_band)
    half = max(1, int(period * cfg.sample_ratio / 2))
    # The boundaries bracket the keyboard, so the outermost pair can sit beyond
    # the frame edge. Clamping to a window that is always at least one column
    # wide keeps those from slicing to nothing and sampling as NaN, which would
    # otherwise poison the comparison against the white reference and misread
    # every key on the board as white.
    width = upper.shape[1]
    samples = np.array(
        [
            upper[:, lo : max(lo + 1, min(width, int(x) + half + 1))].mean()
            for x in boundaries
            for lo in (min(max(0, int(x) - half), width - 1),)
        ]
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
