"""Stage 4: find the falling tiles and say which key each one belongs to.

Three problems, in order of how much trouble they cause:

**Bloom.** Renderers glow heavily, and the halo is a blend of tile colour and
background. Thresholding on "differs from the background" therefore inflates every
blob. Classifying instead on "is close to a known tile colour" excludes the halo
for free, because a blend sits far from both endpoints. That is why the palette is
discovered first and used as the mask, rather than a simple difference.

**Merged tiles.** Adjacent keys played together touch horizontally; repeated notes
touch vertically. Both arrive as one connected region and must be split — the
horizontal case on the key grid, the vertical case at rows where the region
thins out.

**Palette drift.** Gradient-filled tiles vary in lightness down their body, and
bloom adds a whole ramp of dimmer shades, so clustering on colour alone splits
one tile into several palettes. Clusters of the same *hue* are merged instead:
dimming pulls a colour toward the neutral axis without turning it, which is why
merging by chroma distance did not work — measured, two shades of one teal sat
20 apart in chroma but 4 degrees apart in hue.

Two guards fall out of that. Neutral colours are never merged, since two greys
are two voices told apart by lightness and merging them would discard one hand
entirely. And a colour the background itself would match is dropped: a dim halo
can cluster into something nearer the background than the mask's own tolerance,
at which point it discriminates nothing and matches the whole frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .calibrate import Calibration
from .config import Config, DEFAULT
from .video import Frame

log = logging.getLogger(__name__)


class TileError(RuntimeError):
    """Raised when no tiles can be found."""


@dataclass(frozen=True)
class Palette:
    """The tile colours a video uses, discovered from its own pixels."""

    background: np.ndarray  # Lab, shape (3,)
    colors: np.ndarray  # Lab, shape (n, 3)
    counts: np.ndarray  # pixels assigned to each colour

    @property
    def track_count(self) -> int:
        return len(self.colors)


@dataclass(frozen=True)
class Tile:
    """One tile seen in one frame, already resolved to a key."""

    frame: int
    time: float
    pitch: int
    top: float  # pixel row of the tile's top edge
    bottom: float  # pixel row of its bottom edge, clipped at the strike line
    track: int  # palette index; mapped to a hand in stage 7
    left: float
    right: float

    @property
    def height(self) -> float:
        return self.bottom - self.top


def _to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)


def _weighted(lab: np.ndarray, lightness_weight: float) -> np.ndarray:
    """Scale down L so chroma dominates distance comparisons."""
    scaled = lab.copy()
    scaled[..., 0] *= lightness_weight
    return scaled


def discover_palette(
    frames: Sequence[Frame], calibration: Calibration, config: Config = DEFAULT
) -> Palette:
    """Find the background and tile colours from the fall area."""
    cfg = config.tiles
    region = [f.image[: calibration.strike_y] for f in frames]
    if not region or region[0].size == 0:
        raise TileError("no fall area above the strike line")

    background_bgr = np.median(np.stack(region), axis=0).astype(np.uint8)
    background_lab = _to_lab(background_bgr)
    background = np.median(background_lab.reshape(-1, 3), axis=0)

    # Pixels that differ from the static background are tile candidates.
    samples = []
    for image in region:
        lab = _to_lab(image)
        distance = np.linalg.norm(
            _weighted(lab, cfg.lightness_weight)
            - _weighted(background_lab, cfg.lightness_weight),
            axis=2,
        )
        pixels = lab[distance > cfg.background_distance]
        if pixels.size:
            samples.append(pixels)

    if not samples:
        raise TileError("nothing differs from the background; no tiles to detect")

    pixels = np.concatenate(samples)
    if len(pixels) > cfg.max_sample_pixels:
        step = len(pixels) // cfg.max_sample_pixels
        pixels = pixels[::step]

    colors, counts = _cluster(pixels, cfg.max_palettes, cfg.lightness_weight, cfg.merge_distance)

    # Drop colours too rare to be a hand; they are usually antialiasing.
    keep = counts >= counts.sum() * cfg.min_palette_share

    # And drop any that the background itself would match. A dim halo can
    # cluster into a "colour" sitting closer to the background than the mask's
    # own tolerance, at which point it stops discriminating anything: measured
    # on the bloom-heavy theme, such a colour matched 95% of the fall area.
    separation = np.linalg.norm(
        _weighted(colors, cfg.lightness_weight)
        - _weighted(background[None, :], cfg.lightness_weight),
        axis=1,
    )
    keep &= separation >= cfg.color_tolerance

    if not keep.any():
        raise TileError("no tile colour is common enough to be a voice")

    log.debug("palette: %d colours from %d sampled pixels", int(keep.sum()), len(pixels))
    return Palette(background=background, colors=colors[keep], counts=counts[keep])


#: Below this Lab chroma a colour has no meaningful hue — greys and near-whites.
#: Such colours are compared by lightness rather than merged by hue.
MIN_CHROMA = 12.0


def _same_hue(a: np.ndarray, b: np.ndarray, tolerance_degrees: float) -> bool:
    """Whether two Lab colours are the same hue, ignoring how bright or pale."""
    chroma_a = float(np.hypot(a[1] - 128.0, a[2] - 128.0))
    chroma_b = float(np.hypot(b[1] - 128.0, b[2] - 128.0))

    if chroma_a < MIN_CHROMA or chroma_b < MIN_CHROMA:
        # At least one is neutral, so it has no hue to compare. Leave them
        # alone: two greys are two voices told apart by lightness, and merging
        # them would keep one hand's colour and discard the other's, losing
        # every note that hand played.
        return False

    difference = np.degrees(
        np.arctan2(a[2] - 128.0, a[1] - 128.0) - np.arctan2(b[2] - 128.0, b[1] - 128.0)
    )
    return bool(abs((difference + 180.0) % 360.0 - 180.0) < tolerance_degrees)


def _cluster(
    pixels: np.ndarray, k: int, lightness_weight: float, merge_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """k-means in weighted Lab, then merge clusters of near-identical chroma."""
    weighted = np.ascontiguousarray(_weighted(pixels, lightness_weight), dtype=np.float32)
    k = max(1, min(k, len(pixels)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(weighted, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    # Take true (unweighted) means so the stored colours are usable directly.
    present = [i for i in range(k) if np.any(labels == i)]
    colors = np.stack([pixels[labels == i].mean(axis=0) for i in present])
    counts = np.array([int((labels == i).sum()) for i in present])

    # A gradient tile spans lightness but holds hue, so merge by hue *angle*.
    # Chroma distance was the wrong measure of that: dimming a colour pulls it
    # toward the neutral axis, shortening the vector without turning it, so
    # aurora's two shades of one teal sat 20 apart in chroma while being 4
    # apart in degrees — and each hand split into two palettes.
    merged_colors: list[np.ndarray] = []
    merged_counts: list[int] = []
    for color, count in sorted(zip(colors, counts), key=lambda pair: -pair[1]):
        for i, existing in enumerate(merged_colors):
            if _same_hue(color, existing, merge_distance):
                # Keep the dominant member's colour rather than averaging.
                # Clusters arrive largest-first, so `existing` is the tile's own
                # colour and `color` is usually its bloom or a shaded end of a
                # gradient. Averaging the two lands between them — close enough
                # to the halo that the mask then admits it, which turned the
                # whole glow field into detected tiles.
                merged_counts[i] += count
                break
        else:
            merged_colors.append(color)
            merged_counts.append(int(count))

    return np.stack(merged_colors), np.array(merged_counts)


def _track_masks(
    image: np.ndarray, palette: Palette, calibration: Calibration, config: Config
) -> list[np.ndarray]:
    """One binary mask per palette colour, for the area above the strike line."""
    cfg = config.tiles
    lab = _to_lab(image[: calibration.strike_y])
    weighted = _weighted(lab, cfg.lightness_weight)

    targets = _weighted(palette.colors[:, None, None, :], cfg.lightness_weight)
    distances = np.linalg.norm(weighted[None, ...] - targets, axis=3)

    nearest = distances.argmin(axis=0)
    closest = distances.min(axis=0)

    # Only pixels genuinely close to a tile colour count. Bloom is a blend of
    # tile and background, so it sits far from both and drops out here.
    solid = closest < cfg.color_tolerance

    # No morphological opening. A 3x3 open erodes a pixel in every direction,
    # which removes speckle but also annihilates any stroke thinner than three
    # pixels — and an outlined tile's horizontal edges are two pixels thick.
    # Losing them broke each outline into two disconnected vertical bars, each
    # narrower than the minimum tile width and so discarded entirely. Specks are
    # already excluded by the height and width filters on each contour, which
    # cost nothing and do not damage real tiles.
    return [
        ((nearest == index) & solid).astype(np.uint8)
        for index in range(palette.track_count)
    ]


def _split_vertically(mask: np.ndarray, box: tuple[int, int, int, int], config: Config) -> list[tuple[int, int]]:
    """Split a tall region at rows where it thins out.

    Repeated notes on one key render as separate tiles a pixel or two apart. They
    touch after antialiasing, and a tracker that misses the seam reports one long
    note instead of several — the most common way these readers under-count.
    """
    x0, y0, x1, y1 = box
    strip = mask[y0:y1, x0:x1]
    if strip.size == 0:
        return []

    cfg = config.tiles
    fill = strip.mean(axis=1)

    # An outlined tile is hollow: both side edges drawn down its full height,
    # nothing between them. Judged by columns rather than rows because a tile
    # clipped by the top of the frame loses its top edge but keeps its sides.
    #
    # Sparseness alone was not enough: a key whose neighbour bleeds into its
    # edge column is also sparse, and calling that an outline kept it whole and
    # handed it the neighbour's full height.
    columns = strip.mean(axis=0)
    hollow = (
        columns.size > 2
        and columns[0] >= cfg.outline_edge_ratio
        and columns[-1] >= cfg.outline_edge_ratio
        and float(columns[1:-1].mean()) < cfg.outline_fill_ratio
    )
    if hollow:
        return [(y0, y1)] if y1 - y0 >= cfg.min_tile_height else []

    filled = fill >= cfg.row_fill_ratio

    spans: list[tuple[int, int]] = []
    start = None
    for row, on in enumerate(filled):
        if on and start is None:
            start = row
        elif not on and start is not None:
            spans.append((y0 + start, y0 + row))
            start = None
    if start is not None:
        spans.append((y0 + start, y0 + len(filled)))

    return [(a, b) for a, b in spans if b - a >= cfg.min_tile_height]


def detect_in_frame(
    frame: Frame, palette: Palette, calibration: Calibration, config: Config = DEFAULT
) -> list[Tile]:
    """Every tile visible in one frame, resolved to keys."""
    cfg = config.tiles
    tiles: list[Tile] = []

    for track, mask in enumerate(_track_masks(frame.image, palette, calibration, config)):
        # External contours only: an outlined tile is a ring, and its outer
        # contour is exactly the tile's edge.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < cfg.min_tile_height or w < calibration.white_width * cfg.min_tile_width_ratio:
                continue

            pitches = calibration.layout.keys_covered(x, x + w, cfg.min_coverage)
            if not pitches:
                continue

            # Split each key over its *own* columns. Measuring row fill across
            # the whole blob and sharing the result gets adjacent keys of
            # different lengths wrong in both directions: the long one is cut
            # where the short one ends, and the short one inherits the long
            # one's height. Chords whose voices differ in length are ordinary
            # music, so this is not an edge case.
            for pitch in pitches:
                left, right = calibration.layout.key_span(pitch)
                x0 = max(x, int(round(left)))
                x1 = min(x + w, int(round(right)))
                if x1 <= x0:
                    continue

                for top, bottom in _split_vertically(mask, (x0, y, x1, y + h), config):
                    tiles.append(
                        Tile(
                            frame=frame.index,
                            time=frame.time,
                            pitch=pitch,
                            top=float(top),
                            bottom=float(bottom),
                            track=track,
                            left=left,
                            right=right,
                        )
                    )

    return tiles


def detect(
    frames: Sequence[Frame],
    calibration: Calibration,
    palette: Palette | None = None,
    config: Config = DEFAULT,
) -> tuple[Palette, list[list[Tile]]]:
    """Detect tiles across a sequence of frames."""
    palette = palette or discover_palette(frames, calibration, config)
    return palette, [detect_in_frame(f, palette, calibration, config) for f in frames]
