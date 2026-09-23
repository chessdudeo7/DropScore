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

    # How far the typical pixel of each colour sits from it. A flat tile is one
    # colour and its pixels land on top of it; a gradient-filled one is a ramp,
    # and even its typical pixel sits well out. Measured across the corpus, the
    # median member distance is 0-8 for every flat theme and 13-16 for the
    # gradient one, which is what makes it usable as a signal.
    spreads: np.ndarray | None = None

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

    colors, counts = _cluster(
        pixels,
        cfg.max_palettes,
        cfg.lightness_weight,
        cfg.merge_distance,
        background,
        cfg.duplicate_distance,
        cfg.source_near_tie,
    )
    colors, counts = _fold_blends(colors, counts, background, cfg)

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

    # How tightly each colour's own pixels sit around it, for the acceptance
    # radius in _track_masks.
    kept_colors = colors[keep]
    weighted_pixels = _weighted(pixels[:, None, :], cfg.lightness_weight)[:, 0, :]
    weighted_colors = _weighted(kept_colors[:, None, None, :], cfg.lightness_weight)
    weighted_colors = weighted_colors[:, 0, 0, :]
    member_distance = np.linalg.norm(
        weighted_pixels[:, None, :] - weighted_colors[None, :, :], axis=2
    )
    nearest = member_distance.argmin(axis=1)
    closest = member_distance.min(axis=1)

    # Measured over plausible members only. Every pixel differing from the
    # background is a candidate here, and each is assigned to whichever colour
    # is *nearest* — which for a lane separator or a strike line is not its
    # colour in any meaningful sense, merely the least wrong one. Counting
    # those, a single-tile clip measured a spread of 93 and widened its radius
    # to 233, which matched most of the frame and detected nothing at all.
    #
    # The bound has to be loose enough to keep the far end of a gradient ramp,
    # which is the population this exists to measure: on the gradient theme the
    # ramp reaches 32 against a tolerance of 22.
    window = cfg.color_tolerance * cfg.spread_window
    spreads = np.array(
        [
            float(np.median(closest[(nearest == index) & (closest < window)]))
            if np.any((nearest == index) & (closest < window))
            else 0.0
            for index in range(len(kept_colors))
        ]
    )

    log.debug("palette: %d colours from %d sampled pixels", int(keep.sum()), len(pixels))
    return Palette(
        background=background,
        colors=kept_colors,
        counts=counts[keep],
        spreads=spreads,
    )


def _fold_blends(
    colors: np.ndarray, counts: np.ndarray, background: np.ndarray, cfg
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a colour that is only a blend of the background and another into it.

    Bloom and antialiasing are mixtures of a tile's colour with what lies
    behind it, so they sit on the straight line between the two. Hue alone was
    relied on to fold them back, and it decides with a hard cutoff: below
    ``MIN_CHROMA`` a colour counts as neutral and is never merged. A halo dims
    toward that cutoff by construction, so whether a theme's glow became a
    phantom second voice came down to a fraction of a unit of chroma --
    measured, 11.8 against 12.0, flipped by drawing tiles one pixel smaller.

    On the line is a far tighter test than "same hue" and needs no cutoff: that
    halo sat 0.46 from it, a quarter of the way out from the background. Only
    chromatic parents are considered, so two greys on a dark ground -- a pair of
    voices told apart by lightness alone, and every grey lies on the line from
    black to white -- are still left apart as ``_same_hue`` intends.

    Which of a pair is the blend was once read off their counts, on the
    reasoning that a glow is rarer than the tile it surrounds. It is not: a
    glow surrounds *every* tile, so it is pooled where the tile colours are
    split. A theme that tints its tiles by pitch rather than by hand splits
    them across the whole spectrum, and there the single pooled halo was the
    most common colour in the palette -- 48% of sampled pixels against 22% for
    the largest real colour. Being the most common, it was never offered as a
    child, and it survived as a voice of its own: a second, wider blob around
    every tile, 28 to 38 pixels across where a black key is 15. Those blobs
    resolved onto whichever neighbour they were centred on, and a Liszt etude
    in five flats came back with D, E, G, A and B naturals making up 27% of
    the notes -- the white keys either side of the black ones it actually uses.
    Counts say nothing about which is the blend. The geometry already does:
    a blend lies between the background and its parent, so it is always the
    nearer of the two. Colours are offered as children nearest the background
    first, and may only fold outwards.
    """
    if len(colors) < 2:
        return colors, counts

    counts = counts.copy()
    points = _weighted(colors, cfg.lightness_weight)
    origin = _weighted(background[None, :], cfg.lightness_weight)[0]
    keep = np.ones(len(colors), dtype=bool)

    # Children are visited nearest the background first and may only fold into
    # a colour farther out, but the palette keeps the order it arrived in.
    reach = np.linalg.norm(points - origin, axis=1)
    order = [int(i) for i in np.argsort(reach, kind="stable")]
    for rank, index in enumerate(order):
        for parent in order[rank + 1:]:
            if not keep[parent]:
                continue
            chroma = float(np.hypot(colors[parent][1] - 128.0, colors[parent][2] - 128.0))
            if chroma < MIN_CHROMA:
                continue
            direction = points[parent] - origin
            length = float(np.dot(direction, direction))
            if length <= 0:
                continue
            along = float(np.dot(points[index] - origin, direction)) / length
            off = float(np.linalg.norm(points[index] - (origin + along * direction)))
            if 0.0 < along < 1.0 and off < cfg.blend_tolerance:
                counts[parent] += counts[index]
                keep[index] = False
                break

    return colors[keep], counts[keep]


#: Fixed seed for palette clustering; any constant would do.
PALETTE_SEED = 12345

#: k-means restarts; the most compact result is kept.
PALETTE_ATTEMPTS = 30

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


def _distance_to(color: np.ndarray, other: np.ndarray, lightness_weight: float) -> float:
    return float(np.linalg.norm(
        _weighted(color[None, :], lightness_weight)[0] - _weighted(other[None, :], lightness_weight)[0]
    ))


def _farther_from(
    color: np.ndarray, other: np.ndarray, background: np.ndarray, lightness_weight: float, near_tie: float
) -> bool:
    """Is ``color`` the one farther from the background, with ties broken by lightness?"""
    mine = _distance_to(color, background, lightness_weight)
    theirs = _distance_to(other, background, lightness_weight)
    if abs(mine - theirs) > near_tie * max(mine, theirs):
        return mine > theirs
    return abs(float(color[0] - background[0])) > abs(float(other[0] - background[0]))


def _cluster(
    pixels: np.ndarray,
    k: int,
    lightness_weight: float,
    merge_distance: float,
    background: np.ndarray | None = None,
    duplicate_distance: float = 0.0,
    near_tie: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means in weighted Lab, then merge clusters of near-identical chroma."""
    weighted = np.ascontiguousarray(_weighted(pixels, lightness_weight), dtype=np.float32)
    k = max(1, min(k, len(pixels)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Seeded, so that the same video gives the same palette every time. k-means
    # starts from random centres and OpenCV draws them from a process-wide
    # generator, so two identical calls disagreed: colours moved by up to 3 Lab
    # units between runs, and on one clip the palette held three colours on
    # some runs and two on others. A result that can change on its own cannot
    # be compared with the last one, and every regression check depends on
    # exactly that comparison.
    cv2.setRNGSeed(PALETTE_SEED)
    _, labels, _ = cv2.kmeans(weighted, k, None, criteria, PALETTE_ATTEMPTS, cv2.KMEANS_PP_CENTERS)
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
            # Near-identical colours are one colour whether or not they have a
            # hue to compare: the neutral guard below exists for two grey
            # *voices*, which stand far apart, not for a colour and a copy of
            # itself.
            duplicate = float(np.linalg.norm(color - existing)) < duplicate_distance
            if duplicate or _same_hue(color, existing, merge_distance):
                # Keep one member's colour rather than averaging: the average
                # lands between a tile and its bloom, close enough to the halo
                # that the mask admits it, which turned a whole glow field into
                # detected tiles.
                #
                # And keep the one farther from the background, not the more
                # common. Bloom is the tile's colour blended toward what lies
                # behind it, so the source is the member farther out. Keeping
                # the commoner assumed a tile outnumbers its glow, which filled
                # tiles do and neon outlines do not: a capture of them had 0.7%
                # of its fall area in outline and 7.8% in green haze of the same
                # hue, and the palette became the haze.
                #
                # When the two are too close to call, the brighter against a dark
                # ground (the darker against a light one). Glow is the tile's
                # light fading, so it never outshines its source, but it can be
                # the more saturated: one style strokes its tiles near-white over
                # a violet fill that glows violet, and the fill won 72.6 to 71.3.
                # The fill's colour is the glow's, so the mask bridged every pair
                # of neighbouring tiles and read a phantom note between them.
                # Lightness alone is no rule, though: a vivid tile outweighs its
                # own paler highlight, and chose that instead on two themes.
                if background is not None and _farther_from(
                    color, existing, background, lightness_weight, near_tie
                ):
                    merged_colors[i] = color
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
    #
    # How close is "close" depends on the colour. A gradient-filled tile is not
    # one colour but a ramp, so the discovered colour sits mid-ramp and a fixed
    # radius clips both ends whatever it is set to: measured on such a theme,
    # a tile's lower half read 23 to 32 against a tolerance of 22, and the mask
    # stopped 150px short of the tile's bottom edge. Lightness weighting cannot
    # rescue it — the ramp moves through the hue plane too, and even ignoring
    # lightness entirely the far end still measures 24.
    #
    # So the radius is generous relative to how tightly the colour's own pixels
    # cluster around it. On a flat theme the typical pixel sits 0-8 away and
    # this changes nothing; on the gradient one it sits 13-16, and the radius
    # widens to take in the rest of the ramp. Deliberately keyed to the median
    # rather than an upper percentile: the tail is bloom, which is larger on
    # the flat themes than on the gradient one and must stay excluded.
    # Widening is bounded by the other colours. A colour may reach out as far
    # as it likes into empty space, but not so far that it starts claiming
    # pixels belonging to another voice — so the bound is half the distance to
    # the nearest one, which is the point where the two would meet.
    #
    # That bound is what makes the widening safe to apply generally. On a video
    # whose palette is three shades of one colour, discovered from particle
    # effects rather than two distinct hands, the spread is as large as a
    # genuine gradient's and would widen the radius just as far: measured, from
    # 22 to 31, taking the detected blobs from 63 a frame to 91. Those colours
    # sit 22 apart where a real pair of hands sat 76, and the cap tells them
    # apart without needing to know which is which.
    radius = np.full(len(distances), cfg.color_tolerance, dtype=np.float32)
    if palette.spreads is not None and len(palette.spreads) == len(distances):
        wanted = palette.spreads * cfg.spread_multiple
        if len(targets) > 1:
            flat = targets.reshape(len(targets), -1)
            gaps = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
            np.fill_diagonal(gaps, np.inf)
            wanted = np.minimum(wanted, gaps.min(axis=1) / 2.0)
        # Nor so far that it takes in the background. The other colours bound
        # it, but a palette of one colour had nothing to bound it at all: on a
        # capture of neon outlines its radius grew to 57, the black background
        # sat 41 away, and every pixel of the fall area matched -- one blob, a
        # "tile" on every key, every frame.
        ground = _weighted(palette.background[None, :], cfg.lightness_weight)[0]
        to_ground = np.linalg.norm(targets.reshape(len(targets), -1) - ground[None, :], axis=1)
        wanted = np.minimum(wanted, to_ground / 2.0)
        radius = np.maximum(radius, wanted)

    solid = closest < radius[nearest]

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


def _looks_outlined(strip: np.ndarray, cfg) -> bool:
    """Is this a tile drawn as an outline: stroked down both sides, empty inside?

    Judged by the strongest column within each side third, and by the middle of
    the tile rather than every column between the edges. Reading the outermost
    columns and averaging all the rest suited thin strokes on tall tiles, and
    failed a style drawn thick and rounded: a neon outline's rounded corners
    keep even its stroke columns short of full, and on a short tile the top and
    bottom strokes are much of its height, so every interior column read about
    half full. Those tiles were too full to be outlines and too empty to be
    solid, and a whole capture of them came back as two notes.

    The middle of an outline is empty whatever its stroke or its corners, and
    the middle of a filled tile is not. Near-empty fringe columns are trimmed
    first: sparks brushing a tile's edge widened its box by columns 4% full.
    """
    columns = strip.mean(axis=0)
    occupied = np.flatnonzero(columns >= cfg.outline_fringe_ratio)
    if occupied.size < 3:
        return False
    strip = strip[:, occupied[0] : occupied[-1] + 1]
    height, width = strip.shape
    if width < 3 or height < 3:
        return False

    # Sides and middle are read over the same rows. Measured down the whole
    # height instead, a box holding two filled tiles with a gap between them
    # passed as an outline: its side columns are full for both tiles and so
    # read full on average, while its middle rows -- the gap -- read empty.
    # That merged the pair into one note, and on a plain two-voice clip it cost
    # a quarter of the notes their staff.
    top, bottom = int(height * 0.25), max(int(height * 0.75), int(height * 0.25) + 1)
    band = strip[top:bottom]

    columns = band.mean(axis=0)
    side = max(1, int(round(width * 0.3)))
    if min(float(columns[:side].max()), float(columns[-side:].max())) < cfg.outline_side_ratio:
        return False

    # An outline is closed: it is stroked across the top and the bottom as well
    # as down the sides. Only one of the two is required, because a tile
    # reaching past the top of the fall area or down through the strike line
    # keeps just the other. Without this, two thin tiles on neighbouring keys
    # that merged into one box read as an outline -- stroked at either side,
    # empty between -- and were kept whole as a single note on the wrong key.
    rows = strip.mean(axis=1)
    edge = max(1, int(round(height * 0.1)))
    if max(float(rows[:edge].max()), float(rows[-edge:].max())) < cfg.outline_side_ratio:
        return False

    left, right = int(width * 0.35), max(int(width * 0.65), int(width * 0.35) + 1)
    return float(band[:, left:right].mean()) < cfg.outline_fill_ratio


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
    # Each side is judged by its strongest column rather than its outermost.
    # Exactly which column a stroke lands in is not something the caller can
    # settle: a key boundary falls between pixels, and a blob can run a column
    # wide of its key through antialiasing. Reading the outermost column alone,
    # a stroke one column in -- or split across two, at 1.0 and 0.75 -- was
    # read as no stroke at all, and the tile was handed to the solidity rule
    # meant for filled ones. That cost every tile on one key for a whole clip.
    hollow = _looks_outlined(strip, cfg)
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

    # A tile is a solid block of its colour. Effects are not: a video with
    # sparks streaming off the strike line produced ninety blobs a frame where
    # the music had eight, because a wisp crossing a key column fills enough of
    # a row to pass row_fill_ratio even though it fills little of the region it
    # ends up spanning.
    #
    # Solidity is what tells them apart, and height cannot: measured, the
    # trails are as tall as tiles, and rejecting anything short enough to catch
    # them cost 9% of genuine detections while leaving the trails behind. Real
    # tiles measure 0.79 to 0.92 filled across every theme; the trails sit at
    # 0.28 to 0.57. Outlined tiles are hollow by design and measure 0.18, which
    # is why this is checked after the hollow case has already returned.
    return [
        (a, b)
        for a, b in spans
        if b - a >= cfg.min_tile_height
        and float(strip[a - y0 : b - y0].mean()) >= cfg.min_solidity
    ]


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

            # A blob no wider than a white key is one tile, and belongs to the
            # key it is centred on. Coverage is for telling merged tiles apart,
            # and it assumed a black key's tile is as narrow as the key: one
            # style draws them 21px wide over 15.5px keys, and centred on B-flat
            # such a tile still covered 62% of the B beside it. Hovering either
            # side of the 60% bar, it flickered between the two keys and left a
            # phantom B under every B-flat.
            # It must still cover most of that key: a spark's wisp is centred on
            # some key too, and coverage was what kept it from being a note.
            if w <= calibration.white_width:
                nearest = calibration.layout.nearest_key(x + w / 2)
                pitches = []
                if nearest is not None:
                    left, right = calibration.layout.key_span(nearest)
                    covered = (min(x + w, right) - max(x, left)) / (right - left)
                    if covered >= cfg.min_coverage:
                        pitches = [nearest]
            else:
                pitches = calibration.layout.keys_covered(x, x + w, cfg.min_coverage)
                if pitches:
                    pitches = _with_hidden_black_keys(
                        mask, (x, y, x + w, y + h), pitches, calibration, config
                    )
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
                if len(pitches) == 1:
                    # The blob is this key's alone, so take it whole. Cutting
                    # it to the key's span loses whatever the span's rounded
                    # edge trims away, and an outlined tile keeps its strokes
                    # exactly there: on a black key, whose tile fills its span
                    # almost exactly, the right-hand stroke fell outside the
                    # cut entirely. Nothing downstream can recover a stroke
                    # that was never in the strip.
                    x0, x1 = x, x + w
                else:
                    lo, hi = _exposed_span(pitch, pitches, calibration)
                    x0 = max(x, int(round(lo)))
                    x1 = min(x + w, int(round(hi)))
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


def _with_hidden_black_keys(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    pitches: list[int],
    calibration: Calibration,
    config: Config,
) -> list[int]:
    """Add a black key merged into its white neighbour's blob, if its tile is there.

    A black key is not claimed by span alone when a white neighbour is, because
    two merged white tiles cover the boundary between them and would otherwise
    read as a phantom accidental. But a real black tile merged with a white one
    was then suppressed along with the phantoms.

    The pixels tell them apart. Two merged white tiles fill the boundary only
    where the white keys' own columns are filled too. A black tile fills its
    lane on rows where neither neighbour's uncovered columns are -- an F#4 held
    for five seconds beside a G4 that started later filled rows the G4 had not
    reached yet.
    """
    from .keyboard import is_black  # noqa: PLC0415

    cfg = config.tiles
    x0, y0, x1, y1 = box
    layout = calibration.layout
    claimed = list(pitches)

    def filled_rows(lo: float, hi: float, ratio: float) -> np.ndarray:
        a, b = max(x0, int(round(lo))), min(x1, int(round(hi)))
        if b <= a:
            return np.zeros(y1 - y0, dtype=bool)
        return mask[y0:y1, a:b].mean(axis=1) >= ratio

    def longest_run(rows: np.ndarray) -> int:
        best = run = 0
        for filled in rows:
            run = run + 1 if filled else 0
            best = max(best, run)
        return best

    # A tile is solid across its lane and unbroken down it; sparks scattered
    # through the lane are neither. Counting filled rows alone claimed black
    # keys for them -- three phantom accidentals in one frame of a spark clip.
    tall_enough = max(cfg.min_tile_height, int(calibration.white_width * 0.5))

    for pitch in layout.pitches:
        if not is_black(pitch) or pitch in claimed:
            continue
        left, right = layout.key_span(pitch)
        if right <= x0 or left >= x1:
            continue
        # A black tile spans its lane, so a blob holding one covers the lane.
        # Judged only over the sliver of lane that falls inside the blob, the
        # left-hand stroke of an outlined D4 -- four pixels of C#4's sixteen --
        # was solid on every row where the D4's hollow middle was not, and a
        # phantom C#4 was read under the second D4 of every bar.
        if (min(right, x1) - max(left, x0)) / (right - left) < cfg.min_coverage:
            continue

        lane = filled_rows(left, right, cfg.min_solidity)
        neighbours = np.zeros_like(lane)
        for white in (pitch - 1, pitch + 1):
            if white in claimed:
                lo, hi = _exposed_span(white, claimed + [pitch], calibration)
                neighbours |= filled_rows(lo, hi, cfg.row_fill_ratio)
        if longest_run(lane & ~neighbours) >= tall_enough:
            claimed.append(pitch)

    return sorted(claimed)


def _exposed_span(
    pitch: int, together: Sequence[int], calibration: Calibration
) -> tuple[float, float]:
    """The columns of a key that no black key in the same blob also covers.

    A black key's lane sits across the boundary between two white keys, so its
    tile overlaps a third or so of each neighbour's columns. Judged over its
    full width, a white key inside a blob with a black neighbour counts that
    neighbour's tile as its own wherever the two are side by side -- and when
    the neighbour is held longer, the white key reads as reaching all the way
    down to the strike line. On a clip of held chords a G4 lasting five seconds
    sat beside an F#4 lasting as long from earlier: G4 read as filling its
    whole column, its lower edge never seemed to fall, no onset could be read,
    and the note vanished, twice in each of two clips.

    Only keys of the other colour in the same blob are set aside, so a key
    standing alone keeps every column it has.

    A black key sets aside its white neighbour's columns in turn. Its lane lies
    over a third of each neighbour, and a renderer drawing black tiles as wide
    as the lane puts two thirds of it on the white tile's columns: judged over
    them, a D#5 zigzagging with E5 -- E D# E D# E, the turn Fur Elise is made of
    -- read as filled from the top E to the bottom one, one D#5 in place of two,
    timed by the last E's lower edge and so a sixteenth early.
    """
    from .keyboard import is_black  # noqa: PLC0415

    left, right = calibration.layout.key_span(pitch)
    black = is_black(pitch)

    lo, hi = left, right
    for other in together:
        if other == pitch or is_black(other) == black:
            continue
        other_left, other_right = calibration.layout.key_span(other)
        if other_right <= left or other_left >= right:
            continue
        if other_left <= left:
            lo = max(lo, other_right)
        else:
            hi = min(hi, other_left)

    # Too little left to judge by, and the full width is the better evidence.
    # A black key has less to begin with: the D#5 above kept 0.21 of a white
    # key's width, and a black key between two claimed neighbours keeps none.
    floor = 0.15 if black else 0.25
    if hi - lo < calibration.white_width * floor:
        return left, right
    return lo, hi


def detect(
    frames: Sequence[Frame],
    calibration: Calibration,
    palette: Palette | None = None,
    config: Config = DEFAULT,
) -> tuple[Palette, list[list[Tile]]]:
    """Detect tiles across a sequence of frames."""
    palette = palette or discover_palette(frames, calibration, config)
    return palette, [detect_in_frame(f, palette, calibration, config) for f in frames]
