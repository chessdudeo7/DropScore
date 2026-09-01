"""Stage 5: turn per-frame tile detections into timed notes.

The central idea, and the reason this pipeline can beat frame-counting: **timing
comes from pixel geometry, not from frame indices.** A frame is 33ms at 30fps, but
sixteenths at 140 BPM are 107ms apart, so rounding onsets to frames would quantize
badly before any musical quantization happens. Instead, once the scroll speed is
known, a tile's distance from the strike line converts directly to seconds, and
the answer is sub-frame accurate.

Two edges, two events:

* a tile's **bottom** edge crossing the strike line is the note-on;
* its **top** edge crossing the same line is the note-off.

Both are extrapolated from frames where that edge is unclipped, then combined by
median. Reading duration from a tile's height instead would fail for any tile
taller than the screen, and would be wrong on every frame where the tile is partly
off the top or already past the strike line.

Scroll speed is measured by phase correlation on the fall area rather than from
the tracked tiles, so it does not depend on detection being complete — and its
spread across frame pairs is a genuine confidence signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np

from .calibrate import Calibration
from .config import Config, DEFAULT
from .notes import Hand, Note, NoteSequence
from .tiles import Palette, Tile, detect_in_frame
from .video import Frame

log = logging.getLogger(__name__)


class TrackingError(RuntimeError):
    """Raised when tiles cannot be tracked or timed."""


@dataclass
class TileTrack:
    """One tile followed across the frames it appears in."""

    pitch: int
    track: int  # palette index
    times: list[float] = field(default_factory=list)
    tops: list[float] = field(default_factory=list)
    bottoms: list[float] = field(default_factory=list)

    def observe(self, tile: Tile) -> None:
        self.times.append(tile.time)
        self.tops.append(tile.top)
        self.bottoms.append(tile.bottom)

    @property
    def length(self) -> int:
        return len(self.times)

    @property
    def last_time(self) -> float:
        return self.times[-1]

    @property
    def last_bottom(self) -> float:
        return self.bottoms[-1]


@dataclass(frozen=True)
class SpeedEstimate:
    """Scroll speed in pixels per second, and how consistent it was."""

    value: float
    spread: float  # median absolute deviation across frame pairs
    samples: int

    @property
    def confidence(self) -> float:
        """1.0 when every frame pair agreed; falls off as they disagree."""
        if self.value <= 0:
            return 0.0
        return float(max(0.0, 1.0 - (self.spread / self.value) * 10))


def estimate_speed(
    frames: Sequence[Frame],
    calibration: Calibration,
    background: np.ndarray | None = None,
    config: Config = DEFAULT,
) -> SpeedEstimate:
    """Measure the scroll speed by phase correlation between consecutive frames.

    Correlation runs on the background-subtracted residual, not the raw frame:
    the fall area is mostly static background, which would otherwise dominate the
    correlation peak and report a speed of zero. The residual contains only the
    tiles, whose motion is the thing being measured.

    Independent of tile detection on purpose — a missed tile should not move the
    number every timestamp is derived from.
    """
    cfg = config.tracking
    if len(frames) < 2:
        raise TrackingError("need at least two consecutive frames to measure speed")

    region = calibration.strike_y
    images = [f.image[:region] for f in frames]
    if background is None:
        background = np.median(np.stack(images), axis=0).astype(np.uint8)

    reference = cv2.cvtColor(background[:region], cv2.COLOR_BGR2GRAY).astype(np.float32)

    def residual(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.abs(gray - reference)

    window: np.ndarray | None = None
    shifts: list[float] = []

    for previous, current in zip(frames, frames[1:]):
        dt = current.time - previous.time
        if dt <= 0:
            continue

        a = residual(previous.image[:region])
        b = residual(current.image[:region])
        if a.std() < cfg.min_residual or b.std() < cfg.min_residual:
            continue  # nothing is falling in this pair

        if window is None:
            window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)

        (_, dy), response = cv2.phaseCorrelate(a, b, window)
        if response < cfg.min_correlation:
            continue

        speed = dy / dt
        if cfg.min_speed <= speed <= cfg.max_speed:
            shifts.append(speed)

    if len(shifts) < cfg.min_speed_samples:
        raise TrackingError(
            f"could not measure a consistent scroll speed "
            f"({len(shifts)} usable frame pairs)"
        )

    values = np.array(shifts)
    speed = float(np.median(values))
    spread = float(np.median(np.abs(values - speed)))
    log.debug("scroll speed %.2f px/s (mad %.2f over %d pairs)", speed, spread, len(values))
    return SpeedEstimate(value=speed, spread=spread, samples=len(values))


def build_tracks(
    detections: Iterable[tuple[Frame, Sequence[Tile]]],
    speed: float,
    calibration: Calibration,
    config: Config = DEFAULT,
) -> list[TileTrack]:
    """Group per-frame detections into per-tile tracks.

    Tiles only move vertically at a known rate, so association is a
    one-dimensional nearest-prediction match within each (pitch, colour) lane.
    """
    cfg = config.tracking
    open_tracks: dict[tuple[int, int], list[TileTrack]] = {}
    finished: list[TileTrack] = []

    for frame, tiles in detections:
        lanes: dict[tuple[int, int], list[Tile]] = {}
        for tile in tiles:
            lanes.setdefault((tile.pitch, tile.track), []).append(tile)

        seen: set[int] = set()

        for lane, lane_tiles in lanes.items():
            candidates = open_tracks.setdefault(lane, [])

            # Score every plausible pairing in this lane, then take them in
            # order of agreement, each tile and each track used once. Matching
            # tile-by-tile instead let two tiles on one key — two strikes of a
            # repeated note, both on screen — attach to the *same* track, whose
            # observations then interleave two different notes and average into
            # one wrong one. Repeated notes are where this pipeline is weakest,
            # so it is the last place to be careless.
            pairings: list[tuple[float, int, int]] = []
            for tile_index, tile in enumerate(lane_tiles):
                for track_index, candidate in enumerate(candidates):
                    dt = frame.time - candidate.last_time
                    if dt <= 0:
                        continue
                    # The bottom edge stops at the strike line while the tile passes.
                    predicted = min(
                        candidate.last_bottom + speed * dt, calibration.strike_y
                    )
                    error = abs(tile.bottom - predicted)
                    tolerance = max(cfg.min_match_px, cfg.match_ratio * speed * dt)
                    if error < tolerance:
                        pairings.append((error, tile_index, track_index))

            pairings.sort()
            used_tiles: set[int] = set()
            used_tracks: set[int] = set()
            for _error, tile_index, track_index in pairings:
                if tile_index in used_tiles or track_index in used_tracks:
                    continue
                candidate = candidates[track_index]
                candidate.observe(lane_tiles[tile_index])
                used_tiles.add(tile_index)
                used_tracks.add(track_index)
                seen.add(id(candidate))

            # Whatever is left is a tile that just appeared.
            for tile_index, tile in enumerate(lane_tiles):
                if tile_index in used_tiles:
                    continue
                track = TileTrack(pitch=tile.pitch, track=tile.track)
                track.observe(tile)
                candidates.append(track)
                seen.add(id(track))

        # Retire tracks that went unmatched this frame.
        for lane, candidates in list(open_tracks.items()):
            still_open = []
            for candidate in candidates:
                if id(candidate) in seen or frame.time - candidate.last_time <= cfg.max_gap:
                    still_open.append(candidate)
                else:
                    finished.append(candidate)
            open_tracks[lane] = still_open

    for candidates in open_tracks.values():
        finished.extend(candidates)

    return finished


def _cross_time(times: Sequence[float], edges: Sequence[float], strike_y: int, speed: float) -> float:
    """When an edge at these positions reaches the strike line."""
    estimates = [t + (strike_y - edge) / speed for t, edge in zip(times, edges)]
    return float(np.median(estimates))


def track_to_note(
    track: TileTrack, speed: float, calibration: Calibration, config: Config = DEFAULT
) -> Note | None:
    """Convert one tile track into a note, or None if it is not usable."""
    cfg = config.tracking
    if track.length < cfg.min_observations:
        return None

    strike = calibration.strike_y
    margin = cfg.edge_margin

    # Only unclipped edges carry timing information.
    bottom_times, bottom_edges = [], []
    top_times, top_edges = [], []
    for time, top, bottom in zip(track.times, track.tops, track.bottoms):
        if bottom < strike - margin:
            bottom_times.append(time)
            bottom_edges.append(bottom)
        if top > margin:
            top_times.append(time)
            top_edges.append(top)

    if not bottom_times:
        # Never seen before it reached the line; its onset is unrecoverable.
        return None

    onset = _cross_time(bottom_times, bottom_edges, strike, speed)

    if top_times:
        offset = _cross_time(top_times, top_edges, strike, speed)
        duration = offset - onset
    else:
        # Top edge never visible: the tile is taller than the screen, so fall
        # back to the tallest observation, which is a lower bound.
        duration = max(b - t for t, b in zip(track.tops, track.bottoms)) / speed

    if duration < cfg.min_duration:
        return None

    hand: Hand = "L" if track.track == 1 else "R"
    return Note(
        onset=max(0.0, onset),
        pitch=track.pitch,
        duration=duration,
        hand=hand,
    )


def tracks_to_notes(
    tracks: Sequence[TileTrack], speed: float, calibration: Calibration, config: Config = DEFAULT
) -> NoteSequence:
    notes = [track_to_note(t, speed, calibration, config) for t in tracks]
    return NoteSequence.of([n for n in notes if n is not None])


def transcribe(
    frames: Iterator[Frame] | Sequence[Frame],
    calibration: Calibration,
    palette: Palette,
    speed: SpeedEstimate | float,
    config: Config = DEFAULT,
) -> NoteSequence:
    """Run detection, tracking and timing over a frame stream."""
    value = speed.value if isinstance(speed, SpeedEstimate) else float(speed)
    if value <= 0:
        raise TrackingError(f"scroll speed must be positive, got {value}")

    detections = (
        (frame, detect_in_frame(frame, palette, calibration, config)) for frame in frames
    )
    tracks = build_tracks(detections, value, calibration, config)
    sequence = tracks_to_notes(tracks, value, calibration, config)

    log.info("%d tracks -> %d notes", len(tracks), len(sequence))
    return sequence
