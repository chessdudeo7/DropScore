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
spread across frame pairs is a genuine confidence signal. The correlator's own
sub-pixel offset is measured against known displacements of the video's own
frames and subtracted; half a pixel per frame is invisible to look at and worth
a second of drift over a clip.
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
from .video import Frame, VideoReader

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


def _robust_mean(values: np.ndarray) -> float:
    """Mean of the measurements that are not obvious outliers.

    The mean, not the median, because the underlying displacements are
    quantised. A renderer places a tile at whole pixels, so between frames it
    moves 5px or 4px where the true rate is 4.708, and the mixture averages to
    the truth. Where several tiles are on screen their sub-pixel phases cancel
    and the correlation peak lands on a continuous value, so either statistic
    works; where one tile dominates, the two clusters show through and the
    median snaps to whichever is more populous — measured 4.95 against a true
    4.708, a 5% error that put every onset 51ms early.

    The median was there to survive outliers, so that job is done first and
    explicitly. The bound is a multiple of the middle, not of the spread: when
    one cluster holds most of the measurements the spread is tiny, and a bound
    scaled to it discards the other cluster — which is the median's bias, put
    back by the guard meant to make the mean safe. Measured, that alone left
    a sustained clip at +1.43% instead of +0.23%.

    A frame pair reporting less than half or more than twice the typical rate
    is not measuring the scroll, and that holds however tightly the rest agree.
    """
    if not len(values):
        return 0.0
    middle = float(np.median(values))
    if middle <= 0:
        return middle
    ratio = DEFAULT.tracking.outlier_ratio
    keep = values[(values >= middle / ratio) & (values <= middle * ratio)]
    return float(keep.mean()) if len(keep) else middle


@dataclass(frozen=True)
class SpeedEstimate:
    """Scroll speed in pixels per second, and how consistent it was."""

    value: float
    spread: float  # median absolute deviation across frame pairs
    samples: int

    # The per-pair measurements behind the value, so that several probes can
    # be pooled on equal terms. Combining probe *summaries* let a probe resting
    # on 22 frame pairs count for as much as one resting on 38.
    shifts: tuple[float, ...] = ()

    @property
    def confidence(self) -> float:
        """1.0 when every frame pair agreed; falls off as they disagree."""
        if self.value <= 0:
            return 0.0
        return float(max(0.0, 1.0 - (self.spread / self.value) * 10))


def _correlation_bias(
    residuals: Sequence[np.ndarray], window: np.ndarray, cfg
) -> float:
    """The offset phaseCorrelate adds when told a shift it should already know.

    Each residual is displaced by a known number of rows and measured. Whatever
    comes back beyond the displacement is the estimator's own error on this
    content, and is the same error it makes on the real frame pairs.

    Rolling wraps the image, which is precisely the periodicity the DFT behind
    phase correlation assumes, so the probe is a fair one.
    """
    errors = [
        cv2.phaseCorrelate(image, np.roll(image, probe, axis=0), window)[0][1] - probe
        for image in residuals
        for probe in cfg.bias_probe_shifts
    ]
    bias = float(np.median(errors)) if errors else 0.0

    if abs(bias) > cfg.max_bias:
        log.warning(
            "correlation bias of %.2fpx is implausibly large; ignoring it", bias
        )
        return 0.0

    log.debug("correlation bias %.3fpx over %d probes", bias, len(errors))
    return bias


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

    # Phase correlation measures how far the tiles moved between one frame and
    # the next, so the frames must be evenly spaced in time. Handing this the
    # output of sample(), which jumps across the whole video, would otherwise
    # fail obscurely — every pair filtered out as implausible — rather than
    # saying what was wrong.
    gaps = {frames[i + 1].index - frames[i].index for i in range(len(frames) - 1)}
    if len(gaps) > 1 or gaps == {0} or min(gaps) < 0:
        raise TrackingError(
            f"scroll speed needs evenly spaced frames; got gaps of "
            f"{sorted(gaps)} frames. Use reader.frames(), not reader.sample()."
        )

    region = calibration.strike_y
    images = [f.image[:region] for f in frames]
    if background is None:
        background = np.median(np.stack(images), axis=0).astype(np.uint8)

    reference = cv2.cvtColor(background[:region], cv2.COLOR_BGR2GRAY).astype(np.float32)

    def residual(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.abs(gray - reference)

    window = cv2.createHanningWindow(
        (images[0].shape[1], images[0].shape[0]), cv2.CV_32F
    )

    # Measure and remove the estimator's own systematic offset.
    #
    # phaseCorrelate locates a correlation peak to sub-pixel precision, and on
    # some content that estimate carries a constant offset: shifting one theme's
    # frame by a known 3, 5 or 7 pixels came back as 3.499, 5.498 and 7.497.
    # Half a pixel per frame is nothing to look at and everything to a
    # transcription — it made that clip's notes drift a second and a half over
    # twenty seconds, scoring F1 0.06 with the tiles read perfectly.
    #
    # It cannot be predicted from image size or parity — measured, 400 rows was
    # exact while 401, 402 and 403 all skewed — so it is calibrated instead:
    # shift this video's own frame by an amount we know and see what comes back.
    # Same content, same code path, so whatever the estimator does to this video
    # is what gets subtracted. Where there is no bias this measures ~0.002px and
    # changes nothing.
    bias = _correlation_bias(
        [residual(images[i]) for i in (0, len(images) // 2, len(images) - 1)],
        window,
        cfg,
    )

    shifts: list[float] = []

    for previous, current in zip(frames, frames[1:]):
        dt = current.time - previous.time
        if dt <= 0:
            continue

        a = residual(previous.image[:region])
        b = residual(current.image[:region])
        if a.std() < cfg.min_residual or b.std() < cfg.min_residual:
            continue  # nothing is falling in this pair

        (_, dy), response = cv2.phaseCorrelate(a, b, window)
        if response < cfg.min_correlation:
            continue

        speed = (dy - bias) / dt
        if cfg.min_speed <= speed <= cfg.max_speed:
            shifts.append(speed)

    if len(shifts) < cfg.min_speed_samples:
        raise TrackingError(
            f"could not measure a consistent scroll speed "
            f"({len(shifts)} usable frame pairs)"
        )

    values = np.array(shifts)
    speed = _robust_mean(values)
    spread = float(np.median(np.abs(values - speed)))
    log.debug("scroll speed %.2f px/s (mad %.2f over %d pairs)", speed, spread, len(values))
    return SpeedEstimate(
        value=speed, spread=spread, samples=len(values), shifts=tuple(shifts)
    )


def measure_scroll_speed(
    reader: VideoReader,
    calibration: Calibration,
    frames_per_window: int,
    windows: int = 8,
    config: Config = DEFAULT,
) -> SpeedEstimate:
    """Measure the scroll speed from several windows spread across the video.

    One window is not enough. Phase correlation reads vertical motion from the
    ends of things, and a piece of long sustained notes draws tiles as tall
    uniform bars whose bodies look the same wherever they are — the aperture
    problem. Measured on a real recording of a slow arrangement, a window a
    third of the way in yielded one usable pair out of forty, while a window
    over the busier opening gave a clean 6.24px per frame from the same code.

    So the video is probed in several places and the samples pooled. A clip that
    measures cleanly everywhere is unaffected; one with a quiet stretch no
    longer depends on where the single probe happened to land.
    """
    total = reader.info.frame_count or 0
    if total <= 0:
        raise TrackingError("video reports no frame count; cannot place probes")

    # Cover the whole video, ends included. An earlier version kept to the
    # middle on the theory that the start is a title card, but on a slow piece
    # the only measurable stretch was the opening: once the arrangement settles
    # into sustained chords the tiles are bars taller than the fall area, so
    # neither end of one is ever on screen and there is nothing to correlate.
    # A probe that lands on a title card simply finds no motion and drops out,
    # which costs one window and is much cheaper than missing the only good one.
    windows = max(1, windows)
    span = max(1, total - frames_per_window)
    starts = (
        [int(span * i / (windows - 1)) for i in range(windows)]
        if windows > 1
        else [span // 2]
    )

    pooled: list[float] = []
    probe_values: list[float] = []
    spreads: list[float] = []
    failures: list[str] = []
    for start in starts:
        window = list(reader.frames(start=start, stop=start + frames_per_window))
        if len(window) < 2:
            continue
        try:
            estimate = estimate_speed(window, calibration, config=config)
        except TrackingError as exc:
            failures.append(f"frame {start}: {exc}")
            continue
        pooled.extend(estimate.shifts or (estimate.value,))
        probe_values.append(estimate.value)
        spreads.append(estimate.spread)
        log.debug(
            "speed probe at frame %d: %.2f px/s over %d pairs",
            start,
            estimate.value,
            estimate.samples,
        )

    if not pooled:
        raise TrackingError(
            "could not measure a consistent scroll speed from any of "
            f"{len(starts)} probes across the video; "
            + "; ".join(failures)
        )

    values = np.array(pooled)
    speed = _robust_mean(values)

    # Windows that disagree wildly mean one of them locked onto something that
    # was not the tiles, so report the disagreement between probes rather than
    # the within-probe spread — it is the honest measure of confidence here.
    agreement = np.array(probe_values)
    spread = (
        float(np.median(np.abs(agreement - np.median(agreement))))
        if len(agreement) > 1
        else float(spreads[0])
    )
    log.debug(
        "scroll speed %.2f px/s from %d pairs across %d of %d probes",
        speed, len(values), len(probe_values), len(starts),
    )
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


def _frame_interval(times: Sequence[float]) -> float:
    """Seconds between observations, from the track's own timestamps."""
    if len(times) < 2:
        return 0.0
    return float(np.median(np.diff(np.asarray(times, dtype=np.float64))))


def track_to_note(
    track: TileTrack,
    speed: float,
    calibration: Calibration,
    config: Config = DEFAULT,
    hands: dict[int, Hand] | None = None,
) -> Note | None:
    """Convert one tile track into a note, or None if it is not usable."""
    hands = hands or {}
    cfg = config.tracking
    if track.length < cfg.min_observations:
        return None

    strike = calibration.strike_y
    margin = cfg.edge_margin

    # A bottom edge that has reached the strike line stops advancing, and a
    # stationary edge carries no timing: every frame it sits there reports a
    # crossing at that frame, so a note held for five seconds contributes five
    # seconds' worth of contradictory estimates and the median lands in the
    # middle of the note instead of at its start.
    #
    # Testing the position against the strike line does not find them. The
    # detected blob plateaus a few pixels short of the crop edge — measured at
    # 532 against a strike line of 536 — so a fixed margin lets the whole
    # plateau through. What marks a clipped edge is that it has stopped moving,
    # so the plateau is found by where the edge stops rather than by where it
    # is: anything within one frame's travel of the furthest the edge ever got.
    bottoms = np.asarray(track.bottoms, dtype=np.float64)
    step = speed * _frame_interval(track.times)
    plateau = bottoms.max() - step * 0.5

    # Only unclipped edges carry timing information.
    bottom_times, bottom_edges = [], []
    top_times, top_edges = [], []
    for time, top, bottom in zip(track.times, track.tops, track.bottoms):
        if bottom < strike - margin and bottom < plateau:
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

    if duration <= 0:
        # The top edge appears to cross before the bottom did, so the two
        # estimates contradict each other and there is nothing to salvage.
        log.debug("dropping pitch %d: offset precedes onset", track.pitch)
        return None

    # Clamp rather than drop. It looks like these should be fragments — the
    # two edges crossing at once — but measured both ways on the corpus,
    # dropping them costs 23 real notes of recall to remove 22 spurious ones,
    # and the corpus scores worse for it. Most are genuine notes whose top edge
    # was read badly, not phantoms.
    duration = max(duration, cfg.min_duration)

    return Note(
        onset=max(0.0, onset),
        pitch=track.pitch,
        duration=duration,
        hand=hands.get(track.track, "R"),
    )


def hands_by_register(tracks: Sequence[TileTrack]) -> dict[int, Hand]:
    """Decide which palette belongs to which hand, by register.

    Palettes are numbered by how many pixels they cover, which says nothing
    about which hand they are — so mapping index 1 to the left hand was
    arbitrary, and silently collapsed every colour past the second into the
    right hand.

    The lowest-sounding colour is taken as the left hand instead. With a single
    colour there is nothing to distinguish, so everything is nominally right and
    stage 7 splits it by pitch.
    """
    registers: dict[int, list[int]] = {}
    for track in tracks:
        if track.length:
            registers.setdefault(track.track, []).append(track.pitch)

    if len(registers) < 2:
        return {index: "R" for index in registers}

    lowest = min(registers, key=lambda index: float(np.median(registers[index])))
    return {index: ("L" if index == lowest else "R") for index in registers}


def tracks_to_notes(
    tracks: Sequence[TileTrack], speed: float, calibration: Calibration, config: Config = DEFAULT
) -> NoteSequence:
    hands = hands_by_register(tracks)
    notes = [track_to_note(t, speed, calibration, config, hands) for t in tracks]
    return NoteSequence.of(
        _merge_overlaps([n for n in notes if n is not None], config)
    )


def _merge_overlaps(notes: list[Note], config: Config = DEFAULT) -> list[Note]:
    """Fuse notes that are really one note on the same key.

    A key cannot sound twice at once, so an overlap is not two notes: it is one
    tile that broke into two tracks for a frame — a glint of bloom or a stray
    row of antialiasing splitting the blob — and was then timed twice.

    Overlap alone does not catch them all. A fragment whose duration collapses
    to the floor ends almost as soon as it starts, so the next fragment does
    not overlap it and the two mechanisms defeat each other: measured on a real
    recording, one tile produced five notes on one pitch inside 150ms, none of
    them overlapping the next.

    So proximity of *onsets* decides it, not overlap. Onsets are what separates
    the two cases: a genuine repeat is a key released and struck again, so its
    onset sits a whole note-length after the previous one — measured across the
    corpus, never closer than 104ms — while fragments of one tile land within a
    few frames of each other. Note that the *gaps* between genuine repeats go
    down to zero, so the same reasoning applied to gaps would destroy them.

    Nearness alone is still too broad. Applied to every pair it cost one theme
    five real notes, where noisy detection had put two true onsets inside the
    window. So one of the pair must also be too short to be a note in its own
    right: shorter than the repeat interval itself, which is the same bar the
    onsets are held to, and self-consistent — a note that could not be
    distinguished from its neighbour by position cannot be one by length
    either. Testing instead for the duration *floor* was too strict: only four
    of fifteen fragments on a real recording had collapsed that far, the rest
    measuring 33 to 46ms.

    The cost is that a repeat faster than ``min_repeat`` where one of the two
    is also mistimed reads as one note. At 144 BPM that is quicker than a
    thirty-second, and a renderer has to draw two tiles a pixel or two apart
    for it, so it is close to the limit of what tiles can express anyway.
    """
    min_repeat = config.tracking.min_repeat

    by_pitch: dict[int, list[Note]] = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)

    merged: list[Note] = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda note: note.onset)
        current = group[0]
        previous_onset = current.onset
        for following in group[1:]:
            fragment = (
                current.duration < min_repeat or following.duration < min_repeat
            )
            close = (
                fragment and following.onset - previous_onset < min_repeat
            )
            previous_onset = following.onset
            if close or following.onset < current.offset:
                end = max(current.offset, following.offset)
                current = Note(
                    onset=current.onset,
                    pitch=pitch,
                    duration=end - current.onset,
                    hand=current.hand,
                    velocity=current.velocity,
                )
            else:
                merged.append(current)
                current = following
        merged.append(current)

    return merged


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
