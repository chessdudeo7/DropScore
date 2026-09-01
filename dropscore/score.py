"""Stage 7: turn raw note events into something a score can be written from.

Four jobs, in dependency order: work out the beat, work out which beat is the
downbeat, snap onsets to that grid, and work out the key. Hand assignment sits
alongside them and depends on none.

Two principles run through this:

**Never snap blindly.** Quantization that moves a note a long way is how machine
transcriptions become unreadable — a shifted note is worse than an unshifted one
because it looks deliberate. Notes further than a configured fraction of a grid
step from any gridline keep their measured time and are reported instead.

**The grid comes from the music, not from a guess.** The beat period is found by
phase coherence over candidate periods, which asks "do the onsets line up with a
grid of this spacing?" rather than assuming a tempo. That naturally finds the
*tatum* — the finest grid, usually sixteenths — so the beat is recovered by
multiplying up to a plausible tempo rather than by taking the raw winner.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .config import Config, DEFAULT
from .notes import Hand, Note, NoteSequence

log = logging.getLogger(__name__)


class ScoreError(RuntimeError):
    """Raised when a sequence cannot be analysed."""


# Krumhansl-Kessler key profiles: how strongly each scale degree is felt as
# belonging to a major or minor key. Correlating a piece's pitch-class weights
# against rotations of these is the standard key-finding method.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Beat periods worth trying as multiples of the tatum. Covers simple and
# compound metres without admitting arbitrary ratios.
BEAT_MULTIPLES = (1, 2, 3, 4, 6, 8)


@dataclass(frozen=True)
class Analysis:
    """What was inferred about a sequence, and how confident each part is."""

    tempo: float  # BPM
    beat: float  # seconds per beat
    beat_phase: float  # seconds; where the beat grid starts
    downbeat_phase: float  # seconds; where bar one starts
    beats_per_bar: int
    key: str
    tempo_confidence: float  # 0-1, onset alignment to the beat grid
    key_confidence: float  # 0-1, correlation margin over the runner-up

    def __str__(self) -> str:
        return (
            f"{self.tempo:.1f} BPM ({self.tempo_confidence:.2f}), "
            f"{self.key} ({self.key_confidence:.2f}), "
            f"{self.beats_per_bar}/4"
        )


# ── tempo ────────────────────────────────────────────────────────────


def _coherence(onsets: np.ndarray, period: float) -> complex:
    """How well onsets line up with a grid of this spacing.

    Magnitude near 1 means every onset sits at the same offset within the
    period; the angle says what that offset is.
    """
    return complex(np.mean(np.exp(2j * np.pi * onsets / period)))


def estimate_tempo(sequence: NoteSequence, config: Config = DEFAULT) -> tuple[float, float, float]:
    """Return (beat seconds, phase seconds, confidence).

    Coherence is maximal for the finest grid that explains the onsets, and is
    just as high for any divisor of it. The coarsest period within a whisker of
    the best score is therefore the real tatum, and the beat is a small multiple
    of that chosen to land on a plausible tempo.
    """
    cfg = config.score
    onsets = np.array(sorted({round(n.onset, 4) for n in sequence}))
    if len(onsets) < cfg.min_onsets_for_tempo:
        raise ScoreError(
            f"only {len(onsets)} distinct onsets; need "
            f"{cfg.min_onsets_for_tempo} to estimate a tempo"
        )

    periods = np.linspace(cfg.min_tatum, cfg.max_tatum, cfg.tempo_resolution)
    scores = np.array([abs(_coherence(onsets, p)) for p in periods])

    best = scores.max()
    # Prefer the coarsest grid that still explains the onsets: a grid twice as
    # fine fits equally well and would halve the reported tempo.
    acceptable = np.flatnonzero(scores >= best * cfg.tatum_tolerance)
    tatum = float(periods[acceptable[-1]])

    # Phase must be read at the tatum, not at the beat. Onsets spread evenly
    # across a beat's subdivisions cancel out in the beat-period sum — four
    # sixteenths sit at 0, 90, 180 and 270 degrees and average to nothing — so
    # the beat-period angle is meaningless whenever the music is busy. The
    # tatum grid contains every gridline the beat grid has, which is what
    # quantization actually needs.
    z = _coherence(onsets, tatum)
    beat = _beat_from_tatum(tatum, cfg)
    phase = ((math.atan2(z.imag, z.real) * tatum / (2 * math.pi)) % tatum) % beat

    confidence = float(min(1.0, best))
    log.debug("tatum %.4fs -> beat %.4fs (%.1f BPM), phase %.4fs", tatum, beat, 60 / beat, phase)
    return beat, phase, confidence


def _beat_from_tatum(tatum: float, cfg) -> float:
    """Scale the tatum up to the multiple whose tempo is most plausible."""
    candidates = []
    for multiple in BEAT_MULTIPLES:
        beat = tatum * multiple
        bpm = 60.0 / beat
        if cfg.min_bpm <= bpm <= cfg.max_bpm:
            candidates.append((abs(math.log(bpm / cfg.tempo_prior)), beat))

    if not candidates:
        # Nothing plausible: fall back to whichever multiple lands closest.
        return min(
            (tatum * m for m in BEAT_MULTIPLES),
            key=lambda b: abs(math.log((60.0 / b) / cfg.tempo_prior)),
        )
    return min(candidates)[1]


def find_downbeat(
    sequence: NoteSequence, beat: float, phase: float, config: Config = DEFAULT
) -> float:
    """Which beat starts the bar.

    Bass notes fall on downbeats far more often than not, so beat positions are
    scored by onset count weighted toward the low register.
    """
    cfg = config.score
    if not len(sequence):
        return phase

    scores = np.zeros(cfg.beats_per_bar)
    for note in sequence:
        index = int(round((note.onset - phase) / beat)) % cfg.beats_per_bar
        # A note an octave lower counts for roughly twice as much.
        scores[index] += 2.0 ** ((60 - note.pitch) / 12.0)

    return (phase + float(np.argmax(scores)) * beat) % (beat * cfg.beats_per_bar)


# ── key ──────────────────────────────────────────────────────────────


def estimate_key(sequence: NoteSequence) -> tuple[str, float]:
    """Krumhansl-Schmuckler key finding, weighted by sounding time.

    Weighting by duration rather than note count matters: a passing sixteenth
    should not argue as loudly for a key as a held whole note.
    """
    if not len(sequence):
        raise ScoreError("cannot find the key of an empty sequence")

    weights = np.zeros(12)
    for note in sequence:
        weights[note.pitch % 12] += note.duration

    if not weights.any():
        raise ScoreError("no sounding notes to analyse")

    results: list[tuple[float, str]] = []
    for tonic in range(12):
        rotated = np.roll(weights, -tonic)
        for profile, quality in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            correlation = float(np.corrcoef(rotated, profile)[0, 1])
            if math.isnan(correlation):
                correlation = -1.0
            results.append((correlation, f"{PITCH_CLASS_NAMES[tonic]} {quality}"))

    results.sort(reverse=True)
    best, key = results[0]
    runner_up = results[1][0]
    confidence = float(max(0.0, min(1.0, (best - runner_up) / max(abs(best), 1e-6))))
    return key, confidence


# ── hands ────────────────────────────────────────────────────────────


def assign_hands(sequence: NoteSequence, config: Config = DEFAULT) -> NoteSequence:
    """Decide which notes belong to which hand.

    When detection already separated the tiles by colour, both hands are present
    and the only question is which label is which — stage 5 orders palettes by
    pixel count, not by register, so the labels are arbitrary until now. When
    only one colour was found, hands are split by a *moving* pitch boundary
    rather than a fixed middle C, so the split follows the music up and down the
    keyboard instead of cutting through it.
    """
    if not len(sequence):
        return sequence

    mode = config.score.hand_mode
    if mode == "none":
        return _relabel(sequence, lambda note: "R")
    if mode == "split":
        return _relabel(sequence, lambda note: "R" if note.pitch >= 60 else "L")
    if mode == "pitch":
        return _split_by_pitch(sequence, config)

    left = sequence.hand("L")
    right = sequence.hand("R")

    if left and right:
        if np.median([n.pitch for n in left]) > np.median([n.pitch for n in right]):
            flipped = [
                Note(n.onset, n.pitch, n.duration, "R" if n.hand == "L" else "L", n.velocity)
                for n in sequence
            ]
            return NoteSequence.of(flipped, tempo=sequence.tempo, key=sequence.key,
                                   source=sequence.source)
        return sequence

    return _split_by_pitch(sequence, config)


def _relabel(sequence: NoteSequence, hand_of) -> NoteSequence:
    """Rebuild a sequence with each note's hand decided by ``hand_of``."""
    return NoteSequence.of(
        [
            Note(n.onset, n.pitch, n.duration, hand_of(n), n.velocity)
            for n in sequence
        ],
        tempo=sequence.tempo,
        key=sequence.key,
        source=sequence.source,
    )


def _split_by_pitch(sequence: NoteSequence, config: Config = DEFAULT) -> NoteSequence:
    cfg = config.score
    notes = list(sequence)
    pitches = np.array([n.pitch for n in notes], dtype=float)

    global_split = _two_means(pitches)
    assigned: list[Note] = []

    for note in notes:
        window = [
            n.pitch
            for n in notes
            if abs(n.onset - note.onset) <= cfg.hand_window
        ]
        split = _two_means(np.array(window, dtype=float)) if len(window) >= 4 else global_split
        hand: Hand = "R" if note.pitch >= split else "L"
        assigned.append(Note(note.onset, note.pitch, note.duration, hand, note.velocity))

    return NoteSequence.of(
        assigned, tempo=sequence.tempo, key=sequence.key, source=sequence.source
    )


def _two_means(values: np.ndarray) -> float:
    """Boundary between the two clusters in a 1-D set of pitches."""
    if values.size == 0:
        return 60.0
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        return low + 0.5

    centres = np.array([low, high])
    for _ in range(12):
        labels = np.abs(values[:, None] - centres[None, :]).argmin(axis=1)
        for i in (0, 1):
            if np.any(labels == i):
                centres[i] = values[labels == i].mean()
    return float(centres.mean())


# ── quantization ─────────────────────────────────────────────────────



def quantize(
    sequence: NoteSequence, analysis: Analysis, config: Config = DEFAULT
) -> tuple[NoteSequence, int]:
    """Snap onsets and durations to the grid, leaving outliers alone.

    Returns the sequence and the number of notes left unquantized. A note that
    would have to move more than ``max_shift`` of a step is almost certainly
    evidence that the grid is wrong for that passage, and moving it anyway
    produces notation that looks confidently incorrect.
    """
    cfg = config.score
    step = analysis.beat / cfg.steps_per_beat
    tolerance = step * cfg.max_shift

    quantized: list[Note] = []
    skipped = 0

    for note in sequence:
        onset = _snap(note.onset, step, analysis.beat_phase, tolerance)
        if onset is None:
            onset = note.onset
            skipped += 1

        duration = _snap(note.duration, step, 0.0, tolerance)
        if duration is None or duration < step / 2:
            duration = max(note.duration, cfg.min_duration)

        quantized.append(Note(max(0.0, onset), note.pitch, duration, note.hand, note.velocity))

    return (
        NoteSequence.of(
            quantized, tempo=analysis.tempo, key=analysis.key, source=sequence.source
        ),
        skipped,
    )


def _snap(value: float, step: float, phase: float, tolerance: float) -> float | None:
    """Nearest gridline, or None when that is further than the tolerance."""
    snapped = round((value - phase) / step) * step + phase
    return snapped if abs(snapped - value) <= tolerance else None


# ── entry point ──────────────────────────────────────────────────────


def analyze(sequence: NoteSequence, config: Config = DEFAULT) -> Analysis:
    """Infer tempo, downbeat and key, honouring any overrides in the config."""
    cfg = config.score
    beat, phase, tempo_confidence = estimate_tempo(sequence, config)

    if cfg.fixed_tempo:
        # The grid is still anchored on the measured phase; only its spacing is
        # replaced. Confidence becomes 1.0 because it was told, not inferred.
        beat = 60.0 / cfg.fixed_tempo
        phase %= beat
        tempo_confidence = 1.0

    downbeat = find_downbeat(sequence, beat, phase, config)

    if cfg.fixed_key:
        key, key_confidence = cfg.fixed_key, 1.0
    else:
        key, key_confidence = estimate_key(sequence)

    return Analysis(
        tempo=60.0 / beat,
        beat=beat,
        beat_phase=phase,
        downbeat_phase=downbeat,
        beats_per_bar=config.score.beats_per_bar,
        key=key,
        tempo_confidence=tempo_confidence,
        key_confidence=key_confidence,
    )


def postprocess(
    sequence: NoteSequence, config: Config = DEFAULT
) -> tuple[NoteSequence, Analysis]:
    """Hands, tempo, key and quantization in one pass."""
    handed = assign_hands(sequence, config)
    analysis = analyze(handed, config)

    if config.score.steps_per_beat <= 0:
        # Quantization off: keep the measured times, still report what was
        # inferred about the piece.
        handed.tempo, handed.key = analysis.tempo, analysis.key
        return handed, analysis

    result, skipped = quantize(handed, analysis, config)

    if skipped:
        log.info(
            "%d of %d notes were too far from the grid to snap and kept their "
            "measured time",
            skipped,
            len(result),
        )
    return result, analysis
