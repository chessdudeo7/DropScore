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
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

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

# Semitones above the tonic that belong to each scale. The profiles above say
# how *typical* each degree is; these say which are in the key at all, which is
# a different question and the one a correlation cannot answer on its own.
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

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

    # The coarse scan steps by about a millisecond, and being one step out
    # accumulates: over a few seconds the grid slides far enough that notes
    # sitting exactly on it no longer snap. Refine locally before reading the
    # phase off it.
    fine = np.linspace(tatum * 0.98, tatum * 1.02, 201)
    tatum = float(fine[int(np.argmax([abs(_coherence(onsets, p)) for p in fine]))])

    # Phase must be read at the tatum, not at the beat. Onsets spread evenly
    # across a beat's subdivisions cancel out in the beat-period sum — four
    # sixteenths sit at 0, 90, 180 and 270 degrees and average to nothing — so
    # the beat-period angle is meaningless whenever the music is busy. The
    # tatum grid contains every gridline the beat grid has, which is what
    # quantization actually needs.
    z = _coherence(onsets, tatum)
    beat = _beat_from_tatum(tatum, onsets, [n.duration for n in sequence], cfg)
    phase = ((math.atan2(z.imag, z.real) * tatum / (2 * math.pi)) % tatum) % beat

    confidence = float(min(1.0, best))
    log.debug("tatum %.4fs -> beat %.4fs (%.1f BPM), phase %.4fs", tatum, beat, 60 / beat, phase)
    return beat, phase, confidence


def _repeats_at(onsets: np.ndarray, lag: float, tolerance: float) -> float:
    """Share of onsets followed by another one ``lag`` seconds later.

    Autocorrelation of the onset train, which peaks at periods the music
    actually groups by. Unlike coherence this does not cancel when a beat is
    subdivided — four sixteenths spread evenly round a beat sum to nothing in
    the coherence angle, but each still has a partner one beat away.
    """
    if lag <= 0 or not len(onsets):
        return 0.0

    # Only onsets with room for a partner may vote. Counting the rest as
    # misses biases the measure toward short lags, which trivially have more
    # room: on a uniform stream it made every finer beat look better supported
    # than the true one, purely because the clip ends.
    eligible = onsets[onsets + lag <= onsets[-1] + tolerance]
    if not len(eligible):
        return 0.0

    index = np.searchsorted(onsets, eligible + lag - tolerance)
    index = np.clip(index, 0, len(onsets) - 1)
    return float(np.mean(np.abs(onsets[index] - (eligible + lag)) <= tolerance))


def _duration_fit(modal: float, beat: float) -> float:
    """How idiomatic the commonest note value is against this beat.

    Full marks once the modal note is an eighth or longer, falling away below
    that: a piece written almost entirely in sixteenths is rare enough that
    reading one is better evidence of a beat twice too slow than of the piece.
    """
    if modal <= 0 or beat <= 0:
        return 1.0
    return min(1.0, (modal / beat) / 0.5)


def _beat_from_tatum(
    tatum: float, onsets: np.ndarray, durations: Sequence[float], cfg
) -> float:
    """Scale the tatum up to a beat.

    Which multiple is the beat cannot be read off the onsets alone — a stream
    of quarters at 100 BPM and one of eighths at 50 give identical onset times
    — so three things decide it, none sufficient alone.

    Preferring a fixed ``steps_per_beat`` makes the tatum a sixteenth by
    construction, which is how a slow arrangement came back at 50 BPM with
    every note a sixteenth. Choosing by nearness to the prior alone is no
    better: it reported ~95 BPM for pieces at 72 and at 144.
    """
    # How idiomatic the note values look is only evidence when there are
    # values to compare. A study written as an unbroken stream of sixteenths
    # is uniform by design, and reading its single duration as "too fine"
    # argued for a beat a third too fast. Where one value accounts for
    # everything the term is switched off rather than trusted.
    modal, variety = 0.0, 0.0
    if len(durations):
        counts = Counter(round(float(d), 2) for d in durations)
        modal, modal_count = counts.most_common(1)[0]
        variety = 1.0 - modal_count / len(durations)

    best: tuple[float, float] | None = None
    for multiple in BEAT_MULTIPLES:
        beat = tatum * multiple
        bpm = 60.0 / beat
        if not cfg.min_bpm <= bpm <= cfg.max_bpm:
            continue

        support = _repeats_at(onsets, beat, tatum * 0.5)
        prior = math.exp(
            -0.5 * (math.log(bpm / cfg.tempo_prior) / cfg.tempo_prior_width) ** 2
        )
        fit = _duration_fit(modal, beat) ** (cfg.duration_evidence * variety)

        # A mild preference for the conventional four tatums to the beat. On
        # music that says nothing about its own metre — an unbroken stream of
        # equal notes, where every candidate is supported identically — this
        # is the only thing left to go on. Weak enough that any real evidence
        # overrules it.
        conventional = 1.0 if multiple == cfg.steps_per_beat else cfg.other_multiple

        score = support * prior * fit * conventional
        if best is None or score > best[0]:
            best = (score, beat)

    if best is None:
        # Nothing plausible: fall back to whichever multiple lands closest.
        return min(
            (tatum * m for m in BEAT_MULTIPLES),
            key=lambda b: abs(math.log((60.0 / b) / cfg.tempo_prior)),
        )
    return best[1]


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


def estimate_key(
    sequence: NoteSequence, config: Config = DEFAULT
) -> tuple[str, float]:
    """Krumhansl-Schmuckler key finding, weighted by sounding time.

    Weighting by duration rather than note count matters: a passing sixteenth
    should not argue as loudly for a key as a held whole note.

    Correlation alone is not enough. A template match asks how closely the
    music's shape resembles a key's, and answers happily for a key whose
    defining notes never sound: on a real recording with no G# and no D#
    anywhere, E major still beat E minor, because the two share a heavy tonic
    and dominant and nothing charged E major for the C, D and G that rule it
    out. So weight sitting outside a candidate's scale is subtracted from its
    score — see ``out_of_scale_penalty``.
    """
    cfg = config.score
    if not len(sequence):
        raise ScoreError("cannot find the key of an empty sequence")

    weights = np.zeros(12)
    for note in sequence:
        weights[note.pitch % 12] += note.duration

    if not weights.any():
        raise ScoreError("no sounding notes to analyse")

    total = float(weights.sum())

    results: list[tuple[float, str]] = []
    for tonic in range(12):
        rotated = np.roll(weights, -tonic)
        for profile, quality, scale in (
            (MAJOR_PROFILE, "major", MAJOR_SCALE),
            (MINOR_PROFILE, "minor", MINOR_SCALE),
        ):
            correlation = float(np.corrcoef(rotated, profile)[0, 1])
            if math.isnan(correlation):
                correlation = -1.0
            outside = 1.0 - float(rotated[list(scale)].sum()) / total
            score = correlation - cfg.out_of_scale_penalty * outside
            results.append((score, f"{PITCH_CLASS_NAMES[tonic]} {quality}"))

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

    if cfg.steps_per_beat <= 0:
        # Quantization off. Handled here rather than only in postprocess, so
        # that calling this directly — it is public API, and "no grid" is a
        # setting the UI offers — cannot divide by zero.
        return (
            NoteSequence.of(
                list(sequence),
                tempo=analysis.tempo,
                key=analysis.key,
                source=sequence.source,
            ),
            0,
        )

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
        key, key_confidence = estimate_key(sequence, config)

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
    result, skipped = quantize(handed, analysis, config)

    if skipped:
        log.info(
            "%d of %d notes were too far from the grid to snap and kept their "
            "measured time",
            skipped,
            len(result),
        )
    return result, analysis
