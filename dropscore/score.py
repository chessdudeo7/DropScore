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
from dataclasses import dataclass, replace
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

# What each key is called on a page, which is whichever spelling takes fewer
# accidentals: D flat major is written with five flats, C sharp major with
# seven sharps, and nobody writes the second. Where the two are equal -- F
# sharp against G flat major, six apiece -- the commoner name is kept.
MAJOR_KEY_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_KEY_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")

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

    # The note value one beat is written as: 4 for a quarter, 8 for an eighth.
    # A compound beat is a dotted note and its metre is counted in eighths --
    # two dotted quarters a bar is 6/8, not 2/4 -- so the unit cannot be
    # assumed. Quarters unless said otherwise.
    beat_type: int = 4

    # Where each beat falls, for playing that does not hold one tempo. Empty
    # when the beats were not tracked, and everything then falls back to the
    # steady grid that ``beat`` and ``beat_phase`` describe.
    beat_times: tuple[float, ...] = ()

    # Where this stretch of music begins, and the stretches the piece is made
    # of. An arrangement changes tempo and metre part way through -- one page
    # here runs seven bars at 80 and then goes to 130 -- and one tempo cannot
    # describe both: measured on such a recording, the beat came back at 65,
    # which is neither. Each section is analysed on its own and carries its own
    # tempo, metre and grid. Empty for a piece that holds one tempo throughout,
    # where this analysis describes all of it.
    start: float = 0.0
    sections: tuple["Analysis", ...] = ()

    def at(self, when: float) -> "Analysis":
        """The section covering this moment, or the whole analysis."""
        chosen = self
        for section in self.sections:
            if when >= section.start - 1e-9:
                chosen = section
        return chosen

    def beats_before(self, section: "Analysis") -> float:
        """The beat a section begins on, counted on through every section before.

        Each section is counted on its own grid from where it begins, so the
        count carries straight on across the join. Counted from time zero
        instead, a section's own phase put its first beat dozens of beats past
        where the one before left off: on a real page, beat 11 at ten seconds
        and beat 58 at seventeen.
        """
        if not self.sections or section is self.sections[0]:
            return 0.0
        position = 0.0
        for index, (earlier, following) in enumerate(zip(self.sections, self.sections[1:])):
            grid = replace(earlier, sections=())
            if index == 0:
                position = beat_position(following.start, grid)
            else:
                position += beat_position(following.start, grid) - beat_position(earlier.start, grid)
            if following is section:
                return position
        return position

    def __str__(self) -> str:
        return (
            f"{self.tempo:.1f} BPM ({self.tempo_confidence:.2f}), "
            f"{self.key} ({self.key_confidence:.2f}), "
            f"{self.beats_per_bar}/{self.beat_type}"
        )


# ── tempo ────────────────────────────────────────────────────────────


def _coherence(onsets: np.ndarray, period: float) -> complex:
    """How well onsets line up with a grid of this spacing.

    Magnitude near 1 means every onset sits at the same offset within the
    period; the angle says what that offset is.
    """
    return complex(np.mean(np.exp(2j * np.pi * onsets / period)))


def estimate_tempo(sequence: NoteSequence, config: Config = DEFAULT) -> tuple[float, float, float]:
    """Return (beat seconds, phase seconds, confidence)."""
    beat, phase, confidence, _ = estimate_beat(sequence, config)
    return beat, phase, confidence


def estimate_beat(
    sequence: NoteSequence, config: Config = DEFAULT
) -> tuple[float, float, float, float]:
    """Return (beat seconds, phase seconds, confidence, tatum seconds).

    The tatum comes back with the beat because how many tatums the beat holds
    is what says whether it is a simple beat or a compound one, and that
    decides the metre it must be written in.

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
    scores = _grid_scores(onsets, periods, cfg)

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
    accented = _accented_onsets(sequence, tatum * cfg.steps_per_beat, cfg)
    beat = _beat_from_tatum(
        tatum,
        onsets,
        [n.duration for n in sequence],
        cfg,
        accented,
        drifting=bool(scores.max() < cfg.steady_tempo),
    )
    tatum_phase = (math.atan2(z.imag, z.real) * tatum / (2 * math.pi)) % tatum
    phase = _beat_phase(sequence, tatum, tatum_phase, beat)

    confidence = float(min(1.0, best))
    log.debug("tatum %.4fs -> beat %.4fs (%.1f BPM), phase %.4fs", tatum, beat, 60 / beat, phase)
    return beat, phase, confidence, tatum


def _accents(sequence: NoteSequence, beat: float) -> list[float]:
    """How strongly each note marks the start of a beat or a bar.

    Long notes and low ones: a bass note held through the bar is the downbeat
    far more often than a short note high in the melody is.
    """
    notes = list(sequence)
    if not notes:
        return []
    low = float(np.percentile([n.pitch for n in notes], 35))
    return [
        min(n.duration / beat, 4.0) * (3.0 if n.pitch <= low else 1.0) for n in notes
    ]


def _grid_scores(onsets: np.ndarray, periods: np.ndarray, cfg) -> np.ndarray:
    """How well each candidate grid explains the onsets, over the whole clip.

    Measured in windows and averaged, not across the clip at once. A player
    slowing and pushing by a few percent -- which is most playing that is not
    a machine -- leaves no single grid fitting end to end, and the reading
    collapses: on pieces warped by a smooth 4%, the period found was half or a
    third of the true one on half of them, and the tempo came back as 150 BPM
    where the music was at 100. Inside a window the drift is small enough that
    the true grid still stands out, and averaging the windows' scores keeps
    what they agree on.
    """
    whole = np.array([abs(_coherence(onsets, p)) for p in periods])

    span = float(onsets[-1] - onsets[0])
    window = cfg.tempo_window
    if span <= window * 1.5 or whole.max() >= cfg.steady_tempo:
        # One grid fits the clip, so use it: measured in windows and averaged,
        # the peak broadens and the coarsest period still within tolerance sits
        # a little long -- 72 BPM read as 70.4, and one piece halved outright.
        return whole

    spectra = []
    start = float(onsets[0])
    while start < onsets[-1] - window / 2:
        piece = onsets[(onsets >= start) & (onsets < start + window)]
        if len(piece) >= cfg.min_onsets_for_tempo:
            spectra.append([abs(_coherence(piece, p)) for p in periods])
        start += window / 2

    if not spectra:
        return whole
    return np.asarray(spectra, dtype=float).mean(axis=0)


def _beat_phase(
    sequence: NoteSequence, tatum: float, tatum_phase: float, beat: float
) -> float:
    """Which of the tatum's gridlines are the beat.

    The tatum's phase says where the finest grid lies, but a beat of four
    tatums could start on any one of the four, and the phase alone cannot say
    which: taken modulo the tatum it picked whichever line fell first after the
    start of the video. A synthetic clip starts on a downbeat, so that happened
    to be right; a recording starts wherever the button was pressed. On a real
    capture it put every beat half a beat late -- each quarter note written as
    a rest and an offbeat -- and on pieces started at a random point it was
    right for 20 of 40 in four and 10 of 40 in three.

    The beat is where the weight falls, so choose the offset that the long and
    the low notes land on. On those same pieces: 40 of 40 in both.
    """
    notes = list(sequence)
    lines = max(1, int(round(beat / tatum)))
    if lines == 1 or not notes:
        return tatum_phase % beat

    accents = _accents(sequence, beat)
    onsets = np.array([n.onset for n in notes])
    weights = np.array(accents)
    best: tuple[float, float] | None = None
    for line in range(lines):
        candidate = (tatum_phase + line * tatum) % beat
        position = (onsets - candidate) / beat
        on_beat = np.abs(position - np.round(position)) * beat < tatum / 2
        score = float(weights[on_beat].sum())
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1]


def track_beats(
    sequence: NoteSequence, beat: float, phase: float, config: Config = DEFAULT
) -> tuple[float, ...]:
    """Where each beat actually falls, rather than where a steady one would.

    One period and one phase describe a metronome. A person pushes and slows,
    and everything downstream is then measured against a grid the playing
    never followed: on pieces warped by a smooth 4%, fewer than seven notes in
    ten landed in the right subdivision, and at 8% barely a quarter did.

    The beats are chosen together rather than one at a time -- every sequence
    of them is scored on the weight of the notes it lands on, less a penalty
    for changing speed, and the best is taken. Chosen greedily instead, each
    beat snapping to whatever is nearest, subdivisions drag it off course and
    it slips a half beat within a bar or two.

    The penalty is light. Heavy, it holds the beat to the tempo it started at
    and so refuses the drift it exists to follow: the same pieces scored 79%
    and 42% against 100% and 100% once it was loosened.
    """
    cfg = config.score
    notes = sorted(sequence, key=lambda n: n.onset)
    if len(notes) < cfg.min_onsets_for_tempo or beat <= 0:
        return ()

    onsets = np.array([n.onset for n in notes])
    weights = np.array(_accents(sequence, beat))
    start, end = float(onsets[0]), float(onsets[-1])
    resolution = cfg.beat_resolution
    bins = int((end - start) / resolution) + 2
    if bins < 4:
        return ()

    accent = np.zeros(bins)
    for when, weight in zip(onsets, weights):
        accent[int(round((when - start) / resolution))] += weight
    peak = accent.max()
    if peak <= 0:
        return ()
    accent /= peak

    period = beat / resolution
    lo, hi = max(2, int(period * 0.5)), int(period * 2.0)
    steps = np.arange(lo, hi + 1)
    penalty = -cfg.beat_inertia * np.log(steps / period) ** 2

    score = accent.copy()
    previous = np.full(bins, -1, dtype=int)
    for index in range(lo, bins):
        earlier = index - steps
        usable = earlier >= 0
        if not usable.any():
            continue
        candidates = score[earlier[usable]] + penalty[usable]
        best = int(np.argmax(candidates))
        if candidates[best] > 0:
            score[index] += candidates[best]
            previous[index] = earlier[usable][best]

    tail = np.arange(max(0, bins - int(period * 2)), bins)
    index = int(tail[np.argmax(score[tail])])
    chain: list[float] = []
    while index >= 0:
        chain.append(index * resolution + start)
        index = previous[index]
    chain.reverse()
    return tuple(chain) if len(chain) >= 2 else ()


def beat_position(when: float, analysis: Analysis) -> float:
    """Where a moment falls, counted in beats from the first tracked one."""
    if analysis.sections:
        section = analysis.at(when)
        grid = replace(section, sections=())
        if section is analysis.sections[0]:
            return beat_position(when, grid)
        return (
            analysis.beats_before(section)
            + beat_position(when, grid)
            - beat_position(section.start, grid)
        )
    times = analysis.beat_times
    if not times:
        return (when - analysis.beat_phase) / analysis.beat
    grid = np.asarray(times)
    indices = np.arange(len(grid), dtype=float)
    period = float(np.median(np.diff(grid))) if len(grid) > 1 else analysis.beat
    if when <= grid[0]:
        return float((when - grid[0]) / period)
    if when >= grid[-1]:
        return float(len(grid) - 1 + (when - grid[-1]) / period)
    return float(np.interp(when, grid, indices))


def beat_time(position: float, analysis: Analysis) -> float:
    """The moment a beat position falls at: ``beat_position`` reversed."""
    if analysis.sections:
        section = analysis.sections[0]
        for candidate in analysis.sections[1:]:
            if position >= analysis.beats_before(candidate) - 1e-9:
                section = candidate
        grid = replace(section, sections=())
        if section is analysis.sections[0]:
            return beat_time(position, grid)
        local = position - analysis.beats_before(section) + beat_position(section.start, grid)
        return beat_time(local, grid)
    times = analysis.beat_times
    if not times:
        return analysis.beat_phase + position * analysis.beat
    grid = np.asarray(times)
    indices = np.arange(len(grid), dtype=float)
    period = float(np.median(np.diff(grid))) if len(grid) > 1 else analysis.beat
    if position <= 0:
        return float(grid[0] + position * period)
    if position >= len(grid) - 1:
        return float(grid[-1] + (position - (len(grid) - 1)) * period)
    return float(np.interp(position, indices, grid))


def estimate_meter(
    sequence: NoteSequence, beat: float, phase: float, config: Config = DEFAULT
) -> int:
    """Beats in a bar: three or four.

    Read from how the weight of the music repeats. Where the long and low notes
    fall is a series over beats, and in three it comes round every three while
    in four it comes round every four -- so the series is compared with itself
    shifted by each, and the closer match wins. Asking instead which position
    in the bar carries most weight, as the downbeat search does, cannot choose
    between bar lengths: it found three for 2 pieces in 40 that were in three.
    Measured on pieces generated in both, 36 of 40 each way, and a real capture
    of a piece in three read as three where it had always been written in four.

    Two against four is not attempted: the same music is correctly written in
    either, where three against four is not.
    """
    cfg = config.score
    notes = list(sequence)
    if len(notes) < cfg.min_onsets_for_tempo or beat <= 0:
        return 4

    accents = _accents(sequence, beat)
    positions = [(n.onset - phase) / beat for n in notes]
    first = min(int(round(x)) for x in positions)
    series = np.zeros(max(int(round(x)) for x in positions) - first + 1)
    for x, weight in zip(positions, accents):
        if abs(x - round(x)) < 0.2:  # on a beat, not between them
            series[int(round(x)) - first] += weight

    if series.size < 16:
        return 4

    centred = series - series.mean()
    energy = float(np.dot(centred, centred))
    if energy <= 0:
        return 4

    def match(lag: int) -> float:
        return float(np.dot(centred[:-lag], centred[lag:])) / energy

    return 3 if match(3) > match(4) else 4


def _meter(
    sequence: NoteSequence, beat: float, tatum: float, phase: float, config: Config
) -> tuple[int, int, float]:
    """The time signature, and the seconds of the unit it counts.

    A beat holding three tatums is a compound beat -- a dotted note -- and a
    dotted beat's metre is counted in eighths: two of them a bar is 6/8, three
    9/8, four 12/8. Called four-four instead, as it was, the bars came out
    twice the length of the music's: a real Fur Elise, whose bars are a second
    each, was laid out in bars of two seconds with the bar lines through the
    middle of every other one.

    Everything else counts in quarters, where the beat is the quarter itself.
    """
    cfg = config.score
    if cfg.beats_per_bar:
        return cfg.beats_per_bar, cfg.beat_type or 4, beat

    steps = round(beat / tatum) if tatum > 0 else 0
    if steps in (3, 6) and not cfg.fixed_tempo:
        eighth = beat / 3.0
        count = _eighths_in_a_bar(sequence, eighth, phase, cfg)

        # A beat of three tatums is a dotted note only if the music groups in
        # threes. A beat is chosen partly for sitting near a walking tempo, and
        # on a piece whose tatum is a sixteenth that prefers three of them --
        # 119 to the minute -- over the two the music actually moves in, at
        # 178. Fur Elise came back that way: its onsets repeat at two tatums
        # 0.63 of the time against 0.56 at three, and its bar of six tatums was
        # read as six eighths where the page prints three. The bar was the
        # right length, so the metre passed; but every value in it was counted
        # against a unit half the size the page writes, and the eighths the
        # melody holds could not be written at all -- no run of whole units
        # came to one, so they were engraved as sixteenths, 10 of the 11 on the
        # page wrong.
        #
        # Asked directly, the grouping settles it: where two tatums repeat more
        # than three, the tatum is a sixteenth and the unit is two of them.
        if steps == 3 and count % 2 == 0:
            onsets = np.array(sorted({float(n.onset) for n in sequence}))
            tolerance = tatum * cfg.repeat_tolerance
            if _repeats_at(onsets, 2.0 * tatum, tolerance) > _repeats_at(
                onsets, 3.0 * tatum, tolerance
            ):
                eighth, count = 2.0 * tatum, count // 2

        return count, 8, eighth

    return estimate_meter(sequence, beat, phase, config), 4, beat


def _eighths_in_a_bar(sequence: NoteSequence, eighth: float, phase: float, cfg) -> int:
    """Six, nine or twelve: how far apart the bass comes round.

    Asked of the low notes alone. Long notes are no guide to where a bar
    begins in music that carries a tune over an accompaniment -- the tune's
    long notes fall where the phrase wants them -- and mixing the two put the
    bar of a piece whose bass moves every six eighths at eight.
    """
    notes = list(sequence)
    if len(notes) < cfg.min_onsets_for_tempo or eighth <= 0:
        return 6

    cut = float(np.percentile([n.pitch for n in notes], cfg.bass_share))
    positions = [(n.onset - phase) / eighth for n in notes if n.pitch <= cut]
    if len(positions) < cfg.min_onsets_for_tempo:
        return 6

    first, last = int(round(min(positions))), int(round(max(positions)))
    series = np.zeros(last - first + 1)
    for position in positions:
        if abs(position - round(position)) < 0.25:
            series[int(round(position)) - first] += 1.0

    centred = series - series.mean()
    energy = float(centred @ centred)
    if energy <= 0:
        return 6

    scores = {
        lag: float(centred[:-lag] @ centred[lag:]) / energy
        for lag in (6, 9, 12)
        if lag < len(centred)
    }
    return max(scores, key=scores.get) if scores else 6


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


def _accented_onsets(sequence: NoteSequence, beat: float, cfg) -> np.ndarray:
    """When the longer notes begin, both ends of the question.

    Which multiple of the tatum is the beat is asked of these rather than of
    every onset. Any onset serves to find the grid, but not to find the beat
    on it: music moves in eighths and sixteenths, so a shorter lag genuinely
    has more partners than a longer one and the measure prefers a beat too
    fast whatever the music does. The terms beside it held that in check only
    while the tempo held still -- a piece changing pace halfway through was
    read at half its beat 16 times out of 16.

    Long notes fall on beats rather than between them, so the question is put
    to them alone, and a partner must be one of them too. Weighting the notes
    instead of setting the others aside is not enough: the partner is still
    found among the subdivisions, and the same piece stays wrong 16 times out
    of 16. Length only, not register -- taking the lowest notes as well picks
    out a line of its own in plain material and reads its spacing as the beat,
    which turned three pieces in four into dotted ones.
    """
    notes = sorted(sequence, key=lambda n: n.onset)
    if not notes or beat <= 0:
        return np.zeros(0)

    held: dict[float, float] = {}
    for note in notes:
        when = round(note.onset, 4)
        held[when] = max(held.get(when, 0.0), min(note.duration / beat, 4.0))

    onsets = np.array(sorted(held))
    weights = np.array([held[t] for t in onsets])
    keep = weights >= np.quantile(weights, 1.0 - cfg.accent_share)

    # And held longer than the grid itself. In an unbroken stream of sixteenths
    # no note is long, and the longest share is only the ones that happened to
    # measure long -- on one recording, whichever tiles of a repeating
    # arpeggio were drawn tallest. Those recur with the figure, every half
    # bar, and that spacing was read as the beat: 65 BPM for a piece marked
    # 130. Left with too few, the question goes back to every onset.
    keep &= weights >= cfg.accent_min_tatums / cfg.steps_per_beat
    return onsets[keep]


def _repeats_over_windows(
    onsets: np.ndarray, lag: float, tolerance: float, cfg, drifting: bool
) -> float:
    """``_repeats_at``, asked a window at a time and averaged.

    Two notes a beat apart are a beat apart only if the beat has not changed
    in between. Over a whole clip a player drifting a few percent pulls them
    outside any tolerance tight enough to be worth having.
    """
    span = float(onsets[-1] - onsets[0]) if len(onsets) else 0.0
    window = cfg.tempo_window
    whole = _repeats_at(onsets, lag, tolerance)
    if span <= window * 1.5 or not drifting:
        return whole

    scores = []
    start = float(onsets[0])
    while start < onsets[-1] - window / 2:
        piece = onsets[(onsets >= start) & (onsets < start + window)]
        if len(piece) >= cfg.min_onsets_for_tempo:
            scores.append(_repeats_at(piece, lag, tolerance))
        start += window / 2
    return float(np.mean(scores)) if scores else whole


def _duration_fit(modal: float, beat: float, articulation: float = 1.0) -> float:
    """How idiomatic the commonest note value is against this beat.

    Full marks once the modal note is an eighth or longer, falling away below
    that: a piece written almost entirely in sixteenths is rare enough that
    reading one is better evidence of a beat twice too slow than of the piece.

    The measured duration is how long the key was held, which is shorter than
    the value written -- a quarter played detached at 60 BPM lasts around a
    third of a second, not a whole one. Taken literally every piece looks
    written in finer values than it is, and finer values argue for a faster
    beat, so this term voted for double the tempo on exactly the slow pieces
    where it should have argued against. Dividing out the articulation
    compares like with like.
    """
    if modal <= 0 or beat <= 0:
        return 1.0
    written = modal / articulation if articulation > 0 else modal
    return min(1.0, (written / beat) / 0.5)


def _beat_from_tatum(
    tatum: float,
    onsets: np.ndarray,
    durations: Sequence[float],
    cfg,
    accented: np.ndarray | None = None,
    drifting: bool = False,
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
    #
    # Values are counted in steps of the grid, not in measured seconds. How
    # long a key is held varies with how its tile is drawn, and to the nearest
    # hundredth one recording's stream of sixteenths held five different
    # lengths -- varied enough to switch the term back on, which then read the
    # sixteenths as too fine and argued for a beat of three of them.
    modal, variety = 0.0, 0.0
    if len(durations):
        counts = Counter(round(float(d), 2) for d in durations)
        modal = counts.most_common(1)[0][0]
        steps = Counter(max(1, round(float(d) / tatum)) for d in durations)
        variety = 1.0 - steps.most_common(1)[0][1] / len(durations)

    # Too few long notes to measure repetition among, and the measure is
    # noise: a clip of slow held chords left 19 of them, scoring 0.111 at its
    # true beat against 0.222 at half, where every onset together scored 0.956
    # and 0.933. Sparse music is also the music that needs no help here.
    #
    # Nor when they are too far apart to land on beats. Long notes a few beats
    # apart only ever find partners a few beats away, so every shorter candidate
    # scores nothing: a stream of sixteenths over bass notes held for bars left
    # long notes a median 2.95 beats apart, supported only the slowest beat on
    # offer, and was read at half its tempo. Everywhere they are evidence they
    # sit about a beat apart.
    chosen = onsets
    if accented is not None and len(accented) >= cfg.min_accented_onsets:
        spacing = np.diff(np.unique(np.round(accented, 2)))
        spacing = spacing[spacing > tatum / 2]
        conventional_beat = tatum * cfg.steps_per_beat
        if len(spacing) and np.median(spacing) <= cfg.accent_max_gap_beats * conventional_beat:
            chosen = accented

    best: tuple[float, float] | None = None
    for multiple in BEAT_MULTIPLES:
        beat = tatum * multiple
        bpm = 60.0 / beat
        if not cfg.min_bpm <= bpm <= cfg.max_bpm:
            continue

        support = _repeats_over_windows(
            chosen, beat, tatum * cfg.repeat_tolerance, cfg, drifting
        )
        prior = math.exp(
            -0.5 * (math.log(bpm / cfg.tempo_prior) / cfg.tempo_prior_width) ** 2
        )
        fit = _duration_fit(modal, beat, cfg.legato_ratio) ** (
            cfg.duration_evidence * variety
        )

        # A mild preference for the conventional four tatums to the beat. On
        # music that says nothing about its own metre — an unbroken stream of
        # equal notes, where every candidate is supported identically — this
        # is the only thing left to go on. Weak enough that any real evidence
        # overrules it.
        #
        # Rewarding binary multiples generally instead, so that a beat of eight
        # tatums is not punished where the finest grid is a thirty-second, is
        # worse: it hands the same bonus to a beat of two tatums, and reading a
        # slow piece at double speed is the commoner error by far.
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
    sequence: NoteSequence,
    beat: float,
    phase: float,
    config: Config = DEFAULT,
    beats_per_bar: int | None = None,
) -> float:
    """Which beat starts the bar.

    Bass notes fall on downbeats far more often than not, so beat positions are
    scored by onset count weighted toward the low register.
    """
    bar = beats_per_bar or config.score.beats_per_bar or 4
    if not len(sequence):
        return phase

    scores = np.zeros(bar)
    for note in sequence:
        index = int(round((note.onset - phase) / beat)) % bar
        # A note an octave lower counts for roughly twice as much.
        scores[index] += 2.0 ** ((60 - note.pitch) / 12.0)

    return (phase + float(np.argmax(scores)) * beat) % (beat * bar)


# ── key ──────────────────────────────────────────────────────────────


#: The note the two staves are divided at.
MIDDLE_C = 60


#: Pitch classes spelled without an accidental: the white keys.
NATURAL_CLASSES = frozenset((0, 2, 4, 5, 7, 9, 11))


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

            # A key signature claims that certain notes are altered. Claiming
            # one the music never plays is a claim about nothing: on a real
            # capture with no F of either kind anywhere, E minor beat A minor
            # on template shape alone and put a sharp on the page that the
            # piece never sounds. Both fit the notes; only one asserts
            # something unheard.
            unfounded = sum(
                1
                for step in scale
                if (tonic + step) % 12 not in NATURAL_CLASSES
                and weights[(tonic + step) % 12] == 0.0
            )

            score = (
                correlation
                - cfg.out_of_scale_penalty * outside
                - cfg.unfounded_accidental * unfounded
            )
            names = MINOR_KEY_NAMES if quality == "minor" else MAJOR_KEY_NAMES
            results.append((score, f"{names[tonic]} {quality}"))

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

    if left and right and _looks_like_hands(sequence, config):
        if np.median([n.pitch for n in left]) > np.median([n.pitch for n in right]):
            flipped = [
                Note(n.onset, n.pitch, n.duration, "R" if n.hand == "L" else "L", n.velocity)
                for n in sequence
            ]
            return NoteSequence.of(flipped, tempo=sequence.tempo, key=sequence.key,
                                   source=sequence.source)
        return sequence

    return _split_by_pitch(sequence, config)


def _looks_like_hands(sequence: NoteSequence, config: Config = DEFAULT) -> bool:
    """Do the two colour groups sit in different registers, as hands do?

    Stage 5 finds colours, not hands. A video that draws every note in one
    colour can still yield two palettes when the tiles over the black keys
    render darker than the ones over the white keys, and the split that comes
    back is accidentals-versus-naturals: two groups interleaved across the
    whole keyboard. Trusting it puts bass notes on the treble staff and buries
    the result in ledger lines.

    Hands are not perfectly separable -- they cross, and they share the middle
    of the keyboard -- but a single pitch boundary still sorts most of a real
    pair correctly. A split along some other axis does no better than chance.
    """
    cfg = config.score
    notes = list(sequence)
    if len(notes) < cfg.min_hand_notes:
        return True  # too little evidence to overrule the colours

    pitches = np.array([n.pitch for n in notes])
    is_right = np.array([n.hand == "R" for n in notes])

    thresholds = np.arange(pitches.min(), pitches.max() + 2)
    above = pitches[None, :] >= thresholds[:, None]
    agree = (above == is_right[None, :]).mean(axis=1)
    best = float(max(agree.max(), 1.0 - agree.min()))

    # Scored as the gain over putting every note on the larger side, not as a
    # raw accuracy. A boundary drawn below every note already sorts a lopsided
    # split as well as the split is lopsided: 78 to 22 scores 0.78 without
    # pitch separating anything, and on a real capture that let a colour split
    # of accidentals against naturals through a threshold of 0.75 and put 172
    # notes on the wrong staff. Measured as gain, genuine pairs of hands score
    # 0.61 to 0.88 and colour splits along some other axis 0.01 to 0.12.
    majority = float(max(is_right.mean(), 1.0 - is_right.mean()))
    gain = (best - majority) / (1.0 - majority)
    return gain >= cfg.hand_separability


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
    """Split one colour into two hands with a boundary that follows the music.

    The boundary is the midpoint of the register in play around each note --
    halfway between the highest and lowest pitch of its nearest neighbours in
    time. Deliberately not a clustering of those pitches: the left hand plays
    sparse bass under a dense melody, and any rule that weights by how many
    notes sit where gets dragged up into the melody, putting its lower notes on
    the bass staff. Taking only the extremes ignores density, which is the one
    thing that misleads here.

    Neighbours are counted, not timed. A window measured in seconds spans a
    different amount of music at 60bpm than at 144, and the accuracy fell off
    either side of whichever duration was chosen; a fixed count holds across
    the tempo range.
    """
    cfg = config.score
    notes = list(sequence)
    onsets = np.array([n.onset for n in notes], dtype=float)
    pitches = np.array([n.pitch for n in notes], dtype=float)

    global_split = (pitches.min() + pitches.max()) / 2.0
    assigned: list[Note] = []

    for index, note in enumerate(notes):
        nearest = np.argsort(np.abs(onsets - note.onset), kind="stable")
        window = pitches[nearest[: cfg.hand_neighbours]]
        split = (
            (window.min() + window.max()) / 2.0
            if len(window) >= 4
            else global_split
        )
        # Kept near middle C. What is being chosen here is which staff a note
        # is written on, and a staff is chosen by register: the boundary is
        # middle C, give or take. Free to go where it liked, it followed the
        # music -- on a real capture with a pedal note repeating under a high
        # melody it rose above the pedal and sent it to the bass staff, where
        # the printed music keeps it in the treble throughout.
        reach = cfg.staff_boundary_reach
        split = min(max(split, MIDDLE_C - reach), MIDDLE_C + reach)
        # A note exactly on the boundary goes by middle C. Clamped, the
        # boundary lands on a whole pitch, and counting that pitch as upper put
        # a left hand's E2-E3-G#3 on two staves: the G#3 sat on a boundary held
        # at the lower limit, and went to the treble all four times it came.
        hand: Hand = (
            "R" if note.pitch > split or (note.pitch == split and note.pitch >= MIDDLE_C) else "L"
        )

        # Everything around this note within one hand's reach is one part, and
        # is not split between the staves at all. A melody moving through a
        # sixth or so has no bass line in it to separate out, but a boundary
        # held near middle C still cut through it: a real capture's tune dips
        # to D4 with the bass silent, and those D4s went to the bass staff --
        # and the notes before them, left with nothing following on their own
        # staff for three beats, were written short as well.
        if len(window) >= 4 and window.max() - window.min() <= cfg.one_hand_span:
            hand = "R" if float(np.median(window)) >= MIDDLE_C else "L"
        assigned.append(Note(note.onset, note.pitch, note.duration, hand, note.velocity))

    # A note struck again, with nothing struck between, is played by the hand
    # that struck it the first time. Each is judged by its own neighbours, and
    # a boundary lying across a repeated D4 put the first on the treble staff
    # and the second on the bass -- where a bass note three beats later had
    # pulled the boundary above it. Neither was then followed on its own staff
    # by anything close, and both were written short.
    order = sorted(range(len(assigned)), key=lambda i: assigned[i].onset)
    for position in range(1, len(order)):
        first, again = assigned[order[position - 1]], assigned[order[position]]
        if again.pitch != first.pitch or again.hand == first.hand:
            continue
        struck_together = [
            n for n in assigned if abs(n.onset - first.onset) < cfg.repeat_min_gap and n is not first
        ]
        if struck_together or again.onset - first.onset < cfg.repeat_min_gap:
            continue
        assigned[order[position]] = Note(
            again.onset, again.pitch, again.duration, first.hand, again.velocity
        )

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

    # Snapped where the beats are, not where a steady one would have put them.
    # The two are the same for a metronome and part company for a person: a
    # grid laid down at one tempo walks away from playing that drifts, and
    # after a few bars it is snapping notes to the wrong subdivision.
    in_beats = bool(analysis.beat_times)
    grid = 1.0 / cfg.steps_per_beat
    allowed = grid * cfg.max_shift

    quantized: list[Note] = []
    skipped = 0

    for note in sequence:
        if in_beats:
            position = beat_position(note.onset, analysis)
            landed = _snap(position, grid, 0.0, allowed)
            if landed is None:
                onset = note.onset
                skipped += 1
            else:
                onset = beat_time(landed, analysis)
                position = landed
            length = beat_position(note.onset + note.duration, analysis) - position
            snapped = _snap(length, grid, 0.0, allowed)
            if snapped is None or snapped < grid / 2:
                duration = max(note.duration, cfg.min_duration)
            else:
                duration = max(beat_time(position + snapped, analysis) - onset, cfg.min_duration)
        else:
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


def notate_durations(
    sequence: NoteSequence, analysis: Analysis, config: Config = DEFAULT
) -> NoteSequence:
    """Rewrite performed durations as written ones.

    A tile's length is how long the key was held, and a player releasing a
    quarter note a little early is playing it detached, not playing a shorter
    note. Engraved literally that becomes a dotted eighth followed by a
    sixteenth rest -- on a real transcription the dotted eighth was the
    commonest value on the page, 88 notes of 301, purely because the source
    was held at about three quarters of nominal.

    Notation carries the written value and leaves articulation to a slur or a
    staccato dot, so a note covering enough of the way to the next onset in
    its own hand is written as reaching it.

    Shortens only where a note overlaps its immediate neighbour and nothing
    past it: that is one hand playing legato, which the page writes as a slur
    over two notes rather than as a note and a half. A note running on past the
    note after it is a voice genuinely held under a moving one, and cutting it
    would delete a real sustain.
    """
    cfg = config.score
    if cfg.legato_ratio <= 0:
        return sequence

    step = analysis.beat / cfg.steps_per_beat if cfg.steps_per_beat > 0 else 0.0
    written: list[Note] = []

    for hand in ("L", "R"):
        voice = sorted(sequence.hand(hand), key=lambda n: n.onset)
        onsets = [n.onset for n in voice]
        # How far apart this hand's notes usually fall, in beats. A silence
        # counts as articulation only up to this; beyond it a rest is written.
        spacings = sorted(
            beat_position(later, analysis) - beat_position(earlier, analysis)
            for earlier, later in zip(onsets, onsets[1:])
            if later - earlier > 1e-6
        )
        pulse = spacings[len(spacings) // 2] if spacings else 0.0
        for index, note in enumerate(voice):
            # In the beat of the stretch this note falls in, which is not the
            # first stretch's where the piece changes tempo.
            here = analysis.at(note.onset)
            step = here.beat / cfg.steps_per_beat if cfg.steps_per_beat > 0 else 0.0
            duration = note.duration
            # The next *different* onset. Notes struck together are one event,
            # and measuring to the nearest of them gives a gap of nothing, so
            # no note inside a chord was ever written as reaching anything.
            following = next(
                (t for t in onsets[index + 1 :] if t > note.onset + 1e-6), None
            )
            if following is not None:
                gap = following - note.onset
                held = duration / gap if gap > 0 else 0.0
                # Held most of the way: detached, and written as reaching.
                # Or the gap is short enough that no rest would be written
                # there anyway -- a staccato quarter is a quarter with a dot
                # over it, not a sixteenth and three rests. Taken literally,
                # a real capture wrote 5 quarters as 0.58 of a beat, 5 as
                # 0.38, 5 as 0.33 and so on: 32% of its written values
                # matched the printed music.
                #
                # Whether the note is written as reaching is judged on the way
                # to the next onset, but how far it reaches is capped at the
                # hand's own pulse: past that the silence is a rest. Judging
                # the ratio against the capped figure instead sounds tidier and
                # is worse -- it lets a sixteenth held a fifth of the way to a
                # distant onset count as reaching, and cost Pietschmann 3 six
                # values and Fur Elise two.
                if gap > 0 and (
                    cfg.legato_ratio <= held < 1.0
                    or gap <= cfg.articulation_gap * analysis.beat + step / 2
                ):
                    filled = gap
                    if cfg.fill_pulses > 0 and pulse > 0:
                        filled = min(filled, pulse * cfg.fill_pulses * here.beat)
                    if step > 0:
                        filled = round(filled / step) * step
                    # Enforce the invariant after rounding, not before it: a
                    # gap under half a step rounds to nothing, which is not a
                    # note at all.
                    duration = max(duration, filled)

                # A note that outlasts the next onset is a voice held under a
                # moving one, and the gap that matters is from where it ends,
                # not from where it began. Measured from the start, a dotted
                # half under a pulse of quarters had a gap of one beat, which
                # it already outlasted, so it was left as it was played -- two
                # and a quarter beats of a three-beat note, eight times on one
                # page. From its end, the next onset is most of a beat away
                # and inside the articulation gap, like any other.
                end = note.onset + note.duration
                # Overlapping its neighbour and nothing beyond it is legato,
                # not a sustain: the hand has not left the note before taking
                # the next, which the page writes as a slur and not as a
                # longer value. Only ever shortens to where the next note
                # begins, so a held voice -- which runs past a whole figure,
                # and is what the branch below exists for -- is untouched.
                beyond = next(
                    (t for t in onsets[index + 1 :] if t > following + 1e-6), None
                )
                if (
                    cfg.overlap_is_legato
                    and duration <= note.duration + 1e-9
                    and following <= end
                    and (beyond is None or end < beyond - step / 2)
                ):
                    duration = gap
                elif duration <= note.duration + 1e-9 and following <= end:
                    after = next(
                        (t for t in onsets[index + 1 :] if t >= end - step / 2), None
                    )
                    if (
                        after is not None
                        and after - end <= cfg.articulation_gap * here.beat + step / 2
                    ):
                        reach = after - note.onset
                        if step > 0:
                            reach = round(reach / step) * step
                        duration = max(duration, reach)

            # A note starting on a beat fills at least that beat. Released
            # early and engraved as played it becomes a short value followed
            # by a rest that runs over the beat line -- which the page would
            # not write, because a rest is written from a boundary, not across
            # one. So the value is taken out to the end of the beat it starts
            # on, and what is left over becomes the rest.
            if cfg.fill_to_beat > 0:
                position = beat_position(note.onset, analysis)
                onto = abs(position - round(position))
                to_beat = (round(position) + 1 - position) * here.beat
                # Only where the beat is nearly all the silence there is. A
                # note is stretched onto the beat line so the rest after it can
                # start there; if a whole beat or more of rest is left over
                # anyway, the silence stands on its own and the note keeps what
                # it was played at. Without that, the left hand's sixteenths --
                # on a beat, then two beats of rest -- were each written as a
                # whole beat.
                if (
                    following is not None
                    and onto < cfg.fill_to_beat
                    and duration < to_beat
                    and note.onset + to_beat <= following + step / 2
                    and following - (note.onset + to_beat) < here.beat - step / 2
                ):
                    duration = to_beat
            written.append(
                Note(note.onset, note.pitch, duration, note.hand, note.velocity)
            )

    written = _hold_under_figures(written, step, cfg, analysis.beat)

    return NoteSequence.of(
        sorted(written),
        tempo=sequence.tempo,
        key=sequence.key,
        source=sequence.source,
    )


def _hold_under_figures(notes: list[Note], step: float, cfg, beat: float = 0.0) -> list[Note]:
    """Write a bass note held under a figure as lasting until the bass moves.

    A player strikes a low octave, moves the hand up into an arpeggio and lets
    the pedal hold the bass. The tile is as short as the key was down, and the
    page writes what is heard: on a real recording, bass octaves the arpeggio
    ran over for up to three and a half bars were printed as tied whole notes,
    and written as sixteenths -- 3 of 14 bass values right.

    Held until the next note that comes near it, and only across a figure: at
    least ``hold_min_figure_onsets`` onsets in between, every one of them
    ``hold_clear_interval`` semitones above or more. A figure close above is
    the bass line itself moving -- Alberti's C-G-E-G, a fifth apart -- and a
    single chord between two bass notes is oom-pah, written short.

    Only ever lengthens.
    """
    if cfg.hold_min_figure_onsets <= 0:
        return notes
    ordered = sorted(notes, key=lambda n: n.onset)
    result = []
    for note in ordered:
        # A bass octave moves together, so the lower note of one is measured
        # from its upper note. From its own pitch, the lower D of a D octave
        # took the B octave's lower note -- nine semitones up -- for figure, and
        # was held twice as long as the page prints it. Only the lowest note
        # struck, and only with its octave: a B struck with an arpeggio's B
        # above it is the upper note of a bass octave, not the lower.
        together = [
            other.pitch for other in ordered if abs(other.onset - note.onset) < cfg.repeat_min_gap
        ]
        base = note.pitch
        if note.pitch == min(together) and note.pitch + 12 in together:
            base = note.pitch + 12

        # Only a doubled bass is held. A sustained bass is written as an octave
        # -- every one on the page this was measured against is -- while a
        # single low note is as likely to be the first note of the hand's own
        # figure: Fur Elise's E2 is followed by its E3 and G#3, and holding it
        # turned a sixteenth into a whole bar, seven times over.
        if note.pitch + 12 not in together and note.pitch - 12 not in together:
            result.append(note)
            continue

        clear = base + cfg.hold_clear_interval
        figure: set[float] = set()
        until = None
        last = note.onset
        for other in ordered:
            if other.onset < note.onset + cfg.repeat_min_gap:
                continue
            # The figure has to keep going. Where it stops, so does the note:
            # nothing is holding under silence, and one bass note ran 55 beats
            # to the next one that far below it.
            if other.onset - last > cfg.hold_max_gap * beat:
                until = last
                break
            if other.pitch < clear:
                until = other.onset
                break
            figure.add(round(other.onset, 3))
            last = other.onset
        if until is None or len(figure) < cfg.hold_min_figure_onsets:
            result.append(note)
            continue
        reach = until - note.onset
        if step > 0:
            reach = round(reach / step) * step
        result.append(Note(note.onset, note.pitch, max(note.duration, reach), note.hand, note.velocity))
    return result


def _snap(value: float, step: float, phase: float, tolerance: float) -> float | None:
    """Nearest gridline, or None when that is further than the tolerance."""
    snapped = round((value - phase) / step) * step + phase
    return snapped if abs(snapped - value) <= tolerance else None


# ── entry point ──────────────────────────────────────────────────────


def analyze(sequence: NoteSequence, config: Config = DEFAULT) -> Analysis:
    """Infer tempo, downbeat and key, honouring any overrides in the config."""
    cfg = config.score

    # An arrangement that changes tempo part way through is two pieces of music
    # for this purpose, and one beat cannot describe both: a page running seven
    # bars at 80 and then going to 130 came back at 65, which is neither, and
    # every written value in it was wrong. Told where the changes fall, each
    # stretch is analysed on its own. Where they fall is not inferred -- the
    # grid of a rubato performance wanders as far as a real change does, and
    # every rule that caught the change also cut a Fur Elise into ten pieces.
    if cfg.sections:
        return _analyze_sections(sequence, config)
    beat, phase, tempo_confidence, tatum = estimate_beat(sequence, config)

    if cfg.fixed_tempo:
        # The grid is still anchored on the measured phase; only its spacing is
        # replaced. Confidence becomes 1.0 because it was told, not inferred.
        beat = 60.0 / cfg.fixed_tempo
        phase %= beat
        tempo_confidence = 1.0

    # Only where one grid does not fit. A steady performance is already
    # described by its period and phase, and tracking it can only be worse:
    # given an unbroken stream of equal notes there is no weight anywhere to
    # follow, and the tracker wanders -- on such a stream at 120 BPM it laid
    # beats from 0.37s to 0.62s apart where every one of them is 0.5.
    # The metre first: it settles what a beat is counted as, and the beats are
    # tracked in that unit. Tracked in one and reported in another, every
    # written value came out a third of what it should be.
    beats_per_bar, beat_type, beat = _meter(sequence, beat, tatum, phase, config)
    beat_times = (
        track_beats(sequence, beat, phase, config)
        if tempo_confidence < cfg.steady_tempo
        else ()
    )
    downbeat = find_downbeat(sequence, beat, phase, config, beats_per_bar)

    if cfg.fixed_key:
        key, key_confidence = cfg.fixed_key, 1.0
    else:
        key, key_confidence = estimate_key(sequence, config)

    # Reported per quarter note, whatever the beat is written as, so that a
    # piece counted in eighths does not read as 356 BPM.
    return Analysis(
        tempo=60.0 / (beat * beat_type / 4.0),
        beat=beat,
        beat_phase=phase,
        downbeat_phase=downbeat,
        beats_per_bar=beats_per_bar,
        key=key,
        tempo_confidence=tempo_confidence,
        key_confidence=key_confidence,
        beat_type=beat_type,
        beat_times=beat_times,
    )


def _analyze_sections(sequence: NoteSequence, config: Config) -> Analysis:
    """Analyse each declared stretch on its own, and hold them together."""
    cfg = config.score
    starts = [0.0] + [t for t in sorted(cfg.sections) if t > 0.0]
    plain = replace(config, score=replace(cfg, sections=()))

    parts: list[Analysis] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else math.inf
        inside = [n for n in sequence if start - 1e-9 <= n.onset < end]
        if not inside:
            continue
        part = NoteSequence.of(inside, tempo=sequence.tempo, key=sequence.key,
                               source=sequence.source)
        try:
            analysis = analyze(part, plain)
        except ScoreError:
            # Too little music to read a tempo from; the stretch before it
            # describes this one too.
            continue
        parts.append(replace(analysis, start=start))

    if not parts:
        return analyze(sequence, plain)

    # The key is the piece's, not the stretch's. Read from one section alone, a
    # page in D flat major came back as G sharp minor: the fast half of it
    # dwells on the dominant, and eight bars are not enough to tell a key from
    # its neighbours.
    if cfg.fixed_key:
        key, key_confidence = cfg.fixed_key, 1.0
    else:
        key, key_confidence = estimate_key(sequence, config)
    parts = [replace(part, key=key, key_confidence=key_confidence) for part in parts]

    if len(parts) == 1:
        return replace(parts[0], start=0.0)
    return replace(parts[0], sections=tuple(parts))


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
