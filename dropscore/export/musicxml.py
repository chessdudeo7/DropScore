"""MusicXML output.

Read the limitations before trusting the result. Turning a note stream into
*readable* notation is a partly aesthetic problem — voicing, beaming, rests,
enharmonic spelling, cross-staff writing — and this does the mechanical part
only:

* One voice per staff. Notes sharing an onset become a chord; a note beginning
  while another is still sounding truncates the earlier one rather than opening
  a second voice. Real piano writing needs voices, and that is where engraving
  quality is won or lost.
* Notes crossing a barline are split and tied, which is required for the file to
  be valid at all. So is any length that is not a single note value: it is
  written as the values that make it up, tied, and a rest likewise.
* Positions and lengths are rounded to the thirty-second grid, the finest value
  written, so anything stage 7 declined to quantize gets rounded here
  regardless.
* A tempo mark opens the piece, and a new one opens each declared section.
* No beaming, dynamics, articulation, slurs or pedal.

The consequence, stated plainly: **the MIDI is accurate and the notation is
approximate.** When they disagree, the MIDI is right.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET

from ..notes import Note, NoteSequence
from ..score import Analysis
from .midi import key_signature

log = logging.getLogger(__name__)

# Divisions per quarter note. 24 covers sixteenths and triplets exactly.
DIVISIONS = 24

_SHARP_SPELLING = (
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
)
_FLAT_SPELLING = (
    ("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
    ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0),
)

# Note types by duration in divisions, longest first.
_TYPES = (
    (DIVISIONS * 4, "whole"),
    (DIVISIONS * 2, "half"),
    (DIVISIONS, "quarter"),
    (DIVISIONS // 2, "eighth"),
    (DIVISIONS // 4, "16th"),
    (DIVISIONS // 8, "32nd"),
)


def spell(pitch: int, flats: bool) -> tuple[str, int, int]:
    """(step, alter, octave) for a MIDI pitch, spelled to suit the key."""
    step, alter = (_FLAT_SPELLING if flats else _SHARP_SPELLING)[pitch % 12]
    # Neither spelling table crosses an octave boundary — no Cb or B# — so the
    # octave is always the plain MIDI one.
    return step, alter, pitch // 12 - 1


# Lengths a single note or rest can be written as, longest first, in divisions.
_WRITABLE = tuple(
    sorted(
        {value for base, _ in _TYPES for value in (base, base * 3 // 2) if base * 3 % 2 == 0 or value == base},
        reverse=True,
    )
)

# The finest value written: a thirty-second. Anything finer is timing noise.
GRID = DIVISIONS // 8


def _parts(duration: int) -> list[int]:
    """A length as the note values it is written with, tied together.

    A single note carries a single written value, and a length that is not one
    was given the nearest value's name while keeping its own length: a note of
    fifteen divisions drawn as an eighth, which is twelve. A notation program
    draws what the name says, so the rhythm on the page was wrong for 44% of
    the symbols on one transcription. Split into values that each exist, the
    page and the timing agree.
    """
    parts = []
    remaining = duration
    while remaining >= GRID:
        value = next(v for v in _WRITABLE if v <= remaining)
        parts.append(value)
        remaining -= value
    return parts


def note_type(duration: int) -> tuple[str, bool]:
    """Closest written note value, and whether it needs a dot."""
    for base, name in _TYPES:
        if duration >= base:
            return name, duration >= base * 1.5
    return "32nd", False


@dataclass
class _Event:
    """A chord: one or more notes sharing an onset, in divisions."""

    start: int
    duration: int
    pitches: list[int]


VOICES_PER_STAFF = 2


def _lay_out(notes: list[Note], quarter: float, voices: int = 1) -> list[list[_Event]]:
    """Group one staff's notes into up to ``voices`` timelines.

    A single voice cannot hold a note and start another at the same time, so
    everything had to be truncated at the next onset whatever it was. That is
    wrong wherever a hand holds one note under a moving line — which is most
    piano writing — and on a real transcription it turned held notes into a
    page of dotted eighths followed by rests.

    Notes are grouped into chords by onset first, so simultaneous pitches stay
    one event rather than being split across voices. Each chord then goes to
    the first voice that is free, and truncation applies within a voice only:
    a held note in the second voice is no longer cut short by the first voice
    moving above it.
    """
    def to_divisions(seconds: float) -> int:
        # On the thirty-second grid, the finest value written. Timings a
        # division or two off it made rests of one twenty-fourth of a beat.
        return int(round(seconds / quarter * DIVISIONS / GRID)) * GRID

    grouped: dict[int, list[Note]] = {}
    for note in notes:
        grouped.setdefault(to_divisions(note.onset), []).append(note)

    # (start, natural end, pitches) per chord, in time order.
    # At least a thirty-second long: a note that was played is written, and one
    # held for a hair under the grid snapped to nothing and vanished.
    chords = [
        (start, max(start + GRID, max(to_divisions(n.offset) for n in grouped[start])),
         sorted(n.pitch for n in grouped[start]))
        for start in sorted(grouped)
    ]

    lanes: list[list[tuple[int, int, list[int]]]] = [[] for _ in range(max(1, voices))]
    for chord in chords:
        start = chord[0]
        free = next((lane for lane in lanes if not lane or lane[-1][1] <= start), None)
        if free is None:
            # Every voice is still sounding. Put it in the one that frees up
            # soonest and let truncation below shorten what it lands on.
            free = min(lanes, key=lambda lane: lane[-1][1])
        free.append(chord)

    out: list[list[_Event]] = []
    for lane in lanes:
        events: list[_Event] = []
        for index, (start, end, pitches) in enumerate(lane):
            length = end - start
            if index + 1 < len(lane):
                length = min(length, lane[index + 1][0] - start)
            if length < 1:
                continue
            events.append(_Event(start, length, pitches))
        out.append(events)
    return out


def _bar_starts(per_measure: int, changes: list[tuple[int, int]], end: int) -> list[int]:
    """Where every bar begins, in divisions, through any changes of bar length.

    ``changes`` pairs the division a section begins at with its bar length. A
    section begins on a bar line: one declared part way through a bar starts at
    the nearest bar line to it.
    """
    starts = [0]
    length = per_measure
    pending = sorted(changes)
    while starts[-1] < end:
        here = starts[-1]
        while pending and pending[0][0] <= here + length // 2:
            length = pending.pop(0)[1]
        starts.append(here + length)
    return starts


def _split_at_barlines(
    events: list[_Event], per_measure: int | list[int]
) -> dict[int, list[tuple[_Event, bool, bool]]]:
    """Distribute events into measures, tying anything that crosses a barline."""
    import bisect  # noqa: PLC0415

    starts = per_measure if isinstance(per_measure, list) else None
    measures: dict[int, list[tuple[_Event, bool, bool]]] = {}

    for event in events:
        start, remaining = event.start, event.duration
        first = True
        while remaining > 0:
            if starts is None:
                measure = start // per_measure
                offset = start % per_measure
                room = per_measure - offset
            else:
                measure = min(bisect.bisect_right(starts, start) - 1, len(starts) - 2)
                offset = start - starts[measure]
                room = starts[measure + 1] - start
            length = min(remaining, room)
            last = length == remaining
            measures.setdefault(measure, []).append(
                (_Event(offset, length, event.pitches), not first, not last)
            )
            start += length
            remaining -= length
            first = False

    return measures


def _add_note(
    parent: ET.Element,
    pitch: int | None,
    duration: int,
    staff: int,
    flats: bool,
    chord: bool = False,
    tied_from: bool = False,
    tied_to: bool = False,
    voice: int | None = None,
) -> None:
    element = ET.SubElement(parent, "note")
    if chord:
        ET.SubElement(element, "chord")

    if pitch is None:
        ET.SubElement(element, "rest")
    else:
        step, alter, octave = spell(pitch, flats)
        pitch_element = ET.SubElement(element, "pitch")
        ET.SubElement(pitch_element, "step").text = step
        if alter:
            ET.SubElement(pitch_element, "alter").text = str(alter)
        ET.SubElement(pitch_element, "octave").text = str(octave)

    ET.SubElement(element, "duration").text = str(duration)

    for start, kind in ((tied_from, "stop"), (tied_to, "start")):
        if start:
            ET.SubElement(element, "tie", type=kind)

    ET.SubElement(element, "voice").text = str(staff if voice is None else voice)
    name, dotted = note_type(duration)
    ET.SubElement(element, "type").text = name
    if dotted:
        ET.SubElement(element, "dot")
    ET.SubElement(element, "staff").text = str(staff)

    if tied_from or tied_to:
        notations = ET.SubElement(element, "notations")
        for start, kind in ((tied_from, "stop"), (tied_to, "start")):
            if start:
                ET.SubElement(notations, "tied", type=kind)


def _in_uniform_time(
    sequence: NoteSequence, analysis: Analysis
) -> tuple[NoteSequence, Analysis, float]:
    """Rewrite the playing against a steady beat, where it was not played to one.

    A page carries one tempo mark. Pushing and slowing is performance, not
    notation, and a note is written where it falls in the bar rather than at
    the second it happened -- so where the beats were tracked, each note is
    placed by which beat it fell on and how far through it, and the page is
    laid out from that. Without this the notes are quantized to the beats the
    player actually played and then engraved as though those beats were even,
    which puts them back where they started.
    """
    from ..score import beat_position  # noqa: PLC0415

    # Sections too: a piece that changes tempo is laid out in one steady
    # stream of beats, with the change carried by a tempo mark, not by bars of
    # a different length on the page.
    if not analysis.beat_times and not analysis.sections:
        return sequence, analysis, 0.0

    notes = list(sequence)
    if not notes:
        return sequence, analysis, 0.0

    beat = analysis.beat
    # Beats are counted from the first tracked one, so a note before it has a
    # negative position -- and a note cannot start before zero. The steady
    # timeline starts at the earliest note instead.
    shift = max(0.0, -min(beat_position(n.onset, analysis) for n in notes)) * beat
    rewritten = [
        Note(
            beat_position(n.onset, analysis) * beat + shift,
            n.pitch,
            max(
                (beat_position(n.onset + n.duration, analysis)
                 - beat_position(n.onset, analysis)) * beat,
                1e-4,
            ),
            n.hand,
            n.velocity,
        )
        for n in notes
    ]
    steady = replace(
        analysis,
        beat_phase=0.0,
        downbeat_phase=(beat_position(analysis.downbeat_phase, analysis) * beat + shift)
        % (beat * analysis.beats_per_bar),
        beat_times=(),
    )
    return (
        NoteSequence.of(rewritten, tempo=sequence.tempo, key=sequence.key,
                        source=sequence.source),
        steady,
        shift,
    )


def _from_the_downbeat(sequence: NoteSequence, analysis: Analysis) -> NoteSequence:
    return _downbeat_origin(sequence, analysis)[0]


def _downbeat_origin(sequence: NoteSequence, analysis: Analysis) -> tuple[NoteSequence, float]:
    """Measure time from a bar line rather than from the start of the video.

    Notes were placed by their seconds from zero, so the first bar began when
    the video did. A synthetic clip starts on a downbeat and never showed it;
    a recording starts wherever it was started, and every beat and bar line on
    the page sat that far off true however well the beat had been found --
    three quarter notes a bar came out as an eighth rest, a sixteenth and more
    rests, three times over. The analysis knows where the bars fall; the page
    now starts on the last bar line before the first note.
    """
    notes = list(sequence)
    if not notes or analysis.beat <= 0:
        return sequence, 0.0
    bar = analysis.beat * analysis.beats_per_bar
    first = min(n.onset for n in notes)
    origin = analysis.downbeat_phase + math.floor((first - analysis.downbeat_phase) / bar) * bar
    return (
        NoteSequence.of(
            [Note(n.onset - origin, n.pitch, n.duration, n.hand, n.velocity) for n in notes],
            tempo=sequence.tempo,
            key=sequence.key,
            source=sequence.source,
        ),
        origin,
    )


def build(sequence: NoteSequence, analysis: Analysis | None = None) -> ET.ElementTree:
    """Build a two-staff piano score."""
    tempo = (analysis.tempo if analysis else sequence.tempo) or 120.0
    # The counted beat, which the tempo describes only when it is a quarter.
    beat = analysis.beat if analysis else 60.0 / tempo
    beats_per_bar = analysis.beats_per_bar if analysis else 4
    beat_type = analysis.beat_type if analysis else 4
    key = (analysis.key if analysis else sequence.key) or None
    fifths, mode = key_signature(key)
    flats = fifths < 0

    # Durations are written in divisions of a quarter note, so a beat counted
    # as an eighth is half a quarter: everything below measures in quarters.
    quarter = beat * beat_type / 4.0
    per_measure = int(round(beats_per_bar * DIVISIONS * 4 / beat_type))

    # Written values, not held-key times — see notate_durations. Only done
    # here: the MIDI and the JSON stay faithful to what the video showed.
    # Where each section begins on the page, and what it changes to.
    changes: list[tuple[int, int, int, float]] = []  # division, bar length, beats, tempo
    if analysis is not None:
        from ..score import beat_position, notate_durations  # noqa: PLC0415

        original = analysis
        sequence, analysis, shift = _in_uniform_time(sequence, analysis)
        sequence = notate_durations(sequence, analysis)
        sequence, origin = _downbeat_origin(sequence, analysis)

        for section in original.sections[1:]:
            at = beat_position(section.start, original) * beat + shift - origin
            changes.append((
                int(round(at / quarter * DIVISIONS / GRID)) * GRID,
                int(round(section.beats_per_bar * DIVISIONS * 4 / beat_type)),
                section.beats_per_bar,
                section.tempo,
            ))

    # A note goes on the staff its register belongs to, which is not always the
    # hand that played it: in cross-hand writing a hand crosses the other and
    # the page still prints each note where it reads. The sequence carries the
    # hand, so the staves are decided here.
    from ..score import assign_staves  # noqa: PLC0415

    sequence = assign_staves(sequence)

    # Two voices per staff. MusicXML voice numbers are unique across the part,
    # so the staves take 1-2 and 5-6, which is the convention notation editors
    # expect and keeps a voice's identity obvious when reading the file.
    laid = {
        staff: _lay_out(sequence.hand(hand), quarter, VOICES_PER_STAFF)
        for staff, hand in ((1, "R"), (2, "L"))
    }
    end = max(
        (e.start + e.duration for lanes in laid.values() for lane in lanes for e in lane),
        default=per_measure,
    )
    bar_lines = _bar_starts(per_measure, [(at, length) for at, length, _, _ in changes], end)
    staves = {
        staff: [_split_at_barlines(lane, bar_lines) for lane in lanes]
        for staff, lanes in laid.items()
    }
    last_measure = max(
        (m for lanes in staves.values() for lane in lanes for m in lane),
        default=0,
    )

    # The first bar of each section, and what it announces.
    announce: dict[int, tuple[int, float]] = {}
    for at, _, beats, section_tempo in changes:
        bar = min(range(len(bar_lines) - 1), key=lambda i: abs(bar_lines[i] - at))
        announce[bar] = (beats, section_tempo)

    root = ET.Element("score-partwise", version="4.0")
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", id="P1")

    for index in range(last_measure + 1):
        measure = ET.SubElement(part, "measure", number=str(index + 1))

        if index == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
            key_element = ET.SubElement(attributes, "key")
            ET.SubElement(key_element, "fifths").text = str(fifths)
            ET.SubElement(key_element, "mode").text = "minor" if mode else "major"
            time_element = ET.SubElement(attributes, "time")
            ET.SubElement(time_element, "beats").text = str(beats_per_bar)
            ET.SubElement(time_element, "beat-type").text = str(beat_type)
            ET.SubElement(attributes, "staves").text = "2"
            for staff, sign, line in ((1, "G", 2), (2, "F", 4)):
                clef = ET.SubElement(attributes, "clef", number=str(staff))
                ET.SubElement(clef, "sign").text = sign
                ET.SubElement(clef, "line").text = str(line)

        per_measure = bar_lines[index + 1] - bar_lines[index]
        if index == 0:
            _add_tempo(measure, tempo)
        elif index in announce:
            beats, section_tempo = announce[index]
            if beats != beats_per_bar:
                attributes = ET.SubElement(measure, "attributes")
                time_element = ET.SubElement(attributes, "time")
                ET.SubElement(time_element, "beats").text = str(beats)
                ET.SubElement(time_element, "beat-type").text = str(beat_type)
                beats_per_bar = beats
            _add_tempo(measure, section_tempo)

        written = False
        for staff in (1, 2):
            for lane, bars in enumerate(staves[staff]):
                events = sorted(bars.get(index, []), key=lambda item: item[0].start)

                # An empty second voice is left out rather than filled with a
                # bar of rests, which would double every rest on the page.
                if lane and not events:
                    continue

                # Rewind to the start of the bar for every voice after the
                # first: each writes the same span of time over again.
                if written:
                    ET.SubElement(measure, "backup").append(
                        _text("duration", per_measure)
                    )
                written = True

                voice = VOICES_PER_STAFF * (staff - 1) + lane + 1
                if staff == 2:
                    voice += 3  # 5 and 6, leaving the usual gap after 1 and 2

                position = 0
                for event, tied_from, tied_to in events:
                    if event.start > position:
                        _add_rests(measure, event.start - position, staff, flats, voice)
                    values = _parts(event.duration)
                    for part_index, value in enumerate(values):
                        for order, pitch in enumerate(event.pitches):
                            _add_note(
                                measure, pitch, value, staff, flats,
                                chord=order > 0,
                                tied_from=tied_from if part_index == 0 else True,
                                tied_to=tied_to if part_index == len(values) - 1 else True,
                                voice=voice,
                            )
                    position = event.start + sum(values)

                if position < per_measure:
                    _add_rests(measure, per_measure - position, staff, flats, voice)

    return ET.ElementTree(root)


def _add_tempo(measure: ET.Element, tempo: float) -> None:
    """A metronome mark, and the playback tempo that goes with it."""
    direction = ET.SubElement(measure, "direction", placement="above")
    kind = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(kind, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(int(round(tempo)))
    ET.SubElement(direction, "sound", tempo=f"{tempo:.1f}")


def _add_rests(parent: ET.Element, duration: int, staff: int, flats: bool, voice: int) -> None:
    """Rests filling a span, each a value that exists."""
    for part in _parts(duration):
        _add_note(parent, None, part, staff, flats, voice=voice)


def _text(tag: str, value: int) -> ET.Element:
    element = ET.Element(tag)
    element.text = str(value)
    return element


def write(
    sequence: NoteSequence, path: str | Path, analysis: Analysis | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tree = build(sequence, analysis)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)

    log.info("wrote MusicXML to %s", path)
    return path
