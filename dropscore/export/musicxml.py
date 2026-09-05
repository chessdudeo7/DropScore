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
  be valid at all.
* No beaming, dynamics, articulation, slurs or pedal.
* Durations are rounded to the divisions grid, so anything stage 7 declined to
  quantize gets rounded here regardless.

The consequence, stated plainly: **the MIDI is accurate and the notation is
approximate.** When they disagree, the MIDI is right.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


def _lay_out(notes: list[Note], beat: float, voices: int = 1) -> list[list[_Event]]:
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
        return int(round(seconds / beat * DIVISIONS))

    grouped: dict[int, list[Note]] = {}
    for note in notes:
        grouped.setdefault(to_divisions(note.onset), []).append(note)

    # (start, natural end, pitches) per chord, in time order.
    chords = [
        (start, max(to_divisions(n.offset) for n in grouped[start]),
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


def _split_at_barlines(events: list[_Event], per_measure: int) -> dict[int, list[tuple[_Event, bool, bool]]]:
    """Distribute events into measures, tying anything that crosses a barline."""
    measures: dict[int, list[tuple[_Event, bool, bool]]] = {}

    for event in events:
        start, remaining = event.start, event.duration
        first = True
        while remaining > 0:
            measure = start // per_measure
            offset = start % per_measure
            length = min(remaining, per_measure - offset)
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


def build(sequence: NoteSequence, analysis: Analysis | None = None) -> ET.ElementTree:
    """Build a two-staff piano score."""
    tempo = (analysis.tempo if analysis else sequence.tempo) or 120.0
    beat = 60.0 / tempo
    beats_per_bar = analysis.beats_per_bar if analysis else 4
    key = (analysis.key if analysis else sequence.key) or None
    fifths, mode = key_signature(key)
    flats = fifths < 0

    per_measure = beats_per_bar * DIVISIONS

    # Written values, not held-key times — see notate_durations. Only done
    # here: the MIDI and the JSON stay faithful to what the video showed.
    if analysis is not None:
        from ..score import notate_durations  # noqa: PLC0415

        sequence = notate_durations(sequence, analysis)

    # Two voices per staff. MusicXML voice numbers are unique across the part,
    # so the staves take 1-2 and 5-6, which is the convention notation editors
    # expect and keeps a voice's identity obvious when reading the file.
    staves = {
        1: [
            _split_at_barlines(lane, per_measure)
            for lane in _lay_out(sequence.hand("R"), beat, VOICES_PER_STAFF)
        ],
        2: [
            _split_at_barlines(lane, per_measure)
            for lane in _lay_out(sequence.hand("L"), beat, VOICES_PER_STAFF)
        ],
    }
    last_measure = max(
        (m for lanes in staves.values() for lane in lanes for m in lane),
        default=0,
    )

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
            ET.SubElement(time_element, "beat-type").text = "4"
            ET.SubElement(attributes, "staves").text = "2"
            for staff, sign, line in ((1, "G", 2), (2, "F", 4)):
                clef = ET.SubElement(attributes, "clef", number=str(staff))
                ET.SubElement(clef, "sign").text = sign
                ET.SubElement(clef, "line").text = str(line)

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
                        _add_note(
                            measure, None, event.start - position, staff, flats,
                            voice=voice,
                        )
                    for order, pitch in enumerate(event.pitches):
                        _add_note(
                            measure, pitch, event.duration, staff, flats,
                            chord=order > 0, tied_from=tied_from, tied_to=tied_to,
                            voice=voice,
                        )
                    position = event.start + event.duration

                if position < per_measure:
                    _add_note(
                        measure, None, per_measure - position, staff, flats,
                        voice=voice,
                    )

    return ET.ElementTree(root)


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
