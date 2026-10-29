"""Standard MIDI File output, written directly.

No dependency: an SMF is a header chunk and a few track chunks of delta-timed
events, and the format is small enough that adding a library costs more than it
saves — particularly for a project that already pins OpenCV and numpy.

MIDI is the honest output of this pipeline. It says exactly what the tiles said:
these pitches, at these times, for these durations. Notation (see ``musicxml``)
has to make editorial decisions that MIDI never does, which is why this is the
format to trust when the two disagree.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

from ..notes import Note, NoteSequence
from ..score import Analysis

log = logging.getLogger(__name__)

TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 120.0

# Sharps (positive) or flats (negative) in each key signature, by tonic pitch
# class. Minor keys take their relative major's signature.
# Five flats, not seven sharps, for D flat major and B flat minor: the
# signature follows the name a page would carry.
_MAJOR_SIGNATURE = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 5: -1, 10: -2, 3: -3, 8: -4}
_MINOR_SIGNATURE = {9: 0, 4: 1, 11: 2, 6: 3, 1: 4, 8: 5, 3: -6, 10: -5, 2: -1, 7: -2, 0: -3, 5: -4}

_PITCH_CLASSES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def key_signature(key: str | None) -> tuple[int, int]:
    """(sharps or flats, 0 for major / 1 for minor) from a key name."""
    if not key:
        return 0, 0
    parts = key.split()
    pitch_class = _PITCH_CLASSES.get(parts[0])
    if pitch_class is None:
        return 0, 0
    minor = len(parts) > 1 and parts[1].lower().startswith("min")
    table = _MINOR_SIGNATURE if minor else _MAJOR_SIGNATURE
    return table.get(pitch_class, 0), int(minor)


def _vlq(value: int) -> bytes:
    """MIDI variable-length quantity: seven bits per byte, high bit as 'more'."""
    if value < 0:
        raise ValueError(f"delta times cannot be negative, got {value}")
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">I", len(payload)) + payload


def _meta(kind: int, payload: bytes) -> bytes:
    return bytes([0xFF, kind]) + _vlq(len(payload)) + payload


def _track(events: list[tuple[int, bytes]], name: str | None = None) -> bytes:
    """Absolute-time events to a delta-timed track chunk."""
    data = bytearray()
    if name:
        data += _vlq(0) + _meta(0x03, name.encode("ascii", "replace"))

    previous = 0
    # Sort by time, and put note-offs before note-ons at the same instant so a
    # repeated note retriggers instead of being cut short by its predecessor.
    for time, payload in sorted(events, key=lambda e: (e[0], e[1][0] & 0xF0)):
        data += _vlq(time - previous) + payload
        previous = time

    data += _vlq(0) + _meta(0x2F, b"")
    return _chunk(b"MTrk", bytes(data))


def _releases(notes, ticks) -> list[tuple[Note, int]]:
    """Each note with the tick it is released at.

    A key cannot sound twice at once. Rounded to the beat grid, a held note can
    run past the next strike of the same key -- 280 times in one waltz -- and a
    MIDI note-off names only the key, so the first note's release silenced the
    second one early. A note is released by the next strike of its key, and a
    second strike on the same tick is dropped rather than written as a note
    with no length.
    """
    by_key: dict[int, list[Note]] = {}
    for note in notes:
        by_key.setdefault(note.pitch, []).append(note)
    kept: list[tuple[Note, int]] = []
    for strikes in by_key.values():
        strikes.sort(key=lambda n: n.onset)
        for index, note in enumerate(strikes):
            start, end = ticks(note.onset), ticks(note.offset)
            if index + 1 < len(strikes):
                end = min(end, ticks(strikes[index + 1].onset))
            if end > start:
                kept.append((note, end))
            elif index + 1 == len(strikes):
                kept.append((note, start + 1))
    kept.sort(key=lambda pair: pair[0].onset)
    return kept


def write(
    sequence: NoteSequence,
    path: str | Path,
    analysis: Analysis | None = None,
) -> Path:
    """Write a format-1 MIDI file: one conductor track, one track per hand."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tempo = (analysis.tempo if analysis else sequence.tempo) or DEFAULT_TEMPO
    beat = analysis.beat if analysis else 60.0 / tempo
    beats_per_bar = analysis.beats_per_bar if analysis else 4
    beat_type = analysis.beat_type if analysis else 4
    key = (analysis.key if analysis else sequence.key) or None

    # MIDI counts in quarter notes, whatever the beat is written as.
    quarter = beat * beat_type / 4.0
    denominator = 2 if beat_type == 4 else 3  # 2^2 = 4, 2^3 = 8

    def ticks(seconds: float) -> int:
        return max(0, int(round(seconds / quarter * TICKS_PER_BEAT)))

    conductor: list[tuple[int, bytes]] = [
        (0, _meta(0x51, struct.pack(">I", int(round(1_000_000 * quarter)))[1:])),
        (0, _meta(0x58, bytes([beats_per_bar, denominator, 24, 8]))),
        (0, _meta(0x59, bytes([key_signature(key)[0] & 0xFF, key_signature(key)[1]]))),
    ]

    tracks = [_track(conductor, name="DropScore")]

    for channel, (hand, label) in enumerate((("R", "Right hand"), ("L", "Left hand"))):
        notes = sequence.hand(hand)
        if not notes:
            continue
        events: list[tuple[int, bytes]] = []
        for note, release in _releases(notes, ticks):
            velocity = max(1, min(127, note.velocity))
            events.append((ticks(note.onset), bytes([0x90 | channel, note.pitch, velocity])))
            events.append((release, bytes([0x80 | channel, note.pitch, 0])))
        tracks.append(_track(events, name=label))

    header = _chunk(b"MThd", struct.pack(">HHH", 1, len(tracks), TICKS_PER_BEAT))
    path.write_bytes(header + b"".join(tracks))

    log.info("wrote %d notes to %s", len(sequence), path)
    return path
