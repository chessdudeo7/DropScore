"""Stage 8.

The MIDI tests parse the bytes back with a small reader written here rather than
trusting the writer's own view of what it produced. A format written by hand
needs to be read by something that does not share its assumptions.
"""

from __future__ import annotations

import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from dropscore.export import extension, midi, musicxml, write
from dropscore.export.midi import TICKS_PER_BEAT, key_signature
from dropscore.export.musicxml import DIVISIONS, note_type, spell
from dropscore.export.pdf import EngraverNotFound, find_engraver
from dropscore.notes import Note, NoteSequence
from dropscore.score import analyze
from dropscore.synth import generate


# ── a minimal SMF reader, for verification only ──────────────────────


def _read_vlq(data: bytes, index: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index


def parse_midi(path: Path) -> dict:
    """Header fields and note events, decoded independently of the writer."""
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    (length,) = struct.unpack(">I", data[4:8])
    fmt, track_count, division = struct.unpack(">HHH", data[8 : 8 + length])

    tracks: list[list[tuple[int, str, int, int]]] = []
    tempos: list[int] = []
    signatures: list[tuple[int, int]] = []
    index = 8 + length

    while index < len(data):
        assert data[index : index + 4] == b"MTrk"
        (size,) = struct.unpack(">I", data[index + 4 : index + 8])
        body = data[index + 8 : index + 8 + size]
        index += 8 + size

        events: list[tuple[int, str, int, int]] = []
        position = 0
        time = 0
        while position < len(body):
            delta, position = _read_vlq(body, position)
            time += delta
            status = body[position]
            position += 1

            if status == 0xFF:
                kind = body[position]
                position += 1
                payload_length, position = _read_vlq(body, position)
                payload = body[position : position + payload_length]
                position += payload_length
                if kind == 0x51:
                    tempos.append(int.from_bytes(payload, "big"))
                elif kind == 0x59:
                    signatures.append(
                        (int.from_bytes(payload[:1], "big", signed=True), payload[1])
                    )
                elif kind == 0x2F:
                    break
            elif status & 0xF0 in (0x80, 0x90):
                pitch, velocity = body[position], body[position + 1]
                position += 2
                name = "on" if status & 0xF0 == 0x90 and velocity else "off"
                events.append((time, name, pitch, status & 0x0F))
            else:  # pragma: no cover - the writer emits nothing else
                raise AssertionError(f"unexpected status byte {status:#x}")

        tracks.append(events)

    return {
        "format": fmt,
        "track_count": track_count,
        "division": division,
        "tracks": tracks,
        "tempos": tempos,
        "signatures": signatures,
    }


def _two_hands() -> NoteSequence:
    return NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=0.5, hand="R", velocity=90),
            Note(onset=0.5, pitch=64, duration=0.5, hand="R", velocity=80),
            Note(onset=0.0, pitch=48, duration=1.0, hand="L", velocity=70),
        ],
        tempo=120.0,
        key="C major",
    )


# ── MIDI ─────────────────────────────────────────────────────────────


def test_midi_header_is_wellformed(tmp_path: Path) -> None:
    parsed = parse_midi(midi.write(_two_hands(), tmp_path / "a.mid"))
    assert parsed["format"] == 1
    assert parsed["division"] == TICKS_PER_BEAT
    assert parsed["track_count"] == len(parsed["tracks"]) == 3  # conductor + 2 hands


def test_midi_note_times_match_the_tempo(tmp_path: Path) -> None:
    parsed = parse_midi(midi.write(_two_hands(), tmp_path / "a.mid"))
    right = parsed["tracks"][1]

    # At 120 BPM a beat is 0.5s, so a half-second note is exactly one beat.
    starts = [e for e in right if e[1] == "on"]
    assert [e[0] for e in starts] == [0, TICKS_PER_BEAT]
    assert [e[2] for e in starts] == [60, 64]


def test_midi_each_hand_gets_its_own_track(tmp_path: Path) -> None:
    parsed = parse_midi(midi.write(_two_hands(), tmp_path / "a.mid"))
    right = {e[2] for e in parsed["tracks"][1] if e[1] == "on"}
    left = {e[2] for e in parsed["tracks"][2] if e[1] == "on"}
    assert right == {60, 64}
    assert left == {48}


def test_midi_every_note_is_released(tmp_path: Path) -> None:
    parsed = parse_midi(midi.write(generate(seed=2, bars=4), tmp_path / "a.mid"))
    for events in parsed["tracks"][1:]:
        ons = sum(1 for e in events if e[1] == "on")
        offs = sum(1 for e in events if e[1] == "off")
        assert ons == offs


def test_midi_repeated_note_is_released_before_it_restarts(tmp_path: Path) -> None:
    """Otherwise the second strike is silenced by the first note's release."""
    sequence = NoteSequence.of(
        [Note(onset=i * 0.5, pitch=60, duration=0.5) for i in range(4)], tempo=120.0
    )
    parsed = parse_midi(midi.write(sequence, tmp_path / "a.mid"))
    events = [e for e in parsed["tracks"][1]]

    at_beat = [e for e in events if e[0] == TICKS_PER_BEAT]
    assert [e[1] for e in at_beat] == ["off", "on"]


def test_midi_writes_the_tempo_and_key(tmp_path: Path) -> None:
    sequence = _two_hands()
    parsed = parse_midi(midi.write(sequence, tmp_path / "a.mid"))
    assert parsed["tempos"] == [500_000]  # 120 BPM in microseconds per quarter
    assert parsed["signatures"] == [(0, 0)]  # C major


@pytest.mark.parametrize(
    "key,expected",
    [
        ("C major", (0, 0)),
        ("G major", (1, 0)),
        ("F major", (-1, 0)),
        ("A minor", (0, 1)),
        ("E minor", (1, 1)),
        ("D minor", (-1, 1)),
        (None, (0, 0)),
        ("nonsense", (0, 0)),
    ],
)
def test_key_signatures(key: str | None, expected: tuple[int, int]) -> None:
    assert key_signature(key) == expected


def test_midi_uses_the_analysis_over_the_sequence(tmp_path: Path) -> None:
    sequence = generate(seed=5, bars=8, tempo=96.0)
    analysis = analyze(sequence)
    parsed = parse_midi(midi.write(sequence, tmp_path / "a.mid", analysis))
    assert parsed["tempos"][0] == pytest.approx(60_000_000 / analysis.tempo, rel=0.01)


# ── MusicXML ─────────────────────────────────────────────────────────


def test_musicxml_is_parseable_and_has_two_staves(tmp_path: Path) -> None:
    path = musicxml.write(_two_hands(), tmp_path / "a.musicxml")
    root = ET.parse(path).getroot()

    assert root.tag == "score-partwise"
    assert root.findtext(".//attributes/staves") == "2"
    assert {c.findtext("sign") for c in root.findall(".//clef")} == {"G", "F"}


def test_musicxml_keeps_every_pitch(tmp_path: Path) -> None:
    sequence = _two_hands()
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml")).getroot()

    written = set()
    for note in root.findall(".//note"):
        pitch = note.find("pitch")
        if pitch is None:
            continue
        step = pitch.findtext("step")
        alter = int(pitch.findtext("alter") or 0)
        octave = int(pitch.findtext("octave"))
        written.add((step, alter, octave))

    for note in sequence:
        assert spell(note.pitch, flats=False) in written


def test_musicxml_measures_are_full(tmp_path: Path) -> None:
    """Every measure must add up, per voice, or the file is invalid.

    Per *staff* was the right unit while a staff held one voice. With two, a
    staff writes the same span of time twice over, so its notes sum to double
    the bar and only the voices themselves have to balance.
    """
    sequence = generate(seed=8, bars=4, tempo=120.0)
    analysis = analyze(sequence)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analysis)).getroot()

    per_measure = analysis.beats_per_bar * DIVISIONS
    for number, measure in enumerate(root.findall(".//measure"), start=1):
        totals: dict[str, int] = {}
        for note in measure.findall("note"):
            if note.find("chord") is not None:
                continue
            voice = note.findtext("voice")
            totals[voice] = totals.get(voice, 0) + int(note.findtext("duration"))
        assert totals, f"measure {number} has no notes"
        for voice, total in totals.items():
            assert total == per_measure, (
                f"measure {number} voice {voice} sums to {total}, not {per_measure}"
            )


def test_musicxml_holds_a_note_under_a_moving_line(tmp_path: Path) -> None:
    """One hand sustaining while the other fingers move is not truncation.

    With a single voice per staff every note had to be cut at the next onset,
    so a held bass note became a short note and a rest -- on a real
    transcription, a page of dotted eighths.
    """
    notes = []
    for bar in range(4):
        start = bar * 2.0
        notes.append(Note(onset=start, pitch=48, duration=2.0, hand="L"))  # held
        for step in range(4):  # a line moving over it
            notes.append(
                Note(onset=start + step * 0.5, pitch=55 + step, duration=0.5, hand="L")
            )
    sequence = NoteSequence.of(notes, tempo=120.0)
    analysis = analyze(sequence)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analysis)).getroot()

    # The held C should still be two beats' worth wherever it appears, rather
    # than cut to the half-beat before the line's next note.
    per_beat = DIVISIONS
    longest = 0
    for note in root.findall(".//note"):
        pitch = note.find("pitch")
        if pitch is None or pitch.findtext("step") != "C":
            continue
        longest = max(longest, int(note.findtext("duration")))

    assert longest >= 2 * per_beat, (
        f"the held C is only {longest} divisions; it was truncated by the line "
        f"above it (a beat is {per_beat})"
    )


def test_musicxml_ties_notes_across_barlines(tmp_path: Path) -> None:
    sequence = NoteSequence.of(
        [Note(onset=1.5, pitch=60, duration=2.0, hand="R")], tempo=120.0
    )
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml")).getroot()

    ties = root.findall(".//tie")
    assert {t.get("type") for t in ties} == {"start", "stop"}


def test_musicxml_marks_simultaneous_notes_as_a_chord(tmp_path: Path) -> None:
    sequence = NoteSequence.of(
        [Note(onset=0.0, pitch=p, duration=1.0, hand="R") for p in (60, 64, 67)],
        tempo=120.0,
    )
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml")).getroot()
    assert len(root.findall(".//chord")) == 2  # three notes, two of them chorded


def test_musicxml_spells_flats_in_flat_keys(tmp_path: Path) -> None:
    sequence = NoteSequence.of(
        [Note(onset=i * 0.5, pitch=p, duration=0.5) for i, p in enumerate([65, 67, 69, 70, 72])],
        tempo=120.0,
        key="F major",
    )
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml")).getroot()

    assert root.findtext(".//key/fifths") == "-1"
    alters = {
        (p.findtext("step"), int(p.findtext("alter") or 0))
        for p in root.findall(".//pitch")
    }
    assert ("B", -1) in alters  # not A#


@pytest.mark.parametrize(
    "duration,expected",
    [
        (DIVISIONS * 4, ("whole", False)),
        (DIVISIONS * 2, ("half", False)),
        (DIVISIONS * 3, ("half", True)),
        (DIVISIONS, ("quarter", False)),
        (DIVISIONS // 2, ("eighth", False)),
        (DIVISIONS // 4, ("16th", False)),
    ],
)
def test_note_types(duration: int, expected: tuple[str, bool]) -> None:
    assert note_type(duration) == expected


# ── dispatch and PDF ─────────────────────────────────────────────────


def test_extension_lookup() -> None:
    assert extension("midi") == ".mid"
    assert extension("musicxml") == ".musicxml"
    with pytest.raises(ValueError, match="unknown format"):
        extension("ogg")


@pytest.mark.parametrize("format", ["midi", "musicxml", "json"])
def test_write_dispatches_by_name(format: str, tmp_path: Path) -> None:
    path = write(_two_hands(), tmp_path / f"a{extension(format)}", format)
    assert path.exists() and path.stat().st_size > 0


def test_pdf_says_what_to_install_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both routes gone, so the message must name both."""
    from dropscore.export import pdf  # noqa: PLC0415

    monkeypatch.setattr(pdf, "find_engraver", lambda: None)
    monkeypatch.setattr(pdf, "_verovio_available", lambda: False)

    with pytest.raises(EngraverNotFound, match="verovio"):
        pdf.write(_two_hands(), tmp_path / "a.pdf")
    with pytest.raises(EngraverNotFound, match="MuseScore"):
        pdf.write(_two_hands(), tmp_path / "b.pdf")


def test_pdf_engraves_in_process_without_musescore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the verovio route: no GUI application on PATH."""
    from dropscore.export import pdf  # noqa: PLC0415

    if not pdf._verovio_available():
        pytest.skip('verovio route not installed (pip install "dropscore[pdf]")')

    monkeypatch.setattr(pdf, "find_engraver", lambda: None)
    out = pdf.write(_two_hands(), tmp_path / "a.pdf")

    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:5] == b"%PDF-"
    assert not out.with_suffix(".musicxml").exists(), "left its intermediate behind"


def test_pdf_keeps_the_musicxml_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dropscore.export import pdf  # noqa: PLC0415

    if not pdf._verovio_available():
        pytest.skip("verovio route not installed")

    monkeypatch.setattr(pdf, "find_engraver", lambda: None)
    out = pdf.write(_two_hands(), tmp_path / "a.pdf", keep_musicxml=True)

    assert out.with_suffix(".musicxml").exists()
