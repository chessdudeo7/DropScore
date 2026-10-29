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


def test_pdf_writes_no_music_font_characters_into_text() -> None:
    """The PDF route has no music font, so a music-font character in text --
    the note of a tempo mark -- became a black box above the first bar."""
    from dropscore.export.pdf import _plain_text_glyphs  # noqa: PLC0415

    svg = (
        '<tspan font-family="Leipzig" font-size="720px"></tspan>'
        '<tspan font-size="405px">\xa0=\xa0</tspan><tspan font-size="405px">130</tspan>'
    )
    plain = _plain_text_glyphs(svg)

    assert not any("" <= ch <= "" for ch in plain)
    assert "Tempo" in plain and "130" in plain


def test_pdf_leaves_a_musicxml_it_did_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job asking for MusicXML and PDF writes both beside each other, and
    the engraver's intermediate is the same file -- removing it afterwards
    took away the MusicXML the job had been asked for."""
    from dropscore.export import pdf  # noqa: PLC0415

    if not pdf._verovio_available():
        pytest.skip("verovio route not installed")

    monkeypatch.setattr(pdf, "find_engraver", lambda: None)
    asked_for = musicxml.write(_two_hands(), tmp_path / "piece.musicxml")
    pdf.write(_two_hands(), tmp_path / "piece.pdf")

    assert Path(asked_for).exists()


def test_pdf_engraves_from_a_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service engraves in a worker thread, not the one that imported
    verovio -- and there the wheel's default font path pointed at the machine
    it was built on, so every PDF the service was asked for failed."""
    import threading  # noqa: PLC0415

    from dropscore.export import pdf  # noqa: PLC0415

    if not pdf._verovio_available():
        pytest.skip("verovio route not installed")

    monkeypatch.setattr(pdf, "find_engraver", lambda: None)
    outcome: list[object] = []

    def engrave() -> None:
        try:
            outcome.append(pdf.write(_two_hands(), tmp_path / "a.pdf"))
        except Exception as exc:  # noqa: BLE001 -- reported below
            outcome.append(exc)

    worker = threading.Thread(target=engrave)
    worker.start()
    worker.join()

    assert isinstance(outcome[0], Path), outcome[0]
    assert outcome[0].read_bytes()[:5] == b"%PDF-"


def test_pdf_from_a_worker_thread_logs_no_font_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A toolkit built with its default fonts looked on the build machine first
    and printed five errors into the server log on every PDF, before the right
    path was set and the engraving went fine."""
    import threading  # noqa: PLC0415

    from dropscore.export import pdf  # noqa: PLC0415

    if not pdf._verovio_available():
        pytest.skip("verovio route not installed")

    monkeypatch.setattr(pdf, "find_engraver", lambda: None)
    worker = threading.Thread(target=lambda: pdf.write(_two_hands(), tmp_path / "a.pdf"))
    worker.start()
    worker.join()

    assert "could not be loaded" not in capfd.readouterr().err


def test_pdf_keeps_the_musicxml_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dropscore.export import pdf  # noqa: PLC0415

    if not pdf._verovio_available():
        pytest.skip("verovio route not installed")

    monkeypatch.setattr(pdf, "find_engraver", lambda: None)
    out = pdf.write(_two_hands(), tmp_path / "a.pdf", keep_musicxml=True)

    assert out.with_suffix(".musicxml").exists()


def test_bar_lines_fall_on_the_measured_downbeat_not_the_start_of_the_video() -> None:
    """A recording starts wherever it was started. Measuring the page from
    time zero put every bar line that far off true: three quarter notes a bar
    came out as a rest, an offbeat sixteenth and more rests."""
    from dropscore.export.musicxml import _from_the_downbeat  # noqa: PLC0415
    from dropscore.score import analyze  # noqa: PLC0415

    beat, start = 0.6, 0.37
    notes = [Note(onset=start + i * beat, pitch=64, duration=0.15) for i in range(60)]
    for bar in range(4, 20):
        notes.append(Note(onset=start + bar * 3 * beat, pitch=57, duration=1.6))
    sequence = NoteSequence.of(notes)

    analysis = analyze(sequence)
    anchored = _from_the_downbeat(sequence, analysis)
    positions = [n.onset / analysis.beat for n in anchored]
    assert all(abs(p - round(p)) < 0.05 for p in positions), "notes fell between beats"

    bar = analysis.beat * analysis.beats_per_bar
    downbeats = [n.onset for n in anchored if n.pitch == 57]
    assert all(abs(t / bar - round(t / bar)) < 0.05 for t in downbeats), "bars do not start on the bass"


_TYPE_DIVISIONS = {"whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5, "16th": 0.25, "32nd": 0.125}


def _struck(root: ET.Element) -> int:
    """Notes that begin in the file: not rests, and not the far end of a tie."""
    return sum(
        1
        for note in root.iter("note")
        if note.find("pitch") is not None
        and not any(tie.get("type") == "stop" for tie in note.findall("tie"))
    )


def test_musicxml_writes_every_note(tmp_path: Path) -> None:
    """Counted, not merely looked for by pitch. Checking only that each pitch
    appears somewhere passed while a change to the writer dropped 69 of one
    page's 161 notes -- every voice after the first was read from the wrong
    bar, and the pitches all still turned up elsewhere."""
    sequence = generate(seed=8, bars=8, tempo=110.0)
    analysis = analyze(sequence)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analysis)).getroot()
    assert _struck(root) == len(sequence)


def test_musicxml_writes_every_value_as_one_that_exists(tmp_path: Path) -> None:
    """A note's written type must be its length. A note of five sixteenths was
    written as a quarter with the duration of five sixteenths, and a notation
    program draws the quarter: 44% of the symbols on one transcription."""
    beat = 0.5
    notes = [
        Note(onset=i * 2 * beat, pitch=60 + i, duration=beat * length, hand="R")
        for i, length in enumerate((1.25, 0.75, 1.75, 0.25, 1.5, 1.125))
    ]
    sequence = NoteSequence.of(notes)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml")).getroot()

    for note in root.iter("note"):
        written = _TYPE_DIVISIONS[note.findtext("type")] * DIVISIONS
        if note.find("dot") is not None:
            written *= 1.5
        assert int(note.findtext("duration")) == written, (
            f"a {note.findtext('type')} lasting {note.findtext('duration')} divisions"
        )
    assert _struck(root) == len(sequence)


def test_musicxml_marks_the_tempo_and_where_it_changes(tmp_path: Path) -> None:
    from dataclasses import replace  # noqa: PLC0415

    from dropscore.config import DEFAULT  # noqa: PLC0415

    slow, fast = generate(seed=3, bars=8, tempo=70.0), generate(seed=4, bars=8, tempo=120.0)
    boundary = max(n.onset + n.duration for n in slow) + 0.1
    sequence = NoteSequence.of(
        list(slow) + [Note(n.onset + boundary, n.pitch, n.duration, n.hand, n.velocity) for n in fast]
    )
    config = replace(DEFAULT, score=replace(DEFAULT.score, sections=(boundary,)))
    analysis = analyze(sequence, config)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analysis)).getroot()

    tempos = [float(sound.get("tempo")) for sound in root.iter("sound") if sound.get("tempo")]
    assert len(tempos) == 2, f"tempo marks {tempos}"
    assert tempos[0] == pytest.approx(analysis.sections[0].tempo, rel=0.01)
    assert tempos[1] == pytest.approx(analysis.sections[1].tempo, rel=0.01)
    assert root.find(".//measure[@number='1']/direction/direction-type/metronome") is not None
    assert _struck(root) == len(sequence)


def test_musicxml_writes_the_staff_a_note_reads_on_not_the_hand_that_played_it() -> None:
    """A hand crossing over the other still prints where it reads. The
    sequence carries the hand; the staff is decided as the score is written."""
    notes = [Note(onset=i * 0.5, pitch=74 + (i % 3), duration=0.4, hand="L") for i in range(16)]
    root = musicxml.build(NoteSequence.of(notes, tempo=120.0)).getroot()

    # Rests are <note> elements too, and the empty staff is full of them.
    staves = {n.findtext("staff") for n in root.findall(".//note") if n.find("pitch") is not None}
    assert staves == {"1"}, f"a passage well above middle C was written on staff {staves}"


@pytest.mark.parametrize(
    "key,pitch,expected",
    [
        ("C major", 70, ("B", -1, 4)),  # not A#
        ("F# major", 65, ("E", 1, 4)),  # its seventh, not F natural
        ("Gb major", 71, ("C", -1, 5)),  # its fourth, written in the octave above B4
        ("A minor", 68, ("G", 1, 4)),  # the leading tone
        ("C# minor", 72, ("B", 1, 4)),  # the leading tone, B#4 sounding C5
    ],
)
def test_pitches_are_spelled_by_scale_degree(key: str, pitch: int, expected) -> None:
    from dropscore.export.musicxml import spell_in_key  # noqa: PLC0415

    assert spell_in_key(pitch, key) == expected


def test_musicxml_spells_the_seventh_of_f_sharp_major(tmp_path: Path) -> None:
    sequence = NoteSequence.of(
        [Note(onset=i * 0.5, pitch=p, duration=0.5) for i, p in enumerate([66, 68, 70, 65, 66])],
        tempo=120.0,
        key="F# major",
    )
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml")).getroot()

    alters = {
        (p.findtext("step"), int(p.findtext("alter") or 0))
        for p in root.findall(".//pitch")
    }
    assert ("E", 1) in alters
    assert ("F", 0) not in alters


def _structure_problems(root: ET.Element) -> list[str]:
    """Measures that do not add up, read the way a notation program reads them.

    A cursor walks each measure: a note that is not part of a chord moves it
    on, <backup> moves it back and <forward> moves it on. No voice may run past
    the bar or fall short of it, the cursor may never go below the barline,
    every tie that starts must stop on the same pitch and staff, and every beam
    that begins must end in the same bar and voice.
    """
    problems: list[str] = []
    for part in root.iter("part"):
        divisions = beats = beat_type = None
        open_ties: dict[tuple, int] = {}
        for measure in part.iter("measure"):
            number = measure.get("number")
            attributes = measure.find("attributes")
            if attributes is not None:
                if attributes.find("divisions") is not None:
                    divisions = int(attributes.findtext("divisions"))
                time = attributes.find("time")
                if time is not None:
                    beats, beat_type = int(time.findtext("beats")), int(time.findtext("beat-type"))
            length = divisions * 4 * beats // beat_type
            cursor, furthest, voices, beams = 0, 0, {}, set()
            for element in measure:
                if element.tag == "note":
                    for beam in element.findall("beam"):
                        key = (element.findtext("voice"), beam.get("number"))
                        if beam.text == "begin":
                            if key in beams:
                                problems.append(f"m{number}: beam begins twice")
                            beams.add(key)
                        elif beam.text in ("continue", "end") and key not in beams:
                            problems.append(f"m{number}: beam {beam.text} with nothing open")
                        if beam.text == "end":
                            beams.discard(key)
                    pitch = element.find("pitch")
                    if pitch is not None:
                        key = (pitch.findtext("step"), pitch.findtext("alter"),
                               pitch.findtext("octave"), element.findtext("staff"))
                        ties = [tie.get("type") for tie in element.findall("tie")]
                        if "stop" in ties:
                            if not open_ties.get(key):
                                problems.append(f"m{number}: tie stops with nothing open")
                            else:
                                open_ties[key] -= 1
                        if "start" in ties:
                            open_ties[key] = open_ties.get(key, 0) + 1
                    if element.find("chord") is not None:
                        continue
                    duration = int(element.findtext("duration"))
                    cursor += duration
                    voice = element.findtext("voice", "1")
                    voices[voice] = voices.get(voice, 0) + duration
                elif element.tag == "backup":
                    cursor -= int(element.findtext("duration"))
                elif element.tag == "forward":
                    cursor += int(element.findtext("duration"))
                if cursor < 0:
                    problems.append(f"m{number}: backed up past the barline")
                furthest = max(furthest, cursor)
            if beams:
                problems.append(f"m{number}: a beam is never ended")
            if furthest > length:
                problems.append(f"m{number}: runs to {furthest} of {length}")
            for voice, total in voices.items():
                if total != length:
                    problems.append(f"m{number} voice {voice}: {total} of {length}")
        if any(open_ties.values()):
            problems.append(f"{part.get('id')}: a tie is never stopped")
    return problems


def _played_badly(seed: int) -> NoteSequence:
    """Two hands with loose timing, odd lengths and overlaps: a transcription."""
    import random

    rng = random.Random(seed)
    notes, t = [], 0.3
    for _ in range(160):
        t += rng.choice([0.12, 0.25, 0.25, 0.5, 0.5, 0.75, 1.0]) + rng.gauss(0, 0.02)
        hand = rng.choice("LR")
        pitch = rng.randint(36, 59) if hand == "L" else rng.randint(60, 84)
        notes.append(Note(onset=t, pitch=pitch, duration=rng.uniform(0.05, 2.5), hand=hand))
        if rng.random() < 0.2:
            notes.append(Note(onset=t, pitch=pitch + rng.choice([3, 4, 7]),
                              duration=rng.uniform(0.1, 1.5), hand=hand))
    return NoteSequence.of(notes, tempo=100.0)


@pytest.mark.parametrize("seed", range(6))
def test_musicxml_structure_holds_for_messy_playing(seed: int, tmp_path: Path) -> None:
    from dropscore.config import DEFAULT
    from dropscore.score import postprocess

    sequence, analysis = postprocess(_played_badly(seed), DEFAULT)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analysis)).getroot()
    assert _structure_problems(root) == []


@pytest.mark.parametrize("beats_per_bar", [2, 3, 4])
def test_musicxml_structure_holds_in_every_metre(beats_per_bar: int, tmp_path: Path) -> None:
    sequence = generate(seed=3, bars=8, beats_per_bar=beats_per_bar, sustained=True)
    analysis = analyze(sequence)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analysis)).getroot()
    assert _structure_problems(root) == []


def test_structure_check_catches_a_broken_bar(tmp_path: Path) -> None:
    sequence = generate(seed=3, bars=4)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analyze(sequence))).getroot()
    note = next(root.iter("note"))
    note.find("duration").text = str(int(note.findtext("duration")) + 1)
    assert _structure_problems(root)


def test_structure_check_catches_an_unended_beam(tmp_path: Path) -> None:
    sequence = generate(seed=3, bars=4)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml", analyze(sequence))).getroot()
    assert not _structure_problems(root)
    end = next(b for b in root.iter("beam") if b.text == "end")
    end.text = "continue"
    assert _structure_problems(root)


def _sounding(path: Path, seconds_per_tick: float) -> list[tuple[float, int, float]]:
    """(onset, pitch, length) as a synthesizer plays the file: a note-off ends
    whichever strike of that key is sounding."""
    heard = []
    for events in parse_midi(path)["tracks"][1:]:
        down: dict[int, int] = {}
        for time, kind, pitch, _ in events:
            if kind == "on":
                if pitch in down:  # struck again while down: the first stops
                    heard.append((down[pitch], pitch, time - down[pitch]))
                down[pitch] = time
            elif pitch in down:
                start = down.pop(pitch)
                heard.append((start, pitch, time - start))
        assert not down, "a note is never released"
    return sorted((s * seconds_per_tick, p, n * seconds_per_tick) for s, p, n in heard)


def test_midi_note_held_into_its_next_strike_ends_there(tmp_path: Path) -> None:
    # Rounded to the grid, a held C can run past the next C. The file must
    # give the second C its full length, not end it with the first one's
    # release.
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=1.5, hand="R"),
            Note(onset=1.0, pitch=60, duration=1.0, hand="R"),
        ],
        tempo=120.0,
    )
    path = midi.write(sequence, tmp_path / "a.mid")
    heard = _sounding(path, 0.5 / TICKS_PER_BEAT)
    assert [(round(s, 3), p, round(n, 3)) for s, p, n in heard] == [(0.0, 60, 1.0), (1.0, 60, 1.0)]


@pytest.mark.parametrize("seed", range(3))
def test_midi_round_trip_keeps_every_note(seed: int, tmp_path: Path) -> None:
    from dropscore.config import DEFAULT
    from dropscore.score import postprocess

    sequence, analysis = postprocess(_played_badly(seed), DEFAULT)
    path = midi.write(sequence, tmp_path / "a.mid", analysis)
    tick = analysis.beat * analysis.beat_type / 4 / TICKS_PER_BEAT
    heard = _sounding(path, tick)

    by_key: dict[int, list[Note]] = {}
    for note in sequence:
        by_key.setdefault(note.pitch, []).append(note)
    expected = []
    for strikes in by_key.values():
        strikes.sort(key=lambda n: n.onset)
        for note, after in zip(strikes, strikes[1:] + [None]):
            end = note.offset if after is None else min(note.offset, after.onset)
            expected.append((note.onset, note.pitch, end - note.onset))
    expected.sort()

    assert len(heard) == len(expected)
    for (s, p, n), (es, ep, en) in zip(heard, expected):
        assert p == ep and abs(s - es) <= tick and abs(n - en) <= 2 * tick
    tempo = parse_midi(path)["tempos"][0]
    assert tempo == round(1_000_000 * analysis.beat * analysis.beat_type / 4)


def _beams(note: ET.Element) -> dict[str, str]:
    return {b.get("number"): b.text for b in note.findall("beam")}


def test_musicxml_beams_short_notes_a_beat_at_a_time(tmp_path: Path) -> None:
    """Verovio engraves the PDF and does not beam by itself: without beams in
    the file every eighth and sixteenth was printed with its own flag."""
    sixteenth, eighth = 0.125, 0.25  # at 120 bpm
    notes = [Note(onset=i * sixteenth, pitch=60 + i, duration=sixteenth) for i in range(4)]
    notes += [
        Note(onset=0.5, pitch=67, duration=eighth),
        Note(onset=0.75, pitch=65, duration=sixteenth),
        Note(onset=0.875, pitch=64, duration=sixteenth),
        Note(onset=1.0, pitch=62, duration=eighth),  # then a rest: nothing to join
        Note(onset=1.375, pitch=60, duration=sixteenth),
        Note(onset=1.5, pitch=60, duration=0.5),
    ]
    sequence = NoteSequence.of(notes, tempo=120.0)
    root = ET.parse(musicxml.write(sequence, tmp_path / "a.musicxml")).getroot()
    sounding = [n for n in root.find(".//measure").findall("note")
                if n.find("rest") is None and n.findtext("staff") == "1"]

    run = ["begin", "continue", "continue", "end"]
    assert [_beams(n) for n in sounding[:4]] == [{"1": k, "2": k} for k in run]
    assert [_beams(n) for n in sounding[4:7]] == [
        {"1": "begin"}, {"1": "continue", "2": "begin"}, {"1": "end", "2": "end"},
    ]
    assert all(not _beams(n) for n in sounding[7:]), "a rest or a quarter was beamed"


def test_compound_metre_beams_and_marks_the_dotted_quarter() -> None:
    from dropscore.export.musicxml import _add_tempo, _beam_span  # noqa: PLC0415

    assert _beam_span(6, 8, 6 * DIVISIONS // 2) == DIVISIONS * 3 // 2
    assert _beam_span(3, 8, 3 * DIVISIONS // 2) == 3 * DIVISIONS // 2  # 3/8 in one
    assert _beam_span(3, 4, 3 * DIVISIONS) == DIVISIONS

    measure = ET.Element("measure")
    _add_tempo(measure, 72.0, 6, 8)
    assert measure.find(".//beat-unit-dot") is not None
    assert measure.findtext(".//per-minute") == "48"
    assert measure.find(".//sound").get("tempo") == "72.0"  # playback stays in quarters
