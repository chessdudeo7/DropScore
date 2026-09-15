"""Scoring a recording against its printed music.

The real pieces this is for live outside the repository, so these tests build
a small piece of their own: a pulse of quarters over a bass note a bar, in
three, with its printed notes written out alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dropscore.notes import Note, NoteSequence
from dropscore.printed import (
    PrintedResult,
    find_pieces,
    load,
    regressions,
    score,
    score_sequence,
)

FIRST_BEAT = 0.37
BEAT = 0.6
BARS = 16


def _printed_notes() -> list[list]:
    notes = []
    for beat in range(BARS * 3):
        notes.append([64, beat, 1, "R"])
    for bar in range(BARS):
        notes.append([45, bar * 3, 3, "L"])
    return notes


def _write_piece(directory: Path, **overrides) -> Path:
    data = {
        "title": "a pulse over a bass line",
        "video": "recording.mp4",
        "first_beat": FIRST_BEAT,
        "beat": BEAT,
        "window": [0, BARS * 3 - 1],
        "tempo": 100,
        "beats_per_bar": 3,
        "keys": [],
        "notes": _printed_notes(),
    }
    data.update(overrides)
    path = directory / "pulse.printed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _played(notes: list[list], held: float = 0.9) -> NoteSequence:
    """The printed notes as a player gives them: on time, released early."""
    return NoteSequence.of(
        [
            Note(onset=FIRST_BEAT + beat * BEAT, pitch=pitch, duration=length * BEAT * held)
            for pitch, beat, length, _ in notes
        ]
    )


def test_a_faithful_performance_scores_perfectly(tmp_path: Path) -> None:
    piece = load(_write_piece(tmp_path))
    result = score_sequence(piece, _played(_printed_notes()))

    assert result.f1 == pytest.approx(1.0)
    assert result.written_right == result.matched
    assert result.staff_right == result.matched
    assert result.tempo_right and result.meter_right


def test_a_missing_note_and_an_extra_one_are_counted(tmp_path: Path) -> None:
    piece = load(_write_piece(tmp_path))
    notes = _printed_notes()
    played = _played(notes[1:] + [[72, 7.5, 0.5, "R"]])

    result = score_sequence(piece, played)
    assert result.matched == len(notes) - 1
    assert result.detected == len(notes)
    assert result.f1 < 1.0


def test_a_note_written_at_the_wrong_value_is_counted(tmp_path: Path) -> None:
    """Held for a sliver, and a bar later than anything else on its staff, the
    last bass note has nothing to be written as reaching -- so it is written as
    played, and the page is wrong about it."""
    piece = load(_write_piece(tmp_path))
    notes = _printed_notes()
    played = NoteSequence.of(
        [
            Note(
                onset=FIRST_BEAT + beat * BEAT,
                pitch=pitch,
                duration=(0.2 if (pitch == 45 and beat == (BARS - 1) * 3) else length * 0.9) * BEAT,
            )
            for pitch, beat, length, _ in notes
        ]
    )
    result = score_sequence(piece, played)
    assert result.written_right == result.matched - 1


def test_a_relative_video_path_is_found_beside_the_piece_file(tmp_path: Path) -> None:
    piece = load(_write_piece(tmp_path))
    assert piece.video == tmp_path / "recording.mp4"


def test_a_recording_that_is_not_there_is_reported_not_raised(tmp_path: Path) -> None:
    result = score(load(_write_piece(tmp_path)))
    assert result.error and "not found" in result.error


def test_no_directory_means_no_pieces_and_no_noise(tmp_path: Path) -> None:
    assert find_pieces(tmp_path / "nowhere") == []
    _write_piece(tmp_path)
    assert find_pieces(tmp_path) == [tmp_path / "pulse.printed.json"]


def test_regressions_name_what_got_worse() -> None:
    before = PrintedResult(
        name="p", printed=100, detected=100, matched=100, written_right=98, staff_right=100,
        tempo_found=100.0, tempo_expected=100.0, meter_found=3, meter_expected=3,
    )
    same = PrintedResult.from_dict(before.to_dict())
    assert regressions(before, same) == []

    worse = PrintedResult.from_dict({**before.to_dict(), "written_right": 96, "meter_found": 4})
    found = regressions(before, worse)
    assert any("written values" in line for line in found)
    assert any("meter was right" in line for line in found)


def test_a_failed_transcription_is_a_regression() -> None:
    before = PrintedResult(name="p", printed=10, detected=10, matched=10)
    after = PrintedResult(name="p", error="boom")
    assert regressions(before, after) == ["failed: boom"]
