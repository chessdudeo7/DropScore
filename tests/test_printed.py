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
        bar_found=1.8, bar_expected=1.8,
    )
    same = PrintedResult.from_dict(before.to_dict())
    assert regressions(before, same) == []

    # Bars half the length the page prints: the bar lines now fall in the
    # middle of the music's.
    worse = PrintedResult.from_dict({**before.to_dict(), "written_right": 96, "bar_found": 0.9})
    found = regressions(before, worse)
    assert any("written values" in line for line in found)
    assert any("meter was right" in line for line in found)


def test_a_failed_transcription_is_a_regression() -> None:
    before = PrintedResult(name="p", printed=10, detected=10, matched=10)
    after = PrintedResult(name="p", error="boom")
    assert regressions(before, after) == ["failed: boom"]


def test_a_recording_read_as_almost_nothing_is_a_failure_not_a_crash(tmp_path: Path) -> None:
    """Two notes are too few to find a tempo in, and the error that raised
    took the whole eval down with it."""
    piece = load(_write_piece(tmp_path))
    two = NoteSequence.of([Note(onset=1.0, pitch=64, duration=0.2), Note(onset=2.0, pitch=64, duration=0.2)])

    result = score_sequence(piece, two)
    assert result.error and "could not analyse" in result.error
    assert regressions(PrintedResult(name="p", printed=10, detected=10, matched=10), result)


def test_a_beat_map_places_beats_between_its_points(tmp_path: Path) -> None:
    """A performance that slows: beat 3 comes a second after beat 0, beat 6
    a second and a half after that."""
    path = _write_piece(tmp_path, beat_times=[[0, 1.0], [3, 2.0], [6, 3.5]])
    piece = load(path)
    assert piece.seconds(0) == pytest.approx(1.0)
    assert piece.seconds(1.5) == pytest.approx(1.5)
    assert piece.seconds(4.5) == pytest.approx(2.75)
    assert piece.seconds(7.0) == pytest.approx(4.0)  # carries on at the last stretch's pace


def test_a_piece_can_stop_reading_its_recording_early(tmp_path: Path) -> None:
    piece = load(_write_piece(tmp_path, end=145))
    assert piece.end == pytest.approx(145.0)


def test_a_piece_is_read_against_the_section_it_covers(tmp_path: Path) -> None:
    """A page covering the fast half of a recording is checked against that
    half: read against the whole, its tempo came back at half the mark."""
    piece = load(_write_piece(tmp_path, sections=[0.0]))
    assert piece.sections == (0.0,)


# ── scored by order rather than by beat ──────────────────────────────


def _line() -> list[list]:
    """An arpeggio rising and falling, as a figuration does, with the lower
    half of each wave printed on the bass staff."""
    wave = [37, 44, 49, 53, 56, 61, 65, 68, 65, 61, 56, 53, 49, 44]
    return [
        [pitch, index, 0, "R" if pitch >= 56 else "L"]
        for index, pitch in enumerate(wave * 4)
    ]


def _order_piece(directory: Path, notes: list[list], **overrides) -> Path:
    data = {
        "title": "an arpeggio played freely",
        "video": "recording.mp4",
        "window": [0, len(notes) - 1],
        "scored_by": "order",
        "keys": [],
        "notes": notes,
    }
    data.update(overrides)
    path = directory / "figure.printed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _freely(notes: list[list], lead: list[int] = (), tail: list[int] = ()) -> NoteSequence:
    """The notes played with the beat wandering, so no grid describes them."""
    played, when, gap = [], 0.0, 0.14
    for pitch in list(lead) + [n[0] for n in notes] + list(tail):
        played.append(Note(onset=when, pitch=pitch, duration=gap * 0.9))
        when += gap
        gap = 0.10 if gap > 0.13 else 0.17  # push and drag, bar by bar
    return NoteSequence.of(played)


def test_a_piece_scored_by_order_ignores_where_the_notes_fell(tmp_path: Path) -> None:
    """Rubato that no grid describes must not cost the transcription anything
    when every note came back in its turn."""
    notes = _line()
    piece = load(_order_piece(tmp_path, notes))

    result = score_sequence(piece, _freely(notes))

    assert result.by_order
    assert result.matched == len(notes)
    assert result.f1 == pytest.approx(1.0)


def test_order_scoring_finds_the_passage_inside_a_longer_recording(tmp_path: Path) -> None:
    """A page covers eight bars of a recording that runs for a hundred, and
    only the stretch it covers may be counted against its precision."""
    notes = _line()
    piece = load(_order_piece(tmp_path, notes))

    result = score_sequence(piece, _freely(notes, lead=[72, 74, 76], tail=[79] * 40))

    assert result.matched == len(notes)
    assert result.f1 == pytest.approx(1.0), (
        f"the rest of the performance was counted as spurious: {result.detected} detected"
    )


def test_order_scoring_counts_a_dropped_note_and_a_wrong_one(tmp_path: Path) -> None:
    notes = _line()
    piece = load(_order_piece(tmp_path, notes))
    played = [n[0] for n in notes]
    del played[20]
    played[30] = played[30] + 1

    result = score_sequence(piece, _freely([[p, i, 0, "R"] for i, p in enumerate(played)]))

    assert result.matched == len(notes) - 2
    assert result.f1 < 1.0


def test_order_scoring_reports_no_written_values(tmp_path: Path) -> None:
    """There is no grid to judge a value against, and reporting one anyway
    would read as a transcription failure rather than a question not asked."""
    notes = _line()
    piece = load(_order_piece(tmp_path, notes))

    result = score_sequence(piece, _freely(notes))

    assert result.written_right == 0
    assert "written" not in str(result)
    assert "in order" in str(result)
