from __future__ import annotations

from pathlib import Path

import pytest

from dropscore.notes import Note, NoteSequence, pitch_name


def test_pitch_names_use_middle_c_as_c4() -> None:
    assert pitch_name(60) == "C4"
    assert pitch_name(61) == "C#4"
    assert pitch_name(21) == "A0"
    assert pitch_name(108) == "C8"


@pytest.mark.parametrize("pitch", [20, 109, 0, 127])
def test_rejects_pitches_off_the_keyboard(pitch: int) -> None:
    with pytest.raises(ValueError, match="outside the 88-key range"):
        Note(onset=0.0, pitch=pitch, duration=1.0)


def test_rejects_nonsense_timing() -> None:
    with pytest.raises(ValueError, match="duration must be positive"):
        Note(onset=0.0, pitch=60, duration=0.0)
    with pytest.raises(ValueError, match="onset must not be negative"):
        Note(onset=-0.1, pitch=60, duration=1.0)


def test_offset_is_onset_plus_duration() -> None:
    assert Note(onset=1.5, pitch=60, duration=0.25).offset == pytest.approx(1.75)


def test_sequence_sorts_on_construction() -> None:
    late = Note(onset=2.0, pitch=60, duration=0.5)
    early = Note(onset=0.5, pitch=72, duration=0.5)
    assert list(NoteSequence.of([late, early])) == [early, late]


def test_sequence_sorts_simultaneous_notes_by_pitch() -> None:
    high = Note(onset=1.0, pitch=72, duration=0.5)
    low = Note(onset=1.0, pitch=60, duration=0.5)
    assert list(NoteSequence.of([high, low])) == [low, high]


def test_duration_is_the_last_offset_not_the_last_onset() -> None:
    """A long note starting early can outlast one starting later."""
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=10.0),
            Note(onset=1.0, pitch=64, duration=0.5),
        ]
    )
    assert sequence.duration == pytest.approx(10.0)


def test_empty_sequence_has_zero_duration() -> None:
    assert NoteSequence().duration == 0.0
    with pytest.raises(ValueError, match="empty sequence"):
        NoteSequence().pitch_range


def test_between_catches_notes_already_sounding() -> None:
    held = Note(onset=0.0, pitch=60, duration=8.0)
    later = Note(onset=5.0, pitch=64, duration=0.5)
    sequence = NoteSequence.of([held, later])

    # The window starts after the held note began, but it is still sounding.
    assert held in sequence.between(4.0, 4.5)
    assert later not in sequence.between(4.0, 4.5)
    assert sequence.between(20.0, 21.0) == []


def test_hand_filters() -> None:
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=1.0, hand="R"),
            Note(onset=0.0, pitch=48, duration=1.0, hand="L"),
        ]
    )
    assert [n.pitch for n in sequence.hand("L")] == [48]
    assert [n.pitch for n in sequence.hand("R")] == [60]


def test_round_trips_through_json(tmp_path: Path) -> None:
    original = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=0.5, hand="R", velocity=90),
            Note(onset=0.5, pitch=48, duration=1.0, hand="L", velocity=60),
        ],
        tempo=96.0,
        key="F major",
        source="test",
    )
    path = original.save(tmp_path / "notes.json")
    restored = NoteSequence.load(path)

    assert restored.notes == original.notes
    assert restored.tempo == 96.0
    assert restored.key == "F major"
    assert restored.source == "test"
