from __future__ import annotations

import pytest

from dropscore.keyboard import (
    COMMON_RANGES,
    KeyboardLayout,
    is_black,
    is_white,
    white_ordinal,
    white_pitches,
)
from dropscore.notes import MAX_PITCH, MIN_PITCH


def test_black_and_white_classification() -> None:
    # C4 major scale is all white; the sharps between them are all black.
    assert all(is_white(p) for p in (60, 62, 64, 65, 67, 69, 71, 72))
    assert all(is_black(p) for p in (61, 63, 66, 68, 70))
    assert is_white(21) and is_black(22)  # A0, A#0


def test_full_piano_has_52_white_keys() -> None:
    assert len(white_pitches(MIN_PITCH, MAX_PITCH)) == 52


def test_white_ordinal_is_contiguous() -> None:
    """Consecutive white keys must have consecutive ordinals, across octaves."""
    whites = white_pitches(MIN_PITCH, MAX_PITCH)
    ordinals = [white_ordinal(p) for p in whites]
    assert ordinals == list(range(ordinals[0], ordinals[0] + len(whites)))


def test_black_key_shares_ordinal_with_white_below() -> None:
    assert white_ordinal(61) == white_ordinal(60)  # C#4 sits above C4
    assert white_ordinal(70) == white_ordinal(69)  # A#4 sits above A4


def test_layout_rejects_black_endpoints() -> None:
    with pytest.raises(ValueError, match="starts and ends on white keys"):
        KeyboardLayout(first_pitch=22, last_pitch=108)


def test_layout_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="first_pitch must be below"):
        KeyboardLayout(first_pitch=72, last_pitch=60)


def test_white_keys_tile_the_strip_exactly() -> None:
    layout = KeyboardLayout(width=1280.0)
    whites = white_pitches(layout.first_pitch, layout.last_pitch)

    first_left, _ = layout.white_span(whites[0])
    _, last_right = layout.white_span(whites[-1])
    assert first_left == pytest.approx(0.0)
    assert last_right == pytest.approx(1280.0)

    # No gaps and no overlaps between neighbours.
    for lower, upper in zip(whites, whites[1:]):
        assert layout.white_span(lower)[1] == pytest.approx(layout.white_span(upper)[0])


def test_black_key_sits_on_the_boundary_between_its_neighbours() -> None:
    layout = KeyboardLayout(width=1280.0)
    _, c_right = layout.white_span(60)
    d_left, _ = layout.white_span(62)
    assert layout.key_center(61) == pytest.approx(c_right)
    assert layout.key_center(61) == pytest.approx(d_left)


def test_black_keys_are_narrower_than_white() -> None:
    layout = KeyboardLayout(width=1280.0)
    assert layout.key_width(61) < layout.key_width(60)
    left, right = layout.key_span(61)
    assert right - left == pytest.approx(layout.black_width)


@pytest.mark.parametrize("key_range", sorted(COMMON_RANGES))
def test_every_key_center_round_trips(key_range: str) -> None:
    """The core invariant: pitch -> centre -> pitch is the identity.

    Stage 3 fits a layout to a real video; this is the property its fit will be
    judged on, so it had better hold for the layout that draws them.
    """
    first, last = COMMON_RANGES[key_range]
    layout = KeyboardLayout(first_pitch=first, last_pitch=last, width=1280.0)

    for pitch in layout.pitches:
        # Black keys only exist near the far edge of the keybed, so sample there.
        depth = 0.2 if is_black(pitch) else 0.9
        assert layout.pitch_at(layout.key_center(pitch), depth) == pitch


def test_below_the_black_keys_only_white_keys_are_hit() -> None:
    layout = KeyboardLayout(width=1280.0)
    assert layout.pitch_at(layout.key_center(61), y_fraction=0.95) in (60, 62)


def test_pitch_at_returns_none_outside_the_keyboard() -> None:
    layout = KeyboardLayout(x0=100.0, width=500.0)
    assert layout.pitch_at(99.0) is None
    assert layout.pitch_at(601.0) is None
    assert layout.pitch_at(350.0) is not None


def test_offset_layout_shifts_everything() -> None:
    a = KeyboardLayout(width=640.0)
    b = KeyboardLayout(x0=200.0, width=640.0)
    assert b.key_center(60) == pytest.approx(a.key_center(60) + 200.0)


def test_narrow_range_has_wider_keys() -> None:
    full = KeyboardLayout(*COMMON_RANGES["88"], width=1280.0)
    cropped = KeyboardLayout(*COMMON_RANGES["49"], width=1280.0)
    assert cropped.white_width > full.white_width
