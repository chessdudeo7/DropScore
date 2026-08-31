"""Stage 4 scored against stage 2's ground truth.

The central test asks the only question that matters: at a given instant, does the
detector report exactly the set of pitches the renderer drew?
"""

from __future__ import annotations

import numpy as np
import pytest

from dropscore.calibrate import Calibration
from dropscore.config import DEFAULT
from dropscore.keyboard import KeyboardLayout
from dropscore.notes import Note, NoteSequence
from dropscore.synth import RenderSpec, SynthRenderer, generate, get_theme
from dropscore.synth.themes import THEMES
from dropscore.tiles import TileError, detect_in_frame, discover_palette
from dropscore.video import Frame

SPEC = RenderSpec(width=960, height=540, fps=10.0)


def _renderer(theme: str = "classic", sequence: NoteSequence | None = None) -> SynthRenderer:
    spec = RenderSpec(
        width=SPEC.width, height=SPEC.height, fps=SPEC.fps, theme=get_theme(theme)
    )
    return SynthRenderer(sequence or generate(seed=1, bars=4), spec)


def _truth_calibration(renderer: SynthRenderer) -> Calibration:
    """Use the exact geometry, so stage 4 is tested without stage 3's error."""
    return Calibration(
        layout=renderer.layout,
        strike_y=renderer.strike_y,
        keybed_bottom=renderer.spec.height,
        white_width=renderer.layout.white_width,
        confidence=1.0,
    )


def _frames(renderer: SynthRenderer, count: int = 20) -> list[Frame]:
    indices = np.linspace(0, renderer.frame_count - 1, num=count, dtype=int)
    return [
        Frame(int(i), int(i) / renderer.spec.fps, renderer.frame(int(i)), 1.0)
        for i in indices
    ]


def _frame_at(renderer: SynthRenderer, t: float) -> Frame:
    index = int(round(t * renderer.spec.fps))
    return Frame(index, index / renderer.spec.fps, renderer.frame(index), 1.0)


def _expected_pitches(renderer: SynthRenderer, t: float, min_height: int = 0) -> set[int]:
    """Pitches whose tile is on screen at time t, optionally only tall ones.

    A tile entering the frame is a sliver for a frame or two, right at the
    detector's minimum-height cutoff, so assertions bracket rather than demand
    exactness: every clearly visible tile must be found, and nothing may be
    reported that was not drawn at all.
    """
    found = set()
    for note in renderer.notes:
        rect = renderer._tile_rect(note, t)
        if rect is not None and rect[3] - rect[1] >= min_height:
            found.add(note.pitch)
    return found


# ── palette discovery ────────────────────────────────────────────────


def test_finds_two_colours_for_a_two_hand_video() -> None:
    renderer = _renderer("synthesia")  # clearly distinct green and blue
    palette = discover_palette(_frames(renderer), _truth_calibration(renderer))
    assert palette.track_count == 2


def test_hands_of_near_identical_hue_collapse_to_one_track() -> None:
    """classic uses two shades of the same gold, which is a real case.

    Chroma is the only signal kept (lightness is discarded to hold gradients
    together), so these merge. Hand assignment falls back to pitch in stage 7.
    """
    renderer = _renderer("classic")
    palette = discover_palette(_frames(renderer), _truth_calibration(renderer))
    assert palette.track_count == 1


def test_gradient_tiles_stay_one_colour_per_hand() -> None:
    """Lightness varies down an aurora tile; hue does not."""
    renderer = _renderer("aurora")
    palette = discover_palette(_frames(renderer), _truth_calibration(renderer))
    assert palette.track_count == 2


def test_empty_video_has_no_palette() -> None:
    renderer = SynthRenderer(NoteSequence(), SPEC)
    frames = [Frame(i, i / SPEC.fps, renderer.frame(0), 1.0) for i in range(4)]
    with pytest.raises(TileError, match="nothing differs"):
        discover_palette(frames, _truth_calibration(renderer))


# ── detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_detects_the_right_pitches(theme: str) -> None:
    renderer = _renderer(theme)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    for t in (3.0, 5.5, 8.0):
        frame = _frame_at(renderer, t)
        found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration)}
        assert _expected_pitches(renderer, frame.time, min_height=8) <= found, f"{theme} t={t}"
        assert found <= _expected_pitches(renderer, frame.time), f"{theme} t={t}"


def test_bloom_does_not_widen_a_tile_onto_its_neighbour() -> None:
    """A lone note under heavy glow must not read as a cluster."""
    sequence = NoteSequence.of([Note(onset=0.0, pitch=60, duration=1.0)])
    renderer = _renderer("neon", sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    assert {tile.pitch for tile in detect_in_frame(frame, palette, calibration)} == {60}


def test_adjacent_keys_are_split_not_merged() -> None:
    sequence = NoteSequence.of(
        [Note(onset=0.0, pitch=p, duration=1.0) for p in (60, 62, 64)]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    assert {tile.pitch for tile in detect_in_frame(frame, palette, calibration)} == {60, 62, 64}


def test_two_white_neighbours_do_not_invent_the_black_key_between_them() -> None:
    sequence = NoteSequence.of(
        [Note(onset=0.0, pitch=p, duration=1.0) for p in (60, 62)]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration)}
    assert 61 not in found


def test_a_black_key_is_still_found_between_two_silent_whites() -> None:
    sequence = NoteSequence.of([Note(onset=0.0, pitch=61, duration=1.0)])
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    assert {tile.pitch for tile in detect_in_frame(frame, palette, calibration)} == {61}


def test_repeated_notes_are_split_vertically() -> None:
    """Two tiles on one key, close together, must not read as one long note."""
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=0.45),
            Note(onset=0.5, pitch=60, duration=0.45),
        ]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    # Early enough that both tiles are fully on screen at once.
    frame = _frame_at(renderer, renderer.notes[0].onset - 0.6)
    tiles = [t for t in detect_in_frame(frame, palette, calibration) if t.pitch == 60]
    assert len(tiles) == 2


def test_tile_geometry_matches_what_was_drawn() -> None:
    sequence = NoteSequence.of([Note(onset=0.0, pitch=60, duration=0.8)])
    renderer = _renderer("minimal", sequence)  # no glow, no rounding
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    note = renderer.notes[0]
    t = note.onset - 0.4
    frame = _frame_at(renderer, t)
    tile = detect_in_frame(frame, palette, calibration)[0]

    _, top, _, bottom = renderer._tile_rect(note, frame.time)
    assert tile.top == pytest.approx(top, abs=2)
    assert tile.bottom == pytest.approx(bottom, abs=2)


def test_tracks_separate_the_two_hands() -> None:
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=72, duration=1.0, hand="R"),
            Note(onset=0.0, pitch=48, duration=1.0, hand="L"),
        ]
    )
    renderer = _renderer("synthesia", sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    by_pitch = {t.pitch: t.track for t in detect_in_frame(frame, palette, calibration)}
    assert by_pitch[72] != by_pitch[48]


def test_nothing_is_detected_below_the_strike_line() -> None:
    renderer = _renderer()
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, 5.0)
    tiles = detect_in_frame(frame, palette, calibration)
    assert tiles
    assert max(t.bottom for t in tiles) <= calibration.strike_y


def test_detection_uses_the_fitted_layout_not_the_true_one() -> None:
    """A layout shifted by a fraction of a key must still resolve correctly."""
    renderer = _renderer()
    true_layout = renderer.layout
    shifted = KeyboardLayout(
        first_pitch=true_layout.first_pitch,
        last_pitch=true_layout.last_pitch,
        x0=true_layout.x0 + 0.3,
        width=true_layout.width,
    )
    calibration = Calibration(
        layout=shifted,
        strike_y=renderer.strike_y,
        keybed_bottom=renderer.spec.height,
        white_width=shifted.white_width,
        confidence=1.0,
    )
    palette = discover_palette(_frames(renderer), calibration, DEFAULT)

    frame = _frame_at(renderer, 5.0)
    found = {t.pitch for t in detect_in_frame(frame, palette, calibration)}
    assert _expected_pitches(renderer, frame.time, min_height=8) <= found
    assert found <= _expected_pitches(renderer, frame.time)
