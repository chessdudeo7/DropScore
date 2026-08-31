"""Tests for the synthetic renderer.

The important ones are the geometry round-trips: a tile's bottom edge must cross
the strike line exactly at the note's onset, and its height must equal
``duration * speed``. Stage 5 recovers timing by inverting precisely those two
relations, so if they do not hold here the ground truth is a lie.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dropscore.keyboard import COMMON_RANGES, is_black
from dropscore.notes import Note, NoteSequence
from dropscore.synth import RenderSpec, SynthRenderer, generate, get_theme, render
from dropscore.synth.themes import THEMES
from dropscore.video import VideoReader

SMALL = RenderSpec(width=480, height=270, fps=20.0)


def _one_note(pitch: int = 60, onset: float = 0.0, duration: float = 1.0) -> NoteSequence:
    return NoteSequence.of([Note(onset=onset, pitch=pitch, duration=duration)])


# ── generated music ──────────────────────────────────────────────────


def test_generate_is_deterministic() -> None:
    assert generate(seed=7).notes == generate(seed=7).notes


def test_different_seeds_differ() -> None:
    assert generate(seed=1).notes != generate(seed=2).notes


def test_generated_music_has_both_hands_and_a_key() -> None:
    sequence = generate(seed=3)
    assert sequence.hand("L") and sequence.hand("R")
    assert sequence.key in {"C major", "G major", "F major", "D major",
                            "A minor", "E minor", "D minor"}
    assert sequence.tempo == 96.0


def test_generated_music_includes_repeated_notes() -> None:
    """Repeated notes are the case that merges into one blob; it must appear."""
    sequence = generate(seed=0, bars=6)
    by_pitch: dict[int, list[Note]] = {}
    for note in sequence:
        by_pitch.setdefault(note.pitch, []).append(note)

    repeats = [
        (a, b)
        for notes in by_pitch.values()
        for a, b in zip(notes, notes[1:])
        if b.onset - a.offset < 0.2
    ]
    assert repeats, "expected consecutive notes on the same key"


def test_generated_music_includes_simultaneous_notes() -> None:
    sequence = generate(seed=0, bars=8)
    onsets = [n.onset for n in sequence]
    assert len(onsets) > len(set(round(o, 6) for o in onsets))


# ── geometry ─────────────────────────────────────────────────────────


def test_speed_follows_lead_time() -> None:
    renderer = SynthRenderer(_one_note(), SMALL)
    assert renderer.speed == pytest.approx(
        renderer.strike_y / renderer.theme.lead_time
    )


def test_tile_bottom_crosses_the_strike_line_at_onset() -> None:
    renderer = SynthRenderer(_one_note(onset=2.0), SMALL)
    note = renderer.notes[0]  # shifted by lead_in

    rect = renderer._tile_rect(note, note.onset)
    assert rect is not None
    assert rect[3] == pytest.approx(renderer.strike_y, abs=1)


def test_tile_height_encodes_duration() -> None:
    renderer = SynthRenderer(_one_note(onset=2.0, duration=0.8), SMALL)
    note = renderer.notes[0]

    # Sampled early enough that the tile is fully on screen and unclipped.
    t = note.onset - 0.4
    _, top, _, bottom = renderer._tile_rect(note, t)
    assert bottom - top == pytest.approx(note.duration * renderer.speed, abs=2)


def test_tile_is_clipped_at_the_strike_line() -> None:
    renderer = SynthRenderer(_one_note(onset=2.0, duration=1.0), SMALL)
    note = renderer.notes[0]

    # Halfway through the note, the bottom half has passed the strike line.
    rect = renderer._tile_rect(note, note.onset + 0.5)
    assert rect is not None
    assert rect[3] <= renderer.strike_y


def test_tile_disappears_once_fully_played() -> None:
    renderer = SynthRenderer(_one_note(onset=2.0, duration=0.5), SMALL)
    note = renderer.notes[0]
    assert renderer._tile_rect(note, note.offset + 0.5) is None


def test_tile_not_yet_visible_before_its_lead_time() -> None:
    renderer = SynthRenderer(_one_note(onset=6.0), SMALL)
    note = renderer.notes[0]
    assert renderer._tile_rect(note, note.onset - renderer.theme.lead_time - 0.5) is None


def test_tile_sits_over_its_own_key() -> None:
    renderer = SynthRenderer(_one_note(pitch=60, onset=2.0), SMALL)
    note = renderer.notes[0]
    left, _, right, _ = renderer._tile_rect(note, note.onset - 0.5)
    center = renderer.layout.key_center(60)
    assert left <= center <= right


def test_black_key_tiles_are_narrower() -> None:
    white = SynthRenderer(_one_note(pitch=60, onset=2.0), SMALL)
    black = SynthRenderer(_one_note(pitch=61, onset=2.0), SMALL)

    wl, _, wr, _ = white._tile_rect(white.notes[0], white.notes[0].onset - 0.5)
    bl, _, br, _ = black._tile_rect(black.notes[0], black.notes[0].onset - 0.5)
    assert br - bl < wr - wl


# ── rendered pixels ──────────────────────────────────────────────────


def test_frame_has_the_expected_shape() -> None:
    renderer = SynthRenderer(generate(seed=1, bars=2), SMALL)
    frame = renderer.frame(0)
    assert frame.shape == (SMALL.height, SMALL.width, 3)
    assert frame.dtype == np.uint8


def test_no_key_is_lit_on_the_first_frame() -> None:
    """lead_in exists so nothing is already being played at t=0."""
    renderer = SynthRenderer(generate(seed=1, bars=2), SMALL)
    unlit = SynthRenderer(NoteSequence(), SMALL)
    assert np.array_equal(
        renderer.frame(0)[renderer.strike_y :],
        unlit.frame(0)[unlit.strike_y :],
    )


def test_struck_key_lights_up() -> None:
    renderer = SynthRenderer(_one_note(pitch=60, onset=1.0, duration=1.0), SMALL)
    note = renderer.notes[0]
    column = int(renderer.layout.key_center(60))
    row = renderer.strike_y + renderer.keybed_height - 2

    before = renderer.frame(int((note.onset - 0.3) * SMALL.fps))[row, column]
    during = renderer.frame(int((note.onset + 0.3) * SMALL.fps))[row, column]
    assert not np.array_equal(before, during)


def test_tile_is_drawn_above_the_keybed() -> None:
    renderer = SynthRenderer(_one_note(pitch=60, onset=2.0, duration=1.0), SMALL)
    note = renderer.notes[0]
    t = note.onset - 0.5
    frame = renderer.frame(int(t * SMALL.fps))

    left, top, right, bottom = renderer._tile_rect(note, t)
    patch = frame[top + 2 : bottom - 2, left + 2 : right - 2].astype(int)
    background = np.array(renderer.theme.background[::-1], dtype=int)  # RGB -> BGR
    assert np.abs(patch - background).max() > 30


@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_every_theme_renders(theme_name: str) -> None:
    spec = RenderSpec(
        width=SMALL.width, height=SMALL.height, fps=SMALL.fps, theme=get_theme(theme_name)
    )
    renderer = SynthRenderer(generate(seed=2, bars=2), spec)
    frame = renderer.frame(renderer.frame_count // 2)
    assert frame.shape == (spec.height, spec.width, 3)
    assert frame.std() > 0  # not a flat fill


@pytest.mark.parametrize("key_range", sorted(COMMON_RANGES))
def test_every_key_range_renders(key_range: str) -> None:
    spec = RenderSpec(
        width=SMALL.width, height=SMALL.height, fps=SMALL.fps, key_range=key_range
    )
    renderer = SynthRenderer(generate(seed=4, bars=2), spec)
    assert renderer.frame(renderer.frame_count // 2).shape[0] == spec.height


def test_notes_outside_the_shown_range_are_dropped() -> None:
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=24, duration=1.0),  # below a 61-key keyboard
            Note(onset=0.0, pitch=60, duration=1.0),
        ]
    )
    spec = RenderSpec(width=SMALL.width, height=SMALL.height, key_range="61")
    renderer = SynthRenderer(sequence, spec)
    assert [n.pitch for n in renderer.notes] == [60]


# ── truth and output ─────────────────────────────────────────────────


def test_truth_records_the_geometry_that_drew_the_clip() -> None:
    renderer = SynthRenderer(generate(seed=5, bars=2), SMALL)
    truth = renderer.truth()

    geometry = truth["geometry"]
    assert geometry["strike_y"] == renderer.strike_y
    assert geometry["speed_px_per_s"] == pytest.approx(renderer.speed)
    assert geometry["white_key_width"] == pytest.approx(renderer.layout.white_width)
    assert geometry["first_pitch"] == renderer.layout.first_pitch
    assert truth["video"]["fps"] == SMALL.fps


def test_truth_notes_are_the_shifted_ones_actually_rendered() -> None:
    """Truth must describe the video, not the sequence handed in."""
    renderer = SynthRenderer(_one_note(onset=0.0), SMALL)
    restored = NoteSequence.from_dict(renderer.truth()["sequence"])
    assert restored[0].onset == pytest.approx(SMALL.lead_in)


def test_render_writes_a_video_stage_one_can_open(tmp_path: Path) -> None:
    sequence = generate(seed=6, bars=2)
    video_path, truth_path = render(sequence, tmp_path / "clip.mp4", SMALL)

    assert video_path.exists() and video_path.stat().st_size > 0
    assert truth_path is not None and truth_path.exists()

    renderer = SynthRenderer(sequence, SMALL)
    with VideoReader(video_path) as reader:
        assert reader.info.source_width == SMALL.width
        assert reader.info.source_height == SMALL.height
        assert reader.info.fps == pytest.approx(SMALL.fps, abs=0.5)
        assert reader.info.frame_count == pytest.approx(renderer.frame_count, abs=2)


def test_render_can_skip_the_truth_file(tmp_path: Path) -> None:
    _, truth_path = render(_one_note(), tmp_path / "clip.mp4", SMALL, write_truth=False)
    assert truth_path is None


def test_rendering_is_reproducible() -> None:
    a = SynthRenderer(generate(seed=8, bars=2), SMALL).frame(30)
    b = SynthRenderer(generate(seed=8, bars=2), SMALL).frame(30)
    assert np.array_equal(a, b)


def test_black_and_white_keys_are_distinguishable_in_the_keybed() -> None:
    """Stage 3 anchors the key grid on the black-key pattern, so it must be
    visible in the rendered keybed."""
    renderer = SynthRenderer(NoteSequence(), SMALL)
    bed = renderer._keybed
    row = int(renderer.keybed_height * 0.2)  # inside the black keys

    layout = renderer.layout
    blacks = [p for p in layout.pitches if is_black(p)]
    whites = [p for p in layout.pitches if not is_black(p)]

    black_mean = np.mean([bed[row, int(layout.key_center(p))].mean() for p in blacks])
    white_mean = np.mean([bed[row, int(layout.key_center(p))].mean() for p in whites])
    assert abs(white_mean - black_mean) > 50
