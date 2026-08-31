"""Stage 5 scored against stage 2's ground truth.

The headline claim is sub-frame timing accuracy, so the assertions are tighter
than one frame: onsets recovered from tile geometry must beat what rounding to
the nearest frame could achieve.
"""

from __future__ import annotations

import numpy as np
import pytest

from dropscore.calibrate import Calibration
from dropscore.notes import Note, NoteSequence
from dropscore.synth import RenderSpec, SynthRenderer, generate, get_theme
from dropscore.tiles import discover_palette
from dropscore.tracking import TrackingError, estimate_speed, track_to_note, transcribe
from dropscore.video import Frame

SPEC = RenderSpec(width=720, height=360, fps=12.0)


def _renderer(theme: str = "synthesia", sequence: NoteSequence | None = None, bars: int = 2) -> SynthRenderer:
    spec = RenderSpec(
        width=SPEC.width, height=SPEC.height, fps=SPEC.fps, theme=get_theme(theme)
    )
    return SynthRenderer(sequence or generate(seed=3, bars=bars), spec)


def _calibration(renderer: SynthRenderer) -> Calibration:
    return Calibration(
        layout=renderer.layout,
        strike_y=renderer.strike_y,
        keybed_bottom=renderer.spec.height,
        white_width=renderer.layout.white_width,
        confidence=1.0,
    )


def _all_frames(renderer: SynthRenderer) -> list[Frame]:
    return [
        Frame(i, i / renderer.spec.fps, renderer.frame(i), 1.0)
        for i in range(renderer.frame_count)
    ]


def _match(found: NoteSequence, expected: list[Note], tolerance: float = 0.05):
    """Greedily pair detected notes to true ones by pitch and onset."""
    remaining = list(found)
    pairs, missed = [], []
    for note in expected:
        candidates = [
            f for f in remaining
            if f.pitch == note.pitch and abs(f.onset - note.onset) <= tolerance
        ]
        if candidates:
            best = min(candidates, key=lambda f: abs(f.onset - note.onset))
            remaining.remove(best)
            pairs.append((note, best))
        else:
            missed.append(note)
    return pairs, missed, remaining


# ── speed ────────────────────────────────────────────────────────────


def test_measures_the_scroll_speed() -> None:
    renderer = _renderer()
    frames = _all_frames(renderer)[10:40]
    estimate = estimate_speed(frames, _calibration(renderer))

    assert estimate.value == pytest.approx(renderer.speed, rel=0.02)
    assert estimate.confidence > 0.5


def test_speed_is_measured_from_tiles_not_the_static_background() -> None:
    """A frame is mostly background; correlating on it would report zero."""
    renderer = _renderer()
    frames = _all_frames(renderer)[10:40]
    assert estimate_speed(frames, _calibration(renderer)).value > 1.0


@pytest.mark.parametrize("theme", ["classic", "synthesia", "paper", "aurora"])
def test_speed_survives_every_look(theme: str) -> None:
    renderer = _renderer(theme)
    frames = _all_frames(renderer)[10:40]
    assert estimate_speed(frames, _calibration(renderer)).value == pytest.approx(
        renderer.speed, rel=0.05
    )


def test_speed_needs_two_frames() -> None:
    renderer = _renderer()
    with pytest.raises(TrackingError, match="at least two"):
        estimate_speed(_all_frames(renderer)[:1], _calibration(renderer))


def test_still_video_has_no_measurable_speed() -> None:
    renderer = _renderer()
    still = renderer.frame(20)
    frames = [Frame(i, i / SPEC.fps, still.copy(), 1.0) for i in range(10)]
    with pytest.raises(TrackingError, match="consistent scroll speed"):
        estimate_speed(frames, _calibration(renderer))


# ── timing ───────────────────────────────────────────────────────────


def test_onsets_are_more_accurate_than_a_frame() -> None:
    renderer = _renderer()
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)
    palette = discover_palette(frames[::5], calibration)

    found = transcribe(frames, calibration, palette, renderer.speed)
    pairs, missed, _ = _match(found, renderer.notes)

    assert not missed, f"{len(missed)} notes missed"
    errors = np.array([abs(f.onset - t.onset) for t, f in pairs])
    frame_time = 1.0 / renderer.spec.fps
    assert np.median(errors) < frame_time / 3


def test_durations_are_recovered() -> None:
    renderer = _renderer()
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)
    palette = discover_palette(frames[::5], calibration)

    found = transcribe(frames, calibration, palette, renderer.speed)
    pairs, _, _ = _match(found, renderer.notes)

    errors = np.array([abs(f.duration - t.duration) for t, f in pairs])
    assert np.median(errors) < 0.05


def test_finds_every_note_and_invents_none() -> None:
    renderer = _renderer()
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)
    palette = discover_palette(frames[::5], calibration)

    found = transcribe(frames, calibration, palette, renderer.speed)
    _, missed, spurious = _match(found, renderer.notes)

    assert not missed
    assert not spurious


def test_repeated_notes_stay_separate() -> None:
    """The classic under-count: two tiles on one key read as one long note."""
    sequence = NoteSequence.of(
        [Note(onset=i * 0.35, pitch=60, duration=0.28) for i in range(6)]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)
    palette = discover_palette(frames[::5], calibration)

    found = transcribe(frames, calibration, palette, renderer.speed)
    assert len([n for n in found if n.pitch == 60]) == 6


def test_a_note_taller_than_the_screen_is_timed_correctly() -> None:
    """Reading duration off tile height would fail here; two edges do not.

    The tile is never fully visible in any single frame, but its bottom edge
    crosses the strike line early and its top edge crosses later, and both are
    measured from frames where that edge is unclipped.
    """
    lead = get_theme("synthesia").lead_time
    duration = lead * 1.6
    sequence = NoteSequence.of([Note(onset=0.0, pitch=60, duration=duration)])
    renderer = _renderer(sequence=sequence)
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)
    palette = discover_palette(frames[::5], calibration)

    found = transcribe(frames, calibration, palette, renderer.speed)
    assert len(found) == 1
    assert found[0].duration == pytest.approx(duration, abs=0.08)


def test_chords_keep_every_voice() -> None:
    sequence = NoteSequence.of(
        [Note(onset=0.5, pitch=p, duration=1.0) for p in (60, 64, 67)]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)
    palette = discover_palette(frames[::5], calibration)

    found = transcribe(frames, calibration, palette, renderer.speed)
    assert sorted(n.pitch for n in found) == [60, 64, 67]
    assert len({round(n.onset, 2) for n in found}) == 1


def test_short_tracks_are_discarded() -> None:
    from dropscore.tracking import TileTrack  # noqa: PLC0415

    renderer = _renderer()
    track = TileTrack(pitch=60, track=0)
    assert track_to_note(track, renderer.speed, _calibration(renderer)) is None


def test_transcribe_rejects_a_nonpositive_speed() -> None:
    renderer = _renderer()
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)[:5]
    palette = discover_palette(frames, calibration)
    with pytest.raises(TrackingError, match="must be positive"):
        transcribe(frames, calibration, palette, 0.0)


def test_two_tile_colours_produce_two_hands() -> None:
    """Which colour is which hand is arbitrary here — palettes are ordered by
    pixel count, not by register. Stage 7 decides; stage 5 only keeps them apart.
    """
    renderer = _renderer()
    calibration = _calibration(renderer)
    frames = _all_frames(renderer)
    palette = discover_palette(frames[::5], calibration)

    found = transcribe(frames, calibration, palette, renderer.speed)
    assert {n.hand for n in found} == {"L", "R"}
