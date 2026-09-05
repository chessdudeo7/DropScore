"""Stage 5 scored against stage 2's ground truth.

The headline claim is sub-frame timing accuracy, so the assertions are tighter
than one frame: onsets recovered from tile geometry must beat what rounding to
the nearest frame could achieve.
"""

from __future__ import annotations

import numpy as np
import pytest

from dropscore.calibrate import Calibration
from dropscore.config import DEFAULT
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


# ── association is one tile to one track ─────────────────────────────


def _tile(pitch: int, bottom: float, time: float, height: float = 20.0):
    from dropscore.tiles import Tile  # noqa: PLC0415

    return Tile(
        frame=int(time * 100),
        time=time,
        pitch=pitch,
        top=bottom - height,
        bottom=bottom,
        track=0,
        left=0.0,
        right=10.0,
    )


def _plain_calibration(strike_y: int = 400):
    from dropscore.keyboard import KeyboardLayout  # noqa: PLC0415

    layout = KeyboardLayout(width=960.0)
    return Calibration(
        layout=layout,
        strike_y=strike_y,
        keybed_bottom=strike_y + 100,
        white_width=layout.white_width,
        confidence=1.0,
    )


def test_two_tiles_on_one_key_do_not_share_a_track() -> None:
    """Two strikes of a repeated note, both on screen at once.

    Matching tile by tile let the second attach to the same track as the first,
    whose observations then interleaved two notes and averaged into one wrong
    one. Repeated notes are the pipeline's weakest case.
    """
    from dropscore.tracking import build_tracks  # noqa: PLC0415

    calibration = _plain_calibration()
    speed = 100.0

    # Frame 1: two tiles on the same key, 60px apart.
    # Frame 2: both have fallen 10px, so either could match either track.
    detections = [
        (Frame(0, 0.0, np.zeros((1, 1, 3), np.uint8), 1.0), [_tile(60, 100, 0.0), _tile(60, 160, 0.0)]),
        (Frame(1, 0.1, np.zeros((1, 1, 3), np.uint8), 1.0), [_tile(60, 110, 0.1), _tile(60, 170, 0.1)]),
    ]

    tracks = build_tracks(detections, speed, calibration)

    assert len(tracks) == 2, f"expected two tracks, got {len(tracks)}"
    assert all(track.length == 2 for track in tracks), "a track absorbed both tiles"

    # Each track followed one tile, so its bottoms advance by the scroll amount.
    for track in tracks:
        assert track.bottoms[1] - track.bottoms[0] == pytest.approx(10.0)


def test_the_closer_pairing_wins_regardless_of_tile_order() -> None:
    """Assignment is by agreement, not by whichever tile was listed first."""
    from dropscore.tracking import build_tracks  # noqa: PLC0415

    calibration = _plain_calibration()
    first = [_tile(60, 100, 0.0), _tile(60, 200, 0.0)]

    # Same second frame, tiles listed in both orders.
    later = [_tile(60, 210, 0.1), _tile(60, 110, 0.1)]

    forward = build_tracks(
        [
            (Frame(0, 0.0, np.zeros((1, 1, 3), np.uint8), 1.0), first),
            (Frame(1, 0.1, np.zeros((1, 1, 3), np.uint8), 1.0), later),
        ],
        100.0,
        calibration,
    )
    reversed_order = build_tracks(
        [
            (Frame(0, 0.0, np.zeros((1, 1, 3), np.uint8), 1.0), list(reversed(first))),
            (Frame(1, 0.1, np.zeros((1, 1, 3), np.uint8), 1.0), list(reversed(later))),
        ],
        100.0,
        calibration,
    )

    assert len(forward) == len(reversed_order) == 2
    assert sorted(t.bottoms for t in forward) == sorted(t.bottoms for t in reversed_order)


def test_a_single_tile_still_follows_one_track() -> None:
    from dropscore.tracking import build_tracks  # noqa: PLC0415

    calibration = _plain_calibration()
    detections = [
        (Frame(i, i * 0.1, np.zeros((1, 1, 3), np.uint8), 1.0), [_tile(60, 100 + i * 10, i * 0.1)])
        for i in range(5)
    ]

    tracks = build_tracks(detections, 100.0, calibration)
    assert len(tracks) == 1
    assert tracks[0].length == 5


def test_tiles_on_different_keys_never_compete() -> None:
    from dropscore.tracking import build_tracks  # noqa: PLC0415

    calibration = _plain_calibration()
    detections = [
        (
            Frame(i, i * 0.1, np.zeros((1, 1, 3), np.uint8), 1.0),
            [_tile(60, 100 + i * 10, i * 0.1), _tile(62, 100 + i * 10, i * 0.1)],
        )
        for i in range(4)
    ]

    tracks = build_tracks(detections, 100.0, calibration)
    assert len(tracks) == 2
    assert {t.pitch for t in tracks} == {60, 62}
    assert all(t.length == 4 for t in tracks)


# ── palette index is not a hand ──────────────────────────────────────


def _track(track_index: int, pitch: int):
    from dropscore.tracking import TileTrack  # noqa: PLC0415

    made = TileTrack(pitch=pitch, track=track_index)
    made.observe(_tile(pitch, 100.0, 0.0))
    return made


def test_the_lower_colour_becomes_the_left_hand() -> None:
    """Palettes are numbered by pixel count, which says nothing about register."""
    from dropscore.tracking import hands_by_register  # noqa: PLC0415

    # Track 0 is the bass here; the old index rule would have called it right.
    hands = hands_by_register([_track(0, 45), _track(0, 48), _track(1, 79), _track(1, 84)])
    assert hands == {0: "L", 1: "R"}


def test_a_third_colour_is_not_silently_swallowed() -> None:
    """Anything past index 1 used to collapse into the right hand regardless."""
    from dropscore.tracking import hands_by_register  # noqa: PLC0415

    hands = hands_by_register([_track(0, 80), _track(1, 40), _track(2, 84)])
    assert hands[1] == "L", "the lowest colour should be the left hand"
    assert hands[0] == "R" and hands[2] == "R"


def test_a_single_colour_is_left_for_stage_seven() -> None:
    """With one colour there is nothing to tell the hands apart."""
    from dropscore.tracking import hands_by_register  # noqa: PLC0415

    hands = hands_by_register([_track(0, 40), _track(0, 80)])
    assert set(hands.values()) == {"R"}


def test_hands_ignore_tracks_with_no_observations() -> None:
    from dropscore.tracking import TileTrack, hands_by_register  # noqa: PLC0415

    hands = hands_by_register([_track(0, 45), _track(1, 80), TileTrack(pitch=60, track=2)])
    assert set(hands) == {0, 1}


# ── input validation and short notes ─────────────────────────────────


def test_speed_rejects_unevenly_spaced_frames() -> None:
    """sample() jumps across the video; phase correlation needs a steady step."""
    renderer = _renderer()
    calibration = _calibration(renderer)
    scattered = [
        Frame(i, i / renderer.spec.fps, renderer.frame(i), 1.0) for i in (0, 1, 2, 40, 80)
    ]

    with pytest.raises(TrackingError, match="evenly spaced"):
        estimate_speed(scattered, calibration)


def test_speed_accepts_a_uniform_step() -> None:
    """Every second frame is fine: the gap is constant, so dy/dt still holds."""
    renderer = _renderer()
    calibration = _calibration(renderer)
    strided = [
        Frame(i, i / renderer.spec.fps, renderer.frame(i), 1.0) for i in range(0, 40, 2)
    ]

    estimate = estimate_speed(strided, calibration)
    assert estimate.value == pytest.approx(renderer.speed, rel=0.1)


def test_a_very_short_note_is_kept_not_dropped() -> None:
    """Losing a note costs recall permanently; a long one is fixed by quantizing."""
    from dropscore.tracking import TileTrack, track_to_note  # noqa: PLC0415

    calibration = _plain_calibration()
    track = TileTrack(pitch=60, track=0)
    # A tile only a couple of pixels tall: a real, very short note.
    track.observe(_tile(60, 300.0, 0.0, height=2.0))
    track.observe(_tile(60, 310.0, 0.1, height=2.0))

    note = track_to_note(track, 100.0, calibration)
    assert note is not None, "a short note was discarded"
    assert note.duration >= DEFAULT.tracking.min_duration


def test_contradictory_edges_are_dropped() -> None:
    """If the top crosses before the bottom, the two estimates disagree."""
    from dropscore.tracking import TileTrack, track_to_note  # noqa: PLC0415

    calibration = _plain_calibration()
    track = TileTrack(pitch=60, track=0)
    track.times = [0.0, 0.1]
    track.bottoms = [300.0, 310.0]
    track.tops = [340.0, 350.0]  # top below the bottom: inconsistent

    assert track_to_note(track, 100.0, calibration) is None


# ── edges that have stopped moving ───────────────────────────────────


def _falling_then_held(strike: int, speed: float, fps: float, onset: float, held: float):
    """A tile that reaches the strike line and then sits there.

    Its bottom plateaus a few pixels short of the crop edge, exactly as the
    detector reports one on a real clip.
    """
    from dropscore.tracking import TileTrack

    track = TileTrack(pitch=60, track=0)
    plateau = strike - 4.0
    time = 0.0
    while time < onset + held:
        bottom = min(plateau, strike + (time - onset) * speed)
        track.times.append(time)
        track.bottoms.append(bottom)
        track.tops.append(max(0.0, bottom - held * speed))
        time += 1.0 / fps
    return track


def test_onset_ignores_an_edge_that_has_stopped_moving() -> None:
    """A held note's onset is where its edge arrived, not the middle of it.

    Once the bottom reaches the strike line it stops advancing, and every
    frame it sits there reports a crossing at that frame. Left in, five
    seconds of held note contribute five seconds of contradictory estimates
    and the median lands mid-note: measured 3.00s for a note starting at 1.00.
    """
    strike, speed, fps = 536, 208.8, 30.0
    track = _falling_then_held(strike, speed, fps, onset=1.0, held=5.0)
    calibration = _plain_calibration(strike_y=strike)

    note = track_to_note(track, speed, calibration, DEFAULT)

    assert note is not None
    assert note.onset == pytest.approx(1.0, abs=0.05), (
        f"onset {note.onset:.2f}s; the plateau dragged the median into the note"
    )


def test_plateau_is_found_by_motion_not_by_position() -> None:
    """The blob stops short of the crop edge, so a fixed margin misses it.

    Measured on a real clip: a strike line at 536 with the detected bottom
    plateauing at 532, comfortably inside an edge_margin of 2.
    """
    strike, speed, fps = 536, 208.8, 30.0
    track = _falling_then_held(strike, speed, fps, onset=1.0, held=4.0)

    assert max(track.bottoms) < strike - DEFAULT.tracking.edge_margin, (
        "fixture must reproduce a plateau that the position test lets through"
    )

    note = track_to_note(track, speed, _plain_calibration(strike_y=strike), DEFAULT)
    assert note is not None and note.onset == pytest.approx(1.0, abs=0.05)


def test_a_tile_taller_than_the_screen_keeps_its_onset() -> None:
    """Neither edge visible for most of the note, but the arrival was seen."""
    strike, speed, fps = 536, 208.8, 30.0
    track = _falling_then_held(strike, speed, fps, onset=0.8, held=12.0)

    note = track_to_note(track, speed, _plain_calibration(strike_y=strike), DEFAULT)

    assert note is not None
    assert note.onset == pytest.approx(0.8, abs=0.05)


# ── which statistic summarises the frame pairs ───────────────────────


def test_robust_mean_reads_quantised_displacements_correctly() -> None:
    """Whole-pixel rendering makes the pairs bimodal; the mean is the truth.

    A tile drawn at integer positions moves 5px or 4px between frames where
    the true rate is 4.708, and the mixture averages to it. Taking the median
    snaps to whichever cluster is more populous — measured 4.95 against 4.708,
    a 5% error that put every onset 51ms early.
    """
    from dropscore.tracking import _robust_mean

    values = np.array([5.0] * 70 + [4.0] * 30)  # mean 4.7, median 5.0

    assert _robust_mean(values) == pytest.approx(4.7, abs=0.01)
    assert np.median(values) == 5.0, "fixture must actually be bimodal"


def test_robust_mean_keeps_the_smaller_cluster() -> None:
    """Regression: an outlier bound scaled to the spread deletes it.

    With one cluster dominating, the median absolute deviation is near zero,
    so a bound of a few deviations excludes the other cluster entirely — the
    median's bias reintroduced by the guard meant to make the mean safe.
    """
    from dropscore.tracking import _robust_mean

    values = np.array([5.0] * 90 + [4.0] * 10)

    assert float(np.median(np.abs(values - np.median(values)))) == 0.0
    assert _robust_mean(values) == pytest.approx(4.9, abs=0.01)


def test_robust_mean_still_rejects_outliers() -> None:
    """The job the median was doing has to keep being done."""
    from dropscore.tracking import _robust_mean

    clean = np.array([100.0, 101.0, 99.0, 100.5, 99.5])
    with_junk = np.append(clean, [2000.0, 3.0])

    assert _robust_mean(with_junk) == pytest.approx(_robust_mean(clean), abs=0.5)


def test_speed_pools_frame_pairs_not_probe_summaries() -> None:
    """A probe resting on 22 pairs must not weigh as much as one on 38."""
    from dropscore.tracking import SpeedEstimate

    estimate = SpeedEstimate(value=150.0, spread=1.0, samples=3, shifts=(149.0, 150.0, 151.0))

    assert estimate.shifts, "the per-pair measurements must survive for pooling"
    assert len(estimate.shifts) == 3
