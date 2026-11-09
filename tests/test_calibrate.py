"""Stage 3 scored against stage 2's ground truth.

The strict assertion throughout is that ``first_pitch`` must be recovered
*exactly*. A one-key error is not a small error — it transposes everything
downstream — so there is no tolerance to give.
"""

from __future__ import annotations

import numpy as np
import pytest

from dropscore.calibrate import (
    BLACK_PATTERN,
    CalibrationError,
    _bend_grid,
    _boundary_positions,
    anchor_pitch,
    calibrate,
    estimate_period,
    find_keybed,
    median_frame,
    median_images,
)
from dropscore.config import DEFAULT
from dropscore.keyboard import COMMON_RANGES
from dropscore.notes import Note, NoteSequence
from dropscore.synth import RenderSpec, SynthRenderer, generate, get_theme
from dropscore.synth.themes import THEMES
from dropscore.video import Frame

# Wide enough that a white key is ~18px on an 88-key board, which is comparable
# to a real 720p video.
SPEC = RenderSpec(width=960, height=540, fps=10.0)


def _frames(renderer: SynthRenderer, count: int = 24) -> list[Frame]:
    """Sample rendered frames directly, skipping the encode/decode round trip."""
    indices = np.linspace(0, renderer.frame_count - 1, num=count, dtype=int)
    return [
        Frame(index=int(i), time=int(i) / renderer.spec.fps, image=renderer.frame(int(i)), scale=1.0)
        for i in indices
    ]


def _render(theme: str = "classic", key_range: str = "88", seed: int = 1) -> SynthRenderer:
    spec = RenderSpec(
        width=SPEC.width,
        height=SPEC.height,
        fps=SPEC.fps,
        theme=get_theme(theme),
        key_range=key_range,
    )
    return SynthRenderer(generate(seed=seed, bars=4), spec)


# ── unit level ───────────────────────────────────────────────────────


def test_estimate_period_recovers_a_known_grid() -> None:
    length, period = 960, 18.0
    x = np.arange(length)
    profile = (np.abs((x % period)) < 1.0).astype(np.float32)  # a spike per period

    got_period, offset = estimate_period(profile, 4.0, 80.0)
    assert got_period == pytest.approx(period, rel=0.02)
    # Offset should land on a spike, i.e. near 0 modulo the period.
    assert min(offset % period, period - offset % period) < 1.5


def _ramped_profile(length: int, left: float, right: float) -> tuple[np.ndarray, np.ndarray]:
    """Separator spikes whose spacing grows steadily from ``left`` to ``right``."""
    positions = [12.0]
    while positions[-1] < length - 12:
        u = positions[-1] / length
        positions.append(positions[-1] + left + (right - left) * u)
    separators = np.array(positions[:-1])
    profile = np.zeros(length, dtype=np.float32)
    for x in separators:
        profile[int(round(x))] = 1.0
    return np.convolve(profile, np.ones(3, dtype=np.float32) / 3.0, mode="same"), separators


def test_bend_grid_fits_a_keyboard_wider_at_one_end() -> None:
    """A phone's view of a screen: keys 22.5px wide at one end, 26px at the other.

    No evenly spaced grid fits that; the best one is half a key off somewhere.
    The bent grid lands every separator.
    """
    profile, separators = _ramped_profile(1280, 22.5, 26.0)
    x_lo, x_hi = int(separators[0]), int(separators[-1])
    period, offset = estimate_period(profile, 4.0, 80.0)
    flat = _boundary_positions(offset, period, x_lo, x_hi)
    flat_miss = max(min(abs(flat - x)) for x in separators)
    assert flat_miss > period / 3, "a flat grid should not be able to fit this profile"

    bent = _bend_grid(profile, period, offset, x_lo, x_hi, DEFAULT)
    assert bent is not None, "the bent grid was not believed"
    lines, bend = bent
    assert bend > 0
    assert max(min(abs(lines - x)) for x in separators) < period / 10
    # One line per separator, plus the bracketing line past the last one.
    assert lines.size in (separators.size, separators.size + 1)

    # With the margin out of reach the flat grid is kept, which is the bug.
    from dataclasses import replace  # noqa: PLC0415

    strict = replace(DEFAULT, calibration=replace(DEFAULT.calibration, bend_ratio=100.0))
    assert _bend_grid(profile, period, offset, x_lo, x_hi, strict) is None


def test_bend_grid_leaves_an_even_keyboard_alone() -> None:
    profile, separators = _ramped_profile(1280, 24.5, 24.5)
    x_lo, x_hi = int(separators[0]), int(separators[-1])
    period, offset = estimate_period(profile, 4.0, 80.0)
    assert _bend_grid(profile, period, offset, x_lo, x_hi, DEFAULT) is None


def test_estimate_period_rejects_a_flat_profile() -> None:
    with pytest.raises(CalibrationError, match="no vertical structure"):
        estimate_period(np.ones(500, dtype=np.float32), 4.0, 80.0)


def test_anchor_pitch_finds_the_rotation() -> None:
    for rotation in range(7):
        pattern = np.array(
            [BLACK_PATTERN[(i + rotation) % 7] for i in range(30)], dtype=bool
        )
        found, score = anchor_pitch(pattern)
        assert found == rotation
        assert score == 1.0


def test_anchor_pitch_tolerates_a_misread_boundary() -> None:
    pattern = np.array([BLACK_PATTERN[i % 7] for i in range(28)], dtype=bool)
    pattern[5] = not pattern[5]
    rotation, score = anchor_pitch(pattern)
    assert rotation == 0
    assert 0.9 < score < 1.0


def test_anchor_pitch_needs_a_full_octave() -> None:
    with pytest.raises(CalibrationError, match="need at least"):
        anchor_pitch(np.array([True, True, False], dtype=bool))


def test_keybed_is_found_at_the_strike_line() -> None:
    renderer = _render()
    frames = _frames(renderer)
    top, bottom = find_keybed(frames, median_frame(frames), DEFAULT)
    assert top == pytest.approx(renderer.strike_y, abs=3)
    assert bottom == pytest.approx(renderer.spec.height, abs=3)


def test_the_strike_line_lands_on_the_change_not_on_the_first_busy_row() -> None:
    """A shadow band above the keybed can pass for keys, and on held chords does.

    Both tests for "looks like keys" are satisfied by the band a renderer
    draws above the keybed: it carries the key separators, so it is busy right
    across the frame. Normally the rows above it are quiet and the search stops
    anyway -- but a piece of long held chords keeps tiles in the temporal
    median, so the fall area reads as structured too and the edge is placed
    several pixels high. Four pixels there was 28ms of onset error on every
    note of one clip.

    What marks the edge is not which side is busier but that the two differ.
    """
    height, width, strike = 400, 640, 300
    background = np.full((height, width, 3), 30, dtype=np.uint8)

    # Held chords: a few columns lit all the way down the fall area.
    for x in (100, 140, 180):
        background[:strike, x : x + 24] = (200, 180, 120)

    # The shadow band, carrying the key separators through it.
    background[strike - 4 : strike, :] = (70, 70, 70)
    for x in range(0, width, 18):
        background[strike - 4 : strike, x : x + 2] = (210, 210, 210)

    # The keybed itself: white keys with black keys standing on them.
    background[strike:, :] = 245
    for x in range(0, width, 18):
        background[strike : strike + 60, x : x + 8] = 20

    frames = [
        Frame(index=i, time=i / 30.0, image=background.copy(), scale=1.0)
        for i in range(4)
    ]
    frames[1].image[0:50, 300:340] = (255, 255, 255)  # something has to move

    top, _ = find_keybed(frames, background, DEFAULT)
    assert top == strike


def test_calibration_needs_more_than_one_frame() -> None:
    renderer = _render()
    with pytest.raises(CalibrationError, match="at least two frames"):
        calibrate(_frames(renderer, count=1))


def test_static_video_is_rejected() -> None:
    """No moving tiles means this is not a falling-tile video."""
    renderer = SynthRenderer(NoteSequence(), SPEC)
    frames = [
        Frame(index=i, time=i / SPEC.fps, image=renderer.frame(0), scale=1.0)
        for i in range(4)
    ]
    with pytest.raises(CalibrationError, match="nothing moves|no clear boundary"):
        calibrate(frames)


# ── scored against ground truth ──────────────────────────────────────


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_calibrates_every_theme(theme: str) -> None:
    renderer = _render(theme=theme)
    result = calibrate(_frames(renderer))

    assert result.layout.first_pitch == renderer.layout.first_pitch
    assert result.layout.last_pitch == renderer.layout.last_pitch
    assert result.white_width == pytest.approx(renderer.layout.white_width, abs=0.5)
    assert result.strike_y == pytest.approx(renderer.strike_y, abs=3)
    assert result.confidence >= 0.9


@pytest.mark.parametrize("key_range", sorted(COMMON_RANGES))
def test_calibrates_every_key_range(key_range: str) -> None:
    renderer = _render(key_range=key_range)
    result = calibrate(_frames(renderer))

    assert result.layout.first_pitch == renderer.layout.first_pitch
    assert result.layout.white_count == renderer.layout.white_count


def test_every_key_center_maps_back_to_the_right_pitch() -> None:
    """The property that actually matters downstream."""
    renderer = _render()
    fitted = calibrate(_frames(renderer)).layout

    for pitch in renderer.layout.pitches:
        column = renderer.layout.key_center(pitch)
        depth = 0.2 if pitch % 12 in (1, 3, 6, 8, 10) else 0.9
        assert fitted.pitch_at(column, depth) == pitch


def test_finds_the_extent_when_the_keyboard_is_masked_at_the_edges() -> None:
    """Occluding the outer keys should not disturb the grid itself.

    The visible keyboard genuinely starts on a different key, so first_pitch is
    expected to move; what must survive is the period and the pattern lock.
    """
    renderer = _render()
    frames = _frames(renderer)
    margin = 60
    for frame in frames:
        frame.image[:, :margin] = 0
        frame.image[:, -margin:] = 0

    result = calibrate(frames)
    assert result.white_width == pytest.approx(renderer.layout.white_width, abs=0.5)
    assert result.confidence >= 0.9
    assert result.layout.x0 >= margin - result.white_width


def test_survives_jpeg_noise() -> None:
    renderer = _render()
    rng = np.random.default_rng(0)
    frames = _frames(renderer)
    for frame in frames:
        noise = rng.normal(0, 4, frame.image.shape)
        frame.image[:] = np.clip(frame.image.astype(float) + noise, 0, 255).astype(np.uint8)

    result = calibrate(frames)
    assert result.layout.first_pitch == renderer.layout.first_pitch
    assert result.white_width == pytest.approx(renderer.layout.white_width, abs=0.5)


def test_calibration_is_independent_of_what_is_played() -> None:
    a = calibrate(_frames(_render(seed=1)))
    b = calibrate(_frames(_render(seed=9)))
    assert a.layout == b.layout


def test_sparse_unmusical_content_still_calibrates() -> None:
    spec = RenderSpec(width=SPEC.width, height=SPEC.height, fps=SPEC.fps, key_range="61")
    sequence = NoteSequence.of(
        [Note(onset=i * 0.5, pitch=36 + (i * 7) % 60, duration=0.4) for i in range(24)]
    )
    renderer = SynthRenderer(sequence, spec)
    result = calibrate(_frames(renderer))
    assert result.layout.first_pitch == 36


def test_black_classification_survives_lit_keys() -> None:
    """A key lit by the player must not drag the black/white split with it.

    Two-means seeded at the extremes put the line between "normal" and
    "outlier" rather than between black and white, so every real black key
    landed on the white side. Values here are taken from a real recording.
    """
    from dropscore.calibrate import _classify_black

    samples = np.array(
        [83, 43, 20, 89, 16, 87, 32, 88, 209, 127, 184, 86, 84, 85, 82, 17]
    )
    black = _classify_black(samples, white_reference=101.0)

    assert black[samples < 50].all(), "dark boundaries are black keys"
    assert not black[samples > 80].any(), "bright ones, lit or not, are gaps"


def test_keybed_beats_a_burned_in_caption() -> None:
    """A thin band of white-on-black text outscores a keybed on mean structure.

    Real videos carry captions, and a two-line title is denser per row than a
    keyboard is. What separates them is depth, so the band is chosen on total
    structure and the caption loses by six to one.
    """
    from dropscore.calibrate import _runs_above

    structure = np.zeros(720, dtype=np.float32)
    structure[300:420] = 44.0  # a keybed: moderate, deep
    structure[690:706] = 54.0  # a caption: denser, thin

    runs = _runs_above(structure, threshold=20.0, minimum=12)
    top, end = max(runs, key=lambda run: float(structure[run[0] : run[1]].sum()))

    assert (top, end) == (300, 420)


# ── awkward footage ──────────────────────────────────────────────────


def test_a_tilted_keyboard_is_measured_and_levelled() -> None:
    import cv2  # noqa: PLC0415

    from dropscore.video import measure_tilt  # noqa: PLC0415

    renderer = _render()
    frames = _frames(renderer)
    height, width = frames[0].image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.8, 1.0)
    tilted = [
        cv2.warpAffine(f.image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        for f in frames
    ]

    level, _ = measure_tilt([f.image for f in frames])
    degrees, agreement = measure_tilt(tilted)

    assert abs(level) < 0.05
    assert degrees == pytest.approx(-0.8, abs=0.1)
    assert agreement >= DEFAULT.video.level_min_agreement


def test_a_sliver_past_the_board_is_trimmed_rather_than_failing() -> None:
    """A board a few pixels in from the frame edge bracketed 53 white keys on a
    full piano, which has 52, and calibration refused the video outright."""
    renderer = _render()
    frames = _frames(renderer)
    pad = int(renderer.layout.white_width * 0.6)
    padded = []
    for f in frames:
        image = np.zeros((f.image.shape[0], f.image.shape[1] + pad, 3), np.uint8)
        image[:, pad:] = f.image
        image[:, :pad] = f.image[:, :1]  # the frame edge smeared outward, as a keystone does
        padded.append(Frame(f.index, f.time, image, 1.0))

    result = calibrate(padded)

    assert (result.layout.first_pitch, result.layout.last_pitch) == (21, 108)


def _on_a_desktop(image: np.ndarray) -> np.ndarray:
    """The video in a window in one corner, beside another window full of text."""
    import cv2  # noqa: PLC0415

    h, w = image.shape[:2]
    desk = np.full((h, w, 3), (90, 70, 50), np.uint8)
    cv2.rectangle(desk, (w // 2 + 40, 30), (w - 30, h - 60), (245, 245, 245), -1)
    for row in range(6):
        cv2.putText(desk, f"notes line {row}", (w // 2 + 60, 70 + 35 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    cv2.rectangle(desk, (14, 14), (22 + w // 2, 48 + h // 2), (200, 200, 200), -1)
    desk[40 : 40 + h // 2, 18 : 18 + w // 2] = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return desk


def test_finds_the_keyboard_of_a_video_playing_in_a_window() -> None:
    """A screen recording of a window: every row crossed the window edges and
    the other window's text, and the whole fall area was taken for keybed."""
    from dataclasses import replace  # noqa: PLC0415

    renderer = _render()
    frames = [replace(f, image=_on_a_desktop(f.image)) for f in _frames(renderer)]
    result = calibrate(frames)

    assert result.layout.first_pitch == renderer.layout.first_pitch
    assert result.layout.last_pitch == renderer.layout.last_pitch
    for pitch in (21, 60, 61, 108):
        expected = 18 + renderer.layout.key_center(pitch) / 2
        assert result.layout.key_center(pitch) == pytest.approx(expected, abs=1.5)


def test_a_busy_lane_does_not_cut_the_window_in_two() -> None:
    """A key held through most of the sampled frames leaves its lane's median
    off the background colour. On a corner-window copy of the Alkan render
    three such lanes cut the video at its middle and half the keyboard was
    lost; inside the motion they are lanes, not the window's edge."""
    from dataclasses import replace  # noqa: PLC0415

    from dropscore.calibrate import video_columns  # noqa: PLC0415

    renderer = _render()
    frames = [replace(f, image=_on_a_desktop(f.image)) for f in _frames(renderer)]
    fall = renderer.layout.key_span(60)
    lo, hi = 18 + int(fall[0] / 2), 18 + int(fall[1] / 2)
    for f in frames:
        f.image[45 : 40 + int(renderer.strike_y / 2) - 4, lo:hi] = (230, 80, 200)  # held all along
    found = video_columns(frames)
    assert found is not None and found[0] <= 20 and found[1] >= 18 + renderer.layout.width / 2 - 2
    narrow = replace(DEFAULT, calibration=replace(DEFAULT.calibration, video_lane_share=0.0))
    cut = video_columns(frames, narrow)
    assert cut is None or cut[0] > lo or cut[1] < hi


def test_a_board_cut_off_on_the_left_starts_on_the_piano() -> None:
    """A full piano with its lowest four white keys cut off by the picture
    shows 48 white keys from an E. Centred on middle C those would start at
    E0, below the bottom of a piano, and the fit was thrown away: the only
    octave they fit is from E1."""
    from dropscore.calibrate import _first_pitch  # noqa: PLC0415

    assert _first_pitch(2, 48) == 28


def test_the_picture_top_is_the_first_row_that_moves() -> None:
    """A vertical phone video holds the picture under a blank band. A tile
    top at the band's edge is clipped there, not a top, so the calibration
    says where the picture begins."""
    from dataclasses import replace  # noqa: PLC0415

    from dropscore.calibrate import picture_top  # noqa: PLC0415
    from dropscore.video import Frame  # noqa: PLC0415

    frames = []
    for i in range(6):
        image = np.zeros((300, 200, 3), np.uint8)
        image[120 + 10 * i : 140 + 10 * i, 50:80] = 200  # a tile falling below row 120
        frames.append(Frame(i, i / 30.0, image, 1.0))

    assert picture_top(frames) == 120
    assert picture_top([frames[0]]) == 0

    # A player's own bar, a few rows deep, is not the picture's edge: the
    # tops clipped under one timed long notes right on two recordings.
    shallow = [replace(f, image=np.roll(f.image, -110, axis=0)) for f in frames]
    assert picture_top(shallow) == 0


def test_the_temporal_median_is_taken_a_band_at_a_time() -> None:
    """Sixty 720p frames stacked whole are 170MB, and np.median copies the
    stack to partition it; the same median a band at a time holds a twentieth
    of that. The bands must not change a single pixel, including where the
    frame's height is not a multiple of the band."""
    import tracemalloc  # noqa: PLC0415

    from dropscore.calibrate import MEDIAN_BAND_ROWS  # noqa: PLC0415

    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (201, 160, 3), dtype=np.uint8) for _ in range(9)]
    tracemalloc.start()
    try:
        expected = np.median(np.stack(images), axis=0).astype(np.uint8)
        whole = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
        banded = median_images(images)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert np.array_equal(banded, expected)
    assert peak < whole / 2, (peak, whole)
    assert MEDIAN_BAND_ROWS < 201
