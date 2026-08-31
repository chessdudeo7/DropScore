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
    anchor_pitch,
    calibrate,
    estimate_period,
    find_keybed,
    median_frame,
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
