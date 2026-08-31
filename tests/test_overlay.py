from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dropscore.calibrate import Calibration
from dropscore.notes import Note, NoteSequence
from dropscore.overlay import (
    ANCHOR_COLOR,
    annotate,
    draw_grid,
    draw_tiles,
    dump_frames,
    dump_video,
    track_color,
)
from dropscore.synth import RenderSpec, SynthRenderer, generate
from dropscore.tiles import detect_in_frame, discover_palette
from dropscore.video import Frame, VideoReader

SPEC = RenderSpec(width=640, height=320, fps=10.0)


def _renderer(sequence: NoteSequence | None = None) -> SynthRenderer:
    return SynthRenderer(sequence or generate(seed=2, bars=2), SPEC)


def _calibration(renderer: SynthRenderer) -> Calibration:
    return Calibration(
        layout=renderer.layout,
        strike_y=renderer.strike_y,
        keybed_bottom=renderer.spec.height,
        white_width=renderer.layout.white_width,
        confidence=0.97,
    )


def _frame(renderer: SynthRenderer, index: int = 20) -> Frame:
    return Frame(index, index / SPEC.fps, renderer.frame(index), 1.0)


def test_annotate_leaves_the_source_frame_untouched() -> None:
    renderer = _renderer()
    frame = _frame(renderer)
    before = frame.image.copy()

    annotate(frame, _calibration(renderer))
    assert np.array_equal(frame.image, before)


def test_annotated_frame_keeps_its_shape() -> None:
    renderer = _renderer()
    frame = _frame(renderer)
    assert annotate(frame, _calibration(renderer)).shape == frame.image.shape


def test_grid_marks_every_c_on_the_keybed() -> None:
    """The check that matters: bright anchors must land on the real C keys."""
    renderer = _renderer()
    calibration = _calibration(renderer)
    image = draw_grid(renderer.frame(20).copy(), calibration, labels=False)

    row = calibration.strike_y + (calibration.keybed_bottom - calibration.strike_y) // 2
    for pitch in renderer.layout.pitches:
        if pitch % 12:
            continue
        column = int(renderer.layout.white_span(pitch)[0])
        assert tuple(int(v) for v in image[row, column]) == ANCHOR_COLOR


def test_grid_draws_nothing_above_the_strike_line() -> None:
    renderer = _renderer()
    calibration = _calibration(renderer)
    plain = renderer.frame(20)
    drawn = draw_grid(plain.copy(), calibration, labels=False)

    # Everything above the strike line is untouched, bar the strike line itself.
    assert np.array_equal(drawn[: calibration.strike_y - 1], plain[: calibration.strike_y - 1])


def test_tile_boxes_use_one_colour_per_track() -> None:
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=72, duration=1.0, hand="R"),
            Note(onset=0.0, pitch=48, duration=1.0, hand="L"),
        ]
    )
    renderer = SynthRenderer(sequence, RenderSpec(width=640, height=320, fps=10.0))
    calibration = _calibration(renderer)
    frames = [_frame(renderer, i) for i in range(0, renderer.frame_count, 3)]
    palette = discover_palette(frames, calibration)

    frame = _frame(renderer, int(renderer.notes[0].onset * SPEC.fps) - 5)
    tiles = detect_in_frame(frame, palette, calibration)
    image = draw_tiles(frame.image.copy(), tiles, labels=False)

    for tile in tiles:
        expected = track_color(tile.track)
        row = int((tile.top + tile.bottom) / 2)
        column = int(tile.left)
        assert tuple(int(v) for v in image[row, column]) == expected


def test_hud_dims_its_corner() -> None:
    renderer = _renderer()
    frame = _frame(renderer)
    annotated = annotate(frame, _calibration(renderer), speed=120.0)
    assert annotated[0:10, 0:10].mean() < frame.image[0:10, 0:10].mean() + 1


def test_dump_frames_writes_one_png_each(tmp_path: Path) -> None:
    renderer = _renderer()
    frames = [_frame(renderer, i) for i in (5, 10, 15)]
    written = dump_frames(frames, _calibration(renderer), tmp_path)

    assert len(written) == 3
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    assert all(p.suffix == ".png" for p in written)


def test_dump_frames_detects_tiles_when_given_a_palette(tmp_path: Path) -> None:
    renderer = _renderer()
    calibration = _calibration(renderer)
    frames = [_frame(renderer, i) for i in range(0, renderer.frame_count, 4)]
    palette = discover_palette(frames, calibration)

    plain = dump_frames(frames[:2], calibration, tmp_path / "plain")
    boxed = dump_frames(frames[:2], calibration, tmp_path / "boxed", palette=palette)

    import cv2  # noqa: PLC0415

    assert not np.array_equal(cv2.imread(str(plain[0])), cv2.imread(str(boxed[0])))


def test_dump_video_is_readable(tmp_path: Path) -> None:
    renderer = _renderer()
    frames = [_frame(renderer, i) for i in range(20)]
    target = dump_video(
        frames,
        _calibration(renderer),
        tmp_path / "debug.mp4",
        SPEC.fps,
        (SPEC.width, SPEC.height),
    )

    assert target.exists()
    with VideoReader(target) as reader:
        assert reader.info.source_width == SPEC.width
        assert reader.info.frame_count == pytest.approx(20, abs=2)
