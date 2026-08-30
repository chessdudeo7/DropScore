from __future__ import annotations

from pathlib import Path

import pytest

from dropscore.config import Config, VideoConfig
from dropscore.video import VideoError, VideoReader

from .conftest import CLIP_FPS, CLIP_FRAMES, CLIP_HEIGHT, CLIP_WIDTH


def test_reports_metadata(clip: Path) -> None:
    with VideoReader(clip) as reader:
        info = reader.info
        assert info.source_width == CLIP_WIDTH
        assert info.source_height == CLIP_HEIGHT
        assert info.fps == pytest.approx(CLIP_FPS, abs=0.5)
        # Encoders may drop or pad a frame; the count should still be close.
        assert info.frame_count == pytest.approx(CLIP_FRAMES, abs=2)
        assert info.duration == pytest.approx(CLIP_FRAMES / CLIP_FPS, abs=0.2)


def test_does_not_upscale_small_videos(clip: Path) -> None:
    """max_width is a ceiling, not a target."""
    with VideoReader(clip, Config(video=VideoConfig(max_width=4096))) as reader:
        assert reader.info.scale == 1.0
        assert reader.info.width == CLIP_WIDTH


def test_downscales_to_max_width(clip: Path) -> None:
    config = Config(video=VideoConfig(max_width=320))
    with VideoReader(clip, config) as reader:
        assert reader.info.width == 320
        assert reader.info.height == pytest.approx(CLIP_HEIGHT // 2, abs=1)
        frame = next(reader.frames())
        assert frame.image.shape[1] == 320
        assert frame.scale == pytest.approx(0.5, abs=0.01)


def test_iterates_every_frame(clip: Path) -> None:
    with VideoReader(clip) as reader:
        frames = list(reader.frames())
    assert len(frames) == pytest.approx(CLIP_FRAMES, abs=2)
    assert [f.index for f in frames] == list(range(len(frames)))


def test_step_skips_frames(clip: Path) -> None:
    with VideoReader(clip) as reader:
        frames = list(reader.frames(step=3))
    assert [f.index for f in frames] == list(range(0, len(frames) * 3, 3))


def test_start_and_stop_bound_iteration(clip: Path) -> None:
    with VideoReader(clip) as reader:
        frames = list(reader.frames(start=10, stop=20))
    assert [f.index for f in frames] == list(range(10, 20))


def test_frame_time_follows_index(clip: Path) -> None:
    with VideoReader(clip) as reader:
        frames = list(reader.frames(start=30, stop=33))
    for frame in frames:
        assert frame.time == pytest.approx(frame.index / reader.info.fps, abs=1e-6)


def test_step_must_be_positive(clip: Path) -> None:
    with VideoReader(clip) as reader:
        with pytest.raises(ValueError, match="step must be"):
            list(reader.frames(step=0))


def test_sample_returns_requested_count(clip: Path) -> None:
    with VideoReader(clip) as reader:
        frames = reader.sample(8)
    assert len(frames) == 8
    indices = [f.index for f in frames]
    assert indices == sorted(indices)
    # Sampling should skip the very start and end of the clip.
    assert indices[0] > 0
    assert indices[-1] < CLIP_FRAMES - 1


def test_sample_ignores_margin_when_clip_is_short(clip: Path) -> None:
    """Asking for more samples than the margin allows widens to the whole clip."""
    with VideoReader(clip) as reader:
        frames = reader.sample(CLIP_FRAMES)
    assert frames[0].index == 0


def test_missing_file_raises() -> None:
    with pytest.raises(VideoError, match="No such file"):
        VideoReader("nope.mp4")


def test_undecodable_file_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"this is not a video")
    with pytest.raises(VideoError):
        VideoReader(bogus)


def test_close_is_idempotent(clip: Path) -> None:
    reader = VideoReader(clip)
    reader.close()
    reader.close()
