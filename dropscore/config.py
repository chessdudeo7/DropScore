"""Tunable parameters for the transcription pipeline.

Every magic number the pipeline depends on lives here rather than inline, so that
per-renderer presets (stage 3+) are a matter of swapping a Config, and so the
evaluation harness (stage 9) can sweep values without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class VideoConfig:
    """How the video is read."""

    # Frames wider than this are downscaled to it; narrower ones are left alone.
    # 720p is plenty: a white key is ~6px wide at 1280 across, already enough to
    # separate adjacent keys. Higher resolutions cost time and buy nothing.
    max_width: int = 1280

    # Frames sampled across the video for calibration passes (keybed location,
    # palette clustering). Spread evenly rather than taken from the start, since
    # the opening seconds are often a title card with no tiles.
    calibration_samples: int = 60

    # Fraction of the video to ignore at each end when sampling for calibration.
    calibration_margin: float = 0.05


@dataclass(frozen=True)
class Config:
    """Root config. Sub-configs are added by later stages."""

    video: VideoConfig = field(default_factory=VideoConfig)

    def evolve(self, **changes: Any) -> "Config":
        """Return a copy with top-level fields replaced."""
        return replace(self, **changes)


DEFAULT = Config()
