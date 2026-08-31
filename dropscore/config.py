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
class CalibrationConfig:
    """Fitting the keyboard grid (stage 3)."""

    # The strike line is placed where row activity drops most sharply. That drop
    # must be at least this fraction of the busiest row, or the video is rejected
    # as not having a still keyboard under a moving area at all.
    activity_ratio: float = 0.15

    # Fractional depth range searched for the strike line. Keybeds occupy roughly
    # the bottom 15-35% of the frame, so the split is never near the top.
    split_search: tuple[float, float] = (0.50, 0.95)

    # Sanity bounds on the keybed band, as guards against non-piano video.
    min_keybed_px: int = 12
    max_keybed_ratio: float = 0.60
    min_edge_energy: float = 1.0

    # Plausible white-key widths in pixels. Below ~4px adjacent keys cannot be
    # separated at all; above ~80px the video is showing barely an octave.
    min_key_px: float = 4.0
    max_key_px: float = 80.0

    # Depth ranges within the keybed, as fractions from its top edge. The upper
    # band is where black keys live; the lower band is guaranteed white.
    black_band: tuple[float, float] = (0.10, 0.45)
    white_band: tuple[float, float] = (0.75, 0.98)

    # Width of the strip sampled at each boundary, as a fraction of key width.
    sample_ratio: float = 0.40

    # Columns quieter than this fraction of the busiest are outside the keyboard.
    extent_ratio: float = 0.08

    # Reject the fit below this black-key pattern match rate. A one-key offset
    # typically scores around 0.4, so 0.8 separates cleanly.
    min_confidence: float = 0.80


@dataclass(frozen=True)
class Config:
    """Root config. Sub-configs are added by later stages."""

    video: VideoConfig = field(default_factory=VideoConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def evolve(self, **changes: Any) -> "Config":
        """Return a copy with top-level fields replaced."""
        return replace(self, **changes)


DEFAULT = Config()
