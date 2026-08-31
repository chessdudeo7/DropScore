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
class TileConfig:
    """Finding and identifying falling tiles (stage 4)."""

    # Lab lightness is downweighted so a gradient-filled tile stays one colour.
    # The cost is that hands distinguished only by brightness merge into one
    # track; stage 7 separates those by pitch instead.
    lightness_weight: float = 0.35

    # Distance from the static background before a pixel is a tile candidate.
    background_distance: float = 18.0

    # Palette discovery.
    max_palettes: int = 4
    merge_distance: float = 12.0  # chroma distance below which colours are one
    min_palette_share: float = 0.05
    max_sample_pixels: int = 200_000

    # How close a pixel must sit to a palette colour to count as solid tile.
    # Bloom is a blend of tile and background, so it lands outside this and is
    # excluded without any erosion.
    color_tolerance: float = 22.0

    # Blob filtering.
    min_tile_height: int = 3
    min_tile_width_ratio: float = 0.35  # of a white key; black tiles are ~0.62

    # A key must be this covered by a blob to be claimed from it.
    min_coverage: float = 0.60

    # Rows at least this filled are part of a tile; the gaps split repeated notes.
    row_fill_ratio: float = 0.50

    # A region less filled than this is an outlined tile, whose middle is empty
    # by design, rather than two stacked tiles with a seam between them.
    outline_fill_ratio: float = 0.45


@dataclass(frozen=True)
class TrackingConfig:
    """Following tiles and converting them to times (stage 5)."""

    # Scroll-speed measurement. Correlation runs on the background-subtracted
    # residual, so a pair with nothing falling has almost no signal and is
    # skipped rather than contributing a spurious zero.
    min_residual: float = 2.0
    min_correlation: float = 0.05
    min_speed: float = 20.0  # px/s
    max_speed: float = 2000.0
    min_speed_samples: int = 5

    # Association. A tile can only be where the known speed puts it, so the gate
    # is a fraction of the distance it should have travelled since last seen.
    min_match_px: float = 4.0
    match_ratio: float = 0.60
    max_gap: float = 0.12  # seconds a track survives unmatched

    # Timing. Edges within this many pixels of a frame boundary or the strike
    # line are clipped, so they carry no usable position.
    edge_margin: float = 2.0
    min_observations: int = 2
    min_duration: float = 0.02  # seconds; below this it is a detection artefact


@dataclass(frozen=True)
class ScoreConfig:
    """Tempo, key, hands and quantization (stage 7)."""

    # Tempo search. Candidate grid spacings are scanned for phase coherence, so
    # the range is on the tatum (finest subdivision), not on the beat.
    min_onsets_for_tempo: int = 8
    min_tatum: float = 0.06  # seconds
    max_tatum: float = 1.00
    tempo_resolution: int = 1200

    # A grid twice as fine fits an onset set exactly as well, so the coarsest
    # period scoring within this fraction of the best is taken as the tatum.
    tatum_tolerance: float = 0.92

    min_bpm: float = 40.0
    max_bpm: float = 208.0
    tempo_prior: float = 110.0  # multiples of the tatum are judged against this
    beats_per_bar: int = 4

    # Quantization. Anything within half a step snaps to *some* gridline, so a
    # tolerance below 0.5 is what makes leaving outliers alone possible at all.
    steps_per_beat: int = 4  # sixteenths in 4/4
    max_shift: float = 0.35
    min_duration: float = 0.03

    # Seconds either side of a note considered when splitting hands by pitch.
    hand_window: float = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Scoring a transcription against ground truth (stage 9)."""

    # A reference and an estimate note match when pitches are equal and onsets
    # fall within this. 50ms is the conventional tolerance for note transcription
    # and is comfortably tighter than a sixteenth at any realistic tempo.
    onset_tolerance: float = 0.05

    # How far a clip's F1 may fall against the stored baseline before it counts
    # as a regression rather than noise.
    regression_tolerance: float = 0.02


@dataclass(frozen=True)
class Config:
    """Root config. Sub-configs are added by later stages."""

    video: VideoConfig = field(default_factory=VideoConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tiles: TileConfig = field(default_factory=TileConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def evolve(self, **changes: Any) -> "Config":
        """Return a copy with top-level fields replaced."""
        return replace(self, **changes)


DEFAULT = Config()
