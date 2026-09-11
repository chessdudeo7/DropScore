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

    # The strike line is placed where the background's horizontal structure
    # rises most sharply: keybed rows cross white and black keys and so vary
    # hugely, while fall-area rows are near-uniform. Measured across the themes
    # that is a row spread of ~73 against ~2-6, so a modest floor separates
    # them and rejects video that is not a keyboard under a falling area.
    min_keybed_contrast: float = 20.0

    # Renderers often shade the top few rows of the keybed. After the split,
    # rows above it are reclaimed while they still carry this share of the
    # keybed's structure — without it the strike line lands 2-3px low.
    #
    # Both bounds matter. On a short clip the temporal median keeps some tile
    # content, so the fall area is not structureless and an unbounded walk
    # climbed 200px into it; the reclaim is only ever worth a few rows.
    keybed_edge_ratio: float = 0.50
    keybed_edge_max: float = 0.02  # of frame height

    # A row counts as part of the keybed band once its structure rises this far
    # from the frame's quiet baseline toward its peak.
    keybed_band_ratio: float = 0.35

    # Black keys cover this share of a keyboard's depth. Only the part of the
    # keybed they cross stands out as structure — below them it is near-uniform
    # white — so the band that is found is scaled up by this to recover the
    # keybed's full depth.
    black_height_ratio: float = 0.62

    # How much of the frame's width a row must carry structure across before
    # it can be part of the keybed. A row of keys spans nearly all of it;
    # tiles held still long enough to survive into the background span only
    # the register they are written in.
    # How far either side of the walked-back keybed edge to look for the row
    # where the picture actually changes. Small: this is a refinement of an
    # edge already found, and a wider search starts finding the bottoms of the
    # black keys instead.
    strike_refine_px: int = 8

    min_keybed_coverage: float = 0.5

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
    # Hue tolerance in degrees. Colours within this of each other are one
    # voice, which keeps a gradient- or bloom-shaded tile from splitting into
    # several palettes while leaving genuinely different hands apart.
    merge_distance: float = 25.0

    # How far off the line between the background and a tile colour another
    # colour may sit and still be read as that tile's bloom or antialiasing
    # rather than a voice of its own. A glow halo measured 0.46.
    blend_tolerance: float = 6.0
    min_palette_share: float = 0.05
    max_sample_pixels: int = 200_000

    # A colour's acceptance radius is at least this multiple of how far its own
    # pixels typically sit from it, so a gradient-filled tile is not clipped at
    # a radius chosen for flat ones. Below the point where color_tolerance
    # takes over it has no effect at all.
    spread_multiple: float = 2.5

    # When measuring that spread, how far out a pixel may sit and still count
    # as one of the colour's own, as a multiple of color_tolerance. Loose
    # enough to keep a gradient's far end, tight enough to exclude pixels that
    # are only nearest to this colour for want of a better match.
    spread_window: float = 2.0

    # How close a pixel must sit to a palette colour to count as solid tile.
    # Bloom is a blend of tile and background, so it lands outside this and is
    # excluded without any erosion.
    color_tolerance: float = 22.0

    # Blob filtering.
    min_tile_height: int = 3
    min_tile_width_ratio: float = 0.35  # of a white key; black tiles are ~0.62

    # A key must be this covered by a blob to be claimed from it.
    min_coverage: float = 0.60

    # Rows at least this filled are part of a tile. The bar is low because what
    # separates two repeated notes is *background* — a genuine gap is empty,
    # not half full — whereas a rounded cap or a bloom-softened edge is only
    # partly covered and must not be read as a break in the middle of a tile.
    row_fill_ratio: float = 0.20

    # How much of the region it spans a blob must actually fill to be a tile
    # rather than a visual effect. Set from the gap between the two: solid
    # tiles measure 0.79 to 0.92 across every theme, spark trails 0.28 to 0.57.
    # Outlined tiles are exempt -- they are hollow on purpose, and are
    # recognised before this applies.
    min_solidity: float = 0.65

    # A region less filled than this is an outlined tile, whose middle is empty
    # by design, rather than two stacked tiles with a seam between them.
    outline_fill_ratio: float = 0.45

    # ...but only if its side edges run nearly the full height, which is what
    # makes an outline an outline. Judged separately from row_fill_ratio: that
    # bar is deliberately low, and reusing it let a key with a neighbour
    # bleeding into one edge column pass as hollow.
    outline_edge_ratio: float = 0.80


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

    # How far from the typical rate, as a factor, a frame pair may sit before
    # it is treated as an outlier rather than a measurement. Scaled to the rate
    # rather than to the spread of the measurements: displacements are
    # quantised to whole pixels, so they cluster, and a bound scaled to a tight
    # cluster throws away the second one.
    outlier_ratio: float = 2.0

    # How many frames apart the correlated pair sits. More than one because a
    # recording may not update every frame, and a duplicated pair reads zero
    # and is discarded, leaving only the pairs that moved twice as far.
    correlation_lag: int = 5

    # Known displacements used to measure the correlator's own offset, and
    # the largest offset worth believing. Anything past this is not a
    # sub-pixel quirk but a sign the probe itself went wrong.
    bias_probe_shifts: tuple[int, ...] = (4, 6, 8)
    max_bias: float = 2.0

    # Association. A tile can only be where the known speed puts it, so the gate
    # is a fraction of the distance it should have travelled since last seen.
    min_match_px: float = 4.0
    match_ratio: float = 0.60
    max_gap: float = 0.12  # seconds a track survives unmatched

    # Timing. Edges within this many pixels of a frame boundary or the strike
    # line are clipped, so they carry no usable position.
    edge_margin: float = 2.0
    # Frames a tile must be seen in before it is a note. Two was every blob
    # that appeared twice, which is what a visual effect looks like: measured
    # across five clips, spurious tracks have a median of 2 observations where
    # real notes have 57. Three removes 56% of the spurious ones for 3% of the
    # real, and is the best of the range -- 4 and 5 trade more real notes than
    # they save.
    #
    # It cannot go much higher without hurting the outlined theme, whose real
    # tracks are legitimately short: its fifth percentile is 2 observations.
    min_observations: int = 3

    # Particles rise; tiles do not. A spark thrown off a struck key travels
    # upward, so a track whose every measurable edge is *ascending* was not a
    # tile, whatever colour it was drawn in.
    #
    # Stated as "rising", not "falling at the scroll speed", though the second
    # is the truer description of a tile and separates the synthetic clips far
    # better. It cannot be used. A real recording breaks one tile into dozens
    # of partial blobs whose edges are pinned by whatever occludes them: on one
    # capture 69% of tracks had an edge that never moved at all, and demanding
    # they descend threw away 190 notes that were as well aligned to the beat
    # as the ones it kept. Stationary is what fragmentation looks like, and
    # only rising is what a particle looks like.
    #
    # Slightly below zero rather than at it, so that noise in a short fit does
    # not read as ascent.
    min_fall_ratio: float = -0.05

    # Unclipped samples needed before the test is allowed to reject a track. A
    # tile taller than the fall area never shows its top edge and its bottom
    # stops dead at the strike line, so the question cannot always be asked --
    # and unanswerable must not read as no.
    min_fall_samples: int = 4
    min_duration: float = 0.02  # seconds; below this it is a detection artefact

    # Two notes on one key closer together than this are one note found twice,
    # not a repeat. Chosen from the gap between the two populations: across the
    # corpus no genuine repeat has its onset within 104ms of the previous one,
    # while fragments of a single tile land within 60ms.
    min_repeat: float = 0.07


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
    tempo_prior: float = 110.0  # where a listener prefers to hear the beat
    beats_per_bar: int = 4

    # Which multiple of the tatum is the beat cannot be read off the onsets
    # alone: a stream of quarters at 100 BPM and one of eighths at 50 produce
    # identical onset times. The choice is made from three things.
    #
    # First, how strongly the onset train repeats at the candidate period —
    # that rules out levels which are not metrical at all, such as reading a
    # piece grouped in twos as though it were in threes.
    #
    # Second, this prior, as a Gaussian on log BPM. Wide on purpose, and wider
    # than it was: at 1.0 it still pulled hard enough to double the tempo of
    # anything slow, because 60 BPM sits far enough down the curve from the
    # 110 the prior likes that 120 outscored it on the prior alone. It should
    # only break a tie between readings the evidence likes equally, and at 2.0
    # it does. Measured over 120 pieces from 60 to 152 BPM, widening it is
    # worth 11 points of exact accuracy on its own.
    tempo_prior_width: float = 2.0

    # How close a partner must land, as a fraction of the tatum, to count as a
    # repeat. Strictly under a half: at a half the acceptance window is a whole
    # tatum wide and takes in the *neighbouring* gridline, so on grid-aligned
    # music every candidate period scores near one and the measure stops
    # discriminating -- two readings of one piece came back at 0.913 apiece,
    # and noise between saturated scores chose the beat.
    repeat_tolerance: float = 0.25

    # Third, how idiomatic the commonest note value looks against the beat.
    # A piece written almost entirely in sixteenths is far rarer than one in
    # eighths or quarters, so a beat making the modal note a sixteenth is
    # evidence it is twice too slow. Adding this costs one exact match over an
    # eight-tempo sweep and buys something worth more: every remaining error
    # becomes a doubling rather than a halving. Doubling renotates the same
    # music in coarser values, which is harmless; halving turns a stream of
    # eighths into a thicket of sixteenths, which is what a reader notices.
    duration_evidence: float = 0.5

    # Weight given to a beat that is not the conventional `steps_per_beat`
    # tatums long. Only decides matters when nothing else does, so it is close
    # to 1.0: at 0.85 it was strong enough to overturn the onset evidence on
    # any piece whose finest grid is a thirty-second, where the beat is eight
    # tatums rather than four and the bonus backs the wrong reading. Removing
    # it entirely is worse again -- an unbroken stream of equal notes supports
    # every level identically, and then this is the only thing left.
    other_multiple: float = 0.95

    # Quantization. Anything within half a step snaps to *some* gridline, so a
    # tolerance below 0.5 is what makes leaving outliers alone possible at all.
    # Zero disables quantization and keeps the measured times.
    steps_per_beat: int = 4  # sixteenths in 4/4
    max_shift: float = 0.35
    min_duration: float = 0.03

    # How hard a candidate key is penalised for the sounding weight that falls
    # outside its scale, as a fraction of total weight.
    #
    # Krumhansl-Schmuckler correlates the pitch-class profile against a key
    # template, and a template match does not care whether the notes a key
    # *requires* are actually present. On a real recording containing no G#
    # and no D# at all, E major still beat E minor (0.7305 to 0.7128) because
    # both share a heavy tonic and dominant, and the notes that rule E major
    # out — C, D and G, a fifth of the sounding weight — cost it nothing.
    #
    # Plausible candidates sit within about 0.02 correlation of each other, so
    # at 0.3 a key excluding a fifth of the music pays 0.06: enough to break a
    # near-tie, not enough to overturn a genuinely clear winner. Chromatic
    # music is unaffected — accuracy over the generated corpus is identical at
    # every value from 0 to 1.2, even with 30% of notes displaced chromatically.
    out_of_scale_penalty: float = 0.3

    # How much of the way to the next onset a note must cover before it is
    # *written* as reaching it. Held-key time is articulation, not note value:
    # a quarter released at three quarters of its length is a detached quarter,
    # and engraving it literally gives a dotted eighth and a rest. Applies to
    # notation only -- MIDI and JSON keep what was actually played. Zero writes
    # the measured durations unchanged.
    legato_ratio: float = 0.6

    # How many neighbouring notes in time set the local hand boundary. Counted
    # rather than timed: a window in seconds spans different amounts of music
    # across the tempo range, and accuracy fell off either side of whichever
    # duration was picked. Measured on 30 held-out pieces at 60-144bpm, this
    # is flat from 8 to 10 and falls away by 5 and 14.
    hand_neighbours: int = 8

    # How well a single pitch boundary must sort the two colour groups before
    # they are believed to be hands. Real hands cross and share the middle of
    # the keyboard, so this is well under 1.0; a colour split that fell along
    # some other axis -- black keys against white, most often -- lands near
    # chance. Measured over the corpus and two real captures: genuine hands
    # score 0.877 to 0.954, and a one-colour video whose tiles darkened over
    # the accidentals scored 0.664. This sits between them but nearer the
    # failure, because the safer error is to trust the colours -- falling back
    # only helps where they are plainly not hands.
    hand_separability: float = 0.75

    # Below this many notes the separability test is skipped and the colours
    # are taken at their word: a handful of notes can look unseparable by luck.
    min_hand_notes: int = 8

    # How hands are decided:
    #   "color" — trust the tile colours, fall back to pitch when there is one
    #   "pitch" — always split by a moving pitch boundary
    #   "split" — fixed boundary at middle C
    #   "none"  — one staff, everything right hand
    hand_mode: str = "color"

    # Override the inferred tempo or key. Left None, both are estimated.
    fixed_tempo: float | None = None
    fixed_key: str | None = None


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
