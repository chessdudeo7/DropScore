"""Stage 4 scored against stage 2's ground truth.

The central test asks the only question that matters: at a given instant, does the
detector report exactly the set of pitches the renderer drew?
"""

from __future__ import annotations

import numpy as np
import pytest

from dropscore.calibrate import Calibration
from dropscore.config import DEFAULT
from dropscore.keyboard import KeyboardLayout
from dropscore.notes import Note, NoteSequence
from dropscore.synth import RenderSpec, SynthRenderer, generate, get_theme
from dropscore.synth.themes import THEMES
from dropscore.tiles import TileError, detect_in_frame, discover_palette
from dropscore.video import Frame

SPEC = RenderSpec(width=960, height=540, fps=10.0)


def _renderer(theme: str = "classic", sequence: NoteSequence | None = None) -> SynthRenderer:
    spec = RenderSpec(
        width=SPEC.width, height=SPEC.height, fps=SPEC.fps, theme=get_theme(theme)
    )
    return SynthRenderer(sequence or generate(seed=1, bars=4), spec)


def _truth_calibration(renderer: SynthRenderer) -> Calibration:
    """Use the exact geometry, so stage 4 is tested without stage 3's error."""
    return Calibration(
        layout=renderer.layout,
        strike_y=renderer.strike_y,
        keybed_bottom=renderer.spec.height,
        white_width=renderer.layout.white_width,
        confidence=1.0,
    )


def _frames(renderer: SynthRenderer, count: int = 20) -> list[Frame]:
    indices = np.linspace(0, renderer.frame_count - 1, num=count, dtype=int)
    return [
        Frame(int(i), int(i) / renderer.spec.fps, renderer.frame(int(i)), 1.0)
        for i in indices
    ]


def _frame_at(renderer: SynthRenderer, t: float) -> Frame:
    index = int(round(t * renderer.spec.fps))
    return Frame(index, index / renderer.spec.fps, renderer.frame(index), 1.0)


def _expected_pitches(renderer: SynthRenderer, t: float, min_height: int = 0) -> set[int]:
    """Pitches whose tile is on screen at time t, optionally only tall ones.

    A tile entering the frame is a sliver for a frame or two, right at the
    detector's minimum-height cutoff, so assertions bracket rather than demand
    exactness: every clearly visible tile must be found, and nothing may be
    reported that was not drawn at all.
    """
    found = set()
    for note in renderer.notes:
        rect = renderer._tile_rect(note, t)
        if rect is not None and rect[3] - rect[1] >= min_height:
            found.add(note.pitch)
    return found


# ── merged blobs are split per key, not per blob ─────────────────────


def _painted(
    layout, height: int, width: int, spans: dict[int, tuple[int, int]]
) -> np.ndarray:
    """A frame with tiles painted directly onto given keys.

    Drawn without the renderer's inter-tile gap, so adjacent keys touch and
    form a single contour — which is how plenty of real renderers draw them,
    and the case where per-blob splitting goes wrong.
    """
    import cv2  # noqa: PLC0415

    image = np.full((height, width, 3), 12, dtype=np.uint8)
    for pitch, (top, bottom) in spans.items():
        left, right = layout.key_span(pitch)
        cv2.rectangle(
            image, (int(left), top), (int(right), bottom), (110, 215, 245), -1
        )
    return image


def _detect_painted(spans: dict[int, tuple[int, int]]) -> dict[int, list]:
    """Detect on a painted frame, returning tiles grouped by pitch."""
    renderer = _renderer()
    calibration = _truth_calibration(renderer)
    layout = renderer.layout
    size = (renderer.spec.height, renderer.spec.width)

    # Mostly blank, so the temporal median is the *background*. Painting the
    # majority would make the median the tiles themselves and inverted the
    # palette, which is what an earlier version of this fixture did.
    painted = Frame(0, 0.0, _painted(layout, size[0], size[1], spans), 1.0)
    frames = [
        Frame(i, i / SPEC.fps, _painted(layout, size[0], size[1], {}), 1.0)
        for i in range(1, 6)
    ] + [painted]

    palette = discover_palette(frames, calibration)
    tiles = detect_in_frame(painted, palette, calibration)

    grouped: dict[int, list] = {}
    for tile in tiles:
        grouped.setdefault(tile.pitch, []).append(tile)
    return grouped


def test_adjacent_keys_of_different_lengths_keep_their_own_heights() -> None:
    """The bug: row fill measured across the blob and shared between keys.

    C is long, D is short, and they touch. Measuring fill over the whole blob
    gives D the long note's height — a chord whose voices differ in length is
    ordinary music, not an edge case.
    """
    grouped = _detect_painted({60: (40, 200), 62: (150, 200)})

    assert set(grouped) == {60, 62}
    assert len(grouped[60]) == 1 and len(grouped[62]) == 1

    long_tile = grouped[60][0]
    short_tile = grouped[62][0]
    assert long_tile.height == pytest.approx(160, abs=3)
    assert short_tile.height == pytest.approx(50, abs=3)
    assert short_tile.top == pytest.approx(150, abs=3)


def test_a_gap_on_one_key_does_not_split_its_neighbour() -> None:
    """Two strikes on D, one long note on C, touching.

    Per-blob splitting would cut C at D's gap as well.
    """
    import cv2  # noqa: PLC0415

    renderer = _renderer()
    calibration = _truth_calibration(renderer)
    layout = renderer.layout
    height, width = renderer.spec.height, renderer.spec.width

    def paint() -> np.ndarray:
        image = np.full((height, width, 3), 12, dtype=np.uint8)
        for pitch, (top, bottom) in ((60, (40, 200)), (62, (40, 100)), (62, (130, 200))):
            left, right = layout.key_span(pitch)
            cv2.rectangle(image, (int(left), top), (int(right), bottom), (110, 215, 245), -1)
        return image

    painted = Frame(0, 0.0, paint(), 1.0)
    blank = np.full((height, width, 3), 12, dtype=np.uint8)
    frames = [Frame(i, i / SPEC.fps, blank, 1.0) for i in range(1, 6)] + [painted]

    palette = discover_palette(frames, calibration)
    tiles = detect_in_frame(painted, palette, calibration)

    by_pitch: dict[int, list] = {}
    for tile in tiles:
        by_pitch.setdefault(tile.pitch, []).append(tile)

    assert len(by_pitch[60]) == 1, "the unbroken note was split by its neighbour's gap"
    assert len(by_pitch[62]) == 2, "the two strikes were not separated"


# ── palette discovery ────────────────────────────────────────────────


def test_finds_two_colours_for_a_two_hand_video() -> None:
    renderer = _renderer("synthesia")  # clearly distinct green and blue
    palette = discover_palette(_frames(renderer), _truth_calibration(renderer))
    assert palette.track_count == 2


def test_hands_of_near_identical_hue_collapse_to_one_track() -> None:
    """classic uses two shades of the same gold, which is a real case.

    Chroma is the only signal kept (lightness is discarded to hold gradients
    together), so these merge. Hand assignment falls back to pitch in stage 7.
    """
    renderer = _renderer("classic")
    palette = discover_palette(_frames(renderer), _truth_calibration(renderer))
    assert palette.track_count == 1


def test_gradient_tiles_stay_one_colour_per_hand() -> None:
    """Lightness varies down an aurora tile; hue does not."""
    renderer = _renderer("aurora")
    palette = discover_palette(_frames(renderer), _truth_calibration(renderer))
    assert palette.track_count == 2


def test_empty_video_has_no_palette() -> None:
    renderer = SynthRenderer(NoteSequence(), SPEC)
    frames = [Frame(i, i / SPEC.fps, renderer.frame(0), 1.0) for i in range(4)]
    with pytest.raises(TileError, match="nothing differs"):
        discover_palette(frames, _truth_calibration(renderer))


# ── detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_detects_the_right_pitches(theme: str) -> None:
    renderer = _renderer(theme)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    for t in (3.0, 5.5, 8.0):
        frame = _frame_at(renderer, t)
        found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration)}
        assert _expected_pitches(renderer, frame.time, min_height=8) <= found, f"{theme} t={t}"
        assert found <= _expected_pitches(renderer, frame.time), f"{theme} t={t}"


def test_bloom_does_not_widen_a_tile_onto_its_neighbour() -> None:
    """A lone note under heavy glow must not read as a cluster."""
    sequence = NoteSequence.of([Note(onset=0.0, pitch=60, duration=1.0)])
    renderer = _renderer("neon", sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    assert {tile.pitch for tile in detect_in_frame(frame, palette, calibration)} == {60}


def test_adjacent_keys_are_split_not_merged() -> None:
    sequence = NoteSequence.of(
        [Note(onset=0.0, pitch=p, duration=1.0) for p in (60, 62, 64)]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    assert {tile.pitch for tile in detect_in_frame(frame, palette, calibration)} == {60, 62, 64}


def test_two_white_neighbours_do_not_invent_the_black_key_between_them() -> None:
    sequence = NoteSequence.of(
        [Note(onset=0.0, pitch=p, duration=1.0) for p in (60, 62)]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration)}
    assert 61 not in found


def test_a_black_key_is_still_found_between_two_silent_whites() -> None:
    sequence = NoteSequence.of([Note(onset=0.0, pitch=61, duration=1.0)])
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    assert {tile.pitch for tile in detect_in_frame(frame, palette, calibration)} == {61}


def test_repeated_notes_are_split_vertically() -> None:
    """Two tiles on one key, close together, must not read as one long note."""
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=0.45),
            Note(onset=0.5, pitch=60, duration=0.45),
        ]
    )
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    # Early enough that both tiles are fully on screen at once.
    frame = _frame_at(renderer, renderer.notes[0].onset - 0.6)
    tiles = [t for t in detect_in_frame(frame, palette, calibration) if t.pitch == 60]
    assert len(tiles) == 2


def test_tile_geometry_matches_what_was_drawn() -> None:
    sequence = NoteSequence.of([Note(onset=0.0, pitch=60, duration=0.8)])
    renderer = _renderer("minimal", sequence)  # no glow, no rounding
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    note = renderer.notes[0]
    t = note.onset - 0.4
    frame = _frame_at(renderer, t)
    tile = detect_in_frame(frame, palette, calibration)[0]

    _, top, _, bottom = renderer._tile_rect(note, frame.time)
    assert tile.top == pytest.approx(top, abs=2)
    assert tile.bottom == pytest.approx(bottom, abs=2)


def test_tracks_separate_the_two_hands() -> None:
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=72, duration=1.0, hand="R"),
            Note(onset=0.0, pitch=48, duration=1.0, hand="L"),
        ]
    )
    renderer = _renderer("synthesia", sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    by_pitch = {t.pitch: t.track for t in detect_in_frame(frame, palette, calibration)}
    assert by_pitch[72] != by_pitch[48]


def test_nothing_is_detected_below_the_strike_line() -> None:
    renderer = _renderer()
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, 5.0)
    tiles = detect_in_frame(frame, palette, calibration)
    assert tiles
    assert max(t.bottom for t in tiles) <= calibration.strike_y


def test_detection_uses_the_fitted_layout_not_the_true_one() -> None:
    """A layout shifted by a fraction of a key must still resolve correctly."""
    renderer = _renderer()
    true_layout = renderer.layout
    shifted = KeyboardLayout(
        first_pitch=true_layout.first_pitch,
        last_pitch=true_layout.last_pitch,
        x0=true_layout.x0 + 0.3,
        width=true_layout.width,
    )
    calibration = Calibration(
        layout=shifted,
        strike_y=renderer.strike_y,
        keybed_bottom=renderer.spec.height,
        white_width=shifted.white_width,
        confidence=1.0,
    )
    palette = discover_palette(_frames(renderer), calibration, DEFAULT)

    frame = _frame_at(renderer, 5.0)
    found = {t.pitch for t in detect_in_frame(frame, palette, calibration)}
    assert _expected_pitches(renderer, frame.time, min_height=8) <= found
    assert found <= _expected_pitches(renderer, frame.time)


# ── gradient-filled tiles ────────────────────────────────────────────


def _plain_calibration(strike_y: int, width: int) -> Calibration:
    """A keyboard spanning the frame, for tests about colour rather than keys."""
    layout = KeyboardLayout(width=float(width))
    return Calibration(
        layout=layout,
        strike_y=strike_y,
        keybed_bottom=strike_y + 20,
        white_width=layout.white_width,
        confidence=1.0,
    )


def test_gradient_tile_is_not_clipped_where_its_colour_shifts() -> None:
    """A ramp is not one colour, and a fixed radius cuts it in half.

    Measured on the gradient theme: the discovered colour sits mid-ramp, the
    tile's lower half reads 23 to 32 against a tolerance of 22, and the mask
    stops 150px short of the bottom edge — so the tile's arrival is read from
    the wrong row and the note fragments.
    """
    from dropscore.tiles import Palette, _to_lab, _track_masks

    height, width = 200, 60
    image = np.zeros((height + 40, width, 3), dtype=np.uint8)
    # A vertical ramp between two shades of one hue, as a gradient tile draws.
    for row in range(height):
        t = row / (height - 1)
        image[row, :] = (
            int(108 + t * 126),
            int(53 + t * 62),
            int(67 + t * 77),
        )

    calibration = _plain_calibration(strike_y=height, width=width)
    lab = _to_lab(image[:height])
    middle = np.median(lab.reshape(-1, 3), axis=0)
    pixels = lab.reshape(-1, 3)
    spread = float(
        np.median(np.linalg.norm((pixels - middle) * [0.35, 1.0, 1.0], axis=1))
    )
    palette = Palette(
        background=np.array([0.0, 128.0, 128.0], dtype=np.float32),
        colors=np.array([middle], dtype=np.float32),
        counts=np.array([pixels.shape[0]]),
        spreads=np.array([spread]),
    )

    mask = _track_masks(image, palette, calibration, DEFAULT)[0]
    covered = mask.sum(axis=1) > width * 0.5

    assert covered[:height].mean() > 0.9, (
        f"only {covered[:height].mean():.0%} of the ramp was matched; "
        "the radius is still clipping the gradient"
    )


def test_flat_tiles_keep_the_fixed_tolerance() -> None:
    """A tight cluster must not widen the radius and let bloom in."""
    from dropscore.tiles import Palette, _track_masks

    calibration = _plain_calibration(strike_y=80, width=40)
    image = np.zeros((120, 40, 3), dtype=np.uint8)
    image[:80, :] = (200, 90, 120)

    tight = Palette(
        background=np.array([0.0, 128.0, 128.0], dtype=np.float32),
        colors=np.array([[100.0, 150.0, 100.0]], dtype=np.float32),
        counts=np.array([1000]),
        spreads=np.array([1.0]),  # flat: pixels sit on top of the colour
    )
    none = Palette(
        background=tight.background, colors=tight.colors, counts=tight.counts
    )

    a = _track_masks(image, tight, calibration, DEFAULT)[0]
    b = _track_masks(image, none, calibration, DEFAULT)[0]

    assert np.array_equal(a, b), "a tight cluster changed the acceptance radius"
