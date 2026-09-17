"""Stage 4 scored against stage 2's ground truth.

The central test asks the only question that matters: at a given instant, does the
detector report exactly the set of pitches the renderer drew?
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dropscore.calibrate import Calibration
from dropscore.config import DEFAULT
from dropscore.keyboard import KeyboardLayout
from dropscore.notes import Note, NoteSequence
from dropscore.synth import RenderSpec, SynthRenderer, generate, get_theme
from dropscore.synth.themes import THEMES
from dropscore.tiles import (
    TileError,
    _cluster,
    _looks_outlined,
    detect_in_frame,
    discover_palette,
)
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


def test_an_outlined_tile_on_a_black_key_is_detected() -> None:
    """A black key's tile fills its span almost exactly, so its edges sit on
    the span's boundary. Rounding the boundary inward dropped the column the
    right-hand stroke was drawn in: the blob stopped reading as hollow, was
    judged by the solidity rule meant for filled tiles, and one key went
    undetected for a whole clip -- 14 notes, with nothing else missed.
    """
    black = 78
    sequence = NoteSequence.of([Note(onset=0.6, pitch=black, duration=1.2)])
    renderer = _renderer("paper", sequence)
    calibration = _truth_calibration(renderer)
    frame = _frame_at(renderer, 1.0)
    palette = discover_palette(_frames(renderer), calibration)

    found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration)}
    assert black in found, f"black key missed; found {sorted(found)}"


def test_a_held_black_key_beside_a_later_white_one_leaves_both_intact() -> None:
    """An F#4 held from earlier, a G4 starting later, their tiles touching.

    A black key's lane overlaps its white neighbour's columns, so the two form
    one blob. The black key was not claimed -- a black key is only claimed by
    span when no white neighbour is, to keep two merged white tiles from
    reading as a phantom accidental -- and the G4, judged over its full width,
    took the F#4's tile for its own and read as reaching the strike line. Its
    lower edge never fell, and on a clip of held chords it vanished twice.
    """
    renderer = _renderer()
    calibration = _truth_calibration(renderer)
    layout = renderer.layout
    height, width = renderer.spec.height, renderer.spec.width
    import cv2  # noqa: PLC0415

    image = np.full((height, width, 3), 12, dtype=np.uint8)
    colour = (110, 215, 245)
    black_left, black_right = layout.key_span(66)
    white_left, white_right = layout.key_span(67)
    cv2.rectangle(image, (int(black_left), 0), (int(black_right), 400), colour, -1)
    cv2.rectangle(image, (int(white_left), 0), (int(white_right), 180), colour, -1)

    painted = Frame(0, 0.0, image, 1.0)
    blank = np.full((height, width, 3), 12, dtype=np.uint8)
    frames = [Frame(i, i / SPEC.fps, blank, 1.0) for i in range(1, 6)] + [painted]
    palette = discover_palette(frames, calibration)

    tiles = detect_in_frame(painted, palette, calibration)
    by_pitch = {t.pitch: t for t in tiles}
    assert 66 in by_pitch, f"the held black key was lost; found {sorted(by_pitch)}"
    assert by_pitch[67].bottom == pytest.approx(180, abs=4), (
        f"the white key took the black key's tile: bottom {by_pitch[67].bottom}"
    )


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


def test_a_glow_halo_is_folded_into_its_tile_colour() -> None:
    """A halo is the tile's colour blended toward the background, so it lies
    on the line between them. Hue decided this before, with a hard chroma
    cutoff the halo sat 0.2 below, and the glow became a second voice."""
    from dropscore.tiles import _fold_blends  # noqa: PLC0415

    background = np.array([6.0, 128.0, 126.0])
    tile = np.array([221.0, 126.0, 183.0])
    halo = background + 0.24 * (tile - background)  # chroma just under 12
    colors = np.stack([tile, halo])
    counts = np.array([17532, 2706])

    kept, kept_counts = _fold_blends(colors, counts, background, DEFAULT.tiles)
    assert len(kept) == 1
    assert kept_counts[0] == 17532 + 2706


def test_two_grey_voices_are_not_folded_together() -> None:
    """Every grey lies on the line from black to white, so the blend test on
    its own would merge two voices told apart only by lightness."""
    from dropscore.tiles import _fold_blends  # noqa: PLC0415

    background = np.array([5.0, 128.0, 128.0])
    white, grey = np.array([240.0, 128.0, 128.0]), np.array([120.0, 128.0, 128.0])
    kept, _ = _fold_blends(np.stack([white, grey]), np.array([9000, 8000]), background, DEFAULT.tiles)
    assert len(kept) == 2


def test_a_second_hand_of_another_hue_is_not_folded() -> None:
    from dropscore.tiles import _fold_blends  # noqa: PLC0415

    background = np.array([18.0, 128.0, 126.0])
    green, blue = np.array([170.0, 80.0, 170.0]), np.array([140.0, 150.0, 70.0])
    kept, _ = _fold_blends(np.stack([green, blue]), np.array([9000, 7000]), background, DEFAULT.tiles)
    assert len(kept) == 2


def test_the_same_frames_give_the_same_palette_every_time() -> None:
    """Clustering starts from random centres, and unseeded two identical calls
    disagreed -- colours moved by up to 3 Lab units, and one clip's palette
    held three colours on some runs and two on others. Nothing downstream can
    be compared run to run if this can change on its own."""
    renderer = _renderer("aurora")
    calibration = _truth_calibration(renderer)
    frames = _frames(renderer)

    first = discover_palette(frames, calibration).colors
    for _ in range(3):
        assert np.array_equal(discover_palette(frames, calibration).colors, first)


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

    # A theme with effects is adversarial on purpose: sparks are the tiles' own
    # colour, so some get through and detecting nothing spurious is not a bar
    # this pipeline currently clears. Bounded rather than waived, so a change
    # that floods detections still fails here — removing the solidity filter
    # takes the same clip from 67 spurious notes to 247.
    allowance = 8 if get_theme(theme).particles else 0

    for t in (3.0, 5.5, 8.0):
        frame = _frame_at(renderer, t)
        found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration)}
        expected = _expected_pitches(renderer, frame.time)
        assert _expected_pitches(renderer, frame.time, min_height=8) <= found, f"{theme} t={t}"
        spurious = found - expected
        assert len(spurious) <= allowance, (
            f"{theme} t={t}: {len(spurious)} pitches detected with no tile "
            f"({sorted(spurious)})"
        )


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


def test_widening_stops_short_of_a_neighbouring_colour() -> None:
    """A colour may not widen so far that it claims another voice's pixels.

    Two shades of one colour, discovered from particle effects rather than
    from two hands, spread as widely as a genuine gradient does and would
    widen just as far — measured on a real video, from 22 to 31, taking the
    detected blobs from 63 a frame to 91. Crowded colours cap each other.
    """
    from dropscore.tiles import Palette, _to_lab, _track_masks

    calibration = _plain_calibration(strike_y=60, width=40)
    image = np.zeros((90, 40, 3), dtype=np.uint8)
    image[:60, :] = (120, 160, 120)

    # Anchor to what the pixel actually converts to, and put the colour just
    # beyond the fixed tolerance: only a widened radius can reach it.
    pixel = _to_lab(image[:60])[0, 0].astype(np.float64)
    near = pixel + np.array([0.0, 25.0, 0.0])

    def palette(gap: float) -> Palette:
        return Palette(
            background=np.array([0.0, 128.0, 128.0], dtype=np.float32),
            colors=np.array([near, near + [0.0, gap, 0.0]], dtype=np.float32),
            counts=np.array([1000, 1000]),
            spreads=np.array([15.0, 15.0]),  # both loose enough to want ~37
        )

    crowded = sum(int(m.sum()) for m in _track_masks(image, palette(24.0), calibration, DEFAULT))
    spacious = sum(int(m.sum()) for m in _track_masks(image, palette(120.0), calibration, DEFAULT))

    assert spacious > 0, "the widened radius should reach a pixel 25 away"
    assert crowded == 0, (
        f"a colour 24 from its neighbour still matched {crowded} pixels 25 away; "
        "the cap is not limiting how far a crowded palette may widen"
    )


def test_two_filled_tiles_with_a_gap_are_not_read_as_one_outline() -> None:
    """A box holding two tiles of one key, one above the other.

    Read down the whole height, its side columns are full for both tiles and
    the gap between them reads as an empty middle -- an outline, which is kept
    whole as a single note. Sides and middle have to be read over the same
    rows, and there the sides are as empty as the middle is.
    """
    strip = np.zeros((100, 12), dtype=np.uint8)
    strip[:32] = 1
    strip[68:] = 1

    assert not _looks_outlined(strip, DEFAULT.tiles)


def test_a_hollow_tile_is_still_read_as_an_outline() -> None:
    strip = np.zeros((100, 12), dtype=np.uint8)
    strip[:, :2] = 1
    strip[:, -2:] = 1
    strip[:3] = 1
    strip[-3:] = 1

    assert _looks_outlined(strip, DEFAULT.tiles)


def test_two_thin_tiles_side_by_side_are_not_read_as_one_outline() -> None:
    """Two narrow tiles on neighbouring keys, merged into one box.

    Stroked at either side and empty between, the box reads as an outline --
    but nothing closes it top or bottom, which every real outline has.
    """
    strip = np.zeros((222, 23), dtype=np.uint8)
    strip[:, 2:4] = 1
    strip[:, 22] = 1

    assert not _looks_outlined(strip, DEFAULT.tiles)


def test_an_outline_clipped_at_the_top_is_still_an_outline() -> None:
    """A tile running past the top of the fall area keeps only its lower stroke."""
    strip = np.zeros((100, 12), dtype=np.uint8)
    strip[:, :2] = 1
    strip[:, -2:] = 1
    strip[-3:] = 1

    assert _looks_outlined(strip, DEFAULT.tiles)


def test_a_colour_and_a_near_copy_of_it_are_one_voice() -> None:
    """Two greys 1.1 apart are one voice; two grey hands stand far apart."""
    grey = np.full((400, 3), (147.0, 127.0, 129.0), dtype=np.float32)
    near = np.full((100, 3), (148.1, 127.0, 129.0), dtype=np.float32)
    other = np.full((300, 3), (231.0, 127.0, 129.0), dtype=np.float32)

    colors, counts = _cluster(
        np.concatenate([grey, near, other]),
        DEFAULT.tiles.max_palettes,
        DEFAULT.tiles.lightness_weight,
        DEFAULT.tiles.merge_distance,
        np.array([14.0, 128.0, 129.0]),
        DEFAULT.tiles.duplicate_distance,
    )

    assert len(colors) == 2, f"expected two voices, got {np.round(colors, 1).tolist()}"
    assert sorted(counts.tolist()) == [300, 500]


def test_a_bright_stroke_is_kept_over_the_saturated_glow_of_its_own_hue() -> None:
    """Tiles stroked near-white over a violet fill that glows violet.

    The fill and the glow are one colour, so a mask of it bridges neighbouring
    tiles through their glow. The stroke is the tile. Judged by weighted Lab
    distance from the background the fill won, 72.6 to 71.3; the stroke is the
    brighter, and glow is always the dimmer against a dark ground. These are
    the clusters measured on that capture.
    """
    glow = np.full((822, 3), (36.1, 150.3, 93.0), dtype=np.float32)
    fill = np.full((536, 3), (94.2, 168.3, 65.8), dtype=np.float32)
    stroke = np.full((212, 3), (202.6, 138.1, 107.1), dtype=np.float32)

    colors, _ = _cluster(
        np.concatenate([glow, fill, stroke]),
        DEFAULT.tiles.max_palettes,
        DEFAULT.tiles.lightness_weight,
        DEFAULT.tiles.merge_distance,
        np.array([3.0, 132.0, 120.0]),
        DEFAULT.tiles.duplicate_distance,
        DEFAULT.tiles.source_near_tie,
    )

    assert len(colors) == 1
    assert colors[0][0] > 150, f"kept {np.round(colors[0], 1).tolist()}, not the stroke"


def test_a_vivid_tile_is_kept_over_its_own_brighter_highlight() -> None:
    """Brighter is only the tiebreak. A vivid green tile and the paler, brighter
    highlight on it are far apart in distance from the background, and choosing
    the brighter kept the highlight, whose pixels spread 36 from it where the
    tile's own spread 1.2. These are that theme's clusters."""
    body = np.full((1138, 3), (136.7, 187.8, 62.2), dtype=np.float32)
    highlight = np.full((137, 3), (164.3, 167.1, 89.8), dtype=np.float32)

    colors, _ = _cluster(
        np.concatenate([body, highlight]),
        DEFAULT.tiles.max_palettes,
        DEFAULT.tiles.lightness_weight,
        DEFAULT.tiles.merge_distance,
        np.array([2.0, 128.0, 126.0]),
        DEFAULT.tiles.duplicate_distance,
        DEFAULT.tiles.source_near_tie,
    )

    assert len(colors) == 1
    assert colors[0][0] == pytest.approx(136.7, abs=1.0), f"kept {np.round(colors[0], 1).tolist()}"


def test_an_outlined_white_tile_does_not_claim_the_black_key_its_stroke_touches() -> None:
    """An outlined D4 drawn a pixel wider than its key.

    Its left stroke lies in the few columns of C#4's lane that fall inside the
    blob. Judged over only those columns, the lane was solid on every row where
    the D4's hollow middle was empty, and a phantom C#4 was read.
    """
    from dropscore.keyboard import KeyboardLayout  # noqa: PLC0415
    from dropscore.tiles import _with_hidden_black_keys  # noqa: PLC0415

    layout = replace(KeyboardLayout(width=1274.0), black_offsets=(-0.13, 0.19, -0.23, 0.0, 0.23))
    calibration = Calibration(
        layout=layout, strike_y=366, keybed_bottom=500, white_width=layout.white_width, confidence=1.0
    )
    # As measured: the blob 27 pixels wide, starting 5 pixels inside C#4's
    # lane, stroked 4 pixels thick.
    _, sharp_right = layout.key_span(61)
    x0 = int(round(sharp_right)) - 5
    x1 = x0 + 27
    y0, y1 = 100, 122
    mask = np.zeros((366, 1274), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    mask[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4] = 0

    claimed = _with_hidden_black_keys(mask, (x0, y0, x1, y1), [62], calibration, DEFAULT)
    assert claimed == [62]
