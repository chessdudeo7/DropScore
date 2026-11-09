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
    measure_lanes,
    straighten_lanes,
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


def test_a_halo_is_folded_even_when_it_is_the_commonest_colour() -> None:
    """A glow surrounds every tile, so where the tile colours are split the
    pooled halo outnumbers each of them. Deciding which of a pair was the
    blend by their counts never offered that halo as a child, and it survived
    as a voice: on a theme tinting tiles by pitch it took 48% of the sampled
    pixels and drew a second, wider blob around every tile."""
    from dropscore.tiles import _fold_blends  # noqa: PLC0415

    background = np.array([0.0, 128.0, 128.0])
    tile = np.array([198.0, 94.0, 121.0])
    halo = background + 0.32 * (tile - background)
    colors = np.stack([halo, tile])
    counts = np.array([104374, 47085])

    kept, kept_counts = _fold_blends(colors, counts, background, DEFAULT.tiles)
    assert len(kept) == 1
    assert kept[0] == pytest.approx(tile)
    assert kept_counts[0] == 104374 + 47085


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


def test_two_shades_joined_by_a_ramp_are_one_track() -> None:
    """A tile shaded through more hue than the merge tolerance clusters into
    two colours either side of its ramp. On one recording a green ran to
    yellow-green, 51 degrees apart, and split every tile across two masks.
    Its pixels run unbroken between the two; two hands' colours leave a gap.
    """
    from dropscore.tiles import _merge_ramps  # noqa: PLC0415

    rng = np.random.default_rng(3)
    green, yellow_green = np.array([200.0, 92.0, 122.0]), np.array([185.0, 87.0, 160.0])
    colors = np.stack([green, yellow_green])

    # One ramp: pixels spread evenly along the segment between the two.
    along = rng.uniform(-0.1, 1.1, size=9000)[:, None]
    ramp = green + along * (yellow_green - green) + rng.normal(0.0, 2.0, size=(9000, 3))
    assert _merge_ramps(colors, ramp, DEFAULT.tiles).tolist() == [0, 0]

    # Two voices: pixels sit at the ends and nowhere between.
    ends = np.concatenate([
        green + rng.normal(0.0, 2.0, size=(5000, 3)),
        yellow_green + rng.normal(0.0, 2.0, size=(4000, 3)),
    ])
    assert _merge_ramps(colors, ends, DEFAULT.tiles).tolist() == [0, 1]


def test_shades_of_one_track_share_a_mask_without_widening() -> None:
    """Two shades feeding one track are one mask, and neither shade's reach
    grows: merged into one colour at the ramp's middle, the radius widened in
    every direction and took the tile's halo on the keys either side of it --
    ten notes on one recording that were never played."""
    import cv2  # noqa: PLC0415

    from dropscore.tiles import Palette, _to_lab, _track_masks  # noqa: PLC0415

    weight = np.array([DEFAULT.tiles.lightness_weight, 1.0, 1.0])
    shades = np.array([[200.0, 92.0, 122.0], [185.0, 87.0, 160.0]])
    middle = shades.mean(axis=0)
    # Off the ramp, 30 out from its middle in weighted Lab: within a widened
    # radius of the middle, past the tolerance of either shade and of the
    # line between them.
    along = (shades[1] - shades[0]) * weight
    across = np.cross(along, [1.0, 0.0, 0.0])
    halo = middle + 30.0 * across / np.linalg.norm(across) / weight
    mid_ramp = middle + 15.0 * across / np.linalg.norm(across) / weight
    lab = np.stack([shades[0], shades[1], halo, mid_ramp]).clip(0, 255).astype(np.uint8)
    strip = cv2.cvtColor(lab[None], cv2.COLOR_LAB2BGR)[0]
    rows = 40
    image = np.repeat(strip[None], rows + 10, axis=0)
    read = _to_lab(image[:1])[0]  # the colours as detection will see them

    calibration = _plain_calibration(strike_y=rows, width=4)
    background = np.array([0.0, 128.0, 128.0])
    spreads = np.array([16.0, 16.0])
    paired = Palette(background, read[:2], np.array([1, 1]), spreads, tracks=np.array([0, 0]))
    (mask,) = _track_masks(image, paired, calibration, DEFAULT)
    assert mask[:, 0].all() and mask[:, 1].all()
    assert not mask[:, 2].any()
    # Halfway along the ramp and a little off it: past where the two shades
    # meet, and the tile's all the same.
    assert mask[:, 3].all()

    merged = Palette(background, read[:2].mean(axis=0)[None], np.array([2]), spreads[:1])
    (wide,) = _track_masks(image, merged, calibration, DEFAULT)
    assert wide[:, 2].all()  # what the one-colour merge did


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


def test_rounded_repeats_with_no_gap_split_at_their_pinch() -> None:
    """Two rounded tiles drawn flush meet only in a narrower waist, never a gap."""
    from dataclasses import replace  # noqa: PLC0415

    from dropscore.tiles import _split_vertically  # noqa: PLC0415

    mask = np.zeros((200, 40), np.uint8)
    mask[20:180, 5:27] = 255
    # The pinch measured on a real render: 22 columns narrowing to 16 for four
    # rows, every row still far above row_fill_ratio.
    mask[98:100, 5:7] = mask[98:100, 25:27] = 0
    mask[100:104, 5:9] = mask[100:104, 23:27] = 0
    mask[104:106, 5:7] = mask[104:106, 25:27] = 0
    # Each tile's own rounded caps must not be cut.
    mask[20:23, 5:8] = mask[20:23, 24:27] = mask[177:180, 5:8] = mask[177:180, 24:27] = 0

    spans = _split_vertically(mask, (5, 20, 27, 180), DEFAULT)
    assert len(spans) == 2 and spans[0][1] <= spans[1][0]
    assert 98 <= spans[0][1] <= 106

    off = replace(DEFAULT, tiles=replace(DEFAULT.tiles, waist_ratio=0.0))
    assert len(_split_vertically(mask, (5, 20, 27, 180), off)) == 1


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


def test_a_colour_may_widen_away_from_the_neighbour_crowding_it() -> None:
    """Only the neighbour a pixel lies toward should bound the reach to it.

    Bounded by its nearest neighbour in any direction, a colour with company
    on one side was held just as close on the other, where there was nothing
    for twice the distance. Three colours sampled from a ramp did that: the
    middle sat 43 from one side and 86 from the other, half of 43 held it to
    22, and the stretch below it belonged to no colour at all. Tiles coloured
    from there matched nothing -- on a Liszt etude a whole band of the
    keyboard, one of its keys found in 935 frames where it should hold 1560.
    """
    from dropscore.tiles import Palette, _to_lab, _track_masks

    calibration = _plain_calibration(strike_y=60, width=40)
    image = np.zeros((90, 40, 3), dtype=np.uint8)
    image[:60, :] = (255, 255, 255)

    pixel = _to_lab(image[:60])[0, 0].astype(np.float64)
    middle = pixel + [0.0, 36.0, 0.0]          # the pixel is 36 from this one
    crowding = middle + [0.0, 43.0, 0.0]       # 43 away, on the far side
    distant = pixel - [0.0, 50.0, 0.0]         # 86 from the middle, past the pixel

    palette = Palette(
        background=np.array([0.0, 128.0, 128.0], dtype=np.float32),
        colors=np.array([middle, crowding, distant], dtype=np.float32),
        counts=np.array([1000, 1000, 1000]),
        spreads=np.array([17.0, 17.0, 17.0]),  # loose enough to want ~42
    )

    matched = int(_track_masks(image, palette, calibration, DEFAULT)[0].sum())
    assert matched > 0, (
        "a pixel 36 from its colour, with 86 of empty space in that direction, "
        "was refused because another colour sat 43 away on the other side"
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


def test_a_black_key_zigzagging_with_its_white_neighbour_keeps_each_of_its_tiles() -> None:
    """E D# E D# E, drawn with black tiles as wide as their lane.

    Each D# touches the Es above and below it, so the five tiles are one blob.
    The D#'s lane lies two thirds over the E's columns, and judged over all of
    it the lane was full from the top E to the bottom one: one D# in place of
    two, timed by the last E's lower edge.
    """
    from dropscore.keyboard import KeyboardLayout  # noqa: PLC0415
    from dropscore.tiles import _exposed_span, _split_vertically  # noqa: PLC0415

    layout = replace(KeyboardLayout(width=1272.0), black_offsets=(-0.13, 0.19, -0.23, 0.0, 0.23))
    calibration = Calibration(
        layout=layout, strike_y=354, keybed_bottom=500, white_width=layout.white_width, confidence=1.0
    )
    mask = np.zeros((354, 1272), dtype=np.uint8)
    e_left, e_right = layout.key_span(76)
    sharp_left, sharp_right = layout.key_span(75)
    for top in (40, 85, 130):
        mask[top : top + 24, int(e_left) + 1 : int(e_right)] = 1
    for top in (63, 108):
        mask[top : top + 24, int(sharp_left) : int(sharp_right) + 1] = 1

    lo, hi = _exposed_span(75, [75, 76], calibration)
    x0, x1 = int(round(lo)), int(round(hi))
    spans = _split_vertically(mask, (x0, 40, x1, 154), DEFAULT)
    assert len(spans) == 2, f"D#5 read as {spans}"


# ── colours that only draw lines ─────────────────────────────────────


def _grid_and_tiles_frames(grey_tiles: bool = False) -> tuple[list[Frame], Calibration]:
    """Black fall area, solid purple tiles, thin grey lines like a visualizer's grid."""
    import cv2  # noqa: PLC0415

    layout = KeyboardLayout(x0=0.0, width=1280.0)
    calibration = Calibration(
        layout=layout, strike_y=400, keybed_bottom=480,
        white_width=layout.white_width, confidence=1.0,
    )
    frames = []
    for i in range(24):
        image = np.zeros((480, 1280, 3), np.uint8)
        for x in range(60, 1280, 172):
            cv2.line(image, (x, 0), (x, 399), (90, 90, 90), 1)
        cv2.line(image, (0, 100 + i * 5), (1279, 100 + i * 5), (90, 90, 90), 1)
        for k in range(6):
            left = 100 + k * 190
            color = (90, 90, 90) if grey_tiles and k % 2 else (255, 60, 200)
            cv2.rectangle(image, (left, 50 + i * 8), (left + 20, 160 + i * 8), color, -1)
        frames.append(Frame(i, i / 30, image, 1.0))
    return frames, calibration


def test_a_grey_that_only_draws_lines_is_not_a_voice() -> None:
    frames, calibration = _grid_and_tiles_frames()
    palette = discover_palette(frames, calibration, DEFAULT)

    assert palette.track_count == 1
    assert np.hypot(*(palette.colors[0, 1:] - 128.0)) > 20  # the purple


def test_grey_tiles_are_kept() -> None:
    frames, calibration = _grid_and_tiles_frames(grey_tiles=True)
    palette = discover_palette(frames, calibration, DEFAULT)

    greys = np.hypot(palette.colors[:, 1] - 128.0, palette.colors[:, 2] - 128.0) < 8
    assert greys.any()


def _spread(distances, **overrides):
    from dropscore.tiles import _member_spread

    cfg = replace(DEFAULT.tiles, **overrides)
    return _member_spread(np.asarray(distances, dtype=float), cfg)


def test_spread_of_a_flat_colour_ignores_the_smoke_around_it():
    # 41% of the members in one tight band, then a gap, then smoke straggling
    # evenly to 44: a flat tile colour under a particle cloud, as measured on
    # the glow render. Read from the median, the spread would widen the radius
    # until the cloud counted as tile.
    rng = np.random.default_rng(0)
    band = rng.uniform(5.0, 6.5, 410)
    tail = rng.uniform(12.0, 44.0, 590)
    distances = np.concatenate([band, tail])
    assert np.median(distances) > 14.0
    assert _spread(distances) < 7.0


def test_spread_ignores_sparks_that_fill_two_thirds_of_a_short_sample():
    # The first 68 seconds of a sparks render: 33% of the members in the band,
    # the sparks set apart from it, the median out among them at 12.
    rng = np.random.default_rng(4)
    band = rng.uniform(2.0, 5.5, 330)
    sparks = np.concatenate([rng.uniform(10.0, 13.0, 300), rng.uniform(16.0, 44.0, 370)])
    distances = np.concatenate([band, sparks])
    assert np.median(distances) > 10.0
    assert _spread(distances) < 5.0
    assert _spread(distances, spread_core_share=0.35) > 10.0


def test_spread_of_a_gradient_keeps_its_median():
    # A ramp spreads its pixels evenly: no band holds 40% of them.
    distances = np.random.default_rng(1).uniform(0.0, 30.0, 1000)
    assert _spread(distances) == pytest.approx(np.median(distances))


def test_spread_keeps_the_median_when_it_lies_in_the_band():
    rng = np.random.default_rng(2)
    distances = np.concatenate([rng.uniform(9.0, 10.5, 600), rng.uniform(0.0, 3.0, 400)])
    assert _spread(distances) == pytest.approx(np.median(distances))


def test_spread_keeps_the_median_when_the_variation_carries_on_past_the_band():
    # Für Elise's second colour: 41% of its members within 4 of each other, but
    # the rest continue at a steady density with no gap. That is one colour
    # drawn with real variation; read from the band it lost 8 of 104 notes.
    rng = np.random.default_rng(3)
    distances = np.concatenate([rng.uniform(2.2, 4.0, 410), rng.uniform(4.0, 30.0, 590)])
    assert np.median(distances) > 7.0
    assert _spread(distances) == pytest.approx(np.median(distances))


def test_spread_ignores_smoke_on_a_30fps_encode():
    # The same glow render at 30fps: the encoder softens the tile, so its band
    # holds 0.38 of the members and the next 4 units 0.11 of the band. Read
    # from the median there, the radius took in the smoke: 80 phantom notes.
    rng = np.random.default_rng(4)
    band = rng.uniform(3.0, 6.5, 380)
    shoulder = rng.uniform(7.2, 10.8, 42)
    tail = rng.uniform(11.0, 44.0, 578)
    distances = np.concatenate([band, shoulder, tail])
    assert np.median(distances) > 14.0
    assert _spread(distances) < 7.0


def test_spread_keeps_the_median_when_the_band_is_the_far_end():
    # Distances are only gathered out to a window, so a colour can pile up at
    # its edge. That is where the counting stopped, not a flat colour's core.
    rng = np.random.default_rng(5)
    distances = np.concatenate([rng.uniform(40.0, 43.5, 400), rng.uniform(0.0, 30.0, 600)])
    assert _spread(distances) == pytest.approx(np.median(distances))


def _leaning(image: np.ndarray, strike_y: int, a: float, b: float) -> np.ndarray:
    """The picture above the strike line sheared so a lane that meets the
    strike line at column x runs at dx/dy = a + b * x, as a board filmed at
    an angle does."""
    import cv2  # noqa: PLC0415

    height, width = image.shape[:2]
    columns = np.arange(width, dtype=np.float32)[None, :]
    rows = np.arange(strike_y, dtype=np.float32)[:, None]
    # A leaning lane's column x' at row y came from column x at the strike
    # line: x' = x - (strike_y - y)(a + b x), so x = (x' + (strike_y - y) a)
    # / (1 - (strike_y - y) b).
    map_x = (columns + (strike_y - rows) * a) / (1.0 - (strike_y - rows) * b)
    map_y = np.broadcast_to(rows, (strike_y, width)).astype(np.float32)
    out = image.copy()
    out[:strike_y] = cv2.remap(
        image[:strike_y], map_x.astype(np.float32), map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return out


def _lean_fixture(a: float, b: float):
    # Long notes right across the board, so every lane's lean is measurable
    # and the fit has the width to tell a ramp from a constant.
    rng = np.random.default_rng(3)
    notes = [
        Note(onset=i * 0.5, pitch=int(rng.integers(24, 106)), duration=1.5 + (i % 3))
        for i in range(60)
    ]
    renderer = _renderer("classic", NoteSequence.of(notes))
    calibration = _truth_calibration(renderer)
    straight = _frames(renderer, 24)
    leaning = [
        Frame(f.index, f.time, _leaning(f.image, calibration.strike_y, a, b), 1.0)
        for f in straight
    ]
    palette = discover_palette(leaning, calibration)
    return calibration, straight, leaning, palette


def test_leaning_lanes_are_measured_and_straightened() -> None:
    """A board filmed at an angle: lanes lean out at both ends, so a tall
    tile's bounding box reaches over the key beside it and a phantom is read
    there. Measured from the sample frames and corrected, detection on the
    leaning picture reads the same keys as on the straight one."""
    a, b = -0.045, 9.0e-5  # leans left at the left edge, right at the right
    calibration, straight, leaning, palette = _lean_fixture(a, b)

    measured_a, measured_b = measure_lanes(leaning, palette, calibration)
    width = straight[0].image.shape[1]
    assert measured_a == pytest.approx(a, abs=0.006)
    assert measured_a + measured_b * width == pytest.approx(a + b * width, abs=0.006)

    corrected = replace(calibration, lane_slope=(measured_a, measured_b))
    wrong = right = 0
    for before, after in zip(straight, leaning):
        truth = {t.pitch for t in detect_in_frame(before, palette, calibration)}
        naive = {t.pitch for t in detect_in_frame(after, palette, calibration)}
        fixed = {t.pitch for t in detect_in_frame(after, palette, corrected)}
        wrong += len(naive ^ truth)
        right += len(fixed ^ truth)
    assert wrong > 10, "the lean should have hurt the uncorrected read"
    assert right <= wrong // 10


def test_straight_lanes_are_left_alone() -> None:
    """A screen recording measures a hundredth of a pixel of lean and must
    keep its detections to the pixel."""
    calibration, straight, _, _ = _lean_fixture(0.0, 0.0)
    palette = discover_palette(straight, calibration)
    assert measure_lanes(straight, palette, calibration) == (0.0, 0.0)
    assert straighten_lanes(straight, palette, calibration) == calibration


def test_a_black_key_struck_with_its_white_neighbour_is_found() -> None:
    """C#4 and D4 together: one blob a key and a third wide. The old rules
    gave it all to D4, since the black lane had no row of its own; the blob's
    far edge, straight down C#4's outer edge, says the black key is there."""
    sequence = NoteSequence.of([Note(onset=0.0, pitch=p, duration=1.0) for p in (61, 62)])
    renderer = _renderer(sequence=sequence)
    calibration = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), calibration)

    frame = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration)}
    assert found == {61, 62}, found

    off = replace(DEFAULT, tiles=replace(DEFAULT.tiles, black_edge_px=0.0))
    found = {tile.pitch for tile in detect_in_frame(frame, palette, calibration, off)}
    assert found == {62}, found


def test_a_white_tile_off_its_fitted_key_is_not_read_as_the_black_beside_it() -> None:
    """The keystone copy's board was fitted a fifth of a key left of its tiles at
    one end, so each C2's centre sat nearer C#2 and every one was read as C#2.
    A tile as wide as that cannot be a black key's."""
    sequence = NoteSequence.of(
        [Note(onset=0.0, pitch=60, duration=1.0), Note(onset=2.0, pitch=66, duration=1.0)]
    )
    renderer = _renderer(sequence=sequence)
    truth = _truth_calibration(renderer)
    palette = discover_palette(_frames(renderer), truth)
    shifted = replace(truth, layout=replace(truth.layout, x0=truth.layout.x0 - 0.3 * truth.white_width))

    white = _frame_at(renderer, renderer.notes[0].onset - 0.5)
    black = _frame_at(renderer, renderer.notes[1].onset - 0.5)
    def read(frame, near, calibration, config=DEFAULT):
        found = detect_in_frame(frame, palette, calibration, config)
        return {t.pitch for t in found if abs(t.pitch - near) <= 1}

    assert read(white, 60, shifted) == {60}
    assert read(black, 66, truth) == {66}

    off = replace(DEFAULT, tiles=replace(DEFAULT.tiles, black_tile_max_ratio=99.0))
    assert read(white, 60, shifted, off) == {61}


def test_a_seam_blurred_into_the_mask_is_found_in_the_colour() -> None:
    """Recompressed, a copy widens each colour's radius until the rounded
    corners at a seam pass for tile and the mask is full width throughout;
    the colour still turns at both edges for a row or two."""
    from dropscore.config import DEFAULT  # noqa: PLC0415
    from dropscore.tiles import _graded_seams  # noqa: PLC0415

    distance = np.full((120, 24), 7.0, dtype=np.float32)
    distance[:, :2] = distance[:, -2:] = 60.0  # the halo the radius took in
    distance[70:72, 2:5] = 18.0  # both corners turn at the seam
    distance[70:72, 19:22] = 20.0
    assert _graded_seams(distance, 0, 120, DEFAULT.tiles, 100) == [(100, 170), (171, 220)]

    one_side = distance.copy()
    one_side[70:72, 19:22] = 7.0  # a neighbour's edge beside one side only
    assert _graded_seams(one_side, 0, 120, DEFAULT.tiles, 100) == [(100, 220)]


def test_colours_are_found_over_the_board_not_the_desktop() -> None:
    """A video in a window beside a desktop: over the whole frame the desktop
    outweighed the video's background, which was then taken for a tile colour."""
    from dataclasses import replace  # noqa: PLC0415

    from dropscore.calibrate import calibrate  # noqa: PLC0415
    from dropscore.config import DEFAULT  # noqa: PLC0415
    from dropscore.tiles import discover_palette  # noqa: PLC0415
    from tests.test_calibrate import _frames, _on_a_desktop, _render  # noqa: PLC0415

    renderer = _render()
    frames = [replace(f, image=_on_a_desktop(f.image)) for f in _frames(renderer)]
    calibration = calibrate(frames)
    found = discover_palette(frames, calibration)
    assert found.background[0] < 20  # the video's dark ground, not the desktop
    whole = replace(DEFAULT, tiles=replace(DEFAULT.tiles, palette_in_board=False))
    assert discover_palette(frames, calibration, whole).background[0] > 20
