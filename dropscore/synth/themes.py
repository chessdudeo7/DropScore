"""Visual presets for the synthetic renderer.

The point of having several is that stages 3-5 must not overfit to one channel's
look. Each theme varies something the vision code could accidentally depend on:
palette, tile shape, glow, keybed proportions, lane separators, keyboard range.
A change that only works on ``classic`` will show up as a regression on the rest.

Colours are RGB here because that is how they are read and written; the renderer
converts to OpenCV's BGR at the point of drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RGB = tuple[int, int, int]

TileStyle = Literal["flat", "rounded", "outline", "gradient"]


@dataclass(frozen=True)
class Theme:
    """Everything about how a synthetic clip looks."""

    name: str

    # Background above the keyboard.
    background: RGB = (7, 8, 13)
    lane_separators: bool = False
    lane_color: RGB = (24, 26, 34)

    # Falling tiles, by hand.
    right_color: RGB = (245, 215, 110)
    left_color: RGB = (122, 162, 255)
    tile_style: TileStyle = "flat"
    corner_radius: float = 0.0  # as a fraction of tile width
    tile_gap: float = 0.10  # horizontal inset, as a fraction of key width
    glow: float = 0.0  # 0 disables; ~0.5 is a strong bloom

    # Keybed.
    keybed_ratio: float = 0.22  # keybed height as a fraction of frame height
    white_key_color: RGB = (236, 236, 240)
    black_key_color: RGB = (16, 17, 22)
    key_edge_color: RGB = (120, 122, 132)
    keybed_shadow: bool = True  # dark band where the keybed meets the fall area

    # Struck keys light up in the hand's colour, dimmed by this factor.
    highlight_strength: float = 0.85

    # Everything below models what a real capture has and a clean render does
    # not. Each of these hid a bug that only a real video could reach.

    # Dark space below the keyboard, as a fraction of frame height. Real
    # captures are rarely cropped to the keybed, and what sits in that space
    # is where a caption goes.
    bottom_margin: float = 0.0

    # Text burned into the bottom margin, the way an arranger credits a video.
    # White-on-black text has a higher mean row structure than a keyboard does,
    # so a keybed search that ranks bands by mean picks the caption instead.
    caption: str = ""

    # Struck keys bloom past white. A renderer that only tints a key keeps it
    # within the palette; a real one blows it out, and a boundary sample there
    # comes back brighter than any unplayed white key, which drags a two-means
    # black/white split onto the wrong axis.
    highlight_bloom: float = 0.0

    # Hands over the lower keybed, occluding keys and adding their own
    # structure to rows the keybed search has to reject.
    hands: bool = False

    # A bright line at the top of the keybed. Some renderers draw one, some do
    # not — stage 5 must not depend on finding it.
    strike_line: bool = False
    strike_color: RGB = (255, 255, 255)

    # Seconds of lookahead visible above the keybed. Sets the scroll speed
    # together with the frame height, which is what stage 5 has to recover.
    lead_time: float = 3.0

    def color_for(self, hand: str) -> RGB:
        return self.right_color if hand == "R" else self.left_color


THEMES: dict[str, Theme] = {
    # The warm gold look of the reference video: heavy bloom, no separators.
    "classic": Theme(
        name="classic",
        background=(7, 8, 13),
        right_color=(245, 215, 110),
        left_color=(214, 186, 92),
        tile_style="rounded",
        corner_radius=0.30,
        glow=0.55,
        lead_time=3.2,
    ),
    # Stock Synthesia: flat green/blue, lane separators, visible strike line.
    "synthesia": Theme(
        name="synthesia",
        background=(18, 20, 28),
        lane_separators=True,
        lane_color=(32, 35, 46),
        right_color=(64, 196, 99),
        left_color=(72, 132, 232),
        tile_style="flat",
        tile_gap=0.06,
        strike_line=True,
        strike_color=(210, 214, 226),
        keybed_ratio=0.26,
        lead_time=2.6,
    ),
    # Saturated neon on pure black, strong bloom, rounded.
    "neon": Theme(
        name="neon",
        background=(0, 0, 0),
        right_color=(255, 62, 165),
        left_color=(0, 229, 255),
        tile_style="rounded",
        corner_radius=0.45,
        glow=0.75,
        keybed_ratio=0.20,
        lead_time=3.8,
    ),
    # Deliberately awkward: outlined tiles on a light background, thin keybed.
    # If tile detection assumes "bright rectangle on dark", this breaks it.
    "paper": Theme(
        name="paper",
        background=(238, 236, 230),
        lane_separators=True,
        lane_color=(220, 217, 210),
        right_color=(196, 84, 62),
        left_color=(64, 92, 148),
        tile_style="outline",
        tile_gap=0.12,
        glow=0.0,
        keybed_ratio=0.17,
        white_key_color=(252, 252, 252),
        black_key_color=(40, 40, 44),
        key_edge_color=(150, 150, 156),
        keybed_shadow=False,
        strike_line=True,
        strike_color=(90, 90, 96),
        lead_time=2.2,
    ),
    # Vertical gradient tiles, so a tile's colour is not constant down its body.
    "aurora": Theme(
        name="aurora",
        background=(10, 14, 24),
        right_color=(120, 240, 200),
        left_color=(150, 120, 245),
        tile_style="gradient",
        corner_radius=0.20,
        glow=0.35,
        keybed_ratio=0.24,
        lead_time=3.0,
    ),
    # No gap at all, so adjacent keys played together touch and arrive as one
    # blob. Plenty of real renderers draw this way, and every other theme here
    # leaves a gap — which meant the corpus could not exercise horizontal
    # splitting at all, and scored a clean F1 while a merge bug sat in the code.
    "flush": Theme(
        name="flush",
        background=(12, 14, 20),
        right_color=(255, 176, 59),
        left_color=(58, 190, 255),
        tile_style="flat",
        tile_gap=0.0,
        glow=0.0,
        keybed_ratio=0.23,
        lead_time=2.9,
    ),
    # Models a screen capture rather than a clean render: a credit caption
    # burned into dark space below the keyboard, hands over the keys, and
    # struck keys blooming past white. Every one of those broke stage 3 on a
    # real video while the rest of this corpus stayed at F1 0.97.
    "capture": Theme(
        name="capture",
        background=(6, 6, 10),
        right_color=(168, 88, 245),
        left_color=(168, 88, 245),  # one colour for both hands, as many are
        tile_style="flat",
        tile_gap=0.05,
        glow=0.45,
        keybed_ratio=0.23,
        # Dim tan rather than white. A real capture's keybed measured 101 in
        # grey against this corpus's 190, and that gap is the whole bug: a
        # bright, crisp keyboard out-structures a caption on mean row variance,
        # so the caption never wins and the band search is never tested.
        white_key_color=(120, 105, 90),
        black_key_color=(18, 16, 20),
        key_edge_color=(70, 60, 50),  # must stay darker than the dimmed keys
        bottom_margin=0.30,
        caption="ARRANGED BY A. N. OTHER",
        highlight_bloom=0.9,
        hands=True,
        lead_time=2.4,
    ),
    # Minimal: no glow, no separators, tight gaps, fast scroll.
    "minimal": Theme(
        name="minimal",
        background=(20, 20, 20),
        right_color=(230, 230, 230),
        left_color=(140, 140, 140),
        tile_style="flat",
        tile_gap=0.04,
        glow=0.0,
        keybed_ratio=0.19,
        highlight_strength=0.6,
        lead_time=1.8,
    ),
}

DEFAULT_THEME = "classic"


def get_theme(name: str) -> Theme:
    try:
        return THEMES[name]
    except KeyError:
        raise KeyError(
            f"unknown theme {name!r}; available: {', '.join(sorted(THEMES))}"
        ) from None
