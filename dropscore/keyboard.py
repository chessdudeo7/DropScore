"""Piano keyboard geometry: the map between pixel columns and MIDI pitches.

This is the load-bearing abstraction of the whole project. Stage 2 uses it to
*draw* a keyboard; stage 3 will *fit* one to a real video frame. Both sides
speaking the same model means calibration can be scored directly against the
layout that produced a synthetic clip.

The convention matches how these renderers actually draw a keyboard: white keys
are equal-width and tile the strip edge to edge, and each black key is centred on
the boundary between the two white keys it sits between. Real pianos offset black
keys slightly within their groups; Synthesia-style renderers do not, and matching
the renderer is what matters here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .notes import MAX_PITCH, MIN_PITCH

# Pitch classes of the white keys, and each one's index within an octave.
WHITE_PITCH_CLASSES = (0, 2, 4, 5, 7, 9, 11)
_WHITE_INDEX = {pc: i for i, pc in enumerate(WHITE_PITCH_CLASSES)}

# Black pitch class -> the white pitch class immediately below it.
_BLACK_BELOW = {1: 0, 3: 2, 6: 5, 8: 7, 10: 9}


def is_black(pitch: int) -> bool:
    return pitch % 12 not in _WHITE_INDEX


def is_white(pitch: int) -> bool:
    return pitch % 12 in _WHITE_INDEX


def white_ordinal(pitch: int) -> int:
    """Absolute index of a white key, counting every white key from MIDI 0.

    For a black key this returns the ordinal of the white key just below it,
    which is exactly what is needed to place the black key on a boundary.
    """
    pitch_class = pitch % 12
    if pitch_class in _WHITE_INDEX:
        return 7 * (pitch // 12) + _WHITE_INDEX[pitch_class]
    return 7 * (pitch // 12) + _WHITE_INDEX[_BLACK_BELOW[pitch_class]]


def white_pitches(first: int, last: int) -> list[int]:
    return [p for p in range(first, last + 1) if is_white(p)]


@dataclass(frozen=True)
class KeyboardLayout:
    """Maps pitches to pixel columns across a keybed strip.

    ``x0`` and ``width`` describe the horizontal extent of the *white* keys, so
    ``x0`` is the left edge of the lowest white key and ``x0 + width`` the right
    edge of the highest.
    """

    first_pitch: int = MIN_PITCH  # A0
    last_pitch: int = MAX_PITCH  # C8
    x0: float = 0.0
    width: float = 1280.0

    # Black keys as a fraction of white-key width and of keybed height.
    black_width_ratio: float = 0.62
    black_height_ratio: float = 0.62

    def __post_init__(self) -> None:
        if self.first_pitch >= self.last_pitch:
            raise ValueError("first_pitch must be below last_pitch")
        if not is_white(self.first_pitch) or not is_white(self.last_pitch):
            raise ValueError(
                "a keyboard starts and ends on white keys; got "
                f"{self.first_pitch} to {self.last_pitch}"
            )
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")

    # ── counts ───────────────────────────────────────────────────────

    @property
    def white_count(self) -> int:
        return white_ordinal(self.last_pitch) - white_ordinal(self.first_pitch) + 1

    @property
    def white_width(self) -> float:
        return self.width / self.white_count

    @property
    def black_width(self) -> float:
        return self.white_width * self.black_width_ratio

    @property
    def pitches(self) -> range:
        return range(self.first_pitch, self.last_pitch + 1)

    def contains(self, pitch: int) -> bool:
        return self.first_pitch <= pitch <= self.last_pitch

    # ── pitch -> pixels ──────────────────────────────────────────────

    def white_index(self, pitch: int) -> int:
        """Index of ``pitch`` among this keyboard's white keys, from 0."""
        return white_ordinal(pitch) - white_ordinal(self.first_pitch)

    def white_span(self, pitch: int) -> tuple[float, float]:
        """Left and right edges of a white key."""
        if is_black(pitch):
            raise ValueError(f"{pitch} is a black key")
        index = self.white_index(pitch)
        left = self.x0 + index * self.white_width
        return left, left + self.white_width

    def key_center(self, pitch: int) -> float:
        """Centre column of any key.

        White keys sit at the middle of their slot; black keys sit on the
        boundary between the white keys either side of them.
        """
        self._require_in_range(pitch)
        if is_white(pitch):
            left, right = self.white_span(pitch)
            return (left + right) / 2
        # white_ordinal() already resolved to the white key below.
        boundary_index = self.white_index(pitch) + 1
        return self.x0 + boundary_index * self.white_width

    def key_span(self, pitch: int) -> tuple[float, float]:
        """Left and right edges of any key, as drawn."""
        if is_white(pitch):
            return self.white_span(pitch)
        center = self.key_center(pitch)
        half = self.black_width / 2
        return center - half, center + half

    def key_width(self, pitch: int) -> float:
        return self.black_width if is_black(pitch) else self.white_width

    # ── pixels -> pitch ──────────────────────────────────────────────

    def pitch_at(self, x: float, y_fraction: float = 0.0) -> int | None:
        """Which key covers column ``x``, or None if outside the keyboard.

        ``y_fraction`` is the depth down the keybed, 0.0 at the far edge and 1.0
        at the near edge. Black keys only occupy the top ``black_height_ratio``
        of the strip, so below that a column always belongs to a white key —
        which is why tile-to-key mapping should sample near the top.
        """
        if x < self.x0 or x > self.x0 + self.width:
            return None

        if y_fraction <= self.black_height_ratio:
            for pitch in self.pitches:
                if is_black(pitch):
                    left, right = self.key_span(pitch)
                    if left <= x <= right:
                        return pitch

        index = int((x - self.x0) // self.white_width)
        index = min(index, self.white_count - 1)  # right edge lands one past
        whites = white_pitches(self.first_pitch, self.last_pitch)
        return whites[index]

    def _require_in_range(self, pitch: int) -> None:
        if not self.contains(pitch):
            raise ValueError(
                f"pitch {pitch} is outside this keyboard "
                f"({self.first_pitch}-{self.last_pitch})"
            )


# Ranges these videos commonly show. A full 88 is most typical, but plenty of
# channels crop to the played range to make the keys bigger.
COMMON_RANGES: dict[str, tuple[int, int]] = {
    "88": (21, 108),  # A0 - C8, full piano
    "76": (28, 103),  # E1 - G7
    "61": (36, 96),  # C2 - C6, common controller range
    "49": (36, 84),  # C2 - C5
}
