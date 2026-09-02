"""Procedural note sequences to render.

Real MIDI files are the better input, but they are also a dependency and a
licensing question. This generator gives the evaluation corpus something to chew
on with nothing installed: deterministic from a seed, and deliberately covering
the cases that break tile readers rather than aiming to sound good.

Those cases, all of which appear here:

* **Repeated notes** — two tiles on the same key with a hair of a gap. The single
  most common way a tracker under-counts, since the tiles merge into one blob.
* **Chords** — several tiles starting on the same frame, some on adjacent keys,
  which merge horizontally if the gap is misjudged.
* **Held notes under moving ones** — a long tile spanning many short ones.
* **Both hands in the same register** — hand assignment cannot fall back to a
  pitch split.
* **Wide leaps and dense runs** — stresses the key grid at both extremes.
"""

from __future__ import annotations

import random

from ..notes import Note, NoteSequence

# Scale degrees as semitone offsets from the tonic.
MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)

# Triads as scale-degree indices, and the progressions they appear in.
# Every one begins on the tonic. A progression that opens elsewhere — vi IV I V
# was here — spends more bars away from home than on it, and since the left hand
# holds a triad on the bar's degree throughout, the piece's weighted pitch
# content then implies a different key from the one it was generated in. That
# makes it useless as ground truth for key detection.
_PROGRESSIONS = (
    (0, 5, 3, 4),  # I  vi IV V
    (0, 3, 4, 0),  # I  IV V  I
    (0, 3, 5, 4),  # I  IV vi V
    (0, 4, 5, 3),  # I  V  vi IV
)

KEYS = {
    "C major": (60, MAJOR),
    "G major": (67, MAJOR),
    "F major": (65, MAJOR),
    "D major": (62, MAJOR),
    "A minor": (57, MINOR),
    "E minor": (64, MINOR),
    "D minor": (62, MINOR),
}


def _degree_pitch(tonic: int, scale: tuple[int, ...], degree: int) -> int:
    """Pitch for a scale degree, wrapping octaves for degrees outside 0-6."""
    octave, index = divmod(degree, len(scale))
    return tonic + 12 * octave + scale[index]


def generate(
    seed: int = 0,
    bars: int = 16,
    tempo: float = 96.0,
    key: str | None = None,
) -> NoteSequence:
    """Build a deterministic sequence with the awkward cases baked in."""
    rng = random.Random(seed)

    key = key or rng.choice(sorted(KEYS))
    tonic, scale = KEYS[key]
    beat = 60.0 / tempo
    bar = beat * 4

    progression = rng.choice(_PROGRESSIONS)
    notes: list[Note] = []

    for index in range(bars):
        start = index * bar
        # Open and close on the tonic. Without it the progressions — written
        # with major keys in mind — can dwell longer on the subdominant than
        # the tonic, and the piece genuinely implies a different key from the
        # one it was generated in. Ambiguity like that belongs in real music,
        # not in a corpus used as ground truth for key detection.
        degree = (
            0 if index in (0, bars - 1) else progression[index % len(progression)]
        )

        notes.extend(_left_hand(rng, tonic, scale, degree, start, beat, index))
        notes.extend(_right_hand(rng, tonic, scale, degree, start, beat, index))

    # One long pedal-ish tone under the middle section, so something is always
    # held while other tiles come and go.
    hold_start = bar * (bars // 4)
    notes.append(
        Note(
            onset=hold_start,
            pitch=_degree_pitch(tonic - 24, scale, 0),
            duration=bar * 2,
            hand="L",
            velocity=64,
        )
    )

    return NoteSequence.of(
        _drop_impossible_overlaps(notes),
        tempo=tempo,
        key=key,
        source=f"synthetic:seed={seed}",
    )


def _drop_impossible_overlaps(notes: list[Note]) -> list[Note]:
    """Remove notes that restrike a key already sounding.

    The hands are written independently, so both can land on one pitch at once
    — a held left-hand C with a short right-hand C inside it. No piano can do
    that, and the renderer draws the second tile over the first, splitting it
    in two. Ground truth then claims two notes where the video shows three
    fragments, and the pipeline is marked wrong for reading the picture
    correctly.
    """
    sounding: dict[int, float] = {}
    keep: list[Note] = []

    for note in sorted(notes):
        if note.onset < sounding.get(note.pitch, -1.0):
            continue
        sounding[note.pitch] = note.offset
        keep.append(note)

    return keep


def _left_hand(
    rng: random.Random,
    tonic: int,
    scale: tuple[int, ...],
    degree: int,
    start: float,
    beat: float,
    bar_index: int,
) -> list[Note]:
    root = _degree_pitch(tonic - 12, scale, degree)
    triad = [root, _degree_pitch(tonic - 12, scale, degree + 2), root + 12]

    # Raise the seventh on the dominant chord of a minor key, as harmonic minor
    # does. Without a leading tone, natural minor shares every pitch with its
    # relative major and sits a fifth away from its subdominant minor, so the
    # generated music was genuinely ambiguous about its own key — and ground
    # truth that disagrees with its own content is worse than none.
    if scale is MINOR and degree == 4:
        triad[1] += 1

    # Every fourth bar, play the chord as a block instead of broken, so tiles
    # start on the same frame across several keys at once.
    if bar_index % 4 == 3:
        return [
            Note(onset=start, pitch=p, duration=beat * 3.6, hand="L", velocity=70)
            for p in triad
        ]

    notes = []
    for i in range(4):
        pitch = triad[i % 3] if i < 3 else triad[1]
        notes.append(
            Note(
                onset=start + i * beat,
                pitch=pitch,
                duration=beat * rng.uniform(0.75, 0.95),
                hand="L",
                velocity=rng.randint(58, 76),
            )
        )
    return notes


def _right_hand(
    rng: random.Random,
    tonic: int,
    scale: tuple[int, ...],
    degree: int,
    start: float,
    beat: float,
    bar_index: int,
) -> list[Note]:
    notes: list[Note] = []
    position = 0.0

    # Every third bar, hammer one note repeatedly. Adjacent tiles on a single
    # key with a small gap are exactly what merges into one blob.
    if bar_index % 3 == 2:
        pitch = _degree_pitch(tonic, scale, degree + rng.randint(0, 2))
        step = beat / 2
        for i in range(8):
            notes.append(
                Note(
                    onset=start + i * step,
                    pitch=pitch,
                    duration=step * 0.72,  # leaves a real but small gap
                    hand="R",
                    velocity=rng.randint(70, 92),
                )
            )
        return notes

    while beat * 4 - position >= beat / 4:
        length = rng.choice((beat / 4, beat / 2, beat / 2, beat))
        # Never leave a sliver too short to render as a visible tile.
        length = min(length, beat * 4 - position)

        degree_here = degree + rng.randint(0, 6)
        pitch = _degree_pitch(tonic, scale, degree_here)

        # Occasionally drop into the left hand's register, so a pitch split
        # cannot separate the hands.
        if rng.random() < 0.08:
            pitch -= 12

        notes.append(
            Note(
                onset=start + position,
                pitch=pitch,
                duration=length * rng.uniform(0.7, 0.95),
                hand="R",
                velocity=rng.randint(66, 100),
            )
        )

        # Thirds and sixths above the melody note: adjacent-ish tiles that must
        # not be merged horizontally.
        if rng.random() < 0.22:
            notes.append(
                Note(
                    onset=start + position,
                    pitch=_degree_pitch(tonic, scale, degree_here + 2),
                    duration=length * 0.8,
                    hand="R",
                    velocity=rng.randint(60, 84),
                )
            )

        position += length

    return notes
