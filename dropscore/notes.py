"""Note events — the currency the whole pipeline trades in.

The synthetic renderer emits these as ground truth (stage 2), the tile tracker
produces them from video (stage 5), post-processing refines them (stage 7), the
exporters consume them (stage 8), and evaluation compares two sets of them
(stage 9). Keeping one representation means those stages never translate.

Times are in **seconds**, not frames or ticks. Timing is recovered from tile
geometry rather than frame indices (see docs/PIPELINE.md), so it is continuous
and the frame rate is not part of the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Hand = Literal["L", "R"]

# The 88-key range: A0 to C8.
MIN_PITCH = 21
MAX_PITCH = 108

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def pitch_name(pitch: int) -> str:
    """MIDI number to a name like ``F#4``. Middle C (60) is C4."""
    return f"{_PITCH_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True, order=True)
class Note:
    """A single note. Ordered by onset, then pitch, so sorting is stable."""

    onset: float  # seconds from the start of the video
    pitch: int  # MIDI note number
    duration: float  # seconds
    hand: Hand = "R"
    velocity: int = 80

    def __post_init__(self) -> None:
        if not MIN_PITCH <= self.pitch <= MAX_PITCH:
            raise ValueError(
                f"pitch {self.pitch} is outside the 88-key range "
                f"{MIN_PITCH}-{MAX_PITCH}"
            )
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.onset < 0:
            raise ValueError(f"onset must not be negative, got {self.onset}")

    @property
    def offset(self) -> float:
        return self.onset + self.duration

    @property
    def name(self) -> str:
        return pitch_name(self.pitch)


@dataclass
class NoteSequence:
    """An ordered collection of notes, plus whatever is known about the piece."""

    notes: list[Note] = field(default_factory=list)
    tempo: float | None = None  # BPM, when known
    key: str | None = None  # e.g. "F major", when known
    source: str | None = None  # where these came from

    def __post_init__(self) -> None:
        self.notes.sort()

    def __iter__(self) -> Iterator[Note]:
        return iter(self.notes)

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, index: int) -> Note:
        return self.notes[index]

    @property
    def duration(self) -> float:
        """End of the last note to sound, or 0.0 when empty."""
        return max((n.offset for n in self.notes), default=0.0)

    @property
    def pitch_range(self) -> tuple[int, int]:
        if not self.notes:
            raise ValueError("an empty sequence has no pitch range")
        pitches = [n.pitch for n in self.notes]
        return min(pitches), max(pitches)

    def hand(self, hand: Hand) -> list[Note]:
        return [n for n in self.notes if n.hand == hand]

    def add(self, *notes: Note) -> None:
        self.notes.extend(notes)
        self.notes.sort()

    def between(self, start: float, end: float) -> list[Note]:
        """Notes sounding at any point within ``[start, end)``."""
        return [n for n in self.notes if n.onset < end and n.offset > start]

    # ── serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo": self.tempo,
            "key": self.key,
            "source": self.source,
            "notes": [asdict(n) for n in self.notes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteSequence":
        return cls(
            notes=[Note(**n) for n in data.get("notes", [])],
            tempo=data.get("tempo"),
            key=data.get("key"),
            source=data.get("source"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NoteSequence":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def of(cls, notes: Iterable[Note], **meta: Any) -> "NoteSequence":
        return cls(notes=list(notes), **meta)
