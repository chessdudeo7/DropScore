"""Real recordings scored against the printed music they were played from.

The synthetic corpus knows exactly what it drew, and that is its limit: it
only ever contains what it was built to contain. Every clip starts on a
downbeat, is in four, is played legato and holds one tempo, and a transcriber
tuned against it alone came back from a real capture with the meter, the beat,
the bar lines, the key signature and most of the written note values wrong.

A recording with its sheet music is the check that cannot be fooled that way,
but neither the recording nor the notes transcribed from the page belong to
this project. So pieces are described in JSON files kept outside the
repository -- by default in ``out/printed``, which is not tracked -- and
nothing here runs unless such files are present.

A piece file looks like this::

    {
      "title": "Interstellar, bars 1-26",
      "video": "C:/.../recording.mp4",
      "first_beat": -0.260,          # seconds at which beat 0 falls
      "beat": 0.5997,                # seconds per beat
      "window": [0.5, 77.5],         # the beats scored, inclusive
      "tempo": 100, "beats_per_bar": 3,
      "keys": ["A minor", "C major"], # any of these is right
      "lowest": 62, "highest": 84,   # optional: pitches outside are not scored
      "end": 145,                    # optional: seconds of recording to read
      "beat_times": [[0, 0.97], [3, 1.97], ...]  # optional, replaces first_beat and beat
      "sections": [24.385],          # optional: seconds where the tempo changes
      "scored_by": "order",          # optional: see below. "beat" by default
      "notes": [[64, 1, 1, "R"], ...] # pitch, beat, length in beats, staff
    }

``scored_by`` is ``"order"`` for music that cannot be checked against a beat.
A Liszt etude played with real rubato, its figuration written seven notes to
the beat, has its printed onsets within half a gap of each other however
carefully the beats are read off the recording, and the tolerance rather than
the transcriber would decide the score. Such a piece is asked only that every
note it prints comes back, once, in its turn: the two pitch streams are
aligned and what lines up is counted. Its notes carry their position in the
stream where a beat would go, its window bounds that position, and no written
value is judged -- there is no grid to judge one against.

``beat_times`` is for a performance that does not hold one tempo: seconds at
chosen beats, a bar line apiece say, read off the recording. Beats between them
are placed by straight interpolation. It says only where the beats fell -- the
notes, their values and their staves are still read from the page.

Staff means the staff the note is printed on, "R" for treble and "L" for
bass -- which is what is being judged, rather than which hand plays it.

``lowest`` and ``highest`` score only part of the texture: a melody read with
confidence, say, above accompanying voices that could not be. Notes detected
outside the bounds are neither found nor spurious.

Copy the recording in beside its piece file and give ``video`` as a bare file
name. Screen recorders keep their output in scratch folders: the Windows
Snipping Tool holds only its most recent recording, and two reference
recordings were lost from there when the next one was made. A piece whose
recording is missing is reported and skipped, never scored as a regression,
and its baseline is left alone.
"""

from __future__ import annotations

import bisect
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .config import DEFAULT, Config
from .notes import NoteSequence

log = logging.getLogger(__name__)

#: Where piece files are looked for unless told otherwise.
DEFAULT_DIRECTORY = Path("out/printed")

#: How far a detected onset may sit from the printed one and still be that note.
ONSET_TOLERANCE = 0.07

#: How far a written value may be from the printed one, as a fraction of it.
VALUE_TOLERANCE = 0.15

#: Tempo within this fraction of the printed mark counts as right.
TEMPO_TOLERANCE = 0.03

#: How far a bar's length may sit from the printed one and still be that bar.
BAR_TOLERANCE = 0.05


@dataclass(frozen=True)
class PrintedNote:
    pitch: int
    beat: float  # where it is printed, in beats from beat 0
    length: float  # its printed value, in beats
    staff: str  # "R" for the treble staff, "L" for the bass


@dataclass(frozen=True)
class PrintedPiece:
    path: Path
    title: str
    video: Path
    first_beat: float
    beat: float
    window: tuple[float, float]
    tempo: float | None
    beats_per_bar: int | None
    keys: tuple[str, ...]
    notes: tuple[PrintedNote, ...]
    lowest: int | None = None
    highest: int | None = None
    end: float | None = None
    beat_times: tuple[tuple[float, float], ...] = ()
    sections: tuple[float, ...] = ()

    #: "beat" to match each note where the page puts it, "order" to ask only
    #: that the notes come back in the right order. See ``_score_by_order``.
    scored_by: str = "beat"

    def covers(self, pitch: int) -> bool:
        return (self.lowest is None or pitch >= self.lowest) and (
            self.highest is None or pitch <= self.highest
        )

    @property
    def name(self) -> str:
        return self.path.name.removesuffix(".printed.json")

    def seconds(self, beat: float) -> float:
        if len(self.beat_times) >= 2:
            beats = [b for b, _ in self.beat_times]
            times = [t for _, t in self.beat_times]
            # Past either end, carry on at the nearest stretch's pace.
            i = min(max(bisect.bisect_right(beats, beat) - 1, 0), len(beats) - 2)
            (b0, t0), (b1, t1) = self.beat_times[i], self.beat_times[i + 1]
            return t0 + (beat - b0) * (t1 - t0) / (b1 - b0)
        return self.first_beat + beat * self.beat

    def scored(self) -> list[PrintedNote]:
        low, high = self.window
        return [n for n in self.notes if low <= n.beat <= high and self.covers(n.pitch)]

    def bar_seconds(self) -> float | None:
        """How long a printed bar lasts in the recording."""
        if not self.beats_per_bar:
            return None
        return self.seconds(self.beats_per_bar) - self.seconds(0)

    @property
    def baseline_path(self) -> Path:
        return self.path.with_name(f"{self.name}.baseline.json")


def load(path: str | Path) -> PrintedPiece:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    video = Path(data["video"])
    if not video.is_absolute():
        video = path.parent / video
    return PrintedPiece(
        path=path,
        title=str(data.get("title", path.stem)),
        video=video,
        first_beat=float(data.get("first_beat", 0.0)),
        beat=float(data.get("beat", 0.0)),
        window=(float(data["window"][0]), float(data["window"][1])),
        tempo=float(data["tempo"]) if data.get("tempo") else None,
        beats_per_bar=int(data["beats_per_bar"]) if data.get("beats_per_bar") else None,
        keys=tuple(data.get("keys", ())),
        notes=tuple(
            PrintedNote(int(p), float(b), float(length), str(staff))
            for p, b, length, staff in data["notes"]
        ),
        lowest=int(data["lowest"]) if data.get("lowest") is not None else None,
        highest=int(data["highest"]) if data.get("highest") is not None else None,
        end=float(data["end"]) if data.get("end") is not None else None,
        beat_times=tuple(sorted((float(b), float(t)) for b, t in data.get("beat_times", ()))),
        sections=tuple(sorted(float(t) for t in data.get("sections", ()))),
        scored_by=str(data.get("scored_by", "beat")),
    )


def find_pieces(directory: str | Path = DEFAULT_DIRECTORY) -> list[Path]:
    """Piece files in ``directory``; none, quietly, if it does not exist."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.printed.json"))


@dataclass
class PrintedResult:
    name: str
    title: str = ""
    printed: int = 0
    detected: int = 0
    matched: int = 0
    written_right: int = 0
    staff_right: int = 0
    tempo_found: float | None = None
    meter_found: int | None = None
    beat_type_found: int | None = None
    bar_found: float | None = None  # seconds a bar lasts
    bar_expected: float | None = None
    key_found: str | None = None
    tempo_expected: float | None = None
    meter_expected: int | None = None
    keys_expected: list[str] = field(default_factory=list)
    error: str | None = None

    #: Set when the piece was matched by order rather than by beat, where no
    #: written value can be judged and none is reported.
    by_order: bool = False

    @property
    def precision(self) -> float:
        return self.matched / self.detected if self.detected else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.printed if self.printed else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def written_share(self) -> float:
        return self.written_right / self.matched if self.matched else 0.0

    @property
    def staff_share(self) -> float:
        return self.staff_right / self.matched if self.matched else 0.0

    @property
    def tempo_right(self) -> bool | None:
        if self.tempo_expected is None or self.tempo_found is None:
            return None
        return abs(self.tempo_found - self.tempo_expected) / self.tempo_expected <= TEMPO_TOLERANCE

    @property
    def meter_right(self) -> bool | None:
        """Whether the bar lines fall where the page puts them.

        Judged by how long a bar lasts, not by the numerals. A page in 3/8 read
        as 6/8 has every bar line in the right place and every rhythm right; it
        spells the values twice as long, which is the same music written in
        coarser notes. A page in four read as two does not: its bars are half
        the length, and the bar lines land in the middle of the music's.
        """
        if self.bar_expected is None or self.bar_found is None:
            return None
        return abs(self.bar_found - self.bar_expected) / self.bar_expected <= BAR_TOLERANCE

    @property
    def key_right(self) -> bool | None:
        if not self.keys_expected or self.key_found is None:
            return None
        return self.key_found in self.keys_expected

    def __str__(self) -> str:
        if self.error:
            return f"FAILED  {self.error}"
        facts = [
            f"F1 {self.f1:.3f} ({self.matched}/{self.printed} found, "
            f"{self.detected - self.matched} spurious)",
        ]
        if not self.by_order:
            facts.append(f"written {self.written_right}/{self.matched}")
        facts.append(f"staff {self.staff_right}/{self.matched}")
        if self.by_order:
            facts.append("in order")
        for label, right, found in (
            ("tempo", self.tempo_right, f"{self.tempo_found:.1f}" if self.tempo_found else "-"),
            ("meter", self.meter_right, f"{self.meter_found}/{self.beat_type_found or 4}" if self.meter_found else "-"),
            ("key", self.key_right, self.key_found or "-"),
        ):
            if right is not None:
                facts.append(f"{label} {found}{'' if right else ' (wrong)'}")
        return "  ".join(facts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PrintedResult:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _score_by_order(
    piece: PrintedPiece, handed: NoteSequence, detected: int
) -> PrintedResult:
    """Score a piece on the order its notes come in, not on where they fall.

    Some music cannot be checked against a beat. A Liszt etude is played with
    real rubato and its figuration is written seven notes to the beat, so
    beats read off the recording would put the printed onsets within half a
    gap of each other and the tolerance would decide the score. Inventing
    positions for them would measure nothing but the invention.

    What can be asked without inventing anything is that every note the page
    prints comes back, once, in its turn. The two pitch streams are aligned
    and what lines up is counted -- the same question a beat match asks, with
    the timing left out of it.

    Only the stretch of recording the page covers is judged: the alignment
    says which played notes those are, and precision is measured over them
    rather than over a whole performance the page describes eight bars of.
    """
    printed = [n.pitch for n in piece.scored()]
    staves = [n.staff for n in piece.scored()]
    played = sorted(handed, key=lambda n: (n.onset, n.pitch))
    played = [n for n in played if piece.covers(n.pitch)]
    pitches = [n.pitch for n in played]

    # Where in the recording the printed passage starts. Lining the two up is
    # cheap to guess at and dear to do properly, so most offsets are ruled out
    # by counting how many pitches fall in the same place, and only the best
    # few are aligned in full.
    reach = int(len(printed) * 1.6) + 16
    # The guess is made on the opening of the passage, and by alignment rather
    # than by asking how many pitches fall in the same place: a stream with
    # notes missing or over-detected drifts out of step within a bar or two,
    # and counting positions then ranks the right offset below the wrong one.
    head, ahead = printed[:40], 80
    guesses = sorted(
        (
            (len(_longest_in_common(head, pitches[offset : offset + ahead])), -offset)
            for offset in range(max(1, len(pitches) - len(printed) // 2))
            if pitches[offset] == printed[0]
        ),
        reverse=True,
    )[:8]
    guesses = [(size, -back) for size, back in guesses]

    best: tuple[list[tuple[int, int]], int] | None = None
    for _, offset in guesses:
        pairs = _longest_in_common(printed, pitches[offset : offset + reach])
        if best is None or len(pairs) > len(best[0]):
            best = (pairs, offset)

    if best is None or not best[0]:
        return PrintedResult(
            name=piece.name, title=piece.title, printed=len(printed), detected=detected,
            by_order=True, error="the printed passage was not found in the recording",
        )

    pairs, offset = best
    staff_right = sum(played[offset + there].hand == staves[here] for here, there in pairs)
    return PrintedResult(
        name=piece.name,
        title=piece.title,
        printed=len(printed),
        # The played notes the page covers, not the whole performance: a page
        # of eight bars must not be charged for the hundred that follow it.
        detected=pairs[-1][1] - pairs[0][1] + 1,
        matched=len(pairs),
        staff_right=staff_right,
        by_order=True,
    )


def _longest_in_common(printed: list[int], played: list[int]) -> list[tuple[int, int]]:
    """Pair up as many notes as the two streams have in common, in order.

    The longest common subsequence, which is the most notes that can be lined
    up without either stream going backwards. ``difflib`` is the obvious tool
    and is the wrong one: it takes the longest matching block first and
    recurses either side of it, which on a figuration that repeats the same
    wave is not the best it could do -- given an arpeggio of fifty-six notes
    with two wrong, it paired up twenty-eight of them.
    """
    rows, columns = len(printed), len(played)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for here in range(rows - 1, -1, -1):
        row, below, pitch = table[here], table[here + 1], printed[here]
        for there in range(columns - 1, -1, -1):
            row[there] = (
                below[there + 1] + 1 if pitch == played[there]
                else max(below[there], row[there + 1])
            )

    pairs: list[tuple[int, int]] = []
    here = there = 0
    while here < rows and there < columns:
        if printed[here] == played[there]:
            pairs.append((here, there))
            here, there = here + 1, there + 1
        elif table[here + 1][there] >= table[here][there + 1]:
            here += 1
        else:
            there += 1
    return pairs


def score_sequence(
    piece: PrintedPiece, sequence: NoteSequence, config: Config = DEFAULT
) -> PrintedResult:
    """Score an already transcribed sequence against the printed music."""
    from .score import (  # noqa: PLC0415
        ScoreError, assign_hands, beat_position, notate_durations, postprocess,
    )

    # A page covers one stretch of the music, and where the recording changes
    # tempo the stretch it covers is the one it must be read against.
    if piece.sections:
        config = replace(config, score=replace(config.score, sections=piece.sections))

    # A recording the transcriber can barely read yields too few notes to find
    # a tempo in, and that raised out of eval entirely -- on a capture of neon
    # outlines, whose two detected notes crashed the run. That is the failure
    # this check exists to report, so it is reported as one.
    try:
        handed, analysis = postprocess(sequence, config)
    except ScoreError as exc:
        return PrintedResult(
            name=piece.name,
            title=piece.title,
            printed=len(piece.scored()),
            detected=len(sequence),
            error=f"could not analyse {len(sequence)} note(s): {exc}",
        )
    if piece.scored_by == "order":
        # Read as played, not as snapped to a grid. Quantising a figuration
        # this fast puts several notes on one gridline -- 1755 of 2713 on the
        # etude this exists for -- and the order within a gridline is then
        # whatever the sort falls back on, which scrambled every wave: an
        # arpeggio printed 61 65 68 65 61 came back 61 65 61 65 68. The hands
        # are still wanted, and assigning them moves nothing.
        return _score_by_order(piece, assign_hands(sequence, config), len(sequence))

    written = notate_durations(handed, analysis, config)

    low, high = piece.window
    start, end = piece.seconds(low), piece.seconds(high)
    detected = [
        n for n in handed
        if start - ONSET_TOLERANCE <= n.onset <= end + ONSET_TOLERANCE and piece.covers(n.pitch)
    ]
    values = {(n.pitch, round(n.onset, 6)): n for n in written}

    here = analysis.at(piece.seconds(low))
    result = PrintedResult(
        name=piece.name,
        title=piece.title,
        printed=len(piece.scored()),
        detected=len(detected),
        tempo_found=here.tempo,
        meter_found=here.beats_per_bar,
        beat_type_found=here.beat_type,
        bar_found=here.beat * here.beats_per_bar,
        bar_expected=piece.bar_seconds(),
        key_found=analysis.key,
        tempo_expected=piece.tempo,
        meter_expected=piece.beats_per_bar,
        keys_expected=list(piece.keys),
    )

    used: set[int] = set()
    for printed in sorted(piece.scored(), key=lambda n: n.beat):
        when = piece.seconds(printed.beat)
        best = None
        for index, note in enumerate(detected):
            if index in used or note.pitch != printed.pitch:
                continue
            if abs(note.onset - when) <= ONSET_TOLERANCE and (
                best is None or abs(note.onset - when) < abs(detected[best].onset - when)
            ):
                best = index
        if best is None:
            continue
        used.add(best)
        note = detected[best]
        result.matched += 1
        result.staff_right += note.hand == printed.staff

        engraved = values.get((note.pitch, round(note.onset, 6)))
        if engraved is not None:
            length = beat_position(engraved.onset + engraved.duration, analysis) - beat_position(
                engraved.onset, analysis
            )
            # As a share of a bar. A page in 3/8 read as 6/8 writes every value
            # twice as long, and every one of them fills the same part of the
            # same bar; comparing the numbers alone called all 104 of them
            # wrong on music read note for note.
            written = length / here.beats_per_bar
            wanted = printed.length / piece.beats_per_bar if piece.beats_per_bar else printed.length
            result.written_right += abs(written - wanted) / wanted < VALUE_TOLERANCE

    return result


def score(piece: PrintedPiece, config: Config = DEFAULT) -> PrintedResult:
    """Transcribe the recording and score it against the printed music."""
    from .calibrate import calibrate  # noqa: PLC0415
    from .tiles import discover_palette  # noqa: PLC0415
    from .tracking import measure_scroll_speed, transcribe  # noqa: PLC0415
    from .video import VideoReader  # noqa: PLC0415

    if not piece.video.exists():
        return PrintedResult(name=piece.name, title=piece.title,
                             error=f"recording not found at {piece.video}")
    try:
        with VideoReader(piece.video, config, end=piece.end) as reader:
            samples = reader.sample()
            calibration = calibrate(samples, config)
            palette = discover_palette(samples, calibration, config)
            speed = measure_scroll_speed(reader, calibration, 40, config=config, palette=palette)
            sequence = transcribe(reader.frames(), calibration, palette, speed, config)
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        log.exception("could not transcribe %s", piece.video)
        return PrintedResult(name=piece.name, title=piece.title, error=str(exc))
    return score_sequence(piece, sequence, config)


def regressions(before: PrintedResult, after: PrintedResult) -> list[str]:
    """What got worse, in words; empty when nothing did."""
    if after.error:
        return [f"failed: {after.error}"]
    worse = []
    for label, old, new in (
        ("notes F1", before.f1, after.f1),
        ("written values", before.written_share, after.written_share),
        ("staff", before.staff_share, after.staff_share),
    ):
        if new < old - 1e-9:
            worse.append(f"{label} {old:.3f} -> {new:.3f}")
    for label, old, new in (
        ("tempo", before.tempo_right, after.tempo_right),
        ("meter", before.meter_right, after.meter_right),
        ("key", before.key_right, after.key_right),
    ):
        if old and new is False:
            worse.append(f"{label} was right and is now wrong")
    return worse
