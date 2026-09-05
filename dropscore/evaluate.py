"""Stage 9: measure how good the transcription actually is.

Every earlier stage has been checked against ground truth in its own terms —
calibration in pixels, timing in seconds. This turns the whole pipeline into one
number per clip and one number per corpus, and compares that against a stored
baseline so a change that helps one theme and quietly breaks another cannot land
unnoticed.

**Matching is one-to-one and maximum-cardinality.** A reference note and an
estimate match when their pitches are equal and their onsets fall within a
tolerance, and each note on either side can be used at most once. That second
half is what stops a single estimate from "explaining" a whole run of repeated
notes on one key and inflating recall — the exact case where this pipeline is
weakest, so the scorer must not flatter it. Maximum-cardinality matching (Kuhn's
algorithm) is the standard choice and is what ``mir_eval`` uses; for candidacy
windows on a line a nearest-first greedy pass usually reaches the same answer,
but the matching formulation is the one that is right by construction rather
than by argument.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Config, DEFAULT
from .notes import Note, NoteSequence

log = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when a comparison cannot be made."""


@dataclass(frozen=True)
class Metrics:
    """Note-level agreement between a reference and an estimate."""

    reference_count: int
    estimate_count: int
    matched: int
    onset_mae: float  # seconds, over matched pairs
    onset_p95: float
    duration_mae: float
    hand_accuracy: float  # over matched pairs

    @property
    def precision(self) -> float:
        return self.matched / self.estimate_count if self.estimate_count else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.reference_count if self.reference_count else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def missed(self) -> int:
        return self.reference_count - self.matched

    @property
    def spurious(self) -> int:
        return self.estimate_count - self.matched

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(precision=self.precision, recall=self.recall, f1=self.f1)
        return data

    def __str__(self) -> str:
        return (
            f"F1 {self.f1:.3f} (P {self.precision:.3f} R {self.recall:.3f})  "
            f"{self.matched}/{self.reference_count} matched, "
            f"{self.spurious} spurious, onset MAE {self.onset_mae * 1000:.1f}ms"
        )


def _max_matching(candidates: Sequence[Sequence[int]]) -> dict[int, int]:
    """Maximum bipartite matching by augmenting paths (Kuhn's algorithm).

    ``candidates[i]`` lists the estimate indices that reference note ``i`` could
    pair with. Returns estimate index -> reference index.
    """
    match_right: dict[int, int] = {}

    def augment(left: int, seen: set[int]) -> bool:
        for right in candidates[left]:
            if right in seen:
                continue
            seen.add(right)
            if right not in match_right or augment(match_right[right], seen):
                match_right[right] = left
                return True
        return False

    for left in range(len(candidates)):
        augment(left, set())
    return match_right


def compare(
    reference: NoteSequence,
    estimate: NoteSequence,
    config: Config = DEFAULT,
) -> Metrics:
    """Score an estimate against a reference."""
    cfg = config.evaluation
    reference_notes: list[Note] = list(reference)
    estimate_notes: list[Note] = list(estimate)

    by_pitch: dict[int, list[int]] = {}
    for index, note in enumerate(estimate_notes):
        by_pitch.setdefault(note.pitch, []).append(index)

    candidates: list[list[int]] = []
    for note in reference_notes:
        options = [
            index
            for index in by_pitch.get(note.pitch, [])
            if abs(estimate_notes[index].onset - note.onset) <= cfg.onset_tolerance
        ]
        # Nearest first, so the search settles on tight pairs when free to choose.
        options.sort(key=lambda i: abs(estimate_notes[i].onset - note.onset))
        candidates.append(options)

    matches = _max_matching(candidates)

    onset_errors: list[float] = []
    duration_errors: list[float] = []
    hand_hits = 0
    for right, left in matches.items():
        expected, actual = reference_notes[left], estimate_notes[right]
        onset_errors.append(abs(actual.onset - expected.onset))
        duration_errors.append(abs(actual.duration - expected.duration))
        hand_hits += actual.hand == expected.hand

    return Metrics(
        reference_count=len(reference_notes),
        estimate_count=len(estimate_notes),
        matched=len(matches),
        onset_mae=float(np.mean(onset_errors)) if onset_errors else 0.0,
        onset_p95=float(np.percentile(onset_errors, 95)) if onset_errors else 0.0,
        duration_mae=float(np.mean(duration_errors)) if duration_errors else 0.0,
        hand_accuracy=hand_hits / len(matches) if matches else 0.0,
    )


# ── geometry, from the truth sidecar ─────────────────────────────────


@dataclass(frozen=True)
class GeometryError:
    """How far the fitted geometry was from the geometry that drew the clip."""

    first_pitch_correct: bool
    white_width_error: float  # pixels
    strike_y_error: float
    speed_error: float  # px/s, absolute

    def __str__(self) -> str:
        anchor = "ok" if self.first_pitch_correct else "WRONG KEY"
        return (
            f"grid {anchor}, width {self.white_width_error:+.2f}px, "
            f"strike {self.strike_y_error:+.1f}px, speed {self.speed_error:+.1f}px/s"
        )


@dataclass
class ClipResult:
    """Everything measured about one clip."""

    name: str
    metrics: Metrics
    geometry: GeometryError | None = None
    error: str | None = None

    # What the key finder said against what the clip was written in. Scored
    # separately from the notes: a transcription can name every pitch correctly
    # and still be engraved in the wrong key, which is what a reader sees first.
    key_expected: str | None = None
    key_found: str | None = None
    key_confidence: float = 0.0

    @property
    def key_correct(self) -> bool | None:
        if not self.key_expected or not self.key_found:
            return None
        return self.key_expected == self.key_found

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics.to_dict(),
            "geometry": asdict(self.geometry) if self.geometry else None,
            "error": self.error,
            "key_expected": self.key_expected,
            "key_found": self.key_found,
            "key_confidence": self.key_confidence,
        }


@dataclass
class Report:
    """A corpus run, comparable against a stored baseline."""

    clips: list[ClipResult] = field(default_factory=list)

    @property
    def f1(self) -> float:
        """Corpus F1, pooled over notes rather than averaged over clips.

        Averaging per-clip F1 would let a two-note clip outvote a thousand-note
        one. Pooling counts asks the question that matters: across everything,
        what fraction of notes were right?
        """
        matched = sum(c.metrics.matched for c in self.clips)
        reference = sum(c.metrics.reference_count for c in self.clips)
        estimate = sum(c.metrics.estimate_count for c in self.clips)
        if not reference or not estimate:
            return 0.0
        precision, recall = matched / estimate, matched / reference
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @property
    def failures(self) -> list[ClipResult]:
        return [c for c in self.clips if c.error]

    def to_dict(self) -> dict[str, Any]:
        return {"f1": self.f1, "clips": [c.to_dict() for c in self.clips]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def per_clip_f1(self) -> dict[str, float]:
        return {c.name: c.metrics.f1 for c in self.clips}


@dataclass(frozen=True)
class Regression:
    """A clip that got worse, or a clip that vanished from the corpus."""

    name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    def __str__(self) -> str:
        return f"{self.name}: F1 {self.before:.3f} -> {self.after:.3f} ({self.delta:+.3f})"


def find_regressions(
    baseline: dict[str, Any], report: Report, config: Config = DEFAULT
) -> list[Regression]:
    """Clips whose F1 dropped by more than the tolerance.

    A missing clip counts as a regression to zero: silently dropping a case from
    the corpus is exactly how a corpus stops catching anything.
    """
    tolerance = config.evaluation.regression_tolerance
    current = report.per_clip_f1()
    regressions: list[Regression] = []

    for clip in baseline.get("clips", []):
        name = clip["name"]
        before = float(clip["metrics"]["f1"])
        after = current.get(name)
        if after is None:
            regressions.append(Regression(name, before, 0.0))
        elif before - after > tolerance:
            regressions.append(Regression(name, before, after))

    return sorted(regressions, key=lambda r: r.delta)


# ── running the corpus ───────────────────────────────────────────────


def load_truth(path: str | Path) -> tuple[NoteSequence, dict[str, Any]]:
    """Read a ``.truth.json`` sidecar into a sequence and its geometry."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sequence" not in data:
        raise EvaluationError(f"{path} is not a DropScore truth file")
    return NoteSequence.from_dict(data["sequence"]), data.get("geometry", {})


def run_clip(video: str | Path, truth: str | Path, config: Config = DEFAULT) -> ClipResult:
    """Transcribe one clip and score it against its sidecar.

    Failures are captured rather than raised: one clip that cannot be calibrated
    should be reported as a zero, not abort the corpus run.
    """
    from .calibrate import calibrate  # noqa: PLC0415
    from .score import ScoreError, postprocess  # noqa: PLC0415
    from .tiles import discover_palette  # noqa: PLC0415
    from .tracking import measure_scroll_speed, transcribe  # noqa: PLC0415
    from .video import VideoReader  # noqa: PLC0415

    video = Path(video)
    reference, geometry = load_truth(truth)
    name = video.stem

    empty = Metrics(len(reference), 0, 0, 0.0, 0.0, 0.0, 0.0)

    try:
        with VideoReader(video, config) as reader:
            samples = reader.sample()
            calibration = calibrate(samples, config)
            palette = discover_palette(samples, calibration, config)

            speed = measure_scroll_speed(reader, calibration, 40, config=config)

            estimate = transcribe(reader.frames(), calibration, palette, speed, config)

        analysis = None
        try:
            estimate, analysis = postprocess(estimate, config)
        except ScoreError as exc:
            log.warning("%s: leaving notes raw (%s)", name, exc)

    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the run
        log.error("%s failed: %s", name, exc)
        return ClipResult(name=name, metrics=empty, error=str(exc))

    error = None
    if geometry:
        error = GeometryError(
            first_pitch_correct=calibration.layout.first_pitch == geometry.get("first_pitch"),
            white_width_error=calibration.white_width - geometry.get("white_key_width", 0.0),
            strike_y_error=calibration.strike_y - geometry.get("strike_y", 0),
            speed_error=speed.value - geometry.get("speed_px_per_s", 0.0),
        )

    return ClipResult(
        name=name,
        metrics=compare(reference, estimate, config),
        geometry=error,
        key_expected=reference.key,
        key_found=analysis.key if analysis else None,
        key_confidence=analysis.key_confidence if analysis else 0.0,
    )


def run_corpus(
    pairs: Sequence[tuple[Path, Path]], config: Config = DEFAULT
) -> Report:
    report = Report()
    for video, truth in pairs:
        result = run_clip(video, truth, config)
        report.clips.append(result)
        log.info("%s: %s", result.name, result.error or result.metrics)
    return report


def find_clips(directory: str | Path) -> list[tuple[Path, Path]]:
    """Pair every truth sidecar in a directory with its video."""
    directory = Path(directory)
    pairs: list[tuple[Path, Path]] = []

    for truth in sorted(directory.glob("*.truth.json")):
        stem = truth.name[: -len(".truth.json")]
        videos = [p for p in directory.glob(f"{stem}.*") if p.suffix in {".mp4", ".avi", ".mkv"}]
        if videos:
            pairs.append((videos[0], truth))
        else:
            log.warning("no video found alongside %s", truth.name)

    return pairs
