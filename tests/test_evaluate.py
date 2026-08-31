"""Stage 9.

The scorer is the thing every later judgement rests on, so it is tested on cases
where the right answer is known by construction — especially the repeated-note
case, where a greedy matcher would silently under-report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dropscore.config import Config, EvaluationConfig
from dropscore.evaluate import (
    EvaluationError,
    Metrics,
    Report,
    ClipResult,
    compare,
    find_clips,
    find_regressions,
    load_truth,
)
from dropscore.notes import Note, NoteSequence
from dropscore.synth import RenderSpec, generate, render


def _shift(sequence: NoteSequence, delta: float) -> NoteSequence:
    return NoteSequence.of(
        [Note(max(0.0, n.onset + delta), n.pitch, n.duration, n.hand, n.velocity) for n in sequence]
    )


# ── matching ─────────────────────────────────────────────────────────


def test_identical_sequences_score_one() -> None:
    sequence = generate(seed=1, bars=4)
    metrics = compare(sequence, sequence)
    assert metrics.f1 == 1.0
    assert metrics.matched == len(sequence)
    assert metrics.onset_mae == 0.0


def test_empty_estimate_scores_zero() -> None:
    metrics = compare(generate(seed=1, bars=2), NoteSequence())
    assert metrics.f1 == 0.0
    assert metrics.recall == 0.0
    assert metrics.precision == 0.0


def test_empty_reference_and_estimate_do_not_divide_by_zero() -> None:
    metrics = compare(NoteSequence(), NoteSequence())
    assert metrics.f1 == 0.0


def test_small_timing_error_still_matches() -> None:
    sequence = generate(seed=2, bars=2)
    metrics = compare(sequence, _shift(sequence, 0.02))
    assert metrics.matched == len(sequence)
    assert metrics.onset_mae == pytest.approx(0.02, abs=1e-6)


def test_large_timing_error_does_not_match() -> None:
    sequence = generate(seed=2, bars=2)
    assert compare(sequence, _shift(sequence, 0.5)).matched == 0


def test_tolerance_is_configurable() -> None:
    sequence = generate(seed=2, bars=2)
    loose = Config(evaluation=EvaluationConfig(onset_tolerance=0.5))
    assert compare(sequence, _shift(sequence, 0.2), loose).matched == len(sequence)


def test_wrong_pitch_never_matches() -> None:
    reference = NoteSequence.of([Note(onset=0.0, pitch=60, duration=1.0)])
    estimate = NoteSequence.of([Note(onset=0.0, pitch=61, duration=1.0)])
    assert compare(reference, estimate).matched == 0


def test_a_run_of_repeated_notes_matches_one_for_one() -> None:
    """The pipeline's weakest case, so the scorer must count it exactly."""
    reference = NoteSequence.of(
        [Note(onset=i * 0.12, pitch=60, duration=0.08) for i in range(8)]
    )
    estimate = NoteSequence.of(
        [Note(onset=i * 0.12 + 0.01, pitch=60, duration=0.08) for i in range(8)]
    )
    metrics = compare(reference, estimate)
    assert metrics.matched == 8
    assert metrics.f1 == 1.0


def test_a_merged_repeated_note_is_scored_as_missed() -> None:
    """Eight strikes read as one long note must score 1/8 recall, not 8/8.

    One-to-one matching is what enforces this: without it a single estimate
    would be reused for every reference within tolerance.
    """
    reference = NoteSequence.of(
        [Note(onset=i * 0.02, pitch=60, duration=0.02) for i in range(8)]
    )
    estimate = NoteSequence.of([Note(onset=0.0, pitch=60, duration=1.0)])

    metrics = compare(reference, estimate)
    assert metrics.matched == 1
    assert metrics.recall == pytest.approx(1 / 8)


def test_one_estimate_cannot_satisfy_two_references() -> None:
    reference = NoteSequence.of(
        [
            Note(onset=0.00, pitch=60, duration=0.1),
            Note(onset=0.02, pitch=60, duration=0.1),
        ]
    )
    estimate = NoteSequence.of([Note(onset=0.01, pitch=60, duration=0.1)])
    metrics = compare(reference, estimate)
    assert metrics.matched == 1
    assert metrics.missed == 1
    assert metrics.spurious == 0


def test_spurious_notes_cost_precision_not_recall() -> None:
    reference = NoteSequence.of([Note(onset=0.0, pitch=60, duration=0.5)])
    estimate = NoteSequence.of(
        [Note(onset=0.0, pitch=60, duration=0.5), Note(onset=2.0, pitch=72, duration=0.5)]
    )
    metrics = compare(reference, estimate)
    assert metrics.recall == 1.0
    assert metrics.precision == 0.5
    assert metrics.spurious == 1


def test_hand_accuracy_is_measured_over_matches() -> None:
    reference = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=0.5, hand="R"),
            Note(onset=1.0, pitch=48, duration=0.5, hand="L"),
        ]
    )
    estimate = NoteSequence.of(
        [
            Note(onset=0.0, pitch=60, duration=0.5, hand="R"),
            Note(onset=1.0, pitch=48, duration=0.5, hand="R"),  # wrong hand
        ]
    )
    assert compare(reference, estimate).hand_accuracy == pytest.approx(0.5)


def test_duration_error_is_reported() -> None:
    reference = NoteSequence.of([Note(onset=0.0, pitch=60, duration=1.0)])
    estimate = NoteSequence.of([Note(onset=0.0, pitch=60, duration=1.4)])
    assert compare(reference, estimate).duration_mae == pytest.approx(0.4)


# ── reports and regressions ──────────────────────────────────────────


def _metrics(matched: int, reference: int, estimate: int) -> Metrics:
    return Metrics(reference, estimate, matched, 0.0, 0.0, 0.0, 1.0)


def test_corpus_f1_pools_notes_rather_than_averaging_clips() -> None:
    """A tiny perfect clip must not outweigh a large bad one."""
    report = Report(
        clips=[
            ClipResult("tiny", _metrics(2, 2, 2)),  # F1 1.0
            ClipResult("large", _metrics(50, 100, 100)),  # F1 0.5
        ]
    )
    average_of_clips = (1.0 + 0.5) / 2
    assert report.f1 < average_of_clips
    assert report.f1 == pytest.approx(52 / 102 * 2 / 2, rel=0.01)


def test_regression_detected_when_a_clip_gets_worse() -> None:
    baseline = Report(clips=[ClipResult("a", _metrics(10, 10, 10))]).to_dict()
    worse = Report(clips=[ClipResult("a", _metrics(5, 10, 10))])

    regressions = find_regressions(baseline, worse)
    assert len(regressions) == 1
    assert regressions[0].name == "a"
    assert regressions[0].delta < 0


def test_small_wobble_is_not_a_regression() -> None:
    baseline = Report(clips=[ClipResult("a", _metrics(100, 100, 100))]).to_dict()
    slightly_worse = Report(clips=[ClipResult("a", _metrics(99, 100, 100))])
    assert find_regressions(baseline, slightly_worse) == []


def test_improvement_is_not_a_regression() -> None:
    baseline = Report(clips=[ClipResult("a", _metrics(5, 10, 10))]).to_dict()
    better = Report(clips=[ClipResult("a", _metrics(10, 10, 10))])
    assert find_regressions(baseline, better) == []


def test_a_clip_vanishing_from_the_corpus_counts_as_a_regression() -> None:
    """Otherwise dropping an awkward case would look like a clean run."""
    baseline = Report(
        clips=[ClipResult("a", _metrics(10, 10, 10)), ClipResult("b", _metrics(10, 10, 10))]
    ).to_dict()
    partial = Report(clips=[ClipResult("a", _metrics(10, 10, 10))])

    regressions = find_regressions(baseline, partial)
    assert [r.name for r in regressions] == ["b"]
    assert regressions[0].after == 0.0


def test_report_round_trips_through_json(tmp_path: Path) -> None:
    report = Report(clips=[ClipResult("a", _metrics(8, 10, 9))])
    path = report.save(tmp_path / "baseline.json")
    restored = json.loads(path.read_text(encoding="utf-8"))

    assert restored["clips"][0]["name"] == "a"
    assert restored["f1"] == pytest.approx(report.f1)


# ── corpus discovery ─────────────────────────────────────────────────


def test_finds_clips_next_to_their_sidecars(tmp_path: Path) -> None:
    spec = RenderSpec(width=320, height=180, fps=10.0)
    video, truth = render(generate(seed=3, bars=1), tmp_path / "clip.mp4", spec)

    pairs = find_clips(tmp_path)
    assert pairs == [(video, truth)]


def test_sidecar_without_a_video_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "orphan.truth.json").write_text("{}", encoding="utf-8")
    assert find_clips(tmp_path) == []


def test_load_truth_returns_notes_and_geometry(tmp_path: Path) -> None:
    spec = RenderSpec(width=320, height=180, fps=10.0)
    sequence = generate(seed=3, bars=1)
    _, truth = render(sequence, tmp_path / "clip.mp4", spec)

    reference, geometry = load_truth(truth)
    assert len(reference) > 0
    assert geometry["first_pitch"] == 21
    assert "speed_px_per_s" in geometry


def test_load_truth_rejects_a_foreign_json(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(EvaluationError, match="not a DropScore truth file"):
        load_truth(path)
