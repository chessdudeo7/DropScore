"""Stage 7, tested against note data rather than video.

These functions take notes in and give notes out, so they are tested directly on
sequences — including the generator's own output, whose tempo and key are known
exactly.
"""

from __future__ import annotations

import pytest

from dropscore.config import DEFAULT, Config, ScoreConfig
from dropscore.notes import Note, NoteSequence
from dropscore.score import (
    ScoreError,
    analyze,
    assign_hands,
    estimate_key,
    estimate_tempo,
    find_downbeat,
    postprocess,
    quantize,
)
from dropscore.synth import generate


def _grid(tempo: float, count: int = 32, subdivision: int = 4) -> NoteSequence:
    """Notes exactly on a subdivision grid at a known tempo."""
    step = 60.0 / tempo / subdivision
    return NoteSequence.of(
        [
            Note(onset=i * step, pitch=60 + (i % 5), duration=step * 0.8)
            for i in range(count)
        ]
    )


# ── tempo ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tempo", [72.0, 96.0, 120.0, 144.0])
def test_recovers_a_known_tempo(tempo: float) -> None:
    beat, _, confidence = estimate_tempo(_grid(tempo))
    assert 60.0 / beat == pytest.approx(tempo, rel=0.03)
    assert confidence > 0.9


def test_does_not_report_the_subdivision_as_the_beat() -> None:
    """Sixteenths fit a sixteenth grid perfectly; the beat is four of them."""
    beat, _, _ = estimate_tempo(_grid(120.0, subdivision=4))
    assert beat == pytest.approx(0.5, rel=0.05)


def test_phase_puts_offset_onsets_back_on_the_grid() -> None:
    """Phase is read at the tatum, so what matters is that onsets land on it.

    Reading it at the beat period would not work: four sixteenths sit at 0, 90,
    180 and 270 degrees within a beat and average to nothing.
    """
    offset = 0.137
    shifted = NoteSequence.of(
        [Note(n.onset + offset, n.pitch, n.duration, n.hand, n.velocity) for n in _grid(120.0)]
    )
    beat, phase, _ = estimate_tempo(shifted)

    step = beat / DEFAULT.score.steps_per_beat
    residual = (offset - phase) % step
    assert min(residual, step - residual) < 0.02


def test_tempo_survives_jittered_onsets() -> None:
    import random  # noqa: PLC0415

    rng = random.Random(0)
    step = 60.0 / 120.0 / 4
    notes = [
        Note(onset=max(0.0, i * step + rng.gauss(0, 0.008)), pitch=60, duration=step * 0.7)
        for i in range(48)
    ]
    beat, _, _ = estimate_tempo(NoteSequence.of(notes))
    assert 60.0 / beat == pytest.approx(120.0, rel=0.05)


def test_tempo_needs_enough_onsets() -> None:
    sparse = NoteSequence.of([Note(onset=float(i), pitch=60, duration=0.5) for i in range(3)])
    with pytest.raises(ScoreError, match="distinct onsets"):
        estimate_tempo(sparse)


def test_generated_music_reports_its_own_tempo() -> None:
    sequence = generate(seed=4, bars=16, tempo=96.0)
    beat, _, _ = estimate_tempo(sequence)
    assert 60.0 / beat == pytest.approx(96.0, rel=0.04)


# ── downbeat ─────────────────────────────────────────────────────────


def test_downbeat_follows_the_bass() -> None:
    beat = 0.5
    notes = []
    for bar in range(8):
        start = bar * beat * 4
        notes.append(Note(onset=start, pitch=40, duration=beat))  # bass on beat 1
        for offset in (1, 2, 3):
            notes.append(Note(onset=start + offset * beat, pitch=72, duration=beat * 0.5))

    downbeat = find_downbeat(NoteSequence.of(notes), beat, 0.0)
    assert downbeat == pytest.approx(0.0, abs=1e-6)


def test_downbeat_shifts_with_the_music() -> None:
    beat = 0.5
    notes = []
    for bar in range(8):
        start = bar * beat * 4 + beat  # bars begin on the second beat of the grid
        notes.append(Note(onset=start, pitch=40, duration=beat))
        for offset in (1, 2, 3):
            notes.append(Note(onset=start + offset * beat, pitch=72, duration=beat * 0.5))

    assert find_downbeat(NoteSequence.of(notes), beat, 0.0) == pytest.approx(beat, abs=1e-6)


# ── key ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "key,pitches",
    [
        ("C major", [60, 62, 64, 65, 67, 69, 71]),
        ("G major", [67, 69, 71, 72, 74, 76, 78]),
        ("F major", [65, 67, 69, 70, 72, 74, 76]),
    ],
)
def test_detects_major_keys(key: str, pitches: list[int]) -> None:
    tonic = pitches[0]
    notes = [Note(onset=i * 0.5, pitch=p, duration=0.5) for i, p in enumerate(pitches)]
    # Lean on the tonic, as real music does.
    notes.append(Note(onset=len(pitches) * 0.5, pitch=tonic, duration=2.0))

    found, confidence = estimate_key(NoteSequence.of(notes))
    assert found == key
    assert confidence > 0.0


def test_key_of_generated_music_matches_what_it_was_written_in() -> None:
    sequence = generate(seed=11, bars=16)
    found, _ = estimate_key(sequence)
    # Relative major and minor share a pitch-class set, so either is acceptable.
    tonic = found.split()[0]
    assert tonic in sequence.key or sequence.key.split()[0] in found


def test_empty_sequence_has_no_key() -> None:
    with pytest.raises(ScoreError, match="empty sequence"):
        estimate_key(NoteSequence())


# ── hands ────────────────────────────────────────────────────────────


def test_existing_hand_labels_are_swapped_if_inverted() -> None:
    """Stage 5 orders palettes by pixel count, so labels arrive arbitrary."""
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=40, duration=1.0, hand="R"),  # bass labelled right
            Note(onset=0.0, pitch=80, duration=1.0, hand="L"),
        ]
    )
    fixed = assign_hands(sequence)
    by_pitch = {n.pitch: n.hand for n in fixed}
    assert by_pitch[40] == "L"
    assert by_pitch[80] == "R"


def test_correct_hand_labels_are_left_alone() -> None:
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=40, duration=1.0, hand="L"),
            Note(onset=0.0, pitch=80, duration=1.0, hand="R"),
        ]
    )
    by_pitch = {n.pitch: n.hand for n in assign_hands(sequence)}
    assert by_pitch[40] == "L" and by_pitch[80] == "R"


def test_single_track_is_split_by_pitch() -> None:
    notes = []
    for i in range(16):
        notes.append(Note(onset=i * 0.25, pitch=45 + (i % 3), duration=0.2, hand="R"))
        notes.append(Note(onset=i * 0.25, pitch=76 + (i % 3), duration=0.2, hand="R"))

    split = assign_hands(NoteSequence.of(notes))
    assert all(n.hand == "L" for n in split if n.pitch < 60)
    assert all(n.hand == "R" for n in split if n.pitch > 70)


def test_hand_split_follows_the_music_up_the_keyboard() -> None:
    """A fixed middle-C split would cut straight through a passage that moves."""
    notes = []
    for i in range(16):
        base = 40 + i * 2  # both hands climb together
        notes.append(Note(onset=i * 0.25, pitch=base, duration=0.2, hand="R"))
        notes.append(Note(onset=i * 0.25, pitch=base + 24, duration=0.2, hand="R"))

    split = assign_hands(NoteSequence.of(notes))
    pairs = {}
    for note in split:
        pairs.setdefault(round(note.onset, 3), []).append(note)
    for group in pairs.values():
        hands = {n.hand for n in group}
        assert hands == {"L", "R"}, "each simultaneous pair should straddle the split"


def _hand_config(mode: str) -> Config:
    return Config(score=ScoreConfig(hand_mode=mode))


def test_hand_mode_none_puts_everything_on_one_staff() -> None:
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=40, duration=1.0, hand="L"),
            Note(onset=0.0, pitch=80, duration=1.0, hand="R"),
        ]
    )
    assert {n.hand for n in assign_hands(sequence, _hand_config("none"))} == {"R"}


def test_hand_mode_split_uses_a_fixed_middle_c_boundary() -> None:
    sequence = NoteSequence.of(
        [Note(onset=0.0, pitch=p, duration=0.5) for p in (48, 59, 60, 72)]
    )
    by_pitch = {n.pitch: n.hand for n in assign_hands(sequence, _hand_config("split"))}
    assert by_pitch == {48: "L", 59: "L", 60: "R", 72: "R"}


def test_hand_mode_pitch_ignores_existing_labels() -> None:
    """Colour said one thing; the user asked for a pitch split instead."""
    sequence = NoteSequence.of(
        [
            Note(onset=0.0, pitch=40, duration=1.0, hand="R"),
            Note(onset=0.0, pitch=44, duration=1.0, hand="R"),
            Note(onset=0.0, pitch=80, duration=1.0, hand="R"),
            Note(onset=0.0, pitch=84, duration=1.0, hand="R"),
        ]
    )
    by_pitch = {n.pitch: n.hand for n in assign_hands(sequence, _hand_config("pitch"))}
    assert by_pitch[40] == "L" and by_pitch[84] == "R"


def test_fixed_tempo_overrides_the_estimate() -> None:
    sequence = _grid(96.0)
    config = Config(score=ScoreConfig(fixed_tempo=140.0))
    analysis = analyze(sequence, config)
    assert analysis.tempo == pytest.approx(140.0)
    assert analysis.tempo_confidence == 1.0


def test_fixed_key_overrides_the_estimate() -> None:
    sequence = _grid(120.0)
    analysis = analyze(sequence, Config(score=ScoreConfig(fixed_key="Eb minor")))
    assert analysis.key == "Eb minor"
    assert analysis.key_confidence == 1.0


def test_quantize_handles_a_disabled_grid_itself() -> None:
    """It is public API and "no grid" is a setting, so it must not divide by zero."""
    sequence = _grid(120.0)
    analysis = analyze(sequence)

    result, skipped = quantize(sequence, analysis, Config(score=ScoreConfig(steps_per_beat=0)))

    assert skipped == 0
    assert [n.onset for n in result] == [n.onset for n in sequence]
    assert result.tempo == pytest.approx(analysis.tempo)
    assert result.key == analysis.key


def test_quantize_with_no_grid_does_not_mutate_its_input() -> None:
    sequence = _grid(120.0)
    analysis = analyze(sequence)
    before = sequence.tempo

    quantize(sequence, analysis, Config(score=ScoreConfig(steps_per_beat=0)))
    assert sequence.tempo == before


def test_quantization_can_be_switched_off() -> None:
    sequence = generate(seed=3, bars=4, tempo=96.0)
    off = Config(score=ScoreConfig(steps_per_beat=0))
    result, analysis = postprocess(sequence, off)

    assert [n.onset for n in result] == [n.onset for n in sequence]
    assert result.tempo == pytest.approx(analysis.tempo)
    assert result.key == analysis.key


# ── quantization ─────────────────────────────────────────────────────


def test_quantization_snaps_near_misses() -> None:
    sequence = _grid(120.0)
    analysis = analyze(sequence)
    nudged = NoteSequence.of(
        [Note(n.onset + 0.01, n.pitch, n.duration, n.hand, n.velocity) for n in sequence]
    )

    result, skipped = quantize(nudged, analysis)
    assert skipped == 0
    step = analysis.beat / DEFAULT.score.steps_per_beat
    for note in result:
        offset = (note.onset - analysis.beat_phase) % step
        assert min(offset, step - offset) < 1e-6


def test_quantization_leaves_far_outliers_alone() -> None:
    """A note nowhere near the grid keeps its measured time rather than lying."""
    sequence = _grid(120.0)
    analysis = analyze(sequence)
    step = analysis.beat / DEFAULT.score.steps_per_beat
    stray = Note(onset=1.0 + step * 0.45, pitch=61, duration=0.2)

    result, skipped = quantize(NoteSequence.of([*sequence, stray]), analysis)
    assert skipped == 1
    assert any(n.pitch == 61 and n.onset == pytest.approx(stray.onset) for n in result)


def test_quantization_never_produces_a_zero_duration() -> None:
    sequence = NoteSequence.of(
        [Note(onset=i * 0.5, pitch=60, duration=0.004) for i in range(12)]
    )
    analysis = analyze(sequence)
    result, _ = quantize(sequence, analysis)
    assert all(n.duration > 0 for n in result)


def test_tighter_tolerance_skips_more() -> None:
    sequence = _grid(120.0)
    analysis = analyze(sequence)
    nudged = NoteSequence.of(
        [Note(n.onset + 0.03, n.pitch, n.duration, n.hand, n.velocity) for n in sequence]
    )

    loose = quantize(nudged, analysis, Config(score=ScoreConfig(max_shift=0.45)))[1]
    tight = quantize(nudged, analysis, Config(score=ScoreConfig(max_shift=0.05)))[1]
    assert tight > loose


# ── end to end ───────────────────────────────────────────────────────


def test_postprocess_annotates_the_sequence() -> None:
    sequence = generate(seed=6, bars=8, tempo=96.0)
    result, analysis = postprocess(sequence)

    assert result.tempo == pytest.approx(analysis.tempo)
    assert result.key == analysis.key
    assert len(result) == len(sequence)
    assert analysis.tempo == pytest.approx(96.0, rel=0.05)


def test_postprocess_keeps_both_hands() -> None:
    sequence = generate(seed=7, bars=8)
    result, _ = postprocess(sequence)
    assert {n.hand for n in result} == {"L", "R"}


# ── key finding ──────────────────────────────────────────────────────


def test_key_found_for_every_key_the_generator_writes() -> None:
    """Both styles, several seeds, every key."""
    from dropscore.synth.music import KEYS, generate

    wrong = []
    for key in sorted(KEYS):
        for sustained in (False, True):
            for seed in range(4):
                sequence = generate(seed=seed, key=key, sustained=sustained)
                found, _ = estimate_key(sequence)
                if found != key:
                    wrong.append(f"{key}/{'sustained' if sustained else 'normal'}"
                                 f"/seed{seed} -> {found}")

    assert not wrong, "mis-identified: " + ", ".join(wrong)


def test_key_rejects_a_key_whose_defining_notes_never_sound() -> None:
    """The real-recording case, as pitch-class weights.

    Measured off a transcription of an actual video: heavy B and E, no F, and
    — decisively — no G# and no D# anywhere. E major needs both. Plain
    Krumhansl-Schmuckler still ranked E major first (0.7305 to E minor's
    0.7128), because the shared tonic and dominant carry the correlation and
    nothing charges a key for the notes that contradict it.
    """
    weights = {"B": 60.3, "E": 52.9, "A": 28.8, "C": 25.1, "D": 5.1, "G": 4.4}
    pitch_of = {"C": 60, "D": 62, "E": 64, "G": 67, "A": 69, "B": 71}

    notes, onset = [], 0.0
    for name, weight in weights.items():
        # Split each class into a few notes so the sequence is note-like.
        for _ in range(4):
            notes.append(
                Note(onset=onset, pitch=pitch_of[name], duration=weight / 4, hand="R")
            )
            onset += 0.25

    found, _ = estimate_key(NoteSequence.of(notes))

    assert found != "E major", "picked a key needing G# and D#, neither of which sound"
    assert found in {"E minor", "A minor"}, f"expected a natural-minor reading, got {found}"


def test_out_of_scale_penalty_is_off_when_zero() -> None:
    """The penalty is a knob, and zero must restore plain correlation."""
    import dataclasses

    from dropscore.config import DEFAULT
    from dropscore.synth.music import generate

    plain = DEFAULT.evolve(
        score=dataclasses.replace(DEFAULT.score, out_of_scale_penalty=0.0)
    )
    sequence = generate(seed=0, key="G major")

    assert estimate_key(sequence, plain)[0] == "G major"
