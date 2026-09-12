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


def _started_late(sequence: NoteSequence, offset: float) -> NoteSequence:
    """As a recording is: begun at some arbitrary moment, not on a downbeat."""
    return NoteSequence.of(
        [Note(n.onset + offset, n.pitch, n.duration, n.hand, n.velocity) for n in sequence],
        tempo=sequence.tempo,
    )


@pytest.mark.parametrize("beats_per_bar", [3, 4])
@pytest.mark.parametrize("offset_beats", [0.25, 0.5, 0.75, 2.5])
def test_the_beat_is_found_wherever_the_recording_starts(
    beats_per_bar: int, offset_beats: float
) -> None:
    """Every synthetic clip starts on a downbeat at time zero, which hid that
    the phase chose whichever tatum line came first after the start: on a real
    capture that put every beat half a beat late."""
    from dropscore.score import analyze  # noqa: PLC0415

    tempo = 96.0
    beat = 60.0 / tempo
    offset = offset_beats * beat
    piece = _started_late(generate(seed=11, tempo=tempo, beats_per_bar=beats_per_bar), offset)

    result = analyze(piece)
    error = ((result.beat_phase - offset % beat) / beat + 0.5) % 1.0 - 0.5
    assert abs(error) < 0.05, f"beat grid {error * 4:+.2f} sixteenths off"


@pytest.mark.parametrize("beats_per_bar", [3, 4])
def test_the_meter_is_measured_not_assumed(beats_per_bar: int) -> None:
    from dropscore.score import analyze  # noqa: PLC0415

    for seed in (2, 5, 9):
        piece = _started_late(generate(seed=seed, tempo=100.0, beats_per_bar=beats_per_bar), 1.3)
        assert analyze(piece).beats_per_bar == beats_per_bar, f"seed {seed}"


def test_a_steady_pulse_under_a_slow_melody_is_heard_in_three() -> None:
    """The shape of a real capture that came back in four: a short repeated
    note on every beat, and over it a melody and bass moving once a bar in
    dotted halves. The pulse is the same on every beat; only the long notes
    say where the bar is."""
    from dropscore.score import analyze  # noqa: PLC0415

    beat = 0.6
    notes = [Note(onset=0.37 + i * beat, pitch=64, duration=0.15) for i in range(72)]
    melody = [69, 71, 71, 72, 71, 71, 69, 72, 71, 71, 72, 74, 76]
    for bar, pitch in enumerate(melody, start=4):
        start = 0.37 + bar * 3 * beat
        notes.append(Note(onset=start, pitch=pitch, duration=1.6))
        notes.append(Note(onset=start, pitch=pitch - 12, duration=1.6))

    result = analyze(NoteSequence.of(notes))
    assert result.beats_per_bar == 3
    assert result.tempo == pytest.approx(100.0, rel=0.02)


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


def test_colour_labels_are_ignored_when_they_are_not_registers() -> None:
    """A one-colour video whose accidentals render darker yields two palettes.

    The split that comes back is black keys against white, not left against
    right, so the two groups interleave across the whole keyboard. Believing it
    would put bass notes on the treble staff.
    """
    black = {1, 3, 6, 8, 10}

    def coloured(onset: float, pitch: int) -> Note:
        return Note(
            onset=onset,
            pitch=pitch,
            duration=0.2,
            hand="R" if pitch % 12 in black else "L",
        )

    notes = []
    for i in range(16):
        notes.append(coloured(i * 0.25, 45 + (i % 6)))  # bass line
        notes.append(coloured(i * 0.25, 76 + (i % 6)))  # treble line, same instant

    split = assign_hands(NoteSequence.of(notes))
    assert all(n.hand == "L" for n in split if n.pitch < 60)
    assert all(n.hand == "R" for n in split if n.pitch > 70)


def test_a_lopsided_colour_split_is_not_believed_for_being_lopsided() -> None:
    """Most notes in one colour, the rest scattered over the same range.

    Nothing about pitch separates the colours, but a boundary below every
    note already sorts 78% of them correctly. As a raw accuracy against 0.75
    that passed, and on a real capture it put 172 notes on the wrong staff.
    """
    import random  # noqa: PLC0415

    rng = random.Random(3)
    notes = []
    for i in range(200):
        hand = "R" if i % 9 < 2 else "L"  # 22% one way, 78% the other
        notes.append(Note(onset=i * 0.2, pitch=rng.randint(45, 88), duration=0.15, hand=hand))

    split = assign_hands(NoteSequence.of(notes))
    left = [n.pitch for n in split.hand("L")]
    right = [n.pitch for n in split.hand("R")]
    assert sorted(left)[len(left) // 2] < sorted(right)[len(right) // 2] - 12, "the colours were believed"


def test_hands_that_share_the_middle_of_the_keyboard_are_still_believed() -> None:
    """Real hands cross and overlap; only a split along some other axis is
    rejected. These two lines share four semitones and must survive."""
    notes = []
    for i in range(12):
        notes.append(Note(onset=i * 0.25, pitch=52 + (i % 8), duration=0.2, hand="L"))
        notes.append(Note(onset=i * 0.25, pitch=56 + (i % 8), duration=0.2, hand="R"))

    split = assign_hands(NoteSequence.of(notes))
    assert [n.hand for n in split] == [n.hand for n in NoteSequence.of(notes)]


def test_a_handful_of_notes_does_not_overrule_the_colours() -> None:
    """Under the evidence floor an unseparable pair could be chance."""
    notes = [
        Note(onset=0.0, pitch=60, duration=0.2, hand="L"),
        Note(onset=0.5, pitch=64, duration=0.2, hand="R"),
        Note(onset=1.0, pitch=62, duration=0.2, hand="L"),
    ]
    assert [n.hand for n in assign_hands(NoteSequence.of(notes))] == ["L", "R", "L"]


def test_single_track_is_split_by_pitch() -> None:
    notes = []
    for i in range(16):
        notes.append(Note(onset=i * 0.25, pitch=45 + (i % 3), duration=0.2, hand="R"))
        notes.append(Note(onset=i * 0.25, pitch=76 + (i % 3), duration=0.2, hand="R"))

    split = assign_hands(NoteSequence.of(notes))
    assert all(n.hand == "L" for n in split if n.pitch < 60)
    assert all(n.hand == "R" for n in split if n.pitch > 70)


def test_the_split_follows_the_music_but_only_so_far() -> None:
    """A fixed middle-C split would cut straight through a passage that moves,
    so the boundary travels with the music -- within a few semitones of middle
    C, and no further.

    Chasing the music without limit is what put a repeated pedal note on the
    bass staff for a whole page of a real capture: the melody above it lifted
    the boundary over the pedal. Which staff a note is written on is a question
    about register, and the staves meet at middle C.
    """
    notes = []
    for i in range(16):
        base = 52 + (i % 6)  # moves, but stays within reach of middle C
        notes.append(Note(onset=i * 0.25, pitch=base, duration=0.2, hand="R"))
        notes.append(Note(onset=i * 0.25, pitch=base + 14, duration=0.2, hand="R"))

    split = assign_hands(NoteSequence.of(notes))
    pairs: dict[float, list[Note]] = {}
    for note in split:
        pairs.setdefault(round(note.onset, 3), []).append(note)
    for group in pairs.values():
        assert {n.hand for n in group} == {"L", "R"}, "a pair failed to straddle the split"


def test_a_passage_far_above_middle_c_is_written_on_one_staff() -> None:
    """Both voices high is both voices on the treble staff, which is how it is
    printed. The boundary does not climb after them to manufacture a bass part
    out of the lower one."""
    notes = []
    for i in range(16):
        notes.append(Note(onset=i * 0.25, pitch=76 + (i % 3), duration=0.2, hand="R"))
        notes.append(Note(onset=i * 0.25, pitch=88 + (i % 3), duration=0.2, hand="R"))

    split = assign_hands(NoteSequence.of(notes))
    assert all(n.hand == "R" for n in split), "a high passage was split across the staves"


def test_a_sparse_bass_does_not_drag_the_split_up_into_the_melody() -> None:
    """The melody outnumbers the bass, and density must not decide the boundary.

    A left hand playing one note per bar under a right hand playing eight puts
    most of the notes at the top of the register. Any boundary that clusters
    the pitches is pulled up among them and hands the melody's lower notes to
    the bass staff.
    """
    notes = []
    for bar in range(8):
        notes.append(Note(onset=bar * 2.0, pitch=45, duration=1.8, hand="L"))
        for step in range(8):
            pitch = 67 + (step % 5) * 2  # melody sits well above the bass
            notes.append(
                Note(onset=bar * 2.0 + step * 0.25, pitch=pitch, duration=0.2, hand="R")
            )

    split = assign_hands(NoteSequence.of(notes))
    assert all(n.hand == "L" for n in split if n.pitch == 45)
    assert all(n.hand == "R" for n in split if n.pitch >= 67)


def test_a_key_signature_does_not_assert_a_note_the_music_never_plays() -> None:
    """Two keys can fit a piece equally and differ in what they claim.

    The sounding weight here is the one measured on a real capture: B heaviest,
    then E, then C and A, and no F of either kind anywhere. E minor and A minor
    both fit it perfectly -- nothing falls outside either scale -- and E minor
    won on the shape of the template alone, putting an F sharp in the signature
    of a piece that never sounds one. The printed music has no signature at all.
    """
    plan = [(71, 48), (64, 31), (72, 20), (69, 19), (74, 4), (67, 2)]
    notes, when = [], 0.0
    for pitch, weight in plan:
        for _ in range(weight):
            notes.append(Note(onset=when, pitch=pitch, duration=1.0))
            when += 0.6

    key, _ = estimate_key(NoteSequence.of(notes))
    assert key in {"A minor", "C major"}, f"chose {key}, whose signature is never played"


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


# ── which multiple of the tatum is the beat ──────────────────────────


def test_beat_is_not_forced_to_a_fixed_multiple() -> None:
    """A slow piece of eighths and quarters must not read as all sixteenths.

    Taken from a real recording: inter-onset intervals cluster at 0.6s and
    0.3s. Read with a beat of 1.2s those become sixteenths and eighths at an
    implausible 50 BPM; read at 0.6s they are quarters and eighths at 100.
    Preferring a fixed ``steps_per_beat`` always chose the former, because it
    makes the tatum a sixteenth by construction.
    """
    notes, t = [], 0.0
    for bar in range(8):
        for step in range(4):
            notes.append(Note(onset=t, pitch=64, duration=0.55, hand="R"))
            notes.append(Note(onset=t + 0.3, pitch=59, duration=0.28, hand="L"))
            t += 0.6

    beat, _, _ = estimate_tempo(NoteSequence.of(notes))

    assert 60.0 / beat == pytest.approx(100.0, rel=0.05), (
        f"read the beat as {60.0 / beat:.0f} BPM; 0.6s IOIs are quarters, "
        "not eighths"
    )


def test_uniform_stream_falls_back_to_the_conventional_beat() -> None:
    """A study of nothing but sixteenths says nothing about its own metre.

    Every candidate period is supported identically, and the note values carry
    no information either, so the conventional four tatums to the beat is all
    there is to go on and must still be reachable.
    """
    beat, _, _ = estimate_tempo(_grid(96.0, subdivision=4))
    assert 60.0 / beat == pytest.approx(96.0, rel=0.03)


def test_repeats_at_is_not_biased_toward_short_lags() -> None:
    """Only onsets with room for a partner may vote.

    Counting onsets near the end as misses made every short lag look better
    supported than a long one purely because the clip stops.
    """
    import numpy as np

    from dropscore.score import _repeats_at

    onsets = np.arange(32) * 0.125
    short = _repeats_at(onsets, 0.375, 0.06)
    long = _repeats_at(onsets, 0.5, 0.06)

    assert short == pytest.approx(long, abs=1e-9), (
        f"uniform stream scored {short:.3f} at 3 steps and {long:.3f} at 4"
    )


def test_tempo_errors_are_octaves_not_arbitrary() -> None:
    """Across the generator's range, any miss is a power-of-two reading.

    Which metrical level is the beat is not decidable from onsets alone, so
    an octave is a choice rather than a mistake. A ratio like two thirds is
    a mistake, and must not happen.

    Several seeds, because one is not a sample. Written against seed 0 alone
    this passed while a two-thirds reading sat in the generator's range the
    whole time, found only by sweeping seeds by hand.
    """
    import math

    from dropscore.synth.music import generate

    bad = []
    for bpm in (60, 72, 80, 96, 110, 120, 144, 152, 160):
        for seed in (0, 47, 113):
            for sustained in (False, True):
                sequence = generate(seed=seed, tempo=float(bpm), sustained=sustained)
                beat, _, _ = estimate_tempo(sequence)
                octaves = math.log2((60.0 / beat) / bpm)
                if abs(octaves - round(octaves)) > 0.03:
                    bad.append(f"{bpm} BPM (seed {seed}) -> {60.0 / beat:.1f}")

    assert not bad, "non-octave tempo errors: " + ", ".join(bad)


def test_slow_pieces_are_rarely_read_at_double_speed() -> None:
    """Slow tempi were doubled, and by two separate mechanisms.

    The prior sat far enough above them that twice the tempo scored better on
    nearness alone, and the modal *played* duration -- always shorter than the
    value written -- made the music look finer than it was, which argues for a
    faster beat. Both pushed the same way on exactly the pieces where the onset
    evidence already favoured the truth.

    Not zero. Which metrical level is the beat is not decidable from onsets
    alone, and one reading in eighteen here is still doubled; over a wider
    sweep of 120 pieces the rate is 3%, against 18% before. The bound is set to
    catch a regression to that, not to claim the ambiguity is gone.
    """
    from dropscore.synth.music import generate

    doubled = []
    for bpm in (60, 63, 66, 72, 80, 88):
        for seed in (0, 5, 11):
            sequence = generate(seed=seed, tempo=float(bpm))
            beat, _, _ = estimate_tempo(sequence)
            if (60.0 / beat) / bpm > 1.5:
                doubled.append(f"{bpm} BPM (seed {seed}) -> {60.0 / beat:.1f}")

    assert len(doubled) <= 1, "read at double speed: " + ", ".join(doubled)


def test_tempo_never_reads_half_speed() -> None:
    """Doubling renotates in coarser values; halving fills the page with
    sixteenths, which is the failure a reader actually notices."""
    from dropscore.synth.music import generate

    slow = []
    for bpm in (60, 72, 80, 96, 110, 120, 144, 160):
        for sustained in (False, True):
            sequence = generate(seed=0, tempo=float(bpm), sustained=sustained)
            beat, _, _ = estimate_tempo(sequence)
            if (60.0 / beat) / bpm < 0.95:
                slow.append(f"{bpm} BPM -> {60.0 / beat:.1f}")

    assert not slow, "read slower than written: " + ", ".join(slow)


# ── written durations against played ones ────────────────────────────


def test_a_staccato_quarter_is_written_as_a_quarter() -> None:
    """Articulation is a dot over the note, not a shorter note and rests.

    A real capture played its repeated quarters at a quarter of their length.
    Engraved as played, they became sixteenths with rests after them, and only
    39% of the written values on the page matched the printed music.
    """
    from dropscore.score import notate_durations  # noqa: PLC0415

    beat = 0.6
    notes = [Note(onset=i * beat, pitch=64, duration=0.15) for i in range(12)]
    sequence = NoteSequence.of(notes)
    analysis = analyze(sequence)

    written = notate_durations(sequence, analysis)
    values = [n.duration / analysis.beat for n in written][:-1]  # the last has no successor
    assert all(abs(v - 1.0) < 0.05 for v in values), f"wrote {values[:4]}"


def test_a_real_rest_is_still_written() -> None:
    """A note followed by a bar of silence is not a note held for a bar."""
    from dropscore.score import notate_durations  # noqa: PLC0415

    beat = 0.6
    notes = [Note(onset=i * 4 * beat, pitch=64, duration=0.3) for i in range(8)]
    sequence = NoteSequence.of(notes)
    analysis = analyze(sequence)

    written = notate_durations(sequence, analysis)
    values = [n.duration / analysis.beat for n in written][:-1]
    assert all(v < 2.0 for v in values), f"filled a real silence: {values[:4]}"


def test_a_note_inside_a_chord_is_written_as_reaching_the_next_one() -> None:
    """Measured to the nearest onset, a note struck with another had a gap of
    nothing, so nothing in a chord was ever written as reaching anything."""
    from dropscore.score import notate_durations  # noqa: PLC0415

    beat = 0.6
    notes = []
    for i in range(8):
        notes.append(Note(onset=i * beat, pitch=64, duration=0.2))
        notes.append(Note(onset=i * beat, pitch=67, duration=0.2))
    sequence = NoteSequence.of(notes)
    analysis = analyze(sequence)

    written = notate_durations(sequence, analysis)
    values = [n.duration / analysis.beat for n in written if n.onset < 7 * beat - 0.01]
    assert all(abs(v - 1.0) < 0.05 for v in values), f"wrote {values[:4]}"


def _analysis(tempo: float = 100.0, key: str = "E minor"):
    from dropscore.score import Analysis

    return Analysis(
        tempo=tempo, beat=60.0 / tempo, beat_phase=0.0, downbeat_phase=0.0,
        beats_per_bar=4, key=key, tempo_confidence=0.7, key_confidence=0.1,
    )


def test_detached_notes_are_written_at_full_value() -> None:
    """A quarter released early is a detached quarter, not a dotted eighth.

    Held-key time is what a falling tile measures, and a source held at about
    three quarters of nominal made the dotted eighth the commonest value on a
    real transcription -- 88 notes of 301 -- with a sixteenth rest after each.
    """
    from dropscore.score import notate_durations

    beat = 0.6
    notes = [
        Note(onset=i * beat, pitch=64, duration=beat * 0.75, hand="R")
        for i in range(8)
    ]

    written = sorted(notate_durations(NoteSequence.of(notes), _analysis()))

    # The last note has no following onset to reach, so it keeps what it was
    # played at — there is nothing to infer a written value from.
    assert all(n.duration == pytest.approx(beat, abs=1e-6) for n in written[:-1]), (
        f"still writing {sorted({round(n.duration, 3) for n in written[:-1]})}"
    )
    assert written[-1].duration == pytest.approx(beat * 0.75, abs=1e-6)


def test_a_real_rest_is_left_alone() -> None:
    """Half the gap is a rest, not articulation, and must stay a rest.

    Measured over a gap of two beats. It used to be measured over one, where
    this now fills: sheet music for a real capture showed its repeated quarters
    held a quarter of the way and printed as quarters, no rests anywhere, so
    within a single beat the silence is articulation whatever its length.
    Beyond a beat the old reading stands and a rest is a rest.
    """
    from dropscore.score import notate_durations

    beat = 0.6
    notes = [
        Note(onset=i * 2 * beat, pitch=64, duration=beat, hand="R")
        for i in range(6)
    ]

    written = notate_durations(NoteSequence.of(notes), _analysis())

    assert all(n.duration == pytest.approx(beat, abs=1e-6) for n in written)


def test_notation_never_shortens_a_note() -> None:
    """Overlap is two voices, and cutting one deletes a real sustain."""
    from dropscore.score import notate_durations

    notes = [
        Note(onset=0.0, pitch=48, duration=4.0, hand="L"),   # held under
        Note(onset=0.6, pitch=55, duration=0.45, hand="L"),  # moving over it
        Note(onset=1.2, pitch=57, duration=0.45, hand="L"),
    ]

    written = notate_durations(NoteSequence.of(notes), _analysis())
    held = next(n for n in written if n.pitch == 48)

    assert held.duration == pytest.approx(4.0), "shortened a sustained note"
    for original, result in zip(sorted(notes), sorted(written)):
        assert result.duration >= original.duration - 1e-9


def test_notation_durations_are_off_at_zero_ratio() -> None:
    from dropscore.config import DEFAULT
    from dropscore.score import notate_durations
    import dataclasses

    off = DEFAULT.evolve(score=dataclasses.replace(DEFAULT.score, legato_ratio=0.0))
    notes = [Note(onset=i * 0.6, pitch=64, duration=0.45, hand="R") for i in range(4)]
    sequence = NoteSequence.of(notes)

    written = notate_durations(sequence, _analysis(), off)
    assert [n.duration for n in written] == [n.duration for n in sequence]


def test_midi_keeps_what_was_played() -> None:
    """Only notation is rewritten; the performance record stays faithful."""
    import dataclasses

    from dropscore.export import midi
    from dropscore.notes import NoteSequence as NS

    beat = 0.6
    notes = [Note(onset=i * beat, pitch=64, duration=beat * 0.75, hand="R")
             for i in range(4)]
    sequence = NS.of(notes, tempo=100.0)

    # quantize/postprocess are not involved here: the exporter must not be
    # quietly rewriting durations of its own accord.
    assert all(n.duration == pytest.approx(0.45) for n in sequence)
    assert midi is not None
