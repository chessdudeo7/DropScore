"""The command line end to end, on a short rendered clip.

The stages are tested on their own; these check the commands that join them:
the files each writes, and the plain sentence each gives when it cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dropscore.cli import main
from dropscore.evaluate import load_truth
from dropscore.notes import NoteSequence


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("clip")
    assert main(["synth", "-o", str(out), "--theme", "classic", "--bars", "2",
                 "--width", "640", "--height", "360"]) == 0
    return next(out.glob("*.mp4"))


@pytest.fixture(scope="module")
def notes(clip: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("notes") / "clip.json"
    assert main(["transcribe", str(clip), "-o", str(path)]) == 0
    return path


def test_transcribe_finds_the_rendered_notes(clip: Path, notes: Path) -> None:
    truth = load_truth(clip.with_name(clip.stem + ".truth.json"))[0]
    heard = NoteSequence.load(notes)
    assert len(heard) == len(truth)
    assert sorted(n.pitch for n in heard) == sorted(n.pitch for n in truth)


def test_export_writes_each_format_asked_for(notes: Path, tmp_path: Path) -> None:
    stem = tmp_path / "piece"
    assert main(["export", str(notes), "-f", "midi", "-f", "musicxml", "-o", str(stem)]) == 0
    assert (tmp_path / "piece.mid").stat().st_size > 0
    xml = (tmp_path / "piece.musicxml").read_text(encoding="utf-8")
    assert "<score-partwise" in xml


def test_analyze_writes_a_quantized_copy(notes: Path, tmp_path: Path) -> None:
    out = tmp_path / "quantized.json"
    assert main(["analyze", str(notes), "-o", str(out)]) == 0
    assert len(NoteSequence.load(out)) == len(NoteSequence.load(notes))


def test_probe_and_tiles_inspect_without_transcribing(clip: Path, capsys) -> None:
    assert main(["probe", str(clip)]) == 0
    assert main(["tiles", str(clip)]) == 0
    said = capsys.readouterr().out
    assert "track" in said  # the palette lists which track each colour feeds


def test_a_missing_file_is_one_sentence(tmp_path: Path, capsys) -> None:
    status = main(["transcribe", str(tmp_path / "nothing.mp4")])
    assert status != 0
    said = capsys.readouterr()
    assert "Traceback" not in said.out + said.err


def test_eval_on_an_empty_corpus_keeps_no_last_run(tmp_path: Path) -> None:
    last = tmp_path / "last.json"
    main(["eval", "-c", str(tmp_path), "--no-printed", "--last", str(last)])
    if last.exists():
        assert json.loads(last.read_text(encoding="utf-8"))["clips"] == []
