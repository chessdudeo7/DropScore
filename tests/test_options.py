"""The settings panel's choices, and how they reach the pipeline."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pydantic", reason="needs the [service] extra")

from dropscore.config import DEFAULT  # noqa: E402
from dropscore.service.options import TranscribeOptions  # noqa: E402


def test_defaults_match_the_pipeline_defaults() -> None:
    """An untouched panel must not change how the pipeline behaves."""
    config = TranscribeOptions().apply(DEFAULT)
    assert config.score.steps_per_beat == DEFAULT.score.steps_per_beat
    assert config.score.hand_mode == DEFAULT.score.hand_mode
    assert config.score.fixed_tempo is None
    assert config.score.fixed_key is None


@pytest.mark.parametrize(
    "choice,steps",
    [("8", 2), ("16", 4), ("32", 8), ("0", 0)],
)
def test_quantize_choice_maps_to_a_grid(choice: str, steps: int) -> None:
    assert TranscribeOptions(quantize=choice).apply().score.steps_per_beat == steps


@pytest.mark.parametrize("mode", ["color", "split", "pitch", "none"])
def test_hand_modes_pass_through(mode: str) -> None:
    assert TranscribeOptions(hands=mode).apply().score.hand_mode == mode


def test_fixed_tempo_is_only_applied_when_asked() -> None:
    assert TranscribeOptions(tempo="auto", fixed_tempo=90).apply().score.fixed_tempo is None
    assert TranscribeOptions(tempo="fixed", fixed_tempo=90).apply().score.fixed_tempo == 90


def test_key_auto_leaves_detection_alone() -> None:
    assert TranscribeOptions(key="auto").apply().score.fixed_key is None
    assert TranscribeOptions(key="F major").apply().score.fixed_key == "F major"


def test_applying_options_does_not_mutate_the_shared_config() -> None:
    """Config is frozen, but the store hands the same instance to every job."""
    TranscribeOptions(hands="none", quantize="0").apply(DEFAULT)
    assert DEFAULT.score.hand_mode == "color"
    assert DEFAULT.score.steps_per_beat == 4


# ── validation ───────────────────────────────────────────────────────


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown output format"):
        TranscribeOptions(formats=["midi", "ogg"])


def test_json_is_always_included() -> None:
    """The results view is drawn from it, so it cannot be opted out of."""
    assert "json" in TranscribeOptions(formats=["midi"]).formats


def test_empty_format_list_falls_back_to_the_defaults() -> None:
    assert TranscribeOptions(formats=[]).formats == ["json", "midi", "musicxml"]


@pytest.mark.parametrize("bad", ["F", "F sharp", "H major", "major"])
def test_malformed_key_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="key must be"):
        TranscribeOptions(key=bad)


@pytest.mark.parametrize("bad", [0, 19, 500])
def test_implausible_fixed_tempo_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        TranscribeOptions(fixed_tempo=bad)


def test_unknown_hand_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        TranscribeOptions(hands="telepathy")


# ── parsing from either transport ────────────────────────────────────


def test_parses_the_multipart_form_field() -> None:
    raw = json.dumps({"hands": "pitch", "quantize": "32", "formats": ["midi"]})
    options = TranscribeOptions.parse(raw)
    assert options.hands == "pitch"
    assert options.apply().score.steps_per_beat == 8


def test_parses_a_nested_object() -> None:
    assert TranscribeOptions.parse({"hands": "none"}).hands == "none"


@pytest.mark.parametrize("empty", [None, ""])
def test_absent_options_give_defaults(empty) -> None:
    assert TranscribeOptions.parse(empty) == TranscribeOptions()


def test_malformed_json_raises_rather_than_silently_defaulting() -> None:
    with pytest.raises(ValueError):
        TranscribeOptions.parse("{not json")
