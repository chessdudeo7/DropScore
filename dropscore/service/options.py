"""The settings the frontend exposes, and how they reach the pipeline.

Kept apart from the endpoints so both submission paths share one definition:
the upload endpoint receives these as a JSON string in a form field (multipart
has no nested objects), the URL endpoint as a nested object. Both end up here.

Every field maps onto something the pipeline already supported — this is
plumbing, not new behaviour. The panel existed in the UI before the backend did
and was silently ignored, which is worse than not offering it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..config import Config, DEFAULT
from ..export import FORMATS

# UI value -> quantization grid, in steps per beat. "0" means leave the
# measured times alone.
QUANTIZE_STEPS = {"8": 2, "16": 4, "32": 8, "0": 0}

DEFAULT_FORMATS = ("json", "midi", "musicxml")


class TranscribeOptions(BaseModel):
    """What the user chose in the settings panel."""

    hands: Literal["color", "split", "pitch", "none"] = "color"
    quantize: Literal["8", "16", "32", "0"] = "16"
    tempo: Literal["auto", "fixed"] = "auto"
    fixed_tempo: float = Field(default=120.0, gt=20, lt=400)
    key: str = "auto"
    formats: list[str] = Field(default_factory=lambda: list(DEFAULT_FORMATS))

    @field_validator("formats")
    @classmethod
    def _known_formats(cls, value: list[str]) -> list[str]:
        unknown = [f for f in value if f not in FORMATS]
        if unknown:
            raise ValueError(f"unknown output format(s): {', '.join(unknown)}")
        # JSON is always written: it is what the results view is drawn from.
        return sorted({*value, "json"}) if value else list(DEFAULT_FORMATS)

    @field_validator("key")
    @classmethod
    def _known_key(cls, value: str) -> str:
        if value == "auto":
            return value
        parts = value.split()
        if len(parts) != 2 or parts[1] not in {"major", "minor"}:
            raise ValueError(f"key must be 'auto' or like 'F major', got {value!r}")
        return value

    def apply(self, config: Config = DEFAULT) -> Config:
        """Fold these choices into a Config for the pipeline to run under."""
        score = replace(
            config.score,
            hand_mode=self.hands,
            steps_per_beat=QUANTIZE_STEPS[self.quantize],
            fixed_tempo=self.fixed_tempo if self.tempo == "fixed" else None,
            fixed_key=None if self.key == "auto" else self.key,
        )
        return config.evolve(score=score)

    @classmethod
    def parse(cls, raw: str | dict[str, Any] | None) -> "TranscribeOptions":
        """Build from a form field, a nested object, or nothing at all."""
        if raw is None or raw == "":
            return cls()
        if isinstance(raw, str):
            raw = json.loads(raw)
        return cls.model_validate(raw)
