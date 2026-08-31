"""Synthetic falling-tile video generation, for ground truth.

See docs/ROADMAP.md on why this exists before any of the vision code: without it,
stages 3-5 are tuned by eye and regress silently.
"""

from .music import generate
from .renderer import RenderError, RenderSpec, SynthRenderer, render
from .themes import DEFAULT_THEME, THEMES, Theme, get_theme

__all__ = [
    "generate",
    "render",
    "RenderError",
    "RenderSpec",
    "SynthRenderer",
    "THEMES",
    "DEFAULT_THEME",
    "Theme",
    "get_theme",
]
