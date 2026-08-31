"""DropScore — turn falling-tile piano videos into sheet music."""

__version__ = "0.1.0"

from .config import DEFAULT, Config, VideoConfig
from .keyboard import COMMON_RANGES, KeyboardLayout, is_black, is_white
from .notes import Hand, Note, NoteSequence, pitch_name
from .sources import ResolvedSource, SourceError, parse_youtube_id, resolve
from .video import Frame, VideoError, VideoInfo, VideoReader

__all__ = [
    "__version__",
    # config
    "Config",
    "DEFAULT",
    "VideoConfig",
    # video
    "Frame",
    "VideoError",
    "VideoInfo",
    "VideoReader",
    # sources
    "ResolvedSource",
    "SourceError",
    "parse_youtube_id",
    "resolve",
    # music
    "Hand",
    "Note",
    "NoteSequence",
    "pitch_name",
    # keyboard
    "COMMON_RANGES",
    "KeyboardLayout",
    "is_black",
    "is_white",
]
