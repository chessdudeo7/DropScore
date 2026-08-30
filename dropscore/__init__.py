"""DropScore — turn falling-tile piano videos into sheet music."""

__version__ = "0.1.0"

from .config import DEFAULT, Config, VideoConfig
from .sources import ResolvedSource, SourceError, parse_youtube_id, resolve
from .video import Frame, VideoError, VideoInfo, VideoReader

__all__ = [
    "__version__",
    "Config",
    "DEFAULT",
    "VideoConfig",
    "Frame",
    "VideoError",
    "VideoInfo",
    "VideoReader",
    "ResolvedSource",
    "SourceError",
    "parse_youtube_id",
    "resolve",
]
