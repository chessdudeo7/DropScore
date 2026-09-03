"""DropScore — turn falling-tile piano videos into sheet music."""

__version__ = "0.1.0"

from .calibrate import Calibration, CalibrationError, calibrate
from .config import (
    DEFAULT,
    CalibrationConfig,
    Config,
    TileConfig,
    EvaluationConfig,
    ScoreConfig,
    TrackingConfig,
    VideoConfig,
)
from .evaluate import ClipResult, Metrics, Report, compare, run_clip, run_corpus
from .export import EngraverNotFound
from .keyboard import COMMON_RANGES, KeyboardLayout, is_black, is_white
from .notes import Hand, Note, NoteSequence, pitch_name
from .overlay import annotate, dump_frames, dump_video
from .score import Analysis, ScoreError, analyze, assign_hands, postprocess, quantize
from .sources import ResolvedSource, SourceError, parse_youtube_id, resolve
from .tiles import Palette, Tile, TileError, detect, detect_in_frame, discover_palette
from .tracking import (
    SpeedEstimate,
    TileTrack,
    TrackingError,
    estimate_speed,
    measure_scroll_speed,
    hands_by_register,
    transcribe,
)
from .video import Frame, VideoError, VideoInfo, VideoReader

__all__ = [
    "__version__",
    # config
    "CalibrationConfig",
    "TrackingConfig",
    "ScoreConfig",
    "EvaluationConfig",
    "TileConfig",
    "Config",
    "DEFAULT",
    "VideoConfig",
    # calibration
    "Calibration",
    "CalibrationError",
    "calibrate",
    # tiles
    "Palette",
    "Tile",
    "TileError",
    "detect",
    "detect_in_frame",
    "discover_palette",
    # tracking
    "SpeedEstimate",
    "TileTrack",
    "TrackingError",
    "estimate_speed",
    "measure_scroll_speed",
    "hands_by_register",
    "transcribe",
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
    # overlays
    "annotate",
    "dump_frames",
    "dump_video",
    # score
    "Analysis",
    "ScoreError",
    "EngraverNotFound",
    # evaluation
    "ClipResult",
    "Metrics",
    "Report",
    "compare",
    "run_clip",
    "run_corpus",
    "analyze",
    "assign_hands",
    "postprocess",
    "quantize",
    # keyboard
    "COMMON_RANGES",
    "KeyboardLayout",
    "is_black",
    "is_white",
]
