"""Command-line entry point.

Stage 1 ships ``probe``, which is genuinely useful on its own: it answers "can
DropScore open this at all, and what does it think it is?" before any of the
vision work exists to blame. ``transcribe`` is registered but not yet implemented
— it fills in as stages 3-8 land.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT, Config, VideoConfig
from .sources import SourceError, resolve
from .video import VideoError, VideoReader

log = logging.getLogger("dropscore")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s" if verbose else "%(message)s",
        stream=sys.stderr,
    )


def _config_from_args(args: argparse.Namespace) -> Config:
    return DEFAULT.evolve(video=VideoConfig(max_width=args.max_width))


def cmd_probe(args: argparse.Namespace) -> int:
    """Report what the decoder sees, and optionally dump sample frames."""
    source = resolve(args.source)
    config = _config_from_args(args)

    with VideoReader(source.path, config) as reader:
        info = reader.info
        print(info)
        print(f"  source      {source.path}")
        print(f"  label       {source.label}")
        print(f"  decoded     {info.width}x{info.height} (scale {info.scale:.3f})")
        print(f"  frames      {info.frame_count or 'unknown'}")

        if args.dump_frames:
            out_dir = Path(args.dump_frames)
            out_dir.mkdir(parents=True, exist_ok=True)
            frames = reader.sample(args.samples)
            import cv2  # noqa: PLC0415  (only needed when dumping)

            for frame in frames:
                name = out_dir / f"{source.label}_{frame.index:06d}.png"
                cv2.imwrite(str(name), frame.image)
            print(f"  wrote       {len(frames)} frames to {out_dir}")

    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    raise SystemExit(
        "transcribe is not implemented yet — the vision pipeline lands in stages "
        "3-8. See docs/ROADMAP.md. Use `dropscore probe` to check a video opens."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dropscore",
        description="Turn falling-tile piano videos into sheet music.",
    )
    parser.add_argument("--version", action="version", version=f"dropscore {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT.video.max_width,
        help=f"downscale frames to this width (default: {DEFAULT.video.max_width})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="inspect a video without transcribing it")
    probe.add_argument("source", help="path to a video file, or a YouTube URL")
    probe.add_argument("--dump-frames", metavar="DIR", help="write sample frames here")
    probe.add_argument(
        "--samples",
        type=int,
        default=DEFAULT.video.calibration_samples,
        help="how many frames to sample when dumping",
    )
    probe.set_defaults(func=cmd_probe)

    transcribe = sub.add_parser("transcribe", help="transcribe a video (not yet implemented)")
    transcribe.add_argument("source", help="path to a video file, or a YouTube URL")
    transcribe.add_argument("-o", "--output", help="output file")
    transcribe.set_defaults(func=cmd_transcribe)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        return args.func(args)
    except (SourceError, VideoError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
