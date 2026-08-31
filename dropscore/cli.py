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
from .keyboard import COMMON_RANGES
from .sources import SourceError, resolve
from .synth.themes import DEFAULT_THEME, THEMES
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


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit the keyboard grid, optionally scoring it against ground truth."""
    import json  # noqa: PLC0415

    from .calibrate import calibrate  # noqa: PLC0415

    source = resolve(args.source)
    with VideoReader(source.path, _config_from_args(args)) as reader:
        result = calibrate(reader.sample(args.samples))

    print(result)

    truth_path = args.check
    if truth_path is None:
        candidate = source.path.with_suffix(".truth.json")
        truth_path = str(candidate) if candidate.exists() else None

    if truth_path is None:
        return 0

    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))["geometry"]
    checks = [
        ("first_pitch", result.layout.first_pitch, truth["first_pitch"], 0),
        ("last_pitch", result.layout.last_pitch, truth["last_pitch"], 0),
        ("strike_y", result.strike_y, truth["strike_y"], args.tolerance),
        ("white_width", result.white_width, truth["white_key_width"], 0.5),
        ("x0", result.layout.x0, truth["x0"], args.tolerance),
    ]

    failed = 0
    for name, got, want, tolerance in checks:
        ok = abs(got - want) <= tolerance
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:12} got {got:>10.3f}  want {want:>10.3f}")

    return 1 if failed else 0


def cmd_tiles(args: argparse.Namespace) -> int:
    """Report the palette and the tiles found in a few frames."""
    from .calibrate import calibrate  # noqa: PLC0415
    from .notes import pitch_name  # noqa: PLC0415
    from .tiles import detect  # noqa: PLC0415

    source = resolve(args.source)
    with VideoReader(source.path, _config_from_args(args)) as reader:
        frames = reader.sample(args.samples)
        calibration = calibrate(frames)
        palette, per_frame = detect(frames, calibration)

    print(calibration)
    print(f"palette: {palette.track_count} track(s)")
    for index, (color, count) in enumerate(zip(palette.colors, palette.counts)):
        lab = ", ".join(f"{v:.0f}" for v in color)
        print(f"  track {index}  Lab({lab})  {count} px")

    total = sum(len(tiles) for tiles in per_frame)
    print(f"{total} tile(s) across {len(per_frame)} sampled frames")

    for frame, tiles in zip(frames, per_frame):
        if not tiles:
            continue
        names = " ".join(sorted({pitch_name(t.pitch) for t in tiles}))
        print(f"  t={frame.time:7.2f}s  {len(tiles):3d}  {names}")

    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """Render synthetic clips with exact ground truth, for evaluation."""
    from .synth import RenderSpec, generate, get_theme, render  # noqa: PLC0415

    out_dir = Path(args.out_dir)
    themes = sorted(THEMES) if args.theme == "all" else [args.theme]

    for theme_name in themes:
        theme = get_theme(theme_name)
        for index in range(args.count):
            seed = args.seed + index
            sequence = generate(seed=seed, bars=args.bars, tempo=args.tempo)
            spec = RenderSpec(
                width=args.width,
                height=args.height,
                fps=args.fps,
                theme=theme,
                key_range=args.key_range,
            )
            name = f"{theme_name}_{args.key_range}key_seed{seed}"
            video, truth = render(sequence, out_dir / f"{name}.mp4", spec)
            print(f"{video}  ({len(sequence)} notes, {sequence.key})")
            if truth:
                print(f"  truth {truth}")

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

    calib = sub.add_parser(
        "calibrate",
        help="fit the keyboard grid to a video",
        description=(
            "Locates the keybed, measures the white-key grid and anchors it to "
            "pitch via the black-key pattern. Given a .truth.json sidecar (found "
            "automatically next to synthetic clips) it also scores the fit."
        ),
    )
    calib.add_argument("source", help="path to a video file, or a YouTube URL")
    calib.add_argument("--check", metavar="TRUTH", help="ground-truth JSON to score against")
    calib.add_argument(
        "--tolerance", type=float, default=2.0, help="pixel tolerance when scoring"
    )
    calib.add_argument(
        "--samples",
        type=int,
        default=DEFAULT.video.calibration_samples,
        help="frames to sample",
    )
    calib.set_defaults(func=cmd_calibrate)

    tiles = sub.add_parser(
        "tiles",
        help="detect falling tiles and the keys they belong to",
        description=(
            "Discovers the video's tile palette, then reports the tiles found in "
            "sampled frames and which key each one sits over."
        ),
    )
    tiles.add_argument("source", help="path to a video file, or a YouTube URL")
    tiles.add_argument(
        "--samples", type=int, default=12, help="frames to sample"
    )
    tiles.set_defaults(func=cmd_tiles)

    synth = sub.add_parser(
        "synth",
        help="render synthetic falling-tile clips with exact ground truth",
        description=(
            "Generates videos from procedurally written music, each with a "
            ".truth.json sidecar giving the notes and the geometry that drew "
            "them. This is the evaluation corpus the vision stages are scored "
            "against."
        ),
    )
    synth.add_argument("-o", "--out-dir", default="out/synth", help="where to write clips")
    synth.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        choices=[*sorted(THEMES), "all"],
        help="visual preset, or 'all' to render one clip per theme",
    )
    synth.add_argument("--count", type=int, default=1, help="clips per theme")
    synth.add_argument("--seed", type=int, default=0, help="first seed; increments per clip")
    synth.add_argument("--bars", type=int, default=16, help="length in bars")
    synth.add_argument("--tempo", type=float, default=96.0, help="BPM")
    synth.add_argument(
        "--key-range",
        default="88",
        choices=sorted(COMMON_RANGES),
        help="how many keys the keyboard shows",
    )
    synth.add_argument("--width", type=int, default=1280)
    synth.add_argument("--height", type=int, default=720)
    synth.add_argument("--fps", type=float, default=30.0)
    synth.set_defaults(func=cmd_synth)

    transcribe = sub.add_parser("transcribe", help="transcribe a video (not yet implemented)")
    transcribe.add_argument("source", help="path to a video file, or a YouTube URL")
    transcribe.add_argument("-o", "--output", help="output file")
    transcribe.set_defaults(func=cmd_transcribe)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    from .calibrate import CalibrationError  # noqa: PLC0415
    from .synth.renderer import RenderError  # noqa: PLC0415
    from .tiles import TileError  # noqa: PLC0415

    try:
        return args.func(args)
    except (SourceError, VideoError, RenderError, CalibrationError, TileError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
