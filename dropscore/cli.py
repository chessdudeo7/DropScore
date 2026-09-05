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


def _output_stem(path: Path) -> Path:
    """Strip a known export extension so several formats can share a base.

    Only *known* extensions are removed. ``Path.with_suffix`` would replace
    everything after the last dot, so a piece called "prelude.no.2" was written
    as "prelude.no.mid".
    """
    from .export import FORMATS  # noqa: PLC0415

    for suffix in (*(ext for ext, _ in FORMATS.values()), ".notes"):
        if path.name.endswith(suffix) and len(path.name) > len(suffix):
            return _output_stem(path.with_name(path.name[: -len(suffix)]))
    return path


def _with_extension(stem: Path, ext: str) -> Path:
    """Append an extension rather than replacing whatever follows a dot."""
    return stem if stem.name.endswith(ext) else stem.with_name(stem.name + ext)


def _write_formats(sequence, stem: Path, formats: list[str], analysis) -> int:
    """Write each format, reporting failures instead of stopping at the first.

    PDF needs an engraver that may not be installed. Losing the MIDI because
    of it would be the same mistake the service used to make.
    """
    from .export import extension, write  # noqa: PLC0415

    written, failed = [], []
    for format in formats:
        target = _with_extension(stem, extension(format))
        try:
            written.append(write(sequence, target, format, analysis))
        except Exception as exc:  # noqa: BLE001 - one format must not stop the rest
            failed.append((format, exc))

    for path in written:
        print(f"  wrote {path}")
    for format, exc in failed:
        log.error("could not write %s: %s", format, exc)

    return 0 if written else 1


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


def cmd_debug(args: argparse.Namespace) -> int:
    """Write annotated frames or an annotated video showing what was detected."""
    from .calibrate import calibrate  # noqa: PLC0415
    from .overlay import dump_frames, dump_video  # noqa: PLC0415
    from .tiles import discover_palette  # noqa: PLC0415
    from .tracking import measure_scroll_speed  # noqa: PLC0415

    source = resolve(args.source)
    config = _config_from_args(args)
    out_dir = Path(args.out_dir)

    with VideoReader(source.path, config) as reader:
        samples = reader.sample()
        calibration = calibrate(samples, config)
        print(calibration)

        palette = None if args.grid_only else discover_palette(samples, calibration, config)

        speed = None
        if not args.grid_only:
            try:
                speed = measure_scroll_speed(reader, calibration, 40, config=config)
            except Exception as exc:  # noqa: BLE001 - speed is optional here
                log.warning("could not measure scroll speed: %s", exc)

        if args.video:
            frames = reader.frames(step=args.step)
            target = dump_video(
                frames,
                calibration,
                out_dir / f"{source.label}.debug.mp4",
                reader.info.fps / args.step,
                (reader.info.width, reader.info.height),
                palette,
                speed,
                config,
            )
            print(f"wrote {target}")
        else:
            written = dump_frames(
                reader.sample(args.frames),
                calibration,
                out_dir,
                palette,
                speed,
                prefix=source.label,
                config=config,
            )
            print(f"wrote {len(written)} frames to {out_dir}")

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
            sequence = generate(
                seed=seed, bars=args.bars, tempo=args.tempo,
                key=args.key, sustained=args.sustained,
            )
            spec = RenderSpec(
                width=args.width,
                height=args.height,
                fps=args.fps,
                theme=theme,
                key_range=args.key_range,
            )
            suffix = "_sustained" if args.sustained else ""
            if args.key:
                suffix += "_" + args.key.replace(" ", "")
            name = f"{theme_name}_{args.key_range}key_seed{seed}{suffix}"
            video, truth = render(sequence, out_dir / f"{name}.mp4", spec)
            print(f"{video}  ({len(sequence)} notes, {sequence.key})")
            if truth:
                print(f"  truth {truth}")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the web service and serve the frontend from it."""
    try:
        import uvicorn  # noqa: PLC0415

        from .service import create_app  # noqa: PLC0415
    except ImportError as exc:
        log.error('%s (install with: pip install -e ".[service]")', exc)
        return 1

    app = create_app(
        workdir=args.workdir,
        config=_config_from_args(args),
        keep_sources=args.keep_sources,
    )
    print(f"DropScore on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Score the pipeline over a corpus and compare against a baseline."""
    import json  # noqa: PLC0415

    from .evaluate import find_clips, find_regressions, run_corpus  # noqa: PLC0415

    corpus = Path(args.corpus)
    pairs = find_clips(corpus)
    if not pairs:
        log.error(
            "no clips with truth sidecars in %s — run `dropscore synth --theme all "
            "-o %s` first",
            corpus,
            corpus,
        )
        return 1

    report = run_corpus(pairs, _config_from_args(args))

    width = max(len(c.name) for c in report.clips)
    for clip in report.clips:
        if clip.error:
            print(f"  {clip.name:<{width}}  FAILED  {clip.error}")
            continue
        print(f"  {clip.name:<{width}}  {clip.metrics}")
        if clip.geometry and args.verbose:
            print(f"  {'':<{width}}  {clip.geometry}")

    print(f"\ncorpus F1 {report.f1:.4f} over {len(report.clips)} clips")

    scored = [c for c in report.clips if c.key_correct is not None]
    if scored:
        right = [c for c in scored if c.key_correct]
        print(f"key {len(right)}/{len(scored)} correct")
        for clip in scored:
            if not clip.key_correct:
                print(
                    f"  {clip.name:<{width}}  key {clip.key_found} "
                    f"({clip.key_confidence:.2f}), wanted {clip.key_expected}"
                )

    timed = [c for c in report.clips if c.tempo_correct is not None]
    if timed:
        right = [c for c in timed if c.tempo_correct]
        octave = [c for c in timed if c.tempo_octave_correct]
        print(
            f"tempo {len(right)}/{len(timed)} correct, "
            f"{len(octave)}/{len(timed)} correct up to an octave"
        )
        for clip in timed:
            if not clip.tempo_correct:
                print(
                    f"  {clip.name:<{width}}  tempo {clip.tempo_found:.1f} BPM "
                    f"({clip.tempo_ratio:.2f}x), wanted {clip.tempo_expected:.1f}"
                )

    if report.failures:
        print(f"{len(report.failures)} clip(s) failed outright")

    if args.save_baseline:
        print(f"wrote baseline {report.save(args.save_baseline)}")
        return 0

    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            log.error("no baseline at %s; write one with --save-baseline", baseline_path)
            return 1

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        regressions = find_regressions(baseline, report, _config_from_args(args))
        delta = report.f1 - float(baseline.get("f1", 0.0))
        print(f"corpus F1 change {delta:+.4f} against {baseline_path}")

        if regressions:
            print(f"\n{len(regressions)} regression(s):")
            for regression in regressions:
                print(f"  {regression}")
            return 1

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write an existing note JSON out as MIDI, MusicXML or PDF."""
    from .notes import NoteSequence  # noqa: PLC0415
    from .score import ScoreError, analyze  # noqa: PLC0415

    sequence = NoteSequence.load(args.notes)

    analysis = None
    try:
        analysis = analyze(sequence, _config_from_args(args))
    except ScoreError as exc:
        log.warning("could not analyse the notes (%s); using defaults", exc)

    stem = _output_stem(Path(args.output or args.notes))
    return _write_formats(sequence, stem, args.format or ["midi"], analysis)


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run stage 7 over an existing note JSON."""
    from .notes import NoteSequence  # noqa: PLC0415
    from .score import postprocess  # noqa: PLC0415

    sequence = NoteSequence.load(args.notes)
    result, analysis = postprocess(sequence, _config_from_args(args))

    print(analysis)
    print(f"  beat      {analysis.beat:.4f}s, grid from {analysis.beat_phase:.4f}s")
    print(f"  downbeat  {analysis.downbeat_phase:.4f}s")
    print(f"  hands     {len(result.hand('L'))} left, {len(result.hand('R'))} right")

    if args.output:
        result.save(args.output)
        print(f"  wrote     {args.output}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Calibrate, detect, track and time — the pipeline as far as it goes."""
    from .calibrate import calibrate  # noqa: PLC0415
    from .tiles import discover_palette  # noqa: PLC0415
    from .tracking import measure_scroll_speed, transcribe  # noqa: PLC0415

    source = resolve(args.source)
    config = _config_from_args(args)

    with VideoReader(source.path, config) as reader:
        samples = reader.sample()
        calibration = calibrate(samples, config)
        palette = discover_palette(samples, calibration, config)
        log.info("%s", calibration)

        # Speed needs consecutive frames, and several probes rather than one:
        # a stretch of long sustained notes carries no vertical motion to
        # correlate, so a single window can land somewhere unmeasurable.
        speed = measure_scroll_speed(
            reader, calibration, args.speed_frames, config=config
        )
        log.info("scroll speed %.2f px/s (confidence %.2f)", speed.value, speed.confidence)

        sequence = transcribe(reader.frames(), calibration, palette, speed, config)

    sequence.source = f"dropscore:{source.label}"
    analysis = None

    if not args.raw and len(sequence):
        from .score import ScoreError, postprocess  # noqa: PLC0415

        try:
            sequence, analysis = postprocess(sequence, config)
            print(analysis)
        except ScoreError as exc:
            log.warning("could not analyse the notes (%s); leaving them raw", exc)

    stem = _output_stem(Path(args.output or f"{source.label}.notes"))
    print(f"{len(sequence)} notes over {sequence.duration:.1f}s")

    return _write_formats(sequence, stem, args.format or ["json"], analysis)


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

    debug = sub.add_parser(
        "debug",
        help="write annotated frames showing the fitted grid and detected tiles",
        description=(
            "Draws the fitted key grid, strike line and detected tiles onto the "
            "video. If the bright C lines do not land on the real C keys, the "
            "calibration is wrong and everything downstream is transposed."
        ),
    )
    debug.add_argument("source", help="path to a video file, or a YouTube URL")
    debug.add_argument("-o", "--out-dir", default="out/debug", help="where to write")
    debug.add_argument("--frames", type=int, default=12, help="how many stills to write")
    debug.add_argument("--video", action="store_true", help="write an annotated video instead")
    debug.add_argument("--step", type=int, default=1, help="frame step when writing a video")
    debug.add_argument(
        "--grid-only",
        action="store_true",
        help="skip tile detection and draw only the calibration",
    )
    debug.set_defaults(func=cmd_debug)

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
        "--key",
        default=None,
        help="write in this key instead of one chosen from the seed",
    )
    synth.add_argument(
        "--key-range",
        default="88",
        choices=sorted(COMMON_RANGES),
        help="how many keys the keyboard shows",
    )
    synth.add_argument(
        "--sustained",
        action="store_true",
        help="slow held chords in a narrow register, so tiles outgrow the fall area",
    )
    synth.add_argument("--width", type=int, default=1280)
    synth.add_argument("--height", type=int, default=720)
    synth.add_argument("--fps", type=float, default=30.0)
    synth.set_defaults(func=cmd_synth)

    transcribe = sub.add_parser(
        "transcribe",
        help="transcribe a video to raw note events",
        description=(
            "Runs calibration, tile detection, tracking and timing, writing a "
            "note JSON. Timing is recovered from tile geometry and is sub-frame "
            "accurate, but unquantized: tempo, hands and notation come later."
        ),
    )
    transcribe.add_argument("source", help="path to a video file, or a YouTube URL")
    transcribe.add_argument("-o", "--output", help="output file (default: <name>.notes.json)")
    transcribe.add_argument(
        "--speed-frames",
        type=int,
        default=40,
        help="consecutive frames used to measure the scroll speed",
    )
    transcribe.add_argument(
        "--raw",
        action="store_true",
        help="skip tempo, key, hand and quantization analysis",
    )
    transcribe.add_argument(
        "-f",
        "--format",
        action="append",
        choices=["midi", "musicxml", "pdf", "json"],
        help="may be given more than once (default: json)",
    )
    transcribe.set_defaults(func=cmd_transcribe, format=None)

    serve = sub.add_parser(
        "serve",
        help="run the web service and frontend",
        description=(
            "Serves the API and the static frontend from one origin. Needs the "
            'optional extra: pip install -e ".[service]"'
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workdir", default="out/jobs", help="where job output is kept")
    serve.add_argument(
        "--keep-sources",
        action="store_true",
        help="keep uploaded videos after transcribing, for debugging a bad clip",
    )
    serve.set_defaults(func=cmd_serve)

    evaluate = sub.add_parser(
        "eval",
        help="score the pipeline over a corpus of clips with ground truth",
        description=(
            "Transcribes every clip that has a .truth.json sidecar and reports "
            "note-level precision, recall and F1, plus how far the fitted "
            "geometry was from the geometry that drew each clip. With a "
            "baseline it exits non-zero when any clip gets worse."
        ),
    )
    evaluate.add_argument(
        "-c", "--corpus", default="out/synth", help="directory of clips and sidecars"
    )
    evaluate.add_argument("-b", "--baseline", help="baseline JSON to compare against")
    evaluate.add_argument("--save-baseline", metavar="PATH", help="write a new baseline")
    evaluate.set_defaults(func=cmd_eval)

    export = sub.add_parser(
        "export",
        help="write a note JSON as MIDI, MusicXML or PDF",
        description=(
            "MIDI is exactly what the tiles said. MusicXML is mechanically "
            "correct but musically approximate — one voice per staff, no "
            "beaming. PDF needs MuseScore installed separately."
        ),
    )
    export.add_argument("notes", help="a .notes.json written by transcribe")
    export.add_argument(
        "-f",
        "--format",
        action="append",
        choices=["midi", "musicxml", "pdf", "json"],
        help="may be given more than once (default: midi)",
    )
    export.add_argument("-o", "--output", help="output path; extension is set per format")
    export.set_defaults(func=cmd_export, format=None)

    analyse = sub.add_parser(
        "analyze",
        help="infer tempo, key and hands for an existing note JSON",
        description=(
            "Runs stage 7 on notes already transcribed, so the analysis can be "
            "re-run with different settings without re-reading the video."
        ),
    )
    analyse.add_argument("notes", help="a .notes.json written by transcribe")
    analyse.add_argument("-o", "--output", help="write the quantized result here")
    analyse.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    from .calibrate import CalibrationError  # noqa: PLC0415
    from .synth.renderer import RenderError  # noqa: PLC0415
    from .tiles import TileError  # noqa: PLC0415
    from .tracking import TrackingError  # noqa: PLC0415

    try:
        return args.func(args)
    except (
        SourceError,
        VideoError,
        RenderError,
        CalibrationError,
        TileError,
        TrackingError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
