"""Video decoding: metadata, sequential iteration, and calibration sampling.

Everything downstream consumes :class:`Frame` objects from here, so the rest of
the pipeline never touches OpenCV's capture API or has to think about scaling.

Two deliberate choices:

* **Downscaling happens once, here.** Frames wider than
  ``config.video.max_width`` are scaled down to it, and the factor is carried on
  every frame so a measurement can be mapped back to source pixels if it ever
  needs to be. Narrower videos are never upscaled.
* **Sequential iteration never seeks.** ``grab()`` skips undecoded frames far
  more cheaply than ``retrieve()`` decodes them, and unlike frame-index seeking
  it is reliable across containers. Seeking is used only for sampling, where
  landing a frame or two off is harmless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import Config, DEFAULT

log = logging.getLogger(__name__)

# Used when a container reports a nonsensical frame rate. These videos are
# overwhelmingly 30 or 60 fps; 30 is the safer guess, and timing is recovered
# from tile geometry rather than frame counts anyway (see docs/PIPELINE.md).
FALLBACK_FPS = 30.0


# Encoders tried in order when writing. Availability varies by OpenCV build.
ENCODERS = (("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi"))


class VideoError(RuntimeError):
    """Raised when a video cannot be opened, decoded or written."""


def open_writer(
    path: "str | Path", fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, "Path"]:
    """Open a writer, falling back through encoders this build actually has.

    Returns the writer and the path finally used — the extension may change if
    the preferred container is unavailable, so callers must not assume the one
    they passed in.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fourcc, suffix in ENCODERS:
        target = path if path.suffix == suffix else path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, target
        writer.release()

    raise VideoError(
        "No usable video encoder in this OpenCV build "
        f"(tried {', '.join(name for name, _ in ENCODERS)})."
    )


@dataclass(frozen=True)
class VideoInfo:
    """Metadata for a decoded video, after any downscaling."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int  # 0 when the container does not report a usable count
    scale: float  # decoded size / source size, <= 1.0
    source_width: int
    source_height: int

    @property
    def duration(self) -> float:
        """Length in seconds, or 0.0 when the frame count is unknown."""
        return self.frame_count / self.fps if self.frame_count else 0.0

    def __str__(self) -> str:
        size = f"{self.source_width}x{self.source_height}"
        if self.scale < 1.0:
            size += f" -> {self.width}x{self.height}"
        length = f"{self.duration:.1f}s" if self.frame_count else "unknown length"
        return f"{self.path.name}: {size}, {self.fps:.2f} fps, {length}"


@dataclass(frozen=True)
class Frame:
    """One decoded frame."""

    index: int  # index in the source video
    time: float  # seconds from the start
    image: np.ndarray  # BGR, possibly downscaled
    scale: float  # multiply a source-pixel measure by this to get image pixels

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _spread(lo: int, hi: int, count: int) -> np.ndarray:
    """Up to ``count`` distinct indices spread evenly over ``[lo, hi)``.

    Asking for more samples than the range holds used to return the same frame
    several times, which quietly skews the temporal median and per-row activity
    that calibration is built on — a short clip would weight some frames double.
    """
    return np.unique(np.linspace(lo, max(lo, hi - 1), num=count, dtype=int))


def _positive(value: float, fallback: float) -> float:
    """OpenCV reports 0, NaN or negatives for properties it cannot determine."""
    if value is None or not np.isfinite(value) or value <= 0:
        return fallback
    return float(value)


class VideoReader:
    """Reads frames from a local video file.

    Use as a context manager so the underlying capture is always released::

        with VideoReader("clip.mp4") as reader:
            for frame in reader.frames(step=2):
                ...
    """

    def __init__(self, path: str | Path, config: Config = DEFAULT) -> None:
        self.path = Path(path)
        self.config = config
        self._open_iterators = 0

        if not self.path.exists():
            raise VideoError(f"No such file: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise VideoError(
                f"Could not open {self.path.name}. The container or codec may be "
                f"unsupported by this OpenCV build."
            )

        self.info = self._probe()

    # ── lifecycle ────────────────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None  # type: ignore[assignment]

    # ── metadata ─────────────────────────────────────────────────────

    def _probe(self) -> VideoInfo:
        cap = self._capture
        src_w = int(_positive(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0))
        src_h = int(_positive(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0))

        if not src_w or not src_h:
            # Some containers only report dimensions once a frame is decoded.
            ok, probe_frame = cap.read()
            if not ok or probe_frame is None:
                raise VideoError(f"{self.path.name} contains no decodable frames.")
            src_h, src_w = probe_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps = _positive(cap.get(cv2.CAP_PROP_FPS), FALLBACK_FPS)
        if fps == FALLBACK_FPS:
            log.warning("%s reports no usable frame rate; assuming %.0f fps",
                        self.path.name, FALLBACK_FPS)

        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        count = int(raw_count) if raw_count and np.isfinite(raw_count) and raw_count > 0 else 0

        scale = min(1.0, self.config.video.max_width / src_w)
        return VideoInfo(
            path=self.path,
            fps=fps,
            width=int(round(src_w * scale)),
            height=int(round(src_h * scale)),
            frame_count=count,
            scale=scale,
            source_width=src_w,
            source_height=src_h,
        )

    # ── reading ──────────────────────────────────────────────────────

    def _prepare(self, image: np.ndarray, index: int) -> Frame:
        if self.info.scale < 1.0:
            image = cv2.resize(
                image,
                (self.info.width, self.info.height),
                interpolation=cv2.INTER_AREA,  # correct choice when downscaling
            )
        return Frame(
            index=index,
            time=index / self.info.fps,
            image=image,
            scale=self.info.scale,
        )

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[Frame]:
        """Yield frames sequentially.

        Skipped frames are grabbed but not decoded, which is much cheaper than
        decoding and discarding them.
        """
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        # There is one capture behind every iterator, and each read advances it.
        # Two iterators alive at once would interleave their reads and hand both
        # callers the wrong frames, silently. Warn rather than raise: a
        # half-consumed generator stays "open" until it is garbage collected, so
        # refusing outright would reject legitimate use.
        if self._open_iterators:
            log.warning(
                "%s: a second frame iterator was opened while %d is still active; "
                "they share one capture and will interleave",
                self.path.name,
                self._open_iterators,
            )
        self._open_iterators += 1

        try:
            cap = self._capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            index = start

            while stop is None or index < stop:
                if not cap.grab():
                    return
                ok, image = cap.retrieve()
                if not ok or image is None:
                    return
                yield self._prepare(image, index)

                # Skip the next step-1 frames without decoding them.
                for _ in range(step - 1):
                    if not cap.grab():
                        return
                    index += 1
                index += 1
        finally:
            self._open_iterators -= 1

    def read_at(self, index: int) -> Frame | None:
        """Decode a single frame by index, or None if it cannot be read.

        Frame-index seeking is approximate on some containers; the returned
        frame's ``index`` reflects what was requested, not necessarily what the
        decoder landed on. Fine for sampling, not for measurement.
        """
        cap = self._capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, image = cap.read()
        if not ok or image is None:
            return None
        return self._prepare(image, index)

    def sample(self, count: int | None = None) -> list[Frame]:
        """Take frames spread evenly across the video, for calibration passes.

        Skips a margin at each end, since these videos routinely open on a title
        card and close on an outro — neither of which contains any tiles.
        """
        cfg = self.config.video
        count = count or cfg.calibration_samples

        total = self.info.frame_count
        if total <= 0:
            # No usable count: fall back to walking the file and keeping every
            # nth frame. Costs a full pass, but only happens on odd containers.
            log.info("%s reports no frame count; scanning to sample", self.path.name)
            return self._sample_by_scan(count)

        lo = int(total * cfg.calibration_margin)
        hi = int(total * (1.0 - cfg.calibration_margin))
        if hi - lo < count:
            lo, hi = 0, total

        frames: list[Frame] = []
        for i in _spread(lo, hi, count):
            frame = self.read_at(int(i))
            if frame is not None:
                frames.append(frame)

        if not frames:
            raise VideoError(f"Could not read any frames from {self.path.name}.")
        return frames

    def _sample_by_scan(self, count: int) -> list[Frame]:
        """Two passes: count frames without decoding, then decode only the picks.

        Buffering every frame instead would mean gigabytes for a few minutes of
        720p, so the first pass uses ``grab()`` alone.
        """
        cap = self._capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total = 0
        while cap.grab():
            total += 1

        if total == 0:
            raise VideoError(f"Could not read any frames from {self.path.name}.")

        cfg = self.config.video
        lo = int(total * cfg.calibration_margin)
        hi = int(total * (1.0 - cfg.calibration_margin))
        if hi - lo < count:
            lo, hi = 0, total
        wanted = {int(i) for i in _spread(lo, hi, count)}

        return [f for f in self.frames() if f.index in wanted]
