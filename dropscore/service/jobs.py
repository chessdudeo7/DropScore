"""Job queue: run the pipeline in the background and report progress.

Deliberately in-process — a thread pool, a dict of jobs, and a lock. The frontend
polls, so there is no need for Redis or a separate worker to make this work on one
machine, and adding either would mean shipping infrastructure before there is any
evidence it is needed. The interface is narrow enough that swapping in a real
queue later touches this file only.

Stage keys match the frontend's existing step list exactly, so the UI written in
stage 0 needs no new vocabulary to render real progress.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..config import Config, DEFAULT
from ..notes import NoteSequence

log = logging.getLogger(__name__)

# The stages the UI draws, in the order they run. The tiles are followed
# through every frame before the fall speed is worked out from what was found,
# so "notes" -- the long one -- comes before "timing".
STAGES: tuple[tuple[str, str], ...] = (
    ("fetch", "Reading video"),
    ("keys", "Calibrating keyboard geometry"),
    ("tiles", "Finding the tile colours"),
    ("notes", "Following the tiles frame by frame"),
    ("timing", "Measuring the fall speed"),
    ("score", "Quantizing and engraving"),
)

#: Longest video accepted, in seconds. Transcription runs at about real time
#: or slower, and a worker slot held for an hour starves everyone else.
MAX_SECONDS = 15 * 60

#: Fewest frames worth trying. Calibration and the fall speed both compare
#: frames, so a still image or a handful of frames fails further in with a
#: message about the wrong thing ("not a falling-tile video").
MIN_FRAMES = 10

# How long a finished job's results stay downloadable. Retention by count
# alone kept a quiet server's results on disk indefinitely: with fewer than
# 50 jobs ever run, nothing was deleted until a restart.
MAX_AGE_SECONDS = 24 * 60 * 60


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


#: States a job never leaves. Only these are safe to evict.
TERMINAL = frozenset({Status.DONE, Status.ERROR, Status.CANCELLED})

#: The log is a ring: past LOG_LIMIT lines the oldest LOG_TRIM are dropped, so a
#: runaway job cannot grow it without bound.
LOG_LIMIT = 500
LOG_TRIM = 100


class Cancelled(Exception):
    """Raised inside a worker when the job has been cancelled."""


class Rejected(RuntimeError):
    """A video the service declines, with a message meant for the person."""


# What each pipeline failure means to someone who uploaded a video. The
# exception's own text is written for whoever debugs the pipeline -- "keybed
# has no vertical structure to measure" -- and stays in the job log; the error
# shown in the UI says what was wrong with the video and what to try instead.
_NOT_TILES = (
    "This doesn't look like a falling-tile piano video. DropScore needs the "
    "keyboard visible along the bottom of the frame with tiles falling onto it."
)
_EXPLANATIONS: tuple[tuple[str, str], ...] = (
    ("VideoError", "This video could not be read. Try exporting it again as an H.264 MP4."),
    ("CalibrationError", _NOT_TILES),
    ("TileError", "No falling tiles were found in this video. " + _NOT_TILES.split(". ", 1)[1]),
    (
        "TrackingError",
        (
            "The tiles could not be followed as they fell. This happens with videos "
            "that jump, cut between shots, or are not falling-tile videos at all."
        ),
    ),
)


def explain(exc: BaseException) -> str:
    """The message to show for a failed job."""
    if isinstance(exc, Rejected):
        return str(exc)
    names = {cls.__name__ for cls in type(exc).__mro__}
    for name, message in _EXPLANATIONS:
        if name in names:
            return message
    return f"Something went wrong while transcribing: {exc}"


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_JOB_ID = re.compile(r"[0-9a-f]{32}")  # uuid4().hex


def safe_label(raw: str, fallback: str = "video", limit: int = 48) -> str:
    """Turn an untrusted name into one that is safe to put in a path.

    Upload filenames come straight from the client and are used to build output
    paths, so they cannot be trusted. ``Path(name).stem`` alone is not enough:
    it strips ``/`` but leaves ``\\``, which on Windows is a separator, so a file
    called ``..\\..\\evil.mid`` would escape the job directory.

    Everything before the last separator of *either* kind is dropped, the
    extension is removed, anything outside ``[A-Za-z0-9._-]`` becomes an
    underscore, and leading or trailing dots go — so ``..`` cannot survive.
    """
    candidate = re.split(r"[\\/]", str(raw).strip())[-1]
    candidate = Path(candidate).stem
    candidate = _UNSAFE.sub("_", candidate).strip("._")
    return candidate[:limit] or fallback


@dataclass
class Job:
    """One transcription, its progress, and whatever it produced."""

    id: str
    label: str
    workdir: Path
    client: str = ""  # who submitted it, for the per-client limit
    source_path: Path | None = None  # the input video, deleted once it is read
    status: Status = Status.QUEUED
    stage_states: dict[str, str] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    log_offset: int = 0  # absolute index of log[0], once trimming has happened
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    result: dict[str, Any] | None = None
    files: dict[str, Path] = field(default_factory=dict)
    # Formats that were asked for but could not be written, and why. Reported
    # rather than raised, so the outputs that did work stay available.
    failed_formats: dict[str, str] = field(default_factory=dict)
    # How far through the active stage, 0 to 1, where the stage can tell.
    stage_fraction: float = 0.0
    _cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        for key, _ in STAGES:
            self.stage_states.setdefault(key, "pending")

    # ── progress, written by the worker ──────────────────────────────

    def begin(self, stage: str) -> None:
        self.check_cancelled()
        with self._lock:
            self.stage_states[stage] = "active"
            self.stage_fraction = 0.0

    def finish(self, stage: str) -> None:
        with self._lock:
            self.stage_states[stage] = "done"
            self.stage_fraction = 0.0

    def say(self, message: str) -> None:
        with self._lock:
            self.log.append(message)
            if len(self.log) > LOG_LIMIT:
                del self.log[:LOG_TRIM]
                # Clients track how many lines they have seen in absolute terms,
                # so trimming has to be reported. Without this the two sides
                # disagree about what index log[0] is and every line after the
                # first trim is silently skipped.
                self.log_offset += LOG_TRIM
        log.debug("[%s] %s", self.id[:8], message)

    def discard_source(self) -> None:
        """Delete the input video, keeping the much smaller outputs.

        A source is up to 2 GB and is of no further use once the pipeline has
        read it, so holding it until retention eviction — fifty jobs later —
        risks tens of gigabytes. Outputs are kilobytes and stay.
        """
        if self.source_path is None:
            return
        try:
            self.source_path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - platform-specific locking
            log.warning("could not delete %s: %s", self.source_path, exc)
        else:
            self.source_path = None

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def is_finished(self) -> bool:
        """True once the job has reached a state it will not leave."""
        return self.status in TERMINAL

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled()

    @property
    def progress(self) -> float:
        done = sum(1 for state in self.stage_states.values() if state == "done")
        if self.status is Status.RUNNING:
            done += min(max(self.stage_fraction, 0.0), 1.0)
        return done / len(STAGES)

    # ── reading, by the API ──────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "label": self.label,
                "status": self.status.value,
                "progress": self.progress,
                "stages": [
                    {"key": key, "label": label, "state": self.stage_states[key]}
                    for key, label in STAGES
                ],
                "log": list(self.log),
                "log_offset": self.log_offset,
                "error": self.error,
                "elapsed": (self.finished or time.time()) - self.created,
                "result": self.result,
                "formats": sorted(self.files),
                "failed_formats": dict(self.failed_formats),
            }


class JobStore:
    """Holds jobs and runs them. Thread-safe."""

    def __init__(
        self,
        root: Path,
        workers: int = 2,
        retain: int = 50,
        keep_sources: bool = False,
        max_age: float | None = MAX_AGE_SECONDS,
        sweep_every: float = 600.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retain = retain
        self.max_age = max_age
        # Sources are deleted whatever the outcome, not just on success: a
        # service must not depend on failures being rare to stay within its
        # disk budget. Set this to keep them for debugging a bad video.
        self.keep_sources = keep_sources
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._clear_orphans()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dropscore")
        # Expiry has to happen on an idle server too, so it does not wait for
        # the next upload.
        self._stopped = threading.Event()
        if max_age is not None:
            threading.Thread(
                target=self._sweep_until_stopped, args=(sweep_every,),
                name="dropscore-sweep", daemon=True,
            ).start()

    def _sweep_until_stopped(self, every: float) -> None:
        while not self._stopped.wait(every):
            self.expire()

    def expire(self, now: float | None = None) -> int:
        """Delete finished jobs older than ``max_age``. Returns how many."""
        if self.max_age is None:
            return 0
        cutoff = (time.time() if now is None else now) - self.max_age
        with self._lock:
            old = [
                job for job in self._jobs.values()
                if job.is_finished and job.finished is not None and job.finished < cutoff
            ]
            for job in old:
                self._jobs.pop(job.id, None)
                self._order.remove(job.id)
        for job in old:
            shutil.rmtree(job.workdir, ignore_errors=True)
        if old:
            log.info("expired %d finished job(s)", len(old))
        return len(old)

    def _clear_orphans(self) -> None:
        """Delete job directories left by an earlier run of the server.

        Jobs live in memory, so after a restart nothing can reach the outputs a
        previous run left behind -- their ids 404 -- yet the files stayed on
        disk for good, and with ``keep_sources`` the videos too. Only names a
        job would have been given are touched, so pointing the store at a
        directory that holds anything else does not delete it.
        """
        for entry in self.root.iterdir():
            if entry.is_dir() and _JOB_ID.fullmatch(entry.name):
                shutil.rmtree(entry, ignore_errors=True)

    def create(self, label: str) -> Job:
        """Register a job. The label is sanitised here, at the one chokepoint."""
        job_id = uuid.uuid4().hex
        workdir = self.root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, label=safe_label(label), workdir=workdir)

        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._evict()
        return job

    def _evict(self) -> None:
        """Drop the oldest *finished* jobs once past the retention limit.

        Only terminal jobs are eligible. Evicting by age alone would delete a
        running job's working directory underneath its worker — mid-write, on
        the outputs it is producing — and make its id 404 while the thread was
        still going. When nothing is eligible the store is simply allowed over
        the limit until something finishes; growing a little beats corrupting
        work in progress.
        """
        surplus = len(self._order) - self.retain
        if surplus <= 0:
            return

        keep: list[str] = []
        for job_id in self._order:
            job = self._jobs.get(job_id)
            if surplus > 0 and (job is None or job.is_finished):
                if job is not None:
                    self._jobs.pop(job_id, None)
                    shutil.rmtree(job.workdir, ignore_errors=True)
                surplus -= 1
                continue
            keep.append(job_id)

        self._order = keep
        if surplus > 0:
            log.warning(
                "%d job(s) over the retention limit are still running; keeping them",
                surplus,
            )

    def active(self, client: str | None = None) -> int:
        """Jobs queued or running, for one client or for everyone."""
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if not job.is_finished and (client is None or job.client == client)
            )

    def get(self, job_id: str) -> Job | None:
        self.expire()
        with self._lock:
            return self._jobs.get(job_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def discard(self, job: Job) -> None:
        """Forget a job that never ran, and delete anything it wrote.

        Used when submission fails after the job was registered — otherwise it
        would sit in the store as QUEUED forever, counting against retention
        while never doing anything.
        """
        with self._lock:
            self._jobs.pop(job.id, None)
            if job.id in self._order:
                self._order.remove(job.id)
        shutil.rmtree(job.workdir, ignore_errors=True)

    def submit(self, job: Job, work: Callable[[Job], None]) -> None:
        self._pool.submit(self._run, job, work)

    def _run(self, job: Job, work: Callable[[Job], None]) -> None:
        job.status = Status.RUNNING
        try:
            work(job)
            job.status = Status.DONE
        except Cancelled:
            job.status = Status.CANCELLED
            job.say("cancelled")
        except Exception as exc:  # noqa: BLE001 - the API reports the message
            job.status = Status.ERROR
            job.error = explain(exc)
            job.say(f"error: {exc}")
            log.exception("job %s failed", job.id)
        finally:
            job.finished = time.time()
            # In the finally block so a failed or cancelled job releases its
            # video too — the outcome does not change that it is dead weight.
            if not self.keep_sources:
                job.discard_source()

    def shutdown(self) -> None:
        """Stop taking work, and ask anything running to stop.

        ``cancel_futures`` only drops jobs still queued; one already running
        carries on. That matters because ThreadPoolExecutor's threads are not
        daemons, so the interpreter waits for them at exit — without cancelling,
        Ctrl+C on the server would appear to hang until the current
        transcription finished, which for a long video is minutes. Workers check
        between frames, so they stop promptly.
        """
        self._stopped.set()
        with self._lock:
            live = [job for job in self._jobs.values() if not job.is_finished]

        for job in live:
            job.cancel()
        if live:
            log.info("shutdown: cancelled %d running job(s)", len(live))

        self._pool.shutdown(wait=False, cancel_futures=True)


def until_cancelled(
    frames: Iterable[Any], job: Job, report_every: int = 200, total: int = 0
) -> Iterator[Any]:
    """Pass frames through, checking for cancellation between each one.

    The main tracking pass consumes every frame of the video and is by far the
    longest thing a job does. Checking only at stage boundaries meant Cancel
    could not interrupt it: the UI returned to the input screen while the worker
    ran to completion, still holding one of two pool slots, so two cancelled
    long videos starved the service.

    Wrapping the frame stream rather than adding a callback to ``tracking``
    keeps cancellation entirely inside the service layer — the pipeline consumes
    its input lazily, so raising here unwinds the whole pass.
    """
    for index, frame in enumerate(frames, start=1):
        job.check_cancelled()
        if total > 0:
            job.stage_fraction = index / total
        if report_every and index % report_every == 0:
            job.say(f"tracked {index} of {total} frames" if total > 0 else f"tracked {index} frames")
        yield frame


def output_path(job: Job, suffix: str) -> Path:
    """An output path inside the job's directory, verified to stay there.

    ``safe_label`` should already guarantee this. Checking again at the point the
    path is used means a future caller that builds a Job without going through
    ``JobStore.create`` cannot quietly reintroduce an escape.
    """
    root = job.workdir.resolve()
    target = (job.workdir / f"{job.label}{suffix}").resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"refusing to write {target}, which is outside {root}")
    return target


def transcribe_job(
    job: Job,
    video: Path,
    config: Config = DEFAULT,
    formats: Sequence[str] = ("json", "midi", "musicxml"),
    max_seconds: float | None = MAX_SECONDS,
) -> None:
    """The pipeline, stage by stage, reporting as it goes.

    This mirrors ``evaluate.run_clip`` rather than sharing it, because the two
    want different things: the harness wants a score and no chatter, the service
    wants running commentary and the ability to be stopped part-way — including
    inside the long tracking pass, which is most of the runtime.
    """
    from ..calibrate import calibrate  # noqa: PLC0415
    from ..export import extension, write as export_write  # noqa: PLC0415
    from ..score import ScoreError, postprocess  # noqa: PLC0415
    from ..tiles import discover_palette  # noqa: PLC0415
    from ..tracking import transcribe_video  # noqa: PLC0415
    from ..video import VideoReader  # noqa: PLC0415

    # Recorded so the store can delete it once the job settles, whatever the
    # outcome. The reader is closed before then, which matters on Windows.
    job.source_path = Path(video)

    job.begin("fetch")
    with VideoReader(video, config) as reader:
        info = reader.info
        job.say(f"{info.source_width}x{info.source_height} - {info.fps:.2f} fps")
        if max_seconds and info.duration > max_seconds:
            raise Rejected(
                f"This video is {info.duration / 60:.0f} minutes long; the limit is "
                f"{max_seconds / 60:.0f}. Trim it to the part you want transcribed."
            )
        if 0 < info.frame_count < MIN_FRAMES:
            raise Rejected(
                f"This video is only {info.frame_count} "
                f"frame{'s' if info.frame_count > 1 else ''} long. Upload a "
                "recording that shows the tiles falling for at least a second."
            )
        if info.scale < 1.0:
            job.say(f"working at {info.width}x{info.height}")
        samples = reader.sample()
        job.say(f"sampled {len(samples)} frames for calibration")
        job.finish("fetch")

        job.begin("keys")
        calibration = calibrate(samples, config)
        job.say(str(calibration))
        job.finish("keys")

        job.begin("tiles")
        palette = discover_palette(samples, calibration, config)
        job.say(f"{palette.track_count} tile colour(s) discovered")
        job.finish("tiles")

        job.begin("notes")
        frames = until_cancelled(reader.frames(), job, total=info.frame_count or 0)
        sequence, speed = transcribe_video(reader, calibration, palette, config, frames=frames)
        job.finish("notes")
        # Measured inside the pass from the tiles it found; reported as its
        # own stage because it is a separate thing that can go wrong.
        job.begin("timing")
        job.say(f"scroll speed {speed.value:.1f} px/s (confidence {speed.confidence:.2f})")
        job.say(f"{len(sequence)} note events")
        job.finish("timing")

    if not len(sequence):
        raise Rejected(
            "No notes were found. The keyboard was recognised, but no tiles were "
            "seen reaching it."
        )

    job.begin("score")
    analysis = None
    try:
        sequence, analysis = postprocess(sequence, config)
        job.say(str(analysis))
    except ScoreError as exc:
        job.say(f"left unquantized: {exc}")

    sequence.source = f"dropscore:{job.label}"

    # Each format is written independently. PDF needs MuseScore installed on the
    # server, and letting one optional output take a completed transcription
    # down with it — the notes are already correct by this point — would be an
    # absurd trade.
    for format in formats:
        try:
            job.files[format] = export_write(
                sequence, output_path(job, extension(format)), format, analysis
            )
        except Exception as exc:  # noqa: BLE001 - one format must not fail the job
            job.failed_formats[format] = str(exc)
            job.say(f"could not write {format}: {exc}")
            log.warning("job %s: %s export failed: %s", job.id, format, exc)

    if job.files:
        job.say(f"wrote {', '.join(sorted(job.files))}")
    else:
        raise RuntimeError(
            "every requested output failed to write: "
            + "; ".join(f"{fmt}: {why}" for fmt, why in job.failed_formats.items())
        )
    job.finish("score")

    job.result = _summarize(sequence, analysis, calibration, speed)


def _summarize(sequence: NoteSequence, analysis, calibration, speed) -> dict[str, Any]:
    """The payload the results view renders."""
    pitches = [n.pitch for n in sequence]
    return {
        "notes": [
            {
                "t": round(n.onset, 4),
                "dur": round(n.duration, 4),
                "pitch": n.pitch,
                "hand": n.hand,
            }
            for n in sequence
        ],
        "count": len(sequence),
        "duration": round(sequence.duration, 2),
        "tempo": round(analysis.tempo, 1) if analysis else None,
        "key": analysis.key if analysis else None,
        "meter": f"{analysis.beats_per_bar}/{analysis.beat_type}" if analysis else None,
        "lowest": min(pitches) if pitches else None,
        "highest": max(pitches) if pitches else None,
        "confidence": round(
            min(calibration.confidence, speed.confidence) if speed else calibration.confidence, 2
        ),
    }
