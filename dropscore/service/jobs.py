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
from typing import Any, Callable, Iterable, Iterator

from ..config import Config, DEFAULT
from ..notes import NoteSequence

log = logging.getLogger(__name__)

# The stages the UI already knows how to draw.
STAGES: tuple[tuple[str, str], ...] = (
    ("fetch", "Reading video"),
    ("keys", "Calibrating keyboard geometry"),
    ("tiles", "Detecting and tracking tiles"),
    ("timing", "Solving fall speed and onsets"),
    ("notes", "Building note events"),
    ("score", "Quantizing and engraving"),
)


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


#: States a job never leaves. Only these are safe to evict.
TERMINAL = frozenset({Status.DONE, Status.ERROR, Status.CANCELLED})


class Cancelled(Exception):
    """Raised inside a worker when the job has been cancelled."""


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


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
    source_path: Path | None = None  # the input video, deleted once it is read
    status: Status = Status.QUEUED
    stage_states: dict[str, str] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    result: dict[str, Any] | None = None
    files: dict[str, Path] = field(default_factory=dict)
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

    def finish(self, stage: str) -> None:
        with self._lock:
            self.stage_states[stage] = "done"

    def say(self, message: str) -> None:
        with self._lock:
            self.log.append(message)
            # A runaway pipeline must not grow the log without bound.
            if len(self.log) > 500:
                del self.log[:100]
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
                "error": self.error,
                "elapsed": (self.finished or time.time()) - self.created,
                "result": self.result,
                "formats": sorted(self.files),
            }


class JobStore:
    """Holds jobs and runs them. Thread-safe."""

    def __init__(
        self,
        root: Path,
        workers: int = 2,
        retain: int = 50,
        keep_sources: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retain = retain
        # Sources are deleted whatever the outcome, not just on success: a
        # service must not depend on failures being rare to stay within its
        # disk budget. Set this to keep them for debugging a bad video.
        self.keep_sources = keep_sources
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dropscore")

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

    def get(self, job_id: str) -> Job | None:
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
            job.error = str(exc)
            job.say(f"error: {exc}")
            log.exception("job %s failed", job.id)
        finally:
            job.finished = time.time()
            # In the finally block so a failed or cancelled job releases its
            # video too — the outcome does not change that it is dead weight.
            if not self.keep_sources:
                job.discard_source()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def until_cancelled(frames: Iterable[Any], job: Job, report_every: int = 200) -> Iterator[Any]:
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
        if report_every and index % report_every == 0:
            job.say(f"tracked {index} frames")
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


def transcribe_job(job: Job, video: Path, config: Config = DEFAULT) -> None:
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
    from ..tracking import estimate_speed, transcribe  # noqa: PLC0415
    from ..video import VideoReader  # noqa: PLC0415

    # Recorded so the store can delete it once the job settles, whatever the
    # outcome. The reader is closed before then, which matters on Windows.
    job.source_path = Path(video)

    job.begin("fetch")
    with VideoReader(video, config) as reader:
        info = reader.info
        job.say(f"{info.source_width}x{info.source_height} - {info.fps:.2f} fps")
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

        job.begin("timing")
        start = (info.frame_count or 0) // 3
        window = list(reader.frames(start=start, stop=start + 40))
        speed = estimate_speed(window, calibration, config=config)
        job.say(f"scroll speed {speed.value:.1f} px/s (confidence {speed.confidence:.2f})")
        job.finish("timing")

        job.begin("notes")
        sequence = transcribe(
            until_cancelled(reader.frames(), job), calibration, palette, speed, config
        )
        job.say(f"{len(sequence)} note events")
        job.finish("notes")

    job.begin("score")
    analysis = None
    if len(sequence):
        try:
            sequence, analysis = postprocess(sequence, config)
            job.say(str(analysis))
        except ScoreError as exc:
            job.say(f"left unquantized: {exc}")

    sequence.source = f"dropscore:{job.label}"
    for format in ("json", "midi", "musicxml"):
        job.files[format] = export_write(
            sequence, output_path(job, extension(format)), format, analysis
        )
    job.say(f"wrote {', '.join(sorted(job.files))}")
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
        "meter": f"{analysis.beats_per_bar}/4" if analysis else None,
        "lowest": min(pitches) if pitches else None,
        "highest": max(pitches) if pitches else None,
        "confidence": round(
            min(calibration.confidence, speed.confidence) if speed else calibration.confidence, 2
        ),
    }
