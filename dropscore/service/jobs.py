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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

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


class Cancelled(Exception):
    """Raised inside a worker when the job has been cancelled."""


@dataclass
class Job:
    """One transcription, its progress, and whatever it produced."""

    id: str
    label: str
    workdir: Path
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

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

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

    def __init__(self, root: Path, workers: int = 2, retain: int = 50) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retain = retain
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dropscore")

    def create(self, label: str) -> Job:
        job_id = uuid.uuid4().hex
        workdir = self.root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, label=label, workdir=workdir)

        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._evict()
        return job

    def _evict(self) -> None:
        """Drop the oldest finished jobs once past the retention limit."""
        while len(self._order) > self.retain:
            oldest = self._order.pop(0)
            job = self._jobs.pop(oldest, None)
            if job and job.workdir.exists():
                import shutil  # noqa: PLC0415

                shutil.rmtree(job.workdir, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

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

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def transcribe_job(job: Job, video: Path, config: Config = DEFAULT) -> None:
    """The pipeline, stage by stage, reporting as it goes.

    This mirrors ``evaluate.run_clip`` rather than sharing it, because the two
    want different things: the harness wants a score and no chatter, the service
    wants running commentary and the ability to stop between stages.
    """
    from ..calibrate import calibrate  # noqa: PLC0415
    from ..export import extension, write as export_write  # noqa: PLC0415
    from ..score import ScoreError, postprocess  # noqa: PLC0415
    from ..tiles import discover_palette  # noqa: PLC0415
    from ..tracking import estimate_speed, transcribe  # noqa: PLC0415
    from ..video import VideoReader  # noqa: PLC0415

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
        job.check_cancelled()
        sequence = transcribe(reader.frames(), calibration, palette, speed, config)
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
        target = job.workdir / f"{job.label}{extension(format)}"
        job.files[format] = export_write(sequence, target, format, analysis)
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
