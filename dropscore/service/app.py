"""HTTP API, and the static frontend served alongside it.

The endpoint shape follows what the frontend already does: submit a source, poll a
job, download the results. Polling rather than websockets is deliberate — the UI
was written to poll, transcription takes seconds not milliseconds, and a socket
would add reconnection handling for no benefit at this size.

FastAPI is an optional dependency (``pip install -e ".[service]"``). Importing
this module without it raises with an instruction rather than a traceback.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ..config import Config, DEFAULT
from ..sources import SourceError, parse_youtube_id
from .jobs import JobStore, Status, transcribe_job

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The web service needs its optional dependencies: "
        'pip install -e ".[service]"'
    ) from exc

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"

# Uploads are read in chunks and refused past this, so a huge file cannot fill
# the disk before anyone notices.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi"}

CONTENT_TYPES = {
    "json": "application/json",
    "midi": "audio/midi",
    "musicxml": "application/vnd.recordare.musicxml+xml",
    "pdf": "application/pdf",
}


class UrlRequest(BaseModel):
    url: str


def create_app(
    workdir: str | Path = "out/jobs",
    config: Config = DEFAULT,
    serve_frontend: bool = True,
) -> "FastAPI":
    store = JobStore(Path(workdir))
    app = FastAPI(title="DropScore", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": "0.1.0"}

    @app.post("/api/jobs/upload")
    async def submit_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                415, f"{suffix or 'that file'} is not a video DropScore can read"
            )

        # The filename is client-supplied; JobStore.create sanitises it.
        job = store.create(file.filename or "video")
        target = job.workdir / f"source{suffix}"

        written = 0
        with target.open("wb") as out:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job.workdir, ignore_errors=True)
                    raise HTTPException(413, "That file is larger than 2 GB")
                out.write(chunk)

        job.say(f"received {job.label}{suffix} ({written / 1e6:.1f} MB)")
        store.submit(job, lambda j: transcribe_job(j, target, config))
        return {"id": job.id}

    @app.post("/api/jobs/url")
    def submit_url(request: UrlRequest) -> dict[str, Any]:
        video_id = parse_youtube_id(request.url)
        if not video_id:
            raise HTTPException(400, "That does not look like a YouTube link")

        job = store.create(video_id)
        job.say(
            "note: downloading from YouTube is against their Terms of Service; "
            "uploading a file is the supported path"
        )

        def work(current_job) -> None:
            from ..sources import resolve  # noqa: PLC0415

            try:
                resolved = resolve(video_id, current_job.workdir)
            except SourceError as exc:
                raise RuntimeError(str(exc)) from exc
            transcribe_job(current_job, resolved.path, config)

        store.submit(job, work)
        return {"id": job.id}

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        return job.snapshot()

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        job.cancel()
        return {"id": job.id, "status": job.status.value}

    @app.get("/api/jobs/{job_id}/download/{format}")
    def download(job_id: str, format: str) -> FileResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        if job.status is not Status.DONE:
            raise HTTPException(409, f"Job is {job.status.value}, not finished")

        path = job.files.get(format)
        if path is None or not path.exists():
            raise HTTPException(404, f"No {format} output for this job")

        return FileResponse(
            path,
            media_type=CONTENT_TYPES.get(format, "application/octet-stream"),
            filename=path.name,
        )

    @app.on_event("shutdown")
    def stop() -> None:
        store.shutdown()

    if serve_frontend and WEB_ROOT.is_dir():
        app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
    else:
        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse({"ok": True, "frontend": "not bundled"})

    app.state.store = store
    return app
