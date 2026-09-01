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
from pathlib import Path
from typing import Any

from ..config import Config, DEFAULT
from ..sources import SourceError, parse_youtube_id
from .jobs import JobStore, Status, transcribe_job
from .options import TranscribeOptions

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The web service needs its optional dependencies: "
        'pip install -e ".[service]"'
    ) from exc

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

# The docs are mounted at /guide rather than /docs because FastAPI already
# serves its Swagger UI there, and having half a path belong to each would be
# needlessly confusing.
DOCS_MOUNT = "/guide"

# Checked twice: against the declared Content-Length before any work is done,
# and again while copying, for requests that arrive chunked or misdeclare it.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi"}


def _too_large() -> str:
    return f"That file is larger than {MAX_UPLOAD_BYTES / 1024 ** 3:.0f} GB"


CONTENT_TYPES = {
    "json": "application/json",
    "midi": "audio/midi",
    "musicxml": "application/vnd.recordare.musicxml+xml",
    "pdf": "application/pdf",
}


class UrlRequest(BaseModel):
    url: str
    options: TranscribeOptions = Field(default_factory=TranscribeOptions)


def _options(raw: str | None) -> TranscribeOptions:
    """Parse the options form field, turning bad input into a 400."""
    try:
        return TranscribeOptions.parse(raw)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid options: {exc}") from exc


def create_app(
    workdir: str | Path = "out/jobs",
    config: Config = DEFAULT,
    serve_frontend: bool = True,
    keep_sources: bool = False,
) -> "FastAPI":
    store = JobStore(Path(workdir), keep_sources=keep_sources)
    app = FastAPI(title="DropScore", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": "0.1.0"}

    # Deliberately a sync def, not async. Starlette runs sync endpoints in a
    # threadpool, whereas an async one doing blocking file writes stalls the
    # event loop for the whole upload — freezing every other request, including
    # the polling that drives other users' progress bars.
    @app.post("/api/jobs/upload")
    def submit_upload(
        request: Request,
        file: UploadFile = File(...),
        # Multipart has no nested objects, so the settings arrive as JSON text.
        options: str | None = Form(default=None),
    ) -> dict[str, Any]:
        chosen = _options(options)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                415, f"{suffix or 'that file'} is not a video DropScore can read"
            )

        # Starlette has already buffered the body before this runs, so checking
        # the declared length is the only cap that prevents the resources being
        # spent in the first place. The copy is still checked below for requests
        # that arrive chunked or lie about their size.
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, _too_large())

        # The filename is client-supplied; JobStore.create sanitises it.
        job = store.create(file.filename or "video")
        target = job.workdir / f"source{suffix}"

        written = 0
        oversize = False
        with target.open("wb") as out:
            while chunk := file.file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    oversize = True
                    break
                out.write(chunk)

        if oversize:
            # Outside the `with`, so the handle is closed before the directory
            # goes — Windows will not unlink a file that is still open.
            store.discard(job)
            raise HTTPException(413, _too_large())

        job.say(f"received {job.label}{suffix} ({written / 1e6:.1f} MB)")
        store.submit(
            job,
            lambda j: transcribe_job(j, target, chosen.apply(config), chosen.formats),
        )
        return {"id": job.id}

    @app.post("/api/jobs/url")
    def submit_url(request: UrlRequest) -> dict[str, Any]:
        video_id = parse_youtube_id(request.url)
        if not video_id:
            raise HTTPException(400, "That does not look like a YouTube link")

        chosen = request.options
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
            transcribe_job(
                current_job, resolved.path, chosen.apply(config), chosen.formats
            )

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

    if serve_frontend and DOCS_ROOT.is_dir():
        # Mounted before the catch-all below, which would otherwise swallow it.
        app.mount(DOCS_MOUNT, StaticFiles(directory=DOCS_ROOT), name="guide")

    if serve_frontend and WEB_ROOT.is_dir():
        app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
    else:
        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse({"ok": True, "frontend": "not bundled"})

    app.state.store = store
    return app
