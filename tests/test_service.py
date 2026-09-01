"""Stage 10.

Skipped wholesale when the optional service extra is not installed, so the core
suite still runs on a bare install.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from dropscore.synth import RenderSpec, generate, render

fastapi = pytest.importorskip("fastapi", reason="needs the [service] extra")
pytest.importorskip("httpx", reason="needs the [dev] extra")

from fastapi.testclient import TestClient  # noqa: E402

from dropscore.service import (  # noqa: E402
    STAGES,
    JobStore,
    Status,
    safe_label,
    until_cancelled,
)
from dropscore.service.app import create_app  # noqa: E402
from dropscore.service.jobs import output_path  # noqa: E402

SPEC = RenderSpec(width=640, height=360, fps=10.0)


@pytest.fixture
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("service-clip")
    video, _ = render(generate(seed=1, bars=2), directory / "demo.mp4", SPEC)
    return video


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    with TestClient(create_app(workdir=tmp_path / "jobs", serve_frontend=False)) as test_client:
        yield test_client


def _wait(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "error", "cancelled"}:
            return job
        time.sleep(0.2)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


# ── basics ───────────────────────────────────────────────────────────


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json()["ok"] is True


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/nope").status_code == 404


def test_rejects_a_non_video_upload(client: TestClient) -> None:
    response = client.post(
        "/api/jobs/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_rejects_a_non_youtube_url(client: TestClient) -> None:
    response = client.post("/api/jobs/url", json={"url": "https://vimeo.com/12345678"})
    assert response.status_code == 400


# ── the real path ────────────────────────────────────────────────────


def test_upload_runs_the_pipeline_and_reports_progress(client: TestClient, clip: Path) -> None:
    with clip.open("rb") as handle:
        response = client.post(
            "/api/jobs/upload", files={"file": (clip.name, handle, "video/mp4")}
        )
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = _wait(client, job_id)
    assert job["status"] == "done", job.get("error")
    assert job["progress"] == 1.0
    assert [s["state"] for s in job["stages"]] == ["done"] * len(STAGES)
    assert job["log"]


def test_stage_keys_match_the_frontend_step_list(client: TestClient, clip: Path) -> None:
    """The UI renders these keys directly, so they are part of the contract."""
    app_js = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")
    for key, _ in STAGES:
        assert f"key: '{key}'" in app_js, f"frontend has no step for stage {key!r}"


def test_result_carries_notes_the_roll_can_draw(client: TestClient, clip: Path) -> None:
    with clip.open("rb") as handle:
        job_id = client.post(
            "/api/jobs/upload", files={"file": (clip.name, handle, "video/mp4")}
        ).json()["id"]

    job = _wait(client, job_id)
    result = job["result"]
    assert result["count"] == len(result["notes"])
    for note in result["notes"]:
        assert {"t", "dur", "pitch", "hand"} <= note.keys()
        assert 21 <= note["pitch"] <= 108
        assert note["hand"] in {"L", "R"}


def test_outputs_are_downloadable(client: TestClient, clip: Path) -> None:
    with clip.open("rb") as handle:
        job_id = client.post(
            "/api/jobs/upload", files={"file": (clip.name, handle, "video/mp4")}
        ).json()["id"]

    job = _wait(client, job_id)
    assert set(job["formats"]) >= {"json", "midi", "musicxml"}

    midi = client.get(f"/api/jobs/{job_id}/download/midi")
    assert midi.status_code == 200
    assert midi.content[:4] == b"MThd"

    assert client.get(f"/api/jobs/{job_id}/download/pdf").status_code == 404


def test_download_before_completion_is_refused(client: TestClient, tmp_path: Path) -> None:
    store = client.app.state.store
    job = store.create("pending")
    assert client.get(f"/api/jobs/{job.id}/download/midi").status_code == 409


def test_cancel_marks_the_job(client: TestClient) -> None:
    store = client.app.state.store
    job = store.create("pending")
    response = client.delete(f"/api/jobs/{job.id}")
    assert response.status_code == 200
    assert job.cancelled


# ── cancellation actually stops work ─────────────────────────────────


def test_until_cancelled_passes_frames_through(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("x")
    assert list(until_cancelled(range(5), job, report_every=0)) == [0, 1, 2, 3, 4]


def test_until_cancelled_stops_mid_stream(tmp_path: Path) -> None:
    """The point of the fix: a long pass must not run to completion."""
    from dropscore.service.jobs import Cancelled  # noqa: PLC0415

    job = JobStore(tmp_path).create("x")
    consumed = []

    with pytest.raises(Cancelled):
        for value in until_cancelled(range(10_000), job, report_every=0):
            consumed.append(value)
            if len(consumed) == 3:
                job.cancel()

    assert len(consumed) == 3, "iteration continued past the cancel"


def test_until_cancelled_is_lazy(tmp_path: Path) -> None:
    """It must not drain the source before yielding, or a 2 GB video buffers."""
    job = JobStore(tmp_path).create("x")
    pulled = 0

    def source():
        nonlocal pulled
        for i in range(1000):
            pulled += 1
            yield i

    stream = until_cancelled(source(), job, report_every=0)
    next(stream)
    assert pulled == 1


def test_until_cancelled_reports_progress(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("x")
    list(until_cancelled(range(25), job, report_every=10))
    assert [line for line in job.log if "tracked" in line] == [
        "tracked 10 frames",
        "tracked 20 frames",
    ]


def test_cancelling_a_long_pass_releases_the_worker(tmp_path: Path) -> None:
    """A cancelled job must free its pool slot promptly, not at completion."""
    store = JobStore(tmp_path, workers=1)
    slow = store.create("slow")
    started = threading.Event()

    def long_pass(current) -> None:
        def frames():
            started.set()
            while True:
                yield 1
                time.sleep(0.001)

        for _ in until_cancelled(frames(), current, report_every=0):
            pass

    store.submit(slow, long_pass)
    assert started.wait(5), "the slow job never started"

    slow.cancel()
    for _ in range(200):
        if slow.status is Status.CANCELLED:
            break
        time.sleep(0.02)
    assert slow.status is Status.CANCELLED

    # With the single worker released, a following job can run.
    after = store.create("after")
    ran = threading.Event()
    store.submit(after, lambda _job: ran.set())
    assert ran.wait(5), "the worker was still held by the cancelled job"


# ── the source video is not kept ─────────────────────────────────────


def test_source_video_is_deleted_but_outputs_survive(client: TestClient, clip: Path) -> None:
    with clip.open("rb") as handle:
        job_id = client.post(
            "/api/jobs/upload", files={"file": (clip.name, handle, "video/mp4")}
        ).json()["id"]

    job = _wait(client, job_id)
    assert job["status"] == "done", job.get("error")

    stored = client.app.state.store.get(job_id)
    assert not list(stored.workdir.glob("source.*")), "the uploaded video was kept"
    assert stored.workdir.exists()
    for path in stored.files.values():
        assert path.exists()

    # And the outputs are still downloadable afterwards.
    assert client.get(f"/api/jobs/{job_id}/download/midi").status_code == 200


def test_source_is_deleted_even_when_the_job_fails(tmp_path: Path) -> None:
    """A failure does not make a 2 GB upload any less dead weight."""
    store = JobStore(tmp_path)
    job = store.create("x")
    source = job.workdir / "source.mp4"
    source.write_bytes(b"not a video")

    def work(current) -> None:
        current.source_path = source
        raise RuntimeError("boom")

    store.submit(job, work)
    for _ in range(100):
        if job.status is Status.ERROR:
            break
        time.sleep(0.05)

    assert job.status is Status.ERROR
    assert not source.exists()


def test_keep_sources_retains_the_upload(tmp_path: Path) -> None:
    store = JobStore(tmp_path, keep_sources=True)
    job = store.create("x")
    source = job.workdir / "source.mp4"
    source.write_bytes(b"not a video")

    def work(current) -> None:
        current.source_path = source

    store.submit(job, work)
    for _ in range(100):
        if job.status is Status.DONE:
            break
        time.sleep(0.05)

    assert source.exists()


def test_discard_source_is_safe_to_repeat(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("x")
    source = job.workdir / "source.mp4"
    source.write_bytes(b"x")
    job.source_path = source

    job.discard_source()
    job.discard_source()  # already gone, and source_path is now None
    assert not source.exists()
    assert job.source_path is None


def test_discard_source_tolerates_a_missing_file(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("x")
    job.source_path = job.workdir / "never-written.mp4"
    job.discard_source()
    assert job.source_path is None


# ── label sanitising ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("nocturne.mp4", "nocturne"),
        ("my clip.mp4", "my_clip"),
        ("Chopin - Op9 No2.webm", "Chopin_-_Op9_No2"),
        ("/home/user/videos/song.mp4", "song"),
        ("C:\\Users\\me\\song.mp4", "song"),
        # Traversal attempts, under either separator.
        ("../../../etc/passwd", "passwd"),
        ("..\\..\\..\\evil.mid", "evil"),
        ("....//....//evil.mp4", "evil"),
        # Names that sanitise away entirely fall back rather than becoming "".
        ("..", "video"),
        (".", "video"),
        ("", "video"),
        ("   ", "video"),
        ("...mp4", "video"),
        ("曲.mp4", "video"),
    ],
)
def test_safe_label(raw: str, expected: str) -> None:
    assert safe_label(raw) == expected


def test_safe_label_never_contains_a_separator() -> None:
    for raw in ("a/b/c.mp4", "a\\b\\c.mp4", "..\\..\\x", "/", "\\", "a:b.mp4"):
        label = safe_label(raw)
        assert "/" not in label and "\\" not in label and ".." not in label


def test_safe_label_is_length_capped() -> None:
    assert len(safe_label("x" * 500 + ".mp4")) == 48


def test_output_path_stays_inside_the_job_directory(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("..\\..\\evil.mid")
    target = output_path(job, ".mid")
    assert target.is_relative_to(job.workdir.resolve())


def test_output_path_refuses_an_escaping_label(tmp_path: Path) -> None:
    """The second line of defence, for a Job not built via JobStore.create."""
    from dropscore.service.jobs import Job  # noqa: PLC0415

    workdir = tmp_path / "job"
    workdir.mkdir()
    job = Job(id="x", label="../../escaped", workdir=workdir)
    with pytest.raises(ValueError, match="outside"):
        output_path(job, ".mid")


def test_upload_with_a_traversing_filename_writes_inside_the_job(
    client: TestClient, clip: Path
) -> None:
    with clip.open("rb") as handle:
        response = client.post(
            "/api/jobs/upload",
            files={"file": ("..\\..\\..\\pwned.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = _wait(client, job_id)
    assert job["status"] == "done", job.get("error")
    assert job["label"] == "pwned"

    store = client.app.state.store
    workdir = store.get(job_id).workdir.resolve()
    for path in store.get(job_id).files.values():
        assert path.resolve().is_relative_to(workdir)


# ── the store on its own ─────────────────────────────────────────────


def test_job_starts_with_every_stage_pending(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("x")
    assert set(job.stage_states.values()) == {"pending"}
    assert job.progress == 0.0


def test_progress_tracks_finished_stages(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("x")
    job.begin("fetch")
    assert job.progress == 0.0
    job.finish("fetch")
    assert job.progress == pytest.approx(1 / len(STAGES))


def test_beginning_a_stage_after_cancel_raises(tmp_path: Path) -> None:
    from dropscore.service.jobs import Cancelled  # noqa: PLC0415

    job = JobStore(tmp_path).create("x")
    job.cancel()
    with pytest.raises(Cancelled):
        job.begin("fetch")


def test_failures_are_captured_as_job_state(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job = store.create("x")

    def explode(_job) -> None:
        raise RuntimeError("nope")

    store.submit(job, explode)
    for _ in range(100):
        if job.status is Status.ERROR:
            break
        time.sleep(0.05)

    assert job.status is Status.ERROR
    assert job.error == "nope"


def test_log_does_not_grow_without_bound(tmp_path: Path) -> None:
    job = JobStore(tmp_path).create("x")
    for i in range(900):
        job.say(str(i))
    assert len(job.log) <= 500


def _finish(job) -> None:
    """Mark a job terminal without running anything through the pool."""
    job.status = Status.DONE


def test_finished_jobs_are_evicted(tmp_path: Path) -> None:
    store = JobStore(tmp_path, retain=3)
    jobs = []
    for i in range(5):
        job = store.create(f"job{i}")
        _finish(job)
        jobs.append(job)

    assert store.get(jobs[0].id) is None
    assert store.get(jobs[1].id) is None
    assert store.get(jobs[-1].id) is not None
    assert not jobs[0].workdir.exists()
    assert jobs[-1].workdir.exists()


def test_running_jobs_are_never_evicted(tmp_path: Path) -> None:
    """Evicting one would delete its workdir mid-write and 404 a live id."""
    store = JobStore(tmp_path, retain=2)
    live = [store.create(f"live{i}") for i in range(4)]
    for job in live:
        job.status = Status.RUNNING

    store.create("one-more")

    for job in live:
        assert store.get(job.id) is not None, "a running job was evicted"
        assert job.workdir.exists()


def test_eviction_takes_the_finished_job_and_leaves_the_running_one(tmp_path: Path) -> None:
    store = JobStore(tmp_path, retain=2)
    finished = store.create("finished")
    _finish(finished)
    running = store.create("running")
    running.status = Status.RUNNING

    store.create("newest")  # pushes the store over its limit

    assert store.get(finished.id) is None, "the finished job should be reclaimed"
    assert store.get(running.id) is not None, "the running job should survive"


def test_store_may_exceed_retention_while_everything_is_live(tmp_path: Path) -> None:
    store = JobStore(tmp_path, retain=1)
    jobs = [store.create(f"j{i}") for i in range(4)]
    for job in jobs:
        job.status = Status.RUNNING

    store.create("another")
    assert all(store.get(job.id) is not None for job in jobs)


def test_eviction_resumes_once_jobs_finish(tmp_path: Path) -> None:
    store = JobStore(tmp_path, retain=1)
    first = store.create("first")
    first.status = Status.RUNNING
    second = store.create("second")
    second.status = Status.RUNNING

    store.create("third")
    assert store.get(first.id) is not None  # nothing eligible yet

    _finish(first)
    store.create("fourth")
    assert store.get(first.id) is None, "the finished job should now be reclaimed"
