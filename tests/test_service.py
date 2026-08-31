"""Stage 10.

Skipped wholesale when the optional service extra is not installed, so the core
suite still runs on a bare install.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from dropscore.synth import RenderSpec, generate, render

fastapi = pytest.importorskip("fastapi", reason="needs the [service] extra")
pytest.importorskip("httpx", reason="needs the [dev] extra")

from fastapi.testclient import TestClient  # noqa: E402

from dropscore.service import STAGES, JobStore, Status  # noqa: E402
from dropscore.service.app import create_app  # noqa: E402

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


def test_old_jobs_are_evicted(tmp_path: Path) -> None:
    store = JobStore(tmp_path, retain=3)
    jobs = [store.create(f"job{i}") for i in range(5)]
    assert store.get(jobs[0].id) is None
    assert store.get(jobs[-1].id) is not None
    assert not jobs[0].workdir.exists()
