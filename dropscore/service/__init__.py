"""Stage 10: the HTTP service and the frontend it serves.

FastAPI and uvicorn are optional, so importing ``create_app`` fails with an
instruction rather than a traceback when the extra is not installed.
"""

from .jobs import STAGES, Job, JobStore, Status, safe_label, transcribe_job

__all__ = [
    "STAGES",
    "Job",
    "JobStore",
    "Status",
    "safe_label",
    "transcribe_job",
    "create_app",
]


def create_app(*args, **kwargs):
    """Build the FastAPI app. Imported lazily so the extra stays optional."""
    from .app import create_app as build

    return build(*args, **kwargs)
