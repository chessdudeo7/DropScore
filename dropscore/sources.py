"""Resolving a user-supplied source down to a local video file.

The pipeline only ever consumes a decoded local file. Ingestion is deliberately a
thin layer in front of that, so adding or removing a source path never touches
anything downstream.

On the YouTube path: downloading from YouTube is against their Terms of Service.
It exists here as a local-experimentation convenience, is gated behind an optional
dependency, and is not the path to build a service on — see docs/PIPELINE.md.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

_YOUTUBE_HOSTS = {
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "music.youtube.com",
}

_ID_RE = re.compile(r"^[\w-]{11}$")
_PATH_ID_RE = re.compile(r"^/(?:shorts|embed|v|live)/([\w-]{11})")


class SourceError(RuntimeError):
    """Raised when a source cannot be resolved to a local file."""


@dataclass(frozen=True)
class ResolvedSource:
    path: Path
    label: str  # human-readable origin, for logs and output filenames
    is_temporary: bool  # True when we downloaded it and may clean it up


def parse_youtube_id(text: str) -> str | None:
    """Extract an 11-character YouTube id, mirroring the frontend's parser."""
    text = text.strip()
    if not text:
        return None
    if _ID_RE.match(text):
        return text

    parsed = urlparse(text if "//" in text else f"https://{text}")
    host = parsed.hostname or ""
    host = re.sub(r"^(www|m)\.", "", host.lower())
    if host not in _YOUTUBE_HOSTS:
        return None

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if _ID_RE.match(candidate) else None

    values = parse_qs(parsed.query).get("v", [])
    if values and _ID_RE.match(values[0]):
        return values[0]

    match = _PATH_ID_RE.match(parsed.path)
    return match.group(1) if match else None


def looks_like_url(text: str) -> bool:
    return bool(urlparse(text).scheme in {"http", "https"})


def resolve(source: str, dest_dir: Path | None = None) -> ResolvedSource:
    """Turn a path or URL into a local file.

    A local path is used in place; a YouTube URL is downloaded into ``dest_dir``
    (or a temporary directory) if the optional ``youtube`` extra is installed.
    """
    candidate = Path(source).expanduser()
    if candidate.exists():
        return ResolvedSource(path=candidate, label=candidate.stem, is_temporary=False)

    video_id = parse_youtube_id(source)
    if video_id:
        return _download_youtube(video_id, dest_dir)

    if looks_like_url(source):
        raise SourceError(
            f"{source} is not a YouTube URL, and only YouTube URLs and local "
            f"files are supported. Download it and pass the file instead."
        )

    raise SourceError(f"No such file, and not a recognised URL: {source}")


def _download_youtube(video_id: str, dest_dir: Path | None) -> ResolvedSource:
    try:
        import yt_dlp  # noqa: PLC0415  (optional dependency, imported on use)
    except ImportError as exc:
        raise SourceError(
            "Reading from YouTube needs the optional dependency: "
            "pip install 'dropscore[youtube]'. Note that downloading from "
            "YouTube is against their Terms of Service; prefer passing a local "
            "file."
        ) from exc

    import tempfile

    dest_dir = dest_dir or Path(tempfile.mkdtemp(prefix="dropscore-"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    log.warning(
        "Downloading from YouTube violates their Terms of Service. "
        "This path is for local experimentation only."
    )

    options = {
        # 720p is all the pipeline uses; asking for more wastes bandwidth.
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": str(dest_dir / f"{video_id}.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        path = Path(ydl.prepare_filename(info))

    if not path.exists():
        # Merging can change the extension out from under prepare_filename().
        matches = sorted(dest_dir.glob(f"{video_id}.*"))
        if not matches:
            raise SourceError(f"Download reported success but produced no file in {dest_dir}")
        path = matches[0]

    return ResolvedSource(path=path, label=video_id, is_temporary=True)
