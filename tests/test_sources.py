from __future__ import annotations

from pathlib import Path

import pytest

from dropscore.sources import SourceError, looks_like_url, parse_youtube_id, resolve


@pytest.mark.parametrize(
    "text",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "http://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=42",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
        "www.youtube.com/watch?v=dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ],
)
def test_parses_youtube_ids(text: str) -> None:
    assert parse_youtube_id(text) == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "not a url",
        "https://vimeo.com/12345678",
        "https://www.youtube.com/watch?v=tooshort",
        "https://example.com/watch?v=dQw4w9WgXcQ",  # right shape, wrong host
        "https://www.youtube.com/results?search_query=piano",
    ],
)
def test_rejects_non_youtube(text: str) -> None:
    assert parse_youtube_id(text) is None


def test_looks_like_url() -> None:
    assert looks_like_url("https://example.com/a.mp4")
    assert not looks_like_url("C:/videos/a.mp4")
    assert not looks_like_url("a.mp4")


def test_resolve_uses_local_file_in_place(clip: Path) -> None:
    resolved = resolve(str(clip))
    assert resolved.path == clip
    assert resolved.label == clip.stem
    assert resolved.is_temporary is False


def test_resolve_rejects_missing_file() -> None:
    with pytest.raises(SourceError, match="No such file"):
        resolve("definitely/not/here.mp4")


def test_resolve_rejects_non_youtube_url() -> None:
    with pytest.raises(SourceError, match="only YouTube URLs and local"):
        resolve("https://vimeo.com/12345678")
