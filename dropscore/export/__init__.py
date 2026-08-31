"""Stage 8: write the transcription out.

Three formats, in descending order of how much you should trust them:

* **MIDI** — exactly what the tiles said. No editorial decisions.
* **MusicXML** — mechanically correct, musically approximate. One voice per
  staff, no beaming. See ``musicxml`` for the full list of what it does not do.
* **PDF** — MusicXML handed to MuseScore, which must be installed separately.
"""

from __future__ import annotations

from pathlib import Path

from ..notes import NoteSequence
from ..score import Analysis
from . import midi, musicxml, pdf
from .pdf import EngraverNotFound

FORMATS = {
    "midi": (".mid", midi.write),
    "musicxml": (".musicxml", musicxml.write),
    "pdf": (".pdf", pdf.write),
    "json": (".json", None),  # handled by NoteSequence.save
}


def extension(format: str) -> str:
    try:
        return FORMATS[format][0]
    except KeyError:
        raise ValueError(
            f"unknown format {format!r}; available: {', '.join(sorted(FORMATS))}"
        ) from None


def write(
    sequence: NoteSequence,
    path: str | Path,
    format: str,
    analysis: Analysis | None = None,
) -> Path:
    """Write ``sequence`` in one format, choosing the writer by name."""
    if format == "json":
        return sequence.save(path)

    _, writer = FORMATS[format] if format in FORMATS else (None, None)
    if writer is None:
        raise ValueError(
            f"unknown format {format!r}; available: {', '.join(sorted(FORMATS))}"
        )
    return writer(sequence, path, analysis)


__all__ = [
    "FORMATS",
    "EngraverNotFound",
    "extension",
    "write",
    "midi",
    "musicxml",
    "pdf",
]
