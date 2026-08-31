"""PDF engraving, by handing MusicXML to whichever engraver is installed.

Engraving is not something to reimplement: MuseScore and LilyPond exist, do it
far better than a few hundred lines could, and are the tools anyone editing the
result will already have. This writes MusicXML and shells out.

Neither is a Python dependency, so this is the one export that can be
unavailable. When it is, the error says exactly what to install rather than
failing somewhere inside a subprocess.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from ..notes import NoteSequence
from ..score import Analysis
from . import musicxml

log = logging.getLogger(__name__)

# Command names to try, in preference order. MuseScore's CLI converts MusicXML
# to PDF directly; LilyPond needs the MusicXML converted to .ly first.
MUSESCORE_COMMANDS = ("mscore", "musescore", "MuseScore4", "MuseScore3", "mscore4")


class EngraverNotFound(RuntimeError):
    """Raised when no external engraver is installed."""


def find_engraver() -> str | None:
    for command in MUSESCORE_COMMANDS:
        path = shutil.which(command)
        if path:
            return path
    return None


def write(
    sequence: NoteSequence,
    path: str | Path,
    analysis: Analysis | None = None,
    keep_musicxml: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Engrave to PDF via an external engraver."""
    engraver = find_engraver()
    if engraver is None:
        raise EngraverNotFound(
            "PDF output needs MuseScore on PATH (tried: "
            f"{', '.join(MUSESCORE_COMMANDS)}). Install it, or export MusicXML "
            "and open that in any notation editor."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_path = path.with_suffix(".musicxml")
    musicxml.write(sequence, xml_path, analysis)

    try:
        result = subprocess.run(
            [engraver, "-o", str(path), str(xml_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EngraverNotFound(f"{engraver} did not finish within {timeout}s") from exc
    finally:
        if not keep_musicxml and xml_path.exists() and path.exists():
            xml_path.unlink()

    if result.returncode != 0 or not path.exists():
        raise EngraverNotFound(
            f"{Path(engraver).name} failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout).strip()[:400]}"
        )

    log.info("engraved %s", path)
    return path
