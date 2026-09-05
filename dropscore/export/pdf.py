"""PDF engraving.

Engraving is not something to reimplement, and two routes exist that do it
properly. Verovio engraves MusicXML in-process and is a pip install, so it is
tried first: PDF stops being the one export that needs a GUI application on
PATH before it will work, which mattered most for the service, where the
person running it and the person wanting the PDF are not the same.

MuseScore remains the fallback. It is the better engraver of the two and is
what anyone editing the result will already have, so where it is installed
there is no reason not to use it.
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
    """Raised when no engraver is available at all."""


def find_engraver() -> str | None:
    for command in MUSESCORE_COMMANDS:
        path = shutil.which(command)
        if path:
            return path
    return None


def _verovio_available() -> bool:
    """Whether the in-process route has everything it needs.

    Verovio renders to SVG rather than PDF, so the conversion needs svglib and
    reportlab too. All three are pure pip installs; any one missing falls back.
    """
    try:
        import reportlab  # noqa: F401,PLC0415
        import svglib  # noqa: F401,PLC0415
        import verovio  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def available() -> bool:
    """Whether a PDF can be produced by any route."""
    return _verovio_available() or find_engraver() is not None


def _engrave_with_verovio(xml_path: Path, path: Path) -> Path:
    """Engrave in-process: MusicXML to SVG to PDF."""
    from io import BytesIO  # noqa: PLC0415

    import verovio  # noqa: PLC0415
    from reportlab.graphics import renderPDF  # noqa: PLC0415
    from reportlab.pdfgen import canvas as pdf_canvas  # noqa: PLC0415
    from svglib.svglib import svg2rlg  # noqa: PLC0415

    toolkit = verovio.toolkit()
    toolkit.setOptions(
        {
            "pageWidth": 2100,
            "pageHeight": 2970,
            "scale": 38,
            "adjustPageHeight": False,
            "breaks": "auto",
            "footer": "none",
            "header": "none",
        }
    )
    if not toolkit.loadData(xml_path.read_text(encoding="utf-8")):
        raise EngraverNotFound("verovio could not read the MusicXML")

    pages = toolkit.getPageCount()
    if pages < 1:
        raise EngraverNotFound("verovio produced no pages")

    canvas = pdf_canvas.Canvas(str(path))
    for number in range(1, pages + 1):
        svg = toolkit.renderToSVG(number)
        drawing = svg2rlg(BytesIO(svg.encode("utf-8")))
        if drawing is None:
            raise EngraverNotFound(f"could not convert page {number} to PDF")
        canvas.setPageSize((drawing.width, drawing.height))
        renderPDF.draw(drawing, canvas, 0, 0)
        canvas.showPage()
    canvas.save()

    log.info("engraved %s with verovio (%d page(s))", path, pages)
    return path


def write(
    sequence: NoteSequence,
    path: str | Path,
    analysis: Analysis | None = None,
    keep_musicxml: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Engrave to PDF, in-process where possible."""
    engraver = find_engraver()
    if engraver is None and not _verovio_available():
        raise EngraverNotFound(
            "PDF output needs either verovio (pip install "
            'dropscore"[pdf]") or MuseScore on PATH (tried: '
            f"{', '.join(MUSESCORE_COMMANDS)}). Or export MusicXML and open "
            "that in any notation editor."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_path = path.with_suffix(".musicxml")
    musicxml.write(sequence, xml_path, analysis)

    if engraver is None:
        try:
            return _engrave_with_verovio(xml_path, path)
        finally:
            if not keep_musicxml and xml_path.exists() and path.exists():
                xml_path.unlink()

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
