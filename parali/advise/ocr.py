"""OCR for scanned PDFs (Tesseract). Optional: needs the `tesseract` and `pdftoppm` programs.

Many government notices are scans. Some have no text layer at all; others have a poor one
("Jowohor Vyopor" for "Jawahar Vyapar"). `python -m parali ocr` writes a sidecar
`<name>.ocr.txt` next to each PDF (pages separated by form feeds). Ingestion prefers the sidecar,
so the OCR runs once and the text is reused by everyone, including machines without Tesseract.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from parali import config

log = logging.getLogger(__name__)


def available() -> bool:
    return bool(shutil.which("tesseract") and shutil.which("pdftoppm"))


def sidecar(pdf: Path) -> Path:
    return pdf.with_suffix(".ocr.txt")


def ocr_pdf(pdf: Path, lang: str = "eng", dpi: int = 300) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), f"{tmp}/p"], check=True)
        pages = []
        for img in sorted(Path(tmp).glob("p-*.png"), key=lambda p: int(p.stem.split("-")[-1])):
            out = subprocess.run(
                ["tesseract", str(img), "-", "-l", lang, "--psm", "3"],
                check=True,
                capture_output=True,
                text=True,
            )
            pages.append(out.stdout)
    return pages


def run(
    corpus_dir: Path = config.CORPUS_DIR, only_missing_text: bool = False, lang: str = "eng"
) -> list[Path]:
    """OCR every PDF (or only those without a usable text layer) and write sidecars."""
    from parali.advise.ingest import _pdf_pages, _quality

    if not available():
        raise RuntimeError("Install Tesseract OCR and poppler (pdftoppm) to run OCR.")
    written = []
    for pdf in sorted(corpus_dir.rglob("*.pdf")):
        if sidecar(pdf).exists():
            continue
        if only_missing_text:
            texts, _ = _pdf_pages(pdf, use_sidecar=False)
            if all(_quality(t) == "ok" for t in texts if t.strip()):
                continue
        pages = ocr_pdf(pdf, lang)
        sidecar(pdf).write_text("\f".join(pages), encoding="utf-8")
        written.append(sidecar(pdf))
        log.info("OCR %s: %d pages", pdf.name, len(pages))
    return written
