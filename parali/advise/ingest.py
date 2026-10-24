"""Read the document corpus into pages with their source, so every answer can cite a page."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from parali import config
from parali.advise.text import is_hindi, normalize

log = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}


@dataclass
class Page:
    doc_id: str
    title: str
    source: str  # URL or publisher, from sources.csv when given
    page: int
    text: str
    lang: str  # "hi" or "en"
    quality: str  # "ok" or "needs_ocr" (scanned or broken text layer)

    def to_dict(self) -> dict:
        return asdict(self)


def _sources(corpus_dir: Path) -> dict[str, dict]:
    """Optional data/corpus/sources.csv with columns: file,title,url,publisher,date."""
    path = corpus_dir / "sources.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {row["file"]: row for row in csv.DictReader(fh)}


def _quality(text: str) -> str:
    """Text layers from scanned or legacy-font PDFs come out empty or as symbol soup."""
    if len(text) < 40:
        return "needs_ocr"
    letters = sum(c.isalpha() for c in text)
    return "ok" if letters / max(1, len(text)) > 0.5 else "needs_ocr"


def _pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [normalize(p.extract_text() or "") for p in reader.pages]


def _html_pages(path: Path) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return [normalize(soup.get_text("\n"))]


def load_corpus(corpus_dir: Path = config.CORPUS_DIR) -> list[Page]:
    meta = _sources(corpus_dir)
    pages: list[Page] = []
    for path in sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in SUPPORTED):
        rel = path.relative_to(corpus_dir).as_posix()
        info = meta.get(rel) or meta.get(path.name) or {}
        title = info.get("title") or path.stem.replace("_", " ").replace("-", " ")
        source = info.get("url") or info.get("publisher") or rel
        try:
            if path.suffix.lower() == ".pdf":
                texts = _pdf_pages(path)
            elif path.suffix.lower() in {".html", ".htm"}:
                texts = _html_pages(path)
            else:
                texts = [normalize(path.read_text(encoding="utf-8", errors="replace"))]
        except Exception as exc:  # a broken file should not stop the whole corpus
            log.warning("Skipping %s: %s", rel, exc)
            continue
        for i, text in enumerate(texts, start=1):
            pages.append(
                Page(
                    doc_id=rel,
                    title=title,
                    source=source,
                    page=i,
                    text=text,
                    lang="hi" if is_hindi(text) else "en",
                    quality=_quality(text),
                )
            )
    bad = sum(p.quality != "ok" for p in pages)
    log.info("Corpus: %d files, %d pages (%d need OCR)", len({p.doc_id for p in pages}), len(pages), bad)
    return pages
