"""Read the document corpus into pages with their source, so every answer can cite a page."""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
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


NUMERIC_LINE = re.compile(r"^[\d\s.,:%/-]*$|^\d{1,2}-[A-Z][a-z]{2}$")  # "120", "15-Sep", "0 0 0"


def _quality(text: str) -> str:
    """ "ok", "needs_ocr" (scanned or symbol soup) or "chart" (mostly axis labels and numbers from
    a chart: noise for search, so skipped)."""
    if len(text) < 40:
        return "needs_ocr"
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) <= 0.5:
        return "needs_ocr"
    words = [ln for ln in text.split("\n") if ln.strip() and not NUMERIC_LINE.match(ln.strip())]
    if sum(len(ln.split()) for ln in words) < 25:
        return "chart"
    return "ok"


def _drop_numeric_lines(text: str) -> str:
    """Lines that are only numbers or dates are chart axes and labels: noise for search."""
    return "\n".join(ln for ln in text.split("\n") if not NUMERIC_LINE.match(ln.strip()) or not ln.strip())


URL = re.compile(r"https?://\S+")
BOILERPLATE = re.compile(
    r"^(Visitor Counter|Read this release in|Share \d+ words|\(Release ID: \d+\) Visitor"
    r"|Page\s*\w{1,3}?\s*o\w?f\s*\d+$|\d+\s*\|\s*Pag\w*:?$)"  # page markers, also OCR-garbled ones
)
PRINT_STAMP = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},? \d{1,2}:\d{2}( ?[AP]M)?\b")  # browser print header


def _strip_repeated_lines(pages: list[str]) -> tuple[list[str], str | None]:
    """Remove headers/footers: lines that repeat on most pages of a document, browser print
    stamps ("10/24/26, 11:55 AM ...") and page-numbered URL footers. Returns the cleaned pages and
    the URL found in the footer, which becomes the document's citation link."""
    footer_url = None
    lines_per_page = [[ln.strip() for ln in p.split("\n")] for p in pages]
    if len(pages) >= 2:
        counts = Counter(ln for lines in lines_per_page for ln in set(lines) if ln)
        repeated = {ln for ln, n in counts.items() if n >= max(2, 0.6 * len(pages))}
    else:
        repeated = set()
    cleaned = []
    for lines in lines_per_page:
        keep = []
        for ln in lines:
            url = URL.search(ln)
            is_footer_url = url and re.search(r"\s\d+/\d+$", ln)
            if is_footer_url and footer_url is None:
                footer_url = url.group(0)
            if ln in repeated or PRINT_STAMP.match(ln) or BOILERPLATE.match(ln) or is_footer_url:
                continue
            keep.append(ln)
        cleaned.append(normalize("\n".join(keep)))
    return cleaned, footer_url


def _pdf_pages(path: Path, use_sidecar: bool = True) -> tuple[list[str], str | None]:
    """Page texts of a PDF. An OCR sidecar (<name>.ocr.txt, pages split by form feeds; see
    parali/advise/ocr.py) replaces the PDF's own text layer when present."""
    ocr = path.with_suffix(".ocr.txt")
    if use_sidecar and ocr.exists():
        raw = ocr.read_text(encoding="utf-8").split("\f")
    else:
        from pypdf import PdfReader

        raw = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    return _strip_repeated_lines([normalize(t) for t in raw])


def _html_pages(path: Path) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return [normalize(soup.get_text("\n"))]


SNAPSHOT = "pages.jsonl"


def save_snapshot(pages: list[Page], corpus_dir: Path = config.CORPUS_DIR) -> Path:
    """Write the extracted page texts. The repository ships this file instead of the PDFs, so the
    assistant works after a fresh clone; the originals are listed in sources.csv."""
    import json

    path = corpus_dir / SNAPSHOT
    with open(path, "w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_corpus(corpus_dir: Path = config.CORPUS_DIR) -> list[Page]:
    """Parse the documents in corpus_dir; with no documents, fall back to the shipped snapshot."""
    import json

    has_docs = any(
        p.suffix.lower() in SUPPORTED and not p.name.endswith(".ocr.txt") for p in corpus_dir.rglob("*")
    )
    snapshot = corpus_dir / SNAPSHOT
    if not has_docs and snapshot.exists():
        with open(snapshot, encoding="utf-8") as fh:
            return [Page(**json.loads(line)) for line in fh if line.strip()]
    meta = _sources(corpus_dir)
    pages: list[Page] = []
    files = [
        p for p in corpus_dir.rglob("*") if p.suffix.lower() in SUPPORTED and not p.name.endswith(".ocr.txt")
    ]
    for path in sorted(files):
        rel = path.relative_to(corpus_dir).as_posix()
        info = meta.get(rel) or meta.get(path.name) or {}
        title = info.get("title") or path.stem.replace("_", " ").replace("-", " ")
        source = info.get("url") or None
        footer_url = None
        try:
            if path.suffix.lower() == ".pdf":
                texts, footer_url = _pdf_pages(path)
            elif path.suffix.lower() in {".html", ".htm"}:
                texts = _html_pages(path)
            else:
                texts = [normalize(path.read_text(encoding="utf-8", errors="replace"))]
        except Exception as exc:  # a broken file should not stop the whole corpus
            log.warning("Skipping %s: %s", rel, exc)
            continue
        source = source or footer_url or info.get("publisher") or rel
        for i, text in enumerate(texts, start=1):
            if not text.strip():
                continue  # blank trailing pages from browser printing
            pages.append(
                Page(
                    doc_id=rel,
                    title=title,
                    source=source,
                    page=i,
                    text=_drop_numeric_lines(text),
                    lang="hi" if is_hindi(text) else "en",
                    quality=_quality(text),
                )
            )
    bad = sum(p.quality != "ok" for p in pages)
    log.info("Corpus: %d files, %d pages (%d need OCR)", len({p.doc_id for p in pages}), len(pages), bad)
    return pages
