"""Three ways to cut pages into chunks. They are compared on the evaluation set (docs/RAG.md).

Chunks never cross a page boundary, so a citation always points to one page.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np

from parali.advise.ingest import Page
from parali.advise.text import sentences

HEADING = re.compile(r"^(\d+(\.\d+)*[.)]?\s+\S|[A-Z][A-Z\s&/-]{4,}$|[IVX]+\.\s|\(?[a-z]\)\s)")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    page: int
    lang: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        return f"{self.title}, p. {self.page}"


def _make(page: Page, n: int, text: str, strategy: str) -> Chunk:
    return Chunk(
        chunk_id=f"{page.doc_id}#p{page.page}-{strategy}-{n}",
        doc_id=page.doc_id,
        title=page.title,
        source=page.source,
        page=page.page,
        lang=page.lang,
        text=text.strip(),
    )


def fixed(pages: Iterable[Page], size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Every `size` words, overlapping by `overlap`. Simple baseline; ignores document structure."""
    out = []
    for page in pages:
        words = page.text.split()
        step = max(1, size - overlap)
        for n, start in enumerate(range(0, max(1, len(words) - overlap), step)):
            text = " ".join(words[start : start + size])
            if text:
                out.append(_make(page, n, text, "fixed"))
    return out


def structure(pages: Iterable[Page], max_words: int = 220) -> list[Chunk]:
    """Split at blank lines and heading-like lines, then pack paragraphs up to `max_words`.

    Government notices are lists of numbered clauses; keeping a clause whole keeps its condition
    and its amount together.
    """
    out = []
    for page in pages:
        blocks: list[str] = []
        current: list[str] = []
        for line in page.text.split("\n"):
            line = line.strip()
            if not line:
                if current:
                    blocks.append(" ".join(current))
                    current = []
                continue
            if HEADING.match(line) and current:
                blocks.append(" ".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append(" ".join(current))

        n, buf = 0, []
        for block in blocks:
            words = block.split()
            if buf and len(buf) + len(words) > max_words:
                out.append(_make(page, n, " ".join(buf), "structure"))
                n, buf = n + 1, []
            while len(words) > max_words:  # a single huge block: cut it
                out.append(_make(page, n, " ".join(words[:max_words]), "structure"))
                n, words = n + 1, words[max_words:]
            buf.extend(words)
        if buf:
            out.append(_make(page, n, " ".join(buf), "structure"))
    return out


def semantic(
    pages: Iterable[Page],
    embed: Callable[[list[str]], np.ndarray],
    max_words: int = 220,
    breakpoint_percentile: float = 20,
) -> list[Chunk]:
    """Start a new chunk where consecutive sentences are least similar in meaning."""
    out = []
    for page in pages:
        sents = sentences(page.text)
        if not sents:
            continue
        if len(sents) == 1:
            out.append(_make(page, 0, sents[0], "semantic"))
            continue
        vecs = embed(sents)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-9)
        sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
        cut = np.percentile(sims, breakpoint_percentile)
        n, buf = 0, [sents[0]]
        for sent, sim in zip(sents[1:], sims, strict=True):
            words = sum(len(s.split()) for s in buf)
            if sim < cut or words + len(sent.split()) > max_words:
                out.append(_make(page, n, " ".join(buf), "semantic"))
                n, buf = n + 1, []
            buf.append(sent)
        out.append(_make(page, n, " ".join(buf), "semantic"))
    return out


STRATEGIES = ("fixed", "structure", "semantic")
