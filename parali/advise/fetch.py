"""Add an official web page to the corpus from its link (for example a PIB press release).

    python -m parali add-url "https://www.pib.gov.in/PressReleasePage.aspx?PRID=..." --title "..."

The page is saved as HTML in data/corpus/ and recorded in sources.csv with its link, so answers
cite the exact page. Respect each site's terms: only add pages you are allowed to reuse.
"""

from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from parali import config
from parali.pipeline.download import get


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80] or "page"


def add_url(
    url: str, title: str | None = None, publisher: str | None = None, corpus_dir: Path = config.CORPUS_DIR
) -> Path:
    from bs4 import BeautifulSoup

    resp = get(url)
    soup = BeautifulSoup(resp.content, "html.parser")
    title = title or (soup.title.get_text(strip=True) if soup.title else urlparse(url).netloc)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    host = urlparse(url).netloc.removeprefix("www.").split(".")[0] or "web"
    path = corpus_dir / f"{host}_{_slug(title)}.html"
    path.write_bytes(resp.content)
    sources = corpus_dir / "sources.csv"
    new = not sources.exists()
    with open(sources, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["file", "title", "url", "publisher", "date"])
        w.writerow([path.name, title, url, publisher or urlparse(url).netloc, date.today().isoformat()])
    return path
