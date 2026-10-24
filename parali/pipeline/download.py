"""Small, polite HTTP helpers: caching, retries and streaming to disk."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)
USER_AGENT = "parali-watch/0.1 (research project; github.com/sasmit-yadav/paraliWatch)"


def get(url: str, *, params: dict | None = None, retries: int = 5, timeout: int = 120) -> requests.Response:
    """GET with exponential backoff on network errors, 429 and 5xx."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            wait = (60 if status == 429 else 5) * 2**attempt
            log.warning("GET %s failed (%s); retrying in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} after {retries} attempts")


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Stream a file to disk once; later calls reuse the cached copy."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300, headers={"User-Agent": USER_AGENT}) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    log.info("Downloaded %s -> %s (%.1f MB)", url, dest, dest.stat().st_size / 1e6)
    return dest
