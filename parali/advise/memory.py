"""Long-term memory: a farmer's profile (district, crop, land size) reused across questions."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass

from parali import config

DB = config.DATA_DIR / "assistant.db"


@dataclass
class Profile:
    user_id: str
    district: str | None = None
    crop: str | None = None
    land_acres: float | None = None
    language: str | None = None

    def as_context(self) -> str:
        parts = []
        if self.district:
            parts.append(f"district: {self.district}")
        if self.crop:
            parts.append(f"crop: {self.crop}")
        if self.land_acres:
            parts.append(f"land: {self.land_acres:g} acres")
        return "; ".join(parts)


def _conn(path=None) -> sqlite3.Connection:
    path = path or DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, district TEXT, crop TEXT, "
        "land_acres REAL, language TEXT)"
    )
    return conn


def get(user_id: str, path=None) -> Profile:
    with _conn(path) as conn:
        row = conn.execute(
            "SELECT district, crop, land_acres, language FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    return Profile(user_id, *row) if row else Profile(user_id)


def save(profile: Profile, path=None) -> Profile:
    d = asdict(profile)
    with _conn(path) as conn:
        conn.execute(
            "INSERT INTO profiles VALUES (:user_id, :district, :crop, :land_acres, :language) "
            "ON CONFLICT(user_id) DO UPDATE SET district=excluded.district, crop=excluded.crop, "
            "land_acres=excluded.land_acres, language=excluded.language",
            d,
        )
    return profile
