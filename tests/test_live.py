"""Live mode: request windows, season choice, and serving a live season next to the replay."""

import json
import shutil
from datetime import date

import pandas as pd

from parali import config, live


def test_firms_requests_cover_the_season_in_5_day_steps():
    w = live.windows(date(2026, 9, 15), date(2026, 10, 25))
    assert w[0] == (date(2026, 9, 15), 5) and w[-1] == (date(2026, 10, 25), 1)
    assert sum(days for _, days in w) == 41 and all(days <= 5 for _, days in w)


def test_current_season_switches_on_the_season_start():
    assert live.current_season(date(2026, 10, 26)) == 2026
    assert live.current_season(date(2026, 9, 14)) == 2025
    assert live.current_season(date(2027, 3, 1)) == 2026


def test_live_season_is_served_first_and_replay_stays_available(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from parali.app.main import app

    # a fake "live" season: the replay files relabelled, with unknown recent outcomes
    monkeypatch.setattr(config, "LIVE_DIR", tmp_path)
    f = pd.read_parquet(config.DEMO_DIR / "forecast_2025.parquet")
    last = f["date"].max()
    f.loc[f["date"] == last, "fires_next3_count"] = float("nan")
    f.to_parquet(tmp_path / "forecast_2099.parquet")
    for name in ("fires_2025.parquet", "grid.parquet", "districts.geojson"):
        shutil.copy(config.DEMO_DIR / name, tmp_path / name.replace("2025", "2099"))
    (tmp_path / "meta.json").write_text(json.dumps({"season": 2099, "built_at": "2099-10-26T00:00:00+00:00"}))

    client = TestClient(app)
    m = client.get("/api/meta").json()
    assert m["mode"] == "live" and m["season"] == 2099
    assert m["default_date"] == str(pd.Timestamp(last).date())  # live opens on the latest day
    assert [s["mode"] for s in m["seasons"]] == ["live", "replay"]
    assert client.get("/api/meta?season=2025").json()["mode"] == "replay"
    top = client.get(f"/api/top?date={m['default_date']}&n=1&season=2099").json()
    cell = client.get(f"/api/cell/{top[0]['cell']}?date={m['default_date']}&season=2099").json()
    assert cell["fires_next3_actual"] is None  # outcome not known yet
    assert client.get("/api/meta?season=1999").status_code == 404
