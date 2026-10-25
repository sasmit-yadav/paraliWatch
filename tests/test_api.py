"""API tests against the committed demo data (no downloads needed)."""

import pytest
from fastapi.testclient import TestClient

from parali.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def meta():
    return client.get("/api/meta").json()


def test_meta_lists_season_dates(meta):
    assert meta["season"] == 2025
    assert meta["default_date"] in meta["dates"]
    assert meta["metrics"]["test"]["pr_auc"] > meta["metrics"]["best_rule"]["pr_auc"]


def test_risk_returns_hexagons_with_probabilities(meta):
    data = client.get(f"/api/risk?date={meta['default_date']}&min_prob=0.05").json()
    assert data["features"]
    f = data["features"][0]
    assert f["geometry"]["type"] == "Polygon"
    assert 0.05 <= f["properties"]["prob"] <= 1


def test_top_cells_are_sorted_and_explained(meta):
    top = client.get(f"/api/top?date={meta['default_date']}&n=5").json()
    assert len(top) == 5
    detail = client.get(f"/api/cell/{top[0]['cell']}?date={meta['default_date']}").json()
    assert len(detail["reasons"]) == 3
    assert all(r["text"] for r in detail["reasons"])
    assert len(detail["history"]) == 14


def test_bad_inputs_are_rejected(meta):
    assert client.get("/api/risk?date=2020-01-01").status_code == 404
    assert client.get("/api/risk?date=not-a-date").status_code == 400
    assert client.get(f"/api/cell/abc?date={meta['default_date']}").status_code == 404


def test_dashboard_is_served():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_bulletin_endpoint(meta, tmp_path, monkeypatch):
    from parali import config

    monkeypatch.setattr(config, "BULLETIN_DIR", tmp_path)
    b = client.get(f"/api/bulletin?date={meta['default_date']}").json()
    assert b["status"] == "draft" and b["districts"]
    approved = client.post(
        f"/api/bulletin/approve?date={meta['default_date']}", json={"official": "Test Officer"}
    ).json()
    assert approved["status"] == "approved"


def test_profile_roundtrip(tmp_path, monkeypatch):
    from parali.advise import memory

    monkeypatch.setattr(memory, "DB", tmp_path / "a.db")
    client.put("/api/profile/farmer1", json={"district": "Karnal", "land_acres": 3, "language": "hi"})
    assert client.get("/api/profile/farmer1").json()["district"] == "Karnal"


def test_season_timeline_covers_every_forecast_day(meta):
    rows = client.get("/api/season").json()
    assert [r["date"] for r in rows] == meta["dates"]
    assert all(r["fires"] >= 0 and r["expected_cells"] >= 0 for r in rows)
    assert sum(r["fires"] for r in rows) > 0
