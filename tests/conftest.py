import h3
import pandas as pd
import pytest

from parali import config


@pytest.fixture
def small_grid() -> pd.DataFrame:
    """19 cells: a centre cell in Ludhiana district plus two rings around it."""
    centre = h3.latlng_to_cell(30.9, 75.85, config.H3_RES)
    cells = sorted(h3.grid_disk(centre, 2))
    rows = []
    for c in cells:
        lat, lon = h3.cell_to_latlng(c)
        rows.append({"cell": c, "lat": lat, "lon": lon, "district": "Ludhiana", "state": "Punjab"})
    return pd.DataFrame(rows)


@pytest.fixture
def flat_weather(small_grid) -> pd.DataFrame:
    from parali.pipeline.weather import weather_points

    pts = weather_points(small_grid)[["wlat", "wlon"]].drop_duplicates()
    frames = []
    for year in (2023, 2024):
        start, end = config.season_bounds(year)
        for d in pd.date_range(start, end):
            for p in pts.itertuples(index=False):
                frames.append(
                    {
                        "date": d.date(),
                        "wlat": p.wlat,
                        "wlon": p.wlon,
                        "temperature_2m_max": 30.0,
                        "relative_humidity_2m_mean": 60.0,
                        "precipitation_sum": 0.0,
                        "wind_speed_10m_max": 10.0,
                        "wind_direction_10m_dominant": 270.0,
                    }
                )
    return pd.DataFrame(frames)
