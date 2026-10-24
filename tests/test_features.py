from datetime import date

import numpy as np
import pandas as pd

from parali.forecast.features import FEATURES, TARGET, adjacency, build_panel, forward, rolling


def test_rolling_sums_only_past_and_today():
    m = np.array([[1, 0, 2, 0, 3]], dtype=np.float32)
    assert rolling(m, 3).tolist() == [[1, 1, 3, 2, 5]]


def test_forward_sums_only_future_and_marks_incomplete_windows():
    m = np.array([[1, 0, 2, 0, 3, 0]], dtype=np.float32)
    out = forward(m, 3)
    assert out[0, 0] == 2  # days 1..3 = 0 + 2 + 0
    assert out[0, 1] == 5  # days 2..4 = 2 + 0 + 3
    assert np.isnan(out[0, 3])  # window runs past the end


def test_adjacency_is_symmetric_and_has_no_self_loops(small_grid):
    a = adjacency(small_grid["cell"].tolist(), 1).toarray()
    assert (a == a.T).all()
    assert np.diag(a).sum() == 0
    assert a.sum(axis=1).max() == 6  # an inner hexagon has exactly 6 neighbours


def _fires(cell: str, day: date, n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {"acq_date": [day] * n, "cell": [cell] * n, "frp": [5.0] * n, "latitude": 0.0, "longitude": 0.0}
    )


def test_no_future_information_in_features(small_grid, flat_weather):
    """A fire on 10 Oct must show up in the target of 7-9 Oct but in no feature before 10 Oct."""
    cell = small_grid["cell"].iloc[0]
    fires = _fires(cell, date(2024, 10, 10), n=4)
    panel = build_panel(fires, flat_weather, small_grid, years=[2023, 2024])
    rows = panel[(panel["cell"] == cell)].set_index("date")
    rows.index = pd.to_datetime(rows.index).date

    for d in (date(2024, 10, 7), date(2024, 10, 8), date(2024, 10, 9)):
        assert rows.loc[d, TARGET] == 1
        assert rows.loc[d, "fires_next3_count"] == 4
    assert rows.loc[date(2024, 10, 6), TARGET] == 0
    before = rows.loc[[d for d in rows.index if d < date(2024, 10, 10)], FEATURES]
    own = ["fires_1d", "fires_3d", "fires_7d", "fires_14d", "fires_season", "frp_3d"]
    assert (before[own] == 0).all().all()
    assert rows.loc[date(2024, 10, 10), "fires_1d"] == 4


def test_neighbour_feature_counts_fires_next_door(small_grid, flat_weather):
    import h3

    centre = small_grid["cell"].iloc[9]
    neighbour = next(c for c in h3.grid_ring(centre, 1) if c in set(small_grid["cell"]))
    fires = _fires(neighbour, date(2024, 10, 10), n=2)
    panel = build_panel(fires, flat_weather, small_grid, years=[2023, 2024])
    row = panel[(panel["cell"] == centre) & (pd.to_datetime(panel["date"]).dt.date == date(2024, 10, 10))]
    assert row["ring1_1d"].iloc[0] == 2
    assert row["fires_1d"].iloc[0] == 0


def test_last_year_features_come_from_previous_season(small_grid, flat_weather):
    cell = small_grid["cell"].iloc[0]
    fires = _fires(cell, date(2023, 10, 12), n=3)
    panel = build_panel(fires, flat_weather, small_grid, years=[2023, 2024])
    row = panel[(panel["cell"] == cell) & (pd.to_datetime(panel["date"]).dt.date == date(2024, 10, 10))]
    assert row["last_year_next3"].iloc[0] == 3  # 11-13 Oct last year
    assert row["last_year_season"].iloc[0] == 3
    assert row["history_rate"].iloc[0] == 3
