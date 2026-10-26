"""Turn fire points + weather into a cell x day table of features and targets.

Rule that keeps the evaluation honest: every feature for day t uses only information available
by the end of day t. The target looks at days t+1 .. t+HORIZON. Tests in tests/test_features.py
check this.

Internally each season is a (cells x days) matrix, so rolling and neighbour sums are fast array
operations instead of millions of groupbys.
"""

from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd
from scipy import sparse

from parali import config

FEATURES = [
    # the cell's own recent fires
    "fires_1d",
    "fires_3d",
    "fires_7d",
    "fires_14d",
    "fires_season",
    "frp_3d",
    "days_since_fire",
    # fires around the cell
    "ring1_1d",
    "ring1_3d",
    "ring1_7d",
    "ring2_3d",
    # wider activity
    "district_3d",
    "state_3d",
    "state_season_vs_last_year",
    # memory of past seasons
    "last_year_next3",
    "last_year_week",
    "last_year_season",
    "history_rate",
    # calendar
    "season_day",
    # weather observed up to day t
    "tmax",
    "rh",
    "rain",
    "rain_3d",
    "wind",
    "wind_dir_sin",
    "wind_dir_cos",
]
TARGET = "fire_next3"


@dataclass
class Season:
    year: int
    dates: pd.DatetimeIndex
    counts: np.ndarray  # cells x days, fire points
    frp: np.ndarray  # cells x days, summed fire radiative power (MW)


def adjacency(cells: list[str], k: int) -> sparse.csr_matrix:
    """Sparse matrix A where A[i, j] = 1 if cell j is exactly k rings from cell i."""
    index = {c: i for i, c in enumerate(cells)}
    rows, cols = [], []
    for i, c in enumerate(cells):
        for n in h3.grid_ring(c, k):
            j = index.get(n)
            if j is not None:
                rows.append(i)
                cols.append(j)
    n = len(cells)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def season_matrix(fires: pd.DataFrame, cells: list[str], year: int) -> Season:
    start, end = config.season_bounds(year)
    dates = pd.date_range(start, end, freq="D")
    index = {c: i for i, c in enumerate(cells)}
    f = fires[pd.to_datetime(fires["acq_date"]).dt.year == year]
    cell_pos = f["cell"].map(index)
    f, cell_pos = f[cell_pos.notna()], cell_pos[cell_pos.notna()]  # fires outside the grid are ignored
    d_idx = (pd.to_datetime(f["acq_date"]) - dates[0]).dt.days.to_numpy(dtype=np.int64)
    # explicit integer dtype: pandas 3 returns object/float arrays from .map()
    c_idx = cell_pos.to_numpy(dtype=np.int64)
    counts = np.zeros((len(cells), len(dates)), dtype=np.float32)
    frp = np.zeros_like(counts)
    np.add.at(counts, (c_idx, d_idx), 1)
    np.add.at(frp, (c_idx, d_idx), f["frp"].fillna(0).to_numpy(dtype=np.float32))
    return Season(year, dates, counts, frp)


def rolling(m: np.ndarray, window: int) -> np.ndarray:
    """Sum over the last `window` days including today (axis=1)."""
    c = np.cumsum(m, axis=1)
    out = c.copy()
    out[:, window:] = c[:, window:] - c[:, :-window]
    return out


def forward(m: np.ndarray, horizon: int) -> np.ndarray:
    """Sum over days t+1 .. t+horizon; NaN where that window runs past the season."""
    out = np.full(m.shape, np.nan, dtype=np.float32)
    c = np.concatenate([np.zeros((m.shape[0], 1), dtype=m.dtype), np.cumsum(m, axis=1)], axis=1)
    n = m.shape[1]
    for t in range(n - horizon):
        out[:, t] = c[:, t + 1 + horizon] - c[:, t + 1]
    return out


def days_since(m: np.ndarray, cap: int = 30) -> np.ndarray:
    out = np.full(m.shape, cap, dtype=np.float32)
    last = np.full(m.shape[0], -10_000)
    for t in range(m.shape[1]):
        last = np.where(m[:, t] > 0, t, last)
        out[:, t] = np.minimum(t - last, cap)
    return out


def group_rolling(m3: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Total of a (cells x days) matrix within each group, broadcast back to its cells."""
    codes, inv = np.unique(groups, return_inverse=True)
    totals = np.zeros((len(codes), m3.shape[1]), dtype=np.float32)
    np.add.at(totals, inv, m3)
    return totals[inv]


def build_season(
    season: Season,
    prev: Season | None,
    history: np.ndarray,
    grid: pd.DataFrame,
    weather: pd.DataFrame,
    a1: sparse.csr_matrix,
    a2: sparse.csr_matrix,
) -> pd.DataFrame:
    """Feature table for one season. `history` = mean seasonal fire count per cell over earlier
    seasons (all zeros when there are none)."""
    n_cells, n_days = season.counts.shape
    cnt = season.counts
    f3 = rolling(cnt, 3)
    cols: dict[str, np.ndarray] = {
        "fires_1d": cnt,
        "fires_3d": f3,
        "fires_7d": rolling(cnt, 7),
        "fires_14d": rolling(cnt, 14),
        "fires_season": np.cumsum(cnt, axis=1),
        "frp_3d": rolling(season.frp, 3),
        "days_since_fire": days_since(cnt),
        "ring1_1d": a1 @ cnt,
        "ring1_3d": a1 @ f3,
        "ring1_7d": a1 @ rolling(cnt, 7),
        "ring2_3d": a2 @ f3,
        "district_3d": group_rolling(f3, grid["district"].to_numpy()),
        "state_3d": group_rolling(f3, grid["state"].to_numpy()),
        "history_rate": np.repeat(history[:, None], n_days, axis=1),
        "season_day": np.tile(np.arange(n_days, dtype=np.float32), (n_cells, 1)),
    }

    state_cum = group_rolling(np.cumsum(cnt, axis=1), grid["state"].to_numpy())
    if prev is not None:
        pc = prev.counts[:, :n_days]
        cols["last_year_next3"] = np.nan_to_num(forward(pc, config.HORIZON_DAYS))
        cols["last_year_week"] = rolling(np.pad(pc, ((0, 0), (0, 3)))[:, 3:], 7)  # days t-3 .. t+3
        cols["last_year_season"] = np.repeat(pc.sum(axis=1)[:, None], n_days, axis=1)
        prev_state_cum = group_rolling(np.cumsum(pc, axis=1), grid["state"].to_numpy())
        cols["state_season_vs_last_year"] = (state_cum + 1) / (prev_state_cum + 1)
    else:
        for k in ("last_year_next3", "last_year_week", "last_year_season", "state_season_vs_last_year"):
            cols[k] = np.full((n_cells, n_days), np.nan, dtype=np.float32)

    # weather: one series per 0.5 degree point, reused by every cell snapped to it
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w[w["date"].dt.year == season.year]
    key = grid["wlat"].round(4).astype(str) + "," + grid["wlon"].round(4).astype(str)
    w["key"] = w["wlat"].round(4).astype(str) + "," + w["wlon"].round(4).astype(str)

    def wmat(col: str) -> np.ndarray:
        pivot = w.pivot_table(index="key", columns="date", values=col).reindex(columns=season.dates)
        return pivot.reindex(key.to_numpy()).to_numpy(dtype=np.float32)

    rain = np.nan_to_num(wmat("precipitation_sum"))
    wdir = np.deg2rad(wmat("wind_direction_10m_dominant"))
    cols.update(
        tmax=wmat("temperature_2m_max"),
        rh=wmat("relative_humidity_2m_mean"),
        rain=rain,
        rain_3d=rolling(rain, 3),
        wind=wmat("wind_speed_10m_max"),
        wind_dir_sin=np.sin(wdir),
        wind_dir_cos=np.cos(wdir),
    )

    target = forward(cnt, config.HORIZON_DAYS)
    fires_next3 = target
    # keep days that have a full target window and at least a week of history
    keep_days = np.arange(7, n_days - config.HORIZON_DAYS)
    frame = {
        "year": np.full(n_cells * len(keep_days), season.year, dtype=np.int16),
        "date": np.tile(season.dates[keep_days].to_numpy(), n_cells),
        "cell": np.repeat(grid["cell"].to_numpy(), len(keep_days)),
    }
    for name in FEATURES:
        frame[name] = np.asarray(cols[name], dtype=np.float32)[:, keep_days].ravel()
    frame["fires_next3_count"] = fires_next3[:, keep_days].ravel()
    frame[TARGET] = (frame["fires_next3_count"] > 0).astype(np.int8)
    return pd.DataFrame(frame)


def build_panel(
    fires: pd.DataFrame, weather: pd.DataFrame, grid: pd.DataFrame, years=None, only: int | None = None
) -> pd.DataFrame:
    """All seasons stacked. The first year in config.YEARS only seeds 'last year' features.
    only=<year>: build just that season (earlier years still feed its history features)."""
    from parali.pipeline.weather import weather_points

    years = years or config.YEARS
    grid = weather_points(grid).reset_index(drop=True)
    cells = grid["cell"].tolist()
    a1, a2 = adjacency(cells, 1), adjacency(cells, 2)
    seasons = {y: season_matrix(fires, cells, y) for y in years}

    frames = []
    for i, year in enumerate(years[1:], start=1):
        if only is not None and year != only:
            continue
        earlier = [seasons[y].counts.sum(axis=1) for y in years[:i]]
        history = np.mean(earlier, axis=0)
        frames.append(build_season(seasons[year], seasons[years[i - 1]], history, grid, weather, a1, a2))
    return pd.concat(frames, ignore_index=True)
