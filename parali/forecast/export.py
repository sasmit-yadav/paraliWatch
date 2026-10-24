"""Write the small, precomputed files that demo mode serves (committed to the repo)."""

from __future__ import annotations

import pandas as pd

from parali import config
from parali.forecast.train import explain
from parali.pipeline import build

DEMO_YEAR = config.TEST_YEARS[0]
FORECAST = config.DEMO_DIR / f"forecast_{DEMO_YEAR}.parquet"
FIRES = config.DEMO_DIR / f"fires_{DEMO_YEAR}.parquet"
GRID = config.DEMO_DIR / "grid.parquet"

# feature values shown next to each explanation in the dashboard
CONTEXT = ["fires_3d", "ring1_3d", "ring1_7d", "ring2_3d", "last_year_week", "rain_3d", "rh", "tmax"]


def run(booster, test: pd.DataFrame) -> None:
    reasons = explain(booster, test)
    out = pd.concat(
        [
            test[["date", "cell"]].reset_index(drop=True),
            test[["prob", "raw"]].round(4).reset_index(drop=True),
            test[CONTEXT].round(1).reset_index(drop=True),
            test[["fires_next3_count"]].reset_index(drop=True),
            reasons.reset_index(drop=True),
        ],
        axis=1,
    )
    for col in ("cell", "reason1", "reason2", "reason3"):
        out[col] = out[col].astype("category")
    for col in ("reason1_shap", "reason2_shap", "reason3_shap"):
        out[col] = out[col].round(3)
    out.to_parquet(FORECAST, index=False, compression="zstd")

    fires = build.load_fires([DEMO_YEAR])[["latitude", "longitude", "acq_date", "acq_time", "frp", "daynight", "cell"]]
    fires.to_parquet(FIRES, index=False, compression="zstd")
    build.step_grid()[["cell", "lat", "lon", "district", "state"]].to_parquet(GRID, index=False)
