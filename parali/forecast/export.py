"""Write the small, precomputed files that demo mode serves (committed to the repo)."""

from __future__ import annotations

import numpy as np
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
    # the value of each reason's feature, so the dashboard can say "12 fires nearby", not just "ring1_3d"
    values = test.reset_index(drop=True)
    for k in (1, 2, 3):
        names = reasons[f"reason{k}"].to_numpy()
        reasons[f"reason{k}_value"] = np.array(
            [values.at[i, n] for i, n in enumerate(names)], dtype=np.float32
        ).round(2)
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

    fires = build.load_fires([DEMO_YEAR])[
        ["latitude", "longitude", "acq_date", "acq_time", "frp", "daynight", "cell"]
    ]
    fires.to_parquet(FIRES, index=False, compression="zstd")
    build.step_grid()[["cell", "lat", "lon", "district", "state"]].to_parquet(GRID, index=False)


def from_saved_model() -> None:
    """Re-export demo files from the saved model without retraining."""
    import joblib
    import lightgbm as lgb

    from parali.forecast.features import FEATURES
    from parali.forecast.train import CALIBRATOR_PATH, MODEL_PATH

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    calibrator = joblib.load(CALIBRATOR_PATH)
    panel = pd.read_parquet(config.PROCESSED_DIR / "panel.parquet")
    test = panel[panel["year"] == DEMO_YEAR].copy()
    test["raw"] = booster.predict(test[FEATURES])
    test["prob"] = calibrator.predict(test["raw"])
    run(booster, test)
