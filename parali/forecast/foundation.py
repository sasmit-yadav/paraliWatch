"""Zero-shot comparison with a time-series foundation model (Amazon Chronos-Bolt).

Chronos-Bolt was pre-trained on many public time series and is used here WITHOUT any training on
our data. For each cell and day it sees that cell's daily fire counts (the previous season followed
by the current season up to that day) and forecasts the next 3 days. The sum of the forecast is the
cell's risk score, scored exactly like the LightGBM model.

Optional dependencies (heavy, not needed for the app): pip install torch chronos-forecasting
Run: python -m parali.forecast.foundation [tiny|mini|small]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from parali import config
from parali.forecast.compare import _score
from parali.forecast.features import season_matrix
from parali.forecast.train import split
from parali.pipeline import build

BATCH = 4096


def season_scores(pipeline, counts_prev: np.ndarray, counts: np.ndarray, days: np.ndarray):
    """Two risk scores for every (cell, day): the mean forecast and the 90th-percentile forecast,
    each summed over the next 3 days. Shape (cells, days)."""
    import torch

    n_cells, _ = counts.shape
    mean = np.zeros((n_cells, len(days)), dtype=np.float32)
    upper = np.zeros_like(mean)
    for j, t in enumerate(days):
        context = np.concatenate([counts_prev, counts[:, : t + 1]], axis=1)  # up to and including day t
        for i in range(0, n_cells, BATCH):
            ctx = torch.tensor(context[i : i + BATCH], dtype=torch.float32)
            q = pipeline.predict(
                ctx, prediction_length=config.HORIZON_DAYS
            ).numpy()  # (batch, quantiles, horizon)
            mean[i : i + BATCH, j] = q.mean(axis=1).sum(axis=1)
            upper[i : i + BATCH, j] = q[:, -1, :].sum(axis=1)
    return {"mean": mean, "upper": upper}


def run(size: str = "tiny") -> None:
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(f"amazon/chronos-bolt-{size}", device_map="cpu")
    panel = pd.read_parquet(config.PROCESSED_DIR / "panel.parquet")
    _, valid, test = split(panel)
    grid = build.step_grid()
    cells = grid["cell"].tolist()
    years = [y for v in config.VALID_YEARS + config.TEST_YEARS for y in (v - 1, v)]
    fires = build.load_fires(sorted(set(years)))
    n_days = len(pd.date_range(*config.season_bounds(2024)))
    days = np.arange(7, n_days - config.HORIZON_DAYS)

    t0 = time.time()
    scores = {"mean": {}, "upper": {}}
    for year in config.VALID_YEARS + config.TEST_YEARS:
        cur = season_matrix(fires, cells, year).counts
        prev = season_matrix(fires, cells, year - 1).counts
        dates = pd.date_range(*config.season_bounds(year))[days]
        for mode, s in season_scores(pipeline, prev, cur, days).items():
            scores[mode][year] = pd.DataFrame(
                {"cell": np.repeat(cells, len(days)), "date": np.tile(dates, len(cells)), "s": s.ravel()}
            )
    seconds = time.time() - t0

    def align(df: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
        keys = df[["cell", "date"]].assign(date=pd.to_datetime(df["date"]))
        return keys.merge(table, on=["cell", "date"], how="left")["s"].to_numpy()

    results = {}
    for mode, what in (("mean", "mean forecast"), ("upper", "90th-percentile forecast")):
        name = f"chronos_bolt_{size}_{mode}"
        label = f"Chronos-Bolt {size}, zero-shot ({what})"
        results[name] = _score(
            name,
            label,
            align(valid, scores[mode][config.VALID_YEARS[0]]),
            align(test, scores[mode][config.TEST_YEARS[0]]),
            valid,
            test,
            seconds / 2,
        )
    print(json.dumps({k: (v["valid"]["pr_auc"], v["test"]["pr_auc"]) for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "tiny")
