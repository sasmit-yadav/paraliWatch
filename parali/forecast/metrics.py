"""Metrics that match how the forecast is used: officials can visit only a few places a day,
so what matters is how many real fires land in the top-ranked cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_share_by_day(df: pd.DataFrame, score: str, top: float = 0.10) -> tuple[float, float]:
    """For each day, take the top `top` fraction of cells by score.

    Returns (precision, fire_capture):
      precision    = share of those cells that really had a fire in the next 3 days
      fire_capture = share of all next-3-day fire points that fell inside those cells
    """
    hits = cells = fires_in = fires_all = 0.0
    for _, day in df.groupby("date", sort=False):
        k = max(1, int(round(len(day) * top)))
        # stable sort + random tie-break so equal scores are not ranked in file order
        rng = np.random.default_rng(0)
        order = np.lexsort((rng.random(len(day)), -day[score].to_numpy()))[:k]
        chosen = day.iloc[order]
        hits += chosen["fire_next3"].sum()
        cells += k
        fires_in += chosen["fires_next3_count"].sum()
        fires_all += day["fires_next3_count"].sum()
    return hits / cells, (fires_in / fires_all if fires_all else float("nan"))


def evaluate(df: pd.DataFrame, score: str, *, probabilistic: bool = False) -> dict:
    y = df["fire_next3"].to_numpy()
    s = df[score].to_numpy()
    precision10, capture10 = top_share_by_day(df, score, 0.10)
    out = {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "precision_top10pct": float(precision10),
        "fire_capture_top10pct": float(capture10),
        "base_rate": float(y.mean()),
    }
    if probabilistic:
        out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
    return out


def reliability(df: pd.DataFrame, score: str, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed fire rate in equal-width probability bins."""
    b = pd.cut(df[score], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (
        df.groupby(b, observed=True)
        .agg(predicted=(score, "mean"), observed=("fire_next3", "mean"), n=("fire_next3", "size"))
        .reset_index(drop=True)
    )
