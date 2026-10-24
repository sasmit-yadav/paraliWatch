"""Loads the demo files once and answers the API's questions from memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from parali import config
from parali.app.explain import describe


@dataclass
class Store:
    demo_dir: Path = config.DEMO_DIR
    year: int = config.TEST_YEARS[0]

    @cached_property
    def grid(self) -> pd.DataFrame:
        return pd.read_parquet(self.demo_dir / "grid.parquet").set_index("cell")

    @cached_property
    def forecast(self) -> pd.DataFrame:
        df = pd.read_parquet(self.demo_dir / f"forecast_{self.year}.parquet")
        df["cell"] = df["cell"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # percentile rank within each day: "top 5% of cells today"
        df["pct_rank"] = df.groupby("date")["raw"].rank(pct=True, method="max").astype(np.float32)
        return df

    @cached_property
    def fires(self) -> pd.DataFrame:
        df = pd.read_parquet(self.demo_dir / f"fires_{self.year}.parquet")
        df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
        return df

    @cached_property
    def districts(self) -> dict:
        return json.loads((self.demo_dir / "districts.geojson").read_text(encoding="utf-8"))

    @cached_property
    def boundaries(self) -> dict[str, list[list[float]]]:
        out = {}
        for cell in self.grid.index:
            ring = [[round(lon, 5), round(lat, 5)] for lat, lon in h3.cell_to_boundary(cell)]
            out[cell] = ring + [ring[0]]
        return out

    @cached_property
    def dates(self) -> list[date]:
        return sorted(self.forecast["date"].unique())

    def default_date(self) -> date:
        """The day with the most fires in the following 3 days: the most interesting demo day."""
        totals = self.forecast.groupby("date")["fires_next3_count"].sum()
        return totals.idxmax()

    def day(self, d: date) -> pd.DataFrame:
        return self.forecast[self.forecast["date"] == d]

    def risk_geojson(self, d: date, min_prob: float = 0.0) -> dict:
        day = self.day(d)
        day = day[day["prob"] >= min_prob]
        features = []
        for row in day.itertuples(index=False):
            g = self.grid.loc[row.cell]
            features.append(
                {
                    "type": "Feature",
                    "id": row.cell,
                    "properties": {
                        "cell": row.cell,
                        "prob": float(row.prob),
                        "pct_rank": float(row.pct_rank),
                        "district": g["district"],
                        "state": g["state"],
                    },
                    "geometry": {"type": "Polygon", "coordinates": [self.boundaries[row.cell]]},
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def fires_geojson(self, d: date, days: int = 1) -> dict:
        start = d - timedelta(days=days - 1)
        f = self.fires[(self.fires["acq_date"] >= start) & (self.fires["acq_date"] <= d)]
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "date": r.acq_date.isoformat(),
                        "time": str(r.acq_time).zfill(4),
                        "frp": float(r.frp),
                        "daynight": r.daynight,
                    },
                    "geometry": {"type": "Point", "coordinates": [float(r.longitude), float(r.latitude)]},
                }
                for r in f.itertuples(index=False)
            ],
        }

    def top(self, d: date, n: int = 10) -> list[dict]:
        day = self.day(d).nlargest(n, "raw")
        return [self._summary(r) for r in day.itertuples(index=False)]

    def _summary(self, r) -> dict:
        g = self.grid.loc[r.cell]
        return {
            "cell": r.cell,
            "district": g["district"],
            "state": g["state"],
            "lat": round(float(g["lat"]), 4),
            "lon": round(float(g["lon"]), 4),
            "prob": float(r.prob),
            "pct_rank": float(r.pct_rank),
        }

    def cell(self, cell: str, d: date) -> dict | None:
        day = self.day(d)
        rows = day[day["cell"] == cell]
        if rows.empty:
            return None
        r = rows.iloc[0]
        out = self._summary(next(rows.itertuples(index=False)))
        reasons = []
        for k in (1, 2, 3):
            feat = str(r[f"reason{k}"])
            value = r.get(f"reason{k}_value")
            value = None if value is None or pd.isna(value) else float(value)
            reasons.append(
                {
                    "feature": feat,
                    "value": value,
                    "shap": float(r[f"reason{k}_shap"]),
                    "text": describe(feat, value),
                }
            )
        history_start = d - timedelta(days=13)
        hist = self.fires[
            (self.fires["cell"] == cell)
            & (self.fires["acq_date"] >= history_start)
            & (self.fires["acq_date"] <= d)
        ]
        counts = hist.groupby("acq_date").size()
        out.update(
            date=d.isoformat(),
            reasons=reasons,
            fires_next3_actual=int(r["fires_next3_count"]),
            history=[
                {
                    "date": (history_start + timedelta(days=i)).isoformat(),
                    "fires": int(counts.get(history_start + timedelta(days=i), 0)),
                }
                for i in range(14)
            ],
        )
        return out

    def district_summary(self, d: date) -> list[dict]:
        day = self.day(d).merge(self.grid[["district", "state"]], left_on="cell", right_index=True)
        g = (
            day.groupby(["district", "state"])
            .agg(
                expected_fire_cells=("prob", "sum"),
                high_risk_cells=("pct_rank", lambda s: int((s >= 0.9).sum())),
                cells=("cell", "size"),
            )
            .reset_index()
            .sort_values("expected_fire_cells", ascending=False)
        )
        g["expected_fire_cells"] = g["expected_fire_cells"].round(1)
        return g.to_dict(orient="records")
