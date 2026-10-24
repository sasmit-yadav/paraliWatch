from datetime import date

import h3
import pandas as pd

from parali import config
from parali.pipeline.firms import clean


def test_clean_keeps_only_confident_vegetation_fires_in_area_and_season():
    lat, lon = 30.9, 75.85
    cell = h3.latlng_to_cell(lat, lon, config.H3_RES)
    raw = pd.DataFrame(
        {
            "latitude": [lat, lat, lat, lat, 10.0, lat],
            "longitude": [lon, lon, lon, lon, 77.0, lon],
            "acq_date": ["2024-10-10", "2024-10-10", "2024-10-10", "2024-10-10", "2024-10-10", "2024-08-01"],
            "acq_time": [800, 800, 800, 800, 800, 800],
            "confidence": ["n", "h", "l", "n", "n", "n"],
            "frp": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "daynight": ["D"] * 6,
            "type": [0, 0, 0, 2, 0, 0],
        }
    )
    out = clean(raw, {cell}, date(2024, 9, 15), date(2024, 12, 15))
    # kept: rows 0 and 1. dropped: low confidence, static source (type 2), outside box, outside season
    assert out["frp"].tolist() == [5.0, 6.0]
    assert set(out["cell"]) == {cell}
