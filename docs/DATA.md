# Data

Everything is free and downloaded by `python -m parali data`. Raw downloads go to `data/raw/`
(or `PARALI_CACHE`), processed tables to `data/processed/`; neither is committed. The small files
the dashboard needs are committed in `data/demo/`.

## Sources

| Data | Source | Licence / terms | How we use it |
| --- | --- | --- | --- |
| Active fire points | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/), VIIRS S-NPP 375 m, yearly country archive for India | NASA open data; cite FIRMS | Fire counts per cell per day, 2019-2025 |
| Recent fires (live mode) | FIRMS area API, `VIIRS_SNPP_NRT` | Free MAP_KEY | Optional; turns on live mode ([Live mode](LIVE.md)) |
| District and state boundaries | [geoBoundaries](https://www.geoboundaries.org/) gbOpen IND ADM1 / ADM2 (simplified) | CC BY 2.5 IN (ADM1), ODbL 1.0 (ADM2) | Study-area outline, district names |
| Weather | [Open-Meteo historical API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) | CC BY 4.0, non-commercial free tier | Daily temperature, humidity, rain, wind |

## Processing

1. **Study area.** All districts of Punjab and Haryana (43 districts in the geoBoundaries file; it
   predates a few recently created districts such as Malerkotla, which fall inside their parent
   districts here).
2. **Grid.** H3 resolution 6 hexagons whose centre lies inside the area: **2,455 cells**
   (1,296 Punjab, 1,159 Haryana), ~36 km2 each. Each cell gets the district containing its centre.
3. **Fires.** From each yearly VIIRS file we keep points that are inside the area, between 15 Sep
   and 15 Dec, not low-confidence, and `type = 0` (presumed vegetation fire; static industrial
   sources and offshore flares are dropped). Each point is assigned to its H3 cell.
4. **Weather.** Cells are snapped to a 0.5 degree point grid (53 points). Open-Meteo returns daily
   values per point for each season. 0.5 degrees keeps us inside the free API limits; day-level
   weather is regional, and the ablation in `docs/MODEL_SELECTION.md` shows weather adds little.
5. **Feature table.** One row per cell per day: 26 features (see
   `parali/forecast/features.py`) and the target "any fire in the next 3 days". 1.2 million rows
   for 2020-2025 (2019 only seeds "last year" features).

## Fire points per season (study area, 15 Sep - 15 Dec)

| Season | Fire points | Punjab | Haryana | Night detections | Night share |
| --- | --- | --- | --- | --- | --- |
| 2019 | 49,867 | 43,921 | 5,946 | 198 | 0.4% |
| 2020 | 76,280 | 70,904 | 5,376 | 314 | 0.4% |
| 2021 | 84,206 | 74,546 | 9,660 | 210 | 0.2% |
| 2022 | 49,510 | 45,149 | 4,361 | 325 | 0.7% |
| 2023 | 33,114 | 30,274 | 2,840 | 283 | 0.9% |
| 2024 | 10,709 | 8,795 | 1,914 | 556 | 5.2% |
| 2025 | 5,215 | 4,014 | 1,201 | 420 | 8.1% |

## Known limitation: the satellite only sees what is burning when it passes

VIIRS on S-NPP passes over Punjab around 13:30 and 01:30 local time. Daytime detections fell about
17x from 2021 to 2025, while **night-time detections roughly doubled** (from ~200-300 to 420-556)
and their share rose from under 1% to 8%. That pattern is consistent with some burning moving away
from the early-afternoon overpass, but our data alone cannot separate a real fall in burning from a
shift in timing.

What this means for the project:

- The model predicts **satellite-detected** fires, not all fires. Every metric should be read that way.
- The 2025 test season is far quieter than the training seasons. The model still beats every
  simple rule on it, but absolute probabilities from a busy year should not be compared with a
  quiet one.
- Planned: add geostationary fire detections (e.g. from INSAT-3D/3DR), which observe every 15-30
  minutes, and the second VIIRS satellite (NOAA-20) if its archive becomes available for these years.

## Reproducibility

`models/meta.json` stores a hash of the processed fire and weather files the model was trained
on, the exact features, parameters and split years.
