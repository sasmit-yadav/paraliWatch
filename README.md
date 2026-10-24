# paraliWatch

**Parali Watch forecasts where crop-residue (paddy straw) fires are likely in Punjab and Haryana
over the next 3 days, explains why, and shows it on a map**, so officials can send teams and
machines before the burning, not after.

![Dashboard: fire risk by hexagon, with the reasons for a selected cell](docs/images/dashboard.png)

## Why

Every October and November, paddy-straw burning in Punjab and Haryana adds heavy smoke to north
India's air. Satellites count fires after they happen. Parali Watch turns those counts into a
forward-looking, explained risk map at ~36 km2 resolution.

## Results (unseen 2025 season)

| Method | PR-AUC | Fire capture @10% |
| --- | --- | --- |
| **Parali Watch (LightGBM)** | **0.266** | **36.8%** |
| Best simple rule (fires in and around the cell, last 3 days) | 0.191 | 33.6% |
| Random guess | 0.049 | 10.0% |

*Fire capture @10%: if officials watched only the 10% of cells ranked highest each day, the share
of the next 3 days' fires they would cover.* Full results: [`eval/RESULTS.md`](eval/RESULTS.md).
How the model was chosen (7 model families, bootstrap tests, ablation):
[`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md).

## Quick start

Needs Python 3.10+ and [uv](https://docs.astral.sh/uv/). The repository ships the trained model
and 2025 demo data, so the dashboard runs without downloading anything.

```bash
git clone https://github.com/sasmit-yadav/paraliWatch.git
cd paraliWatch
uv sync
uv run python -m parali serve
```

Open http://127.0.0.1:8000 (API docs at http://127.0.0.1:8000/docs).

Or with Docker: `docker compose up --build`.

### Rebuild everything from the raw data

```bash
uv run python -m parali data    # downloads ~330 MB of FIRMS archives, boundaries, weather (~5 min)
uv run python -m parali train   # trains, evaluates, writes eval/ and data/demo/ (~2 min)
uv run pytest                   # 16 tests
```

## How it works

1. **Data:** NASA FIRMS VIIRS fire points (2019-2025), Open-Meteo ERA5 weather, geoBoundaries
   districts. See [`docs/DATA.md`](docs/DATA.md).
2. **Grid:** 2,455 H3 hexagons (~36 km2) covering both states.
3. **Features:** 26 per cell per day: recent fires in the cell and around it, last year's fires on
   the same days, past-season history, day of season, weather. Only data available by that day.
4. **Model:** LightGBM, trained on 2020-2023, tuned and calibrated on 2024, tested once on 2025.
5. **Explanations:** the top 3 SHAP reasons per cell, written as plain sentences ("15 fires in the
   6 surrounding cells over 7 days").
6. **Dashboard:** FastAPI + MapLibre. Pick a day, see risk, click a cell for its reasons, its
   14-day history and what actually happened next.

Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Honest limitations

- It predicts **satellite-detected** fires. Detected daytime fires fell ~17x from 2021 to 2025 while
  night detections rose, so some burning may be happening when the satellite is not overhead.
- The 2025 test season was much quieter than the training seasons.
- Weather adds little today because only observed weather is used; forecast weather is planned.

## Roadmap

Phase 1 (this release): data, forecast, model comparison, dashboard. Next: a Hindi assistant
grounded in government scheme documents (RAG), alert agents and an MCP server, burn-scar detection
from satellite images, and voice. See [`docs/ROADMAP.md`](docs/ROADMAP.md) and the
[development log](docs/DEVLOG.md).

## Data credits

NASA FIRMS (VIIRS S-NPP) · Open-Meteo / Copernicus ERA5 · geoBoundaries · OpenStreetMap
contributors (basemap).

## Licence

MIT. See [LICENSE](LICENSE).
