# Development log

Decisions and findings, newest first. Each entry says what changed and why.

## 2026-10-24: Phase 1 core (data, forecast, dashboard)

**Data**
- Used FIRMS yearly **country archives** (VIIRS S-NPP, India) instead of the area API for past
  seasons: they need no key and cover full years. Live mode will use the API with a key.
- geoBoundaries names carry diacritics ("Haryāna"); names are normalised to ASCII before matching.
- Open-Meteo returned HTTP 429 when we requested 175 points at 0.25 degrees for 7 seasons. Moved to
  a 0.5 degree point grid (53 points), longer backoff on 429, and a pause between batches. All 7
  seasons then downloaded in about 2.5 minutes.
- Found that VIIRS daytime detections fell ~17x from 2021 to 2025 while night detections roughly
  doubled. Documented in `docs/DATA.md` as a limitation: the model predicts satellite-detected fires.

**Features**
- Each season is held as a cells x days matrix; neighbour sums are sparse-matrix products. Building
  1.2 M rows takes ~15 s.
- Tests check that no feature on day *t* sees fires from day *t+1* onward.

**Model**
- First LightGBM (lr 0.05, 63 leaves) stopped at 25 trees: it fit the high-fire 2020-2021 seasons
  and did not transfer to 2024. Slower and smaller settings (lr 0.02, 15 leaves, 1,000 rows per
  leaf) improved validation PR-AUC from 0.341 to 0.351.
- Class weighting beat plain training on validation (0.3507 vs 0.3452); kept.
- Compared 7 model families plus an ensemble. All learned models are within bootstrap noise of
  each other; all beat the best rule clearly. Kept LightGBM for speed and exact SHAP. Full write-up:
  `docs/MODEL_SELECTION.md`.
- Isotonic calibration beat Platt scaling on cross-fitted 2024 data; ranking stays on raw scores
  because isotonic creates ties.
- Test season 2025: PR-AUC 0.266 (best rule 0.191, random 0.049); the top 10% of cells each day
  held 36.8% of the next 3 days' fires (best rule 33.6%).

**Dashboard**
- First basemap (CARTO raster) showed an "API key required" watermark; switched to OpenStreetMap
  tiles, desaturated so the risk colours stand out.
- Cells under 5% chance are left clear; the first version coloured almost the whole map.
- "Top 1%" replaces "riskier than 100% of cells", which read badly.
