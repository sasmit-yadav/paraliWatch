# Architecture

One Python package, one process, no database server (SQLite files only, for feedback and the
optional MLflow store). The full design (all phases) is in the
project design document; this page describes what exists in the repository today.

```
NASA FIRMS ──┐
geoBoundaries ├─> parali/pipeline ──> data/processed/*.parquet ──> parali/forecast ──> models/
Open-Meteo ──┘   (download, clean,     (fires, weather, grid,       (features, train,    eval/
                  H3 grid)              feature table)               compare, export)    data/demo/
                                                                                              │
                                                    browser <── parali/app (FastAPI + map) <─┘
```

## Packages

| Path | Responsibility |
| --- | --- |
| `parali/config.py` | Every setting: area, grid resolution, seasons, split years, source URLs |
| `parali/pipeline/download.py` | HTTP with retries, backoff on rate limits, cached downloads |
| `parali/pipeline/boundaries.py` | State and district polygons for Punjab and Haryana |
| `parali/pipeline/grid.py` | H3 cells covering the area, district per cell |
| `parali/pipeline/firms.py` | FIRMS archive (past seasons) and area API (live), cleaning rules |
| `parali/pipeline/weather.py` | Open-Meteo ERA5 daily weather per 0.5 degree point |
| `parali/pipeline/build.py` | Runs the steps above, caching each output |
| `parali/forecast/features.py` | Cell x day feature table; leak-free by construction (tested) |
| `parali/forecast/train.py` | LightGBM training, class-weight trial, calibration, SHAP reasons |
| `parali/forecast/compare.py` | Model-family comparison, bootstrap tests, ablation |
| `parali/forecast/metrics.py` | PR-AUC, fire capture @10%, reliability tables |
| `parali/forecast/report.py` | Writes `eval/RESULTS.md` and `eval/metrics.json` |
| `parali/forecast/export.py` | Writes the demo files the dashboard reads |
| `parali/app/` | FastAPI API, plain-language explanations, MapLibre dashboard (map, bulletin, assistant) |
| `parali/advise/ingest.py`, `ocr.py`, `fetch.py` | Read PDFs/HTML/text with page numbers, OCR sidecars, header/footer removal, add pages by URL |
| `parali/advise/chunking.py`, `index.py`, `retrieve.py` | Three chunking strategies; LanceDB vectors + BM25+; hybrid fusion; multilingual reranking |
| `parali/advise/answer.py`, `guard.py`, `llm.py`, `memory.py`, `graph.py` | Cited answers, guardrails, any OpenAI-compatible LLM, farmer profiles, knowledge graph |
| `parali/advise/evaluate.py` | Retrieval and answer evaluation on `eval/rag/questions.jsonl` |
| `parali/agents/bulletin.py` | LangGraph monitor -> analyst -> writer bulletin with approval |
| `parali/agents/mcp_server.py` | MCP server exposing 7 tools |
| `parali/forecast/foundation.py` | Chronos-Bolt zero-shot comparison |
| `parali/detect/` | Sentinel-2 access, Prithvi-EO-2.0 and U-Net burn-scar models, training, bootstrap comparison, Punjab check |
| `parali/voice/` | Whisper speech-to-text and its WER evaluation |
| `parali/tune/` | Synthetic training data, LoRA fine-tuning, DPO, an OpenAI-compatible server for local models |
| `parali/advise/judge.py` | LLM-as-judge with meta-evaluation |
| `parali/ops/tracking.py` | MLflow tracking, model registry, champion/challenger rule |
| `parali/ops/drift.py` | Feature, prediction, label and calibration drift per season and per week |
| `parali/ops/trace.py`, `feedback.py` | Request tracing (OpenTelemetry-style spans) and the user feedback loop |
| `deploy/huggingface/` | Space card and deploy script for a public demo |

## API

| Endpoint | Returns |
| --- | --- |
| `GET /api/meta` | Available dates, default date, headline metrics |
| `GET /api/risk?date=&min_prob=` | GeoJSON hexagons with calibrated probability and daily rank |
| `GET /api/fires?date=&days=` | GeoJSON fire points |
| `GET /api/top?date=&n=` | Highest-risk cells that day |
| `GET /api/season` | Per-day totals for the season: fires detected, expected fire cells, high-risk cells (timeline) |
| `GET /api/cell/{cell}?date=` | Probability, rank, top 3 reasons in plain language, 14-day history, what happened next |
| `GET /api/district-summary?date=` | Expected fire cells and high-risk cells per district |
| `GET /api/districts` | District boundaries |
| `POST /api/ask` | Cited answer from the documents (`{"question": ..., "user_id": ...}`) |
| `POST /api/transcribe` | Speech to text (raw audio body; `?language=hi|en` optional). Needs `uv sync --group voice` |
| `GET/PUT /api/profile/{user_id}` | Farmer profile used to personalise answers |
| `GET /api/bulletin?date=` | Agent-written district bulletin (EN + HI), created as a draft |
| `POST /api/bulletin/approve?date=` | Record an official's approval |
| `POST /api/feedback` | Thumbs up/down on an answer (`{"trace_id", "rating": 1 or -1, "comment"}`) |
| `GET /api/ops/summary?last=` | Latency per answer step (p50/p95), answer modes, feedback counts |
| `GET /api/ops/trace/{trace_id}` | Every step of one answer, with timings and what each step saw |
| `GET /docs` | Interactive API documentation (generated by FastAPI) |

## Design choices

- **H3 hexagons instead of villages:** no village shapefiles needed, equal-area cells, and
  neighbours are one function call away.
- **Season split, never random rows:** prevents the model from learning a week's fires from the
  same week.
- **Matrices, not group-bys:** each season is a cells x days array, so rolling and neighbour sums
  are vectorised; building 1.2 million rows takes ~15 seconds.
- **Raw score for ranking, calibrated probability for display:** see `docs/MODEL_SELECTION.md`.
- **Vanilla JavaScript dashboard:** one HTML file, one script, MapLibre from a CDN; nothing to
  build. A framework can come later if the interface grows.

## Dashboard design

The dashboard is a dark, high-contrast "control room" layout. The visual language is inspired
by NVIDIA's product pages: near-black surfaces, one lime accent colour, bold condensed uppercase
type and thin rules. It is a general look, not a copy: no logos, no brand assets, our own icon.

- **Colour has one job each.** The lime accent marks interface state only: the selected day,
  the selected cell, the active tab, buttons and focus. Risk uses a single orange ramp, brighter
  meaning riskier, checked as an ordinal palette against the dark map: monotone lightness,
  visible steps, lowest band above 2:1 contrast. Alert levels in the bulletin use the status
  colours plus a text label, never colour alone.
- **Dark basemap without a paid tile service.** The OpenStreetMap raster is desaturated and its
  brightness inverted in MapLibre's paint properties, so the risk colours stand out.
- **Time is the main control.** A season timeline (fires detected per day, from `/api/season`)
  sits above the map. Click or drag to jump; hover for that day's numbers. Play animates the
  season one day per second; the arrow keys step through days.
- **Three tabs instead of one long column:** Risk (selected cell with its reasons, top cells),
  Bulletin, Assistant. Clicking a cell on the map opens its details on the Risk tab.
- **Answers are shown as cited statements:** each quote or sentence in its own block with its
  source number, then the reviewed knowledge-graph facts, then the source list.
- **Headline numbers in the header** (fire capture at the top 10% of cells, PR-AUC, each against
  the best simple rule) come from `/api/meta`, so they always match the model being served.
- **Phone layout:** under 900 px the map and timeline come first, the tabs below, and the
  legend leaves the map.

Plain HTML, CSS and JavaScript, three files and no build step. Fonts: Barlow and Barlow
Condensed (Google Fonts, SIL Open Font Licence).
