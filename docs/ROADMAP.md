# Roadmap

| Phase | Weeks | Scope | Status |
| --- | --- | --- | --- |
| 1. Core | 1-3 | Data pipeline, forecast with baselines, calibration, SHAP, model comparison, map dashboard, tests, CI | Done (deploy pending) |
| 2. GenAI layer | 4-5 | Hybrid RAG over Haryana crop-residue scheme PDFs, GraphRAG, evaluation set, LangGraph alert agents, MCP server, time-series foundation model comparison | Next |
| 3. Vision + voice | 6-7 | Burn-scar segmentation (Prithvi-EO-2.0 vs U-Net), Hindi speech in/out, QLoRA small model | Planned |
| 4. Ops + polish | 8 | MLflow registry, drift checks, tracing, human feedback loop, public demo | Planned |

## Phase 1 follow-ups

- [ ] Live mode: pull today's fires with a FIRMS key and forecast the current season
- [ ] Weather forecast (next 3 days) as features, not only observed weather
- [ ] Geostationary fire detections to reduce the satellite-timing blind spot
- [ ] Separate forecasts for day +1, +2, +3
- [ ] Conformal intervals on each cell's probability
- [ ] Deploy demo mode to Hugging Face Spaces
