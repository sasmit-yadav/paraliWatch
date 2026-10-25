# Roadmap

| Phase | Weeks | Scope | Status |
| --- | --- | --- | --- |
| 1. Core | 1-3 | Data pipeline, forecast with baselines, calibration, SHAP, model comparison, map dashboard, tests, CI | Done (deploy pending) |
| 2. GenAI layer | 4-5 | Hybrid RAG with multilingual reranking and OCR, GraphRAG, 50-question evaluation, LangGraph bulletin agents with approval, MCP server, Chronos comparison | Done |
| 3. Vision + voice | 6-7 | Burn-scar segmentation (Prithvi-EO-2.0 vs U-Net), Hindi speech in/out, LoRA small model | Done |
| 4. Ops + polish | 8 | MLflow registry, drift checks, tracing, human feedback loop, public demo | Planned |

## Phase 1 follow-ups

- [ ] Live mode: pull today's fires with a FIRMS key and forecast the current season
- [ ] Weather forecast (next 3 days) as features, not only observed weather
- [ ] Geostationary fire detections to reduce the satellite-timing blind spot
- [ ] Separate forecasts for day +1, +2, +3
- [ ] Conformal intervals on each cell's probability
- [ ] Deploy demo mode to Hugging Face Spaces

## Phase 2 follow-ups

- [ ] Official notices for farmer schemes (incentive, machinery subsidy, penalties) to replace the unverified summary
- [ ] Claim-level answer verification (each sentence checked against one passage)
- [ ] A held-out question set for the abstention cut-off
- [ ] Evaluate written answers with a 7B+ model

## Phase 3 follow-ups

- [ ] Label 200-300 Punjab fields (burned / harvested / standing) and fine-tune the burn-scar model on them
- [ ] Two-date (change) input for the burn-scar model; then measure how many burns VIIRS misses
- [ ] Record real farmer questions (Hindi, Punjabi, outdoors) as the speech test set; try large-v3-turbo on GPU
- [ ] Distil cited answers from a strong model (filtered by the guardrails) to train the small model
