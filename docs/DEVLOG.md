# Development log

Decisions and findings, newest first. Each entry says what changed and why.

## 2026-10-25: Phase 4, operations, evaluation with a judge, preference tuning, deployment

**Docker, verified for the first time**
- Built and ran the image: dashboard, forecast API and cited answers work (1.4 GB).
- Docker Hub rate-limited the base image pull: the base image is now a build argument.
- Running as a non-root user (as Hugging Face Spaces do) failed: libraries cache models under
  `HOME`, which was `/`. Set `HOME`, `HF_HOME` and the model cache inside a writable `/app/data`;
  re-tested as user 1000 with warm-up, first answer in 9 s.

**MLflow**
- Backfilled 30 runs into 5 experiments; the production LightGBM model is registered as
  `parali-forecast` with alias `champion`; loaded back, predictions are identical.
- Champion/challenger on the validation season only (unit-tested).

**Drift**
- PSI per feature: 14 of 26 features drifted majorly in 2025 (season fire counts down >80%),
  score PSI 0.62. Yet lift 5.4x (validation 4.1x) and predicted fire rate within 10%: the action
  is "investigate the data source", not retrain.
- First version compared each week with the whole training season and flagged the normal rise
  of the season as drift; now each week is compared with the same weeks of past seasons. Then
  2024 week 1 reads 0.08 (stable) and 2025 week 1 already 0.83: an early warning before any
  outcome is known.

**Tracing and feedback**
- OpenTelemetry-style spans for every answer; the reranker is the slow step (~5 s on 2 cores).
- The first down-voted answer in testing exposed quotes cut at "w.e.f."; abbreviations no longer
  end a sentence; quote-mode fact recall without the graph 64% -> 68%.

**Lesson (again): one heavy job at a time.** A container test running next to the judge model
pushed the machine out of memory and killed the judge mid-run; it was re-run afterwards.

## 2026-10-24 (evening): Phase 3, burn-scar segmentation, voice input, LoRA

**Burn scars**
- First tried weak labels for Punjab (dNBR near VIIRS points). Fires were only ~1.1x more likely
  than random pixels to sit on a large dNBR drop: harvest, haze and VIIRS's 375 m pixel swamp the
  signal. Dropped it rather than train on noise; switched to NASA's labelled HLS Burn Scars set.
- Vendored Prithvi-EO-2.0 model code and removed its `timm` dependency (it pulled a torchvision
  that broke on CPU torch). Proved the local block is identical: unit test vs PyTorch attention, and
  masked-reconstruction loss 0.043 with official weights vs 0.83 random.
- Four runs, same data/steps/seed, no checkpoint picking on validation: fine-tuned Prithvi 0.758
  IoU, Prithvi from scratch 0.706, U-Net 0.700, frozen encoder 0.627. Paired bootstrap: +0.05 to
  +0.06 over both baselines, CI above zero. Pre-training, not architecture, is what helped.
- On Sentinel-2 over Tarn Taran it flags 81-96% of post-harvest cropland as burned: domain shift.
  Short-interval "new scar" check gives only 1.15-1.25x enrichment with VIIRS. Not shown on the
  dashboard; local labels are the next step.
- Memory lesson: 2 CPUs / ~6 GB for all jobs. Running the LLM server beside the reranker or
  training got processes OOM-killed; heavy jobs now run strictly one at a time.

**Voice**
- `/api/transcribe` (faster-whisper, int8 CPU) and a mic button; answers are read aloud with the
  browser's own speech synthesis, so no TTS model or licence is needed (MMS-TTS is non-commercial).
- PyAV 17+ breaks faster-whisper 1.2's decoder; pinned `av<17`.
- FLEURS Hindi, 100 utterances: tiny WER 1.06 (Latin script, loops), base 1.04 (mostly Urdu
  script), small 0.57 / CER 0.26. Chose small.

**LoRA fine-tune of Qwen2.5-0.5B**
- Prompting failed: base 0.5B cited sources in 0/37 answers (1.5B: 1/37).
- Synthetic data: doc2query on 23 held-out chunks -> 114 questions -> 68 kept by the retrieval
  round-trip filter -> grounded quote targets + 12 reranker-verified "not found" negatives.
- First model repeated sentences: overlapping chunks had put duplicate sentences in 14 of 80
  targets. Fixed `_extractive` de-duplication (also in the live assistant) and rebuilt the data.
- OOM twice: full-vocabulary logits (~2 GB at 3,300 tokens) -> loss on answer positions only
  (verified identical); quadratic attention memory -> cap 2,304 tokens.
- Result: 19/37 answers pass guardrails (0 missing citations), but fact recall 0.61 vs 0.79 for
  plain quotes and too many Hindi refusals. Not the default; teacher quality is the limit.
- LLM refusals are now recorded as `llm_declined` instead of being counted as guardrail failures
  (the helpline number in the "not found" text was tripping the number check).

## 2026-10-24 (later): Phase 2, assistant, agents, MCP, foundation model

**Documents**
- Government sites (agriharyana, agriwelfare, CAQM, CREAMS) block automated downloads and web
  search is disabled for this account, so the corpus was collected by hand: 7 CAQM directions and
  advisories, 2 PIB releases, the Haryana Mera Pani Meri Virasat note, 4 ICAR-CREAMS bulletins.
  Two off-topic PDFs ("residue" in refineries and tea pesticides) were excluded.
- Five CAQM notices were scans (two with no text). Added Tesseract OCR with text sidecars.
- Printed web pages repeat a header and footer on every page: removed, and the footer URL became the
  citation link. Chart pages in the bulletins were mostly axis numbers: filtered.
- The project author supplied a summary of Haryana measures (incentives, subsidies, penalties) with
  no source. It is in the corpus labelled UNVERIFIED in every citation; the assistant is told to say
  so and to show both figures where it conflicts with official text (Mera Pani Meri Virasat:
  Rs 7,000 in the official 2020-21 note, Rs 8,000 in the summary).

**Retrieval** (50-question set; details in docs/RAG.md)
- Reranking with multilingual bge-reranker-v2-m3: recall@1 0.61 -> 0.95. An English reranker
  dropped Hindi recall@1 to 0.10.
- Fixed 180-word chunks beat structure-aware and semantic chunking on these OCR'd documents.
- MiniLM embeddings beat the 4x larger mpnet.
- Reranker latency ~6 s on 2 cores with 20 candidates; 10 candidates keep recall@1 at 0.95 (2.8 s).
- BM25 gave zero scores on a tiny test corpus (IDF of a word in half the documents is 0): switched
  to BM25+.

**Generation**
- A local Qwen2.5-1.5B answered fluently but without citations in 36 of 37 cases; rejected drafts
  swapped two states' figures and changed dates. Kept quote mode as the default.
- The local LLM server was killed by the memory limit when it ran next to the reranker; evaluation
  now retrieves and generates in separate stages.
- Grammar-constrained JSON on the local server took ~5 minutes per call; graph extraction now parses
  free-form output instead.
- Bulletins: the small model passed the schema and number checks yet contradicted the trend and
  produced broken Hindi. Added consistency checks and made the template the default.

- Quote-mode answers were walls of OCR text (legal sentences of 80+ words). Long sentences are now
  split at clause breaks and the 5 closest clauses are quoted: fact recall 59% -> 64%, citations on
  a correct page 84% -> 87%. A reviewed knowledge graph adds related facts: 64% -> 79% (optimistic;
  see docs/RAG.md).
- File transfers between the build machine and the repo folder reused staging paths and silently
  wrote stale copies of three files; caught by checksums, fixed, and every transferred file verified.

**Forecast**
- Chronos-Bolt (small, zero-shot) beat the own-cell rule but lost to LightGBM by 0.068 PR-AUC
  (bootstrap CI above zero): it can't see neighbouring cells.

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
