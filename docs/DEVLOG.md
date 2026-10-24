# Development log

Decisions and findings, newest first. Each entry says what changed and why.

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
