# Farmer and official assistant (RAG)

The assistant answers questions about crop-residue rules, schemes and fire statistics **only from
documents in the knowledge base, with a page citation for every answer**, in English or Hindi. If
the documents don't contain the answer it says so and points to the Kisan Call Centre, instead of
guessing.

## 1. Knowledge base

| Documents | Pages used | What they cover |
| --- | --- | --- |
| CAQM Directions 38, 40, 42, 92 and Advisories 5, 6, 11 | 29 | Stubble-burning action plans, ex-situ straw use, co-firing in power plants and brick kilns, pellet price and quality |
| PIB releases (May 2026 CAQM review; bio-bitumen) | 6 | 2026 preparedness, wheat-season fire counts, hotspot districts, straw-to-bitumen |
| Haryana: Mera Pani Meri Virasat note | 2 | Crop diversification incentive and districts |
| ICAR-IARI CREAMS fire bulletins (4) | 25 text pages | Daily fire counts by state, satellites used, monitoring limits |
| Haryana measures summary from the project author (**marked UNVERIFIED**) | 1 | Incentives, machinery subsidy, penalties; shown with its "unverified" label in every citation |

`data/corpus/sources.csv` lists each file with title, link and publisher. The repository ships the
extracted text (`data/corpus/pages.jsonl`), not the PDFs, so it works after a fresh clone.

**Getting clean text was half the work:**

- Five CAQM notices were scans. Two had **no text layer**; three had a broken one ("Jowohor Vyopor"
  for "Jawahar Vyapar", "lmplemenlolion"). `python -m parali ocr` runs Tesseract and saves a
  `<file>.ocr.txt` sidecar per PDF, which ingestion then prefers.
- Printed web pages carry a browser header and footer on every page. Lines repeated on most pages
  of a document are removed, and the page link in the footer becomes the citation URL.
- CREAMS bulletins are mostly charts; their text layer is axis labels ("0 1 2 ... 15-Sep"). Lines
  that are only numbers or dates are dropped, and pages with under 25 real words are skipped.

## 2. Pipeline

```
question ─> guard ─> hybrid search (BM25 + vectors, fused) ─> rerank top 10 ─> relevant? ─no─> "not in my documents"
                                                                                  │yes
                                                                                  v
                                 answer from sources with [n] citations (LLM) ──checks fail──> quote the sources
```

| Step | Choice | Why |
| --- | --- | --- |
| Chunking | Fixed 180 words, 40 overlap, never across pages | Won the evaluation (below); OCR text breaks heading detection |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (ONNX, 220 MB) | Hindi and English in one vector space; beat the larger mpnet model |
| Keyword search | BM25+ with a small Hindi-English term list (पराली -> paddy straw) | Exact names and numbers; bridges Hindi questions to English documents |
| Fusion | Reciprocal rank fusion (k = 60) | Combines lists without calibrating BM25 against cosine scores |
| Reranker | `bge-reranker-v2-m3` (multilingual, int8 ONNX) | Biggest single gain: recall@1 0.61 -> 0.95 |
| Relevance cut-off | Reranker score -0.8 | Separates answerable from unanswerable questions best on the evaluation set |
| Vector store | LanceDB (embedded) | No server to run |
| Generation | Any OpenAI-compatible LLM; without one, quotes the best sentences | Works with zero setup; better with a model |

## 3. Guardrails

| Risk | Guard |
| --- | --- |
| Question not covered by the documents | Below the relevance cut-off -> "I could not find this in the documents" + helpline |
| LLM invents an amount or date | Every number in the answer must appear in a cited source, its title or the question; otherwise the answer is replaced by quotes from the sources |
| Answer without a source | An answer with no [n] citation is rejected the same way |
| Prompt injection ("ignore previous instructions...") | Documents and question are wrapped as data; the prompt says to ignore instructions inside them; suspicious questions are flagged |
| Unverified or conflicting sources | The source title carries "UNVERIFIED"; the prompt asks the model to say so and to give both figures when sources disagree |

## 4. Evaluation

`eval/rag/questions.jsonl`: **50 questions** written from the documents, each with the page(s)
that answer it and the key facts an answer must contain.

- 38 answerable: 28 English, 10 Hindi
- 12 unanswerable: off-topic (weather, PM-KISAN) and **hard negatives** that sound covered but aren't
  (drip irrigation subsidy, baler price, CRM application deadline)

Recall@k = share of answerable questions with a correct page in the top k. MRR = mean of
1 / rank of the first correct page. Latency on a 2-core CPU.

### 4.1 Retrieval: what each step adds

Fixed chunking, MiniLM embeddings (the chosen setup), all 38 answerable questions:

| Search | Recall@1 | Recall@5 | MRR | Latency |
| --- | --- | --- | --- | --- |
| Keyword only (BM25+) | 0.55 | 0.79 | 0.651 | <1 ms |
| Vectors only | 0.47 | 0.76 | 0.586 | 58 ms |
| Hybrid (RRF) | 0.61 | 0.92 | 0.714 | 58 ms |
| **Hybrid + bge-reranker-v2-m3 (chosen)** | **0.95** | **0.97** | **0.961** | 6.2 s (20 candidates) / 2.8 s (10) |
| Hybrid + English ms-marco reranker | 0.63 | 0.84 | 0.710 | 1.2 s |

**Hindi vs English** (same chosen setup; 10 Hindi and 28 English questions):

| Search | English recall@1 | Hindi recall@1 | Hindi recall@5 |
| --- | --- | --- | --- |
| Keyword only | 0.68 | 0.20 | 0.30 |
| Vectors only | 0.46 | 0.50 | 0.80 |
| Hybrid | 0.68 | 0.40 | 0.90 |
| Hybrid + multilingual reranker | 0.93 | **1.00** | 1.00 |
| Hybrid + English reranker | 0.82 | **0.10** | 0.50 |

Findings:

- **Keyword search is strong in English but nearly blind in Hindi** (recall@1 0.20): the documents
  are English. Multilingual vectors bridge the languages; fusion gets the best of both at rank 5.
- **The reranker is the biggest single improvement** (recall@1 0.61 -> 0.95), and it must be
  multilingual: an English-only reranker *destroys* Hindi results (1.00 -> 0.10).
- **Latency is the price.** bge-reranker-v2-m3 takes ~6 s per question on 2 CPU cores when scoring
  20 candidates. Scoring 10 keeps recall@1 at 0.95 (recall@5 0.97 -> 0.95) at 2.8 s, so the app uses 10.

### 4.2 Chunking strategy and embedding model

Recall@1 / MRR with the multilingual reranker (MiniLM embeddings):

| Chunking | Hybrid recall@5 | + rerank recall@1 | + rerank MRR |
| --- | --- | --- | --- |
| **Fixed 180 words, 40 overlap (chosen)** | 0.92 | **0.95** | **0.961** |
| Structure (split at headings and blank lines) | 0.84 | 0.89 | 0.930 |
| Semantic (split where sentence meaning shifts) | 0.79 | 0.87 | 0.900 |

The "smarter" strategies lost. Government notices here are OCR'd scans and printed web pages:
numbered clauses arrive as "12. And further the..." mid-line and headings get merged into text, so
heading detection and sentence similarity find the wrong boundaries. Plain overlapping windows are
robust to that noise.

Embedding model (hybrid search, fixed chunks): `paraphrase-multilingual-MiniLM-L12-v2` recall@5
**0.92** vs `paraphrase-multilingual-mpnet-base-v2` 0.82 (and 0.37 vs 0.47 recall@1 for vectors
alone), despite mpnet being 4x larger. MiniLM was kept; mpnet was not run through the reranker.

### 4.3 Knowing when not to answer

The top reranker score separates answerable from unanswerable questions: answerable questions
mostly score above +0.5, unanswerable ones below -0.9. A cut-off of **-0.8** classified 47 of 50
questions correctly. Caveat: the cut-off was chosen on the same 50 questions it is reported on.

### 4.4 Answers

Generated from the retrieved passages of each question (`python -m parali.advise.evaluate answers`).
In quote mode the assistant picks the 5 sentences or clauses closest in meaning to the question;
long legal sentences are split at ";", ":" and "(ii)" so quotes stay readable. (Splitting first cut
fact recall from 59% to 48% with 3 quotes; 5 shorter quotes recovered it to 64%.) The LLM column
was measured before this change.

| Measure | Quotes only (no LLM) | With Qwen2.5-1.5B (local, 4-bit) |
| --- | --- | --- |
| Answers citing a correct page (answerable) | 87% | 84% |
| Key facts present in the answer (no graph) | 64% (English 65%, Hindi 60%) | 59% |
| Wrong "not found" on answerable questions | 1 of 38 | 1 of 38 |
| Answered an unanswerable question | 3 of 12 | 3 of 12 |
| LLM answers that passed the guardrails | - | **1 of 37** |
| Median time to compose an answer | 1.9 s | 19 s |

**What the small LLM did, and why the guardrails matter.** Its answers were fluent but 36 of 37
had no [n] citation, so they were rejected and the assistant fell back to quoting the sources. The
rejected drafts show why that is the right call: in a sample of six, one **swapped Punjab's and
Haryana's brick-kiln counts** (940 and 2,050) and one **changed the dates of the co-firing targets**
(November -> January, and invented "July 2027"). Both swapped numbers do appear in the sources, so
the number check alone would not have caught the first error; the citation requirement did.
Recommendation: use a stronger model (7B+ locally, or a hosted 70B through an OpenAI-compatible API)
for written answers; keep quote mode as the default.

**The 3 unanswerable questions it still answered** sit right next to covered topics (DSR
application process, tractor subsidy under a named scheme, fires on 5 Nov 2025): the passages
retrieved were about the right scheme or season, so the reranker scored them as relevant. In quote
mode the user sees the passage and its source, not an invented answer.

### 4.5 Knowledge graph (GraphRAG)

`data/corpus/knowledge_graph.json` holds 20 reviewed facts linking schemes, machines, power
plants, brick kilns, pellets and penalties. Each fact names its document and page and carries an
exact quote that is checked against the page text. When a question mentions an entity (in English
or Hindi, via aliases such as "मेरा पानी मेरी विरासत" or "ईंट भट्टों"), its facts and their
one-hop neighbours are added to the answer as "Related facts" and to the LLM's context.

| Measure (quote mode) | Without graph | With graph |
| --- | --- | --- |
| Key facts present in the answer | 64% | **79%** |
| English / Hindi | 65% / 60% | 78% / 80% |
| Citing a correct page | 87% | 87% |
| Median answer length | 134 words | 201 words |

The graph helps most where facts are spread out or conflict: the Mera Pani Meri Virasat question
now shows both Rs 7,000 (official 2020-21 note) and Rs 8,000 (unverified summary) with their
sources, and Hindi questions get English facts they would otherwise miss.

**Treat the +15 points as optimistic.** The graph was curated by the same person who wrote the
evaluation questions, after reading the same documents, so its facts overlap the questions more
than a graph built independently would. A fair test needs new questions written without seeing the
graph. Automatic extraction exists (`python -m parali graph`, LLM -> validated triples whose quotes
must appear in the chunk -> `data/index/graph_extracted.json` for review), but with the 1.5B local
model it took 1-5 minutes per chunk and was stopped; it is meant for a stronger model.

## 5. Reproduce

```bash
python -m parali ocr                            # only if you add scanned PDFs (needs Tesseract)
python -m parali index                          # builds data/index from data/corpus
python -m parali.advise.evaluate retrieval      # all strategies x models x search modes
python -m parali.advise.evaluate answers retrieve
python -m parali.advise.evaluate answers generate   # set PARALI_LLM_* to test an LLM
PARALI_USE_GRAPH=0 python -m parali.advise.evaluate answers generate   # without the knowledge graph
```

## 6. Limits

- 50 questions is enough to compare setups and catch failures, not to quote tight percentages.
- Hindi questions are answered from English documents; in quote mode the quotes are English.
- The knowledge base is thin on farmer-facing scheme details (incentive amounts, how to apply);
  most of that currently comes from the **unverified** summary. Adding the official notices is the
  single most useful improvement.
