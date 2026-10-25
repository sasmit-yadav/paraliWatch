# Learning guide: every AI concept in this project, where it lives, and what it showed

Use this page to study the project or to prepare for questions about it. Each concept has a
plain-language explanation, the file that implements it, the document with the details, and the
result it produced here. Read the sections in order: each builds on the previous one.

Suggested route through the docs:
[README](../README.md) -> [Architecture](ARCHITECTURE.md) -> [Model selection](MODEL_SELECTION.md)
-> [RAG](RAG.md) -> [Agents and MCP](AGENTS_AND_MCP.md) -> [Burn scars](BURN_SCARS.md)
-> [Voice](VOICE.md) -> [Fine-tuning](FINE_TUNING.md) -> [Operations](OPS.md)
-> [Interview guide](INTERVIEW_GUIDE.md).

---

## 1. Data and problem framing

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Problem framing | Turn "help officials act early" into a prediction target: will this cell see a fire in the next 3 days? | `parali/forecast/features.py` | Binary target per cell-day, 1.2M rows |
| Spatial indexing (H3) | Cover the map with equal-area hexagons so every place has neighbours at equal distances | `parali/pipeline/grid.py` | 2,455 cells over 43 districts |
| Data leakage | Using information that wouldn't exist at prediction time; makes offline scores look better than reality | `features.py`, `tests/test_features.py` | Every feature uses only data up to the prediction day, checked by a test |
| Time-based split | Train on the past, test on a later period, never on random rows | `parali/config.py` | Train 2020-23, validate 2024, test 2025 |

## 2. Classical machine learning

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Gradient boosting (LightGBM) | Many small decision trees, each fixing the errors of the previous ones | `parali/forecast/train.py` | Chosen model, PR-AUC 0.266 on 2025 |
| Baselines | Simple rules a model must beat to be worth its cost | `train.py` (`BASELINES`) | Best rule 0.191 |
| Model comparison | Same data, same split, many model families | `parali/forecast/compare.py` | 7 families, all learned models statistically tied |
| Class imbalance | Fires are rare (5-18% of rows), so weight positives more | `train.py` | Weighted training chosen on validation |
| PR-AUC, lift, capture@k | Metrics for rare events and for "check the top 10% first" | `parali/forecast/metrics.py` | Top 10% of cells hold 36.8% of fires |
| Calibration (isotonic vs Platt) | Turn scores into probabilities that match reality | `train.py` | Brier 0.074 -> 0.041 |
| Bootstrap confidence intervals | Resample days to see if a difference is real or noise | `compare.py` | Model beats best rule by +0.112 (CI +0.067 to +0.157) |
| Ablation | Remove feature groups to see what matters | `compare.py` | Past seasons' and neighbours' fires matter most |
| SHAP explanations | Split each prediction into per-feature contributions | `train.py` (`explain`) | Top 3 reasons shown for every cell |

## 3. Foundation models

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Time-series foundation model, zero-shot | A model pre-trained on many time series, used without training on ours | `parali/forecast/foundation.py` | Chronos-Bolt beats the own-cell rule, loses to LightGBM (can't see neighbours) |
| Vision Transformer (ViT) | Images cut into patches, processed like words in a sentence | `parali/detect/prithvi_mae.py` | Prithvi-EO-2.0 encoder |
| Masked autoencoder pre-training | Learn by hiding 75% of an image and reconstructing it | (pre-trained by NASA/IBM) | Our equivalence check: reconstruction loss 0.043 vs 0.83 random |
| Transfer learning: frozen vs fine-tuned vs scratch | How much of a pre-trained model to reuse | `parali/detect/train.py` | Fine-tuned 0.758 IoU > scratch 0.706 > frozen 0.627 |

## 4. Computer vision

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Semantic segmentation | A label for every pixel (burned / not burned) | `parali/detect/models.py` | U-Net 0.700 IoU, Prithvi 0.758 |
| U-Net | Encoder-decoder CNN with skip connections | `models.py` | Strong baseline with 6x fewer parameters |
| Dice + cross-entropy loss | Losses that cope with small burned areas | `parali/detect/train.py` | No-data pixels ignored (tested) |
| IoU | Overlap of predicted and true areas | `train.py`, `compare.py` | Paired bootstrap CI for every gap |
| Domain shift | A model trained in one place fails in another | `parali/detect/punjab.py` | US wildfire model flags 81-96% of post-harvest Punjab as burned |
| Satellite data access (STAC, COGs) | Search and read only the needed pixels from cloud-hosted images | `parali/detect/sentinel.py` | Sentinel-2 tiles read without an account |

## 5. Retrieval-augmented generation (RAG)

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| OCR | Read text from scanned pages | `parali/advise/ocr.py` | 5 scanned notices made searchable |
| Chunking | Split documents into passages for search | `parali/advise/chunking.py` | Fixed windows beat structure and semantic splits |
| Embeddings | Turn text into vectors where similar meanings are close | `parali/advise/embed.py` | Multilingual model, English and Hindi |
| BM25 keyword search | Classic word-match ranking | `parali/advise/index.py` | Recall@1 0.55 |
| Hybrid search + rank fusion | Combine keyword and vector results by rank | `parali/advise/retrieve.py` | Recall@1 0.61 |
| Cross-encoder reranking | Read question and passage together to score relevance | `retrieve.py`, `embed.py` | Recall@1 0.95; Hindi 0.10 -> 1.00 with a multilingual reranker |
| Abstention | Say "not found" when nothing relevant is retrieved | `parali/advise/answer.py` | 47 of 50 answer/refuse decisions correct |
| Guardrails | Check citations and numbers; flag prompt injection | `parali/advise/guard.py` | Caught swapped figures in LLM drafts |
| GraphRAG | A reviewed knowledge graph of facts, linked to answers | `parali/advise/graph.py` | Key-fact recall 68% -> 79% (quote mode) |
| RAG evaluation | Score retrieval and answers separately against gold pages and facts | `parali/advise/evaluate.py` | 50-question set incl. 10 Hindi and 12 unanswerable |
| LLM-as-judge (+ meta-evaluation) | A second model grades answers; we measure how far to trust it | `parali/advise/judge.py` | Faithfulness verdicts usable (35/37 quotes judged supported); relevance too lenient (kappa 0.18-0.53) |

## 6. Agents and tools

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Multi-agent workflow (LangGraph) | Several steps with their own role and state | `parali/agents/bulletin.py` | Monitor -> analyst -> writer bulletin, EN + HI |
| Structured outputs | The LLM must return data matching a schema | `bulletin.py` (Pydantic) | Numbers come from code, not the LLM |
| Human in the loop | An official approves before publishing | `/api/bulletin/approve` | Approval recorded with name and time |
| Tool use via MCP | A standard way for AI assistants to call our functions | `parali/agents/mcp_server.py` | 7 tools (risk, explanations, bulletin, search...) |
| Memory / personalisation | Remember a farmer's district, crop and language | `parali/advise/memory.py` | Used to rewrite questions |
| Agentic retrieval | Rewrite the query and search again when results look weak | `answer.py` (`retrieve`) | Only with an LLM connected |

## 7. Speech

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Speech-to-text (Whisper) | Transcribe spoken questions | `parali/voice/asr.py` | Small: WER 0.57 on FLEURS Hindi |
| WER / CER | Share of wrong words / characters | `parali/voice/evaluate.py` | Tiny and base unusable (wrong scripts) |
| Text-to-speech | Read the answer aloud | `parali/app/static/app.js` | Browser voices: no model, no licence, no server cost |

## 8. Fine-tuning and alignment

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Synthetic data (doc2query) | An LLM writes questions a passage answers | `parali/tune/data.py` | 114 questions -> 68 kept |
| Round-trip filtering | Keep a question only if search finds its source | `data.py` | Removes vague questions automatically |
| LoRA | Train small add-on matrices instead of the whole model | `parali/tune/lora.py` | 1.75% of weights; cited answers 0/37 -> 19/37 |
| Loss masking | Learn only from the answer tokens, not the prompt | `lora.py` | Also saves 2 GB of memory |
| DPO (preference optimisation) | Learn from pairs: this answer is better than that one | `parali/tune/dpo.py` | Wrong refusals 17 -> 0, but correct refusals 3 -> 0: the pairs define the behaviour |
| Teacher limits | A student model copies its teacher, good and bad | [Fine-tuning](FINE_TUNING.md) | Format learned; content capped by the quote teacher |

## 9. Operations (MLOps / LLMOps)

| Concept | In plain words | Where | Result here |
| --- | --- | --- | --- |
| Experiment tracking | Every run's settings, metrics and files in one place | `parali/ops/tracking.py` | 30 runs across 5 experiments in MLflow |
| Model registry, champion/challenger | Versioned models; a new one goes live only if it is better | `tracking.py` | Rule unit-tested |
| Drift (PSI, KS) | Has the input or output distribution changed? | `parali/ops/drift.py` | 2025: big input drift, ranking still holds |
| Tracing | Time and details of each step of one request | `parali/ops/trace.py` | Reranker is the slow step (~5 s) |
| Feedback loop | User ratings become new test cases | `parali/ops/feedback.py` | Found and fixed a sentence-splitting bug |
| Testing and CI | Automatic checks on every change | `tests/`, `.github/workflows/ci.yml` | Unit tests for leakage, metrics, guardrails, DPO maths |
| Containers and deployment | Same environment everywhere; public demo | `Dockerfile`, `deploy/huggingface/` | Image verified; Space script tested |

## 10. Not covered (and why)

- **Training an LLM from scratch or full RLHF with a reward model**: needs GPUs and far more
  data. DPO covers preference learning without a reward model.
- **Image generation (diffusion)**: no use in this problem.
- **Vision-language models**: the burn-scar task needs pixel masks, which a segmentation model
  gives directly.
- **Graph neural networks**: the hexagon grid is a graph; neighbour features already carry most
  of that signal (see the ablation). A GNN would be a reasonable next experiment.
- **Conformal prediction** (guaranteed uncertainty intervals) for the forecast: listed in the
  roadmap.
