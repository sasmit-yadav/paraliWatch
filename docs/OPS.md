# Running it for real: tracking, drift, tracing, feedback, deployment (Phase 4)

A model that scores well once is not a product. This page covers what keeps Parali Watch
trustworthy after it ships, and how to deploy it. Each section answers four questions:
**what is it, why does it matter here, how is it built, and what did we find.**

| Part | Command / endpoint | Code |
| --- | --- | --- |
| Experiment tracking + model registry | `python -m parali track`, `PARALI_TRACKING=1 python -m parali train` | `parali/ops/tracking.py` |
| Drift monitoring | `python -m parali drift [season ...]` | `parali/ops/drift.py` |
| Tracing | every `POST /api/ask`; `GET /api/ops/summary`, `GET /api/ops/trace/{id}` | `parali/ops/trace.py` |
| Feedback loop | thumbs in the dashboard, `POST /api/feedback`, `python -m parali.ops.feedback export` | `parali/ops/feedback.py` |
| Container | `docker compose up` | `Dockerfile` |
| Public demo | `bash deploy/huggingface/deploy.sh <user>/<space>` | `deploy/huggingface/` |

Install the optional tracking dependency with `uv sync --group ops` (MLflow). Everything else
runs on the core install.

---

## 1. Experiment tracking and the model registry (MLflow)

**What.** Experiment tracking records every training run: settings (parameters), results
(metrics), and outputs (artifacts such as the model file), in one searchable place. A *model
registry* keeps numbered versions of a model and labels (aliases) such as `champion` that say
which version is live.

**Why here.** The project already has 30+ experiments spread over JSON files: seven forecast
model families, bootstrap tests, four burn-scar models, three speech models, LLM variants. Without
a tracker the only record of "why is this the production model?" is a document. With one, every
number can be traced to the run that produced it, and retraining can't silently replace a better
model with a worse one.

**How.**

- Storage is local: a SQLite database and an artifact folder under `data/ops/` (not committed).
  Browse it with `mlflow ui --backend-store-uri sqlite:///data/ops/mlflow.db`.
- `python -m parali track` (backfill) loads everything already in `eval/` into five experiments:
  `forecast-model-comparison` (16 runs), `forecast-production`, `burn-scars` (4 runs with loss
  curves), `speech-to-text` (3), `answer-llm` (5). It refuses to run twice unless `--force`.
- `PARALI_TRACKING=1 python -m parali train` logs a new training run with its data hash,
  registers the LightGBM model (MLflow's LightGBM format, so `mlflow.lightgbm.load_model` and
  serving tools work) and stores the calibrator and metadata alongside.
- **Champion/challenger rule** (`promote_if_better`): a new version gets the `champion` alias
  only if its PR-AUC on the *validation* season beats the current champion. The test season is
  never used to choose. A rejected version is tagged `challenger-rejected`, so the attempt stays
  visible. Unit-tested with three versions (better, worse, better).

**Checked.** The registered champion, loaded back through MLflow, gives exactly the same
predictions as the model file (maximum difference 0.0 on 1,000 rows).

---

## 2. Drift monitoring

**What.** *Drift* means the data a model sees after deployment no longer looks like its training
data. Three kinds matter:

- **Feature (input) drift.** The inputs change, for example fewer fires detected per cell.
- **Prediction drift.** The model's output scores change. This needs no labels, so it can be
  watched every day of a live season.
- **Label / performance drift.** The outcome rate changes, or the model ranks worse. This can
  only be seen once outcomes are known.

**How it is measured.**

- **PSI (Population Stability Index).** Cut the training values of a feature into 10 quantile
  bins, then compare the share of new values in each bin:
  `PSI = sum((new% - train%) * ln(new% / train%))`.
  The usual reading, from credit scoring: below 0.10 stable, 0.10 to 0.25 moderate, above 0.25
  major. Fire counts are mostly zero, so identical bin edges are merged and zero gets one bin
  instead of ten (unit-tested on zero-heavy data).
- **KS statistic.** The largest gap between the two cumulative distributions, as a second opinion.
- **Weekly score drift is compared with the same weeks of the training seasons.** Comparing
  October with "the whole training season" would flag the normal rise of the burning season as
  drift.
- **Calibration.** Mean predicted probability against the actual fire rate.

**Actions are tied to impact, not to drift alone.** A drift alarm that never leads to a decision
gets ignored.

- **Retrain** if lift (PR-AUC divided by the fire rate) falls more than 25% below its value on the
  validation season.
- **Recalibrate** if the predicted fire rate is off by more than 30%.
- **Investigate** if inputs moved but neither of the above happened.

**What we found** (`eval/drift/REPORT.md`).

| Season | Fire rate (train -> season) | Score PSI | Features with major drift | Lift (validation 4.1x) | Predicted / actual fire rate | Action |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | 0.184 -> 0.086 | 0.18 moderate | 9 of 26 | 4.1x | 1.00* | investigate |
| 2025 | 0.184 -> 0.049 | 0.62 major | 14 of 26 | 5.4x | 1.10 | investigate |

\*2024 is the season the calibrator was fitted on, so 1.00 is by construction.

- The inputs drifted massively. Season-level fire counts fell by more than 80% (for example
  `state_3d`, fires in the state over the last 3 days: mean 1,156 -> 96). This is the satellite
  detection drop already documented in Phase 1. Only the weather features and the day of the
  season stayed stable.
- **Yet the model still ranks well (lift 5.4x) and its probabilities are within 10% of reality.**
  Every fire feature fell together, so the *relative* order of cells held, and the fire rate fell
  with the scores. So the right call is "investigate the data source", not "retrain".
  Retraining on drift alone would have been wasted work.
- **Early warning.** In 2025 the weekly score PSI was already 0.83 in the first week of the
  season (2024: 0.08), weeks before the outcome data shows anything. That is the signal that
  would have prompted checking the satellite data at the start of the season.

---

## 3. Tracing: what happened inside one answer

**What.** A *trace* is the record of one request. Its *spans* are the steps, nested like a call
tree, each with a duration and attributes. Here the steps are: input guard, retrieval (hybrid
search, rerank, optional query rewrite), knowledge-graph facts, LLM, output guard, and quoting
the sources.

**Why.** When a user says "the answer was wrong" or "it was slow", logs alone can't tell you
which step failed. With the trace id returned in every answer, `/api/ops/trace/{id}` shows which
passages were retrieved (top three by page), their scores, whether the LLM ran, and what the
guardrails decided.

**How.**

- About 150 lines with no extra service (`parali/ops/trace.py`). Python's `contextvars`
  track the current span, so nested `with trace.span(...)` blocks build the tree automatically,
  and requests handled at the same time in different threads never mix their spans.
- Field names follow OpenTelemetry (`trace_id`, `span_id`, `parent_id`, `attributes`), so the
  records could be shipped to Jaeger, Grafana Tempo or Langfuse later without changing the code
  that creates them.
- Finished traces are appended to `data/ops/traces.jsonl`. Writing never breaks an answer: file
  errors are swallowed. Set `PARALI_TRACE=0` to turn it off.
- `/api/ops/summary` reports p50/p95 latency per step and how often each answer mode was used.

**What we found** (2-core CPU): the reranker dominates warm latency, about 5 s per question for
10 candidate passages, against under 0.1 s for hybrid search. The first question after start-up
also paid for index building and model loading; warm-up (on by default, `PARALI_WARMUP=0` turns it off) now does that at start-up (section 7).

---

## 4. The feedback loop

**What.** Users rate answers with a thumbs up or down. Down-voted answers become candidates for
the evaluation set, so every real failure turns into a test that future versions must pass.

**How.**

1. Every answer from `/api/ask` is stored with its trace id: question, answer, mode, cited pages
   (SQLite, `data/ops/feedback.db`).
2. The dashboard shows "Was this useful?" under each answer, which calls `POST /api/feedback`
   with `{"trace_id", "rating": 1 | -1, "comment"}`. An unknown trace id gets a 404.
3. `python -m parali.ops.feedback export` writes new down-voted answers to
   `eval/rag/feedback_candidates.jsonl` in the evaluation format, with a TODO for a person to fill
   in the gold page and key facts. There are no duplicates across exports.
4. Reviewed lines move into `eval/rag/questions.jsonl`. Retrieval and answer evaluations then
   cover them.

**It already paid off.** While testing the loop, the first down-voted answer began with a broken
quote: "...for a period of one-year w.e.f. [1]". The sentence splitter had cut at the full stop
in "w.e.f." (with effect from), an abbreviation all over Indian government notices. Fixing it
(common abbreviations such as w.e.f., No., Rs., Sh., i.e. no longer end a sentence; unit-tested)
raised key-fact recall in quote mode without the knowledge graph from 64% to **68%**.

Privacy: the store holds user questions, so it stays local and is git-ignored. A deployment
should state that ratings and questions are kept to improve answers.

---

## 5. Docker

`docker compose up` builds and runs the app on port 8000.

**Verified in Phase 4.** The image builds (1.4 GB, CPU-only). In the running container
`/api/health`, the forecast endpoints and the dashboard answer. `/api/ask` returns cited answers
after downloading the search models on first use.

Changes made while verifying:

- The base image is a build argument (`--build-arg BASE_IMAGE=...`), useful when Docker Hub
  rate-limits.
- Experiment-only weights (LoRA adapters, burn-scar model) stay out of the build context.
- The container must work as a non-root user, as on Hugging Face Spaces. Running it with
  `-u 1000` exposed a real bug: model downloads failed with "Permission denied" because the
  libraries write caches under `HOME`, which was `/`. The image now sets `HOME`, `HF_HOME` and
  `PARALI_MODEL_CACHE` inside a writable `/app/data`. Re-tested as user 1000 with
  `PARALI_WARMUP=1`: no errors, and the first question answered in 9 s with the models already
  loaded.

---

## 6. Public demo on Hugging Face Spaces

A free CPU Space (2 vCPU, 16 GB) runs the full app, including the document assistant without an
LLM (quote mode).

1. Create a Space at https://huggingface.co/new-space with SDK **Docker** and hardware
   **CPU basic**.
2. `pip install -U huggingface_hub && huggingface-cli login` (a token with *write* access).
3. From the repository root: `bash deploy/huggingface/deploy.sh <your-username>/<space-name>`.

The script copies only git-tracked files and uses `deploy/huggingface/README.md` as the Space
card, which tells Hugging Face to use port 8000. It leaves out the 25 MB burn-scar weights, which
the app doesn't use (Spaces need Git LFS above 10 MB), and adds `PARALI_WARMUP=1` so the search
models load while the Space starts. It was tested against a local git remote. The real push needs
your Hugging Face login.

To add written answers on the Space, set the Space secrets `PARALI_LLM_BASE_URL`,
`PARALI_LLM_MODEL` and `PARALI_LLM_API_KEY` to any OpenAI-compatible provider. The guardrails
still apply.

---

## 7. Speed

Measured on a 2-core cloud machine (a normal laptop is faster). Each change was checked against
the evaluation so nothing got faster by getting worse.

| What | Before | After | How |
| --- | --- | --- | --- |
| First question after the app starts | 56 s (models loaded on demand) | answered at normal speed once warm-up finishes: 9 s after start (75 s on the very first start) | Warm-up on by default: index, both models and every sentence vector load in the background at start-up; sentence vectors are saved to disk |
| Picking the quotes for an answer | 0.7-1.2 s | 0.07 s | Sentence vectors computed once and reused |
| Reranking 10 passages | 3.3 s average, up to 8.5 s when one passage is long | 2.9 s average, 2.5 s worst case | Reranker reads at most 320 tokens per pair, so one long passage no longer pads the whole batch. Recall@1 unchanged (0.95); 256 tokens was faster but dropped it to 0.87, so it was rejected |
| Asking the same question again | full answer time | 4 ms | Cache of the last 256 answers (keyed by question, profile and LLM) |
| One day's risk map download | 517 KB | 77 KB | GZip compression |
| Stepping / playing through the season | a request per day | instant | The browser keeps day data and fetches the next two days ahead; responses carry cache headers |

Also tried and rejected: a smaller rerank pool (8 or 6 passages instead of 10) is 20-40% faster
but recall@1 fell from 0.95 to 0.89 / 0.87.

The remaining cost is the reranker itself (a 568M-parameter multilingual model, needed for
Hindi). On a GPU, or with an LLM-free cached answer, answers are effectively instant.

## 8. What is not done (and would be next)

- **Alerts.** Drift and latency are reports, not notifications. Next would be a scheduled job
  (cron or GitHub Actions) running `python -m parali drift` weekly during the season and posting
  to a channel when an action other than "none" appears.
- **A/B testing.** The champion/challenger rule compares models offline. Comparing them on live
  traffic would need both served at once and the feedback split by version.
- **Trace storage at scale.** A JSONL file is fine for a demo. Production would export the same
  spans to an OpenTelemetry collector.
