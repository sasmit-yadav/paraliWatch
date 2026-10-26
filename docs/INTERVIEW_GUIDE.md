# Interview guide: explaining Parali Watch

How to present this project in an interview, what to say about each design decision, and answers
to the questions interviewers usually ask. Every number here comes from `eval/` and can be shown
live.

## 1. The 30-second pitch

> Every October and November, farmers in Punjab and Haryana burn paddy straw, and the smoke reaches
> Delhi. Satellites only count fires after they happen. I built Parali Watch, which forecasts which
> ~36 km2 areas are likely to see fires in the next 3 days and explains why, so officials can act
> before the burning. On the unseen 2025 season, the 10% of areas it ranks highest each day contain
> about 37% of the fires that follow, and it beats every simple rule I tested, including
> "fire spreads where fire already is". I compared seven model families, checked the differences
> with a bootstrap test, and documented a data problem I found: daytime fire detections fell 17x
> since 2021 while night detections doubled.

## 2. The 2-minute walkthrough

Say it in this order. Each step has one decision worth defending.

1. **Problem framing.** Binary classification: for each cell and day, "will at least one fire be
   detected here in the next 3 days?". *Decision:* a 3-day window because officials need lead
   time, and 1-day targets are very noisy.
2. **Data.** NASA FIRMS VIIRS fire points 2019-2025 (309k points in the area), ERA5 weather from
   Open-Meteo, district boundaries from geoBoundaries. *Decision:* yearly archives, no API key.
3. **Grid.** 2,455 H3 hexagons at resolution 6. *Decision:* hexagons have equal area and every
   neighbour is the same distance away, unlike square grids or villages of very different sizes.
4. **Features.** 26 per cell-day: the cell's recent fires, its neighbours' fires, last year's fires
   on the same days, past-season history, day of season, weather. *Decision:* every feature uses
   only data available by the end of that day (no leakage), checked by a unit test.
5. **Split.** Train 2020-2023, validate 2024, test 2025. *Decision:* split by season, never by
   random rows.
6. **Model.** LightGBM with class weighting, isotonic calibration. *Decision:* chosen on validation
   after comparing 7 families; all learned models were statistically tied, LightGBM won on speed and
   exact SHAP explanations.
7. **Output.** A map with each cell's probability, rank and top 3 reasons in plain language.

## 3. Numbers to remember

| What | Value |
| --- | --- |
| Area | Punjab + Haryana, 43 districts, 2,455 H3 cells (~36 km2 each) |
| Training rows | 805k cell-days (2020-2023); 1.2M rows overall |
| Test season fire rate | 4.9% of cell-days (random PR-AUC = 0.049) |
| Model PR-AUC on 2025 | **0.266** |
| Best simple rule PR-AUC on 2025 | 0.191 |
| Fire capture @10% on 2025 | **36.8%** (best rule 33.6%, random 10%) |
| ROC-AUC on 2025 | 0.849 |
| Brier score | 0.074 raw -> 0.041 calibrated |
| Model vs best rule (bootstrap, 2024) | +0.112 PR-AUC, 95% CI +0.067 to +0.157 |
| Fires detected per season | 84,206 (2021) -> 5,215 (2025) |
| Night share of detections | 0.2% (2021) -> 8.1% (2025) |
| Live 2026 so far (31 scored days to 22 Oct) | PR-AUC 0.227 vs best rule 0.144; capture@10% 72.6% vs 54.7% |

## 4. The decisions interviewers probe, with answers

### "Why did you split by season and not randomly?"

Fires cluster in space and time. With a random split, a cell's fire on 3 Nov would be in training
and its neighbour's fire on 3 Nov in testing, so the model would "predict" using the same event.
Splitting by whole seasons tests what we actually need: predicting a season the model has never
seen. Random splits would have given much better-looking, meaningless numbers.

### "How do you know there's no data leakage?"

Two places leakage can hide, both handled in `parali/forecast/features.py`:
- **Features** use `rolling()`, which sums today and earlier days only.
- **Target** uses `forward()`, which sums days t+1 to t+3 only.

`tests/test_features.py::test_no_future_information_in_features` puts 4 fires on 10 Oct and
checks they appear in the target of 7-9 Oct but in no feature before 10 Oct. History features
(`history_rate`) only average seasons *before* the current one.

### "Why PR-AUC and not accuracy or ROC-AUC?"

Only ~5-9% of cell-days have a fire. A model that always says "no fire" is 95% accurate and
useless. ROC-AUC also looks good on rare events because it rewards ranking the many easy
negatives. PR-AUC (average precision) focuses on how well real fire cells rise to the top, which is
exactly how the forecast is used. I also report "fire capture @10%" because officials understand
"if you watch these 10% of areas, you cover 37% of the fires".

### "Why LightGBM? Did you try deep learning?"

Yes. I compared logistic regression, an MLP neural network, random forest, scikit-learn's
HistGradientBoosting, XGBoost, two LightGBM settings and an ensemble. On the validation season they
scored between 0.337 and 0.353 PR-AUC. A day-level bootstrap showed the differences between the
learned models are within noise (LightGBM vs MLP: +0.003, 95% CI -0.034 to +0.044), and retraining
LightGBM with different random seeds alone moves it by about 0.003. So the algorithm matters less
than the features. I chose LightGBM because it was best on validation, trains in ~20 seconds and
gives exact SHAP values through TreeSHAP. Random forest scored slightly higher on the test season,
but I didn't switch, because choosing by the test set would make the test number meaningless.

### "What did tuning change?"

The first LightGBM (learning rate 0.05, 63 leaves) stopped after 25 trees. It learned the very
busy 2020-2021 seasons quickly and did not transfer to the quieter 2024. A slower, smaller, more
regularised model (0.02, 15 leaves, at least 1,000 rows per leaf) trained ~260 trees and improved
validation PR-AUC from 0.341 to 0.351. All tuning used 2024 only.

### "How did you handle class imbalance?"

I compared plain training with `scale_pos_weight` (negatives / positives, about 4.4 in the training seasons) on validation.
Weighting won slightly (0.3507 vs 0.3452), so I kept it. Because weighting distorts probabilities,
I calibrate afterwards.

### "What is calibration and why do you need it?"

A model score of 0.6 doesn't mean a 60% chance. I fit isotonic regression on 2024 to map scores to
observed fire rates. On 2025 the Brier score dropped from 0.074 to 0.041, and predicted vs observed
rates match within about a point in each band. I compared isotonic with Platt scaling using
alternate days of 2024 (isotonic was slightly better). Isotonic creates ties, so ranking uses the
raw score and only the displayed chance uses the calibrated one.

### "Which features matter most?"

From the ablation (remove one group, retrain, measure validation PR-AUC):
past seasons' fires (-0.029) > the cell's own recent fires (-0.011) > neighbours (-0.008) >
calendar (-0.004) > weather (-0.002) > district/state totals (0.000). Farmers burn the same fields
at similar times each year, so last year is the strongest signal. Weather barely helps because I
only use weather observed up to today; forecast rain for the next 3 days is the obvious next step.

### "How do you explain a prediction?"

LightGBM's `pred_contrib=True` returns exact SHAP values (TreeSHAP). For each cell I keep the 3
features that push the risk up most, with their values, and turn them into sentences such as
"15 fire(s) in the 6 surrounding cells over 7 days" (`parali/app/explain.py`).

### "What surprised you?"

Daytime fire detections fell ~17x between 2021 and 2025, while night detections roughly doubled
and rose from 0.2% to 8.1% of all detections. The satellite passes around 13:30, so this fits some
burning moving to times the satellite doesn't see. My data can't separate real reduction from a
timing shift, so I state that the model predicts *detected* fires, and I plan to add geostationary
satellites that observe every 15-30 minutes.

### "How would you deploy and maintain it?"

The repo ships a trained model and demo data; `python -m parali serve` starts FastAPI and the map.
There is a Dockerfile and a GitHub Actions CI that runs lint and 16 tests. For a live season:
pull today's fires daily with a FIRMS API key, rebuild that day's features, predict, and retrain
weekly. `models/meta.json` stores the data hash, features, parameters and split, so every model
can be traced to the data that produced it. Drift checks and a model registry are planned.

### "What would you do with more time?"

1. Forecast weather (next 3 days) as features.
2. Geostationary fire data to cover the timing blind spot.
3. Separate day +1/+2/+3 forecasts and confidence intervals (conformal prediction).
4. Evaluate on more seasons as they come in, especially a busy one.

## 5. Code you must be able to explain line by line

| File | What to know |
| --- | --- |
| `parali/forecast/features.py` | `rolling()` vs `forward()`; why seasons are cells x days matrices; how `adjacency()` builds neighbour sums with a sparse matrix; why `history_rate` uses only earlier seasons |
| `parali/forecast/train.py` | `split()` by season; early stopping on 2024; class-weight trial; `IsotonicRegression`; `explain()` with `pred_contrib` |
| `parali/forecast/metrics.py` | `top_share_by_day()`: per day, take the top 10% of cells, count hits and fires |
| `parali/forecast/compare.py` | how the bootstrap resamples whole days; how the ablation removes feature groups |
| `parali/pipeline/firms.py` | cleaning rules: drop low confidence, keep `type = 0`, keep the season, map to H3 |
| `parali/app/store.py` | how the API finds a cell's reasons and 14-day history |

Practice: open `features.py`, close your notes, and explain `rolling()` and `forward()` with a
5-day example on paper. Do the same for `top_share_by_day()`.

## 6. Limitations to state before they are pointed out

- Predicts **satellite-detected** fires; some burning may be missed (see section 4).
- The test season was much quieter (4.9% fire rate) than training seasons (14-21%).
- Weather is observed only, not forecast.
- District boundaries are from an older release (no Malerkotla, for example).
- Hyperparameters were tuned lightly by hand on validation, not by a full search.

Saying these first shows you understand your own system.

## 7. Phase 2: the assistant, agents and MCP

### Pitch addition (20 seconds)

> On top of the forecast I built a document assistant for officials and farmers. It answers in
> English or Hindi from CAQM directions, PIB releases, Haryana scheme notes and ICAR bulletins,
> always with a page citation, and says "not in my documents" when it should. Adding a multilingual
> reranker took the right page from rank 1 in 61% of questions to 95%. I also built LangGraph agents
> that write a daily district bulletin, and an MCP server so any AI assistant can query the forecast.

### Numbers to remember

| What | Value |
| --- | --- |
| Evaluation set | 50 questions: 38 answerable (10 Hindi), 12 unanswerable, gold pages marked |
| Recall@1: BM25 / vectors / hybrid / + reranker | 0.55 / 0.47 / 0.61 / **0.95** |
| Hindi recall@1 with English vs multilingual reranker | 0.10 vs **1.00** |
| Best chunking | Fixed 180-word windows (0.95) beat structure (0.89) and semantic (0.87) |
| Answer/abstain cut-off | 47 of 50 classified correctly |
| Small local LLM (1.5B) answers passing guardrails | 1 of 37 |
| Chronos-Bolt zero-shot vs LightGBM (validation PR-AUC) | 0.286 vs 0.351 |

### Questions and answers

**"Why hybrid search? Why not just embeddings?"** Keyword search finds exact names and numbers
("Direction No. 92", "Rs 2.32") that embeddings blur; embeddings find meaning and cross languages,
which keyword search can't (Hindi recall@1 0.20). Reciprocal rank fusion combines the two lists by
rank, so I don't have to calibrate a BM25 score against a cosine similarity. Hybrid recall@5 was
0.92 vs 0.79 and 0.76 for each alone.

**"What does the reranker do and why did it help so much?"** Embedding search encodes the question
and passage separately; a cross-encoder reads them together and scores relevance directly, which is
much more accurate but too slow for the whole corpus. So: retrieve 10 candidates cheaply, rerank
them precisely. It must be multilingual: an English reranker dropped Hindi recall@1 to 0.10.

**"Why did fixed-size chunks beat 'smarter' chunking?"** The documents are OCR'd scans and printed
web pages. Headings merge into text and clause numbers appear mid-line, so heading detection and
sentence-similarity splitting cut in the wrong places. Overlapping windows don't depend on layout.
I'd revisit this with clean digital documents.

**"How do you stop hallucinations?"** Four layers: (1) answer only above a relevance cut-off,
otherwise say "not found"; (2) the prompt allows only the numbered sources and requires [n]
citations; (3) every number in the answer must appear in a cited source; (4) if any check fails,
fall back to quoting the source sentences. The small model's rejected drafts swapped Punjab's and
Haryana's brick-kiln counts and changed target dates: the citation check caught them. The number
check alone wouldn't have caught the swap, because both numbers are in the sources. That's a
limitation I'd fix with claim-level verification (check each sentence against one passage).

**"How did you evaluate a RAG system?"** Separately at each stage: retrieval with recall@k and MRR
against hand-marked gold pages; abstention with unanswerable questions, including hard negatives
that sound covered; answers with citation accuracy, key-fact recall and number support. Because
retrieval and generation are evaluated separately, I can tell which stage failed.

**"How did you handle scanned PDFs?"** Two had no text and three had a broken OCR layer. I run
Tesseract once and save a text sidecar that ingestion prefers, strip repeated headers and footers,
and drop chart-axis noise. Clean text mattered as much as model choice.

**"Why are bulletin numbers written by code, not the LLM?"** The analyst step computes every
figure; the LLM may only phrase them, must pass a JSON schema, and every number must exist in the
facts. A small model still produced contradictions and broken Hindi, so the default is a
deterministic template and the LLM writer is opt-in.

**"What is MCP and why add it?"** Model Context Protocol is a standard way for AI assistants to
call external tools. Exposing get_risk, explain_cell, daily_bulletin and search_schemes means any
MCP client can answer "where should Sangrur's teams go today?" with live project data.

**"You compared a foundation model. What happened?"** Chronos-Bolt, zero-shot, beat the rule that
uses the same information (a cell's own history) but lost to LightGBM (+0.068 PR-AUC, CI above
zero) because it sees one cell at a time and can't use neighbours, the second-strongest signal.

### Limitations to state first

- The knowledge base is thin on farmer scheme details; some figures come from an unverified summary,
  and the assistant labels them that way.
- The abstention cut-off was tuned on the same 50 questions.
- The 1.5B local model can't follow the citation format; quality answers need a larger model.

## 8. Phase 3: satellite images, voice, and a fine-tuned small LLM

### Pitch addition (20 seconds)

> I also fine-tuned NASA and IBM's Prithvi geospatial foundation model to map burn scars from
> satellite images. On NASA's labelled dataset it beats a U-Net and, more importantly, beats the
> same network trained without pre-training by 5 IoU points, so I can show what pre-training is
> worth. When I applied it to Punjab it failed: harvested fields look burned to a model trained on
> US wildfires. I measured that failure instead of hiding it. Farmers can also ask by voice in
> Hindi, and I LoRA-fine-tuned a 0.5B model to answer with citations.

### Numbers to remember

| What | Value |
| --- | --- |
| Burn-scar IoU: Prithvi fine-tuned / scratch / U-Net / frozen | **0.758** / 0.706 / 0.700 / 0.627 |
| Gain from pre-training (paired bootstrap) | +0.052 IoU, 95% CI +0.029 to +0.084 |
| Training cost | 15-40 min per model on a 2-core CPU; 0.3 s per 512 x 512 scene |
| Punjab: share of post-harvest land flagged as burned | 81-96% (domain shift) |
| Hindi speech WER (FLEURS, 100 utterances): tiny / base / small | 1.06 / 1.04 / **0.57** (CER 0.26) |
| LoRA 0.5B: answers passing guardrails, base -> fine-tuned | 0/37 -> **19/37** (1.5B base: 1/37) |
| LoRA cost | 8.8 M trainable params (1.75%), 80 synthetic examples, 74 min on CPU |

### Questions and answers

**"Why a foundation model for burn scars?"** Labelled satellite data is scarce and expensive.
Prithvi was pre-trained on a huge unlabelled HLS archive by hiding 75% of image patches and
reconstructing them, so it already knows what fields, water, cloud and soil look like in these six
bands. Fine-tuning needs far fewer labels than training from scratch.

**"How do you know pre-training helped, and not just a bigger model?"** I trained the identical
architecture from random weights with the same data, steps and seed: 0.706 vs 0.758. A paired
bootstrap over the 264 validation scenes puts the gain at +0.03 to +0.08 IoU. The U-Net, with 6x
fewer parameters, matched the un-pre-trained transformer, so architecture alone bought nothing.

**"Why did the frozen encoder do worst?"** Its recall was 0.92 but precision 0.66: the general
features find dark, changed surfaces but can't separate burned ground from dark soil or shadow.
That last distinction needs the encoder itself to adapt, which is why full fine-tuning (with the
encoder at a 5x lower learning rate so it doesn't forget) won.

**"Did you tune on the validation set?"** No. Every model trains for a fixed 1,200 steps and is
evaluated once, with no early stopping or checkpoint picking. The dataset only has a train and a
validation split, so that was the only honest option.

**"The model fails in Punjab. Isn't that a problem for your project?"** It's the most useful result
in Phase 3. US training scenes show black scars in green or grey landscapes. In Punjab in November
every harvested field is brown stubble or bare soil, so the model calls 81-96% of the land burned.
I checked new scars between two images three days apart against VIIRS fires: only 1.15-1.25x better
than random. So it isn't on the dashboard. The fix is 200-300 labelled local fields and fine-tuning
on image pairs so the model learns the change, not the look. I also tried weak labels from dNBR
first and dropped them because they were barely better than random (1.1x).

**"You removed a dependency from NASA's code. How do you know you didn't break it?"** Two checks: a
unit test compares my attention block with PyTorch's reference implementation, and with the official
weights the model reconstructs masked patches of real images with loss 0.043, against 0.83 for
random weights. A wrong layer would destroy that.

**"How does voice work, and why not a text-to-speech model?"** Browser records, faster-whisper small
transcribes on CPU, the normal cited-answer pipeline answers, and the browser's built-in speech
synthesis reads the answer. Hindi TTS models are large, slow on CPU, or non-commercial (MMS-TTS is
CC-BY-NC); Android phones already ship good Hindi voices, at zero server cost.

**"WER 0.57 sounds bad."** It is high, and FLEURS is clean read speech, so real farmers will be
worse. But the transcript only has to retrieve the right document: the multilingual embedding and
reranker tolerate spelling noise, and the user sees and can edit the text before asking. The smaller
models were unusable: tiny wrote Hindi in Latin letters and looped, base mostly wrote Urdu script.
On a GPU I'd use large-v3-turbo.

**"Why fine-tune instead of prompting?"** I tried prompting first: a 0.5B model cited sources in 0
of 37 answers, a 1.5B in 1 of 37, even with an example in the prompt. Small models don't follow
format instructions reliably, but they can learn a format from examples, and LoRA does that cheaply:
8.8 M trainable parameters, 80 examples, 74 minutes on a laptop CPU.

**"Where did the training data come from?"** Generated and filtered automatically: a 1.5B model
wrote questions for document chunks (doc2query), I kept only questions where the retriever finds
the source chunk in its top 3 (round-trip filter), the targets are grounded quote answers that pass
the guardrails by construction, and negatives are pairs the reranker confirms are irrelevant. All
chunks on the evaluation's gold pages were excluded, so the test questions stay unseen.

**"Did it work?"** Partly, and I can say exactly which part. Answers passing the guardrails went
from 0 to 19 of 37, with no missing citations and 17 of 19 citing the right page. But the model
learned what its teacher did, copying sentences, and chooses them worse than the ranker that made
its targets, so key-fact recall fell from 0.79 to 0.61. It also refuses too often in Hindi. So it's
not the default. The lesson: fine-tuning transfers format easily, but quality is capped by the
teacher. The next step is distillation from a strong model with the same guardrails as a filter.

**"What went wrong on the way?"** Three things, all in the DEVLOG. The model learned to repeat
sentences because overlapping chunks put duplicate sentences in 14 of 80 targets; I fixed the
de-duplication in the live assistant too. Training ran out of memory: full-vocabulary logits for
3,300 tokens are about 2 GB, so I compute logits only for the answer tokens and checked that the loss
is identical. And attention memory grows with the square of the length, so I measured it and capped
examples at 2,304 tokens.

### Limitations to state first

- Burn scars: no Punjab labels, so the in-domain number (0.758) says nothing about Punjab.
- Voice: tested on read speech in standard Hindi only; no Punjabi.
- The fine-tuned model is a format experiment, not the default: its answers carry fewer key facts
  than plain quotes, and it was evaluated on only 37 questions.

## 9. Phase 4: running it like a product, judging answers, preference tuning

### Pitch addition (20 seconds)

> Then I made it operable. Every training run and model is tracked in MLflow with a
> champion/challenger rule, drift is monitored per season and per week, every answer is traced
> step by step, and users' thumbs-downs become new test cases (the first one exposed a real bug).
> I verified the Docker image as a non-root user, which exposed another bug, and packaged a
> one-command public demo. On the LLM side I added an LLM-as-judge with a check on how far the
> judge can be trusted, and DPO preference tuning, which fixed the small model's wrong refusals
> but overshot, a clear lesson about what preference data really specifies.

### Numbers to remember

| What | Value |
| --- | --- |
| MLflow | 30 runs in 5 experiments; registered model with `champion` alias |
| 2025 drift | 14 of 26 features major (PSI > 0.25), score PSI 0.62, yet lift 5.4x and calibration within 10% |
| Early warning | Weekly score PSI 0.83 in the first week of 2025 (2024: 0.08) |
| Slowest step per answer | Reranker, ~5 s on 2 CPU cores |
| Feedback-found bug | Quotes cut at "w.e.f."; fix raised quote-mode fact recall 64% -> 68% |
| DPO | Guardrail-passing answers 19 -> 37 of 37, wrong refusals 17 -> 0, correct refusals 3 -> 0 of 3 |
| LLM judge (Phi-3.5-mini) on verbatim quotes | 35 of 37 called supported (faithfulness verdicts usable) |
| Judge vs key-fact check, relevance | Kappa 0.18-0.53; too lenient on fluent answers (13 misses on DPO answers) |

### Questions and answers

**"Your inputs drifted massively. Why didn't you retrain?"** Because drift is a symptom, not a
verdict. I tie actions to impact: retrain only if lift falls more than 25% below validation,
recalibrate if predicted rates are off by more than 30%. In 2025 the fire counts fell by 80%+ but
fell everywhere together, so the ranking held (lift 5.4x vs 4.1x on validation) and probabilities
were within 10%. The action is to investigate the satellite data source, which matches the
daytime-detection drop found in Phase 1. Retraining on a drift alarm alone would have been busywork.

**"How do you detect drift before you have labels?"** Prediction drift. Score PSI needs no
outcomes. I compare each week with the same weeks of past seasons, because comparing October
with the whole season flags the normal start of burning as drift (my first version did exactly
that). With that fix, 2025 is flagged in its first week.

**"What is a champion/challenger rule and why only validation?"** A new model version becomes
the served one only if it beats the current one on the validation season. The test season is
for the final report; choosing with it would leak it into model selection.

**"Why build your own tracer instead of using a library?"** It's about 150 lines, needs no service
to run, and uses OpenTelemetry's field names, so moving to Jaeger or Langfuse later only changes
the exporter. It showed the reranker is the bottleneck, which is where optimisation should go
(a smaller reranker, or caching).

**"Give an example of the feedback loop working."** The first down-voted answer began with a
broken quote, "...for a period of one-year w.e.f. [1]". The splitter treated the full stop in
"w.e.f." as a sentence end. That abbreviation is all over Indian government notices. Fixing it
improved quote-mode fact recall from 64% to 68%, and the case is now a unit test.

**"How do you know an LLM judge is right?"** You measure it like any model. I graded answers with
Phi-3.5-mini, deliberately a different model family from the Qwen answer models to avoid
self-preference, and compared its verdicts with labels I trust: quote-mode answers are copied
word for word, so a good judge must call them supported, and "answers the question" was compared
with the key-fact check using Cohen's kappa. Result: it called 35 of 37 verbatim quotes
supported, so its faithfulness verdicts are usable. But on relevance kappa was only 0.18-0.53, and
it was most often fooled by fluent answers that missed the fact. Once it even claimed an answer
stated a grant amount that wasn't there. So it's a screening tool; the fact check stays the metric.

**"What is DPO and what happened?"** Direct Preference Optimisation trains on pairs (better
answer, worse answer for the same prompt) against a frozen reference model, with no reward model.
I built pairs to stop wrong refusals: grounded answer over "not found" when the passages answer,
and the reverse when they are off-topic, balanced 1:1. Wrong refusals went from 17 to 0 and every
answer passed the guardrails. But it also stopped refusing the 3 unanswerable questions it used to
refuse correctly, because my refusal pairs used obviously off-topic passages while real failures
are near misses. The pairs are the specification: next time, hard negatives.

**"DPO on a CPU?"** The DPO gradient factorises into a weight times the difference of two
gradients. I compute the weight without gradients, then back-propagate the chosen and rejected
answers one at a time, so memory equals plain fine-tuning. A unit test shows it equals the
autograd gradient of the DPO loss.

**"What broke when you tested Docker properly?"** Running as a non-root user, as Hugging Face
Spaces do, made model downloads fail: libraries write caches under HOME, which was `/`. Setting
HOME and the cache paths inside a writable folder fixed it. Tests that only run as root would
never have caught it.

**"Why does the dashboard look like this?"** It's a monitoring tool, so it uses a dark
"control room" layout where the data is the brightest thing on screen. Colour has one job each:
lime only for interface state (selection, focus, the current day), one orange ramp for risk
(brighter = riskier, checked for monotone lightness and contrast on the dark map), and status
colours with text labels for alert levels. Time is the main control, so a season timeline with
play sits above the map. Still plain JavaScript, three files, no build step.

### Limitations to state first

- The Hugging Face Space and CI need the owner's accounts to go live; both are scripted and
  tested locally.
- The judge is a 3.8B model on 4-bit weights; it is a screening tool, not ground truth.
- DPO and LoRA used 60-80 synthetic pairs; the effects are clear but the sample is small.

## 9b. Live mode: "does it work today, or only on old data?"

**Q: Your test season is 2025. Is this just a replay?**
No. With a free NASA key the app fetches this season's near-real-time fire points (same VIIRS
sensor as training) and forecast weather, builds features with the exact training code, and
scores today. It rebuilds every 3 hours. 2025 stays as the replay because it is the only season
with a complete, untouched test.

**Q: How do you evaluate a season that hasn't finished?**
A forecast for day d is about d+1..d+3, so its label exists only once data for d+3 does. Days up
to (last data - 3) are scored; the last three are shown as "not known yet" rather than as zero
fires, which would be a silent label error. 2026 so far: 0.227 vs 0.144 PR-AUC against the best
simple rule.

**Q: 72.6% capture in 2026 vs 36.8% in 2025: did the model get better?**
No, and I would not claim that. Early October fires are few and concentrated (two districts have
half of them), so ranking the top 10% is easier. The comparable signal is the margin over the
rule on the same days, which held. That is the online-evaluation habit: compare against a
baseline on identical data, not against last year's headline.

**Q: What breaks in production that didn't in the notebook?**
API limits (FIRMS allows 5 days per request: windowed fetches, caching of settled windows),
outages (keep serving the last good build and expose the error), and caches (build-versioned
URLs so browsers never show an old forecast as new). The key is a secret in `.env` or a Space
secret, never in git.

## 10. Questions to ask back

- "How do you handle distribution shift between training and production data in your team?"
- "Do you evaluate models on time-based splits?"
- "How do you decide when a simpler model is good enough?"
