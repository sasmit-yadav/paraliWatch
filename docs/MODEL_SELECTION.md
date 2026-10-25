# Model selection: what we tested, what won, and why

**Decision:** the forecast uses **LightGBM (tuned)**. It had the best validation score of any single
model, but the honest finding is bigger than that: every machine-learning model we tried landed
within a few thousandths of each other, while all of them beat the best hand-written rule by a
wide, statistically clear margin. The features carry the signal; the algorithm is a secondary
choice. LightGBM wins that secondary choice on speed, exact explanations and simplicity.

All numbers come from `eval/model_comparison/` and can be reproduced with
`python -m parali.forecast.compare <model>` (see the end of this page).

## 1. The setup every model shared

- **Task:** for each H3 cell (~36 km2) in Punjab and Haryana and each day, predict whether NASA
  VIIRS will detect at least one fire in that cell in the next 3 days.
- **Same 26 features** for every model (listed in `parali/forecast/features.py`). Each feature for
  day *t* only uses data available by the end of day *t*.
- **Same split by season**, never random rows (random rows would let the model peek at the same
  week's fires):
  - train: 2020-2023 seasons (805,240 cell-days)
  - validation: 2024 season (201,310 cell-days): used for early stopping and **for choosing the model**
  - test: 2025 season (201,310 cell-days): reported for completeness, **never used to choose**
- **Main metric: PR-AUC** (average precision). Fires are rare (8.6% of cell-days in 2024, 4.9% in
  2025), and PR-AUC rewards putting real fire cells at the top of the list, which is exactly how
  officials would use the forecast. A random guess scores PR-AUC equal to the fire rate.
- **Practical metric: fire capture @10%**: if officials watch only the top 10% of cells each day,
  what share of the next 3 days' fires fall inside those cells?

## 2. What we tested

| Family | Model | Why it was in the race |
| --- | --- | --- |
| Rules (no learning) | Fires in the cell, last 3 days | "Fire spreads where fire already is" |
| Rules | Fires in and around the cell, last 3 days | Adds the neighbours |
| Rules | Same cell, same days last year | Farmers burn at a similar time every year |
| Rules | Cell's average fires per past season | Some places always burn more |
| Linear | Logistic regression (log-scaled counts, standardised, balanced classes) | Simplest learned model; shows how much non-linearity adds |
| Neural network | MLP, 2 hidden layers (64, 32), early stopping | A deep-learning baseline on tabular data |
| Bagging | Random forest, 200 trees, depth 14, balanced | Classic strong tabular model |
| Boosting | scikit-learn HistGradientBoosting | Boosting without extra dependencies |
| Boosting | XGBoost (hist, depth 6, class weight, early stopping) | Industry-standard boosting |
| Boosting | LightGBM, first settings (lr 0.05, 63 leaves) | Our first model |
| Boosting | **LightGBM, tuned** (lr 0.02, 15 leaves, min 1,000 rows per leaf, class weight) | Tuned on validation only |
| Ensemble | Mean percentile rank of LightGBM, MLP, XGBoost, logistic regression, random forest | Does combining help? |

Linear models and the neural network got log-transformed, standardised inputs because fire
counts are heavy-tailed (most cells 0, a few over 100). Tree models used raw values; they don't
need scaling.

## 3. Results

Sorted by validation PR-AUC (the score used for choosing).

| Model | Valid PR-AUC | Valid capture @10% | Test PR-AUC | Test ROC-AUC | Test capture @10% | Fit time |
| --- | --- | --- | --- | --- | --- | --- |
| Ensemble (mean rank of 5) | 0.353 | 43.4% | 0.266 | 0.850 | 36.9% | sum of members |
| **LightGBM, tuned (chosen)** | **0.351** | **43.3%** | **0.266** | **0.849** | **36.8%** | 21 s |
| MLP neural network | 0.349 | 41.3% | 0.235 | 0.835 | 35.3% | 33 s |
| XGBoost | 0.344 | 42.0% | 0.258 | 0.846 | 36.4% | 10 s |
| Logistic regression | 0.343 | 42.0% | 0.248 | 0.836 | 35.1% | 2 s |
| LightGBM, first settings | 0.341 | 42.8% | 0.260 | 0.845 | 36.4% | 9 s |
| Random forest | 0.341 | 43.3% | 0.268 | 0.851 | 36.9% | 51 s |
| HistGradientBoosting | 0.337 | 41.9% | 0.237 | 0.838 | 34.8% | 54 s |
| Rule: fires in and around the cell | 0.242 | 35.0% | 0.191 | 0.748 | 33.6% | - |
| Rule: same days last year | 0.234 | 34.4% | 0.093 | 0.608 | 23.3% | - |
| Rule: fires in the cell | 0.167 | 29.6% | 0.107 | 0.602 | 25.7% | - |
| Rule: past-season average | 0.166 | 28.5% | 0.084 | 0.692 | 20.6% | - |

Fire rate (what a random guess scores): 0.086 in validation, 0.049 in test. Fit times are on a
2-core machine.

## 4. Are the differences real? (bootstrap)

A gap of 0.002 could be luck. We resampled whole **days** of the validation season 300 times
(days, not rows, because fires cluster in time) and recomputed the PR-AUC difference each time.

| Comparison | Mean PR-AUC gain | 95% interval | Times A won |
| --- | --- | --- | --- |
| LightGBM vs best rule (neighbourhood) | +0.112 | +0.067 to +0.157 | 100% |
| LightGBM vs logistic regression | +0.009 | -0.009 to +0.030 | 81% |
| LightGBM vs MLP | +0.003 | -0.034 to +0.044 | 53% |
| Ensemble vs LightGBM | +0.002 | -0.008 to +0.010 | 67% |

**Reading:** learning beats rules, clearly. Among the learned models, the intervals include zero,
so no model is proven better than the others.

Retraining LightGBM with three different random seeds gave validation PR-AUC 0.3507, 0.3516 and
0.3535: seed noise alone is about +/-0.003, the same size as the gaps between the top models.

## 5. Which features matter (ablation)

Retrain the chosen LightGBM with one group of features removed (validation season):

| Removed group | Features left | Valid PR-AUC | Change |
| --- | --- | --- | --- |
| nothing | 26 | 0.351 | - |
| past seasons (last year's fires, history) | 22 | 0.322 | **-0.029** |
| the cell's own recent fires | 19 | 0.340 | -0.011 |
| neighbouring cells' fires | 22 | 0.343 | -0.008 |
| calendar (day of season) | 25 | 0.346 | -0.004 |
| weather | 19 | 0.348 | -0.002 |
| district and state totals | 23 | 0.351 | 0.000 |

- **Memory of past seasons is the strongest signal.** The same fields burn at roughly the same
  time each year.
- **Weather barely helps** in its current form. We only use weather observed up to today; rain
  *forecast* for the next 3 days is a likely improvement (planned).
- **District and state totals add nothing.** They stay for now because they make the dashboard's
  explanations readable, but they are the first candidates to drop.

## 6. Why LightGBM, not the others

| Option | Why not chosen |
| --- | --- |
| Ensemble | +0.002 on validation, not significant, but needs 5 models to train, ship and explain. Not worth it. |
| MLP neural network | Close on validation, but weaker on test (0.235) and less stable; explanations need slower approximate SHAP. |
| Random forest | Best single test score (0.268), but we choose on validation, where it ranked 6th. Picking it now would be choosing by the test set. Also 2.5x slower to train and a much larger model file. |
| XGBoost | Close, but no gain over LightGBM and one more dependency. |
| Logistic regression | Very respectable (and a useful sanity check), but trails on test (0.248) because it can't capture interactions such as "fires nearby *and* harvest week". |
| HistGradientBoosting | Lowest of the learned models. |

LightGBM also gives us:

- **Exact SHAP explanations for free** (`pred_contrib=True`, TreeSHAP), which the dashboard uses to
  show "why is this cell high-risk?".
- **Fast training** (~20 s) and a small model file, so retraining every week is cheap.
- **Raw inputs** with no scaling or log transforms to keep in sync between training and serving.

### How the tuned settings were found

The first LightGBM (lr 0.05, 63 leaves) stopped after only 25 trees: it learned the high-fire
seasons of 2020-2021 too quickly and did not carry over to 2024, when fires were far fewer. A
slower, smaller, more regularised model (lr 0.02, 15 leaves, at least 1,000 rows per leaf) trained
for ~260 trees and scored better on validation. We also compared plain training with positive-class
weighting (fires are ~1 in 8 cell-days in training): weighting won slightly (0.3507 vs 0.3452) and
was kept. Every one of these choices was made on the 2024 season only.

## 7. Limits of this comparison

- Model settings for the non-LightGBM models are sensible defaults, lightly chosen, not exhaustively
  tuned. A full hyperparameter search for every model could reorder the top few, but the bootstrap
  says the top few are already within noise of each other.
- Fire counts fell about 16x from 2021 to 2025, so every model is tested on a season quieter than
  anything it trained on. Results on a busy season could differ.
- Time-series foundation models were compared separately in section 10.

## 8. Reproduce

```bash
python -m parali data                       # builds data/processed/panel.parquet
pip install xgboost                         # only needed for the XGBoost entry
for m in baseline_persistence_3d baseline_neighbourhood_3d baseline_same_days_last_year \
         baseline_history_rate logreg mlp random_forest hist_gb xgboost lightgbm_default lightgbm_tuned; do
  python -m parali.forecast.compare $m
done
python -m parali.forecast.compare ensemble
python -m parali.forecast.compare bootstrap lightgbm_tuned mlp
python -m parali.forecast.compare ablation
python -m parali.forecast.compare table     # writes eval/model_comparison/summary.csv
```

## 9. Calibration: isotonic vs Platt scaling

Model scores are ranked well but are not probabilities (raw Brier score on 2025: 0.074). We map
them to probabilities with a calibrator fitted on the 2024 season, and compared two standard
methods by fitting on alternate days of 2024 and scoring the other days (so the choice again uses
validation data only):

| Method | Brier (cross-fitted, 2024) | Log loss (cross-fitted, 2024) | Distinct output values (2025) |
| --- | --- | --- | --- |
| **Isotonic regression (chosen)** | **0.0642** | **0.2226** | 170 |
| Platt scaling (logistic on log-odds) | 0.0648 | 0.2239 | 4,557 |

Isotonic is slightly better and was kept. Its one drawback is visible in the dashboard: it maps
nearby scores to the same probability, so several top cells can all show "45%". Ranking (the top
list, the "Top 1%" figure, fire capture) therefore uses the raw score, and the calibrated value is
only shown as the chance of fire. On the 2025 test season calibration cut the Brier score from
0.074 to 0.041, and the predicted and observed fire rates match within about a percentage point in
every band (see `eval/RESULTS.md`).

## 10. Time-series foundation model: Chronos-Bolt, zero-shot

Foundation models for time series are pre-trained on large collections of public series and can
forecast a new series without any training. We tested Amazon's
[Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-small) (tiny, 9M parameters; small, 48M)
with **no training on our data**. For each cell and day it reads that cell's daily fire counts
(the whole previous season, then the current season up to that day) and forecasts the next 3 days.
The forecast sum is the risk score, scored exactly like every other model.
Code: `parali/forecast/foundation.py`.

| Model | Valid PR-AUC | Test PR-AUC | Test capture @10% | Run time (2 CPU cores, both seasons) |
| --- | --- | --- | --- | --- |
| LightGBM, tuned (chosen) | 0.351 | 0.266 | 36.8% | ~20 s training |
| Chronos-Bolt small, mean forecast | 0.286 | 0.154 | 30.6% | ~10 min inference |
| Chronos-Bolt small, 90th percentile | 0.278 | 0.152 | 30.4% | (same run) |
| Rule: fires in and around the cell | 0.242 | 0.191 | 33.6% | - |
| Chronos-Bolt tiny, 90th percentile | 0.227 | 0.135 | 29.8% | ~2 min inference |
| Rule: fires in the cell | 0.167 | 0.107 | 25.7% | - |

Bootstrap on the 2024 season (whole days resampled, 300 times):

| Comparison | Mean PR-AUC gain | 95% interval |
| --- | --- | --- |
| Chronos-Bolt small vs rule "fires in the cell" | +0.120 | +0.087 to +0.153 |
| Chronos-Bolt small vs rule "fires in and around the cell" | +0.044 | +0.009 to +0.077 |
| LightGBM vs Chronos-Bolt small | +0.068 | +0.027 to +0.110 |

**What this shows:**

- **Zero-shot works, up to a point.** Without seeing any of our data, Chronos-Bolt small clearly
  beats the rule that uses the same information (the cell's own history), and on validation it
  even beats the neighbourhood rule.
- **It can't see neighbours.** Chronos reads one cell's series at a time. Our ablation shows
  neighbouring fires and last year's fires are the strongest signals; a single-series model gets
  only part of the second and none of the first. That is why the tuned tabular model wins by
  +0.068 PR-AUC, with an interval well above zero.
- **It generalised worse to the quiet 2025 season** (test 0.154, below the neighbourhood rule).
- **Bigger helped:** small (48M) beat tiny (9M) by about 0.06 on validation.
- **Cost:** ~10 minutes of CPU inference for two seasons, versus ~20 seconds to train LightGBM and
  milliseconds to predict.

**Decision:** keep LightGBM. A useful next experiment would be to feed the Chronos forecast into
LightGBM as one more feature, or to fine-tune Chronos on these series, so the foundation model's
view of each cell's own history is combined with the spatial features.

## 11. Burn-scar segmentation: Prithvi-EO-2.0 vs U-Net (Phase 3)

Four models, same data (NASA HLS Burn Scars, 540 training / 264 validation scenes), same 1,200
steps, same augmentation and loss, no checkpoint selection on validation. Burn-class IoU:

| Model | IoU (95% CI) | Trained params |
| --- | --- | --- |
| **Prithvi-EO-2.0-tiny, pre-trained, fine-tuned** | **0.758 (0.706-0.802)** | 6.2 M |
| Prithvi architecture, random init | 0.706 (0.645-0.758) | 6.2 M |
| U-Net from scratch | 0.700 (0.646-0.749) | 1.1 M |
| Prithvi pre-trained, encoder frozen | 0.627 (0.571-0.682) | 0.6 M |

Chosen: fine-tuned Prithvi; paired bootstrap +0.05 IoU over both baselines, CI above zero. It does
not transfer to post-harvest Punjab without local labels. Full write-up: [BURN_SCARS.md](BURN_SCARS.md).

## 12. Speech recognition for Hindi questions (Phase 3)

faster-whisper on 100 FLEURS Hindi utterances (CPU, int8): tiny WER 1.06, base 1.04 (mostly Urdu
script), **small 0.57** (CER 0.26, chosen). Text-to-speech: the browser's built-in voices, not a
model (no licence, no server cost). Details: [VOICE.md](VOICE.md).

## 13. Small LLM: prompting vs LoRA fine-tuning (Phase 3)

Answers passing the citation and number guardrails (37 answerable questions): Qwen2.5-0.5B base
0/37, 1.5B base 1/37, **0.5B + LoRA 19/37**. But the fine-tuned answers carry fewer key facts than
plain quotes (fact recall 0.61 vs 0.79), so quote mode stays the default. Details:
[FINE_TUNING.md](FINE_TUNING.md).

## 14. Preference tuning: SFT vs SFT + DPO (Phase 4)

| Qwen2.5-0.5B variant | Guardrail-passing answers | Wrong refusals | Correct refusals (3 near-miss unanswerable) | Key facts vs quotes on same questions |
| --- | --- | --- | --- | --- |
| LoRA (SFT) | 19 / 37 | 17 | 3 / 3 | 12 vs 20 |
| LoRA + DPO | 37 / 37 | 0 | 0 / 3 | 26 vs 38 |

DPO removed the failure it was given pairs for and overshot on the opposite one. Neither replaces
quote mode as the default. Details: [FINE_TUNING.md, section 5](FINE_TUNING.md).

## 15. Choosing the judge model (Phase 4)

Phi-3.5-mini (3.8B, MIT) was chosen over a bigger Qwen because a judge from the *same family* as
the answering models tends to favour its own style, and because it fits in memory next to nothing
else on a 2-core, 6 GB machine. Meta-evaluation: 95% of verbatim quotes judged supported
(faithfulness usable), relevance agreement with the key-fact check kappa 0.18-0.53 (not usable as
the headline metric). Details: [RAG.md, section 5](RAG.md).
