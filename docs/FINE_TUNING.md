# Fine-tuning a small LLM to answer with citations (Phase 3, LoRA)

Phase 2 found that small local models can't follow the assistant's rules. Qwen2.5-1.5B passed the
guardrails (every fact cited as [n], every number present in a cited source) on **1 of 37**
answerable questions. Qwen2.5-0.5B passed on **0 of 37**: its drafts were often reasonable but
never had a single [n] citation. So the assistant falls back to quoting source sentences, which is
safe but not a written answer.

Question: can a cheap LoRA fine-tune on CPU teach a 0.5B model the format (cite every sentence,
copy numbers exactly, say "not found" when the passages don't answer)?

## 1. Training data, with no human labelling and no paid API

`python -m parali.tune.data select | questions | build` (three stages, so the LLM server and the
retrieval models never share the 6 GB memory limit):

1. **Hold out the evaluation.** Only chunks that are *not* on any gold page of the 50 evaluation
   questions, and not near-duplicates of one (cosine < 0.9), are used: 23 chunks.
2. **doc2query.** A local Qwen2.5-1.5B writes up to 5 questions each chunk answers: 114 questions;
   about 30% (23) are then translated into Hindi.
3. **Round-trip filter.** A question is kept only if the real retriever puts its source chunk in the
   top 3: 68 kept. This removes vague or wrong questions without reading them by hand.
4. **Targets.** The answer is the grounded quote answer (the 3 sentences closest to the question,
   each with its [n]), so every target passes the guardrails by construction. Overlapping chunks
   produced duplicate sentences in 14 of the first 80 targets, and the first trained model learned
   to repeat itself; duplicates are now removed in `_extractive` (which also fixed 5 of 50
   quote-mode answers in the live assistant) and the data was rebuilt.
5. **Negatives.** For about a third of questions, the same question with passages from another
   topic, *verified irrelevant by the reranker*, and the target "I could not find this in the
   documents...": 12 examples.

Result: 80 examples (`data/tune/train.jsonl`, statistics in `data/tune/stats.json`). The prompt is
built by `answer.build_messages`, the same function used at inference, so training and serving
can't drift apart.

## 2. Training

| Setting | Value | Why |
| --- | --- | --- |
| Base model | Qwen2.5-0.5B-Instruct (Apache-2.0) | Runs on any laptop CPU |
| Method | LoRA, rank 16, alpha 32, dropout 0.05 | Adapter of 8.8 M parameters (1.75% of 503 M); 34 MB on disk |
| Target layers | q, k, v, o, gate, up, down projections | Attention + MLP; format learning needs both |
| Loss | Answer tokens only | The model should learn to answer, not to reproduce the prompt |
| Optimiser | AdamW, lr 2e-4, one-cycle, grad-clip 1.0 | Standard LoRA settings |
| Batch | 1 sequence, gradient accumulation 4 | No padding waste on CPU |
| Epochs | 2 | Small data; loss flattens in the second epoch |
| Max length | 2,304 tokens (69 of 80 examples fit) | See below |

Why not QLoRA (4-bit base)? bitsandbytes 4-bit needs a GPU. On CPU the base stays in float32 and
only the adapter trains, which is what QLoRA would do minus the quantisation.

**Two memory problems, solved properly:**
- A normal forward pass builds logits for every position over the 151,936-token vocabulary:
  3,300 x 151,936 floats is about 2 GB, plus its gradient. `answer_loss` runs the transformer, then
  applies the output layer only to the ~100 answer positions. Checked to give the same loss as the
  library's own loss (9.168901 vs 9.168902 on a test example).
- Attention memory grows with the square of the length: measured peak 3.5 GB at 1,000 tokens,
  4.5 GB at 2,000, over 6 GB at 3,300 even with gradient checkpointing. So examples are capped at
  2,304 tokens.

Training loss (answer tokens): 0.60 after 5 steps, 0.18 at the end of epoch 1, 0.08 at the end of
epoch 2 (`eval/tune/train_log.json`); 74 minutes on 2 CPU cores.

## 3. Evaluation

Same 50 questions, same retrieved passages (cached), same guardrails as Phase 2; only the model
behind the OpenAI-compatible endpoint changes (`python -m parali.tune.serve` serves a Hugging Face
model with or without the adapter). Greedy decoding with a mild repetition penalty (1.1) for both.

Results on the 37 answerable questions (`eval/rag/answer_results_qwen2.5-0.5b-{base,lora}.json`):

| | Base 0.5B | **LoRA 0.5B** | Phase 2: base 1.5B |
| --- | --- | --- | --- |
| LLM answers accepted by the guardrails | 0 / 37 | **19 / 37** | 1 / 37 |
| Rejected: no citation | 37 | **0** | - |
| Rejected: number not in a cited source | 3 | **1** | - |
| Declined ("not found") although relevant passages were retrieved | 0 | 17 (8 of the 10 Hindi questions) | - |
| Accepted answers that cite a gold page | - | 17 / 19 | - |
| Key facts in the accepted answers | - | 12 / 32 | - |
| Same questions answered by quoting (no LLM) | - | 22 / 32 key facts | - |
| Overall fact recall (with quote fallback) | 0.79 | 0.61 | - |
| Median seconds per answer (2-core CPU) | 11 | 12 | - |

Unanswerable questions are decided by the retrieval cut-off before any LLM runs, so both models
give the same 3 false answers out of 12.

**What the fine-tune achieved:** the format. Every answer it gives has [n] citations (0 rejected
for missing citations, vs 37 of 37 for the base model), numbers are copied exactly (1 failure vs
3), and 17 of 19 accepted answers cite the right page. Format compliance went from 0% to 51%.

**What it did not achieve:** better answers. It learned what its teacher did, which is copy
sentences, and it picks sentences less well than the embedding ranker that wrote its targets
(12 vs 22 key facts on the same questions), sometimes including signatures and boilerplate. It
also over-learned "not found", especially for Hindi (8 of 10), and 4 of 19 answers still repeat a
sentence. So overall fact recall falls from 0.79 to 0.61.

**Decision:** the assistant's default stays as before: an LLM is optional, and without one, or
when its draft fails the checks, the user gets the quoted sources. The adapter is kept as a
documented experiment (`models/answer_lora/`, not committed; rebuild with the commands below).

**What would make it work**, in order:
1. A better teacher. Targets written by a strong model (7B+ local, or a hosted API) that
   *synthesises* a short answer, kept only if it passes the same guardrails (distillation with
   verification). The format has been shown to be learnable; the content needs a better example.
2. More and more varied data: 23 chunks is narrow. Use all documents, and evaluate on questions
   whose documents were never seen in training.
3. Hindi targets in Hindi (now they are English quotes with a Hindi preface), and fewer negatives
   for Hindi.

## 4. A judge model's view

The answers were also graded by an LLM judge (Phi-3.5-mini, a different model family), see
[RAG, section 5](RAG.md). It agrees with the direction of the results above: it says
"answers the question" for 32 of 37 quote answers, 27 of 37 for LoRA (with quote fallbacks) and
23 of 37 for LoRA + DPO. But it is too lenient on fluent answers that miss the key fact (13 such
cases for DPO), so the fact check stays the headline metric.

## 5. Preference tuning with DPO (Phase 4)

**The problem it targets.** The LoRA model's most common failure was refusing questions its
passages did answer (17 of 37). It did refuse correctly, though: for the 3 unanswerable
questions whose passages slipped past the retrieval cut-off, it said "not found" every time.
The goal was to keep the good refusals and remove the wrong ones.

**What DPO is.** Direct Preference Optimisation (Rafailov et al., 2023) trains on *pairs* of
answers to the same prompt: a preferred one (chosen) and a worse one (rejected). It raises the
probability of the chosen answer relative to the rejected one, measured against a frozen copy of
the starting model (the reference) so the model can't drift far from what it already knows:

```
loss = -log sigmoid( beta * [ (log p(chosen) - log ref(chosen)) - (log p(rejected) - log ref(rejected)) ] )
```

Unlike RLHF there is no separate reward model and no reinforcement-learning loop; the preference
data *is* the reward signal. `beta` (0.1 here) sets how far the model may move from the reference.

**Pairs, built automatically from the SFT prompts** (`python -m parali.tune.dpo pairs`):

| Prompt type | Chosen | Rejected | Pairs |
| --- | --- | --- | --- |
| Passages answer the question | the grounded, cited answer | "I could not find this..." | 68 |
| Passages are about something else | "I could not find this..." | quotes from those passages, cited as if they answered | 12 |

69 of the 80 fit the 2,304-token limit. They were **balanced 1:1** for training (30 answer pairs;
the 10 refusal pairs each used three times), so the model sees both directions equally.

**Training.** It starts from the SFT model (its LoRA merged into the base) with a fresh LoRA
adapter; the reference is that SFT model with the new adapter switched off. Settings: 1 epoch,
60 pairs, effective batch 4, learning rate 5e-5. It took 53 minutes on 2 CPU cores.

Memory: the DPO gradient factorises as `-beta * w * (grad log p(chosen) - grad log p(rejected))`
with `w = sigmoid(-beta * margin)`. The code computes `w` without gradients, then back-propagates
the chosen and the rejected answer one after the other, so only one sequence's activations are in
memory at a time. A unit test checks this gives exactly the gradient of the DPO loss.
Sanity check: the first step's loss is 0.6931 = ln 2, as it must be when the model equals the
reference. Training loss fell to 0.08, and the model preferred the chosen answer in 100% of the
last 8 pairs.

**Results** (same 50 questions, same retrieved passages, same guardrails):

| | LoRA (SFT) | **LoRA + DPO** |
| --- | --- | --- |
| LLM answers accepted by the guardrails | 19 / 37 | **37 / 37** |
| Wrongly refused (relevant passages retrieved) | 17 | **0** |
| Rejected for missing citation / unsupported number | 0 / 1 | 0 / 0 |
| Accepted answers citing a gold page | 17 / 19 | 35 / 37 |
| Hindi questions answered by the LLM | 1 / 10 | 9 / 10 |
| Correct refusals on the 3 unanswerable questions that passed retrieval | **3 / 3** | 0 / 3 |
| Key facts: LLM answers vs quoting the same passages (no graph) | 12 vs 20 | 26 vs 38 |
| Answers repeating a sentence | 4 | 1 |

What it did:
1. **It fixed the targeted failure completely.** No more wrong refusals, every answer passes the
   citation and number checks, and Hindi questions now get answers.
2. **It overshot.** It also stopped refusing when it should: the 3 unanswerable questions it used
   to refuse correctly are now answered from unrelated passages. The refusal pairs were "obviously
   off-topic passages"; the evaluation's unanswerable questions have *near-miss* passages (right
   scheme, wrong year or state). DPO learned "refuse only when it is obvious". The fix is
   **hard negatives**: refusal pairs whose passages look relevant but don't contain the answer.
3. **Content is still capped by the teacher.** Its answers carry about two-thirds of the key facts
   that quoting the same passages gives (26 vs 38), because both SFT targets and DPO's chosen
   answers are quotes chosen by the embedding ranker.

**Decision:** the default stays quote mode. This experiment shows preference tuning moves exactly
the behaviour you build pairs for, and *only* that behaviour, in both directions. The pairs are
the specification.

Reproduce: `python -m parali.tune.dpo pairs && python -m parali.tune.dpo train`, then serve with
both adapters (`... serve Qwen/Qwen2.5-0.5B-Instruct models/answer_lora,models/answer_dpo 8082`)
and run the same evaluation with `PARALI_LLM_MODEL=qwen2.5-0.5b-lora-dpo`.

## 6. Reproduce (SFT)

```bash
uv sync --group tune
python -m parali.tune.data select
PARALI_LLM_BASE_URL=http://localhost:8081/v1 PARALI_LLM_MODEL=qwen2.5-1.5b python -m parali.tune.data questions 5
python -m parali.tune.data build
python -m parali.tune.lora Qwen/Qwen2.5-0.5B-Instruct 2        # ~75 min on 2 CPU cores
python -m parali.tune.serve Qwen/Qwen2.5-0.5B-Instruct models/answer_lora 8082 &
PARALI_LLM_BASE_URL=http://localhost:8082/v1 PARALI_LLM_MODEL=qwen2.5-0.5b-lora \
  python -c "from parali.advise.evaluate import run_answers; run_answers(stage='generate')"
```
