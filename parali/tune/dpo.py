"""DPO (Direct Preference Optimisation) on top of the LoRA model, aimed at one failure:
**refusing questions that the passages do answer** (17 of 37 in the LoRA evaluation).

Supervised fine-tuning (SFT) shows the model one good answer per prompt. DPO shows it two answers
to the same prompt, a preferred one ("chosen") and a worse one ("rejected"), and pushes the
model's probabilities towards the first and away from the second, while a frozen copy of the
starting model (the "reference") keeps it from drifting too far. No reward model is needed.

    margin = (log p(chosen) - log ref(chosen)) - (log p(rejected) - log ref(rejected))
    loss   = -log sigmoid(beta * margin)

Preference pairs, built from the same training prompts as the SFT data (evaluation pages still
held out), no extra labelling:
- answerable prompt:  chosen = grounded, cited answer   rejected = "I could not find this..."
- unanswerable prompt: chosen = "I could not find this..."  rejected = an answer quoting the
  unrelated passages as if they answered it
So the model is taught *both* directions: answer when the passages contain the answer, refuse
when they do not. Training on only the first kind would just teach it never to refuse, so the two
kinds are balanced 1:1 (36 answer pairs, the 12 refusal pairs each used three times).

Memory trick (CPU, 6 GB): the DPO gradient is  -beta * w * (grad log p(chosen) - grad log p(rejected))
with  w = sigmoid(-beta * margin).  We compute w first without gradients, then back-propagate the
chosen and the rejected answer one after the other. Only one sequence's activations are in memory
at a time, and the gradient is exactly the DPO gradient (unit-tested).

Usage:
  python -m parali.tune.dpo pairs
  python -m parali.tune.dpo train
  python -m parali.tune.serve Qwen/Qwen2.5-0.5B-Instruct models/answer_lora,models/answer_dpo 8082
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time

import torch

from parali import config
from parali.advise.answer import NOT_FOUND
from parali.advise.text import sentences
from parali.tune.lora import BASE, LOG_DIR, MAX_TOKENS, encode

SFT_ADAPTER = config.ROOT / "models" / "answer_lora"
DPO_ADAPTER = config.ROOT / "models" / "answer_dpo"
PAIRS = config.DATA_DIR / "tune" / "dpo_pairs.jsonl"
BETA = 0.1
LR = 5e-5
SEED = 0


def _first_passage(user_content: str) -> str:
    m = re.search(r"\[1\][^\n]*\n(.*?)(?:\n\[2\]|\n</sources>)", user_content, re.S)
    return m.group(1).strip() if m else ""


def build_pairs() -> dict:
    rows = [json.loads(x) for x in open(config.DATA_DIR / "tune" / "train.jsonl", encoding="utf-8")]
    out, stats = [], {"answer_vs_refusal": 0, "refusal_vs_wrong_answer": 0, "skipped": 0}
    for r in rows:
        prompt, target = r["messages"][:-1], r["messages"][-1]["content"]
        refusal = NOT_FOUND[r["lang"]].format(helpline=config.HELPLINE)
        if r["kind"] == "answer":
            chosen, rejected = target, refusal
            stats["answer_vs_refusal"] += 1
        else:
            quotes = sentences(_first_passage(prompt[1]["content"]))[:2]
            if not quotes:
                stats["skipped"] += 1
                continue
            chosen, rejected = refusal, " ".join(f"{q} [1]" for q in quotes)
            stats["refusal_vs_wrong_answer"] += 1
        out.append(
            {"kind": r["kind"], "lang": r["lang"], "prompt": prompt, "chosen": chosen, "rejected": rejected}
        )
    PAIRS.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    stats["pairs"] = len(out)
    return stats


def answer_logprob(model, ex: dict) -> torch.Tensor:
    """Sum of log-probabilities of the answer tokens (vocabulary logits only at those positions)."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(input_ids=ex["input_ids"]).last_hidden_state[0]
    labels = ex["labels"][0]
    pos = (labels[1:] != -100).nonzero(as_tuple=True)[0]
    logp = torch.log_softmax(inner.lm_head(hidden[pos]).float(), dim=-1)
    return logp.gather(1, labels[1:][pos].unsqueeze(1)).sum()


def dpo_weight(pc: float, pr: float, rc: float, rr: float, beta: float = BETA) -> tuple[float, float]:
    """(loss, w) for one pair: loss = -log sigmoid(beta * margin), w = sigmoid(-beta * margin)."""
    margin = torch.tensor(beta * ((pc - rc) - (pr - rr)))
    return float(-torch.nn.functional.logsigmoid(margin)), float(torch.sigmoid(-margin))


def train(epochs: int = 1, accum: int = 4) -> None:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(os.cpu_count() or 2)
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    sft = PeftModel.from_pretrained(base, str(SFT_ADAPTER)).merge_and_unload()  # start from the SFT model
    sft.gradient_checkpointing_enable()
    sft.enable_input_require_grads()
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(sft, lora)

    data = []
    for p in (json.loads(x) for x in open(PAIRS, encoding="utf-8")):
        c = encode(tok, p["prompt"] + [{"role": "assistant", "content": p["chosen"]}])
        r = encode(tok, p["prompt"] + [{"role": "assistant", "content": p["rejected"]}])
        if c is not None and r is not None:
            data.append({"kind": p["kind"], "c": c, "r": r})
    print(f"{len(data)} pairs fit in {MAX_TOKENS} tokens", flush=True)
    # Balance the two directions (68 answer pairs vs 12 refusal pairs): 36 answer pairs and each
    # refusal pair three times. Otherwise DPO mostly learns "never refuse".
    answer_pairs = [d for d in data if d["kind"] == "answer"]
    refusal_pairs = [d for d in data if d["kind"] != "answer"]
    random.Random(SEED).shuffle(answer_pairs)
    data = answer_pairs[: 3 * len(refusal_pairs)] + refusal_pairs * 3
    print(f"balanced: {len(data)} pairs", flush=True)

    # Reference log-probabilities: the SFT model with the new adapter switched off, computed once.
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for d in data:
            d["ref_c"], d["ref_r"] = (
                float(answer_logprob(model, d["c"])),
                float(answer_logprob(model, d["r"])),
            )
    print("reference log-probs done", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    model.train()
    log, start, step, window = [], time.time(), 0, []
    for epoch in range(epochs):
        random.shuffle(data)
        for i, d in enumerate(data, start=1):
            with torch.no_grad():
                pc, pr = float(answer_logprob(model, d["c"])), float(answer_logprob(model, d["r"]))
            loss, w = dpo_weight(pc, pr, d["ref_c"], d["ref_r"])
            (-BETA * w * answer_logprob(model, d["c"]) / accum).backward()  # raise chosen
            (BETA * w * answer_logprob(model, d["r"]) / accum).backward()  # lower rejected
            margin = (pc - d["ref_c"]) - (pr - d["ref_r"])
            window.append({"loss": loss, "margin": margin, "correct": margin > 0})
            if i % accum == 0 or i == len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                log.append(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "loss": round(sum(x["loss"] for x in window) / len(window), 4),
                        "reward_margin": round(sum(x["margin"] for x in window) / len(window), 3),
                        "pref_accuracy": round(sum(x["correct"] for x in window) / len(window), 3),
                        "minutes": round((time.time() - start) / 60, 1),
                    }
                )
                window = []
                print(log[-1], flush=True)
    DPO_ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DPO_ADAPTER)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "dpo_log.json").write_text(
        json.dumps(
            {
                "start_from": "SFT LoRA merged into " + BASE,
                "pairs": len(data),
                "epochs": epochs,
                "beta": BETA,
                "lr": LR,
                "effective_batch": accum,
                "train_minutes": round((time.time() - start) / 60, 1),
                "curve": log,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["pairs"]:
        print(build_pairs())
    elif sys.argv[1:2] == ["train"]:
        train()
    else:
        print(__doc__)
