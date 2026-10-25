"""LoRA fine-tuning of a small instruct LLM (default Qwen2.5-0.5B-Instruct, Apache-2.0) on CPU.

Only the low-rank adapter matrices on the attention and MLP projections train (~1% of the weights);
the loss is computed on the assistant's answer tokens only, not on the prompt.

Usage: python -m parali.tune.lora [base_model] [epochs]
Output: models/answer_lora/ (adapter weights, a few MB) + eval/tune/train_log.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

import torch

from parali import config

BASE = os.environ.get("PARALI_TUNE_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = config.ROOT / "models" / "answer_lora"
LOG_DIR = config.ROOT / "eval" / "tune"
MAX_TOKENS = 2304  # memory grows with the square of length; 3,300-token examples hit the 6 GB limit
SEED = 0


def encode(tok, messages: list[dict]) -> dict | None:
    """input_ids for the full chat, labels = -100 everywhere except the assistant answer."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + messages[-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_TOKENS:
        return None
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": torch.tensor([ids]), "labels": torch.tensor([labels])}


def answer_loss(model, ex: dict) -> torch.Tensor:
    """Next-token loss on the answer tokens only, computing vocabulary logits just for those
    positions. A full forward would build a (3,300 x 151,936) logit matrix, about 2 GB in float32
    plus its gradient, which does not fit next to the model on an 8 GB machine."""
    inner = model.get_base_model()  # Qwen2ForCausalLM with LoRA layers injected
    hidden = inner.model(input_ids=ex["input_ids"]).last_hidden_state[0]  # (T, d)
    labels = ex["labels"][0]
    pos = (labels[1:] != -100).nonzero(as_tuple=True)[0]  # predict token t+1 from position t
    logits = inner.lm_head(hidden[pos])
    return torch.nn.functional.cross_entropy(logits.float(), labels[1:][pos])


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(os.cpu_count() or 2)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} of {total:,} ({trainable / total:.2%})", flush=True)

    rows = [json.loads(line) for line in open(config.DATA_DIR / "tune" / "train.jsonl", encoding="utf-8")]
    examples = [e for e in (encode(tok, r["messages"]) for r in rows) if e is not None]
    print(f"{len(examples)} of {len(rows)} examples fit in {MAX_TOKENS} tokens", flush=True)

    accum = 4  # effective batch size 4, one sequence at a time (no padding waste on CPU)
    steps = epochs * len(examples) // accum
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=max(steps, 1), pct_start=0.1)
    model.train()
    log, start, step, running = [], time.time(), 0, []
    for epoch in range(epochs):
        random.shuffle(examples)
        for i, ex in enumerate(examples, start=1):
            loss = answer_loss(model, ex) / accum
            loss.backward()
            running.append(loss.item() * accum)
            if i % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    log.append({"step": step, "epoch": epoch + 1, "loss": round(sum(running) / len(running), 4),
                                "minutes": round((time.time() - start) / 60, 1)})
                    running = []
                    print(log[-1], flush=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "train_log.json").write_text(json.dumps({
        "base": base, "examples": len(examples), "epochs": epochs, "effective_batch": accum,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": lora.target_modules and sorted(lora.target_modules)},
        "trainable_params": trainable, "total_params": total,
        "train_minutes": round((time.time() - start) / 60, 1), "loss_curve": log,
    }, indent=2))


if __name__ == "__main__":
    main()
