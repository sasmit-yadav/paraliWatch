"""A minimal OpenAI-compatible /v1/chat/completions server for a Hugging Face model, optionally
with a LoRA adapter. Lets the normal answer pipeline and evaluation use the fine-tuned model
unchanged (PARALI_LLM_BASE_URL=http://localhost:8082/v1).

Usage: python -m parali.tune.serve [base_model] [adapter_dir|none] [port]
"""

from __future__ import annotations

import os
import sys
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.0
    max_tokens: int = 400


def create_app(base: str, adapter: str | None) -> FastAPI:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 2)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    name = base + ("+lora" if adapter else "")
    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict:
        return {"data": [{"id": name}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict:
        ids = tok.apply_chat_template(req.messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            # greedy decoding; a mild repetition penalty stops small models repeating a quoted sentence
            out = model.generate(**ids, max_new_tokens=req.max_tokens, do_sample=False, repetition_penalty=1.1)
        new = out[0, ids["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(ids["input_ids"].shape[1]), "completion_tokens": int(len(new))},
        }

    return app


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8082
    uvicorn.run(create_app(base, adapter), host="127.0.0.1", port=port, log_level="warning")
