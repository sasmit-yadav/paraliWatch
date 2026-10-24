"""Minimal client for any OpenAI-compatible chat endpoint (Ollama, llama.cpp, Groq, Gemini...).

Every call is logged (latency, tokens, model) to data/logs/llm.jsonl so cost and speed can be
tracked. No vendor SDK is needed.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx

from parali import config

LOG = config.DATA_DIR / "logs" / "llm.jsonl"


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(config.LLM_BASE_URL and config.LLM_MODEL)


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_mode: bool = False,
    purpose: str = "chat",
    timeout: float | None = None,
) -> str:
    if not available():
        raise LLMUnavailable("Set PARALI_LLM_BASE_URL and PARALI_LLM_MODEL to use an LLM.")
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}
    start = time.perf_counter()
    try:
        resp = httpx.post(
            config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout or config.LLM_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"LLM request failed: {exc}") from exc
    data = resp.json()
    text = data["choices"][0]["message"]["content"] or ""
    _log(purpose, time.perf_counter() - start, data.get("usage", {}))
    return text.strip()


def _log(purpose: str, seconds: float, usage: dict) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "purpose": purpose,
                        "model": config.LLM_MODEL,
                        "seconds": round(seconds, 2),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    }
                )
                + "\n"
            )
    except OSError:
        pass  # logging must never break an answer


def parse_json(text: str) -> dict:
    """Small models sometimes wrap JSON in ``` fences or add a sentence; take the outer object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in LLM output")
    return json.loads(text[start : end + 1])
