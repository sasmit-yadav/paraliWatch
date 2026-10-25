"""LLM-as-judge: a second model grades each answer, and we check how far the grader can be trusted.

Why: string checks ("does the answer contain '14%'?") are precise but narrow. A judge model can
read the answer the way a person would: is every claim backed by the cited passage
(faithfulness), and does it actually answer the question (relevance)?

Why not trust it blindly: judges have biases (preferring long answers, their own model family,
confident wording). So this script also runs a **meta-evaluation**: it compares the judge's
verdicts with labels we already trust and reports agreement and Cohen's kappa.

- The judge is **Phi-3.5-mini** (Microsoft, MIT licence), a different model family from the
  answering models (Qwen), to avoid a model grading its own style. Run it behind any
  OpenAI-compatible endpoint (llama.cpp server here).
- Faithfulness sanity check: quote-mode answers are copied word for word from the passages, so a
  reliable judge must call almost all of them supported.
- Relevance check: the reference label is "the answer contains at least one of the question's
  key facts" (from eval/rag/questions.jsonl).

Usage:
  PARALI_LLM_BASE_URL=http://localhost:8081/v1 PARALI_LLM_MODEL=phi-3.5-mini \\
    python -m parali.advise.judge no_llm qwen2.5-0.5b-lora
Writes eval/judge/<system>.json and eval/judge/REPORT.md.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np

from parali import config
from parali.advise import llm
from parali.advise.guard import cited_indices

RAG = config.EVAL_DIR / "rag"
OUT = config.EVAL_DIR / "judge"

PROMPT = """You are grading an answer written for a farmer from official documents.

<passages>
{passages}
</passages>

<question>{question}</question>

<answer>{answer}</answer>

Grade two things.
SUPPORTED: is every statement in the answer backed by the passages? (yes / partly / no)
ANSWERS: does the answer give the information the question asks for? (yes / partly / no)
Reply in exactly this format, nothing else:
SUPPORTED: <yes|partly|no>
ANSWERS: <yes|partly|no>
REASON: <one short sentence>"""

LINE = re.compile(r"^\s*(SUPPORTED|ANSWERS|REASON)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def parse(text: str) -> dict | None:
    found = {k.upper(): v.strip() for k, v in LINE.findall(text)}
    out = {}
    for key in ("SUPPORTED", "ANSWERS"):
        word = found.get(key, "").lower().split()
        if not word or word[0].strip(".,") not in ("yes", "partly", "no"):
            return None
        out[key.lower()] = word[0].strip(".,")
    out["reason"] = found.get("REASON", "")[:300]
    return out


def _passages(record: dict, hits: list[dict]) -> str:
    relevant = [h for h in hits if h["score"] >= config.MIN_RERANK_SCORE]
    used = cited_indices(record["answer"], len(relevant)) or list(range(1, min(3, len(relevant)) + 1))
    return "\n\n".join(f"[{i}] {relevant[i - 1]['chunk']['text']}" for i in used)


def judge(system: str) -> dict:
    data = json.loads((RAG / f"answer_results_{system}.json").read_text(encoding="utf-8"))
    contexts = json.loads((RAG / "retrieved_contexts.json").read_text(encoding="utf-8"))
    questions = {
        json.loads(x)["id"]: json.loads(x) for x in (RAG / "questions.jsonl").read_text().splitlines() if x
    }
    rows = []
    for r in data["records"]:
        if not r["answerable"] or r["mode"] == "not_found":
            continue
        q = questions[r["id"]]
        # the "Related facts" appendix comes from the reviewed graph, not the passages: grade the quotes only
        answer = r["answer"].split("\n\nRelated facts:")[0]
        prompt = PROMPT.format(
            passages=_passages(r, contexts[r["id"]]["hits"]), question=q["question"], answer=answer
        )
        verdict = None
        for _ in range(2):  # one retry if the format is not followed
            raw = llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120, purpose="judge"
            )
            verdict = parse(raw)
            if verdict:
                break
        rows.append(
            {
                "id": r["id"],
                "lang": r["lang"],
                "mode": r["mode"],
                "answer_words": len(answer.split()),
                "facts_found": len(r["facts_found"]),
                "facts_total": r["facts_total"],
                "cited_gold_page": r["cited_gold_page"],
                "verdict": verdict,
            }
        )
        print(r["id"], r["mode"], verdict, flush=True)
    result = {"system": system, "judge_model": config.LLM_MODEL, "rows": rows, "meta": meta_evaluate(rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{system}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa: agreement corrected for the agreement expected by chance."""
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return round((po - pe) / (1 - pe), 3) if pe < 1 else float("nan")


def meta_evaluate(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["verdict"]]
    if not ok:
        return {"judged": 0}
    supported = [r["verdict"]["supported"] for r in ok]
    answers = [r["verdict"]["answers"] for r in ok]
    judge_rel = [int(v != "no") for v in answers]
    ref_rel = [int(r["facts_found"] > 0) for r in ok]
    words = np.array([r["answer_words"] for r in ok])
    return {
        "judged": len(ok),
        "unparseable": len(rows) - len(ok),
        "supported": {k: supported.count(k) for k in ("yes", "partly", "no")},
        "answers": {k: answers.count(k) for k in ("yes", "partly", "no")},
        "relevance_vs_fact_check": {
            "agreement": round(float(np.mean(np.array(judge_rel) == np.array(ref_rel))), 3),
            "cohen_kappa": kappa(judge_rel, ref_rel),
            "judge_yes_but_no_fact": sum(j == 1 and f == 0 for j, f in zip(judge_rel, ref_rel, strict=True)),
            "judge_no_but_fact_found": sum(
                j == 0 and f == 1 for j, f in zip(judge_rel, ref_rel, strict=True)
            ),
        },
        # length bias: do longer answers get "yes" more often, beyond what the facts explain?
        "length_bias_corr": round(float(np.corrcoef(words, judge_rel)[0, 1]), 3)
        if len(set(judge_rel)) > 1
        else None,
    }


def report() -> str:
    lines = [
        "# LLM-as-judge results",
        "",
        "Generated by `python -m parali.advise.judge`. Judge: Phi-3.5-mini (different family from the answer "
        "models). Only answerable questions that got an answer are graded.",
        "",
        "| System | Graded | Supported yes / partly / no | Answers yes / partly / no "
        "| Agreement with fact check "
        "| Cohen's kappa | Length bias (corr.) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in sorted(OUT.glob("*.json")):
        m = json.loads(f.read_text())["meta"]
        if not m.get("judged"):
            continue
        s, a, rv = m["supported"], m["answers"], m["relevance_vs_fact_check"]
        lines.append(
            f"| {f.stem} | {m['judged']} | {s['yes']} / {s['partly']} / {s['no']} | "
            f"{a['yes']} / {a['partly']} / {a['no']} | {rv['agreement']:.0%} | {rv['cohen_kappa']} | "
            f"{m['length_bias_corr']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if not llm.available():
        raise SystemExit("Set PARALI_LLM_BASE_URL and PARALI_LLM_MODEL to the judge model's endpoint.")
    for name in sys.argv[1:] or ["no_llm"]:
        judge(name)
    (OUT / "REPORT.md").write_text(report())
    print(report())
