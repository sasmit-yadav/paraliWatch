"""GraphRAG: a small knowledge graph of schemes, benefits, machines, eligibility rules and penalties.

The graph used at question time is data/corpus/knowledge_graph.json: reviewed facts, each with its
document, page and an exact supporting quote. `python -m parali graph` runs LLM extraction over the
chunks into data/index/graph_extracted.json, for a person to review and merge. At question time,
facts about the entities named in the question are added to the context. This helps multi-hop
questions such as "which machines can a cooperative get, and at what subsidy?" where the answer is
spread over several pages.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from pydantic import BaseModel, Field, ValidationError

from parali import config
from parali.advise import llm
from parali.advise.text import expand_bilingual

log = logging.getLogger(__name__)
GRAPH = config.CORPUS_DIR / "knowledge_graph.json"  # reviewed graph, shipped with the repo
EXTRACTED = config.INDEX_DIR / "graph_extracted.json"  # raw LLM extraction, to be reviewed
TYPES = ["scheme", "benefit", "machine", "practice", "eligibility", "authority", "penalty", "place"]
RELATIONS = [
    "provides",
    "covers_machine",
    "eligible",
    "amount",
    "administered_by",
    "applies_in",
    "penalty_for",
    "alternative_to",
    "how_to_apply",
]


class Triple(BaseModel):
    subject: str = Field(min_length=2, max_length=120)
    subject_type: str
    relation: str
    object: str = Field(min_length=1, max_length=200)
    object_type: str
    quote: str = Field(default="", max_length=300)


class Extraction(BaseModel):
    triples: list[Triple] = Field(default_factory=list)


PROMPT = (
    "Extract facts about crop residue management schemes from the text as JSON: "
    '{{"triples": [{{"subject": ..., "subject_type": ..., "relation": ..., "object": ..., '
    '"object_type": ..., "quote": ...}}]}}.\n'
    "Types: " + ", ".join(TYPES) + ".\nRelations: " + ", ".join(RELATIONS) + ".\n"
    "Only facts stated in the text. Copy amounts exactly. quote = the exact words supporting the fact. "
    'Return {{"triples": []}} if there are none.\n\nText:\n{text}'
)


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def extract(chunks) -> dict:
    """Run the LLM over every chunk and keep only triples that validate and whose quote is real."""
    triples = []
    for c in chunks:
        try:
            raw = llm.chat(
                [{"role": "user", "content": PROMPT.format(text=c.text)}],
                max_tokens=700,
                json_mode=False,  # grammar-constrained JSON is very slow on local servers; parsed instead
                purpose="graph",
            )
            parsed = Extraction.model_validate(llm.parse_json(raw))
        except (llm.LLMUnavailable, ValueError, ValidationError) as exc:
            log.debug("chunk %s skipped: %s", c.chunk_id, exc)
            continue
        for t in parsed.triples:
            if t.relation not in RELATIONS or t.subject_type not in TYPES or t.object_type not in TYPES:
                continue
            if t.quote and _norm(t.quote)[:40] not in _norm(c.text):
                continue  # the model invented the supporting quote: drop the fact
            triples.append({**t.model_dump(), "doc_id": c.doc_id, "title": c.title, "page": c.page})
    entities = {}
    for t in triples:
        for side in ("subject", "object"):
            key = _norm(t[side])
            entities.setdefault(key, {"name": t[side], "type": t[f"{side}_type"], "aliases": []})
    graph = {"entities": entities, "triples": triples, "reviewed": False}
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Graph: %d entities, %d facts", len(entities), len(triples))
    return graph


@lru_cache(maxsize=1)
def load() -> dict | None:
    if not GRAPH.exists():
        return None
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def facts_for(question: str, limit: int = 8) -> list[str]:
    """Facts touching any entity whose name (or alias) appears in the question, plus one hop."""
    graph = load()
    if not graph:
        return []
    q = _norm(expand_bilingual(question))
    names = {
        k
        for k, e in graph["entities"].items()
        if len(k) > 3 and (k in q or any(_norm(a) in q for a in e["aliases"]))
    }
    if not names:
        return []
    hop = {_norm(t["object"]) for t in graph["triples"] if _norm(t["subject"]) in names}
    keys = names | hop
    out = []
    for t in graph["triples"]:
        if _norm(t["subject"]) in keys or _norm(t["object"]) in names:
            relation = t["relation"].replace("_", " ")
            out.append(f"- {t['subject']} -> {relation} -> {t['object']} ({t['title']}, p. {t['page']})")
        if len(out) >= limit:
            break
    return out
