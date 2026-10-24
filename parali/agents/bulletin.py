"""Daily district bulletin written by a three-step agent workflow (LangGraph).

    monitor  -> reads today's forecast and the last 6 days of detected fires
    analyst  -> plans where to focus: ranks districts, sets an alert level, gathers reasons
    writer   -> writes the bulletin in English and Hindi as validated JSON (LLM, or a template)

A bulletin is saved as a draft; an official approves it before it is shared (human in the loop).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from parali import config
from parali.advise import guard, llm
from parali.app.store import Store

log = logging.getLogger(__name__)
Level = Literal["high", "medium", "low"]


class CellNote(BaseModel):
    cell: str
    prob: float
    reasons: list[str]


class DistrictAlert(BaseModel):
    district: str
    state: str
    level: Level
    expected_fire_cells: float
    high_risk_cells: int
    fires_last_3d: int
    fires_prev_3d: int
    trend: Literal["rising", "falling", "steady"]
    top_cells: list[CellNote]


class Text(BaseModel):
    headline: str = Field(min_length=5, max_length=160)
    summary: str = Field(min_length=20, max_length=900)
    actions: list[str] = Field(min_length=1, max_length=5)


class Bulletin(BaseModel):
    date: date
    generated_at: str
    status: Literal["draft", "approved"] = "draft"
    approved_by: str | None = None
    approved_at: str | None = None
    writer: Literal["llm", "template"]
    districts: list[DistrictAlert]
    en: Text
    hi: Text


class State(TypedDict, total=False):
    date: date
    observations: dict
    focus: list[DistrictAlert]
    bulletin: Bulletin


def _store() -> Store:
    return Store()


def monitor(state: State) -> State:
    s, d = _store(), state["date"]
    fires = s.fires.merge(s.grid[["district"]], left_on="cell", right_index=True)
    last3 = fires[(fires["acq_date"] > d - timedelta(days=3)) & (fires["acq_date"] <= d)]
    prev3 = fires[(fires["acq_date"] > d - timedelta(days=6)) & (fires["acq_date"] <= d - timedelta(days=3))]
    return {
        "observations": {
            "summary": s.district_summary(d),
            "fires_last_3d": last3.groupby("district").size().to_dict(),
            "fires_prev_3d": prev3.groupby("district").size().to_dict(),
        }
    }


def _level(expected: float, high_cells: int) -> Level:
    if expected >= 10 or high_cells >= 20:
        return "high"
    if expected >= 3 or high_cells >= 5:
        return "medium"
    return "low"


def analyst(state: State, n_districts: int = 5, cells_per_district: int = 2) -> State:
    """Plan: focus on the districts expected to have the most fire cells, and explain why."""
    s, d, obs = _store(), state["date"], state["observations"]
    focus = []
    day = s.day(d).merge(s.grid[["district"]], left_on="cell", right_index=True)
    for row in obs["summary"][:n_districts]:
        last, prev = (
            obs["fires_last_3d"].get(row["district"], 0),
            obs["fires_prev_3d"].get(row["district"], 0),
        )
        trend = (
            "rising" if last > 1.5 * max(prev, 1) else "falling" if prev > 1.5 * max(last, 1) else "steady"
        )
        cells = day[day["district"] == row["district"]].nlargest(cells_per_district, "raw")
        notes = []
        for cell in cells["cell"]:
            info = s.cell(cell, d)
            notes.append(
                CellNote(cell=cell, prob=round(info["prob"], 3), reasons=[r["text"] for r in info["reasons"]])
            )
        focus.append(
            DistrictAlert(
                district=row["district"],
                state=row["state"],
                level=_level(row["expected_fire_cells"], row["high_risk_cells"]),
                expected_fire_cells=row["expected_fire_cells"],
                high_risk_cells=row["high_risk_cells"],
                fires_last_3d=int(last),
                fires_prev_3d=int(prev),
                trend=trend,
                top_cells=notes,
            )
        )
    return {"focus": focus}


LEVEL_HI = {"high": "उच्च", "medium": "मध्यम", "low": "कम"}
TREND_HI = {"rising": "बढ़ रही हैं", "falling": "घट रही हैं", "steady": "स्थिर हैं"}


def _template(d: date, focus: list[DistrictAlert]) -> tuple[Text, Text]:
    top = focus[0]
    high = [f.district for f in focus if f.level == "high"]
    where = ", ".join(high or [top.district])
    en = Text(
        headline=f"Fire risk {top.level} in {top.district}; "
        f"{len(high)} district(s) on high alert for the next 3 days",
        summary=" ".join(
            f"{f.district} ({f.state}): {f.level} risk, "
            f"about {f.expected_fire_cells:g} cells likely to see fires; "
            f"{f.fires_last_3d} fires in the last 3 days ({f.trend})."
            for f in focus
        ),
        actions=[
            f"Send field teams and crop-residue machinery to the highest-risk cells in {where}.",
            "Contact village heads in the listed cells about in-situ and ex-situ alternatives to burning.",
            "Recheck the map tomorrow; risk is updated daily.",
        ],
    )
    hi = Text(
        headline=f"{top.district} में आग का जोखिम {LEVEL_HI[top.level]}; "
        f"अगले 3 दिनों के लिए {len(high)} ज़िले उच्च अलर्ट पर",
        summary=" ".join(
            f"{f.district}: जोखिम {LEVEL_HI[f.level]}, "
            f"लगभग {f.expected_fire_cells:g} क्षेत्रों में आग की संभावना; "
            f"पिछले 3 दिनों में {f.fires_last_3d} आग की घटनाएँ (घटनाएँ {TREND_HI[f.trend]})।"
            for f in focus
        ),
        actions=[
            f"{where} के सबसे अधिक जोखिम वाले क्षेत्रों में टीमें और फसल अवशेष मशीनें भेजें।",
            "सूचीबद्ध क्षेत्रों के ग्राम प्रधानों से पराली जलाने के विकल्पों के बारे में संपर्क करें।",
            "कल फिर से नक्शा देखें; जोखिम रोज़ अपडेट होता है।",
        ],
    )
    return en, hi


WRITER_PROMPT = (
    "You write a short daily bulletin for district agriculture officers about crop-residue fire risk. "
    "Use ONLY the facts in the JSON below. Copy numbers exactly. Return JSON: "
    '{{"en": {{"headline": str, "summary": str, "actions": [str]}}, '
    '"hi": {{"headline": str, "summary": str, "actions": [str]}}}}. '
    "Hindi must be natural Devanagari. Summary at most 5 sentences; 2-4 concrete actions.\n\nFacts:\n{facts}"
)


def writer(state: State) -> State:
    d, focus = state["date"], state["focus"]
    facts = json.dumps([f.model_dump() for f in focus], ensure_ascii=False)
    used = "template"
    en, hi = _template(d, focus)
    if llm.available():
        try:
            raw = llm.parse_json(
                llm.chat(
                    [{"role": "user", "content": WRITER_PROMPT.format(facts=facts)}],
                    max_tokens=900,
                    json_mode=True,
                    purpose="bulletin",
                )
            )
            cand_en, cand_hi = Text.model_validate(raw["en"]), Text.model_validate(raw["hi"])
            written = " ".join([cand_en.headline, cand_en.summary, *cand_en.actions])
            if not guard.unsupported_numbers(written, [facts]):
                en, hi, used = cand_en, cand_hi, "llm"
            else:
                log.info("Bulletin LLM output had numbers not in the facts; using template")
        except (llm.LLMUnavailable, ValueError, KeyError, ValidationError) as exc:
            log.info("Bulletin LLM unavailable or invalid (%s); using template", exc)
    bulletin = Bulletin(
        date=d,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        writer=used,
        districts=focus,
        en=en,
        hi=hi,
    )
    return {"bulletin": bulletin}


def workflow():
    g = StateGraph(State)
    g.add_node("monitor", monitor)
    g.add_node("analyst", analyst)
    g.add_node("writer", writer)
    g.set_entry_point("monitor")
    g.add_edge("monitor", "analyst")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", END)
    return g.compile()


def path_for(d: date):
    return config.BULLETIN_DIR / f"{d.isoformat()}.json"


def generate(d: date, save: bool = True) -> Bulletin:
    bulletin = workflow().invoke({"date": d})["bulletin"]
    if save:
        config.BULLETIN_DIR.mkdir(parents=True, exist_ok=True)
        path_for(d).write_text(bulletin.model_dump_json(indent=1), encoding="utf-8")
    return bulletin


def load(d: date) -> Bulletin | None:
    p = path_for(d)
    return Bulletin.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def approve(d: date, official: str) -> Bulletin:
    bulletin = load(d) or generate(d)
    bulletin.status, bulletin.approved_by = "approved", official
    bulletin.approved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path_for(d).write_text(bulletin.model_dump_json(indent=1), encoding="utf-8")
    return bulletin
