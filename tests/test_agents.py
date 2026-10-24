import asyncio
from datetime import date

from parali.agents import bulletin

DAY = date(2025, 11, 7)


def test_bulletin_workflow_produces_valid_bilingual_draft():
    b = bulletin.generate(DAY, save=False)
    assert b.status == "draft"
    assert 1 <= len(b.districts) <= 5
    assert b.districts[0].expected_fire_cells >= b.districts[-1].expected_fire_cells
    assert all(d.top_cells and d.top_cells[0].reasons for d in b.districts)
    assert any("ऀ" <= ch <= "ॿ" for ch in b.hi.summary)  # Hindi text present
    assert b.en.actions


def test_bulletin_approval_is_recorded(tmp_path, monkeypatch):
    from parali import config

    monkeypatch.setattr(config, "BULLETIN_DIR", tmp_path)
    approved = bulletin.approve(DAY, "District Agriculture Officer")
    assert approved.status == "approved"
    assert bulletin.load(DAY).approved_by == "District Agriculture Officer"


def test_mcp_server_exposes_tools():
    from parali.agents.mcp_server import mcp

    async def run():
        names = {t.name for t in await mcp.list_tools()}
        result = await mcp.call_tool("get_risk", {"date": DAY.isoformat(), "district": "Sangrur", "top_n": 3})
        return names, result

    names, result = asyncio.run(run())
    assert {
        "get_risk",
        "explain_cell",
        "get_fires",
        "district_summary",
        "daily_bulletin",
        "search_schemes",
    } <= names
    assert "Sangrur" in str(result)


def test_bulletin_checks_reject_fluent_but_wrong_text():
    b = bulletin.generate(DAY, save=False)
    # an LLM output seen in testing: wrong trend talk, raw cell IDs, no top district
    bad_en = bulletin.Text(
        headline="High Fire Risk in Punjab: Urgent Action Needed",
        summary="Risk is rising. The top cells are 863d16c67ffffff and 863d1689fffffff.",
        actions=["Act now."],
    )
    problems = bulletin.consistency_problems(bad_en, b.hi, b.districts)
    assert "raw cell IDs in text" in problems
    assert any("top district" in p for p in problems)
    assert bulletin.consistency_problems(b.en, b.hi, b.districts) == []  # the template passes
