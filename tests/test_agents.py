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
