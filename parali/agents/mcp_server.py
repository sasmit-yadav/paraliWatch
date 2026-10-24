"""MCP server: lets any MCP client (desktop AI assistants, IDE assistants, other agents) use Parali Watch
as tools. Run over stdio:

    python -m parali mcp

Example client config:
    {"mcpServers": {"parali-watch": {"command": "uv", "args": ["run", "python", "-m", "parali", "mcp"],
                                     "cwd": "/path/to/paraliWatch"}}}
"""

from __future__ import annotations

from datetime import date

import h3
from mcp.server.mcpserver import MCPServer

from parali import config
from parali.app.store import Store

mcp = MCPServer(
    "parali-watch",
    instructions=(
        "Crop-residue fire risk forecasts, explanations, bulletins and scheme answers for Punjab and Haryana."
    ),
)
_store = Store()


def _day(value: str | None) -> date:
    if not value:
        return _store.default_date()
    d = date.fromisoformat(value)
    if d not in set(_store.dates):
        raise ValueError(f"No forecast for {d}. Available: {_store.dates[0]} to {_store.dates[-1]}.")
    return d


@mcp.tool()
def available_dates() -> dict:
    """Dates that have a forecast, and the default (busiest) date."""
    return {
        "first": _store.dates[0].isoformat(),
        "last": _store.dates[-1].isoformat(),
        "default": _store.default_date().isoformat(),
    }


@mcp.tool()
def get_risk(date: str | None = None, district: str | None = None, top_n: int = 10) -> list[dict]:
    """Highest-risk H3 cells for a date (YYYY-MM-DD), optionally within one district. Each item has
    the calibrated probability of a detected fire in the next 3 days and its rank that day."""
    d = _day(date)
    rows = _store.top(d, n=10_000 if district else max(1, min(top_n, 100)))
    if district:
        rows = [r for r in rows if r["district"].lower() == district.lower()][: max(1, min(top_n, 100))]
    return rows


@mcp.tool()
def explain_cell(cell: str, date: str | None = None) -> dict:
    """Why a cell is at risk: top 3 reasons in plain language, 14-day fire history, and what was
    actually detected in the following 3 days (demo season only)."""
    out = _store.cell(cell, _day(date))
    if out is None:
        raise ValueError("Cell is not in the study area.")
    return out


@mcp.tool()
def get_fires(date: str | None = None, days: int = 1, district: str | None = None) -> dict:
    """Fire points detected by NASA VIIRS on the given date (and up to 6 days before it)."""
    d = _day(date)
    fc = _store.fires_geojson(d, max(1, min(days, 7)))
    if district:
        cells = set(_store.grid.index[_store.grid["district"].str.lower() == district.lower()])
        fc["features"] = [f for f in fc["features"] if _in_district(f, cells)]
    return {"count": len(fc["features"]), "fires": fc["features"][:500]}


def _in_district(feature: dict, cells: set[str]) -> bool:
    lon, lat = feature["geometry"]["coordinates"]
    return h3.latlng_to_cell(lat, lon, config.H3_RES) in cells


@mcp.tool()
def district_summary(date: str | None = None) -> list[dict]:
    """Per district: expected number of cells with fires in the next 3 days, and high-risk cells."""
    return _store.district_summary(_day(date))


@mcp.tool()
def daily_bulletin(date: str | None = None) -> dict:
    """The agent-written district bulletin (English + Hindi) for a date; generated if missing."""
    from parali.agents import bulletin

    d = _day(date)
    return (bulletin.load(d) or bulletin.generate(d)).model_dump(mode="json")


@mcp.tool()
def search_schemes(question: str) -> dict:
    """Answer a question about crop-residue schemes, subsidies or alternatives to burning, from
    official documents, with page citations. Works in English and Hindi."""
    from parali.advise.answer import answer

    try:
        return answer(question).to_dict()
    except FileNotFoundError:
        return {"error": "The document index has not been built. Run: python -m parali index"}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
