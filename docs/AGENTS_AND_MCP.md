# Agents and MCP server

## Daily bulletin agents

District officers don't want a map of 2,455 hexagons every morning; they want "where should my
teams go today, and why". `parali/agents/bulletin.py` produces that with a three-step
[LangGraph](https://github.com/langchain-ai/langgraph) workflow:

```
monitor ──> analyst ──> writer ──> draft bulletin ──> official approves ──> shared
```

| Step | What it does | Uses an LLM? |
| --- | --- | --- |
| **monitor** | Reads the day's forecast per district and the fires detected in the last 6 days | No |
| **analyst** | Plans the focus: the 5 districts with the most expected fire cells; alert level (high / medium / low); trend (fires in the last 3 days vs the 3 before); top 2 cells per district with their SHAP reasons | No |
| **writer** | Writes headline, summary and 2-4 actions in **English and Hindi** as JSON | Yes, if configured; otherwise a template |

Design choices:

- **Facts come from code, words from the model.** The analyst computes every number. The writer
  only phrases them. Its output must validate against a Pydantic schema (`Text`, `Bulletin`), and
  every number it writes must exist in the analyst's facts (`guard.unsupported_numbers`). If either
  check fails, the bulletin falls back to the template, so a bad model output can never publish a
  wrong figure.
- **Template by default, LLM by choice.** We ran the writer with a small local model (Qwen2.5-1.5B,
  4-bit, llama.cpp). Its output passed the schema and number checks, yet the English contradicted
  the facts ("an expected increase ... a trend of falling fire incidents"), listed raw cell IDs, and
  the Hindi was largely meaningless (with Marathi words mixed in). Validation is not the same as
  quality. So the bulletin uses the template unless `PARALI_BULLETIN_LLM=1` is set, and LLM text
  must also pass `consistency_problems()`: the top district must be named, no raw cell IDs, no
  "rising/falling" claims the data doesn't support, and the Hindi must be mostly Devanagari. Use a
  strong model (for example a 70B model through an OpenAI-compatible API) if you enable it.
- **Human in the loop.** Bulletins are saved as `draft`. `POST /api/bulletin/approve` (or the
  Approve button in the dashboard) records who approved it and when.
- **Expected fire cells** = the sum of calibrated probabilities over a district's cells: the
  number of cells we expect to see at least one detected fire in the next 3 days.
- **Alert levels:** high if expected fire cells >= 10 or high-risk cells (top 10% that day) >= 20;
  medium if >= 3 or >= 5; otherwise low. These are simple, readable rules, not learned.

Run it: `python -m parali bulletin 2025-11-07`, or `GET /api/bulletin?date=2025-11-07`.

## MCP server

The [Model Context Protocol](https://modelcontextprotocol.io) lets AI assistants call external
tools. `parali/agents/mcp_server.py` exposes Parali Watch as seven tools, so an assistant can answer
"which villages around Sangrur are likely to burn this week, and what should the officer do?"
using live project data instead of guessing.

| Tool | Returns |
| --- | --- |
| `available_dates` | First, last and default forecast dates |
| `get_risk(date, district, top_n)` | Highest-risk cells with probability and rank |
| `explain_cell(cell, date)` | Top 3 reasons, 14-day history, what happened next |
| `get_fires(date, days, district)` | Detected fire points |
| `district_summary(date)` | Expected fire cells and high-risk cells per district |
| `daily_bulletin(date)` | The agent-written bulletin (EN + HI) |
| `search_schemes(question)` | Cited answer from the scheme documents |

Run over stdio: `python -m parali mcp`. Client configuration (for example Claude Desktop or an
IDE assistant):

```json
{
  "mcpServers": {
    "parali-watch": {
      "command": "uv",
      "args": ["run", "python", "-m", "parali", "mcp"],
      "cwd": "/path/to/paraliWatch"
    }
  }
}
```

Built on the official MCP Python SDK (v2, `MCPServer`). `tests/test_agents.py` lists the tools
and calls `get_risk` in-process.
