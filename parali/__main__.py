"""Command line: python -m parali <command>; run `python -m parali --help` for the list."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from parali import config

PANEL = config.PROCESSED_DIR / "panel.parquet"


def cmd_data(args) -> None:
    from parali.forecast.features import build_panel
    from parali.pipeline import build

    build.run(force=args.force)
    build_panel(build.load_fires(), build.load_weather(), build.step_grid()).to_parquet(PANEL, index=False)
    logging.info("Feature table written to %s", PANEL)


def cmd_train(args) -> None:
    from parali.forecast import export, report, train
    from parali.pipeline import build

    out = train.run(pd.read_parquet(PANEL))
    fires = build.load_fires()
    counts = pd.to_datetime(fires["acq_date"]).dt.year.value_counts().sort_index()
    report.write(out["results"], counts)
    export.run(out["booster"], out["test"])
    logging.info("Model, eval/RESULTS.md and demo files written")


def cmd_index(args) -> None:
    from parali.advise import index

    for strategy in args.strategies:
        index.build(strategy=strategy)


def cmd_add_url(args) -> None:
    from parali.advise.fetch import add_url

    logging.info("Saved %s. Run `python -m parali index` to include it.", add_url(args.url, args.title))


def cmd_ocr(args) -> None:
    from parali.advise import ocr

    for path in ocr.run(only_missing_text=not args.all):
        logging.info("wrote %s", path)


def cmd_graph(args) -> None:
    from parali.advise import graph, index
    from parali.advise.ingest import load_corpus

    graph.extract(index.make_chunks(load_corpus(), "structure"))


def cmd_drift(args) -> None:
    from parali.ops import drift

    drift.run(args.seasons or [2024, 2025])
    logging.info("Report written to %s", drift.OUT / "REPORT.md")


def cmd_track(args) -> None:
    import json

    from parali.ops import tracking

    print(json.dumps(tracking.backfill(force=args.force), indent=2))


def cmd_ask(args) -> None:
    import json

    from parali.advise.answer import answer

    print(json.dumps(answer(" ".join(args.question)).to_dict(), ensure_ascii=False, indent=2))


def cmd_bulletin(args) -> None:
    from datetime import date

    from parali.agents import bulletin

    b = bulletin.generate(date.fromisoformat(args.date))
    print(b.en.headline)
    print(b.hi.headline)
    logging.info("Saved to %s", bulletin.path_for(b.date))


def cmd_mcp(args) -> None:
    from parali.agents.mcp_server import main as run_mcp

    run_mcp()


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("parali.app.main:app", host=args.host, port=args.port, reload=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="parali", description="Parali Watch command line")
    sub = parser.add_subparsers(dest="command", required=True)
    p_data = sub.add_parser("data", help="download sources and build the feature table")
    p_data.add_argument("--force", action="store_true", help="rebuild even if cached")
    sub.add_parser("train", help="train, evaluate and export demo files")
    p_serve = sub.add_parser("serve", help="run the dashboard and API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_index = sub.add_parser("index", help="index the documents in data/corpus for the assistant")
    p_index.add_argument("--strategies", nargs="+", default=[config.CHUNK_STRATEGY])
    p_url = sub.add_parser("add-url", help="add an official web page to the corpus by its link")
    p_url.add_argument("url")
    p_url.add_argument("--title")
    p_ocr = sub.add_parser("ocr", help="OCR scanned PDFs in data/corpus (needs Tesseract)")
    p_ocr.add_argument("--all", action="store_true", help="OCR every PDF, not only those without text")
    sub.add_parser("graph", help="extract the knowledge graph from the documents (needs an LLM)")
    p_ask = sub.add_parser("ask", help="ask the assistant a question")
    p_ask.add_argument("question", nargs="+")
    p_bul = sub.add_parser("bulletin", help="write the district bulletin for a date")
    p_bul.add_argument("date")
    sub.add_parser("mcp", help="run the MCP server over stdio")
    p_drift = sub.add_parser("drift", help="drift report: new seasons vs the training seasons")
    p_drift.add_argument("seasons", nargs="*", type=int)
    p_track = sub.add_parser("track", help="log all recorded experiments to MLflow (needs the 'ops' group)")
    p_track.add_argument("--force", action="store_true", help="log again even if already done")
    p_all = sub.add_parser("all", help="data + train")
    p_all.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config.ensure_dirs()
    if args.command == "data":
        cmd_data(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "add-url":
        cmd_add_url(args)
    elif args.command == "ocr":
        cmd_ocr(args)
    elif args.command == "graph":
        cmd_graph(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "bulletin":
        cmd_bulletin(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "drift":
        cmd_drift(args)
    elif args.command == "track":
        cmd_track(args)
    elif args.command == "all":
        cmd_data(args)
        cmd_train(args)


if __name__ == "__main__":
    main()
