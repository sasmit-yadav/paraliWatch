"""Command line: python -m parali <data|train|serve|all>"""

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
    elif args.command == "all":
        cmd_data(args)
        cmd_train(args)


if __name__ == "__main__":
    main()
