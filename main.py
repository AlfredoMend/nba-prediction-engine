"""
NBA Prediction Engine - master runner.

Examples
--------
python main.py --update-and-predict                 # daily job: refresh GitHub-hosted data, retrain, predict today
python main.py --backtest                           # 2025-26 walk-forward backtest + report + plots
python main.py --update-and-predict --backtest      # everything
python main.py --source synthetic --date 2026-10-26 --update-and-predict --backtest   # offline demo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update-and-predict", action="store_true", help="ingest data, rebuild features, retrain, predict")
    p.add_argument("--backtest", action="store_true", help="run the 2025-26 walk-forward backtest")
    p.add_argument("--skip-props-backtest", action="store_true", help="backtest game model only (faster)")
    p.add_argument("--source", choices=["sportsdataverse", "nba_api", "synthetic"], default=None,
                   help="data source (default: NBA_DATA_SOURCE env var or sportsdataverse)")
    p.add_argument("--date", default=None, help="treat this YYYY-MM-DD as today")
    p.add_argument("--full-refresh", action="store_true", help="re-download every season (nba_api)")
    p.add_argument("--no-ingest", action="store_true", help="reuse cached raw data")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.date:
        os.environ["NBA_TODAY"] = args.date
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    from src import config
    from src.data.storage import Storage
    from src import pipeline

    if not (args.update_and_predict or args.backtest):
        print("Nothing to do: pass --update-and-predict and/or --backtest (see --help)")
        return 1
    source = args.source or config.DATA_SOURCE
    as_of = config.today()
    store = Storage()
    log = logging.getLogger("main")
    t0 = time.time()

    if not args.no_ingest:
        log.info("Ingesting data (source=%s, as_of=%s)", source, as_of)
        pipeline.ingest(store, source, as_of, full_refresh=args.full_refresh)
    log.info("Building features")
    fs = pipeline.load_features(store)

    if args.update_and_predict:
        log.info("Training final models and predicting upcoming games")
        print(json.dumps(pipeline.train_and_predict(fs, store)))
    if args.backtest:
        from src.backtest.run_backtest import run_backtest

        lines = store.read("lines")
        report = run_backtest(fs, lines, store, data_source=store.get_meta().get("data_source", source),
                              skip_props=args.skip_props_backtest)
        print(json.dumps({k: report[k] for k in ("data_source", "scores", "betting")}, indent=2, default=str))
    store.set_meta(last_run=datetime.now().isoformat(timespec="seconds"))
    log.info("Done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
