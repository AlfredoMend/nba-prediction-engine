"""
Run the full 2025-26 walk-forward backtest and write the JSON report + plots.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.backtest.metrics import betting_metrics, prop_metrics, probability_metrics, score_metrics
from src.backtest.plots import plot_all
from src.backtest.walk_forward import walk_forward_games, walk_forward_props
from src.data.build import FeatureSet
from src.data.storage import Storage

log = logging.getLogger(__name__)


def run_backtest(fs: FeatureSet, lines: pd.DataFrame, store: Storage, data_source: str,
                 season: str = config.BACKTEST_SEASON, skip_props: bool = False) -> dict:
    t0 = time.time()
    games = walk_forward_games(fs.games, lines, season=season)
    log.info("game backtest done in %.0fs", time.time() - t0)
    props = pd.DataFrame()
    if not skip_props:
        t1 = time.time()
        props = walk_forward_props(fs.players, season=season)
        log.info("props backtest done in %.0fs", time.time() - t1)

    report = {
        "season": season,
        "data_source": data_source,
        "data_source_note": ("SYNTHETIC simulated league - validates pipeline mechanics only; "
                             "not real NBA accuracy" if data_source == "synthetic" else "stats.nba.com via nba_api"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "type": "chronological walk-forward",
            "game_retrain_days": config.GAME_RETRAIN_DAYS,
            "prop_retrain_days": config.PROP_RETRAIN_DAYS,
            "monte_carlo_sims": config.N_SIMULATIONS,
            "leakage_check_passed": bool((games["train_cutoff"] < games["game_date"]).all()
                                         and (props.empty or (props["train_cutoff"] < props["game_date"]).all())),
        },
        "scores": score_metrics(games),
        "betting": betting_metrics(games),
        "win_probability": probability_metrics(games),
        "props": prop_metrics(props) if not props.empty else {},
    }
    store.write("bt_games", games)
    if not props.empty:
        store.write("bt_props", props)
        # calibrate prop over/under probabilities on this season's walk-forward predictions
        from src.models.prop_edges import PropCalibrator
        calibrator = PropCalibrator().fit(props)
        calibrator.save()
        report["prop_calibration"] = calibrator.report
        report["plots"] = [str(Path(p).relative_to(config.ROOT)) for p in plot_all(games, props, report)]
    config.BACKTEST_REPORT.write_text(json.dumps(report, indent=2, default=str))
    log.info("backtest report -> %s (%.0fs total)", config.BACKTEST_REPORT, time.time() - t0)
    return report
