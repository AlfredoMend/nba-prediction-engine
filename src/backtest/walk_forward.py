"""
Strict chronological walk-forward backtest.

The evaluation season is cut into consecutive windows of `step_days`. For each
window:
  1. train a fresh model on every completed game with game_date < window_start
     (all prior seasons + the evaluation season so far),
  2. predict every game inside [window_start, window_end),
  3. record `train_cutoff` on each prediction so leakage can be audited.

Features were built with only pre-game information (shifted EWMAs, merge_asof
with allow_exact_matches=False), so the only remaining leakage vector - fitting
on future targets - is ruled out by step 1. Unit tests assert this invariant.
"""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from src import config
from src.models.game_model import GameModel
from src.models.monte_carlo import simulate_games
from src.models.props_model import PropsModel

log = logging.getLogger(__name__)


def _windows(dates: pd.Series, step_days: int):
    start, last = dates.min().normalize(), dates.max().normalize()
    while start <= last:
        end = start + pd.Timedelta(days=step_days)
        yield start, end
        start = end


def walk_forward_games(games: pd.DataFrame, lines: pd.DataFrame | None = None,
                       season: str = config.BACKTEST_SEASON, step_days: int = config.GAME_RETRAIN_DAYS,
                       model_factory: Callable[[], GameModel] = GameModel,
                       n_sims: int = config.N_SIMULATIONS, min_train: int = 300) -> pd.DataFrame:
    """Walk-forward predictions + Monte Carlo probabilities for every game of `season`.
    Windows with fewer than `min_train` completed games of history are skipped."""
    done = games.dropna(subset=["home_pts", "away_pts"])
    test = done[done["season"] == season]
    if test.empty:
        raise ValueError(f"no completed games for season {season}")
    out = []
    for i, (start, end) in enumerate(_windows(test["game_date"], step_days)):
        window = test[(test["game_date"] >= start) & (test["game_date"] < end)]
        if window.empty:
            continue
        train = done[done["game_date"] < start]
        if len(train) < min_train:
            log.warning("skipping window %s: only %d training games", start.date(), len(train))
            continue
        model = model_factory().fit(train)
        pred = model.predict(window)
        pred["train_cutoff"] = train["game_date"].max()
        pred["n_train"] = len(train)
        pred = pred.merge(window[["game_id", "home_pts", "away_pts"]], on="game_id")
        if lines is not None and not lines.empty:
            pred = pred.merge(lines[["game_date", "home", "away", "spread", "total"]],
                              on=["game_date", "home", "away"], how="left")
        else:
            pred["spread"], pred["total"] = float("nan"), float("nan")
        pred = simulate_games(pred, model.sigma_home, model.sigma_away, model.rho, n_sims=n_sims, seed=i)
        out.append(pred)
        log.info("game window %s -> %s: %d games (train n=%d)", start.date(), end.date(), len(window), len(train))
    return pd.concat(out, ignore_index=True)


def walk_forward_props(players: pd.DataFrame, season: str = config.BACKTEST_SEASON,
                       step_days: int = config.PROP_RETRAIN_DAYS,
                       model_factory: Callable[[], PropsModel] = PropsModel,
                       min_train: int = 2000) -> pd.DataFrame:
    """Walk-forward quantile projections for every player-game of `season`."""
    done = players[(players["is_upcoming"] == 0) & players["min"].notna()]
    test = done[done["season"] == season]
    out = []
    for start, end in _windows(test["game_date"], step_days):
        window = test[(test["game_date"] >= start) & (test["game_date"] < end)]
        if window.empty:
            continue
        train = done[done["game_date"] < start]
        if len(train) < min_train:
            continue
        model = model_factory().fit(train)
        pred = model.predict(window)
        for s in config.PROP_STATS:
            pred[s] = window[s].values
        pred["actual_min"] = window["min"].values
        pred["train_cutoff"] = train["game_date"].max()
        out.append(pred)
        log.info("props window %s -> %s: %d player-games", start.date(), end.date(), len(window))
    return pd.concat(out, ignore_index=True)
