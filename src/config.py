"""
Central configuration for the NBA Prediction Engine.

Every tunable constant lives here so that the data pipeline, models, backtest
and UI all agree on paths, seasons, feature windows and betting conventions.
Values can be overridden with environment variables (see `.env.example`).
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("NBA_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.getenv("NBA_DB_PATH", DATA_DIR / "nba.duckdb"))
MODELS_DIR = Path(os.getenv("NBA_MODELS_DIR", ROOT / "models_store"))
ASSETS_DIR = ROOT / "assets"
BACKTEST_REPORT = ROOT / "backtest_2025_2026_results.json"

for _d in (DATA_DIR, MODELS_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
# "sportsdataverse" -> ESPN box scores + schedules published daily on GitHub
#                      (github.com/sportsdataverse/sportsdataverse-data). Default.
# "nba_api"         -> stats.nba.com (often blocked from cloud/CI IP ranges)
# "synthetic"       -> deterministic simulated league with the same schema, used for
#                      offline development and unit tests. NOT real NBA results.
DATA_SOURCE = os.getenv("NBA_DATA_SOURCE", "sportsdataverse")

# --------------------------------------------------------------------------- #
# Seasons
# --------------------------------------------------------------------------- #
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
BACKTEST_SEASON = "2025-26"
CURRENT_SEASON = "2026-27"
SEASON_TYPES = ["Regular Season", "Playoffs"]


def today() -> date:
    """'Today' for the pipeline. Overridable for reproducible runs/backtests."""
    override = os.getenv("NBA_TODAY")
    return datetime.strptime(override, "%Y-%m-%d").date() if override else date.today()


def season_for_date(d: date | datetime) -> str:
    """Map a calendar date to an NBA season string, e.g. 2026-01-10 -> '2025-26'."""
    start_year = d.year if d.month >= 8 else d.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
EWMA_SPANS = (5, 10, 20)          # exponential moving average spans (in games)
MAX_REST_DAYS = 7                 # cap rest-day feature (offseason -> 7)
ROTATION_LOOKBACK_DAYS = 21       # a player is "in the rotation" if he played for the team this recently
ROTATION_MIN_MINUTES = 8.0        # ... and averages at least this many minutes
MAX_PLAYER_MINUTES = 44.0         # cap for minutes redistribution
TEAM_MINUTES = 240.0              # regulation team minutes (5 x 48)

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
N_SIMULATIONS = int(os.getenv("NBA_N_SIMS", 10_000))
RANDOM_SEED = 42
PROP_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "pra"]
QUANTILES = (0.10, 0.50, 0.90)    # 10th/50th/90th -> 80% interval

# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
GAME_RETRAIN_DAYS = int(os.getenv("NBA_GAME_RETRAIN_DAYS", 7))     # walk-forward step for game model
PROP_RETRAIN_DAYS = int(os.getenv("NBA_PROP_RETRAIN_DAYS", 28))    # walk-forward step for props model
EDGE_THRESHOLDS = (0.0, 2.0, 4.0)  # minimum |model - line| (points) required to place a bet

# --------------------------------------------------------------------------- #
# Betting conventions
# --------------------------------------------------------------------------- #
# Spreads are stored from the HOME team's perspective: -5.5 means home favoured
# by 5.5. Home covers when (home_pts - away_pts) + spread > 0.
STANDARD_ODDS = -110
WIN_PAYOUT = 100 / 110            # profit per 1 unit risked at -110
BREAKEVEN_WIN_RATE = 110 / 210    # ~52.38%

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
