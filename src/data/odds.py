"""
Betting lines.

`nba_api` carries no odds, so lines come from one of:
  1. `data/lines.csv` / `data/prop_lines.csv` (user-supplied, e.g. exported
     historical closing lines for the 2025-26 backtest), or
  2. The Odds API (https://the-odds-api.com) for live lines when ODDS_API_KEY is set.

Spread convention: home-team perspective (-5.5 = home favoured by 5.5).
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from src import config
from src.data.teams import NAME_TO_ID, TEAMS

log = logging.getLogger(__name__)

LINES_CSV = config.DATA_DIR / "lines.csv"            # game_date,home,away,spread,total
PROP_LINES_CSV = config.DATA_DIR / "prop_lines.csv"  # game_date,player_name,stat,line


def load_lines_csv() -> pd.DataFrame:
    if not LINES_CSV.exists():
        return pd.DataFrame(columns=["game_date", "home", "away", "spread", "total"])
    df = pd.read_csv(LINES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_prop_lines_csv() -> pd.DataFrame:
    if not PROP_LINES_CSV.exists():
        return pd.DataFrame(columns=["game_date", "player_name", "stat", "line"])
    df = pd.read_csv(PROP_LINES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def fetch_live_lines() -> pd.DataFrame:
    """Consensus (median across books) spread & total for upcoming games."""
    if not config.ODDS_API_KEY:
        return pd.DataFrame(columns=["game_date", "home", "away", "spread", "total"])
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {"apiKey": config.ODDS_API_KEY, "regions": "us", "markets": "spreads,totals",
              "oddsFormat": "american"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    rows = []
    for ev in resp.json():
        hid, aid = NAME_TO_ID.get(ev["home_team"]), NAME_TO_ID.get(ev["away_team"])
        if not hid or not aid:
            continue
        spreads, totals = [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk["outcomes"]:
                    if mk["key"] == "spreads" and oc["name"] == ev["home_team"]:
                        spreads.append(oc["point"])
                    elif mk["key"] == "totals" and oc["name"] == "Over":
                        totals.append(oc["point"])
        rows.append({
            # commence_time is UTC; convert to US/Eastern calendar date
            "game_date": pd.Timestamp(ev["commence_time"]).tz_convert("US/Eastern").tz_localize(None).normalize(),
            "home": TEAMS[hid][0], "away": TEAMS[aid][0],
            "spread": float(pd.Series(spreads).median()) if spreads else None,
            "total": float(pd.Series(totals).median()) if totals else None,
        })
    return pd.DataFrame(rows)
