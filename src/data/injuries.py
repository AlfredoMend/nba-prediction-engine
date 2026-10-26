"""
Injury report ingestion.

Sources, merged in order (later wins per player):
  1. ESPN's public injuries feed (site.api.espn.com) - best effort; skipped
     silently if the host is unreachable or the payload changes
  2. data/injuries.csv (player_name, team, status) - manual overrides

Players whose status is OUT or DOUBTFUL are removed from projected lineups;
their minutes/usage are redistributed by `lineup.redistribute`.
"""
from __future__ import annotations

import logging
import re

import pandas as pd
import requests

from src import config
from src.data.sportsdataverse import ESPN_ABBR

log = logging.getLogger(__name__)

ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
INJURIES_CSV = config.DATA_DIR / "injuries.csv"
COLS = ["player_id", "player_name", "team", "status", "detail", "source"]


def _athlete_id(athlete: dict) -> int | None:
    if str(athlete.get("id", "")).isdigit():
        return int(athlete["id"])
    for link in athlete.get("links", []):
        m = re.search(r"/id/(\d+)", link.get("href", ""))
        if m:
            return int(m.group(1))
    return None


def fetch_espn_injuries() -> pd.DataFrame:
    try:
        resp = requests.get(ESPN_INJURIES, timeout=20)
        resp.raise_for_status()
        rows = []
        for team in resp.json().get("injuries", []):
            for inj in team.get("injuries", []):
                ath = inj.get("athlete", {})
                abbr = ath.get("team", {}).get("abbreviation", "")
                rows.append({"player_id": _athlete_id(ath), "player_name": ath.get("displayName"),
                             "team": ESPN_ABBR.get(abbr, abbr), "status": str(inj.get("status", "")).upper(),
                             "detail": inj.get("shortComment", ""), "source": "espn"})
        log.info("ESPN injury report: %d entries", len(rows))
        return pd.DataFrame(rows, columns=COLS)
    except Exception as exc:  # network blocked, schema change, ...
        log.warning("ESPN injury feed unavailable (%s); using data/injuries.csv only", exc)
        return pd.DataFrame(columns=COLS)


def load_injuries(player_games: pd.DataFrame, use_espn: bool = True) -> pd.DataFrame:
    """Combined injury report with player ids resolved against our box scores."""
    frames = [fetch_espn_injuries()] if use_espn else []
    if INJURIES_CSV.exists():
        manual = pd.read_csv(INJURIES_CSV).assign(source="manual", detail="")
        manual["status"] = manual["status"].str.upper()
        frames.append(manual)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLS)
    inj = pd.concat(frames, ignore_index=True)
    by_name = player_games.drop_duplicates("player_name", keep="last").set_index("player_name")["player_id"]
    inj["player_id"] = pd.to_numeric(inj.get("player_id"), errors="coerce").fillna(inj["player_name"].map(by_name))
    inj = inj.dropna(subset=["player_id"]).astype({"player_id": "int64"})
    return inj.drop_duplicates("player_id", keep="last")[COLS].reset_index(drop=True)
