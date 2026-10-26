"""
GitHub-hosted NBA data from the sportsdataverse project (hoopR / ESPN).

https://github.com/sportsdataverse/sportsdataverse-data publishes ESPN team box
scores, player box scores and full schedules as parquet files on GitHub
releases, refreshed daily during the season by the project's own automation.
Downloads come from github.com, so this source works from GitHub Actions and
most cloud hosts where stats.nba.com is blocked.

Produces the same normalized schema as `nba_fetch` (see that module).
ESPN season N == NBA season "(N-1)-N", e.g. ESPN 2026 -> "2025-26".
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import config
from src.data.teams import ABBR_TO_ID, TEAMS

log = logging.getLogger(__name__)

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
FILES = {
    "team": "espn_nba_team_boxscores/team_box_{year}.parquet",
    "player": "espn_nba_player_boxscores/player_box_{year}.parquet",
    "schedule": "espn_nba_schedules/nba_schedule_{year}.parquet",
}
RAW_DIR = config.DATA_DIR / "raw"

# ESPN abbreviations that differ from stats.nba.com
ESPN_ABBR = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "UTAH": "UTA", "NO": "NOP", "WSH": "WAS"}
SEASON_TYPES = {2: "Regular Season", 3: "Playoffs", 5: "PlayIn"}   # 1 = preseason (dropped)
POSITION = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}


def espn_year(season: str) -> int:
    """'2025-26' -> 2026."""
    return int(season[:4]) + 1


def nba_season(year: int) -> str:
    """2026 -> '2025-26'."""
    return f"{year - 1}-{year % 100:02d}"


def _abbr(espn: pd.Series) -> pd.Series:
    return espn.replace(ESPN_ABBR)


def download(kind: str, year: int, refresh: bool = False, retries: int = 4) -> pd.DataFrame:
    """Download (or read the cached copy of) one season file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / Path(FILES[kind].format(year=year)).name
    if refresh or not path.exists():
        url = f"{BASE}/{FILES[kind].format(year=year)}"
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 404:
                    log.warning("%s not published yet", url)
                    return pd.DataFrame()
                r.raise_for_status()
                path.write_bytes(r.content)
                break
            except requests.RequestException as exc:
                if attempt == retries:
                    raise
                wait = 2 ** (attempt + 1)
                log.warning("download failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalize_player_box(raw: pd.DataFrame) -> pd.DataFrame:
    """ESPN player box -> internal player_games (players who actually played)."""
    df = raw[raw["season_type"].isin(SEASON_TYPES)].copy()
    df["team"] = _abbr(df["team_abbreviation"])
    df["opp"] = _abbr(df["opponent_team_abbreviation"])
    df = df[df["team"].isin(ABBR_TO_ID) & df["opp"].isin(ABBR_TO_ID)]          # drops All-Star games
    df = df[~df["did_not_play"].fillna(False).astype(bool) & (df["minutes"].fillna(0) > 0)]
    out = pd.DataFrame({
        "season": df["season"].astype(int).map(nba_season),
        "season_type": df["season_type"].map(SEASON_TYPES),
        "game_id": df["game_id"].astype(str),
        "game_date": pd.to_datetime(df["game_date"]),
        "player_id": df["athlete_id"].astype("int64"),
        "player_name": df["athlete_display_name"],
        "team_id": df["team"].map(ABBR_TO_ID).astype("int64"),
        "opp_id": df["opp"].map(ABBR_TO_ID).astype("int64"),
        "is_home": (df["home_away"] == "home").astype(int),
        "min": df["minutes"].astype(float),
    })
    cols = {"points": "pts", "rebounds": "reb", "offensive_rebounds": "oreb", "defensive_rebounds": "dreb",
            "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
            "field_goals_made": "fgm", "field_goals_attempted": "fga", "three_point_field_goals_made": "fg3m",
            "three_point_field_goals_attempted": "fg3a", "free_throws_made": "ftm",
            "free_throws_attempted": "fta", "fouls": "pf"}
    for src, dst in cols.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0).values
    out["_position"] = df["athlete_position_abbreviation"].map(POSITION).fillna("F").values
    return out.reset_index(drop=True)


def normalize_team_box(raw: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """ESPN team box -> internal team_games. Team minutes come from player minutes
    (240 regulation + 25 per overtime), since ESPN team boxes omit them."""
    df = raw[raw["season_type"].isin(SEASON_TYPES)].copy()
    df["team"] = _abbr(df["team_abbreviation"])
    df["opp"] = _abbr(df["opponent_team_abbreviation"])
    df = df[df["team"].isin(ABBR_TO_ID) & df["opp"].isin(ABBR_TO_ID)]
    out = pd.DataFrame({
        "season": df["season"].astype(int).map(nba_season),
        "season_type": df["season_type"].map(SEASON_TYPES),
        "game_id": df["game_id"].astype(str),
        "game_date": pd.to_datetime(df["game_date"]),
        "team_id": df["team"].map(ABBR_TO_ID).astype("int64"),
        "team": df["team"],
        "opp_id": df["opp"].map(ABBR_TO_ID).astype("int64"),
        "opp": df["opp"],
        "is_home": (df["team_home_away"] == "home").astype(int),
    })
    cols = {"team_score": "pts", "field_goals_made": "fgm", "field_goals_attempted": "fga",
            "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
            "free_throws_made": "ftm", "free_throws_attempted": "fta", "offensive_rebounds": "oreb",
            "defensive_rebounds": "dreb", "total_rebounds": "reb", "assists": "ast", "steals": "stl",
            "blocks": "blk", "fouls": "pf"}
    for src, dst in cols.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce").values
    # total_turnovers includes team turnovers (shot clock etc.) - what possessions need
    tov = pd.to_numeric(df["total_turnovers"], errors="coerce")
    out["tov"] = tov.fillna(pd.to_numeric(df["turnovers"], errors="coerce")).values
    mins = player_games.groupby(["game_id", "team_id"])["min"].sum()
    played = pd.Series(list(zip(out["game_id"], out["team_id"]))).map(mins).values
    ot_periods = np.clip(np.round((np.nan_to_num(played, nan=240.0) - 240) / 25), 0, None)
    out.insert(9, "min", 240 + 25 * ot_periods)
    out = out.dropna(subset=["pts", "fga"])
    # keep only games where both sides are present
    both = out.groupby("game_id")["team_id"].transform("count") == 2
    return out[both].reset_index(drop=True)


def normalize_schedule(raw: pd.DataFrame, start: date, days: int) -> pd.DataFrame:
    """Not-yet-completed games in [start, start + days)."""
    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home"] = _abbr(df["home_abbreviation"])
    df["away"] = _abbr(df["away_abbreviation"])
    end = pd.Timestamp(start + timedelta(days=days))
    df = df[(~df["status_type_completed"].astype(bool)) & (df["game_date"] >= pd.Timestamp(start))
            & (df["game_date"] < end) & df["home"].isin(ABBR_TO_ID) & df["away"].isin(ABBR_TO_ID)
            & df["season_type"].isin(SEASON_TYPES)]
    return pd.DataFrame({
        "game_id": df["game_id"].astype(str), "game_date": df["game_date"],
        "season": df["season"].astype(int).map(nba_season),
        "home_id": df["home"].map(ABBR_TO_ID).astype("int64"), "away_id": df["away"].map(ABBR_TO_ID).astype("int64"),
        "home": df["home"], "away": df["away"], "status": df["status_type_detail"].astype(str),
    }).sort_values(["game_date", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fetch_all(seasons: list[str] | None = None, as_of: date | None = None, refresh_current: bool = True,
              schedule_days: int = 3):
    """Return (team_games, player_games, players, schedule).

    Past seasons are cached under data/raw/; the current season is re-downloaded
    on every run so new games appear as soon as sportsdataverse publishes them.
    """
    seasons = seasons or config.SEASONS
    as_of = as_of or config.today()
    current = espn_year(config.season_for_date(as_of))
    teams, players = [], []
    for season in seasons:
        year = espn_year(season)
        refresh = refresh_current and year >= current
        pg_raw = download("player", year, refresh)
        tg_raw = download("team", year, refresh)
        if pg_raw.empty or tg_raw.empty:
            continue
        pg = normalize_player_box(pg_raw)
        players.append(pg)
        teams.append(normalize_team_box(tg_raw, pg))
        log.info("season %s: %d team rows, %d player rows", season, len(teams[-1]), len(pg))
    player_games = pd.concat(players, ignore_index=True)
    team_games = pd.concat(teams, ignore_index=True)
    # completed games only, and only those with box scores on both sides
    player_games = player_games[player_games["game_id"].isin(team_games["game_id"])]
    player_games = player_games[player_games["game_date"] < pd.Timestamp(as_of)]
    team_games = team_games[team_games["game_date"] < pd.Timestamp(as_of)]

    roster = (player_games.sort_values("game_date")
              .drop_duplicates("player_id", keep="last")[["player_id", "player_name", "_position"]]
              .rename(columns={"_position": "position"}))
    player_games = player_games.drop(columns="_position")

    sched_raw = download("schedule", current, refresh=refresh_current)
    schedule = normalize_schedule(sched_raw, as_of, schedule_days) if not sched_raw.empty else pd.DataFrame()
    return team_games.reset_index(drop=True), player_games.reset_index(drop=True), roster, schedule


def data_freshness(team_games: pd.DataFrame) -> str:
    """Latest completed game date in the dataset (for the UI / logs)."""
    return str(pd.to_datetime(team_games["game_date"]).max().date()) if not team_games.empty else "n/a"


__all__ = ["fetch_all", "data_freshness", "TEAMS"]
