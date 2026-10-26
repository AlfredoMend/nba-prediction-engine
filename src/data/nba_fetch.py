"""
Live data ingestion from stats.nba.com via `nba_api`.

Produces the normalized internal schema shared with the synthetic source:

team_games   season, season_type, game_id, game_date, team_id, team, opp_id, opp,
             is_home, min, pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb,
             ast, stl, blk, tov, pf
player_games season, season_type, game_id, game_date, player_id, player_name,
             team_id, opp_id, is_home, min, pts, reb, oreb, dreb, ast, stl, blk,
             tov, fgm, fga, fg3m, fg3a, ftm, fta, pf
players      player_id, player_name, position (G / F / C)
schedule     game_id, game_date, season, home_id, away_id, home, away, status

stats.nba.com rate-limits aggressively, so every call is wrapped in retries
with exponential backoff and a polite delay between requests.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from src import config
from src.data.teams import ABBR_TO_ID, TEAMS

log = logging.getLogger(__name__)

REQUEST_DELAY_S = 0.8
STAT_COLS = ["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
             "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]


def _call(fn, *args, retries: int = 4, **kwargs):
    """Call an nba_api endpoint with exponential backoff (2s, 4s, 8s, 16s)."""
    for attempt in range(retries + 1):
        try:
            out = fn(*args, timeout=60, **kwargs)
            time.sleep(REQUEST_DELAY_S)
            return out
        except Exception as exc:  # network errors, JSON decode errors, timeouts
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            log.warning("nba_api call failed (%s); retrying in %ss", exc, wait)
            time.sleep(wait)


def _normalize_log(df: pd.DataFrame, season: str, season_type: str, player: bool) -> pd.DataFrame:
    """Convert a LeagueGameLog frame (UPPERCASE columns) to the internal schema."""
    if df.empty:
        return df
    out = pd.DataFrame({
        "season": season,
        "season_type": season_type,
        "game_id": df["GAME_ID"].astype(str),
        "game_date": pd.to_datetime(df["GAME_DATE"]),
        "team_id": df["TEAM_ID"].astype("int64"),
        "team": df["TEAM_ABBREVIATION"],
        # MATCHUP is "LAL vs. BOS" for home games and "LAL @ BOS" for away games
        "is_home": df["MATCHUP"].str.contains(" vs. ").astype(int),
        "opp": df["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[-1].str.strip(),
    })
    out["opp_id"] = out["opp"].map(ABBR_TO_ID).astype("Int64")
    for c in STAT_COLS:
        out[c] = pd.to_numeric(df[c.upper()], errors="coerce")
    if player:
        out.insert(4, "player_id", df["PLAYER_ID"].astype("int64"))
        out.insert(5, "player_name", df["PLAYER_NAME"])
    return out


def fetch_game_logs(seasons: list[str] | None = None,
                    season_types: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download team and player game logs for every requested season."""
    from nba_api.stats.endpoints import leaguegamelog

    seasons = seasons or config.SEASONS
    season_types = season_types or config.SEASON_TYPES
    teams, players = [], []
    for season in seasons:
        for stype in season_types:
            for kind, bucket in (("T", teams), ("P", players)):
                log.info("Fetching %s %s logs (%s)", season, stype, kind)
                ep = _call(leaguegamelog.LeagueGameLog, season=season,
                           season_type_all_star=stype, player_or_team_abbreviation=kind)
                raw = ep.get_data_frames()[0]
                bucket.append(_normalize_log(raw, season, stype, player=(kind == "P")))
    team_games = pd.concat([t for t in teams if not t.empty], ignore_index=True)
    player_games = pd.concat([p for p in players if not p.empty], ignore_index=True)
    return team_games, player_games


def fetch_players(seasons: list[str] | None = None) -> pd.DataFrame:
    """Player -> primary position bucket (G/F/C) from the PlayerIndex endpoint."""
    from nba_api.stats.endpoints import playerindex

    frames = []
    for season in seasons or config.SEASONS:
        try:
            df = _call(playerindex.PlayerIndex, season=season).get_data_frames()[0]
        except Exception as exc:
            log.warning("PlayerIndex failed for %s: %s", season, exc)
            continue
        frames.append(pd.DataFrame({
            "player_id": df["PERSON_ID"].astype("int64"),
            "player_name": df["PLAYER_FIRST_NAME"] + " " + df["PLAYER_LAST_NAME"],
            "position": df["POSITION"].fillna("F").str[0].replace({"": "F"}),
        }))
    if not frames:
        return pd.DataFrame(columns=["player_id", "player_name", "position"])
    # latest season wins if a player's listed position changed
    return pd.concat(frames).drop_duplicates("player_id", keep="last").reset_index(drop=True)


def fetch_schedule(start: date | None = None, days: int = 3) -> pd.DataFrame:
    """Upcoming (not yet final) games for the next `days` days.

    Tries ScoreboardV3 first (current endpoint) and falls back to ScoreboardV2.
    """
    start = start or config.today()
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        try:
            rows.extend(_scoreboard_v3(d))
        except Exception as exc:
            log.warning("ScoreboardV3 failed for %s (%s); falling back to V2", d, exc)
            rows.extend(_scoreboard_v2(d))
    sched = pd.DataFrame(rows, columns=["game_id", "game_date", "season", "home_id",
                                        "away_id", "home", "away", "status"])
    if not sched.empty:
        sched["game_date"] = pd.to_datetime(sched["game_date"])
        sched = sched[~sched["status"].str.lower().str.startswith("final")]
    return sched.reset_index(drop=True)


def _row(gid, d, hid, aid, status):
    return (str(gid), d, config.season_for_date(d), int(hid), int(aid),
            TEAMS.get(int(hid), ("?",))[0], TEAMS.get(int(aid), ("?",))[0], str(status))


def _scoreboard_v3(d: date) -> list[tuple]:
    from nba_api.stats.endpoints import scoreboardv3

    data = _call(scoreboardv3.ScoreboardV3, game_date=d.isoformat(), league_id="00").get_dict()
    return [_row(g["gameId"], d, g["homeTeam"]["teamId"], g["awayTeam"]["teamId"], g.get("gameStatusText", ""))
            for g in data["scoreboard"]["games"]]


def _scoreboard_v2(d: date) -> list[tuple]:
    from nba_api.stats.endpoints import scoreboardv2

    hdr = _call(scoreboardv2.ScoreboardV2, game_date=d.isoformat()).game_header.get_data_frame()
    return [_row(r.GAME_ID, d, r.HOME_TEAM_ID, r.VISITOR_TEAM_ID, r.GAME_STATUS_TEXT)
            for r in hdr.itertuples()]
