"""
Player prop lines with prices, from any book.

Sources (merged; later wins for the same player/stat/book):
  1. data/prop_lines.csv  - lines you paste in, e.g. from Novig:
         game_date,player_name,stat,line,over_odds,under_odds,book
         2026-10-27,Jayson Tatum,pts,23.5,-105,-105,novig
     odds are American; if omitted, -110 is assumed.
  2. The Odds API player-prop markets (needs ODDS_API_KEY). Books are chosen
     with ODDS_PROP_BOOKMAKERS (comma-separated keys, e.g. "novig,draftkings").
     Each event x market costs API credits, so props are fetched only for the
     next game day.

Player names are matched to our player ids after normalisation (case, accents,
punctuation and Jr./III suffixes are ignored).
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata

import pandas as pd
import requests

from src import config
from src.data.teams import NAME_TO_ID

log = logging.getLogger(__name__)

PROP_LINES_CSV = config.DATA_DIR / "prop_lines.csv"
COLS = ["game_date", "player_id", "player_name", "stat", "line", "over_odds", "under_odds", "book"]
# The Odds API market keys -> our stat names
ODDS_API_MARKETS = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
                    "player_threes": "fg3m", "player_steals": "stl", "player_blocks": "blk",
                    "player_points_rebounds_assists": "pra"}
STAT_ALIASES = {"points": "pts", "pts": "pts", "rebounds": "reb", "reb": "reb", "assists": "ast", "ast": "ast",
                "threes": "fg3m", "3pm": "fg3m", "fg3m": "fg3m", "3-pt made": "fg3m", "steals": "stl", "stl": "stl",
                "blocks": "blk", "blk": "blk", "pra": "pra", "pts+reb+ast": "pra", "points+rebounds+assists": "pra"}


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def attach_player_ids(lines: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Fill player_id from names (players: player_id, player_name)."""
    lookup = (players.assign(key=players["player_name"].map(normalize_name))
              .drop_duplicates("key", keep="last").set_index("key")["player_id"])
    out = lines.copy()
    ids = out["player_name"].map(normalize_name).map(lookup)
    out["player_id"] = pd.to_numeric(out.get("player_id"), errors="coerce").fillna(ids)
    missing = out["player_id"].isna()
    if missing.any():
        log.warning("prop lines: %d rows with unknown players, e.g. %s", int(missing.sum()),
                    out.loc[missing, "player_name"].head(5).tolist())
    return out[~missing].astype({"player_id": "int64"})


def load_prop_lines_csv() -> pd.DataFrame:
    if not PROP_LINES_CSV.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(PROP_LINES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["stat"] = df["stat"].astype(str).str.strip().str.lower().map(lambda s: STAT_ALIASES.get(s, s))
    for c in ("over_odds", "under_odds", "player_id"):
        if c not in df:
            df[c] = pd.NA
    if "book" not in df:
        df["book"] = "manual"
    return df.reindex(columns=COLS)


def _odds_api_bookmakers() -> str:
    return os.getenv("ODDS_PROP_BOOKMAKERS", "").strip()


def fetch_odds_api_props(next_game_day: pd.Timestamp | None) -> pd.DataFrame:
    """Player props for the next game day's events from The Odds API."""
    books = _odds_api_bookmakers()
    if not config.ODDS_API_KEY or not books or next_game_day is None:
        return pd.DataFrame(columns=COLS)
    base = "https://api.the-odds-api.com/v4/sports/basketball_nba"
    try:
        events = requests.get(f"{base}/events", params={"apiKey": config.ODDS_API_KEY}, timeout=30)
        events.raise_for_status()
    except Exception as exc:
        log.warning("The Odds API events unavailable (%s)", exc)
        return pd.DataFrame(columns=COLS)
    rows, seen_books = [], set()
    for ev in events.json():
        day = pd.Timestamp(ev["commence_time"]).tz_convert("America/New_York").tz_localize(None).normalize()
        if day != next_game_day or ev.get("home_team") not in NAME_TO_ID:
            continue
        try:
            r = requests.get(f"{base}/events/{ev['id']}/odds", timeout=30, params={
                "apiKey": config.ODDS_API_KEY, "bookmakers": books, "markets": ",".join(ODDS_API_MARKETS),
                "oddsFormat": "american"})
            r.raise_for_status()
        except Exception as exc:
            log.warning("prop odds failed for %s (%s)", ev.get("id"), exc)
            continue
        for bk in r.json().get("bookmakers", []):
            seen_books.add(bk["key"])
            for mk in bk.get("markets", []):
                stat = ODDS_API_MARKETS.get(mk["key"])
                if not stat:
                    continue
                book_rows: dict[tuple, dict] = {}
                for oc in mk.get("outcomes", []):
                    key = (oc.get("description"), oc.get("point"))
                    row = book_rows.setdefault(key, {"game_date": day, "player_id": pd.NA, "player_name": key[0],
                                                     "stat": stat, "line": key[1], "over_odds": pd.NA,
                                                     "under_odds": pd.NA, "book": bk["key"]})
                    row["over_odds" if oc.get("name") == "Over" else "under_odds"] = oc.get("price")
                rows.extend(book_rows.values())
    log.info("Odds API props: %d lines from books %s (requested %s)", len(rows), sorted(seen_books), books)
    return pd.DataFrame(rows, columns=COLS)


def load_prop_lines(players: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """All prop lines for upcoming games, with player ids resolved."""
    next_day = pd.to_datetime(schedule["game_date"]).min() if schedule is not None and not schedule.empty else None
    frames = [f for f in (load_prop_lines_csv(), fetch_odds_api_props(next_day)) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLS)
    lines = pd.concat(frames, ignore_index=True).dropna(subset=["line"])
    lines = attach_player_ids(lines, players)
    return lines.drop_duplicates(["game_date", "player_id", "stat", "book"], keep="last").reset_index(drop=True)
