"""
Market lines: Vegas sportsbooks vs Kalshi prediction markets.

Vegas
  ESPN scoreboard odds (free, no key): spread, total and moneylines from the
  book ESPN displays. The Odds API (consensus across books) is used instead
  when ODDS_API_KEY is set; see `odds.py`.

Kalshi  (https://api.elections.kalshi.com/trade-api/v2, public market data, no key)
  KXNBAGAME    one market per team per game: "will <team> win?"
  KXNBASPREAD  ladder per team: "will <team> win by over X points?"
  KXNBATOTAL   ladder per game: "will over X points be scored?"
  Prices are probabilities ($0.01-$0.99 per $1 contract). We turn them into
  a no-vig home win probability, an implied spread and an implied total, so
  they sit side by side with the Vegas numbers and the model.

Both sources are best effort: if a host is unreachable or a payload changes
shape, the functions log a warning and return an empty frame, and the rest of
the pipeline carries on.

Conventions: spreads are from the HOME team's perspective (-5.5 = home
favoured by 5.5); team codes are stats.nba.com abbreviations.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from src.data.teams import ABBR_TO_ID

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = {"winner": "KXNBAGAME", "spread": "KXNBASPREAD", "total": "KXNBATOTAL"}
KALSHI_FEE_RATE = 0.07      # taker fee = ceil(0.07 * C * P * (1 - P)) dollars, per Kalshi's fee schedule

# ESPN / Kalshi / other abbreviations -> stats.nba.com
ALT_ABBR = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "UTAH": "UTA", "NO": "NOP", "WSH": "WAS",
            "PHO": "PHX", "BRK": "BKN", "CHO": "CHA", "NOR": "NOP", "UTH": "UTA"}
LINE_COLS = ["game_date", "home", "away", "spread", "total", "home_ml", "away_ml", "book"]
KALSHI_COLS = ["game_date", "home", "away", "kalshi_event", "kalshi_home_prob", "kalshi_home_bid",
               "kalshi_home_ask", "kalshi_away_bid", "kalshi_away_ask", "kalshi_volume",
               "kalshi_spread", "kalshi_total"]


def norm_abbr(code: str | None) -> str | None:
    if not code:
        return None
    code = str(code).upper().strip()
    code = ALT_ABBR.get(code, code)
    return code if code in ABBR_TO_ID else None


# --------------------------------------------------------------------------- #
# Odds maths
# --------------------------------------------------------------------------- #
def american_to_prob(ml) -> float:
    """American odds -> implied probability (still includes the bookmaker's margin)."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(ml) or ml == 0:
        return float("nan")
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def no_vig(p_home: float, p_away: float) -> float:
    """Remove the overround by normalising the two implied probabilities."""
    s = p_home + p_away
    return p_home / s if s and not math.isnan(s) else float("nan")


def kalshi_fee(price: float, contracts: int = 1) -> float:
    """Kalshi taker fee in dollars for buying `contracts` at `price` (rounded up to the cent)."""
    if price is None or math.isnan(price) or not 0 < price < 1:
        return float("nan")
    return math.ceil(KALSHI_FEE_RATE * contracts * price * (1 - price) * 100 - 1e-9) / 100


def kalshi_ev(p_true: float, ask: float) -> float:
    """Expected profit per $1 contract bought at `ask`, after the fee."""
    if any(v is None or math.isnan(v) for v in (p_true, ask)) or not 0 < ask < 1:
        return float("nan")
    return p_true - ask - kalshi_fee(ask)


# --------------------------------------------------------------------------- #
# Vegas via ESPN scoreboard
# --------------------------------------------------------------------------- #
def _parse_details(details: str, home: str, away: str) -> float:
    """'BOS -5.5' -> home spread. 'EVEN'/'PK' -> 0."""
    if not details:
        return float("nan")
    if details.strip().upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = re.match(r"\s*([A-Za-z]+)\s+([-+]?\d+(?:\.\d+)?)", details)
    if not m:
        return float("nan")
    fav, pts = norm_abbr(m.group(1)), float(m.group(2))
    if fav == home:
        return pts
    if fav == away:
        return -pts
    return float("nan")


def _moneyline(side: dict, odds: dict, key: str):
    """Moneyline from either the classic (homeTeamOdds.moneyLine) or the newer
    (moneyline.home.close.odds) ESPN odds layout."""
    if side and side.get("moneyLine") is not None:
        return side["moneyLine"]
    ml = (odds.get("moneyline") or {}).get(key) or {}
    for k in ("close", "current", "open"):
        v = (ml.get(k) or {}).get("odds")
        if v not in (None, "", "OFF"):
            return str(v).replace("+", "")
    return None


def parse_espn_scoreboard(payload: dict) -> pd.DataFrame:
    rows = []
    for ev in payload.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): norm_abbr(c.get("team", {}).get("abbreviation")) for c in comp.get("competitors", [])}
        home, away = teams.get("home"), teams.get("away")
        odds_list = comp.get("odds") or []
        if not home or not away or not odds_list:
            continue
        o = odds_list[0]
        spread = _parse_details(o.get("details", ""), home, away)
        ml_h = _moneyline(o.get("homeTeamOdds") or {}, o, "home")
        ml_a = _moneyline(o.get("awayTeamOdds") or {}, o, "away")
        when = pd.Timestamp(ev.get("date") or comp.get("date"))
        if when.tzinfo is not None:  # ESPN times are UTC; NBA game dates are US/Eastern
            when = when.tz_convert("America/New_York").tz_localize(None)
        rows.append({"game_date": when.normalize(), "home": home, "away": away, "spread": spread,
                     "total": pd.to_numeric(o.get("overUnder"), errors="coerce"),
                     "home_ml": pd.to_numeric(ml_h, errors="coerce"), "away_ml": pd.to_numeric(ml_a, errors="coerce"),
                     "book": (o.get("provider") or {}).get("name", "ESPN")})
    return pd.DataFrame(rows, columns=LINE_COLS)


def fetch_espn_lines(start: date, days: int = 3) -> pd.DataFrame:
    frames = []
    for i in range(days):
        d = start + timedelta(days=i)
        try:
            r = requests.get(ESPN_SCOREBOARD, params={"dates": d.strftime("%Y%m%d")}, timeout=20)
            r.raise_for_status()
            frames.append(parse_espn_scoreboard(r.json()))
        except Exception as exc:
            log.warning("ESPN odds unavailable for %s (%s)", d, exc)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=LINE_COLS)
    out = pd.concat(frames, ignore_index=True)
    log.info("ESPN odds: %d games", len(out))
    return out


# --------------------------------------------------------------------------- #
# Kalshi
# --------------------------------------------------------------------------- #
def _price(m: dict, field: str) -> float:
    """Price in dollars. Newer payloads carry `<field>_dollars` strings, older ones integer cents."""
    v = m.get(f"{field}_dollars")
    if v not in (None, ""):
        return float(v)
    v = m.get(field)
    return float(v) / 100 if v not in (None, "") else float("nan")


def _mid(m: dict) -> float:
    """Best estimate of the market probability: bid/ask midpoint when the book is
    reasonably tight, otherwise the last traded price."""
    bid, ask, last = _price(m, "yes_bid"), _price(m, "yes_ask"), _price(m, "last_price")
    if 0 < bid < 1 and 0 < ask < 1 and ask - bid <= 0.15:
        return (bid + ask) / 2
    return last if 0 < last < 1 else float("nan")


_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
                                       "OCT", "NOV", "DEC"], start=1)}


def event_date(event_ticker: str) -> pd.Timestamp | None:
    """'KXNBAGAME-26OCT26WASCHA' -> 2026-10-26."""
    m = _DATE_RE.search(event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    return pd.Timestamp(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def _suffix(ticker: str) -> tuple[str | None, float | None]:
    """'...-OKC7' -> ('OKC', 7.0); '...-224' -> (None, 224.0); '...-HOU' -> ('HOU', None)."""
    tail = (ticker or "").rsplit("-", 1)[-1].upper()
    m = re.match(r"([A-Z]*)(\d+(?:\.\d+)?)?$", tail)
    if not m:
        return None, None
    code = norm_abbr(m.group(1)) if m.group(1) else None
    return code, float(m.group(2)) if m.group(2) else None


def _strike(m: dict, ticker_num: float | None) -> float | None:
    for k in ("floor_strike", "cap_strike"):
        if m.get(k) not in (None, ""):
            return float(m[k])
    text = " ".join(str(m.get(k, "")) for k in ("yes_sub_title", "subtitle", "title"))
    found = re.search(r"over (\d+(?:\.\d+)?)", text, re.I)
    if found:
        return float(found.group(1))
    return ticker_num - 0.5 if ticker_num is not None else None   # tickers use the integer above the .5 line


def fetch_kalshi_markets(series: str, status: str = "open", max_pages: int = 20) -> list[dict]:
    """All markets in a series (paginated)."""
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"series_ticker": series, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_API}/markets", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def _interp_half(points: list[tuple[float, float]]) -> float:
    """Given (x, P(X > x)) pairs, the x where the probability crosses 0.5 (the market's median)."""
    if len(points) < 2:
        return float("nan")
    pts = sorted(points)
    xs = np.array([p[0] for p in pts])
    ps = np.minimum.accumulate(np.array([p[1] for p in pts]))   # enforce a non-increasing survival curve
    if ps[0] < 0.5 or ps[-1] > 0.5:
        return float("nan")                                       # median lies outside the ladder
    # np.interp needs increasing x-coordinates: interpolate on the reversed curve
    return float(np.interp(0.5, ps[::-1], xs[::-1]))


def parse_kalshi(winner: list[dict], spread: list[dict], total: list[dict]) -> pd.DataFrame:
    """Combine the three NBA series into one row per game (teams unordered until matched)."""
    games: dict[str, dict] = {}

    def slot(event: str) -> dict:
        key = re.sub(r"^[A-Z]+-", "", event)          # same game across series: drop the series prefix
        return games.setdefault(key, {"event": event, "teams": {}, "spread": [], "total": [], "volume": 0.0})

    for m in winner:
        team, _ = _suffix(m.get("ticker"))
        if not team:
            continue
        g = slot(m.get("event_ticker", ""))
        g["teams"][team] = {"mid": _mid(m), "bid": _price(m, "yes_bid"), "ask": _price(m, "yes_ask")}
        g["volume"] += float(m.get("volume") or m.get("volume_fp") or 0)
    for m in spread:
        team, num = _suffix(m.get("ticker"))
        x, p = _strike(m, num), _mid(m)
        if team and x is not None and not math.isnan(p):
            slot(m.get("event_ticker", ""))["spread"].append((team, x, p))
    for m in total:
        _, num = _suffix(m.get("ticker"))
        x, p = _strike(m, num), _mid(m)
        if x is not None and not math.isnan(p):
            slot(m.get("event_ticker", ""))["total"].append((x, p))

    rows = []
    for key, g in games.items():
        teams = set(g["teams"]) | {t for t, _, _ in g["spread"]}
        d = event_date(g["event"])
        if len(teams) != 2 or d is None:
            continue
        rows.append({"key": key, "game_date": d, "teams": frozenset(teams), "event": g["event"],
                     "winner": g["teams"], "spread": g["spread"], "total": g["total"], "volume": g["volume"]})
    return pd.DataFrame(rows)


def kalshi_game_lines(parsed: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Match parsed Kalshi games to the schedule (same teams, date within a day) and
    express everything from the home team's perspective."""
    if parsed.empty or schedule.empty:
        return pd.DataFrame(columns=KALSHI_COLS)
    sched = schedule.assign(game_date=pd.to_datetime(schedule["game_date"]))
    rows = []
    for g in parsed.itertuples():
        cand = sched[(sched["home"].isin(g.teams)) & (sched["away"].isin(g.teams))
                     & ((sched["game_date"] - g.game_date).abs() <= pd.Timedelta(days=1))]
        if cand.empty:
            continue
        s = cand.iloc[0]
        home, away = s["home"], s["away"]
        wh, wa = g.winner.get(home, {}), g.winner.get(away, {})
        p_home = no_vig(wh.get("mid", np.nan), wa.get("mid", np.nan))
        if math.isnan(p_home):                                    # only one side quoted
            p_home = wh.get("mid", np.nan) if not math.isnan(wh.get("mid", np.nan)) else 1 - wa.get("mid", np.nan)
        # spread ladder -> P(home margin > x) points
        pts = []
        for team, x, p in g.spread:
            pts.append((x, p) if team == home else (-x, 1 - p))
        margin = _interp_half(pts)
        rows.append({"game_date": s["game_date"], "home": home, "away": away, "kalshi_event": g.event,
                     "kalshi_home_prob": p_home,
                     "kalshi_home_bid": wh.get("bid", np.nan), "kalshi_home_ask": wh.get("ask", np.nan),
                     "kalshi_away_bid": wa.get("bid", np.nan), "kalshi_away_ask": wa.get("ask", np.nan),
                     "kalshi_volume": g.volume,
                     "kalshi_spread": -margin if not math.isnan(margin) else np.nan,
                     "kalshi_total": _interp_half(list(g.total))})
    return pd.DataFrame(rows, columns=KALSHI_COLS)


def fetch_kalshi_lines(schedule: pd.DataFrame) -> pd.DataFrame:
    try:
        markets = {k: fetch_kalshi_markets(s) for k, s in KALSHI_SERIES.items()}
    except Exception as exc:
        log.warning("Kalshi API unavailable (%s)", exc)
        return pd.DataFrame(columns=KALSHI_COLS)
    log.info("Kalshi markets: %s", {k: len(v) for k, v in markets.items()})
    out = kalshi_game_lines(parse_kalshi(markets["winner"], markets["spread"], markets["total"]), schedule)
    log.info("Kalshi: matched %d of %d scheduled games", len(out), len(schedule))
    return out


# --------------------------------------------------------------------------- #
# Attach to predictions
# --------------------------------------------------------------------------- #
def attach_markets(preds: pd.DataFrame, lines: pd.DataFrame, kalshi: pd.DataFrame) -> pd.DataFrame:
    """Add Vegas + Kalshi columns and model-vs-market comparisons to game predictions.
    Always returns the full set of columns (NaN where a market is missing)."""
    out = preds.drop(columns=[c for c in ("spread", "total") if c in preds], errors="ignore")
    keys = ["game_date", "home", "away"]
    out["game_date"] = pd.to_datetime(out["game_date"])
    lines = (lines if lines is not None and not lines.empty else pd.DataFrame(columns=LINE_COLS)).copy()
    for c in LINE_COLS:
        if c not in lines:
            lines[c] = np.nan
    lines["game_date"] = pd.to_datetime(lines["game_date"])
    out = out.merge(lines[LINE_COLS].drop_duplicates(keys, keep="last"), on=keys, how="left")
    kalshi = (kalshi if kalshi is not None and not kalshi.empty else pd.DataFrame(columns=KALSHI_COLS)).copy()
    kalshi["game_date"] = pd.to_datetime(kalshi["game_date"])
    out = out.merge(kalshi[KALSHI_COLS].drop_duplicates(keys, keep="last"), on=keys, how="left")
    for c in ("spread", "total", "home_ml", "away_ml", *KALSHI_COLS[3:]):
        if c != "kalshi_event":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["vegas_home_prob"] = [no_vig(american_to_prob(h), american_to_prob(a))
                              for h, a in zip(out["home_ml"], out["away_ml"])]
    return out


def add_kalshi_value(preds: pd.DataFrame) -> pd.DataFrame:
    """Expected value per $1 Kalshi contract (after fees) of backing the side the model prefers."""
    out = preds.copy()
    ev_home = [kalshi_ev(p, a) for p, a in zip(out["home_win_prob"], out["kalshi_home_ask"])]
    ev_away = [kalshi_ev(1 - p, a) for p, a in zip(out["home_win_prob"], out["kalshi_away_ask"])]
    out["kalshi_ev_home"], out["kalshi_ev_away"] = ev_home, ev_away
    best_home = np.nan_to_num(ev_home, nan=-9) >= np.nan_to_num(ev_away, nan=-9)
    out["kalshi_best_side"] = np.where(np.isnan(ev_home) & np.isnan(ev_away), "",
                                       np.where(best_home, out["home"], out["away"]))
    out["kalshi_best_ev"] = np.where(best_home, ev_home, ev_away)
    out["model_minus_kalshi"] = out["home_win_prob"] - out["kalshi_home_prob"]
    out["model_minus_vegas"] = out["home_win_prob"] - out["vegas_home_prob"]
    out["vegas_minus_kalshi"] = out["vegas_home_prob"] - out["kalshi_home_prob"]
    return out
