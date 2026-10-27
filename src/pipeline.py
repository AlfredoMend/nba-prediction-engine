"""
End-to-end orchestration: ingest -> features -> train -> predict -> persist.

Used by `main.py` (CLI), the GitHub Actions daily job and the Docker image.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from src import config
from src.data import markets, odds, prop_odds
from src.data.build import FeatureSet, build_features
from src.data.storage import Storage
from src.models.game_model import GameModel
from src.models.monte_carlo import simulate_games
from src.models.prop_edges import PropCalibrator, evaluate_prop_bets
from src.models.props_model import PropsModel

log = logging.getLogger(__name__)
INJURIES_CSV = config.DATA_DIR / "injuries.csv"   # player_name,team,status


# --------------------------------------------------------------------------- #
# 1. Ingestion
# --------------------------------------------------------------------------- #
def ingest(store: Storage, source: str, as_of: date, full_refresh: bool = False) -> None:
    """Refresh raw tables in the local DuckDB cache from the chosen source."""
    if source == "synthetic":
        from src.data.synthetic import generate_league

        tg, pg, players, sched, lines, prop_lines, inj = generate_league(through=as_of)
        for name, df in (("raw_team_games", tg), ("raw_player_games", pg), ("players", players),
                         ("schedule", sched), ("lines", lines), ("prop_lines", prop_lines), ("injuries", inj)):
            store.write(name, df)
    elif source == "sportsdataverse":
        from src.data import sportsdataverse
        from src.data.injuries import load_injuries

        tg, pg, players, sched = sportsdataverse.fetch_all(as_of=as_of, refresh_current=True)
        store.write("raw_team_games", tg)
        store.write("raw_player_games", pg)
        store.write("players", players)
        store.write("schedule", sched)
        refresh_markets(store, sched, as_of)
        store.write("prop_lines", prop_odds.load_prop_lines(pg[["player_id", "player_name"]], sched))
        store.write("injuries", load_injuries(pg))
        store.set_meta(latest_game=sportsdataverse.data_freshness(tg))
    elif source == "nba_api":
        from src.data import nba_fetch

        have = store.exists("raw_team_games") and not full_refresh
        # incremental: only re-download the active season once history is cached
        seasons = [config.season_for_date(as_of)] if have else config.SEASONS
        tg, pg = nba_fetch.fetch_game_logs(seasons)
        if have:
            old_tg, old_pg = store.read("raw_team_games"), store.read("raw_player_games")
            tg = pd.concat([old_tg[~old_tg["season"].isin(seasons)], tg], ignore_index=True)
            pg = pd.concat([old_pg[~old_pg["season"].isin(seasons)], pg], ignore_index=True)
        store.write("raw_team_games", tg)
        store.write("raw_player_games", pg)
        players = nba_fetch.fetch_players(seasons if have else None)
        store.write("players", _merge_players(store.read("players"), players) if have else players)
        sched = nba_fetch.fetch_schedule(as_of, days=3)
        store.write("schedule", sched)
        refresh_markets(store, sched, as_of)
        store.write("prop_lines", prop_odds.load_prop_lines(pg[["player_id", "player_name"]], sched))
        store.write("injuries", _load_injuries(pg))
    else:
        raise ValueError(f"unknown data source {source!r}")
    store.set_meta(data_source=source, as_of=as_of.isoformat(),
                   ingested_at=datetime.now().isoformat(timespec="seconds"))


def refresh_markets(store: Storage, schedule: pd.DataFrame, as_of: date) -> None:
    """Vegas lines (CSV < ESPN < The Odds API, later sources win) and Kalshi prices.
    Every run also appends a timestamped snapshot to `line_history`, building the
    record of market lines needed to backtest against Vegas and Kalshi later."""
    frames = [f for f in (odds.load_lines_csv(), markets.fetch_espn_lines(as_of, days=3), odds.fetch_live_lines())
              if not f.empty]
    if frames:
        lines = pd.concat(frames, ignore_index=True)
        lines = (lines.assign(game_date=pd.to_datetime(lines["game_date"]))
                 .drop_duplicates(["game_date", "home", "away"], keep="last"))
    else:
        lines = pd.DataFrame(columns=markets.LINE_COLS)
    kalshi = markets.fetch_kalshi_lines(schedule)
    store.write("lines", lines)
    store.write("kalshi", kalshi)
    snap = markets.attach_markets(schedule[["game_date", "home", "away"]].copy(), lines, kalshi)
    snap["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    store.write("line_history", snap, mode="append")
    store.set_meta(vegas_games=int(lines["spread"].notna().sum()) if not lines.empty else 0,
                   kalshi_games=len(kalshi))


def _merge_players(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([old, new]).drop_duplicates("player_id", keep="last")


def _load_injuries(pg: pd.DataFrame) -> pd.DataFrame:
    """Read data/injuries.csv (player_name, team, status) and attach player ids."""
    cols = ["player_id", "player_name", "team", "status"]
    if not INJURIES_CSV.exists():
        return pd.DataFrame(columns=cols)
    inj = pd.read_csv(INJURIES_CSV)
    ids = pg.drop_duplicates("player_name", keep="last").set_index("player_name")["player_id"]
    inj["player_id"] = inj["player_name"].map(ids)
    return inj.dropna(subset=["player_id"]).astype({"player_id": "int64"})[cols]


# --------------------------------------------------------------------------- #
# 2. Features
# --------------------------------------------------------------------------- #
def load_features(store: Storage) -> FeatureSet:
    return build_features(store.read("raw_team_games"), store.read("raw_player_games"), store.read("players"),
                          store.read("schedule"), store.read("injuries"))


# --------------------------------------------------------------------------- #
# 3. Train + predict upcoming games
# --------------------------------------------------------------------------- #
def train_and_predict(fs: FeatureSet, store: Storage) -> dict:
    game_model = GameModel().fit(fs.games)
    props_model = PropsModel().fit(fs.players)
    game_model.save()
    props_model.save()

    upcoming = fs.games[fs.games["is_upcoming"] == 1]
    preds = markets.attach_markets(game_model.predict(upcoming), store.read("lines"), store.read("kalshi"))
    preds = simulate_games(preds, game_model.sigma_home, game_model.sigma_away, game_model.rho)
    preds = markets.add_kalshi_value(preds)
    preds["spread_edge"] = preds["pred_margin"] + preds["spread"]      # >0 favours home ATS
    preds["total_edge"] = preds["pred_total"] - preds["total"]         # >0 favours the over
    preds["model_version"] = game_model.trained_through
    preds["predicted_at"] = datetime.now().isoformat(timespec="seconds")

    up_players = fs.players[fs.players["is_upcoming"] == 1]
    props = props_model.predict(up_players)
    props["listed_out"] = up_players["listed_out"].values
    props["base_pra"] = up_players["base_pra"].values
    teams = fs.team_feats[["team_id", "team"]].drop_duplicates("team_id").set_index("team_id")["team"]
    props["team"] = props["team_id"].map(teams)
    props["opp"] = props["opp_id"].map(teams)
    bets = evaluate_prop_bets(props[props["listed_out"] == 0], store.read("prop_lines"), PropCalibrator.load())
    props = _attach_prop_lines(props, bets)
    store.write("prop_bets", bets)

    store.write("pred_games", preds)
    store.write("pred_props", props)
    store.write("pred_history", preds, mode="append")
    store.write("upcoming_game_features", upcoming)
    store.write("upcoming_player_features", up_players)
    store.write("feature_importance", game_model.feature_importance())
    store.set_meta(trained_through=game_model.trained_through, sigma_home=game_model.sigma_home,
                   sigma_away=game_model.sigma_away, rho=game_model.rho,
                   predicted_at=datetime.now().isoformat(timespec="seconds"))
    log.info("predicted %d games and %d player lines", len(preds), len(props))
    return {"games": len(preds), "props": len(props)}


def _attach_prop_lines(props: pd.DataFrame, bets: pd.DataFrame) -> pd.DataFrame:
    """Per stat, add the best-value line on offer (book, line, odds, model vs market
    over-probability, EV) as wide columns for the Props Hub."""
    fields = {"line": "line", "book": "book", "over_odds": "over_odds", "under_odds": "under_odds",
              "model_over_prob": "over_prob", "market_over_prob": "market_over_prob", "best_ev": "ev",
              "best_side": "side"}
    for st in config.PROP_STATS:
        for f in fields.values():
            props[f"{st}_{f}"] = "" if f in ("book", "side") else np.nan
    if bets is None or bets.empty:
        return props
    best = bets.sort_values("best_ev", ascending=False).drop_duplicates(["game_date", "player_id", "stat"])
    for st, grp in best.groupby("stat"):
        wide = grp[["game_date", "player_id", *fields]].rename(columns={k: f"{st}_{v}" for k, v in fields.items()})
        props = props.drop(columns=[c for c in wide.columns if c not in ("game_date", "player_id")]).merge(
            wide, on=["game_date", "player_id"], how="left")
    return props
