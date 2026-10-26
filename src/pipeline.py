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
from src.data import odds
from src.data.build import FeatureSet, build_features
from src.data.storage import Storage
from src.models.game_model import GameModel
from src.models.monte_carlo import simulate_games
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
        lines = pd.concat([odds.load_lines_csv(), odds.fetch_live_lines()], ignore_index=True)
        store.write("lines", lines.drop_duplicates(["game_date", "home", "away"], keep="last"))
        store.write("prop_lines", odds.load_prop_lines_csv())
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
        store.write("schedule", nba_fetch.fetch_schedule(as_of, days=3))
        lines = pd.concat([odds.load_lines_csv(), odds.fetch_live_lines()], ignore_index=True)
        store.write("lines", lines.drop_duplicates(["game_date", "home", "away"], keep="last"))
        store.write("prop_lines", odds.load_prop_lines_csv())
        store.write("injuries", _load_injuries(pg))
    else:
        raise ValueError(f"unknown data source {source!r}")
    store.set_meta(data_source=source, as_of=as_of.isoformat(),
                   ingested_at=datetime.now().isoformat(timespec="seconds"))


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
    lines = store.read("lines")
    preds = game_model.predict(upcoming)
    if not lines.empty:
        lines["game_date"] = pd.to_datetime(lines["game_date"])
        preds = preds.merge(lines[["game_date", "home", "away", "spread", "total"]],
                            on=["game_date", "home", "away"], how="left")
    else:
        preds["spread"], preds["total"] = np.nan, np.nan
    preds = simulate_games(preds, game_model.sigma_home, game_model.sigma_away, game_model.rho)
    preds["spread_edge"] = preds["pred_margin"] + preds["spread"]      # >0 favours home ATS
    preds["total_edge"] = preds["pred_total"] - preds["total"]         # >0 favours the over
    preds["model_version"] = game_model.trained_through
    preds["predicted_at"] = datetime.now().isoformat(timespec="seconds")

    up_players = fs.players[fs.players["is_upcoming"] == 1]
    props = props_model.predict(up_players)
    props["listed_out"] = up_players["listed_out"].values
    props["base_pra"] = up_players["base_pra"].values
    props = _attach_prop_lines(props, store.read("prop_lines"))
    teams = fs.team_feats[["team_id", "team"]].drop_duplicates("team_id").set_index("team_id")["team"]
    props["team"] = props["team_id"].map(teams)
    props["opp"] = props["opp_id"].map(teams)

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


def _attach_prop_lines(props: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Merge any available prop lines and compute model over-probabilities."""
    for s in config.PROP_STATS:
        props[f"{s}_line"] = np.nan
    if lines is None or lines.empty:
        return props
    lines = lines.assign(game_date=pd.to_datetime(lines["game_date"]))
    key = "player_id" if "player_id" in lines and lines["player_id"].notna().all() else "player_name"
    wide = lines.pivot_table(index=["game_date", key], columns="stat", values="line", aggfunc="last")
    wide.columns = [f"{c}_line" for c in wide.columns]
    props = props.drop(columns=[c for c in wide.columns if c in props]).merge(
        wide.reset_index(), on=["game_date", key], how="left")
    for s in config.PROP_STATS:
        if f"{s}_line" not in props:
            props[f"{s}_line"] = np.nan
        props[f"{s}_over_prob"] = PropsModel.over_probability(props[f"{s}_p10"], props[f"{s}_p50"],
                                                              props[f"{s}_p90"], props[f"{s}_line"])
    return props
