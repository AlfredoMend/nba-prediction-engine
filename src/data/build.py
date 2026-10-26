"""
Feature-store builder: raw tables -> (team features, game matrix, player matrix).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.features import build_game_matrix, compute_team_features, team_rows
from src.data.lineup import compute_lineup_features, player_usage
from src.data.player_features import build_player_matrix


@dataclass
class FeatureSet:
    team_feats: pd.DataFrame
    games: pd.DataFrame     # one row per game (home/away features + targets)
    players: pd.DataFrame   # one row per player-game (features + targets)


def build_features(team_games: pd.DataFrame, player_games: pd.DataFrame, players: pd.DataFrame,
                   schedule: pd.DataFrame | None = None, injuries: pd.DataFrame | None = None) -> FeatureSet:
    team_games = team_games.assign(game_date=pd.to_datetime(team_games["game_date"]))
    pg = player_games.assign(game_date=pd.to_datetime(player_games["game_date"]))
    pg["usg"] = player_usage(pg, team_games)

    rows = team_rows(team_games, schedule)
    tf = compute_team_features(rows)
    team_lineup, player_lineup = compute_lineup_features(rows, pg, injuries)
    tf = tf.merge(team_lineup, on=["game_id", "team_id"], how="left")
    games = build_game_matrix(tf)
    pm = build_player_matrix(pg, player_lineup, players, tf)
    return FeatureSet(team_feats=tf, games=games, players=pm)
