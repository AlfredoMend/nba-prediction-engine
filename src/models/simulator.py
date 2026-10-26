"""
Lineup & Injury what-if simulator.

Given an upcoming game and a set of players toggled OUT, recompute:
  * each team's missing-minutes / missing-usage features -> re-run the game
    ensemble -> new predicted scores, spread, total and Monte Carlo win/cover probs
  * minutes & usage redistribution among the available players -> re-run the
    props quantile models -> new stat projections

It reuses `lineup.redistribute` and `player_features.add_expectation_features`,
i.e. exactly the transformations the models saw during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.data.lineup import redistribute
from src.data.player_features import add_expectation_features
from src.models.game_model import GameModel
from src.models.monte_carlo import simulate_games
from src.models.props_model import PropsModel


@dataclass
class ScenarioResult:
    game: pd.DataFrame        # one-row frame: predictions + MC probabilities
    players: pd.DataFrame     # player features after redistribution
    props: pd.DataFrame       # prop projections for available players


def apply_lineup(game_row: pd.DataFrame, players: pd.DataFrame, out_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (game_row, players) with lineup-dependent features recomputed for `out_ids`."""
    g = game_row.copy()
    p = players.copy()
    p["listed_out"] = p["player_id"].isin(out_ids).astype(int)
    for side, team_id in (("h", int(g["home_id"].iloc[0])), ("a", int(g["away_id"].iloc[0]))):
        team = p[p["team_id"] == team_id]
        avail, miss = team[team["listed_out"] == 0], team[team["listed_out"] == 1]
        m_min = float(miss["base_min"].sum())
        m_usg = float((miss["base_usg"] * miss["base_min"]).sum() / 100)
        g[f"{side}_missing_min"] = m_min
        g[f"{side}_missing_usg_min"] = m_usg
        g[f"{side}_missing_pra"] = float(miss["base_pra"].sum())
        g[f"{side}_missing_top_min"] = float(miss["base_min"].max()) if len(miss) else 0.0
        proj_min, proj_usg = redistribute(dict(zip(avail["player_id"], avail["base_min"])),
                                          dict(zip(avail["player_id"], avail["base_usg"])), m_min, m_usg)
        idx = team.index
        p.loc[idx, "proj_min"] = p.loc[idx, "player_id"].map(proj_min).fillna(0.0)
        p.loc[idx, "proj_usg"] = p.loc[idx, "player_id"].map(proj_usg).fillna(0.0)
        p.loc[idx, "team_missing_usg_min"] = m_usg
    g["d_missing_usg_min"] = g["h_missing_usg_min"] - g["a_missing_usg_min"]
    return g, add_expectation_features(p)


def run_scenario(game_row: pd.DataFrame, players: pd.DataFrame, out_ids: set[int], game_model: GameModel,
                 props_model: PropsModel, spread: float | None = None, total: float | None = None,
                 n_sims: int = config.N_SIMULATIONS) -> ScenarioResult:
    g, p = apply_lineup(game_row, players, out_ids)
    pred = game_model.predict(g)
    pred["spread"] = spread if spread is not None else float("nan")
    pred["total"] = total if total is not None else float("nan")
    pred = simulate_games(pred, game_model.sigma_home, game_model.sigma_away, game_model.rho, n_sims=n_sims)
    avail = p[p["listed_out"] == 0]
    props = props_model.predict(avail) if len(avail) else pd.DataFrame()
    return ScenarioResult(game=pred, players=p, props=props)
