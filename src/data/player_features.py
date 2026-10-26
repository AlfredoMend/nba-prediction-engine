"""
Player-level feature engineering for the props model.

Each row = one player in one game (completed or upcoming expected participant).
All rolling features are shifted one game inside the player's history, and
opponent positional-defense features are joined with `merge_asof(...,
allow_exact_matches=False)` so only strictly earlier games are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

ROLL_STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "pra", "usg", "ts", "fga", "fta", "tov"]
DEF_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "pra"]


def add_player_box_metrics(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.copy()
    pg["pra"] = pg["pts"] + pg["reb"] + pg["ast"]
    tsa = 2 * (pg["fga"] + 0.44 * pg["fta"])
    pg["ts"] = (pg["pts"] / tsa).where(tsa > 0)              # True Shooting %
    return pg


def grouped_ewm(df: pd.DataFrame, by: str, cols: list[str], span: int, shift: bool = True) -> pd.DataFrame:
    """Vectorised per-group EWMA (optionally shifted one row so it is pre-game)."""
    ew = df.groupby(by, sort=False)[cols].ewm(span=span, min_periods=1).mean().reset_index(level=0, drop=True)
    ew = ew.sort_index()
    return ew.groupby(df[by]).shift(1) if shift else ew


def positional_defense(pg: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Per-48 production allowed by each defence to each position bucket (post-game EWMA)."""
    d = pg.assign(position=positions.values)
    agg = d.groupby(["game_id", "game_date", "opp_id", "position"], as_index=False)[DEF_STATS + ["min"]].sum()
    for s in DEF_STATS:
        agg[f"opp_pos_{s}48"] = 48 * agg[s] / agg["min"].replace(0, np.nan)
    agg = agg.sort_values("game_date").reset_index(drop=True)
    cols = [f"opp_pos_{s}48" for s in DEF_STATS]
    agg["key"] = agg["opp_id"].astype(str) + "_" + agg["position"]
    agg[cols] = grouped_ewm(agg, "key", cols, span=15, shift=False)
    return agg[["game_date", "opp_id", "position", *cols]]


def add_expectation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lineup-adjusted expected stat lines: per-minute rate x projected minutes,
    with scoring stats scaled by the usage change. Re-run by the injury simulator
    after `proj_min` / `proj_usg` are recomputed."""
    df = df.copy()
    usage_boost = (df["proj_usg"] / df["base_usg"].replace(0, np.nan)).clip(0.7, 1.6).fillna(1.0)
    for s in config.PROP_STATS:
        boost = usage_boost if s in ("pts", "fg3m", "pra") else 1.0
        df[f"{s}_exp"] = df[f"{s}_per_min"] * df["proj_min"] * boost
    df["min_delta"] = df["proj_min"] - df["base_min"]
    return df


def build_player_matrix(player_games: pd.DataFrame, player_lineup: pd.DataFrame, players: pd.DataFrame,
                        team_feats: pd.DataFrame) -> pd.DataFrame:
    """Combine lineup projections, rolling form, matchup and context features."""
    pg = add_player_box_metrics(player_games)
    stats_cols = ["game_id", "player_id", "player_name", *ROLL_STATS, "oreb", "dreb", "fgm", "fg3a", "ftm"]
    stats_cols = list(dict.fromkeys(c for c in stats_cols if c in pg.columns))
    df = player_lineup.merge(pg[stats_cols], on=["game_id", "player_id"], how="left")

    # names/positions (players table wins; fall back to names seen in box scores)
    pos = players.drop_duplicates("player_id").set_index("player_id")
    df["position"] = df["player_id"].map(pos["position"]).fillna("F")
    df["player_name"] = df["player_id"].map(pos["player_name"]).fillna(df["player_name"])
    last_name = pg.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
    df["player_name"] = df["player_name"].fillna(df["player_id"].map(last_name))

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    # Rolling form: EWMA 5/10/20 of box stats, usage and TS%
    for span in config.EWMA_SPANS:
        ew = grouped_ewm(df, "player_id", ROLL_STATS, span)
        df[[f"{c}_ewm{span}" for c in ROLL_STATS]] = ew.values
    # Baseline: plain last-10-game average (what a naive bettor would use)
    l10 = df.groupby("player_id", sort=False)[config.PROP_STATS].transform(
        lambda s: s.rolling(10, min_periods=1).mean().shift(1))
    df[[f"{s}_l10" for s in config.PROP_STATS]] = l10.values

    # Per-minute rates and lineup-adjusted expectations
    for s in config.PROP_STATS:
        df[f"{s}_per_min"] = df[f"{s}_ewm10"] / df["min_ewm10"].replace(0, np.nan)
    df = add_expectation_features(df)

    # Experience / schedule
    g = df.groupby("player_id", sort=False)
    df["games_played"] = g.cumcount()
    df["rest_days"] = (df["game_date"] - g["game_date"].shift(1)).dt.days.clip(upper=config.MAX_REST_DAYS).fillna(config.MAX_REST_DAYS)
    df["pos_code"] = df["position"].map({"G": 0, "F": 1, "C": 2}).fillna(1)

    # Opponent positional defence (strictly earlier games)
    completed = df[df["is_upcoming"] == 0]
    pdef = positional_defense(completed[["game_id", "game_date", "opp_id", *DEF_STATS, "min"]].dropna(subset=["min"]),
                              completed.loc[completed["min"].notna(), "position"])
    df = df.sort_values("game_date")
    df = pd.merge_asof(df, pdef, on="game_date", by=["opp_id", "position"], allow_exact_matches=False)

    # Team / opponent context from team features (already pre-game)
    ctx = team_feats[["game_id", "team_id", "pace_ewm10", "ortg_ewm10", "drtg_ewm10", "b2b", "travel_miles"]]
    df = df.merge(ctx.rename(columns={"pace_ewm10": "team_pace10", "ortg_ewm10": "team_ortg10",
                                      "drtg_ewm10": "team_drtg10", "b2b": "team_b2b", "travel_miles": "team_travel"}),
                  on=["game_id", "team_id"], how="left")
    opp = ctx[["game_id", "team_id", "pace_ewm10", "drtg_ewm10"]].rename(
        columns={"team_id": "opp_id", "pace_ewm10": "opp_pace10", "drtg_ewm10": "opp_drtg10"})
    df = df.merge(opp, on=["game_id", "opp_id"], how="left")
    df["exp_pace"] = (df["team_pace10"] + df["opp_pace10"]) / 2
    return df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)


PROP_FEATURES = (
    [f"{c}_ewm{s}" for s in config.EWMA_SPANS for c in ROLL_STATS]
    + [f"{s}_per_min" for s in config.PROP_STATS] + [f"{s}_exp" for s in config.PROP_STATS]
    + [f"opp_pos_{s}48" for s in DEF_STATS]
    + ["base_min", "base_usg", "proj_min", "proj_usg", "min_delta", "team_missing_usg_min", "is_home",
       "games_played", "rest_days", "pos_code", "team_pace10", "team_ortg10", "team_drtg10", "team_b2b",
       "team_travel", "opp_pace10", "opp_drtg10", "exp_pace"]
)
