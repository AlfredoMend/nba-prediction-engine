"""
Team-level feature engineering.

Leakage rule: every feature attached to a game row is computed ONLY from games
that finished before that row's tip-off. Rolling statistics are therefore
always `.shift(1)`-ed inside each team's chronological history. Upcoming games
are appended as placeholder rows with NaN box scores, so they receive features
through exactly the same code path as historical games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.teams import arena_distance

BOX = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]
RATING_COLS = ["ortg", "drtg", "net", "pace", "efg", "tov_pct", "oreb_pct", "ft_rate", "pts", "opp_pts"]


# --------------------------------------------------------------------------- #
# Basic box-score derived metrics
# --------------------------------------------------------------------------- #
def possessions(fga, oreb, tov, fta):
    """Basic possession estimate: FGA - OREB + TOV + 0.44 * FTA."""
    return fga - oreb + tov + 0.44 * fta


def add_box_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-game efficiency metrics. Expects team columns plus `opp_*` box columns."""
    df = df.copy()
    own = possessions(df["fga"], df["oreb"], df["tov"], df["fta"])
    opp = possessions(df["opp_fga"], df["opp_oreb"], df["opp_tov"], df["opp_fta"])
    df["poss"] = (own + opp) / 2                                   # average of both sides
    minutes = df["min"].where(df["min"] > 0, config.TEAM_MINUTES)
    df["pace"] = df["poss"] * config.TEAM_MINUTES / minutes        # possessions per 48
    df["ortg"] = 100 * df["pts"] / df["poss"]
    df["drtg"] = 100 * df["opp_pts"] / df["poss"]
    df["net"] = df["ortg"] - df["drtg"]
    df["efg"] = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"]
    df["tov_pct"] = df["tov"] / (df["fga"] + 0.44 * df["fta"] + df["tov"])
    df["oreb_pct"] = df["oreb"] / (df["oreb"] + df["opp_dreb"])
    df["ft_rate"] = df["fta"] / df["fga"]
    return df


# --------------------------------------------------------------------------- #
# Assemble historical + upcoming team rows
# --------------------------------------------------------------------------- #
def team_rows(team_games: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team per game (completed and upcoming) with opponent box joined."""
    tg = team_games.copy()
    tg["is_upcoming"] = 0
    if schedule is not None and not schedule.empty:
        up = []
        for side, opp_side, home in (("home", "away", 1), ("away", "home", 0)):
            up.append(pd.DataFrame({
                "season": schedule["season"], "season_type": "Regular Season",
                "game_id": schedule["game_id"], "game_date": pd.to_datetime(schedule["game_date"]),
                "team_id": schedule[f"{side}_id"], "team": schedule[side],
                "opp_id": schedule[f"{opp_side}_id"], "opp": schedule[opp_side],
                "is_home": home, "min": np.nan, "is_upcoming": 1,
            }))
        tg = pd.concat([tg, *up], ignore_index=True)
        # never let an upcoming placeholder duplicate a completed game
        tg = tg.drop_duplicates(["game_id", "team_id"], keep="first")
    opp = tg[["game_id", "team_id", *BOX]].rename(columns={c: f"opp_{c}" for c in BOX})
    opp = opp.rename(columns={"team_id": "opp_id"})
    tg = tg.merge(opp, on=["game_id", "opp_id"], how="left")
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg["opp_id"] = tg["opp_id"].astype("int64")
    return tg.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Rolling / contextual team features
# --------------------------------------------------------------------------- #
def _games_in_last_days(dates: np.ndarray, days: int) -> np.ndarray:
    """For each date, number of earlier games within the previous `days` days."""
    d = dates.astype("datetime64[D]").astype(np.int64)
    left = np.searchsorted(d, d - days, side="left")
    return np.arange(len(d)) - left


def compute_team_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe rolling features for every team-game row (see module docstring)."""
    df = add_box_metrics(rows).sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    g = df.groupby("team_id", sort=False)

    # Exponentially weighted moving averages of team ratings, shifted by one game
    for span in config.EWMA_SPANS:
        for col in RATING_COLS:
            df[f"{col}_ewm{span}"] = g[col].transform(lambda s, sp=span: s.ewm(span=sp, min_periods=1).mean().shift(1))

    # Home/away split: net rating EWMA over previous games at the same venue type
    df["venue_net_ewm10"] = df.groupby(["team_id", "is_home"], sort=False)["net"].transform(
        lambda s: s.ewm(span=10, min_periods=1).mean().shift(1))

    # Rest, back-to-backs, schedule density
    prev_date = g["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - prev_date).dt.days.clip(upper=config.MAX_REST_DAYS).fillna(config.MAX_REST_DAYS)
    df["b2b"] = (df["rest_days"] == 1).astype(int)
    df["games_last_7d"] = g["game_date"].transform(lambda s: pd.Series(_games_in_last_days(s.values, 7), index=s.index))

    # Travel: distance from the previous game's arena to this game's arena
    df["venue_id"] = np.where(df["is_home"] == 1, df["team_id"], df["opp_id"])
    prev_venue = g["venue_id"].shift(1)
    df["travel_miles"] = [arena_distance(int(p), int(v)) if pd.notna(p) else 0.0
                          for p, v in zip(prev_venue, df["venue_id"])]
    df.loc[df["rest_days"] >= config.MAX_REST_DAYS, "travel_miles"] = 0.0  # offseason reset

    df["season_game_no"] = df.groupby(["team_id", "season"]).cumcount() + 1
    return df


# --------------------------------------------------------------------------- #
# Game-level matrix (one row per game, home vs away)
# --------------------------------------------------------------------------- #
TEAM_FEATURES = (
    [f"{c}_ewm{s}" for s in config.EWMA_SPANS for c in RATING_COLS]
    + ["venue_net_ewm10", "rest_days", "b2b", "games_last_7d", "travel_miles", "season_game_no",
       "missing_min", "missing_usg_min", "missing_pra", "missing_top_min"]
)


def build_game_matrix(team_feats: pd.DataFrame) -> pd.DataFrame:
    """Pivot team-game features into home/away columns plus key differentials."""
    keep = ["game_id", "game_date", "season", "team_id", "team", "is_upcoming", "pts", *TEAM_FEATURES]
    keep = [c for c in keep if c in team_feats.columns]
    home = team_feats.loc[team_feats["is_home"] == 1, keep].add_prefix("h_")
    away = team_feats.loc[team_feats["is_home"] == 0, keep].add_prefix("a_")
    gm = home.merge(away, left_on="h_game_id", right_on="a_game_id")
    gm = gm.rename(columns={"h_game_id": "game_id", "h_game_date": "game_date", "h_season": "season",
                            "h_is_upcoming": "is_upcoming", "h_team": "home", "a_team": "away",
                            "h_team_id": "home_id", "a_team_id": "away_id",
                            "h_pts": "home_pts", "a_pts": "away_pts"})
    gm = gm.drop(columns=["a_game_id", "a_game_date", "a_season", "a_is_upcoming"])
    for s in config.EWMA_SPANS:
        gm[f"d_net_ewm{s}"] = gm[f"h_net_ewm{s}"] - gm[f"a_net_ewm{s}"]
        # expected matchup efficiency: own offence vs opponent defence
        gm[f"h_matchup_ortg{s}"] = (gm[f"h_ortg_ewm{s}"] + gm[f"a_drtg_ewm{s}"]) / 2
        gm[f"a_matchup_ortg{s}"] = (gm[f"a_ortg_ewm{s}"] + gm[f"h_drtg_ewm{s}"]) / 2
        gm[f"exp_pace{s}"] = (gm[f"h_pace_ewm{s}"] + gm[f"a_pace_ewm{s}"]) / 2
    if "h_missing_usg_min" in gm:
        gm["d_missing_usg_min"] = gm["h_missing_usg_min"] - gm["a_missing_usg_min"]
    gm["d_rest"] = gm["h_rest_days"] - gm["a_rest_days"]
    return gm.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def game_feature_columns(gm: pd.DataFrame) -> list[str]:
    """All model input columns of the game matrix (excludes ids/targets)."""
    exclude = {"game_id", "game_date", "season", "is_upcoming", "home", "away", "home_id", "away_id",
               "home_pts", "away_pts"}
    return [c for c in gm.columns if c not in exclude and pd.api.types.is_numeric_dtype(gm[c])]
