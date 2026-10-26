"""
Rotation tracking, injury adjustment and usage redistribution.

For every team-game (completed or upcoming) we reconstruct the team's *expected
rotation* from players who recently played for it, compare it with who actually
played (history) or who is not ruled OUT (upcoming), and derive:

Team level  (fed to the game model -> team ratings are scaled automatically)
  missing_min       expected minutes of rotation players who are out
  missing_usg_min   usage-weighted minutes that are out (usage% * minutes / 100)
  missing_pra       expected PRA production that is out
  missing_top_min   minutes of the most important missing player

Player level (fed to the props model)
  base_min / base_usg   pre-game EWMA minutes / usage
  proj_min / proj_usg   after redistributing the missing minutes & usage

The same `redistribute()` function powers the UI's Lineup & Injury Simulator,
so what-if scenarios use exactly the logic the models were trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

ALPHA = 2 / (10 + 1)        # EWMA(span=10) smoothing for the rotation state
DEFAULT_NEW_MIN = 12.0      # prior minutes for a player with no history
DEFAULT_USG = 18.0


@dataclass
class PlayerState:
    team_id: int
    last_date: pd.Timestamp
    min_ewm: float
    usg_ewm: float
    pra_ewm: float


def player_usage(pg: pd.DataFrame, team_games: pd.DataFrame) -> pd.Series:
    """Usage rate (USG%) per player-game using team totals:
    100 * (FGA + 0.44 FTA + TOV) * (TeamMin / 5) / (Min * (TmFGA + 0.44 TmFTA + TmTOV))."""
    tm = team_games[["game_id", "team_id", "min", "fga", "fta", "tov"]].rename(
        columns={"min": "t_min", "fga": "t_fga", "fta": "t_fta", "tov": "t_tov"})
    m = pg[["game_id", "team_id", "min", "fga", "fta", "tov"]].merge(tm, on=["game_id", "team_id"], how="left")
    num = (m["fga"] + 0.44 * m["fta"] + m["tov"]) * (m["t_min"] / 5)
    den = m["min"] * (m["t_fga"] + 0.44 * m["t_fta"] + m["t_tov"])
    usg = 100 * num / den.replace(0, np.nan)
    return pd.Series(usg.clip(0, 60).fillna(0).values, index=pg.index)


def redistribute(base_min: dict, base_usg: dict, out_min: float, out_usg_min: float,
                 cap: float = config.MAX_PLAYER_MINUTES) -> tuple[dict, dict]:
    """Hand minutes and usage of absent players to the available ones.

    Minutes are shared in proportion to each available player's current minutes;
    usage-minutes in proportion to usage x minutes. Minutes are capped at `cap`
    and any overflow is passed on to uncapped players, so the total is conserved
    whenever the cap allows it.
    """
    ids = list(base_min)
    if not ids:
        return {}, {}
    b = np.array([base_min[i] for i in ids], dtype=float)
    u = np.array([base_usg[i] for i in ids], dtype=float)
    proj = b.copy()
    remaining = out_min
    for _ in range(5):  # iterate to push overflow from capped players to the rest
        free = proj < cap
        if remaining <= 1e-9 or not free.any():
            break
        w = b * free
        add = remaining * w / w.sum() if w.sum() > 0 else np.where(free, remaining / free.sum(), 0)
        new = np.minimum(proj + add, cap)
        remaining -= (new - proj).sum()
        proj = new
    um = u * b + out_usg_min * 100 * (u * b) / max((u * b).sum(), 1e-9)
    proj_usg = np.clip(um / np.maximum(proj, 1e-9), 0, 60)
    return dict(zip(ids, proj)), dict(zip(ids, proj_usg))


def compute_lineup_features(team_rows: pd.DataFrame, player_games: pd.DataFrame,
                            injuries: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk through all team-games chronologically building rotation state.

    team_rows    output of features.team_rows (completed + upcoming)
    player_games completed player box scores incl. a `usg` column
    injuries     upcoming injury report (player_id, status) - OUT/DOUBTFUL players are excluded

    Returns (team_lineup, player_lineup). `player_lineup` contains one row per
    player per game, including *expected* participants of upcoming games.
    """
    out_ids = set()
    if injuries is not None and not injuries.empty:
        out_ids = set(injuries.loc[injuries["status"].str.upper().isin(["OUT", "DOUBTFUL"]), "player_id"].astype(int))

    played = {k: v for k, v in player_games.groupby(["game_id", "team_id"])}
    games = team_rows[["game_id", "team_id", "opp_id", "is_home", "game_date", "season", "is_upcoming"]]
    games = games.sort_values(["game_date", "game_id"]).itertuples(index=False)

    state: dict[int, PlayerState] = {}
    team_out, player_out = [], []
    for gm in games:
        lookback = gm.game_date - pd.Timedelta(days=config.ROTATION_LOOKBACK_DAYS)
        rotation = {pid for pid, st in state.items()
                    if st.team_id == gm.team_id and st.last_date >= lookback and st.min_ewm >= config.ROTATION_MIN_MINUTES}
        box = played.get((gm.game_id, gm.team_id))
        if gm.is_upcoming:
            participants = sorted(rotation - out_ids)
        else:
            participants = [] if box is None else box["player_id"].astype(int).tolist()
        missing = rotation - set(participants)

        base_min = {p: state[p].min_ewm if p in state else DEFAULT_NEW_MIN for p in participants}
        base_usg = {p: state[p].usg_ewm if p in state else DEFAULT_USG for p in participants}
        m_min = sum(state[m].min_ewm for m in missing)
        m_usg = sum(state[m].usg_ewm * state[m].min_ewm for m in missing) / 100
        proj_min, proj_usg = redistribute(base_min, base_usg, m_min, m_usg)

        team_out.append((gm.game_id, gm.team_id, m_min, m_usg,
                         sum(state[m].pra_ewm for m in missing),
                         max((state[m].min_ewm for m in missing), default=0.0)))
        for p in participants:
            player_out.append((gm.game_id, p, gm.team_id, gm.opp_id, gm.is_home, gm.game_date, gm.season,
                               gm.is_upcoming, base_min[p], base_usg[p], proj_min[p], proj_usg[p], m_usg,
                               state[p].pra_ewm if p in state else 0.0, 0))
        if gm.is_upcoming:
            # keep ruled-out rotation players too (listed_out=1, projected 0 min) so the
            # Lineup & Injury Simulator can toggle them back IN
            for p in sorted(missing):
                st = state[p]
                player_out.append((gm.game_id, p, gm.team_id, gm.opp_id, gm.is_home, gm.game_date, gm.season,
                                   1, st.min_ewm, st.usg_ewm, 0.0, 0.0, m_usg, st.pra_ewm, 1))

        # update rotation state with the realised box score (completed games only)
        if box is not None and not gm.is_upcoming:
            for r in box[["player_id", "min", "usg", "pts", "reb", "ast"]].itertuples(index=False):
                pid, pra = int(r.player_id), r.pts + r.reb + r.ast
                st = state.get(pid)
                if st is None or (gm.game_date - st.last_date).days > 120:  # new player / new season
                    state[pid] = PlayerState(gm.team_id, gm.game_date, r.min, r.usg, pra)
                else:
                    st.team_id, st.last_date = gm.team_id, gm.game_date
                    st.min_ewm += ALPHA * (r.min - st.min_ewm)
                    st.usg_ewm += ALPHA * (r.usg - st.usg_ewm)
                    st.pra_ewm += ALPHA * (pra - st.pra_ewm)

    team_lineup = pd.DataFrame(team_out, columns=["game_id", "team_id", "missing_min", "missing_usg_min",
                                                  "missing_pra", "missing_top_min"])
    player_lineup = pd.DataFrame(player_out, columns=["game_id", "player_id", "team_id", "opp_id", "is_home",
                                                      "game_date", "season", "is_upcoming", "base_min", "base_usg",
                                                      "proj_min", "proj_usg", "team_missing_usg_min",
                                                      "base_pra", "listed_out"])
    return team_lineup, player_lineup
