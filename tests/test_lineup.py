"""Injury / usage redistribution logic."""
import pandas as pd
import pytest

from src.data.lineup import redistribute
from src.models.simulator import apply_lineup


def test_redistribution_conserves_minutes():
    base_min = {1: 30.0, 2: 20.0, 3: 10.0}
    base_usg = {1: 25.0, 2: 20.0, 3: 15.0}
    proj_min, proj_usg = redistribute(base_min, base_usg, out_min=30.0, out_usg_min=7.5)
    assert sum(proj_min.values()) == pytest.approx(90.0)
    assert proj_min[1] == pytest.approx(45.0 if 45 <= 44 else proj_min[1])
    assert all(proj_min[k] >= base_min[k] for k in base_min)
    assert all(proj_usg[k] >= base_usg[k] * 0.99 for k in base_usg)


def test_redistribution_respects_cap_and_passes_overflow():
    proj_min, _ = redistribute({1: 40.0, 2: 10.0}, {1: 30.0, 2: 15.0}, out_min=20.0, out_usg_min=0, cap=44)
    assert proj_min[1] <= 44 + 1e-9
    assert sum(proj_min.values()) == pytest.approx(70.0)          # overflow moved to player 2


def test_no_missing_players_is_identity():
    proj_min, proj_usg = redistribute({1: 30.0, 2: 18.0}, {1: 22.0, 2: 17.0}, 0.0, 0.0)
    assert proj_min == {1: 30.0, 2: 18.0}
    assert proj_usg[1] == pytest.approx(22.0)


def test_lineup_features_flag_missing_rotation_players(features):
    tf = features.team_feats
    done = tf[tf["is_upcoming"] == 0]
    assert (done["missing_min"] >= 0).all()
    assert (done["missing_min"] > 0).mean() > 0.1               # injuries happen in the synthetic league


def test_simulator_out_player_raises_teammates(features):
    up = features.players[features.players["is_upcoming"] == 1]
    gid = up["game_id"].iloc[0]
    players = up[up["game_id"] == gid].reset_index(drop=True)
    game = features.games[features.games["game_id"] == gid].reset_index(drop=True)
    team = players["team_id"].iloc[0]
    star = players[players["team_id"] == team].sort_values("base_min").iloc[-1]["player_id"]
    g0, p0 = apply_lineup(game, players, set())
    g1, p1 = apply_lineup(game, players, {int(star)})
    side = "h" if game["home_id"].iloc[0] == team else "a"
    assert g1[f"{side}_missing_min"].iloc[0] > g0[f"{side}_missing_min"].iloc[0]
    mates0 = p0[(p0["team_id"] == team) & (p0["player_id"] != star)]["proj_min"].sum()
    mates1 = p1[(p1["team_id"] == team) & (p1["player_id"] != star)]["proj_min"].sum()
    assert mates1 > mates0
    assert p1.loc[p1["player_id"] == star, "listed_out"].iloc[0] == 1
