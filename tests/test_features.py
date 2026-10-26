"""Data transformation tests: box metrics, leakage safety, rest/travel, game matrix."""
import numpy as np
import pandas as pd
import pytest

from src.data.features import add_box_metrics, compute_team_features, possessions, team_rows
from src.data.teams import arena_distance, haversine_miles


def _two_team_rows():
    """Three games between two teams; every stat is easy to compute by hand."""
    rows = []
    for i, d in enumerate(["2025-11-01", "2025-11-02", "2025-11-05"]):
        for team, opp, home in ((1610612747, 1610612738, 1), (1610612738, 1610612747, 0)):
            rows.append(dict(season="2025-26", season_type="Regular Season", game_id=f"G{i}",
                             game_date=pd.Timestamp(d), team_id=team, team=str(team), opp_id=opp, opp=str(opp),
                             is_home=home if i != 1 else 1 - home, min=240,
                             pts=100 + 10 * i if home else 90, fgm=40, fga=90, fg3m=10, fg3a=30, ftm=10, fta=10,
                             oreb=10, dreb=35, reb=45, ast=25, stl=8, blk=5, tov=12, pf=20))
    return pd.DataFrame(rows)


def test_possessions_formula():
    assert possessions(90, 10, 12, 10) == pytest.approx(90 - 10 + 12 + 4.4)


def test_box_metrics_ratings():
    df = add_box_metrics(team_rows(_two_team_rows()))
    r = df.iloc[0]
    assert r["poss"] == pytest.approx(96.4)
    assert r["ortg"] == pytest.approx(100 * r["pts"] / 96.4)
    assert r["net"] == pytest.approx(r["ortg"] - r["drtg"])
    assert r["efg"] == pytest.approx((40 + 5) / 90)


def test_rolling_features_are_shifted_no_leakage():
    tf = compute_team_features(team_rows(_two_team_rows()))
    lal = tf[tf["team_id"] == 1610612747].reset_index(drop=True)
    assert np.isnan(lal.loc[0, "ortg_ewm5"])                       # first game: no history
    assert lal.loc[1, "ortg_ewm5"] == pytest.approx(lal.loc[0, "ortg"])  # only game 0 is visible


def test_future_results_do_not_change_past_features():
    base = _two_team_rows()
    changed = base.copy()
    changed.loc[changed["game_id"] == "G2", "pts"] += 50          # alter the LAST game only
    a = compute_team_features(team_rows(base)).set_index(["game_id", "team_id"])
    b = compute_team_features(team_rows(changed)).set_index(["game_id", "team_id"])
    cols = [c for c in a.columns if "_ewm" in c]
    pd.testing.assert_frame_equal(a[cols], b[cols])               # all pre-game features identical


def test_rest_b2b_and_travel():
    tf = compute_team_features(team_rows(_two_team_rows()))
    lal = tf[tf["team_id"] == 1610612747].reset_index(drop=True)
    assert list(lal["rest_days"]) == [7, 1, 3]
    assert list(lal["b2b"]) == [0, 1, 0]
    # game 1 is played in Boston (LAL away) after game 0 in LA
    assert lal.loc[1, "travel_miles"] == pytest.approx(arena_distance(1610612747, 1610612738))
    assert 2500 < lal.loc[1, "travel_miles"] < 2700


def test_haversine_zero_and_symmetry():
    assert haversine_miles(10, 10, 10, 10) == 0
    assert arena_distance(1610612747, 1610612738) == pytest.approx(arena_distance(1610612738, 1610612747))


def test_game_matrix_pairs_home_and_away(features):
    gm = features.games
    assert gm["game_id"].is_unique
    done = gm[gm["is_upcoming"] == 0]
    assert done["home_pts"].notna().all() and done["away_pts"].notna().all()
    assert (gm["home_id"] != gm["away_id"]).all()
    assert gm["game_date"].is_monotonic_increasing


def test_upcoming_games_get_features_without_targets(features):
    up = features.games[features.games["is_upcoming"] == 1]
    assert len(up) > 0
    assert up["home_pts"].isna().all()
    assert up["h_net_ewm10"].notna().all()


def test_synthetic_box_scores_are_consistent(league):
    tg, pg = league["team_games"], league["player_games"]
    team_sum = pg.groupby(["game_id", "team_id"])[["pts", "reb", "ast", "fgm", "fga"]].sum().reset_index()
    m = team_sum.merge(tg, on=["game_id", "team_id"], suffixes=("_p", "_t"))
    for c in ["pts", "reb", "ast", "fgm", "fga"]:
        assert (m[f"{c}_p"] == m[f"{c}_t"]).all(), c
    assert (pg["fgm"] <= pg["fga"]).all() and (pg["fg3m"] <= pg["fg3a"]).all()


def test_player_features_leakage_safe(features):
    p = features.players
    done = p[p["is_upcoming"] == 0].sort_values(["player_id", "game_date"])
    first = done.groupby("player_id").head(1)
    assert first["pts_ewm10"].isna().all()                        # no history before debut
    # positional defence only uses earlier dates: first league date has no values
    day0 = done["game_date"].min()
    assert done.loc[done["game_date"] == day0, "opp_pos_pts48"].isna().all()
