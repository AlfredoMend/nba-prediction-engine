"""Prop bet evaluation: model probability vs book price, EV, Kelly, calibration, name matching."""
import numpy as np
import pandas as pd
import pytest

from src.data.prop_odds import attach_player_ids, normalize_name
from src.models.prop_edges import PropCalibrator, american_profit, evaluate_prop_bets
from src.models.props_model import PropsModel


def _props():
    # Tatum: median 26, 80% range 17-35
    return pd.DataFrame({"game_date": pd.to_datetime(["2026-10-27"]), "player_id": [1], "player_name": ["Jayson Tatum"],
                         "team": ["BOS"], "opp": ["BKN"], "proj_min": [36.0],
                         "pts_p10": [17.0], "pts_p50": [26.0], "pts_p90": [35.0], "pts_l10": [24.0]})


def _line(line, over, under, book="novig"):
    return pd.DataFrame({"game_date": pd.to_datetime(["2026-10-27"]), "player_id": [1], "player_name": ["Jayson Tatum"],
                         "stat": ["pts"], "line": [line], "over_odds": [over], "under_odds": [under], "book": [book]})


def test_american_profit():
    assert american_profit(-110) == pytest.approx(100 / 110)
    assert american_profit(120) == pytest.approx(1.2)


def test_tatum_example_over_has_positive_ev():
    bets = evaluate_prop_bets(_props(), _line(23.5, -105, -105))
    r = bets.iloc[0]
    p = PropsModel.over_probability(17, 26, 35, 23.5)
    assert r["model_over_prob"] == pytest.approx(p)
    assert r["market_over_prob"] == pytest.approx(0.5)
    assert r["best_side"] == "Over" and r["edge_pts"] == pytest.approx(2.5)
    assert r["best_ev"] == pytest.approx(p * (100 / 105) - (1 - p))
    assert r["value"] and r["kelly_stake"] > 0
    assert r["player_name"] == "Jayson Tatum" and r["team"] == "BOS"


def test_line_above_projection_favours_under():
    r = evaluate_prop_bets(_props(), _line(29.5, -110, -110)).iloc[0]
    assert r["best_side"] == "Under" and r["best_ev"] > 0


def test_fair_line_at_median_is_not_a_bet_after_vig():
    r = evaluate_prop_bets(_props(), _line(26.0, -110, -110)).iloc[0]
    assert r["best_ev"] < 0 and not r["value"] and r["kelly_stake"] == 0


def test_missing_odds_assume_minus_110_and_no_lines_is_empty():
    r = evaluate_prop_bets(_props(), _line(23.5, np.nan, np.nan)).iloc[0]
    assert r["over_odds"] == -110 and r["under_odds"] == -110
    assert evaluate_prop_bets(_props(), pd.DataFrame()).empty


def test_calibrator_corrects_optimistic_probabilities():
    rng = np.random.default_rng(0)
    n = 4000
    p50 = rng.uniform(5, 30, n)
    actual = np.round(p50 + rng.normal(-1.0, 6, n))          # truth sits 1 point below the model's median
    bt = pd.DataFrame({"game_date": pd.date_range("2025-10-21", periods=n, freq="h"), "proj_min": 30.0,
                       "pts": actual, "pts_p10": p50 - 7.7, "pts_p50": p50, "pts_p90": p50 + 7.7})
    cal = PropCalibrator().fit(bt.assign(**{f"{s}{q}": bt[f"pts{q}"] for s in ("reb", "ast", "fg3m", "stl", "blk", "pra")
                                             for q in ("", "_p10", "_p50", "_p90")}))
    rep = cal.report["pts"]
    assert rep["mean_raw"] > rep["observed_rate"] + 0.03                 # raw model too optimistic
    assert abs(rep["mean_calibrated"] - rep["observed_rate"]) < 0.02     # calibrated close to reality
    assert rep["brier_calibrated"] < rep["brier_raw"]


def test_name_matching_handles_accents_and_suffixes():
    assert normalize_name("Nikola Jokić") == "nikola jokic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    players = pd.DataFrame({"player_id": [7, 8], "player_name": ["Nikola Jokic", "Jaren Jackson Jr."]})
    lines = pd.DataFrame({"player_name": ["Nikola Jokić", "JAREN JACKSON JR", "Nobody Here"], "player_id": [pd.NA] * 3})
    out = attach_player_ids(lines, players)
    assert list(out["player_id"]) == [7, 8]
