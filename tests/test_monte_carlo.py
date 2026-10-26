"""Monte Carlo simulator properties."""
import numpy as np
import pandas as pd
import pytest

from src.models.monte_carlo import simulate_games, simulate_scores


def test_even_matchup_is_coin_flip():
    h, a = simulate_scores([110], [110], 12, 12, 0.3, n_sims=20000, seed=1)
    assert (h > a).mean() == pytest.approx(0.5, abs=0.02)
    assert not (h == a).any()                                    # overtime always breaks ties


def test_better_team_wins_more_and_cover_is_monotonic():
    preds = pd.DataFrame({"pred_home_pts": [115, 115, 115], "pred_away_pts": [105, 105, 105],
                          "spread": [-3.5, -10.0, -15.5], "total": [220.5, 220.5, 220.5]})
    out = simulate_games(preds, 12, 12, 0.3, n_sims=10000)
    assert (out["home_win_prob"] > 0.7).all()
    assert out["home_cover_prob"].is_monotonic_decreasing
    assert out["spread_push_prob"].iloc[1] > 0                     # whole-number line can push
    assert out["spread_push_prob"].iloc[0] == 0                    # half-point line cannot
    assert out["over_prob"].iloc[0] == pytest.approx(0.5, abs=0.03)


def test_simulated_means_match_predictions():
    h, a = simulate_scores([112.3], [104.1], 11, 11, 0.4, n_sims=50000, seed=3)
    assert h.mean() == pytest.approx(112.3 + 0.2, abs=0.5)        # small upward shift from overtime
    assert np.corrcoef(h[0], a[0])[0, 1] == pytest.approx(0.4, abs=0.05)


def test_missing_lines_give_nan_probabilities():
    out = simulate_games(pd.DataFrame({"pred_home_pts": [110], "pred_away_pts": [100],
                                       "spread": [np.nan], "total": [np.nan]}), 12, 12, 0.3, n_sims=1000)
    assert np.isnan(out["home_cover_prob"].iloc[0]) and np.isnan(out["over_prob"].iloc[0])


def test_probabilities_do_not_depend_on_batch():
    one = simulate_games(pd.DataFrame({"pred_home_pts": [113.0], "pred_away_pts": [108.0]}), 11, 11, 0.2)
    many = simulate_games(pd.DataFrame({"pred_home_pts": [100.0, 113.0], "pred_away_pts": [120.0, 108.0]}), 11, 11, 0.2)
    assert one["home_win_prob"].iloc[0] == many["home_win_prob"].iloc[1]
