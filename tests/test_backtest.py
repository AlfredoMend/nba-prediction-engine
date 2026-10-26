"""Walk-forward backtest logic and betting metrics."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.backtest.metrics import _settle, betting_metrics, cumulative_units, prop_metrics, score_metrics
from src.backtest.walk_forward import walk_forward_games


class SpyModel:
    """Records the training data it was fitted on; predicts the training mean."""
    calls: list = []

    def __init__(self):
        self.sigma_home = self.sigma_away = 12.0
        self.rho = 0.3

    def fit(self, games):
        SpyModel.calls.append(games["game_date"].max())
        self.mh, self.ma = games["home_pts"].mean(), games["away_pts"].mean()
        return self

    def predict(self, games):
        out = games[["game_id", "game_date", "home", "away"]].copy()
        out["pred_home_pts"], out["pred_away_pts"] = self.mh, self.ma
        out["pred_margin"] = self.mh - self.ma
        out["pred_spread"] = -out["pred_margin"]
        out["pred_total"] = self.mh + self.ma
        return out


def test_walk_forward_never_trains_on_future(features, league):
    SpyModel.calls = []
    season = features.games["season"].iloc[-1]
    preds = walk_forward_games(features.games, league["lines"], season=season, step_days=7,
                               model_factory=SpyModel, n_sims=500, min_train=50)
    assert (preds["train_cutoff"] < preds["game_date"]).all()
    assert len(SpyModel.calls) >= 3
    assert SpyModel.calls == sorted(SpyModel.calls)              # training window only grows
    done = features.games[(features.games["season"] == season) & features.games["home_pts"].notna()]
    first = preds["game_date"].min()
    assert set(preds["game_id"]) == set(done.loc[done["game_date"] >= first, "game_id"])  # no gaps after warm-up
    assert preds["game_id"].is_unique
    assert preds["n_train"].min() >= 50


def test_settle_wins_losses_pushes_and_roi():
    edge = np.array([3.0, -3.0, 1.0, 5.0, 0.0])
    result = np.array([2.0, 4.0, -1.0, 0.0, 9.0])   # +: home covered, 0: push
    r = _settle(edge, result, threshold=0.0)
    assert (r["bets"], r["wins"], r["losses"], r["pushes"]) == (4, 1, 2, 1)
    assert r["units"] == pytest.approx(1 * config.WIN_PAYOUT - 2)
    assert r["roi"] == pytest.approx(r["units"] / 4)
    r2 = _settle(edge, result, threshold=2.0)
    assert r2["bets"] == 3


def _toy_games():
    return pd.DataFrame({
        "game_id": ["a", "b", "c"], "game_date": pd.to_datetime(["2025-11-01"] * 3),
        "home_pts": [110, 100, 120], "away_pts": [100, 105, 100],
        "pred_home_pts": [112, 104, 110], "pred_away_pts": [101, 102, 104],
        "spread": [-5.5, -2.0, -8.5], "total": [210.5, 205.5, 222.5],
    }).assign(pred_margin=lambda d: d.pred_home_pts - d.pred_away_pts,
              pred_total=lambda d: d.pred_home_pts + d.pred_away_pts)


def test_betting_metrics_ats_by_hand():
    m = betting_metrics(_toy_games(), thresholds=(0.0,))
    ats = m["ats"][0]
    # edges: 11-5.5=5.5 (home, covers 10-5.5>0 W); 2-2=0 (no bet); 6-8.5=-2.5 (away; 20-8.5>0 home covered -> L)
    assert (ats["bets"], ats["wins"], ats["losses"]) == (2, 1, 1)
    ou = m["over_under"][0]
    # totals: pred 213 vs 210.5 over (actual 210 -> L); 206 vs 205.5 over (205 -> L); 214 vs 222.5 under (220 -> W)
    assert (ou["wins"], ou["losses"]) == (1, 2)


def test_cumulative_units_matches_settle():
    g = _toy_games()
    cu = cumulative_units(g, "ats")
    assert cu["cum_units"].iloc[-1] == pytest.approx(betting_metrics(g, (0.0,))["ats"][0]["units"])


def test_score_metrics():
    s = score_metrics(_toy_games())
    assert s["home_pts"]["mae"] == pytest.approx((2 + 4 + 10) / 3)
    assert s["straight_up_accuracy"] == pytest.approx(2 / 3)
    assert "market_baseline" in s


def test_no_lines_reports_unavailable():
    g = _toy_games().assign(spread=np.nan, total=np.nan)
    assert betting_metrics(g)["available"] is False


def test_prop_metrics_coverage():
    df = pd.DataFrame({"pts": [10, 20, 30, 40], "pts_p10": [5, 15, 31, 30], "pts_p50": [10, 18, 35, 38],
                       "pts_p90": [15, 25, 40, 45], "pts_l10": [12, 22, 28, 30]})
    m = prop_metrics(df, stats=["pts"])["pts"]
    assert m["coverage_80"] == pytest.approx(0.75)
    assert m["mae"] == pytest.approx((0 + 2 + 5 + 2) / 4)
    assert m["baseline_l10_mae"] == pytest.approx((2 + 2 + 2 + 10) / 4)
