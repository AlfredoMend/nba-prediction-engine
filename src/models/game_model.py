"""
Game outcome engine: LightGBM + XGBoost + CatBoost ensemble predicting home and
away points independently. Spread and total are derived from the two scores.

During `fit`, the most recent 15% of the training window (chronologically) is
held out to estimate the residual standard deviations and the home/away
residual correlation. Those drive the Monte Carlo simulator. The ensemble is
then refit on the full window so no data is wasted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

from src import config
from src.data.features import game_feature_columns


def _make_models(seed: int, fast: bool = False) -> dict:
    n = 250 if fast else 450
    return {
        "lgbm": lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                  random_state=seed, verbose=-1, n_jobs=4),
        "xgb": xgb.XGBRegressor(n_estimators=n, learning_rate=0.03, max_depth=4, subsample=0.8,
                                colsample_bytree=0.7, min_child_weight=10, reg_lambda=1.0,
                                random_state=seed, n_jobs=4, tree_method="hist"),
        "cat": CatBoostRegressor(iterations=n, learning_rate=0.05, depth=5, l2_leaf_reg=5,
                                 random_seed=seed, verbose=0, thread_count=4, allow_writing_files=False),
    }


class ScoreEnsemble:
    """Equal-weight average of three gradient-boosted regressors for one target."""

    def __init__(self, seed: int = config.RANDOM_SEED, fast: bool = False):
        self.models = _make_models(seed, fast)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScoreEnsemble":
        for m in self.models.values():
            m.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models.values()], axis=0)

    def predict_members(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {k: m.predict(X) for k, m in self.models.items()}


@dataclass
class GameModel:
    fast: bool = False
    features: list[str] = field(default_factory=list)
    home: ScoreEnsemble | None = None
    away: ScoreEnsemble | None = None
    sigma_home: float = 11.5
    sigma_away: float = 11.5
    rho: float = 0.4
    trained_through: str = ""

    # ------------------------------------------------------------------ #
    def fit(self, games: pd.DataFrame, calibrate: bool = True) -> "GameModel":
        train = games.dropna(subset=["home_pts", "away_pts"]).sort_values("game_date")
        self.features = game_feature_columns(train)
        if calibrate and len(train) > 500:
            cut = int(len(train) * 0.85)
            fit_part, hold = train.iloc[:cut], train.iloc[cut:]
            h = ScoreEnsemble(fast=True).fit(fit_part[self.features], fit_part["home_pts"])
            a = ScoreEnsemble(fast=True).fit(fit_part[self.features], fit_part["away_pts"])
            rh = hold["home_pts"].values - h.predict(hold[self.features])
            ra = hold["away_pts"].values - a.predict(hold[self.features])
            self.sigma_home, self.sigma_away = float(np.std(rh)), float(np.std(ra))
            self.rho = float(np.corrcoef(rh, ra)[0, 1])
        self.home = ScoreEnsemble(fast=self.fast).fit(train[self.features], train["home_pts"])
        self.away = ScoreEnsemble(fast=self.fast).fit(train[self.features], train["away_pts"])
        self.trained_through = str(train["game_date"].max().date())
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        X = games[self.features]
        out = games[["game_id", "game_date", "home", "away"]].copy()
        out["pred_home_pts"] = self.home.predict(X)
        out["pred_away_pts"] = self.away.predict(X)
        out["pred_margin"] = out["pred_home_pts"] - out["pred_away_pts"]   # home perspective
        out["pred_spread"] = -out["pred_margin"]                            # betting-line convention
        out["pred_total"] = out["pred_home_pts"] + out["pred_away_pts"]
        return out

    def feature_importance(self) -> pd.DataFrame:
        imp = (self.home.models["lgbm"].feature_importances_ + self.away.models["lgbm"].feature_importances_)
        return (pd.DataFrame({"feature": self.features, "importance": imp})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    # ------------------------------------------------------------------ #
    def save(self, path: Path | str = config.MODELS_DIR / "game_model.joblib") -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = config.MODELS_DIR / "game_model.joblib") -> "GameModel":
        return joblib.load(path)
