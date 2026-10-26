"""
Player props engine: LightGBM quantile regression.

For each stat in config.PROP_STATS (PTS, REB, AST, 3PM, STL, BLK, PRA) three
models are trained at the 10th, 50th and 90th percentiles. The median is the
point projection and [p10, p90] is the 80% prediction interval. Quantiles are
sorted after prediction so the interval never crosses.

Over/under probabilities against a line are read off a two-piece normal fitted
to the three quantiles (different spread below/above the median, which captures
the right skew of counting stats).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.data.player_features import PROP_FEATURES

Z90 = norm.ppf(0.9)


class PropsModel:
    def __init__(self, stats=config.PROP_STATS, quantiles=config.QUANTILES, n_estimators: int = 300):
        self.stats, self.quantiles, self.n_estimators = list(stats), tuple(quantiles), n_estimators
        self.models: dict[tuple[str, float], lgb.LGBMRegressor] = {}
        self.features = list(PROP_FEATURES)
        self.trained_through = ""

    def fit(self, players: pd.DataFrame) -> "PropsModel":
        train = players[(players["is_upcoming"] == 0) & players["min"].notna() & (players["games_played"] > 0)]
        X = train[self.features]
        for stat in self.stats:
            for q in self.quantiles:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=self.n_estimators,
                                      learning_rate=0.05, num_leaves=31, min_child_samples=60,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                      random_state=config.RANDOM_SEED, verbose=-1, n_jobs=4)
                self.models[(stat, q)] = m.fit(X, train[stat])
        self.trained_through = str(train["game_date"].max().date())
        return self

    def predict(self, players: pd.DataFrame) -> pd.DataFrame:
        X = players[self.features]
        keep = ["game_id", "game_date", "player_id", "player_name", "team_id", "opp_id", "position",
                "is_home", "proj_min", "proj_usg", "base_min", "base_usg"]
        out = players[[c for c in keep if c in players.columns]].copy()
        for stat in self.stats:
            qs = np.sort(np.column_stack([self.models[(stat, q)].predict(X) for q in self.quantiles]), axis=1)
            qs = np.clip(qs, 0, None)
            for i, q in enumerate(self.quantiles):
                out[f"{stat}_p{int(q * 100)}"] = qs[:, i]
            if f"{stat}_l10" in players:
                out[f"{stat}_l10"] = players[f"{stat}_l10"].values
        return out

    @staticmethod
    def over_probability(p10, p50, p90, line):
        """P(stat > line) from a two-piece normal through (p10, p50, p90)."""
        p10, p50, p90, line = map(np.asarray, (p10, p50, p90, line))
        lo = np.maximum((p50 - p10) / Z90, 0.25)
        hi = np.maximum((p90 - p50) / Z90, 0.25)
        z = np.where(line <= p50, (line - p50) / lo, (line - p50) / hi)
        return 1 - norm.cdf(z)

    def save(self, path: Path | str = config.MODELS_DIR / "props_model.joblib") -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = config.MODELS_DIR / "props_model.joblib") -> "PropsModel":
        return joblib.load(path)
