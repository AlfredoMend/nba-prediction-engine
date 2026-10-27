"""
Player prop edges: model probability vs a book's price.

Example: the model projects Jayson Tatum for a 26-point median; the book (e.g.
Novig) offers Over 23.5 at -105 / Under 23.5 at -105.
  1. model P(over 23.5) comes from the p10/p50/p90 quantiles (two-piece normal),
     then passes through an isotonic calibrator learned on the 2025-26 walk-forward
     backtest (raw probabilities were ~2pp too optimistic on overs);
  2. the market's P(over) is the no-vig implied probability of the two prices;
  3. EV per $1 staked = p * profit_if_win - (1 - p); the side with the higher EV
     is the suggested bet, flagged when EV clears a threshold;
  4. a fractional Kelly stake is shown as a sizing guide.

A gap in points (26 vs 23.5) is not enough on its own: what matters is whether
the model's probability beats the price after the bookmaker's margin.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src import config
from src.data.markets import american_to_prob
from src.models.props_model import PropsModel

CALIBRATOR_PATH = config.MODELS_DIR / "prop_calibration.joblib"
KELLY_FRACTION = 0.25          # quarter Kelly: full Kelly is far too aggressive for model error
DEFAULT_ODDS = -110            # assumed price when a line is supplied without odds


def american_profit(ml) -> float:
    """Profit per $1 staked at American odds (e.g. -110 -> 0.909, +120 -> 1.2)."""
    ml = float(ml)
    return ml / 100 if ml > 0 else 100 / -ml


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
class PropCalibrator:
    """Per-stat isotonic map from raw quantile-model P(over) to observed frequency."""

    def __init__(self):
        self.maps: dict[str, IsotonicRegression] = {}
        self.report: dict = {}

    @staticmethod
    def _training_pairs(bt: pd.DataFrame, stat: str) -> tuple[np.ndarray, np.ndarray]:
        """Raw P(over) and outcomes at several realistic half-point lines per player-game."""
        d = bt.dropna(subset=[stat, f"{stat}_p50"])
        d = d[d["proj_min"] >= 10]
        width = (d[f"{stat}_p90"] - d[f"{stat}_p10"]).clip(lower=1)
        ps, ys = [], []
        for k in (-0.5, -0.25, 0.0, 0.25, 0.5):             # lines spread around the median
            line = np.floor(d[f"{stat}_p50"] + k * width) + 0.5
            line = np.maximum(line, 0.5)
            ps.append(PropsModel.over_probability(d[f"{stat}_p10"], d[f"{stat}_p50"], d[f"{stat}_p90"], line))
            ys.append((d[stat].values > line).astype(float))
        return np.concatenate(ps), np.concatenate(ys)

    def fit(self, bt: pd.DataFrame, validate: bool = True) -> "PropCalibrator":
        bt = bt.sort_values("game_date")
        for stat in config.PROP_STATS:
            if validate:  # honest check: fit on the first half of the season, score the second half
                half = bt["game_date"].quantile(0.5)
                p_tr, y_tr = self._training_pairs(bt[bt["game_date"] <= half], stat)
                p_te, y_te = self._training_pairs(bt[bt["game_date"] > half], stat)
                iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p_tr, y_tr)
                self.report[stat] = {
                    "brier_raw": float(np.mean((p_te - y_te) ** 2)),
                    "brier_calibrated": float(np.mean((iso.predict(p_te) - y_te) ** 2)),
                    "mean_raw": float(p_te.mean()), "mean_calibrated": float(iso.predict(p_te).mean()),
                    "observed_rate": float(y_te.mean()), "n": int(len(y_te)),
                }
            p, y = self._training_pairs(bt, stat)
            self.maps[stat] = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, y)
        return self

    def apply(self, stat: str, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if stat not in self.maps:
            return p
        out = np.full_like(p, np.nan)
        ok = ~np.isnan(p)
        out[ok] = self.maps[stat].predict(p[ok])
        return out

    def save(self, path: Path | str = CALIBRATOR_PATH) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = CALIBRATOR_PATH) -> "PropCalibrator | None":
        return joblib.load(path) if Path(path).exists() else None


# --------------------------------------------------------------------------- #
# Bet evaluation
# --------------------------------------------------------------------------- #
def evaluate_prop_bets(props: pd.DataFrame, lines: pd.DataFrame, calibrator: PropCalibrator | None = None,
                       min_ev: float = 0.03) -> pd.DataFrame:
    """One row per (player, stat, book line) with model vs market probability and EV.

    props  model output (one row per player-game, `<stat>_p10/_p50/_p90` columns)
    lines  long format: game_date, player_id, stat, line, over_odds, under_odds, book
    """
    cols = ["game_date", "player_id", "player_name", "team", "opp", "stat", "book", "line", "over_odds",
            "under_odds", "p10", "p50", "p90", "l10", "proj_min", "edge_pts", "model_over_prob",
            "market_over_prob", "ev_over", "ev_under", "best_side", "best_ev", "best_odds", "kelly_stake", "value"]
    if lines is None or lines.empty or props.empty:
        return pd.DataFrame(columns=cols)
    lines = lines.assign(game_date=pd.to_datetime(lines["game_date"]))
    lines = lines[lines["stat"].isin(config.PROP_STATS) & lines["player_id"].notna()]
    base = props[[c for c in ("game_date", "player_id", "player_name", "team", "opp", "proj_min") if c in props]]
    rows = []
    for stat, grp in lines.groupby("stat"):
        need = {f"{stat}_p10": "p10", f"{stat}_p50": "p50", f"{stat}_p90": "p90", f"{stat}_l10": "l10"}
        grp = grp.drop(columns=[c for c in ("player_name", "team", "opp", "proj_min") if c in grp])  # model's names win
        m = grp.merge(base.join(props[[c for c in need if c in props]].rename(columns=need)),
                      on=["game_date", "player_id"], how="inner")
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=cols)
    d = pd.concat(rows, ignore_index=True)
    for c in ("over_odds", "under_odds"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce").fillna(DEFAULT_ODDS)
    d["book"] = d.get("book", pd.Series("", index=d.index)).fillna("")
    raw = PropsModel.over_probability(d["p10"], d["p50"], d["p90"], d["line"])
    d["model_over_prob"] = raw
    if calibrator is not None:
        for stat, idx in d.groupby("stat").groups.items():
            d.loc[idx, "model_over_prob"] = calibrator.apply(stat, raw[d.index.get_indexer(idx)])
    po, pu = d["over_odds"].map(american_to_prob), d["under_odds"].map(american_to_prob)
    d["market_over_prob"] = po / (po + pu)
    p = d["model_over_prob"]
    d["ev_over"] = p * d["over_odds"].map(american_profit) - (1 - p)
    d["ev_under"] = (1 - p) * d["under_odds"].map(american_profit) - p
    over_better = d["ev_over"] >= d["ev_under"]
    d["best_side"] = np.where(over_better, "Over", "Under")
    d["best_ev"] = np.where(over_better, d["ev_over"], d["ev_under"])
    d["best_odds"] = np.where(over_better, d["over_odds"], d["under_odds"])
    # Kelly fraction of bankroll: (b*p - q) / b, scaled down, never negative
    b = d["best_odds"].map(american_profit)
    p_side = np.where(over_better, p, 1 - p)
    d["kelly_stake"] = np.clip((b * p_side - (1 - p_side)) / b, 0, None) * KELLY_FRACTION
    d["edge_pts"] = d["p50"] - d["line"]
    d["value"] = d["best_ev"] >= min_ev
    return d.reindex(columns=cols).sort_values("best_ev", ascending=False).reset_index(drop=True)
