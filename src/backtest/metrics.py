"""
Backtest metrics: score error, betting performance and probability calibration.

Betting convention (see config): spread is home-perspective, so the market's
expected home margin is `-spread`. Model edge on the spread is
`pred_margin - (-spread) = pred_margin + spread` (points in favour of home).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def score_metrics(df: pd.DataFrame) -> dict:
    """MAE/RMSE of predicted home, away, margin and total against actual results."""
    actual_margin = df["home_pts"] - df["away_pts"]
    actual_total = df["home_pts"] + df["away_pts"]
    out = {
        "n_games": int(len(df)),
        "home_pts": {"mae": mae(df["pred_home_pts"], df["home_pts"]), "rmse": rmse(df["pred_home_pts"], df["home_pts"])},
        "away_pts": {"mae": mae(df["pred_away_pts"], df["away_pts"]), "rmse": rmse(df["pred_away_pts"], df["away_pts"])},
        "margin": {"mae": mae(df["pred_margin"], actual_margin), "rmse": rmse(df["pred_margin"], actual_margin)},
        "total": {"mae": mae(df["pred_total"], actual_total), "rmse": rmse(df["pred_total"], actual_total)},
        "straight_up_accuracy": float(np.mean((df["pred_margin"] > 0) == (actual_margin > 0))),
    }
    lined = df.dropna(subset=["spread", "total"]) if {"spread", "total"} <= set(df.columns) else df.iloc[0:0]
    if len(lined):
        out["market_baseline"] = {
            "n_games": int(len(lined)),
            "margin_mae_market": mae(-lined["spread"], lined["home_pts"] - lined["away_pts"]),
            "margin_mae_model": mae(lined["pred_margin"], lined["home_pts"] - lined["away_pts"]),
            "total_mae_market": mae(lined["total"], lined["home_pts"] + lined["away_pts"]),
            "total_mae_model": mae(lined["pred_total"], lined["home_pts"] + lined["away_pts"]),
        }
    return out


def _settle(edge: np.ndarray, result: np.ndarray, threshold: float) -> dict:
    """Bet the side the edge points to when |edge| >= threshold (and edge != 0).
    `result` > 0 means the 'positive' side (home / over) won; 0 is a push."""
    bet = (np.abs(edge) >= threshold) & (edge != 0)
    side = np.sign(edge[bet])
    res = np.sign(result[bet])
    wins = int(np.sum(side * res > 0))
    losses = int(np.sum(side * res < 0))
    pushes = int(np.sum(res == 0))
    decided = wins + losses
    units = wins * config.WIN_PAYOUT - losses
    return {"threshold": threshold, "bets": int(bet.sum()), "wins": wins, "losses": losses, "pushes": pushes,
            "win_rate": wins / decided if decided else None,
            "units": float(units), "roi": float(units / bet.sum()) if bet.sum() else None}


def betting_metrics(df: pd.DataFrame, thresholds=config.EDGE_THRESHOLDS) -> dict:
    """ATS and over/under performance at several minimum-edge thresholds."""
    d = df.dropna(subset=["spread", "total"])
    if d.empty:
        return {"available": False, "reason": "no betting lines supplied for this season"}
    margin = (d["home_pts"] - d["away_pts"]).values
    total = (d["home_pts"] + d["away_pts"]).values
    ats_edge = (d["pred_margin"] + d["spread"]).values
    ou_edge = (d["pred_total"] - d["total"]).values
    return {
        "available": True,
        "breakeven_win_rate": config.BREAKEVEN_WIN_RATE,
        "ats": [_settle(ats_edge, margin + d["spread"].values, t) for t in thresholds],
        "over_under": [_settle(ou_edge, total - d["total"].values, t) for t in thresholds],
    }


def cumulative_units(df: pd.DataFrame, market: str = "ats", threshold: float = 0.0) -> pd.DataFrame:
    """Per-bet running profit (units) in chronological order, for ROI curves."""
    d = df.dropna(subset=["spread", "total"]).sort_values(["game_date", "game_id"])
    if market == "ats":
        edge = d["pred_margin"] + d["spread"]
        res = d["home_pts"] - d["away_pts"] + d["spread"]
    else:
        edge = d["pred_total"] - d["total"]
        res = d["home_pts"] + d["away_pts"] - d["total"]
    bet = (edge.abs() >= threshold) & (edge != 0)
    outcome = np.sign(edge[bet]) * np.sign(res[bet])
    profit = np.where(outcome > 0, config.WIN_PAYOUT, np.where(outcome < 0, -1.0, 0.0))
    return pd.DataFrame({"game_date": d.loc[bet, "game_date"].values, "profit": profit,
                         "cum_units": np.cumsum(profit), "bet_no": np.arange(1, len(profit) + 1)})


def probability_metrics(df: pd.DataFrame, bins: int = 10) -> dict:
    """Brier score, log loss and a reliability table for home win probability."""
    y = (df["home_pts"] > df["away_pts"]).astype(float).values
    p = np.clip(df["home_win_prob"].values, 1e-4, 1 - 1e-4)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    table = [{"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]), "n": int((idx == i).sum()),
              "mean_pred": float(p[idx == i].mean()), "observed": float(y[idx == i].mean())}
             for i in range(bins) if (idx == i).any()]
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier_baseline_home_rate": float(np.mean((y.mean() - y) ** 2)),
            "calibration": table}


def prop_metrics(df: pd.DataFrame, stats=config.PROP_STATS) -> dict:
    """Per-stat MAE of the median projection vs a last-10-game baseline, plus
    empirical coverage of the 80% [p10, p90] interval."""
    out = {}
    for s in stats:
        d = df.dropna(subset=[s, f"{s}_p50"])
        base = d.dropna(subset=[f"{s}_l10"]) if f"{s}_l10" in d else d.iloc[0:0]
        out[s] = {
            "n": int(len(d)),
            "mae": mae(d[f"{s}_p50"], d[s]),
            "baseline_l10_mae": mae(base[f"{s}_l10"], base[s]) if len(base) else None,
            "coverage_80": float(np.mean((d[s] >= d[f"{s}_p10"]) & (d[s] <= d[f"{s}_p90"]))),
            "mean_interval_width": float(np.mean(d[f"{s}_p90"] - d[f"{s}_p10"])),
        }
    return out
