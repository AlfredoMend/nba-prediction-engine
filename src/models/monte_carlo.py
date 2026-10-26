"""
Monte Carlo outcome simulation.

Each game is simulated N times (default 10,000) by drawing correlated home/away
scores from a bivariate normal centred on the ensemble's predicted scores, with
the residual sigmas and correlation estimated out-of-sample by GameModel.
Scores are rounded to whole points; regulation ties go to a simulated overtime.

Outputs per game: win probability, cover probability vs the spread, over
probability vs the total, push probabilities, and score/margin quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def simulate_scores(pred_home, pred_away, sigma_home: float, sigma_away: float, rho: float,
                    n_sims: int = config.N_SIMULATIONS, seed: int = config.RANDOM_SEED):
    """Return integer score samples of shape (n_games, n_sims) for home and away.

    Common random numbers: every game uses the same standard-normal draws, so a
    game's probabilities do not depend on which other games are in the batch and
    a what-if scenario differs from the baseline only through its inputs.
    """
    rng = np.random.default_rng(seed)
    ph = np.atleast_1d(np.asarray(pred_home, dtype=float))[:, None]
    pa = np.atleast_1d(np.asarray(pred_away, dtype=float))[:, None]
    z1 = rng.standard_normal((1, n_sims))
    z2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 0)) * rng.standard_normal((1, n_sims))
    home = np.rint(ph + sigma_home * z1)
    away = np.rint(pa + sigma_away * z2)
    # Overtime: 5-minute periods (~5/48 of a game) until the tie is broken
    tie = home == away
    while tie.any():
        scale = 5 / 48
        home = np.where(tie, home + np.rint(ph * scale + 3.0 * rng.standard_normal((1, n_sims))), home)
        away = np.where(tie, away + np.rint(pa * scale + 3.0 * rng.standard_normal((1, n_sims))), away)
        tie = home == away
    return home, away


def simulate_games(preds: pd.DataFrame, sigma_home: float, sigma_away: float, rho: float,
                   lines: pd.DataFrame | None = None, n_sims: int = config.N_SIMULATIONS,
                   seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Vectorised simulation for many games. `preds` needs pred_home_pts/pred_away_pts;
    optional `spread`/`total` columns (home-perspective spread) give cover/over probs."""
    out = preds.copy()
    home, away = simulate_scores(out["pred_home_pts"].values, out["pred_away_pts"].values,
                                 sigma_home, sigma_away, rho, n_sims, seed)
    margin, total = home - away, home + away
    out["home_win_prob"] = (margin > 0).mean(axis=1)
    out["sim_margin_mean"] = margin.mean(axis=1)
    out["sim_total_mean"] = total.mean(axis=1)
    for q in (10, 50, 90):
        out[f"margin_p{q}"] = np.percentile(margin, q, axis=1)
        out[f"total_p{q}"] = np.percentile(total, q, axis=1)
    if "spread" in out:
        sp = out["spread"].values[:, None]
        res = margin + sp
        out["home_cover_prob"] = np.where(np.isnan(sp[:, 0]), np.nan, (res > 0).mean(axis=1))
        out["spread_push_prob"] = np.where(np.isnan(sp[:, 0]), np.nan, (res == 0).mean(axis=1))
    if "total" in out:
        tl = out["total"].values[:, None]
        out["over_prob"] = np.where(np.isnan(tl[:, 0]), np.nan, (total > tl).mean(axis=1))
        out["total_push_prob"] = np.where(np.isnan(tl[:, 0]), np.nan, (total == tl).mean(axis=1))
    return out


def simulate_single(pred_home: float, pred_away: float, sigma_home: float, sigma_away: float,
                    rho: float, n_sims: int = config.N_SIMULATIONS, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Full sample table for one game (used by the UI for distribution charts)."""
    h, a = simulate_scores([pred_home], [pred_away], sigma_home, sigma_away, rho, n_sims, seed)
    return pd.DataFrame({"home": h[0], "away": a[0], "margin": h[0] - a[0], "total": h[0] + a[0]})
