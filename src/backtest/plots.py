"""
Static evaluation plots written to /assets (PNG, dark theme).

Colours follow a validated categorical palette (dark-surface steps): series 1
blue, series 2 orange; text and gridlines stay in neutral ink.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.backtest.metrics import cumulative_units  # noqa: E402

SURFACE, TEXT, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"
S1, S2 = "#3987e5", "#d95926"


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, loc="left", fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_visible(False)


def _fig(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def plot_all(games: pd.DataFrame, props: pd.DataFrame, report: dict, out_dir: Path = config.ASSETS_DIR) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Cumulative units (ATS & O/U, all bets at -110); remove any stale curve when lines are absent
    (out_dir / "roi_curve.png").unlink(missing_ok=True)
    if report["betting"].get("available"):
        fig, ax = _fig()
        for market, color, label in (("ats", S1, "Against the spread"), ("ou", S2, "Over/Under")):
            cu = cumulative_units(games, market)
            ax.plot(cu["bet_no"], cu["cum_units"], color=color, linewidth=2, label=label)
            ax.annotate(label, (cu["bet_no"].iloc[-1], cu["cum_units"].iloc[-1]), color=TEXT,
                        fontsize=9, xytext=(4, 0), textcoords="offset points", va="center")
        ax.axhline(0, color=MUTED, linewidth=1)
        _style(ax, "Cumulative profit, 1 unit per game at -110", "Bet number", "Units")
        ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT, loc="lower left")
        paths.append(_save(fig, out_dir / "roi_curve.png"))

    # 2. Win-probability calibration
    cal = pd.DataFrame(report["win_probability"]["calibration"])
    fig, ax = _fig(5.5, 5)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.plot(cal["mean_pred"], cal["observed"], color=S1, linewidth=2, marker="o", markersize=7)
    _style(ax, "Home win probability calibration", "Predicted", "Observed")
    paths.append(_save(fig, out_dir / "calibration.png"))

    # 3. Margin residual distribution
    resid = games["pred_margin"] - (games["home_pts"] - games["away_pts"])
    fig, ax = _fig()
    ax.hist(resid, bins=40, color=S1, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1)
    _style(ax, f"Spread residuals (predicted - actual margin), MAE {np.mean(np.abs(resid)):.2f}", "Points", "Games")
    paths.append(_save(fig, out_dir / "margin_residuals.png"))

    # 4. Props MAE: model vs last-10 baseline
    pm = pd.DataFrame(report["props"]).T
    x = np.arange(len(pm))
    fig, ax = _fig()
    ax.bar(x - 0.2, pm["mae"].astype(float), width=0.38, color=S1, label="Model (median)")
    ax.bar(x + 0.2, pm["baseline_l10_mae"].astype(float), width=0.38, color=S2, label="Last-10 average")
    ax.set_xticks(x, [s.upper() for s in pm.index])
    _style(ax, "Player prop MAE by stat", "", "Mean absolute error")
    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT)
    paths.append(_save(fig, out_dir / "props_mae.png"))

    # 5. Interval coverage vs 80% target
    fig, ax = _fig()
    ax.bar(x, pm["coverage_80"].astype(float) * 100, width=0.6, color=S1)
    ax.axhline(80, color=MUTED, linewidth=1, linestyle="--")
    ax.text(len(pm) - 0.5, 81, "80% target", color=MUTED, ha="right", fontsize=9)
    ax.set_xticks(x, [s.upper() for s in pm.index])
    ax.set_ylim(0, 100)
    _style(ax, "Share of outcomes inside the 80% interval", "", "%")
    paths.append(_save(fig, out_dir / "props_coverage.png"))
    return paths


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return str(path)
