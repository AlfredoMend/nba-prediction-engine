"""
Plotly chart builders for the Streamlit UI (dark theme, validated palette).

Series colours are assigned by identity in a fixed order (blue, orange, aqua),
text stays in neutral ink, grids are recessive, and every chart keeps Plotly's
hover tooltips on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

S1, S2, S3 = "#3987e5", "#d95926", "#199e70"
TEXT, MUTED, GRID = "#f0efec", "#c3c2b7", "#2c2c2a"


def _layout(fig: go.Figure, title: str, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=TEXT)),
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED), hoverlabel=dict(bgcolor="#262624", font_color=TEXT),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def margin_distribution(sims: pd.DataFrame, home: str, away: str, spread: float | None) -> go.Figure:
    counts = sims["margin"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values / len(sims) * 100, marker_color=S1,
                           marker_line_width=0, hovertemplate="Margin %{x}: %{y:.1f}%<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    if spread is not None and not np.isnan(spread):
        fig.add_vline(x=-spread, line_color=S2, line_width=2, line_dash="dash",
                      annotation_text=f"Market {home} {spread:+.1f}", annotation_font_color=TEXT)
    return _layout(fig, f"Simulated final margin ({home} minus {away}), 10,000 runs",
                   xaxis_title=f"{home} margin", yaxis_title="% of simulations", bargap=0.08)


def total_distribution(sims: pd.DataFrame, line: float | None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=sims["total"], nbinsx=50, marker_color=S1, marker_line_width=0,
                                 histnorm="percent", hovertemplate="Total %{x}: %{y:.1f}%<extra></extra>"))
    if line is not None and not np.isnan(line):
        fig.add_vline(x=line, line_color=S2, line_width=2, line_dash="dash",
                      annotation_text=f"Market {line:.1f}", annotation_font_color=TEXT)
    return _layout(fig, "Simulated total points", xaxis_title="Total", yaxis_title="% of simulations", bargap=0.08)


def prop_ranges(row: pd.Series, stats: list[str]) -> go.Figure:
    """80% interval bars per stat with median, last-10 average and market line."""
    labels = [s.upper() for s in stats]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[row[f"{s}_p90"] - row[f"{s}_p10"] for s in stats],
                         base=[row[f"{s}_p10"] for s in stats], orientation="h", marker_color=S1,
                         opacity=0.35, name="80% interval", width=0.5,
                         customdata=[[row[f"{s}_p10"], row[f"{s}_p90"]] for s in stats],
                         hovertemplate="%{y}: %{customdata[0]:.1f} to %{customdata[1]:.1f}<extra>80% interval</extra>"))
    fig.add_trace(go.Scatter(y=labels, x=[row[f"{s}_p50"] for s in stats], mode="markers", name="Projection (median)",
                             marker=dict(color=S1, size=12, line=dict(color="#121211", width=2)),
                             hovertemplate="%{y} projection %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(y=labels, x=[row.get(f"{s}_l10") for s in stats], mode="markers", name="Last-10 avg",
                             marker=dict(color=S3, size=10, symbol="diamond", line=dict(color="#121211", width=2)),
                             hovertemplate="%{y} last-10 avg %{x:.1f}<extra></extra>"))
    lines = [row.get(f"{s}_line") for s in stats]
    if any(pd.notna(v) for v in lines):
        fig.add_trace(go.Scatter(y=labels, x=lines, mode="markers", name="Market line",
                                 marker=dict(color=S2, size=14, symbol="line-ns-open", line=dict(width=3, color=S2)),
                                 hovertemplate="%{y} line %{x:.1f}<extra></extra>"))
    return _layout(fig, f"{row['player_name']}: projected range", height=80 + 55 * len(stats),
                   yaxis=dict(autorange="reversed"))


def roi_curve(curves: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for (name, cu), color in zip(curves.items(), (S1, S2, S3)):
        fig.add_trace(go.Scatter(x=cu["bet_no"], y=cu["cum_units"], mode="lines", name=name,
                                 line=dict(color=color, width=2),
                                 customdata=np.stack([cu["game_date"].astype(str)], axis=-1),
                                 hovertemplate="Bet %{x} (%{customdata[0]}): %{y:+.1f} units<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    return _layout(fig, "Cumulative profit, 1 unit per bet at -110", xaxis_title="Bet number", yaxis_title="Units",
                   hovermode="x unified")


def calibration(cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash", width=1),
                             name="Perfect", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["observed"], mode="lines+markers", name="Model",
                             line=dict(color=S1, width=2), marker=dict(size=9, line=dict(color="#121211", width=2)),
                             customdata=cal["n"],
                             hovertemplate="Predicted %{x:.2f} / observed %{y:.2f} (n=%{customdata})<extra></extra>"))
    return _layout(fig, "Home win probability calibration", xaxis_title="Predicted", yaxis_title="Observed",
                   xaxis_range=[0, 1], yaxis_range=[0, 1])


def residual_hist(resid: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=resid, nbinsx=45, marker_color=S1, marker_line_width=0,
                                 hovertemplate="%{x}: %{y} games<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    return _layout(fig, title, xaxis_title="Predicted minus actual (points)", yaxis_title="Games", bargap=0.08)


def grouped_bars(df: pd.DataFrame, x: str, series: dict[str, str], title: str, ytitle: str,
                 ref: float | None = None) -> go.Figure:
    fig = go.Figure()
    for (col, name), color in zip(series.items(), (S1, S2, S3)):
        fig.add_trace(go.Bar(x=df[x], y=df[col], name=name, marker_color=color, marker_line_width=0,
                             hovertemplate="%{x}: %{y:.2f}<extra>" + name + "</extra>"))
    if ref is not None:
        fig.add_hline(y=ref, line_color=MUTED, line_dash="dash", annotation_text=f"{ref:g} target",
                      annotation_font_color=MUTED)
    return _layout(fig, title, yaxis_title=ytitle, barmode="group", bargap=0.25, bargroupgap=0.08)


def monthly_line(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col, color in zip([c for c in df.columns if c != "month"], (S1, S2, S3)):
        fig.add_trace(go.Scatter(x=df["month"], y=df[col], mode="lines+markers", name=col,
                                 line=dict(color=color, width=2), marker=dict(size=8),
                                 hovertemplate="%{x}: %{y:.2f}<extra>" + col + "</extra>"))
    return _layout(fig, title, yaxis_title="MAE (points)", hovermode="x unified")
