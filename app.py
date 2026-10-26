"""
NBA Prediction Engine - Streamlit UI.

Run:  streamlit run app.py

Views
  1. Today's Games     predicted scores, model vs market spread/total, win & cover probabilities
  2. Player Props Hub  quantile projections with 80% bands, last-10 baseline and market lines
  3. Lineup Simulator  toggle players IN/OUT and re-run the game + props models live
  4. Model Accuracy    2025-26 walk-forward backtest diagnostics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src import config  # noqa: E402
from src.backtest.metrics import cumulative_units  # noqa: E402
from src.data.storage import Storage  # noqa: E402
from src.models.monte_carlo import simulate_single  # noqa: E402
from src.ui import charts  # noqa: E402

st.set_page_config(page_title="NBA Prediction Engine", page_icon="🏀", layout="wide")

STAT_LABELS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3PM", "stl": "Steals",
               "blk": "Blocks", "pra": "PTS+REB+AST"}
VALUE_EDGE_PTS = 2.0      # minimum |model - market| for a "value" flag
VALUE_PROB = 0.55         # minimum model probability for a "value" flag


# --------------------------------------------------------------------------- #
# Data access (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def load_table(name: str) -> pd.DataFrame:
    return Storage().read(name)


@st.cache_data(ttl=300)
def load_meta() -> dict:
    return Storage().get_meta()


@st.cache_resource
def load_models():
    from src.models.game_model import GameModel
    from src.models.props_model import PropsModel
    return GameModel.load(), PropsModel.load()


def fmt_spread(team: str, spread: float) -> str:
    return "n/a" if pd.isna(spread) else f"{team} {spread:+.1f}"


# --------------------------------------------------------------------------- #
# Layout chrome
# --------------------------------------------------------------------------- #
meta = load_meta()
st.sidebar.title("🏀 NBA Prediction Engine")
page = st.sidebar.radio("View", ["Today's Games", "Player Props Hub", "Lineup & Injury Simulator",
                                 "2025-26 Model Accuracy"])
st.sidebar.caption(f"Season {config.CURRENT_SEASON}")
if meta:
    st.sidebar.caption(f"Data source: **{meta.get('data_source', '?')}**  \n"
                       f"Data as of: {meta.get('as_of', '?')}  \n"
                       f"Latest completed game: {meta.get('latest_game', '?')}  \n"
                       f"Models trained through: {meta.get('trained_through', '?')}  \n"
                       f"Last run: {meta.get('last_run', meta.get('predicted_at', '?'))}")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if not meta:
    st.error("No data yet. Run `python main.py --update-and-predict` first.")
    st.stop()
if meta.get("data_source") == "synthetic":
    st.warning("**Demo mode: synthetic league.** Teams are real franchises but players, games and lines are "
               "simulated, so nothing here is a real NBA prediction or real accuracy. Run with "
               "`--source sportsdataverse` for real data.", icon="⚠️")


# --------------------------------------------------------------------------- #
# 1. Today's games
# --------------------------------------------------------------------------- #
def page_games():
    preds = load_table("pred_games")
    if preds.empty:
        st.info("No upcoming games found.")
        return
    preds["game_date"] = pd.to_datetime(preds["game_date"])
    dates = sorted(preds["game_date"].dt.date.unique())
    c1, c2 = st.columns([1, 3])
    day = c1.selectbox("Game date", dates, format_func=lambda d: d.strftime("%a %b %d, %Y"))
    only_value = c2.toggle("Only show value plays", value=False,
                           help=f"|model - market| ≥ {VALUE_EDGE_PTS} pts and model probability ≥ {VALUE_PROB:.0%}")
    g = preds[preds["game_date"].dt.date == day].copy()

    g["ats_pick"] = np.where(g["spread_edge"] > 0, g["home"], g["away"])
    g["ats_prob"] = np.where(g["spread_edge"] > 0, g["home_cover_prob"], 1 - g["home_cover_prob"] - g["spread_push_prob"].fillna(0))
    g["ou_pick"] = np.where(g["total_edge"] > 0, "Over", "Under")
    g["ou_prob"] = np.where(g["total_edge"] > 0, g["over_prob"], 1 - g["over_prob"] - g["total_push_prob"].fillna(0))
    g["ats_value"] = (g["spread_edge"].abs() >= VALUE_EDGE_PTS) & (g["ats_prob"] >= VALUE_PROB)
    g["ou_value"] = (g["total_edge"].abs() >= VALUE_EDGE_PTS) & (g["ou_prob"] >= VALUE_PROB)
    if only_value:
        g = g[g["ats_value"] | g["ou_value"]]

    k = st.columns(4)
    k[0].metric("Games", len(g))
    k[1].metric("Spread value plays", int(g["ats_value"].sum()))
    k[2].metric("Total value plays", int(g["ou_value"].sum()))
    k[3].metric("Avg predicted total", f"{g['pred_total'].mean():.1f}" if len(g) else "n/a")

    table = pd.DataFrame({
        "Matchup": g["away"] + " @ " + g["home"],
        "Pred score": g.apply(lambda r: f"{r.away} {r.pred_away_pts:.0f} - {r.home} {r.pred_home_pts:.0f}", axis=1),
        "Home win %": g["home_win_prob"] * 100,
        "Model spread": [fmt_spread(h, s) for h, s in zip(g["home"], g["pred_spread"])],
        "Market spread": [fmt_spread(h, s) for h, s in zip(g["home"], g["spread"])],
        "Spread edge": g["spread_edge"].abs(),
        "ATS pick": g["ats_pick"] + np.where(g["ats_value"], " ⭐", ""),
        "Cover %": g["ats_prob"] * 100,
        "Model total": g["pred_total"],
        "Market total": g["total"],
        "O/U pick": g["ou_pick"] + np.where(g["ou_value"], " ⭐", ""),
        "O/U %": g["ou_prob"] * 100,
    })
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        "Home win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Spread edge": st.column_config.NumberColumn(format="%.1f pts"),
        "Cover %": st.column_config.NumberColumn(format="%.1f%%"),
        "Model total": st.column_config.NumberColumn(format="%.1f"),
        "Market total": st.column_config.NumberColumn(format="%.1f"),
        "O/U %": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.caption("⭐ = value play. Spreads are shown from the home team's perspective. Probabilities come from "
               "10,000 Monte Carlo simulations per game.")

    st.subheader("Game detail")
    if g.empty:
        return
    pick = st.selectbox("Matchup", g["game_id"], format_func=lambda gid: " @ ".join(
        g.loc[g["game_id"] == gid, ["away", "home"]].iloc[0]))
    r = g[g["game_id"] == pick].iloc[0]
    sims = simulate_single(r.pred_home_pts, r.pred_away_pts, float(meta.get("sigma_home", 11.5)),
                           float(meta.get("sigma_away", 11.5)), float(meta.get("rho", 0.3)))
    m = st.columns(4)
    m[0].metric(f"{r.home} win probability", f"{r.home_win_prob:.1%}")
    m[1].metric("80% margin range", f"{r.margin_p10:+.0f} to {r.margin_p90:+.0f}")
    m[2].metric("80% total range", f"{r.total_p10:.0f} to {r.total_p90:.0f}")
    m[3].metric("Model vs market spread", f"{r.pred_spread:+.1f}",
                delta=None if pd.isna(r.spread) else f"{r.pred_spread - r.spread:+.1f} vs {r.spread:+.1f}",
                delta_color="off")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.margin_distribution(sims, r.home, r.away, r.spread), use_container_width=True)
    c2.plotly_chart(charts.total_distribution(sims, r.total), use_container_width=True)


# --------------------------------------------------------------------------- #
# 2. Player props hub
# --------------------------------------------------------------------------- #
def page_props():
    props = load_table("pred_props")
    if props.empty:
        st.info("No prop projections available.")
        return
    props["game_date"] = pd.to_datetime(props["game_date"])
    props = props[props["listed_out"] == 0]
    c = st.columns([1, 1, 2, 1, 1])
    day = c[0].selectbox("Date", sorted(props["game_date"].dt.date.unique()))
    stat = c[1].selectbox("Stat", config.PROP_STATS, format_func=STAT_LABELS.get)
    search = c[2].text_input("Search player", "")
    min_minutes = c[3].slider("Min projected minutes", 0, 40, 15)
    only_lines = c[4].toggle("Only with lines", value=False)
    teams = sorted(props["team"].dropna().unique())
    sel_teams = st.multiselect("Teams", teams, default=[])

    p = props[(props["game_date"].dt.date == day) & (props["proj_min"] >= min_minutes)]
    if sel_teams:
        p = p[p["team"].isin(sel_teams)]
    if search:
        p = p[p["player_name"].str.contains(search, case=False, na=False)]
    if only_lines:
        p = p[p[f"{stat}_line"].notna()]
    p = p.sort_values(f"{stat}_p50", ascending=False)

    edge = p[f"{stat}_p50"] - p[f"{stat}_line"]
    over = p[f"{stat}_over_prob"] if f"{stat}_over_prob" in p else pd.Series(np.nan, index=p.index)
    table = pd.DataFrame({
        "Player": p["player_name"], "Team": p["team"], "Opp": p["opp"],
        "Proj min": p["proj_min"], "Low (p10)": p[f"{stat}_p10"], "Projection": p[f"{stat}_p50"],
        "High (p90)": p[f"{stat}_p90"], "Last-10 avg": p[f"{stat}_l10"], "Line": p[f"{stat}_line"],
        "Edge": edge, "Over %": over * 100,
        "Lean": np.where(edge.isna(), "", np.where(over >= VALUE_PROB, "Over ⭐",
                                                   np.where(over <= 1 - VALUE_PROB, "Under ⭐", ""))),
    })
    st.dataframe(table, hide_index=True, use_container_width=True, height=460, column_config={
        c: st.column_config.NumberColumn(format="%.1f")
        for c in ["Proj min", "Low (p10)", "Projection", "High (p90)", "Last-10 avg", "Line", "Edge"]
    } | {"Over %": st.column_config.NumberColumn(format="%.0f%%")})
    st.caption(f"{len(p)} players. Projection = median of the quantile model; Low/High = 80% interval. "
               "⭐ = model over/under probability ≥ 55%.")

    if not p.empty:
        who = st.selectbox("Player detail", p["player_id"], format_func=lambda i: p.loc[p["player_id"] == i, "player_name"].iloc[0])
        row = p[p["player_id"] == who].iloc[0]
        st.plotly_chart(charts.prop_ranges(row, config.PROP_STATS), use_container_width=True)


# --------------------------------------------------------------------------- #
# 3. Lineup & injury simulator
# --------------------------------------------------------------------------- #
def page_simulator():
    from src.models.simulator import run_scenario

    gf = load_table("upcoming_game_features")
    pf = load_table("upcoming_player_features")
    preds = load_table("pred_games")
    if gf.empty or pf.empty:
        st.info("Run the pipeline first to populate upcoming games.")
        return
    game_model, props_model = load_models()
    gf["game_date"] = pd.to_datetime(gf["game_date"])
    label = {r.game_id: f"{r.game_date:%b %d}: {r.away} @ {r.home}" for r in gf.itertuples()}
    gid = st.selectbox("Game", list(label), format_func=label.get)
    game_row = gf[gf["game_id"] == gid].reset_index(drop=True)
    players = pf[pf["game_id"] == gid].reset_index(drop=True)
    base = preds[preds["game_id"] == gid].iloc[0]
    home, away = game_row["home"].iloc[0], game_row["away"].iloc[0]

    st.caption("Untick a player to rule him OUT (or tick an injured player back IN). Minutes and usage are "
               "redistributed and both models re-run instantly.")
    cols = st.columns(2)
    out_ids: set[int] = set()
    for col, team_id, team in ((cols[0], game_row["away_id"].iloc[0], away), (cols[1], game_row["home_id"].iloc[0], home)):
        tp = players[players["team_id"] == team_id].sort_values("base_min", ascending=False)
        edit = col.data_editor(
            pd.DataFrame({"IN": tp["listed_out"].values == 0, "Player": tp["player_name"].values,
                          "Avg min": tp["base_min"].round(1).values, "USG%": tp["base_usg"].round(1).values,
                          "player_id": tp["player_id"].values}),
            hide_index=True, use_container_width=True, disabled=["Player", "Avg min", "USG%"],
            column_config={"player_id": None, "IN": st.column_config.CheckboxColumn(width="small")},
            key=f"lineup_{gid}_{team_id}")
        col.markdown(f"**{team}**: {int(edit['IN'].sum())} available")
        out_ids |= set(edit.loc[~edit["IN"], "player_id"].astype(int))

    res = run_scenario(game_row, players, out_ids, game_model, props_model,
                       spread=base["spread"] if pd.notna(base["spread"]) else None,
                       total=base["total"] if pd.notna(base["total"]) else None)
    s = res.game.iloc[0]
    m = st.columns(4)
    m[0].metric("Model spread", fmt_spread(home, s.pred_spread), f"{s.pred_spread - base.pred_spread:+.1f} vs report",
                delta_color="off")
    m[1].metric(f"{home} win probability", f"{s.home_win_prob:.1%}", f"{(s.home_win_prob - base.home_win_prob) * 100:+.1f} pp")
    m[2].metric("Model total", f"{s.pred_total:.1f}", f"{s.pred_total - base.pred_total:+.1f}")
    if pd.notna(base["spread"]):
        m[3].metric(f"{home} cover prob ({base['spread']:+.1f})", f"{s.home_cover_prob:.1%}",
                    f"{(s.home_cover_prob - base.home_cover_prob) * 100:+.1f} pp")
    st.caption("Deltas compare the scenario with the published prediction (which uses the official injury report).")

    st.subheader("Usage & minutes redistribution")
    base_props = load_table("pred_props")
    base_props = base_props[base_props["game_id"] == gid][["player_id", "pts_p50", "pra_p50"]]
    p = res.players.merge(res.props[["player_id", "pts_p50", "reb_p50", "ast_p50", "pra_p50"]], on="player_id", how="left")
    p = p.merge(base_props.rename(columns={"pts_p50": "pts_base", "pra_p50": "pra_base"}), on="player_id", how="left")
    p["Team"] = np.where(p["team_id"] == game_row["home_id"].iloc[0], home, away)
    tbl = pd.DataFrame({
        "Team": p["Team"], "Player": p["player_name"], "Status": np.where(p["listed_out"] == 1, "OUT", "IN"),
        "Avg min": p["base_min"], "Proj min": p["proj_min"], "Avg USG%": p["base_usg"], "Proj USG%": p["proj_usg"],
        "PTS (report)": p["pts_base"], "PTS (scenario)": p["pts_p50"],
        "PRA (report)": p["pra_base"], "PRA (scenario)": p["pra_p50"],
    }).sort_values(["Team", "Proj min"], ascending=[True, False])
    st.dataframe(tbl, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in tbl.columns[3:]})


# --------------------------------------------------------------------------- #
# 4. Backtest accuracy hub
# --------------------------------------------------------------------------- #
def page_accuracy():
    if not config.BACKTEST_REPORT.exists():
        st.info("No backtest yet. Run `python main.py --backtest`.")
        return
    rep = json.loads(config.BACKTEST_REPORT.read_text())
    bt = load_table("bt_games")
    bp = load_table("bt_props")
    if rep.get("data_source") == "synthetic":
        st.error("These diagnostics come from the **synthetic** league. They show the backtest machinery works; "
                 "they are not real 2025-26 NBA accuracy.")
    st.caption(f"{rep['method']['type']} · game model retrained every {rep['method']['game_retrain_days']} days, "
               f"props every {rep['method']['prop_retrain_days']} days · leakage check "
               f"{'passed ✅' if rep['method']['leakage_check_passed'] else 'FAILED ❌'} · generated {rep['generated_at']}")

    sc, bet, wp = rep["scores"], rep["betting"], rep["win_probability"]
    k = st.columns(6)
    k[0].metric("Games", sc["n_games"])
    k[1].metric("Spread MAE", f"{sc['margin']['mae']:.2f}")
    k[2].metric("Total MAE", f"{sc['total']['mae']:.2f}")
    k[3].metric("Winner accuracy", f"{sc['straight_up_accuracy']:.1%}")
    k[4].metric("Brier score", f"{wp['brier']:.3f}", f"{wp['brier'] - wp['brier_baseline_home_rate']:+.3f} vs base rate",
                delta_color="inverse")
    if bet.get("available"):
        a0 = bet["ats"][0]
        k[5].metric("ATS win rate (all)", f"{a0['win_rate']:.1%}", f"ROI {a0['roi']:+.1%}", delta_color="off")

    if bet.get("available"):
        thr = st.select_slider("Minimum model edge to bet (points)", options=list(config.EDGE_THRESHOLDS),
                               value=config.EDGE_THRESHOLDS[0])
        bt["game_date"] = pd.to_datetime(bt["game_date"])
        curves = {"Against the spread": cumulative_units(bt, "ats", thr), "Over/Under": cumulative_units(bt, "ou", thr)}
        st.plotly_chart(charts.roi_curve(curves), use_container_width=True)
        rows = []
        for market, key in (("ATS", "ats"), ("Over/Under", "over_under")):
            for r in bet[key]:
                rows.append({"Market": market, "Min edge": r["threshold"], "Bets": r["bets"], "W": r["wins"],
                             "L": r["losses"], "P": r["pushes"],
                             "Win %": (r["win_rate"] or 0) * 100, "Units": r["units"], "ROI %": (r["roi"] or 0) * 100})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
            "Win %": st.column_config.NumberColumn(format="%.1f%%"), "Units": st.column_config.NumberColumn(format="%+.1f"),
            "ROI %": st.column_config.NumberColumn(format="%+.1f%%")})
        st.caption(f"Break-even at -110 is {config.BREAKEVEN_WIN_RATE:.2%}.")
        if "market_baseline" in sc:
            mb = sc["market_baseline"]
            st.caption(f"Market comparison: spread MAE model {mb['margin_mae_model']:.2f} vs market "
                       f"{mb['margin_mae_market']:.2f}; total MAE model {mb['total_mae_model']:.2f} vs market "
                       f"{mb['total_mae_market']:.2f}.")
    else:
        st.info("No betting lines were available for this season, so ATS/O-U/ROI are not computed. "
                "Add closing lines to `data/lines.csv` to enable them.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.calibration(pd.DataFrame(wp["calibration"])), use_container_width=True)
    c2.plotly_chart(charts.residual_hist(bt["pred_margin"] - (bt["home_pts"] - bt["away_pts"]),
                                         "Spread residuals"), use_container_width=True)

    bt["month"] = pd.to_datetime(bt["game_date"]).dt.to_period("M").astype(str)
    monthly = bt.groupby("month").apply(lambda d: pd.Series({
        "Spread MAE": np.mean(np.abs(d["pred_margin"] - (d["home_pts"] - d["away_pts"]))),
        "Total MAE": np.mean(np.abs(d["pred_total"] - (d["home_pts"] + d["away_pts"]))),
    }), include_groups=False).reset_index()
    st.plotly_chart(charts.monthly_line(monthly, "Error by month"), use_container_width=True)

    if rep.get("props"):
        st.subheader("Player props")
        pm = pd.DataFrame(rep["props"]).T.reset_index(names="stat")
        pm["Stat"] = pm["stat"].map(STAT_LABELS)
        pm["coverage_pct"] = pm["coverage_80"].astype(float) * 100
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.grouped_bars(pm, "Stat", {"mae": "Model (median)", "baseline_l10_mae": "Last-10 average"},
                                            "MAE by stat", "Mean absolute error"), use_container_width=True)
        c2.plotly_chart(charts.grouped_bars(pm, "Stat", {"coverage_pct": "Inside 80% interval"},
                                            "Interval coverage", "% of outcomes", ref=80), use_container_width=True)
        st.dataframe(pm[["Stat", "n", "mae", "baseline_l10_mae", "coverage_pct", "mean_interval_width"]].rename(columns={
            "n": "Player-games", "mae": "Model MAE", "baseline_l10_mae": "L10 MAE", "coverage_pct": "Coverage %",
            "mean_interval_width": "Avg interval width"}), hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.2f")
                           for c in ["Model MAE", "L10 MAE", "Coverage %", "Avg interval width"]})


{"Today's Games": page_games, "Player Props Hub": page_props, "Lineup & Injury Simulator": page_simulator,
 "2025-26 Model Accuracy": page_accuracy}[page]()
