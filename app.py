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


MARKET_COLS = ["spread", "total", "home_ml", "away_ml", "book", "vegas_home_prob", "kalshi_event",
               "kalshi_home_prob", "kalshi_home_ask", "kalshi_away_ask", "kalshi_volume", "kalshi_spread",
               "kalshi_total", "kalshi_best_side", "kalshi_best_ev", "spread_edge", "total_edge"]


def with_market_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Predictions made before the Kalshi integration (or in synthetic mode) lack market columns."""
    for c in MARKET_COLS:
        if c not in df:
            df[c] = "" if c in ("book", "kalshi_event", "kalshi_best_side") else np.nan
    return df


def styled(table: pd.DataFrame, fmt: dict[str, str]):
    """Show missing values as '—' instead of 'None'. Columns with gaps are rendered as
    formatted text; complete columns stay numeric (and sortable) with the same format."""
    out = table.copy()
    numeric_fmt = {}
    for c in out.columns:
        f = fmt.get(c)
        has_gaps = out[c].isna().any() or (not pd.api.types.is_numeric_dtype(out[c])
                                           and (out[c].astype(str).str.strip() == "").any())
        if f and has_gaps:
            out[c] = [("—" if pd.isna(v) else f.format(v)) for v in out[c]]
        elif f:
            numeric_fmt[c] = f
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: "—" if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else v)
    return out.style.format(numeric_fmt) if numeric_fmt else out


def drop_empty(table: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Hide columns with no data at all (instead of a column full of 'None')."""
    empty = [c for c in cols if c in table and (table[c].isna().all() or (table[c].astype(str).str.strip() == "").all())]
    return table.drop(columns=empty), empty


def fmt_spread(team: str, spread: float) -> str:
    return "n/a" if pd.isna(spread) else f"{team} {spread:+.1f}"


# --------------------------------------------------------------------------- #
# Layout chrome
# --------------------------------------------------------------------------- #
meta = load_meta()
st.sidebar.title("🏀 NBA Prediction Engine")
page = st.sidebar.radio("View", ["Today's Games", "Vegas vs Kalshi", "Player Props Hub", "Prop Bet Finder",
                                 "Lineup & Injury Simulator",
                                 "2025-26 Model Accuracy"])
st.sidebar.caption(f"Season {config.CURRENT_SEASON}")
if meta:
    st.sidebar.caption(f"Data source: **{meta.get('data_source', '?')}**  \n"
                       f"Data as of: {meta.get('as_of', '?')}  \n"
                       f"Latest completed game: {meta.get('latest_game', '?')}  \n"
                       f"Models trained through: {meta.get('trained_through', '?')}  \n"
                       f"Last run: {meta.get('last_run', meta.get('predicted_at', '?'))}")


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets file configured
        return default


def refresh_panel():
    """Sidebar: quick in-app re-price, and a full GitHub Actions rerun."""
    from src.ui import github_actions as gha

    st.sidebar.divider()
    st.sidebar.subheader("Refresh")
    if meta.get("quick_refreshed_at"):
        st.sidebar.caption(f"Last quick refresh: {meta['quick_refreshed_at'].replace('T', ' ')}")
    password = secret("REFRESH_PASSWORD")
    entered = st.sidebar.text_input("Refresh password", type="password") if password else ""
    allowed = not password or entered == password

    if st.sidebar.button("⚡ Quick refresh", use_container_width=True, disabled=not allowed,
                         help="About 30 seconds: latest injury report, Vegas odds, Kalshi prices and prop lines, "
                              "re-run through the current models. Does not add new games or retrain."):
        from src.pipeline import quick_refresh
        with st.sidebar.status("Re-pricing today's slate...", expanded=True) as box:
            try:
                res = quick_refresh(Storage())
            except Exception as exc:
                box.update(label="Quick refresh failed", state="error")
                st.sidebar.error(f"{type(exc).__name__}: {exc}")
                return
            box.update(label="Updated", state="complete")
        st.session_state["refresh_result"] = res
        st.cache_data.clear()
        st.rerun()
    res = st.session_state.pop("refresh_result", None)
    if res:
        if res.get("note"):
            st.sidebar.info(res["note"])
        else:
            lines = [f"{res['games']} games re-priced; {res['out_players']} rotation players out ({res['injury_source']})."]
            lines += [f"{m}: {a:.0%} → {b:.0%}" for m, a, b in res["changes"]] or ["No win probability moved 2+ points."]
            st.sidebar.success("  \n".join(lines))

    token, repo = secret("GITHUB_TOKEN"), secret("GITHUB_REPO", "AlfredoMend/nba-prediction-engine")
    if not token:
        st.sidebar.caption("Full rerun button: add `GITHUB_TOKEN` to the app's secrets to enable it.")
        return
    try:
        run = gha.latest_run(token, repo)
    except Exception as exc:
        st.sidebar.caption(f"Could not reach GitHub ({exc.__class__.__name__}).")
        return
    ok, why = gha.can_trigger(run)
    if st.sidebar.button("🔄 Full rerun (new games + retrain)", use_container_width=True, disabled=not (ok and allowed),
                         help="Starts the GitHub workflow: downloads the latest box scores, retrains, re-predicts. "
                              "Takes about 10 minutes; the site updates when it finishes."):
        try:
            gha.trigger_run(token, repo)
            st.sidebar.success("Started. The site updates automatically in about 10 minutes.")
        except Exception as exc:
            st.sidebar.error(str(exc))
    if run:
        state = run["conclusion"] or run["status"]
        st.sidebar.caption(f"Last full run: [{state}]({run['url']}), {run['minutes_ago']:.0f} min ago. {why}")


if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
refresh_panel()

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
    preds = with_market_cols(preds)
    preds["game_date"] = pd.to_datetime(preds["game_date"])
    dates = sorted(preds["game_date"].dt.date.unique())
    c1, c2 = st.columns([1, 3])
    day = c1.selectbox("Game date", dates, format_func=lambda d: d.strftime("%a %b %d, %Y"))
    only_value = c2.toggle("Only show value plays", value=False,
                           help=f"|model - market| ≥ {VALUE_EDGE_PTS} pts and model probability ≥ {VALUE_PROB:.0%}")
    g = preds[preds["game_date"].dt.date == day].copy()

    # picks only exist when a market line is available
    g["ats_pick"] = np.where(g["spread_edge"].isna(), "", np.where(g["spread_edge"] > 0, g["home"], g["away"]))
    g["ats_prob"] = np.where(g["spread_edge"] > 0, g["home_cover_prob"], 1 - g["home_cover_prob"] - g["spread_push_prob"].fillna(0))
    g["ou_pick"] = np.where(g["total_edge"].isna(), "", np.where(g["total_edge"] > 0, "Over", "Under"))
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
        "Vegas spread": [fmt_spread(h, s) for h, s in zip(g["home"], g["spread"])],
        "Kalshi spread": [fmt_spread(h, s) for h, s in zip(g["home"], g["kalshi_spread"])],
        "Spread edge": g["spread_edge"].abs(),
        "ATS pick": g["ats_pick"] + np.where(g["ats_value"], " ⭐", ""),
        "Cover %": g["ats_prob"] * 100,
        "Model total": g["pred_total"],
        "Vegas total": g["total"],
        "Kalshi total": g["kalshi_total"],
        "O/U pick": g["ou_pick"] + np.where(g["ou_value"], " ⭐", ""),
        "O/U %": g["ou_prob"] * 100,
    })
    table["Vegas spread"] = table["Vegas spread"].replace("n/a", None)
    table["Kalshi spread"] = table["Kalshi spread"].replace("n/a", None)
    table, hidden = drop_empty(table, ["Vegas spread", "Kalshi spread", "Spread edge", "ATS pick", "Cover %",
                                       "Vegas total", "Kalshi total", "O/U pick", "O/U %"])
    st.dataframe(styled(table, {"Spread edge": "{:.1f} pts", "Cover %": "{:.1f}%", "Model total": "{:.1f}",
                                "Vegas total": "{:.1f}", "Kalshi total": "{:.1f}", "O/U %": "{:.1f}%"}),
                 hide_index=True, use_container_width=True, column_config={
                     "Home win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    st.caption("⭐ = value play against the Vegas line. Spreads are shown from the home team's perspective. "
               "Probabilities come from 10,000 Monte Carlo simulations per game. Kalshi spread/total = the line where "
               "Kalshi's price ladder crosses 50%.")
    if hidden:
        st.info(f"Hidden until market data arrives: {', '.join(hidden)}. Vegas lines (ESPN) and Kalshi prices are "
                f"fetched on each daily run; last run found {meta.get('vegas_games', 0)} Vegas lines and "
                f"{meta.get('kalshi_games', 0)} Kalshi games.")

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
    m[3].metric("Model vs Vegas spread", f"{r.pred_spread:+.1f}",
                delta=None if pd.isna(r.spread) else f"{r.pred_spread - r.spread:+.1f} vs {r.spread:+.1f}",
                delta_color="off")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.margin_distribution(sims, r.home, r.away, r.spread), use_container_width=True)
    c2.plotly_chart(charts.total_distribution(sims, r.total), use_container_width=True)


# --------------------------------------------------------------------------- #
# 1b. Vegas vs Kalshi
# --------------------------------------------------------------------------- #
KALSHI_MIN_EV = 0.03   # flag Kalshi bets worth at least 3 cents per $1 contract after fees


def page_markets():
    preds = load_table("pred_games")
    if preds.empty:
        st.info("No upcoming games found.")
        return
    preds = with_market_cols(preds)
    preds["game_date"] = pd.to_datetime(preds["game_date"])
    c1, c2 = st.columns([1, 3])
    day = c1.selectbox("Game date", sorted(preds["game_date"].dt.date.unique()),
                       format_func=lambda d: d.strftime("%a %b %d, %Y"), key="mk_day")
    side = c2.radio("Show probabilities for", ["Home team", "Favourite"], horizontal=True)
    g = preds[preds["game_date"].dt.date == day].copy()
    g["label"] = g["away"] + " @ " + g["home"]

    has_v, has_k = g["vegas_home_prob"].notna() | g["spread"].notna(), g["kalshi_home_prob"].notna()
    gap = (g["vegas_home_prob"] - g["kalshi_home_prob"]).abs()
    value = g["kalshi_best_ev"] >= KALSHI_MIN_EV
    k = st.columns(4)
    k[0].metric("Games with Vegas lines", f"{int(has_v.sum())} / {len(g)}")
    k[1].metric("Games on Kalshi", f"{int(has_k.sum())} / {len(g)}")
    k[2].metric("Avg Vegas vs Kalshi gap", f"{gap.mean() * 100:.1f} pp" if gap.notna().any() else "n/a",
                help="Average absolute difference between the no-vig Vegas win probability and Kalshi's price")
    k[3].metric("Kalshi value plays", int(value.sum()),
                help=f"Model expects at least {KALSHI_MIN_EV * 100:.0f}¢ profit per $1 contract after Kalshi fees")

    if not has_v.any() and not has_k.any():
        meta_v, meta_k = meta.get("vegas_games", "?"), meta.get("kalshi_games", "?")
        st.info(f"No market data for this date yet (last run: {meta_v} Vegas lines, {meta_k} Kalshi games). Vegas lines "
                "come from ESPN's scoreboard odds and Kalshi prices from its public API on every daily run.")
        return

    # Probabilities from the chosen team's perspective
    flip = (side == "Favourite") & (g["vegas_home_prob"].fillna(g["kalshi_home_prob"]).fillna(g["home_win_prob"]) < 0.5)
    team = np.where(flip, g["away"], g["home"])
    def persp(col):
        return np.where(flip, 1 - g[col], g[col])
    g["p_model"], g["p_vegas"], g["p_kalshi"] = persp("home_win_prob"), persp("vegas_home_prob"), persp("kalshi_home_prob")
    g["label"] = g["label"] + " · " + team

    st.plotly_chart(charts.market_comparison(
        g, {"p_model": "Model", "p_vegas": "Vegas (no-vig)", "p_kalshi": "Kalshi"},
        "Win probability: model vs Vegas vs Kalshi", f"{side} win probability"), use_container_width=True)

    ev_txt = [f"{s} @ {a * 100:.0f}¢" if s else "" for s, a in zip(
        g["kalshi_best_side"], np.where(g["kalshi_best_side"] == g["home"], g["kalshi_home_ask"], g["kalshi_away_ask"]))]
    table = pd.DataFrame({
        "Matchup": g["away"] + " @ " + g["home"], "Team": team,
        "Model %": g["p_model"] * 100, "Vegas %": g["p_vegas"] * 100, "Kalshi %": g["p_kalshi"] * 100,
        "Vegas–Kalshi (pp)": (g["p_vegas"] - g["p_kalshi"]) * 100,
        "Model spread": [fmt_spread(h, s) for h, s in zip(g["home"], g["pred_spread"])],
        "Vegas spread": [fmt_spread(h, s) for h, s in zip(g["home"], g["spread"])],
        "Kalshi spread": [fmt_spread(h, s) for h, s in zip(g["home"], g["kalshi_spread"])],
        "Model total": g["pred_total"], "Vegas total": g["total"], "Kalshi total": g["kalshi_total"],
        "Kalshi bet": [t + (" ⭐" if v else "") for t, v in zip(ev_txt, value)],
        "EV / $1": g["kalshi_best_ev"] * 100, "Kalshi volume": g["kalshi_volume"],
    })
    st.dataframe(styled(table, {"Model %": "{:.1f}%", "Vegas %": "{:.1f}%", "Kalshi %": "{:.1f}%",
                                "Vegas–Kalshi (pp)": "{:+.1f}", "Model total": "{:.1f}", "Vegas total": "{:.1f}",
                                "Kalshi total": "{:.1f}", "EV / $1": "{:+.1f}¢", "Kalshi volume": "{:,.0f}"}),
                 hide_index=True, use_container_width=True)
    book = ", ".join(sorted({b for b in g["book"].dropna().astype(str) if b}))
    st.caption(
        f"Vegas: {book or 'n/a'} moneylines with the bookmaker margin removed. Kalshi: midpoint of the yes bid/ask "
        "for each team, normalised to sum to 100%. Kalshi bet = the side the model prefers, bought at the current ask; "
        f"EV is expected profit per $1 contract after Kalshi's taker fee. ⭐ = EV ≥ {KALSHI_MIN_EV * 100:.0f}¢. "
        "Prices move; check the live market before trading.")


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

    def col(name, default=np.nan):
        return p[f"{stat}_{name}"] if f"{stat}_{name}" in p else pd.Series(default, index=p.index)
    ev = col("ev")
    table = pd.DataFrame({
        "Player": p["player_name"], "Team": p["team"], "Opp": p["opp"],
        "Proj min": p["proj_min"], "Low (p10)": p[f"{stat}_p10"], "Projection": p[f"{stat}_p50"],
        "High (p90)": p[f"{stat}_p90"], "Last-10 avg": p[f"{stat}_l10"],
        "Book": col("book", ""), "Line": col("line"), "Over odds": col("over_odds"), "Under odds": col("under_odds"),
        "Model over %": col("over_prob") * 100, "Market over %": col("market_over_prob") * 100,
        "Bet": [f"{sd} {ln:g} ⭐" if pd.notna(e) and e >= PROP_MIN_EV else "" for sd, ln, e in
                zip(col("side", ""), col("line"), ev)],
        "EV %": ev * 100,
    })
    table, hidden = drop_empty(table, ["Book", "Line", "Over odds", "Under odds", "Model over %", "Market over %",
                                       "Bet", "EV %"])
    fmt = {c: "{:.1f}" for c in ["Proj min", "Low (p10)", "Projection", "High (p90)", "Last-10 avg", "Line"]}
    fmt |= {"Over odds": "{:+.0f}", "Under odds": "{:+.0f}", "Model over %": "{:.0f}%", "Market over %": "{:.0f}%",
            "EV %": "{:+.1f}%"}
    st.dataframe(styled(table, fmt), hide_index=True, use_container_width=True, height=460)
    st.caption(f"{len(p)} players. Projection = median of the quantile model; Low/High = 80% interval. "
               "Model over % is calibrated on the 2025-26 backtest; Market over % is the book's no-vig price. "
               f"⭐ = expected value ≥ {PROP_MIN_EV:.0%} per $1. See **Prop Bet Finder** for every line ranked by value.")
    if hidden:
        st.info("No prop lines loaded for this date, so book columns are hidden. Add lines (e.g. from Novig) to "
                "`data/prop_lines.csv`, or set `ODDS_API_KEY` + `ODDS_PROP_BOOKMAKERS`.")

    if not p.empty:
        who = st.selectbox("Player detail", p["player_id"], format_func=lambda i: p.loc[p["player_id"] == i, "player_name"].iloc[0])
        row = p[p["player_id"] == who].iloc[0]
        st.plotly_chart(charts.prop_ranges(row, config.PROP_STATS), use_container_width=True)


# --------------------------------------------------------------------------- #
# 2b. Prop bet finder
# --------------------------------------------------------------------------- #
PROP_MIN_EV = 0.03


def page_prop_bets():
    bets = load_table("prop_bets")
    st.markdown("Every prop line we have prices for, ranked by the model's **expected value** against that price. "
                "A projection above the line is not enough by itself: the model's chance of going over has to "
                "beat what the odds imply.")
    if bets.empty:
        st.info("No prop lines loaded yet. To compare against Novig (or any book):\n\n"
                "1. Copy the lines into `data/prop_lines.csv` with columns "
                "`game_date,player_name,stat,line,over_odds,under_odds,book` "
                "(example: `2026-10-27,Jayson Tatum,pts,23.5,-105,-105,novig`), commit it, and run the daily workflow; or\n"
                "2. Add an `ODDS_API_KEY` secret and an `ODDS_PROP_BOOKMAKERS` variable (e.g. `novig,draftkings`) "
                "in the repo settings so the daily run pulls them automatically.")
        return
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    c = st.columns([1, 1, 1, 1, 2])
    day = c[0].selectbox("Date", sorted(bets["game_date"].dt.date.unique()), key="pb_day")
    stats = c[1].multiselect("Stats", config.PROP_STATS, default=[], format_func=STAT_LABELS.get)
    books = c[2].multiselect("Books", sorted(bets["book"].dropna().unique()), default=[])
    min_ev = c[3].number_input("Min EV %", value=float(PROP_MIN_EV * 100), step=1.0) / 100
    search = c[4].text_input("Search player", "", key="pb_search")
    b = bets[(bets["game_date"].dt.date == day) & (bets["best_ev"] >= min_ev)]
    if stats:
        b = b[b["stat"].isin(stats)]
    if books:
        b = b[b["book"].isin(books)]
    if search:
        b = b[b["player_name"].str.contains(search, case=False, na=False)]

    k = st.columns(3)
    k[0].metric("Lines compared", int((bets["game_date"].dt.date == day).sum()))
    k[1].metric("Bets above threshold", len(b))
    k[2].metric("Best EV", f"{b['best_ev'].max():+.1%}" if len(b) else "n/a")

    table = pd.DataFrame({
        "Player": b["player_name"], "Team": b["team"], "Opp": b["opp"], "Stat": b["stat"].map(STAT_LABELS),
        "Book": b["book"], "Bet": b["best_side"] + " " + b["line"].map("{:g}".format),
        "Odds": b["best_odds"], "Projection": b["p50"], "80% range": b["p10"].round(0).astype(int).astype(str) + "–"
        + b["p90"].round(0).astype(int).astype(str),
        "Model %": np.where(b["best_side"] == "Over", b["model_over_prob"], 1 - b["model_over_prob"]) * 100,
        "Market %": np.where(b["best_side"] == "Over", b["market_over_prob"], 1 - b["market_over_prob"]) * 100,
        "EV %": b["best_ev"] * 100, "Stake (% bankroll)": b["kelly_stake"] * 100,
    })
    st.dataframe(table, hide_index=True, use_container_width=True, height=min(480, 40 + 35 * max(len(table), 1)),
                 column_config={
        "Odds": st.column_config.NumberColumn(format="%+d"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Model %": st.column_config.NumberColumn(format="%.1f%%"),
        "Market %": st.column_config.NumberColumn(format="%.1f%%"),
        "EV %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Stake (% bankroll)": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.caption("Model % = calibrated chance the bet wins; Market % = the book's no-vig implied chance. "
               "EV = expected profit per $1 staked. Stake = quarter-Kelly sizing guide, capped by your own judgement. "
               "Prop lines are not yet backtested against real closing prices, so treat edges as leads to check, "
               "not guarantees.")

    if len(b):
        r = b.iloc[0]
        p_side = r.model_over_prob if r.best_side == "Over" else 1 - r.model_over_prob
        m_side = r.market_over_prob if r.best_side == "Over" else 1 - r.market_over_prob
        st.markdown(
            f"**How to read the top row:** the model projects {r.player_name} for **{r.p50:.1f}** "
            f"{STAT_LABELS[r.stat].lower()} (80% range {r.p10:.0f}–{r.p90:.0f}) against a {r.book} line of "
            f"**{r.line:g}**. It gives the {r.best_side.lower()} a **{p_side:.0%}** chance; the {int(r.best_odds):+d} "
            f"price implies {m_side:.0%} after removing the vig. On average, a \\$100 bet is worth about "
            f"**{r.best_ev * 100:+.0f} dollars**.")


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


{"Today's Games": page_games, "Vegas vs Kalshi": page_markets, "Player Props Hub": page_props, "Prop Bet Finder": page_prop_bets, "Lineup & Injury Simulator": page_simulator,
 "2025-26 Model Accuracy": page_accuracy}[page]()
