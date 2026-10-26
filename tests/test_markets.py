"""Vegas (ESPN) and Kalshi parsing, odds maths and model-vs-market comparisons.

Payloads follow the documented ESPN scoreboard and Kalshi v2 `/markets` shapes.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.data import markets as mk


def _sched():
    return pd.DataFrame({"game_id": ["1", "2"], "game_date": pd.to_datetime(["2026-10-26", "2026-10-26"]),
                         "home": ["CHA", "OKC"], "away": ["WAS", "PHX"], "home_id": [1, 2], "away_id": [3, 4]})


# --------------------------------------------------------------------------- #
# odds maths
# --------------------------------------------------------------------------- #
def test_american_to_prob_and_no_vig():
    assert mk.american_to_prob(-150) == pytest.approx(0.6)
    assert mk.american_to_prob(150) == pytest.approx(0.4)
    p = mk.no_vig(mk.american_to_prob(-110), mk.american_to_prob(-110))
    assert p == pytest.approx(0.5)
    assert math.isnan(mk.american_to_prob(None))


def test_kalshi_fee_and_ev():
    assert mk.kalshi_fee(0.50) == pytest.approx(0.02)       # ceil(0.07 * 0.25 * 100) = 2 cents
    assert mk.kalshi_fee(0.95) == pytest.approx(0.01)
    assert mk.kalshi_ev(0.60, 0.50) == pytest.approx(0.60 - 0.50 - 0.02)
    assert math.isnan(mk.kalshi_ev(0.6, float("nan")))


# --------------------------------------------------------------------------- #
# ESPN Vegas lines
# --------------------------------------------------------------------------- #
def _espn_event(home, away, details, ou, ml_h=None, ml_a=None, new_layout=False):
    odds = {"provider": {"name": "ESPN BET"}, "details": details, "overUnder": ou}
    if new_layout:
        odds["moneyline"] = {"home": {"close": {"odds": ml_h}}, "away": {"close": {"odds": ml_a}}}
    else:
        odds["homeTeamOdds"] = {"moneyLine": ml_h}
        odds["awayTeamOdds"] = {"moneyLine": ml_a}
    return {"date": "2026-10-26T23:00Z", "competitions": [{
        "competitors": [{"homeAway": "home", "team": {"abbreviation": home}},
                        {"homeAway": "away", "team": {"abbreviation": away}}],
        "odds": [odds]}]}


def test_parse_espn_scoreboard_spread_sign_and_moneylines():
    payload = {"events": [_espn_event("CHA", "WSH", "CHA -6.5", 228.5, -250, 205),
                          _espn_event("OKC", "PHX", "PHX -1.5", 221.0, "+105", "-125", new_layout=True),
                          _espn_event("GS", "DEN", "EVEN", 230.0)]}
    df = mk.parse_espn_scoreboard(payload)
    assert list(df["home"]) == ["CHA", "OKC", "GSW"] and list(df["away"]) == ["WAS", "PHX", "DEN"]
    assert list(df["spread"]) == [-6.5, 1.5, 0.0]           # home perspective
    assert df.loc[0, "home_ml"] == -250 and df.loc[1, "home_ml"] == 105 and df.loc[1, "away_ml"] == -125
    assert df.loc[0, "game_date"] == pd.Timestamp("2026-10-26")   # 23:00Z is 7pm Eastern


# --------------------------------------------------------------------------- #
# Kalshi
# --------------------------------------------------------------------------- #
def _m(ticker, event, bid=None, ask=None, last=None, strike=None, dollars=True, volume=100):
    m = {"ticker": ticker, "event_ticker": event, "volume": volume}
    for f, v in (("yes_bid", bid), ("yes_ask", ask), ("last_price", last)):
        if v is not None:
            m[f"{f}_dollars" if dollars else f] = f"{v:.4f}" if dollars else int(round(v * 100))
    if strike is not None:
        m["floor_strike"] = strike
    return m


def _kalshi_payload():
    ev = "26OCT26WASCHA"
    winner = [_m(f"KXNBAGAME-{ev}-CHA", f"KXNBAGAME-{ev}", 0.70, 0.72),
              _m(f"KXNBAGAME-{ev}-WAS", f"KXNBAGAME-{ev}", 0.28, 0.31, dollars=False),     # legacy cents
              _m("KXNBAGAME-26OCT26PHXOKC-OKC", "KXNBAGAME-26OCT26PHXOKC", 0.55, 0.57),
              _m("KXNBAGAME-26OCT26PHXOKC-PHX", "KXNBAGAME-26OCT26PHXOKC", 0.43, 0.45)]
    spread = [  # P(CHA wins by > x)
        _m(f"KXNBASPREAD-{ev}-CHA3", f"KXNBASPREAD-{ev}", 0.64, 0.66, strike=2.5),
        _m(f"KXNBASPREAD-{ev}-CHA6", f"KXNBASPREAD-{ev}", 0.54, 0.56, strike=5.5),
        _m(f"KXNBASPREAD-{ev}-CHA9", f"KXNBASPREAD-{ev}", 0.44, 0.46, strike=8.5),
        _m(f"KXNBASPREAD-{ev}-WAS3", f"KXNBASPREAD-{ev}", 0.20, 0.22, strike=2.5),   # P(WAS wins by > 2.5)
    ]
    total = [_m(f"KXNBATOTAL-{ev}-220", f"KXNBATOTAL-{ev}", 0.70, 0.72),
             _m(f"KXNBATOTAL-{ev}-226", f"KXNBATOTAL-{ev}", 0.50, 0.52),
             _m(f"KXNBATOTAL-{ev}-232", f"KXNBATOTAL-{ev}", 0.30, 0.32)]
    return winner, spread, total


def test_event_date_and_suffix():
    assert mk.event_date("KXNBAGAME-26OCT26WASCHA") == pd.Timestamp("2026-10-26")
    assert mk._suffix("KXNBASPREAD-26OCT26WASCHA-CHA6") == ("CHA", 6.0)
    assert mk._suffix("KXNBATOTAL-26OCT26WASCHA-226") == (None, 226.0)
    assert mk._suffix("KXNBAGAME-26OCT26WASCHA-GS") == ("GSW", None)


def test_kalshi_game_lines_home_perspective():
    parsed = mk.parse_kalshi(*_kalshi_payload())
    k = mk.kalshi_game_lines(parsed, _sched()).set_index("home")
    cha = k.loc["CHA"]
    assert cha["away"] == "WAS"
    assert cha["kalshi_home_prob"] == pytest.approx(0.71 / (0.71 + 0.295), abs=1e-6)   # no-vig mids
    assert cha["kalshi_home_ask"] == pytest.approx(0.72) and cha["kalshi_away_ask"] == pytest.approx(0.31)
    # P(margin > x) crosses 0.5 between 5.5 (0.55) and 8.5 (0.45) -> 7.0 -> home spread -7.0
    assert cha["kalshi_spread"] == pytest.approx(-7.0)
    # totals: ticker 226 -> strike 225.5 at P=0.51; 231.5 at 0.31 -> crosses 0.5 at 225.8
    assert cha["kalshi_total"] == pytest.approx(225.5 + 6 * 0.01 / 0.20, abs=1e-6)
    assert math.isnan(k.loc["OKC", "kalshi_spread"])                          # no ladder for that game


def test_interp_half_outside_ladder_is_nan():
    assert math.isnan(mk._interp_half([(1.0, 0.8), (3.0, 0.6)]))


def test_attach_markets_and_value_columns():
    preds = pd.DataFrame({"game_id": ["1", "2"], "game_date": pd.to_datetime(["2026-10-26"] * 2),
                          "home": ["CHA", "OKC"], "away": ["WAS", "PHX"], "home_win_prob": [0.80, 0.50]})
    lines = mk.parse_espn_scoreboard({"events": [_espn_event("CHA", "WSH", "CHA -6.5", 228.5, -250, 205)]})
    kalshi = mk.kalshi_game_lines(mk.parse_kalshi(*_kalshi_payload()), _sched())
    out = mk.add_kalshi_value(mk.attach_markets(preds, lines, kalshi))
    cha, okc = out.iloc[0], out.iloc[1]
    assert cha["spread"] == -6.5 and cha["vegas_home_prob"] == pytest.approx(0.7143 / (0.7143 + 0.3279), abs=1e-3)
    assert cha["kalshi_best_side"] == "CHA"
    assert cha["kalshi_best_ev"] == pytest.approx(0.80 - 0.72 - mk.kalshi_fee(0.72))
    assert np.isnan(okc["spread"]) and np.isnan(okc["vegas_home_prob"])       # no Vegas line for OKC
    assert okc["kalshi_best_side"] in ("OKC", "PHX")


def test_attach_markets_without_any_markets():
    preds = pd.DataFrame({"game_id": ["1"], "game_date": pd.to_datetime(["2026-10-26"]), "home": ["CHA"],
                          "away": ["WAS"], "home_win_prob": [0.6]})
    out = mk.add_kalshi_value(mk.attach_markets(preds, pd.DataFrame(), pd.DataFrame()))
    for c in ("spread", "total", "vegas_home_prob", "kalshi_home_prob", "kalshi_spread", "kalshi_best_ev"):
        assert c in out and np.isnan(out[c].iloc[0])
    assert out["kalshi_best_side"].iloc[0] == ""


def test_refresh_markets_writes_tables(tmp_path, monkeypatch):
    """Pipeline wiring: ESPN + Kalshi fetchers -> lines / kalshi / line_history tables."""
    from datetime import date

    from src import pipeline
    from src.data.storage import Storage

    payload = {"events": [_espn_event("CHA", "WSH", "CHA -6.5", 228.5, -250, 205)]}
    monkeypatch.setattr(mk, "fetch_espn_lines", lambda start, days=3: mk.parse_espn_scoreboard(payload))
    monkeypatch.setattr(mk, "fetch_kalshi_lines",
                        lambda sched: mk.kalshi_game_lines(mk.parse_kalshi(*_kalshi_payload()), sched))
    monkeypatch.setattr(pipeline.odds, "load_lines_csv", lambda: pd.DataFrame())
    monkeypatch.setattr(pipeline.odds, "fetch_live_lines", lambda: pd.DataFrame())
    store = Storage(tmp_path / "t.duckdb")
    pipeline.refresh_markets(store, _sched(), date(2026, 10, 26))
    pipeline.refresh_markets(store, _sched(), date(2026, 10, 26))       # second snapshot appends
    assert len(store.read("lines")) == 1 and len(store.read("kalshi")) == 2
    hist = store.read("line_history")
    assert len(hist) == 4 and hist["kalshi_home_prob"].notna().sum() == 4
    assert store.get_meta()["vegas_games"] == "1" and store.get_meta()["kalshi_games"] == "2"
