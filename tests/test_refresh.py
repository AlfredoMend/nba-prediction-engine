"""In-app quick refresh and the GitHub Actions trigger helpers."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import pipeline
from src.data.storage import Storage
from src.models.game_model import GameModel
from src.models.props_model import PropsModel
from src.ui import github_actions as gha


@pytest.fixture(scope="module")
def models(features):
    return GameModel(fast=True).fit(features.games, calibrate=False), PropsModel(n_estimators=20).fit(features.players)


def test_quick_refresh_applies_new_injury(features, models, tmp_path, monkeypatch):
    game_model, props_model = models
    store = Storage(tmp_path / "q.duckdb")
    up_g = features.games[features.games["is_upcoming"] == 1]
    up_p = features.players[features.players["is_upcoming"] == 1]
    store.write("schedule", up_g[["game_id", "game_date", "home", "away", "home_id", "away_id"]])
    pipeline.publish_predictions(store, game_model, props_model, up_g, up_p)

    # rule the biggest-minutes available player of the first game OUT via the "live" injury report
    gid = up_g["game_id"].iloc[0]
    cand = up_p[(up_p["game_id"] == gid) & (up_p["listed_out"] == 0)].sort_values("base_min").iloc[-1]
    star = int(cand["player_id"])
    inj = pd.DataFrame({"player_id": [star], "player_name": [cand["player_name"]], "team": ["X"],
                        "status": ["OUT"], "detail": [""], "source": ["espn"]})
    monkeypatch.setattr(GameModel, "load", staticmethod(lambda *a, **k: game_model))
    monkeypatch.setattr(PropsModel, "load", staticmethod(lambda *a, **k: props_model))
    monkeypatch.setattr("src.data.injuries.load_injuries", lambda players, use_espn=True: inj)
    monkeypatch.setattr(pipeline, "refresh_markets", lambda store, sched, as_of: None)
    monkeypatch.setattr(pipeline.prop_odds, "load_prop_lines", lambda players, sched: pd.DataFrame())

    before = store.read("pred_games").set_index("game_id")["home_win_prob"]
    res = pipeline.quick_refresh(store, as_of=pd.Timestamp(up_g["game_date"].min()).date())
    after = store.read("pred_games").set_index("game_id")["home_win_prob"]

    assert res["games"] == len(up_g) and res["injury_source"] == "live report"
    props = store.read("pred_props")
    assert props.loc[props["player_id"] == star, "listed_out"].iloc[0] == 1
    assert after[gid] != before[gid]                              # the injury moved that game
    assert "quick_refreshed_at" in store.get_meta()


def _run(status="completed", minutes=60):
    return {"status": status, "conclusion": "success", "url": "u", "event": "schedule",
            "started": datetime.now(timezone.utc) - timedelta(minutes=minutes), "minutes_ago": minutes}


def test_can_trigger_rules():
    assert gha.can_trigger(None)[0]
    assert gha.can_trigger(_run())[0]
    assert not gha.can_trigger(_run("in_progress", 3))[0]
    assert not gha.can_trigger(_run("completed", 5))[0]           # cooldown


def test_trigger_run_raises_on_error(monkeypatch):
    class Resp:
        status_code, text = 403, "Resource not accessible by personal access token"
    monkeypatch.setattr(gha.requests, "post", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError, match="403"):
        gha.trigger_run("t", "o/r")
    Resp.status_code = 204
    gha.trigger_run("t", "o/r")                                   # no exception on success
