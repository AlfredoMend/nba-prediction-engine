"""ESPN (sportsdataverse) -> internal schema normalisation, using tiny in-memory frames."""
from datetime import date

import pandas as pd

from src.data.sportsdataverse import nba_season, normalize_player_box, normalize_schedule, normalize_team_box
from src.data.teams import ABBR_TO_ID

STATS = ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
         "three_point_field_goals_attempted", "free_throws_made", "free_throws_attempted",
         "offensive_rebounds", "defensive_rebounds", "assists", "steals", "blocks", "fouls"]


def _player(aid, team, opp, ha, mins, pts, dnp=False, stype=2):
    return dict(game_id=1, season=2026, season_type=stype, game_date="2026-01-10", athlete_id=aid,
                athlete_display_name=f"P{aid}", team_abbreviation=team, opponent_team_abbreviation=opp,
                home_away=ha, minutes=mins, points=pts, rebounds=5, turnovers=2, did_not_play=dnp,
                athlete_position_abbreviation="PG", **{s: 1 for s in STATS})


def _team(team, opp, ha, pts):
    return dict(game_id=1, season=2026, season_type=2, game_date="2026-01-10", team_abbreviation=team,
                opponent_team_abbreviation=opp, team_home_away=ha, team_score=pts, total_rebounds=40,
                total_turnovers=14, turnovers=12, **{s: 30 for s in STATS})


def test_season_mapping():
    assert nba_season(2026) == "2025-26"


def test_player_box_maps_teams_and_drops_dnp_and_preseason():
    raw = pd.DataFrame([_player(1, "NY", "SA", "home", 30, 20), _player(2, "NY", "SA", "home", 0, 0, dnp=True),
                        _player(3, "SA", "NY", "away", 25, 15), _player(4, "NY", "SA", "home", 20, 9, stype=1)])
    pg = normalize_player_box(raw)
    assert list(pg["player_id"]) == [1, 3]
    assert pg.loc[0, "team_id"] == ABBR_TO_ID["NYK"] and pg.loc[0, "opp_id"] == ABBR_TO_ID["SAS"]
    assert pg.loc[0, "is_home"] == 1 and pg.loc[0, "season"] == "2025-26"
    assert pg.loc[0, "_position"] == "G"


def test_team_box_minutes_from_players_and_overtime():
    pg = normalize_player_box(pd.DataFrame(
        [_player(i, "NY", "SA", "home", 53, 10) for i in range(5)] + [_player(9, "SA", "NY", "away", 240, 99)]))
    tg = normalize_team_box(pd.DataFrame([_team("NY", "SA", "home", 110), _team("SA", "NY", "away", 104)]), pg)
    assert len(tg) == 2
    mins = dict(zip(tg["team"], tg["min"]))
    assert mins["NYK"] == 265 and mins["SAS"] == 240      # 5 x 53 = 265 -> one overtime
    assert (tg["tov"] == 14).all()                           # total_turnovers (incl. team turnovers)


def test_schedule_keeps_only_future_incomplete_games():
    raw = pd.DataFrame({
        "game_id": [1, 2, 3], "game_date": ["2026-10-25", "2026-10-26", "2026-10-27"], "season": 2027,
        "season_type": 2, "status_type_completed": [True, False, False],
        "home_abbreviation": ["GS", "UTAH", "WSH"], "away_abbreviation": ["NO", "LAL", "BOS"],
        "status_type_detail": ["Final", "7:00 PM", "7:30 PM"],
    })
    s = normalize_schedule(raw, date(2026, 10, 26), days=1)
    assert list(s["game_id"]) == ["2"]
    assert s.loc[0, "home"] == "UTA" and s.loc[0, "season"] == "2026-27"
