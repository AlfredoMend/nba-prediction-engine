"""Shared pytest fixtures: a small synthetic league (~3 months of 2020-21)."""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import build_features  # noqa: E402
from src.data.synthetic import generate_league  # noqa: E402


@pytest.fixture(scope="session")
def league():
    tg, pg, players, sched, lines, prop_lines, inj = generate_league(through=date(2021, 1, 20), seed=7)
    return dict(team_games=tg, player_games=pg, players=players, schedule=sched, lines=lines,
                prop_lines=prop_lines, injuries=inj)


@pytest.fixture(scope="session")
def features(league):
    return build_features(league["team_games"], league["player_games"], league["players"],
                          league["schedule"], league["injuries"])
