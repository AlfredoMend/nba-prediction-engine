"""
Synthetic NBA league generator (offline / CI / demo mode).

!!! These are SIMULATED games, not real NBA results. !!!

The generator exists so that the full pipeline - features, models, walk-forward
backtest, UI - can be developed and tested without network access to
stats.nba.com. It emits exactly the same normalized schema as `nba_fetch`, so
every downstream module is source-agnostic.

The simulated league has the structure the models are supposed to learn:
  * latent team offence/defence strength that drifts within and across seasons
  * players with tiers (star/starter/bench), positions, usage and per-minute rates
  * player impact on team efficiency -> injuries genuinely move spreads
  * home-court advantage, back-to-back fatigue and travel effects
  * pace variation, overtime, and internally consistent box scores
    (team PTS == sum of player PTS, FGM <= FGA, ...)
  * a noisy-but-sharp "market" producing spreads/totals and prop lines
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import config
from src.data.teams import TEAMS

FIRST = ["Jalen", "Marcus", "Tyrese", "Devin", "Andre", "Malik", "Isaiah", "Darius", "Caleb", "Jordan",
         "Elijah", "Trey", "Noah", "Miles", "Xavier", "Quentin", "Zion", "Kobe", "Luka", "Nikola",
         "Theo", "Omar", "Rasheed", "Bennett", "Cole", "Dante", "Emeka", "Felix", "Gabe", "Hassan"]
LAST = ["Walker", "Brooks", "Hayes", "Carter", "Price", "Coleman", "Reed", "Bryant", "Foster", "Sims",
        "Ward", "Barnes", "Hughes", "Bishop", "Grant", "Holloway", "Okafor", "Petrovic", "Lindqvist",
        "Mbeki", "Tanaka", "Moreau", "Silva", "Kowalski", "Adeyemi", "Novak", "Duarte", "Fischer", "Quinn", "Vance"]

SEASON_START = {s: date(int(s[:4]), 10, 20) for s in config.SEASONS}
SEASON_DAYS = 172  # ~ late October -> mid April

# Tier templates: (count per roster, base minutes, usage %, net impact per 48)
TIERS = [("star", 1, 34.5, 30.0, 5.0), ("starter", 4, 29.0, 20.0, 1.2),
         ("rotation", 4, 18.0, 17.0, -0.5), ("bench", 4, 7.0, 15.0, -2.0)]
POS_RATES = {  # per-36 rebounds, assists, steals, blocks, share of shots that are 3s
    "G": dict(reb=4.2, ast=6.2, stl=1.3, blk=0.3, three=0.48),
    "F": dict(reb=7.0, ast=3.2, stl=1.0, blk=0.7, three=0.40),
    "C": dict(reb=11.0, ast=2.6, stl=0.8, blk=1.7, three=0.12),
}


@dataclass
class Player:
    pid: int
    name: str
    pos: str
    tier: str
    base_min: float
    usage: float
    impact: float
    rates: dict
    team: int
    injured_games: int = 0


@dataclass
class League:
    rng: np.random.Generator
    players: list[Player] = field(default_factory=list)
    team_off: dict = field(default_factory=dict)
    team_def: dict = field(default_factory=dict)
    team_pace: dict = field(default_factory=dict)


def _make_player(lg: League, pid: int, team: int, tier, pos: str) -> Player:
    name, n, mins, usg, imp = tier
    r = lg.rng
    rates = {k: v * r.uniform(0.7, 1.35) for k, v in POS_RATES[pos].items()}
    return Player(pid=pid, name=f"{r.choice(FIRST)} {r.choice(LAST)}", pos=pos, tier=name,
                  base_min=mins * r.uniform(0.9, 1.08), usage=usg * r.uniform(0.85, 1.15),
                  impact=imp + r.normal(0, 1.0), rates=rates, team=team)


def _init_league(seed: int) -> League:
    lg = League(rng=np.random.default_rng(seed))
    pid = 9_000_000  # synthetic ids live far from real nba ids
    for tid in TEAMS:
        lg.team_off[tid] = lg.rng.normal(0, 2.0)
        lg.team_def[tid] = lg.rng.normal(0, 2.0)
        lg.team_pace[tid] = lg.rng.normal(99.5, 2.0)
        positions = iter(["G", "G", "F", "F", "C", "G", "F", "C", "G", "F", "F", "C", "G"])
        for tier in TIERS:
            for _ in range(tier[1]):
                lg.players.append(_make_player(lg, pid, tid, tier, next(positions)))
                pid += 1
    return lg


def _offseason(lg: League) -> None:
    """Between seasons: team strength regresses + drifts, players age and move."""
    r = lg.rng
    for tid in TEAMS:
        lg.team_off[tid] = 0.6 * lg.team_off[tid] + r.normal(0, 1.5)
        lg.team_def[tid] = 0.6 * lg.team_def[tid] + r.normal(0, 1.5)
        lg.team_pace[tid] = 0.7 * lg.team_pace[tid] + 0.3 * 99.5 + r.normal(0, 1.0)
    for p in lg.players:
        p.impact += r.normal(0, 0.8)
        p.usage = float(np.clip(p.usage + r.normal(0, 1.0), 10, 36))
        p.injured_games = 0
    # ~12% of players switch teams (swap pairs of the same tier to keep rosters balanced)
    by_tier: dict[str, list[Player]] = {}
    for p in lg.players:
        by_tier.setdefault(p.tier, []).append(p)
    for plist in by_tier.values():
        idx = r.permutation(len(plist))
        for a, b in zip(idx[: len(idx) // 16 * 2: 2], idx[1: len(idx) // 16 * 2: 2]):
            plist[a].team, plist[b].team = plist[b].team, plist[a].team


def _schedule_season(lg: League, season: str) -> list[tuple[date, int, int]]:
    """Random but realistic schedule: ~82 games per team, ~7 games per night."""
    r = lg.rng
    start = SEASON_START[season]
    tids = list(TEAMS)
    games_played = {t: 0 for t in tids}
    home_count = {t: 0 for t in tids}
    out = []
    for day in range(SEASON_DAYS):
        d = start + timedelta(days=day)
        n_games = int(np.clip(r.normal(7.2, 2.5), 2, 13))
        avail = [t for t in tids if games_played[t] < 82]
        avail.sort(key=lambda t: games_played[t] + r.uniform(0, 3))
        chosen = avail[: 2 * min(n_games, len(avail) // 2)]
        r.shuffle(chosen)
        for a, b in zip(chosen[::2], chosen[1::2]):
            home, away = (a, b) if home_count[a] <= home_count[b] else (b, a)
            out.append((d, home, away))
            games_played[a] += 1
            games_played[b] += 1
            home_count[home] += 1
    return out


def _allocate(rng, total: int, weights: np.ndarray) -> np.ndarray:
    """Multinomial split of an integer team total across players."""
    if total <= 0 or weights.sum() <= 0:
        return np.zeros(len(weights), dtype=int)
    return rng.multinomial(int(total), weights / weights.sum())


def _roster_state(lg: League, tid: int, is_b2b: bool):
    """Active players, minutes allocation and the lineup's net-strength adjustment."""
    r = lg.rng
    roster = [p for p in lg.players if p.team == tid]
    active = []
    for p in roster:
        if p.injured_games > 0:
            p.injured_games -= 1
            continue
        if r.random() < 0.022:  # new injury: mostly short, occasionally long
            p.injured_games = int(r.geometric(0.25)) if r.random() < 0.85 else int(r.integers(10, 30))
            continue
        if is_b2b and p.tier == "star" and r.random() < 0.15:  # load management
            continue
        active.append(p)
    base = np.array([p.base_min for p in active])
    mins = np.clip(base * config.TEAM_MINUTES / base.sum() + r.normal(0, 2.5, len(base)), 1, 46)
    mins = mins * config.TEAM_MINUTES / mins.sum()
    # minutes-weighted impact: losing a star hands his minutes to lower-impact players
    strength = float(np.sum([p.impact * m / 48 for p, m in zip(active, mins)]))
    return active, mins, strength


def _team_box(rng, pts: int, poss: float) -> dict:
    """Build an internally consistent team box score that sums to `pts`."""
    fta = max(int(round(poss * rng.normal(0.25, 0.04))), 0)
    ftm = int(rng.binomial(fta, 0.78))
    fg3a = max(int(round(poss * rng.normal(0.40, 0.04))), 1)
    fg3m = int(rng.binomial(fg3a, 0.36))
    while pts - ftm - 3 * fg3m < 0:
        fg3m -= 1
    two_pts = pts - ftm - 3 * fg3m
    if two_pts % 2:  # odd remainder -> one extra free throw
        ftm += 1
        fta = max(fta, ftm)
        two_pts -= 1
    fg2m = two_pts // 2
    fg2a = max(int(round(fg2m / np.clip(rng.normal(0.54, 0.03), 0.4, 0.7))), fg2m)
    return dict(pts=pts, fgm=fg2m + fg3m, fga=fg2a + fg3a, fg3m=fg3m, fg3a=fg3a, ftm=ftm, fta=fta,
                tov=max(int(round(poss * rng.normal(0.13, 0.02))), 3), pf=int(rng.normal(19.5, 3.5)))


def generate_league(through: date | None = None, seed: int = config.RANDOM_SEED,
                    schedule_days: int = 3):
    """Simulate seasons 2020-21 -> current up to `through` (exclusive).

    Returns (team_games, player_games, players, schedule, lines, prop_lines, injuries).
    Games on/after `through` are returned in `schedule` (no box score) for the
    next `schedule_days` days, together with market lines.
    """
    through = through or config.today()
    lg = _init_league(seed)
    team_rows, player_rows, sched_rows, line_rows, prop_rows, inj_rows = [], [], [], [], [], []
    last_played: dict[int, date] = {}
    last_city: dict[int, int] = {}
    gid = 0
    for si, season in enumerate(config.SEASONS):
        if SEASON_START[season] >= through + timedelta(days=schedule_days):
            break
        if si:
            _offseason(lg)
        for d, h, a in _schedule_season(lg, season):
            if d >= through + timedelta(days=schedule_days):
                break
            gid += 1
            game_id = f"S{gid:08d}"
            r = lg.rng
            # in-season random-walk drift of latent strengths
            for t in (h, a):
                lg.team_off[t] += r.normal(0, 0.08)
                lg.team_def[t] += r.normal(0, 0.08)
            b2b = {t: last_played.get(t) == d - timedelta(days=1) for t in (h, a)}
            travel = {t: _travel(last_city.get(t, t), h) for t in (h, a)}
            states = {t: _roster_state(lg, t, b2b[t]) for t in (h, a)}
            ortg = {}
            for t, o in ((h, a), (a, h)):
                adj = 1.1 if t == h else -1.1                       # home court (+2.2 net)
                adj -= 1.6 if b2b[t] else 0.0                        # fatigue
                adj -= 0.4 * travel[t] / 1000                        # travel
                ortg[t] = (114 + lg.team_off[t] - lg.team_def[o] + adj
                           + 0.55 * states[t][2] - 0.45 * states[o][2])
            poss = (lg.team_pace[h] + lg.team_pace[a]) / 2

            if d >= through:
                # future game: publish schedule + market lines only
                exp_h, exp_a = poss * ortg[h] / 100, poss * ortg[a] / 100
                sched_rows.append(dict(game_id=game_id, game_date=pd.Timestamp(d), season=season,
                                       home_id=h, away_id=a, home=TEAMS[h][0], away=TEAMS[a][0],
                                       status="Scheduled"))
                line_rows.append(_market_line(r, d, h, a, exp_h, exp_a))
                prop_rows.extend(_market_props(r, d, states, ortg, poss))
                # synthetic injury report: rostered players not active for this game
                for t in (h, a):
                    active_ids = {p.pid for p in states[t][0]}
                    inj_rows.extend(dict(game_date=pd.Timestamp(d), player_id=p.pid, player_name=p.name,
                                         team=TEAMS[t][0], status="OUT")
                                    for p in lg.players if p.team == t and p.pid not in active_ids)
                # keep injury counters ticking but do not mark as played
                continue

            poss_g = poss + r.normal(0, 3.0)
            pts = {t: int(round(poss_g * (ortg[t] + r.normal(0, 8.5)) / 100)) for t in (h, a)}
            minutes = 240
            while pts[h] == pts[a]:  # overtime
                minutes += 25
                for t in (h, a):
                    pts[t] += int(max(r.normal(11, 3.5), 2))
            boxes = {t: _team_box(r, pts[t], poss_g) for t in (h, a)}
            for t, o in ((h, a), (a, h)):  # rebounds/steals/blocks depend on the opponent
                miss_o = boxes[o]["fga"] - boxes[o]["fgm"]
                miss_t = boxes[t]["fga"] - boxes[t]["fgm"]
                boxes[t]["oreb"] = int(r.binomial(miss_t, 0.24))
                boxes[t]["dreb"] = int(r.binomial(miss_o, 0.74))
                boxes[t]["reb"] = boxes[t]["oreb"] + boxes[t]["dreb"]
                boxes[t]["ast"] = int(r.binomial(boxes[t]["fgm"], 0.62))
                boxes[t]["stl"] = int(r.binomial(boxes[o]["tov"], 0.52))
                boxes[t]["blk"] = int(r.binomial(boxes[o]["fga"] - boxes[o]["fg3a"], 0.085))
            for t, o in ((h, a), (a, h)):
                b = boxes[t]
                team_rows.append(dict(season=season, season_type="Regular Season", game_id=game_id,
                                      game_date=pd.Timestamp(d), team_id=t, team=TEAMS[t][0], opp_id=o,
                                      opp=TEAMS[o][0], is_home=int(t == h), min=minutes, **b))
                player_rows.extend(_player_boxes(r, states[t], b, minutes, season, game_id, d, t, o, int(t == h)))
            # historical closing lines for the backtest
            line_rows.append(_market_line(r, d, h, a, poss * ortg[h] / 100, poss * ortg[a] / 100))
            for t in (h, a):
                last_played[t] = d
            last_city[h] = h
            last_city[a] = h

    players = pd.DataFrame([dict(player_id=p.pid, player_name=p.name, position=p.pos) for p in lg.players])
    return (pd.DataFrame(team_rows), pd.DataFrame(player_rows), players,
            pd.DataFrame(sched_rows), pd.DataFrame(line_rows), pd.DataFrame(prop_rows),
            pd.DataFrame(inj_rows))


def _travel(prev_city: int, city: int) -> float:
    from src.data.teams import arena_distance
    return arena_distance(prev_city, city)


def _player_boxes(r, state, team_box, team_minutes, season, game_id, d, t, o, is_home):
    """Split the team box score across active players (sums are preserved)."""
    active, mins, _ = state
    mins = mins * team_minutes / config.TEAM_MINUTES
    usg = np.array([p.usage for p in active])
    three = np.array([p.rates["three"] for p in active])
    w_shot = usg * mins
    cols = {
        "fg3m": _allocate(r, team_box["fg3m"], w_shot * three),
        "fg2m": _allocate(r, team_box["fgm"] - team_box["fg3m"], w_shot * (1 - three)),
        "ftm": _allocate(r, team_box["ftm"], w_shot),
        "reb": _allocate(r, team_box["reb"], np.array([p.rates["reb"] for p in active]) * mins),
        "ast": _allocate(r, team_box["ast"], np.array([p.rates["ast"] for p in active]) * mins),
        "stl": _allocate(r, team_box["stl"], np.array([p.rates["stl"] for p in active]) * mins),
        "blk": _allocate(r, team_box["blk"], np.array([p.rates["blk"] for p in active]) * mins),
        "tov": _allocate(r, team_box["tov"], w_shot),
        "pf": _allocate(r, max(team_box["pf"], 0), mins),
    }
    # attempts: makes + allocated misses so attempts >= makes always holds
    fg3a = cols["fg3m"] + _allocate(r, team_box["fg3a"] - team_box["fg3m"], w_shot * three)
    fg2a = cols["fg2m"] + _allocate(r, (team_box["fga"] - team_box["fg3a"]) - (team_box["fgm"] - team_box["fg3m"]),
                                    w_shot * (1 - three))
    fta = cols["ftm"] + _allocate(r, team_box["fta"] - team_box["ftm"], w_shot)
    oreb = np.minimum(cols["reb"], _allocate(r, team_box["oreb"], np.array([p.rates["reb"] for p in active]) * mins))
    rows = []
    for i, p in enumerate(active):
        fgm = int(cols["fg2m"][i] + cols["fg3m"][i])
        rows.append(dict(
            season=season, season_type="Regular Season", game_id=game_id, game_date=pd.Timestamp(d),
            player_id=p.pid, player_name=p.name, team_id=t, opp_id=o, is_home=is_home,
            min=round(float(mins[i]), 1), pts=int(2 * cols["fg2m"][i] + 3 * cols["fg3m"][i] + cols["ftm"][i]),
            reb=int(cols["reb"][i]), oreb=int(oreb[i]), dreb=int(cols["reb"][i] - oreb[i]),
            ast=int(cols["ast"][i]), stl=int(cols["stl"][i]), blk=int(cols["blk"][i]), tov=int(cols["tov"][i]),
            fgm=fgm, fga=int(fg2a[i] + fg3a[i]), fg3m=int(cols["fg3m"][i]), fg3a=int(fg3a[i]),
            ftm=int(cols["ftm"][i]), fta=int(fta[i]), pf=int(cols["pf"][i]),
        ))
    return rows


def _market_line(r, d, h, a, exp_h, exp_a) -> dict:
    """Sharp-but-imperfect market: true expectation + noise, rounded to 0.5."""
    spread = -(exp_h - exp_a) + r.normal(0, 1.6)
    total = exp_h + exp_a + r.normal(0, 3.0)
    return dict(game_date=pd.Timestamp(d), home=TEAMS[h][0], away=TEAMS[a][0],
                spread=round(spread * 2) / 2, total=round(total * 2) / 2)


def _market_props(r, d, states, ortg, poss) -> list[dict]:
    """Approximate prop lines for upcoming games (expected stat + noise -> x.5)."""
    rows = []
    for t, (active, mins, _) in states.items():
        team_pts = poss * ortg[t] / 100
        usg = np.array([p.usage for p in active])
        share = usg * mins / (usg * mins).sum()
        for i, p in enumerate(active):
            if mins[i] < 15:
                continue
            exp = {"pts": team_pts * share[i], "reb": p.rates["reb"] * mins[i] / 36,
                   "ast": p.rates["ast"] * mins[i] / 36, "fg3m": team_pts * 0.33 / 3 * share[i] * p.rates["three"] / 0.4}
            exp["pra"] = exp["pts"] + exp["reb"] + exp["ast"]
            for stat, v in exp.items():
                rows.append(dict(game_date=pd.Timestamp(d), player_name=p.name, player_id=p.pid, stat=stat,
                                 line=max(0.5, np.floor(v * r.normal(1, 0.08)) + 0.5)))
    return rows
