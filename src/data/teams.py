"""
Static NBA team metadata: official stats.nba.com team ids, abbreviations and
approximate arena coordinates (used for travel-distance features).
"""
from __future__ import annotations

import math

# team_id: (abbreviation, full name, latitude, longitude)
TEAMS: dict[int, tuple[str, str, float, float]] = {
    1610612737: ("ATL", "Atlanta Hawks", 33.757, -84.396),
    1610612738: ("BOS", "Boston Celtics", 42.366, -71.062),
    1610612751: ("BKN", "Brooklyn Nets", 40.683, -73.975),
    1610612766: ("CHA", "Charlotte Hornets", 35.225, -80.839),
    1610612741: ("CHI", "Chicago Bulls", 41.881, -87.674),
    1610612739: ("CLE", "Cleveland Cavaliers", 41.496, -81.688),
    1610612742: ("DAL", "Dallas Mavericks", 32.790, -96.810),
    1610612743: ("DEN", "Denver Nuggets", 39.749, -105.008),
    1610612765: ("DET", "Detroit Pistons", 42.341, -83.055),
    1610612744: ("GSW", "Golden State Warriors", 37.768, -122.388),
    1610612745: ("HOU", "Houston Rockets", 29.751, -95.362),
    1610612754: ("IND", "Indiana Pacers", 39.764, -86.155),
    1610612746: ("LAC", "LA Clippers", 33.945, -118.343),
    1610612747: ("LAL", "Los Angeles Lakers", 34.043, -118.267),
    1610612763: ("MEM", "Memphis Grizzlies", 35.138, -90.051),
    1610612748: ("MIA", "Miami Heat", 25.781, -80.188),
    1610612749: ("MIL", "Milwaukee Bucks", 43.045, -87.917),
    1610612750: ("MIN", "Minnesota Timberwolves", 44.979, -93.276),
    1610612740: ("NOP", "New Orleans Pelicans", 29.949, -90.082),
    1610612752: ("NYK", "New York Knicks", 40.751, -73.993),
    1610612760: ("OKC", "Oklahoma City Thunder", 35.463, -97.515),
    1610612753: ("ORL", "Orlando Magic", 28.539, -81.384),
    1610612755: ("PHI", "Philadelphia 76ers", 39.901, -75.172),
    1610612756: ("PHX", "Phoenix Suns", 33.446, -112.071),
    1610612757: ("POR", "Portland Trail Blazers", 45.532, -122.667),
    1610612758: ("SAC", "Sacramento Kings", 38.580, -121.500),
    1610612759: ("SAS", "San Antonio Spurs", 29.427, -98.438),
    1610612761: ("TOR", "Toronto Raptors", 43.643, -79.379),
    1610612762: ("UTA", "Utah Jazz", 40.768, -111.901),
    1610612764: ("WAS", "Washington Wizards", 38.898, -77.021),
}

ABBR_TO_ID = {v[0]: k for k, v in TEAMS.items()}
NAME_TO_ID = {v[1]: k for k, v in TEAMS.items()}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in miles."""
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def arena_distance(team_a: int, team_b: int) -> float:
    """Miles between two teams' home arenas (0 if unknown)."""
    if team_a not in TEAMS or team_b not in TEAMS:
        return 0.0
    _, _, la, oa = TEAMS[team_a]
    _, _, lb, ob = TEAMS[team_b]
    return haversine_miles(la, oa, lb, ob)
