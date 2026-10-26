# NBA Game & Player Prop Prediction Engine

End-to-end system for the 2026-27 NBA season: daily data from the GitHub-hosted
[sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) NBA release files
(ESPN box scores and schedules, refreshed every day), leakage-safe feature
engineering, a LightGBM/XGBoost/CatBoost score ensemble with 10,000-run Monte Carlo simulation,
LightGBM quantile models for player props, a strict walk-forward backtest on 2025-26, and a dark
Streamlit UI.

```
main.py                     CLI: --update-and-predict, --backtest
app.py                      Streamlit UI (4 views)
src/config.py               paths, seasons, feature windows, betting conventions
src/pipeline.py             ingest -> features -> train -> predict -> persist
src/data/
  sportsdataverse.py        default source: ESPN team/player box scores + schedule from GitHub releases
  injuries.py               ESPN injury feed (best effort) + data/injuries.csv overrides
  nba_fetch.py              alternative source: stats.nba.com (LeagueGameLog, PlayerIndex, ScoreboardV3/V2)
  synthetic.py              offline simulated league with the same schema (demo/CI only)
  odds.py                   lines from CSV or The Odds API
  storage.py                DuckDB cache
  features.py               team ratings + EWMA(5/10/20), rest, B2B, travel, venue splits
  lineup.py                 rotation tracking, injury adjustment, minutes/usage redistribution
  player_features.py        player form, usage, TS%, positional defence allowed, matchup context
  build.py                  assembles the feature store
src/models/
  game_model.py             home/away score ensemble + out-of-sample residual calibration
  monte_carlo.py            correlated score simulation -> win/cover/over probabilities
  props_model.py            p10/p50/p90 quantile models for PTS/REB/AST/3PM/STL/BLK/PRA
  simulator.py              lineup what-if engine used by the UI
src/backtest/
  walk_forward.py           chronological walk-forward on 2025-26
  metrics.py                MAE/RMSE, ATS, O/U, ROI, Brier, calibration, interval coverage
  plots.py / run_backtest.py  report -> backtest_2025_2026_results.json + assets/*.png
tests/                      pytest suite (features, leakage, lineup logic, Monte Carlo, backtest maths)
```

## Quick start

```bash
pip install -r requirements.txt
python main.py --update-and-predict --backtest   # real data from GitHub (sportsdataverse)
streamlit run app.py
pytest -q
```

Other sources: `--source nba_api` (stats.nba.com, often blocked from cloud IPs) or the offline
synthetic demo league:

```bash
python main.py --source synthetic --date 2026-10-26 --update-and-predict --backtest
streamlit run app.py      # shows a "Demo mode: synthetic league" banner
```

Synthetic-mode numbers only show that the pipeline works. They say nothing about real NBA accuracy.

## How it works

**Leakage control.** Every rolling feature is an EWMA shifted by one game inside the team's or
player's own history. Opponent positional defence is joined with `merge_asof(allow_exact_matches=False)`.
Upcoming games are appended as placeholder rows so they get features through the same code path.
The walk-forward backtest retrains on `game_date < window_start` only (weekly for games, every 28
days for props) and records `train_cutoff` on every prediction. The report's
`leakage_check_passed` field asserts `train_cutoff < game_date` for every row, and unit tests
check that changing a future result leaves past features unchanged.

**Injuries and lineups.** `lineup.py` rebuilds each team's expected rotation from players who
played for it in the last 21 days (≥ 8 min EWMA). Rotation players missing from a game (or listed
OUT in `injuries.csv`) produce team features: missing minutes, usage-minutes, PRA and top-player
minutes. These drive the game model's rating adjustment. Their minutes and usage are
redistributed to available teammates (proportional to minutes and usage x minutes, capped at 44
min), which produces `proj_min` / `proj_usg` for the props model. The UI simulator calls the same
functions.

**Game engine.** LightGBM, XGBoost and CatBoost regressors (equal-weight average) predict home and
away points separately. The most recent 15% of each training window is held out to estimate the
residual sigmas and the home/away correlation. A bivariate normal Monte Carlo (10,000 runs,
integer scores, simulated overtime) then gives win, cover and over probabilities.

**Props engine.** For each stat, LightGBM quantile models at 0.1/0.5/0.9 produce the projection
(median) and an 80% interval. Over-probabilities against a line come from a two-piece normal
fitted through the three quantiles.

## Limitations

- Data freshness follows sportsdataverse, which usually publishes a night's games within about
  a day. The sidebar shows the latest completed game in the dataset.
- Neither data source includes betting lines. Supply `data/lines.csv`, `data/prop_lines.csv`
  (templates in `data/templates/`), or set `ODDS_API_KEY` for live game lines. Injuries come from
  ESPN's public feed when reachable, plus `data/injuries.csv` overrides.
  Without historical closing lines the backtest reports score and probability metrics but no ATS/ROI.
- Player tracking metrics are not ingested. Box scores are used throughout.
- The `nba_api` source has not been exercised against the live API (stats.nba.com was unreachable
  while this was built). The sportsdataverse source is the tested default.
