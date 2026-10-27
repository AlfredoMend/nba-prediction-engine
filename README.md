# NBA Game & Player Prop Prediction Engine

End-to-end system for the 2026-27 NBA season: daily data from the GitHub-hosted
[sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) NBA release files
(ESPN box scores and schedules, refreshed every day), leakage-safe feature
engineering, a LightGBM/XGBoost/CatBoost score ensemble with 10,000-run Monte Carlo simulation,
LightGBM quantile models for player props, a strict walk-forward backtest on 2025-26, and a dark
Streamlit UI.

```
main.py                     CLI: --update-and-predict, --backtest
app.py                      Streamlit UI (6 views)
src/config.py               paths, seasons, feature windows, betting conventions
src/pipeline.py             ingest -> features -> train -> predict -> persist
src/data/
  sportsdataverse.py        default source: ESPN team/player box scores + schedule from GitHub releases
  injuries.py               ESPN injury feed (best effort) + data/injuries.csv overrides
  nba_fetch.py              alternative source: stats.nba.com (LeagueGameLog, PlayerIndex, ScoreboardV3/V2)
  synthetic.py              offline simulated league with the same schema (demo/CI only)
  odds.py                   lines from CSV or The Odds API (consensus spread/total/moneyline)
  prop_odds.py              player prop lines + prices (CSV, or The Odds API for chosen books)
  markets.py                free Vegas lines (ESPN scoreboard odds) + Kalshi prices, no-vig maths, Kalshi EV
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
  prop_edges.py             prop bets: calibrated model probability vs book price, EV, Kelly sizing
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

## Refreshing from the site

The sidebar's **⚡ Quick refresh** re-prices today's games in seconds from the latest injuries,
odds and Kalshi prices using the current models. **🔄 Full rerun** starts the GitHub workflow
(new games + retrain, about 10 minutes). Setup is in [DEPLOYMENT.md](DEPLOYMENT.md).

## Vegas vs Kalshi

Every run fetches two kinds of market line for upcoming games:

- **Vegas**: spread, total and moneylines from ESPN's scoreboard odds (free, no key). If
  `ODDS_API_KEY` is set, The Odds API's consensus across US books takes priority.
- **Kalshi**: public market data from `api.elections.kalshi.com` (no key needed for prices).
  `KXNBAGAME` gives each team's win price; the `KXNBASPREAD` and `KXNBATOTAL` ladders are
  turned into an implied spread and total (the line where the price crosses 50%).

The **Vegas vs Kalshi** view puts model, no-vig Vegas and Kalshi win probabilities side by side,
along with the three spreads and totals. It also shows the expected value of buying the model's
side on Kalshi at the current ask, after Kalshi's taker fee (`ceil(0.07 x P x (1 - P))` per
contract). Each run appends a timestamped snapshot to the `line_history` table, so closing lines
build up for a future backtest against both markets.

## Player props vs a book (e.g. Novig)

The **Prop Bet Finder** view ranks every prop line you have a price for by expected value:

1. The model's P(over) comes from its p10/p50/p90 projection. An isotonic calibrator fitted on
   the 2025-26 walk-forward backtest adjusts it (the raw probabilities were about 2 points too
   high on overs; calibration lowered the out-of-sample Brier score for every stat).
2. The book's price becomes a no-vig probability.
3. EV per $1 = p x profit_if_win - (1 - p). The side with the higher EV is shown, flagged at
   EV >= 3%, with a quarter-Kelly stake as a sizing guide.

Example: if the model's median is 26 for Jayson Tatum and the book has Over 23.5 at -105, the
over is a bet only if the model's calibrated chance of 24+ points is above about 51%, the
break-even rate at -105.

**Getting Novig lines in.** Novig has no public API; its odds are resold by third-party feeds.
Two options:
- Paste lines into `data/prop_lines.csv` (`game_date,player_name,stat,line,over_odds,under_odds,book`;
  stats: pts, reb, ast, fg3m, stl, blk, pra) and commit. The next run picks them up.
- Set the `ODDS_API_KEY` secret and an `ODDS_PROP_BOOKMAKERS` repository variable (e.g.
  `novig,draftkings,fanduel`). The run log lists which books actually returned props, so you can
  see whether your plan includes Novig.

## Limitations

- Data freshness follows sportsdataverse, which usually publishes a night's games within about
  a day. The sidebar shows the latest completed game in the dataset.
- Historical lines are not available yet, so the 2025-26 backtest cannot score ATS/ROI against
  Vegas or Kalshi. `line_history` starts collecting them from the first daily run. Supply
  `data/lines.csv` with past closing lines to backfill. Prop lines need `data/prop_lines.csv`
  (templates in `data/templates/`). Injuries come from
  ESPN's public feed when reachable, plus `data/injuries.csv` overrides.
  Without historical closing lines the backtest reports score and probability metrics but no ATS/ROI.
- Player tracking metrics are not ingested. Box scores are used throughout.
- The `nba_api` source has not been exercised against the live API (stats.nba.com was unreachable
  while this was built). The sportsdataverse source is the tested default.
- The ESPN odds and Kalshi parsers were built from the documented response formats and tested
  against sample payloads. Both fail soft: an unreachable host or changed payload logs a warning
  and leaves those columns blank. Kalshi does not list every game, and markets may open late.
- Prop edges have not been backtested against real historical prop prices. The calibration
  check uses lines placed around the model's own projection, not real book lines.
