# Deployment guide

The app is a Streamlit server that reads a local DuckDB file (`data/nba.duckdb`) and
saved models (`models_store/`). Those are produced by `python main.py --update-and-predict`.
Any host that can run a long-lived Python process works.

> **Data source.** The default source downloads parquet files from GitHub releases
> (sportsdataverse), so the daily refresh works on GitHub-hosted runners and any cloud host.
> The optional `nba_api` source talks to stats.nba.com, which blocks many cloud IP ranges.

## 1. Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py --update-and-predict --backtest     # first run downloads 2020-21 -> today (~40 MB)
streamlit run app.py                              # http://localhost:8501
```

Offline demo (no network needed): `python main.py --source synthetic --date 2026-10-26 --update-and-predict --backtest`.

## 2. Docker

```bash
docker build -t nba-predictor .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models_store:/app/models_store" nba-predictor \
       python main.py --update-and-predict
docker run --rm -p 8501:8501 -v "$PWD/data:/app/data" -v "$PWD/models_store:/app/models_store" nba-predictor
```

## 3. Streamlit Community Cloud

1. Push the repo (including `data/nba.duckdb`, `models_store/`, `backtest_2025_2026_results.json`) to GitHub.
2. On share.streamlit.io choose **New app**, select the repo, main file `app.py`, Python 3.11.
3. Add `ODDS_API_KEY` under **Secrets** if you use live odds (only needed by the pipeline, not the UI).
4. The daily GitHub Action commits refreshed data; Streamlit Cloud redeploys on each push.

## 4. Render

1. **New → Web Service**, connect the repo, runtime **Docker** (uses the `Dockerfile`; `$PORT` is honoured).
2. Attach a **Persistent Disk** mounted at `/app/data` (1 GB is plenty) so the DuckDB cache survives deploys.
3. Optional **Cron Job** service with the same image and command
   `python main.py --update-and-predict` on schedule `17 14 * * *`, sharing the same disk.

## 5. Vercel

Vercel runs serverless functions and static sites; it cannot host a Streamlit server (Streamlit
needs a persistent process and WebSockets). Options if you are tied to Vercel:
- host the UI on Streamlit Cloud/Render and point a Vercel domain at it with a rewrite/redirect, or
- export predictions to JSON (`pred_games`, `pred_props` tables) and build a static Vercel frontend
  that reads them.

## 6. GitHub Actions daily refresh

`.github/workflows/daily_update.yml` runs tests, then `main.py --update-and-predict`, and commits
the refreshed DuckDB file, models and report back to the repo. Trigger it manually with
**Run workflow** (tick *backtest* to re-run the 2025-26 backtest).

- Add the `ODDS_API_KEY` repository secret for live lines.
- Raw season files are cached between runs; only the current season is re-downloaded.

## Betting lines and injuries

Vegas lines (ESPN scoreboard odds), Kalshi prices and injuries are fetched automatically on each run.
The hosts are `site.api.espn.com` and `api.elections.kalshi.com`, plus `api.the-odds-api.com` if
`ODDS_API_KEY` is set. The CSVs below add historical lines or overrides (templates in `data/templates/`):

| File | Columns | Used for |
|---|---|---|
| `lines.csv` | game_date, home, away, spread, total[, home_ml, away_ml] | historical closing lines (backtest ATS/O-U/ROI) and manual lines |
| `prop_lines.csv` | game_date, player_name, stat, line | prop edges and over/under probabilities |
| `injuries.csv` | player_name, team, status | players marked OUT/DOUBTFUL are removed from projected lineups |

Spreads are from the home team's perspective (`-5.5` = home favoured by 5.5).
