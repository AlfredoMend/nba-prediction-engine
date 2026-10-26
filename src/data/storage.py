"""
DuckDB-backed local cache.

All pipeline stages read and write pandas DataFrames through this thin wrapper
so the storage engine can be swapped (e.g. for SQLite) in one place.

Tables
------
raw_team_games    one row per team per completed game (nba_api LeagueGameLog schema)
raw_player_games  one row per player per completed game
players           player_id -> position lookup
schedule          upcoming games (no scores yet)
lines             game betting lines (home spread, total) - optional
prop_lines        player prop lines - optional
injuries          current injury report (player_id, status)
pred_games        latest game predictions
pred_props        latest player prop predictions
bt_games / bt_props  walk-forward backtest predictions
meta              key/value metadata (data source, last update, ...)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from src import config


class Storage:
    """Tiny DataFrame <-> DuckDB helper. Opens a short-lived connection per call
    so the Streamlit app and the pipeline never hold competing locks for long."""

    def __init__(self, path: Path | str | None = None):
        self.path = str(path or config.DB_PATH)

    @contextmanager
    def _conn(self, read_only: bool = False, retries: int = 10):
        # DuckDB allows one writer process; retry briefly if the pipeline holds the lock
        for attempt in range(retries):
            try:
                con = duckdb.connect(self.path, read_only=read_only and Path(self.path).exists())
                break
            except duckdb.IOException:
                if attempt == retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        try:
            yield con
        finally:
            con.close()

    # ------------------------------------------------------------------ #
    def write(self, table: str, df: pd.DataFrame, mode: str = "replace") -> None:
        """Write a DataFrame. mode='replace' overwrites, 'append' inserts rows."""
        with self._conn() as con:
            con.register("_df", df)
            if mode == "replace" or not self._exists(con, table):
                con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _df')
            else:
                existing = {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()}
                if set(df.columns) <= existing:
                    con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM _df')
                else:  # schema grew (e.g. new market columns): union old and new rows
                    con.execute(f'CREATE OR REPLACE TABLE "{table}" AS '
                                f'SELECT * FROM "{table}" UNION ALL BY NAME SELECT * FROM _df')
            con.unregister("_df")

    def read(self, table: str) -> pd.DataFrame:
        """Read a whole table; returns an empty DataFrame if it does not exist."""
        if not Path(self.path).exists():
            return pd.DataFrame()
        with self._conn(read_only=True) as con:
            if not self._exists(con, table):
                return pd.DataFrame()
            return con.execute(f'SELECT * FROM "{table}"').df()

    def query(self, sql: str) -> pd.DataFrame:
        with self._conn(read_only=True) as con:
            return con.execute(sql).df()

    def exists(self, table: str) -> bool:
        if not Path(self.path).exists():
            return False
        with self._conn(read_only=True) as con:
            return self._exists(con, table)

    # ------------------------------------------------------------------ #
    def set_meta(self, **kv) -> None:
        meta = self.get_meta()
        meta.update({k: str(v) for k, v in kv.items()})
        self.write("meta", pd.DataFrame({"key": list(meta), "value": list(meta.values())}))

    def get_meta(self) -> dict:
        df = self.read("meta")
        return dict(zip(df["key"], df["value"])) if not df.empty else {}

    @staticmethod
    def _exists(con, table: str) -> bool:
        return bool(
            con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
            ).fetchone()[0]
        )
