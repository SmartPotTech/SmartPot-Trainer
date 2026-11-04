"""Real reading store for training (SQLite on the shared service volume).

The crop ID is stored pseudonymized (SHA-256 hash): the service learns from each crop's time series
without knowing which crop it is or who owns it. If the data directory is not writable, the store
runs in memory and learned state is lost upon restart.
"""

import hashlib
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from app.knowledge.profiles import PROFILES

log = logging.getLogger(__name__)

# Solo los cultivos reales enseñan: los virtuales son datos del simulador.
_REAL_ONLY = "c.kind = 'REAL'"
_TIMEZONE = re.compile(r"^[A-Za-z]+(?:[/_+-][A-Za-z0-9]+)*$")


def _checked_timezone(name: str) -> str:
    """Validates the timezone name before interpolating it into SQL; PostgreSQL verifies its existence."""
    if not _TIMEZONE.match(name):
        raise ValueError(f"Invalid timezone: {name}")
    return name


COLUMNS = ("temperature", "humidity", "brightness", "ph", "tds", "atmosphere", "soil_moisture")
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS readings (
    crop_key TEXT NOT NULL,
    crop_type TEXT NOT NULL,
    measured_at REAL NOT NULL,
    local_hour INTEGER,
    {", ".join(f"{column} REAL" for column in COLUMNS)},
    received_at REAL NOT NULL,
    PRIMARY KEY (crop_key, measured_at)
);
CREATE INDEX IF NOT EXISTS readings_type_time ON readings (crop_type, measured_at);
"""


def crop_key(crop_id: str) -> str:
    return hashlib.sha256(crop_id.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class StoredReading:
    crop_id: str
    crop_type: str
    measured_at: float
    local_hour: int | None
    values: dict[str, float | None]


@dataclass
class Series:
    """Readings for a crop species sorted by crop and timestamp; missing values are NaN."""

    crops: np.ndarray
    times: np.ndarray
    hours: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def subset(self, mask: np.ndarray) -> "Series":
        return Series(self.crops[mask], self.times[mask], self.hours[mask], self.values[mask])


class ReadingStore:
    def __init__(self, data_dir: str | Path | None, database_url: str | None = None, timezone: str = "America/Bogota"):
        self.path = self._resolve(data_dir)
        self.database_url = database_url
        self.timezone = _checked_timezone(timezone)
        self._source_logged = False
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path or ":memory:", check_same_thread=False)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def _adbc(self, query: str) -> pl.DataFrame | None:
        """Executes a read-only query on TimescaleDB and returns it as a Polars DataFrame (zero-copy Arrow)."""
        try:
            import adbc_driver_postgresql.dbapi as adbc_dbapi

            with adbc_dbapi.connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(query)
                df = pl.from_arrow(cur.fetch_arrow_table())
        except Exception as error:  # noqa: BLE001
            log.warning("TimescaleDB unavailable via ADBC (%s); using local store", error)
            return None
        if not self._source_logged:
            log.info("Training source: TimescaleDB (real crops only)")
            self._source_logged = True
        return df  # type: ignore[return-value]

    @property
    def persistent(self) -> bool:
        return self.path is not None

    @staticmethod
    def _resolve(data_dir: str | Path | None) -> str | None:
        if not data_dir:
            return None
        folder = Path(data_dir)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return str(folder / "learning.db")
        except OSError as error:
            log.warning("Cannot write to %s (%s): store running in memory", folder, error)
            return None

    def add(self, readings: list[StoredReading]) -> int:
        """Stores new readings; duplicates (same crop and timestamp) are ignored."""
        now = time.time()
        rows = [
            (
                crop_key(r.crop_id),
                r.crop_type,
                r.measured_at,
                r.local_hour,
                *(r.values.get(column) for column in COLUMNS),
                now,
            )
            for r in readings
        ]
        placeholders = ", ".join("?" for _ in range(len(COLUMNS) + 5))
        with self._lock:
            before = self._connection.total_changes
            self._connection.executemany(f"INSERT OR IGNORE INTO readings VALUES ({placeholders})", rows)
            self._connection.commit()
            return self._connection.total_changes - before

    def counts(self) -> dict[str, dict[str, float]]:
        if self.database_url:
            df = self._adbc(
                "SELECT c.type AS crop_type, COUNT(*) AS readings, COUNT(DISTINCT r.crop_id) AS crops, "
                "EXTRACT(EPOCH FROM MAX(r.measured_at))::double precision AS last_received "
                f"FROM readings r JOIN crops c ON r.crop_id = c.id WHERE {_REAL_ONLY} GROUP BY c.type"
            )
            if df is not None and not df.is_empty():
                return {
                    row[0]: {"readings": row[1], "crops": row[2], "last_received": row[3]} for row in df.iter_rows()
                }

        with self._lock:
            rows = self._connection.execute(
                "SELECT crop_type, COUNT(*), COUNT(DISTINCT crop_key), MAX(received_at) FROM readings "
                "GROUP BY crop_type"
            ).fetchall()
        return {row[0]: {"readings": row[1], "crops": row[2], "last_received": row[3]} for row in rows}

    def total(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM readings").fetchone()[0])

    def load_df(self, crop_type: str, limit: int) -> pl.DataFrame:
        """Loads the `limit` most recent readings for the species directly into a Polars DataFrame.
        With `database_url`, reads from TimescaleDB via ADBC (real crops only, pseudonymized ID, and local hour);
        otherwise, or on failure, falls back to the local/in-memory SQLite store."""
        if self.database_url and crop_type in PROFILES:
            cols = ", ".join(f"r.{c}" for c in COLUMNS)
            df = self._adbc(
                f"SELECT r.crop_id AS crop_key, EXTRACT(EPOCH FROM r.measured_at)::double precision AS measured_at, "
                f"EXTRACT(HOUR FROM r.measured_at AT TIME ZONE '{self.timezone}')::INTEGER AS local_hour, {cols} "
                f"FROM readings r JOIN crops c ON r.crop_id = c.id "
                f"WHERE c.type = '{crop_type}' AND {_REAL_ONLY} "
                f"ORDER BY r.measured_at DESC LIMIT {int(limit)}"
            )
            if df is not None and not df.is_empty():
                df = df.with_columns(
                    pl.col("crop_key").map_elements(lambda value: crop_key(value.strip()), return_dtype=pl.Utf8)
                )
                return df.sort(["crop_key", "measured_at"])

        columns = ", ".join(COLUMNS)
        query = (
            f"SELECT crop_key, measured_at, local_hour, {columns} FROM "
            f"(SELECT * FROM readings WHERE crop_type = ? ORDER BY measured_at DESC LIMIT ?) "
            f"ORDER BY crop_key, measured_at"
        )
        with self._lock:
            return pl.read_database(query, self._connection, execute_options={"parameters": (crop_type, limit)})

    def load(self, crop_type: str, limit: int) -> Series:
        """Loads the `limit` most recent readings for the species, sorted by crop and timestamp."""
        df = self.load_df(crop_type, limit)
        if df.is_empty():
            empty = np.empty(0)
            return Series(np.empty(0, dtype=object), empty, empty, np.empty((0, len(COLUMNS))))
        crops = df["crop_key"].to_numpy()
        times = df["measured_at"].to_numpy()
        hours = df["local_hour"].cast(pl.Float64).to_numpy()
        values = df.select(COLUMNS).to_numpy()
        return Series(crops, times, hours, values)

    def forget(self, crop_id: str) -> int:
        with self._lock:
            cursor = self._connection.execute("DELETE FROM readings WHERE crop_key = ?", (crop_key(crop_id),))
            self._connection.commit()
            return cursor.rowcount

    def prune(self, retention_days: int, max_rows_per_type: int) -> int:
        """Deletes the oldest readings by age and by per-species row cap."""
        cutoff = time.time() - retention_days * 86_400
        with self._lock:
            removed = self._connection.execute("DELETE FROM readings WHERE measured_at < ?", (cutoff,)).rowcount
            for (crop_type,) in self._connection.execute("SELECT DISTINCT crop_type FROM readings").fetchall():
                removed += self._connection.execute(
                    "DELETE FROM readings WHERE crop_type = ? AND measured_at < ("
                    "SELECT measured_at FROM readings WHERE crop_type = ? ORDER BY measured_at DESC "
                    "LIMIT 1 OFFSET ?)",
                    (crop_type, crop_type, max(0, max_rows_per_type - 1)),
                ).rowcount
            self._connection.commit()
        return removed

    def close(self) -> None:
        with self._lock:
            self._connection.close()
