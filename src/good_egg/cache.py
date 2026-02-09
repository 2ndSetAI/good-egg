"""SQLite-backed cache with category-based TTLs."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_DB_PATH = Path.home() / ".cache" / "good-egg" / "cache.db"

# Default TTLs in seconds per category
DEFAULT_TTLS: dict[str, int] = {
    "repo_metadata": 7 * 24 * 3600,    # 7 days
    "user_profile": 1 * 24 * 3600,     # 1 day
    "user_prs": 14 * 24 * 3600,        # 14 days
}


class Cache:
    """SQLite-backed cache with WAL mode and category-based TTLs."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, ttls: dict[str, int] | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttls = {**DEFAULT_TTLS, **(ttls or {})}
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_category ON cache(category)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)"
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        """Get a cached value by key. Returns None if expired or missing."""
        row = self._conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if time.time() > expires_at:
            self.invalidate(key)
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, category: str) -> None:
        """Set a cached value with category-based TTL."""
        ttl = self.ttls.get(category, 3600)  # default 1 hour
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO cache (key, value, category, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            (key, json.dumps(value), category, now, now + ttl),
        )
        self._conn.commit()

    def get_or_fetch(self, key: str, category: str, fetch_fn: Callable[[], T]) -> T:
        """Get from cache or fetch and cache the result."""
        cached = self.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        result = fetch_fn()
        self.set(key, result, category)
        return result

    def invalidate(self, key: str) -> None:
        """Remove a specific cache entry."""
        self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        self._conn.commit()

    def invalidate_category(self, category: str) -> None:
        """Remove all cache entries for a category."""
        self._conn.execute("DELETE FROM cache WHERE category = ?", (category,))
        self._conn.commit()

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        cursor = self._conn.execute(
            "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
        )
        self._conn.commit()
        return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        expired = self._conn.execute(
            "SELECT COUNT(*) FROM cache WHERE expires_at < ?", (time.time(),)
        ).fetchone()[0]
        categories = self._conn.execute(
            "SELECT category, COUNT(*) FROM cache GROUP BY category"
        ).fetchall()
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "categories": {cat: count for cat, count in categories},
            "db_size_bytes": db_size,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
