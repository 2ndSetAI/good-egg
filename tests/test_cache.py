"""Tests for the cache module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from good_egg.cache import Cache


@pytest.fixture
def cache_db(tmp_path: Path) -> Path:
    return tmp_path / "test_cache.db"


@pytest.fixture
def cache(cache_db: Path) -> Cache:
    c = Cache(db_path=cache_db, ttls={"test_category": 3600, "short_ttl": 1})
    yield c
    c.close()


class TestCache:
    def test_set_and_get(self, cache: Cache) -> None:
        cache.set("key1", {"data": "value"}, "test_category")
        result = cache.get("key1")
        assert result == {"data": "value"}

    def test_get_missing_key(self, cache: Cache) -> None:
        assert cache.get("nonexistent") is None

    def test_get_expired_key(self, cache: Cache) -> None:
        cache.set("key1", "value", "short_ttl")
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_set_overwrites(self, cache: Cache) -> None:
        cache.set("key1", "old", "test_category")
        cache.set("key1", "new", "test_category")
        assert cache.get("key1") == "new"

    def test_invalidate(self, cache: Cache) -> None:
        cache.set("key1", "value", "test_category")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_invalidate_category(self, cache: Cache) -> None:
        cache.set("key1", "v1", "test_category")
        cache.set("key2", "v2", "test_category")
        cache.set("key3", "v3", "short_ttl")
        cache.invalidate_category("test_category")
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "v3"

    def test_cleanup_expired(self, cache: Cache) -> None:
        cache.set("key1", "value", "short_ttl")
        time.sleep(1.1)
        removed = cache.cleanup_expired()
        assert removed == 1

    def test_get_or_fetch_cached(self, cache: Cache) -> None:
        cache.set("key1", "cached_value", "test_category")
        result = cache.get_or_fetch("key1", "test_category", lambda: "fresh_value")
        assert result == "cached_value"

    def test_get_or_fetch_fresh(self, cache: Cache) -> None:
        result = cache.get_or_fetch("key1", "test_category", lambda: "fresh_value")
        assert result == "fresh_value"
        assert cache.get("key1") == "fresh_value"

    def test_stats(self, cache: Cache) -> None:
        cache.set("key1", "v1", "test_category")
        cache.set("key2", "v2", "test_category")
        cache.set("key3", "v3", "short_ttl")
        stats = cache.stats()
        assert stats["total_entries"] == 3
        assert stats["categories"]["test_category"] == 2
        assert stats["categories"]["short_ttl"] == 1
        assert stats["db_size_bytes"] > 0

    def test_json_serializable_values(self, cache: Cache) -> None:
        cache.set("list", [1, 2, 3], "test_category")
        cache.set("nested", {"a": {"b": [1, 2]}}, "test_category")
        cache.set("string", "hello", "test_category")
        cache.set("number", 42, "test_category")
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("nested") == {"a": {"b": [1, 2]}}
        assert cache.get("string") == "hello"
        assert cache.get("number") == 42

    def test_default_ttl_for_unknown_category(self, cache: Cache) -> None:
        cache.set("key1", "value", "unknown_category")
        assert cache.get("key1") == "value"

    def test_wal_mode_enabled(self, cache_db: Path) -> None:
        c = Cache(db_path=cache_db)
        mode = c._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        c.close()
