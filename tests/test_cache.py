"""Tests for forge_iml.cache -- Layer 1 exact-match cache."""

import time

from forge_iml.cache import IMLCache, build_cache_key


def test_build_cache_key():
    k1 = build_cache_key("get", "weather", user_handle="alice")
    k2 = build_cache_key("get", "weather", user_handle="alice")
    assert k1 == k2
    assert len(k1) == 64


def test_build_cache_key_differs():
    k1 = build_cache_key("get", "weather", user_handle="alice")
    k2 = build_cache_key("get", "balance", user_handle="alice")
    assert k1 != k2


def test_put_and_get():
    cache = IMLCache()
    key = "abc123"
    cache.put(key, {"result": "sunny"}, ttl_category="weather")
    result = cache.get(key)
    assert result == {"result": "sunny"}


def test_miss():
    cache = IMLCache()
    result = cache.get("nonexistent_key")
    assert result is None


def test_invalidate():
    cache = IMLCache()
    key = "to_remove"
    cache.put(key, {"data": 1})
    assert cache.get(key) is not None
    removed = cache.invalidate(key)
    assert removed is True
    assert cache.get(key) is None


def test_stats():
    cache = IMLCache()
    cache.get("miss1")
    cache.put("hit1", {"v": 1})
    cache.get("hit1")

    s = cache.stats()
    assert s["layer1_exact_hits"] == 1
    assert s["layer3_misses"] == 1
    assert s["total_lookups"] == 2
    assert s["hit_rate"] == 0.5


def test_lru_eviction():
    cache = IMLCache()
    # Manually set a small max for testing
    import forge_iml.cache as cache_mod
    old_max = cache_mod.MAX_ENTRIES
    cache_mod.MAX_ENTRIES = 3

    try:
        cache2 = IMLCache()
        cache2.put("k1", {"v": 1})
        cache2.put("k2", {"v": 2})
        cache2.put("k3", {"v": 3})
        cache2.put("k4", {"v": 4})  # should evict k1

        assert cache2.get("k1") is None
        assert cache2.get("k4") == {"v": 4}
    finally:
        cache_mod.MAX_ENTRIES = old_max
