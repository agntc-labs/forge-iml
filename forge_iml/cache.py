"""
IML Cache -- Three-layer cache for pipeline results.

Layer 1: Exact match (in-memory OrderedDict, <1ms)
Layer 2: Semantic match (via MemoryProvider, <10ms)
Layer 3: Full pipeline (miss -- caller runs parse/plan/execute)
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

from forge_iml.providers.base import MemoryProvider

logger = logging.getLogger("forge_iml.cache")

# -- TTL Map ------------------------------------------------------------------

TTL_MAP = {
    "weather": 300,
    "balance": 900,
    "tool_output": 3600,
    "contacts": 86400,
    "facts": 0,
    "default": 600,
}

MAX_ENTRIES = 1000
CLEANUP_INTERVAL = 60


# -- Cache Key Builder --------------------------------------------------------

def build_cache_key(verb: str, obj: str, constraints: str = "",
                    user_handle: str = "") -> str:
    """Build a deterministic cache key from canonicalized fields."""
    raw = f"{verb}|{obj}|{constraints}|{user_handle}"
    return hashlib.sha256(raw.encode()).hexdigest()


# -- IMLCache -----------------------------------------------------------------

class IMLCache:
    """Three-layer cache: exact -> semantic -> miss.

    Args:
        memory_provider: Optional MemoryProvider for Layer 2 semantic lookups.
            If not provided, only Layer 1 (exact match) is used.
    """

    def __init__(self, memory_provider: Optional[MemoryProvider] = None):
        self._lock = threading.Lock()
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._last_cleanup = time.time()
        self._memory = memory_provider

        self._hits_exact = 0
        self._hits_semantic = 0
        self._misses = 0

    # -- Public API -----------------------------------------------------------

    def get(self, cache_key: str, semantic_query: str = "") -> Optional[dict]:
        """Check all layers. Returns cached result dict or None."""
        self._maybe_cleanup()

        # Layer 1: exact match
        with self._lock:
            if cache_key in self._store:
                entry = self._store[cache_key]
                ttl = entry["ttl"]
                if ttl == 0 or (time.time() - entry["timestamp"]) < ttl:
                    entry["hits"] += 1
                    self._store.move_to_end(cache_key)
                    self._hits_exact += 1
                    logger.debug("cache HIT exact: %s", cache_key[:16])
                    return entry["result"]
                else:
                    del self._store[cache_key]

        # Layer 2: semantic match via memory provider
        if semantic_query and self._memory is not None:
            result = self._semantic_lookup(semantic_query)
            if result is not None:
                self._hits_semantic += 1
                logger.debug("cache HIT semantic: %s", cache_key[:16])
                return result

        # Layer 3: miss
        self._misses += 1
        logger.debug("cache MISS: %s", cache_key[:16])
        return None

    def put(self, cache_key: str, result: dict,
            ttl_category: str = "default") -> None:
        """Store result in Layer 1 (exact cache)."""
        ttl = TTL_MAP.get(ttl_category, TTL_MAP["default"])
        with self._lock:
            while len(self._store) >= MAX_ENTRIES:
                self._store.popitem(last=False)
            self._store[cache_key] = {
                "result": result,
                "timestamp": time.time(),
                "ttl": ttl,
                "hits": 0,
            }
        logger.debug("cache PUT: %s (ttl=%s/%ds)", cache_key[:16],
                      ttl_category, ttl)

    def invalidate(self, cache_key: str) -> bool:
        """Remove a specific key from Layer 1. Returns True if found."""
        with self._lock:
            if cache_key in self._store:
                del self._store[cache_key]
                return True
        return False

    def stats(self) -> dict:
        """Hit/miss rates per layer."""
        total = self._hits_exact + self._hits_semantic + self._misses
        with self._lock:
            size = len(self._store)
        return {
            "layer1_exact_hits": self._hits_exact,
            "layer2_semantic_hits": self._hits_semantic,
            "layer3_misses": self._misses,
            "total_lookups": total,
            "hit_rate": (self._hits_exact + self._hits_semantic) / total
                        if total > 0 else 0.0,
            "exact_hit_rate": self._hits_exact / total if total > 0 else 0.0,
            "layer1_size": size,
            "layer1_max": MAX_ENTRIES,
        }

    # -- Internal -------------------------------------------------------------

    def _semantic_lookup(self, query: str) -> Optional[dict]:
        """Layer 2: search memory provider for a similar prior result."""
        try:
            results = self._memory.search(query, limit=1)
            if not results:
                return None
            top = results[0]
            content = top.get("content", "")
            if "iml_audit" in top.get("type", "") or "IML" in content:
                return {"semantic_match": True, "content": content,
                        "memory_id": top.get("id")}
        except Exception as e:
            logger.debug("semantic lookup failed: %s", e)
        return None

    def _maybe_cleanup(self) -> None:
        """Evict expired entries, but only every CLEANUP_INTERVAL."""
        now = time.time()
        if now - self._last_cleanup < CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        with self._lock:
            expired = [
                k for k, v in self._store.items()
                if v["ttl"] > 0 and (now - v["timestamp"]) >= v["ttl"]
            ]
            for k in expired:
                del self._store[k]
            if expired:
                logger.debug("cache cleanup: evicted %d expired entries",
                             len(expired))
