"""
Unified cache manager for SmarterRouter.

Provides thread-safe caching with TTL, eviction, and consistent invalidation.
"""

import threading
import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class Cache(Generic[T]):
    """Thread-safe cache with TTL and LRU eviction."""

    def __init__(self, default_ttl: float = 60.0, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of entries before LRU eviction
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._data: dict[str, tuple[T, float]] = {}  # key -> (value, expiration_time)
        self._access_order: OrderedDict[str, float] = (
            OrderedDict()
        )  # key -> last_access_time for LRU
        self._lock = threading.RLock()

    def get(self, key: str) -> T | None:
        """Get value from cache if present and not expired."""
        with self._lock:
            if key not in self._data:
                return None

            value, expire_time = self._data[key]
            now = time.monotonic()

            if now > expire_time:
                # Expired
                self._delete_key(key)
                return None

            # Update access order for LRU
            self._access_order.move_to_end(key)
            return value

    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """Set value in cache with optional TTL (uses default if not specified)."""
        with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            expire_time = time.monotonic() + ttl
            self._data[key] = (value, expire_time)
            self._access_order[key] = expire_time
            self._access_order.move_to_end(key)

            # Evict if exceeds max size (remove oldest)
            if len(self._data) > self.max_size:
                oldest_key = next(iter(self._access_order))
                self._delete_key(oldest_key)

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        with self._lock:
            self._delete_key(key)

    def _delete_key(self, key: str) -> None:
        """Internal method to delete key from all structures."""
        self._data.pop(key, None)
        self._access_order.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._data.clear()
            self._access_order.clear()

    def invalidate(self, key: str | None = None) -> None:
        """
        Invalidate cache entry.

        If key is None, clear entire cache.
        """
        if key is None:
            self.clear()
        else:
            self.delete(key)

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._data:
                return False

            _, expire_time = self._data[key]
            return time.monotonic() <= expire_time

    def size(self) -> int:
        """Return number of non-expired entries."""
        with self._lock:
            # Clean expired entries first
            now = time.monotonic()
            expired = [k for k, (_, exp) in self._data.items() if now > exp]
            for k in expired:
                self._delete_key(k)
            return len(self._data)

    def keys(self) -> list[str]:
        """Return list of non-expired keys."""
        with self._lock:
            now = time.monotonic()
            return [k for k, (_, exp) in self._data.items() if now <= exp]


class CacheManager:
    """Manager for multiple named caches."""

    def __init__(self):
        self._caches: dict[str, Cache] = {}
        self._lock = threading.RLock()

    def get_cache(self, name: str, default_ttl: float = 60.0, max_size: int = 1000) -> Cache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = Cache(default_ttl=default_ttl, max_size=max_size)
            return self._caches[name]

    def invalidate_all(self) -> None:
        """Invalidate all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache(name: str, default_ttl: float = 60.0, max_size: int = 1000) -> Cache:
    """Get a named cache from the global cache manager."""
    return _cache_manager.get_cache(name, default_ttl, max_size)


def invalidate_all_caches() -> None:
    """Invalidate all caches in the global cache manager."""
    _cache_manager.invalidate_all()
