"""
Redis cache backend for SmarterRouter.

Provides distributed caching using Redis with TTL and LRU-like eviction.
Implements the synchronous Cache interface.
"""

import logging
import pickle
from typing import Any

import redis
from redis.exceptions import RedisError

from router.cache import Cache

logger = logging.getLogger(__name__)


class RedisCache(Cache):
    """Redis-backed cache with TTL and LRU eviction.

    This implementation uses Redis for distributed caching, allowing
    multiple router instances to share cache entries.

    Features:
    - TTL support via Redis EXPIRE
    - LRU eviction via Redis maxmemory-policy (should be set to allkeys-lru)
    - Atomic operations for thread safety
    - Automatic connection pooling
    - Graceful fallback on Redis errors (returns None, doesn't raise)
    """

    def __init__(
        self,
        default_ttl: float = 60.0,
        max_size: int = 1000,
        redis_url: str = "redis://localhost:6379/0",
        max_connections: int = 20,
        key_prefix: str = "smarterrouter:",
    ):
        """
        Initialize Redis cache.

        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum number of entries (informational, actual limit via Redis config)
            redis_url: Redis connection URL
            max_connections: Maximum connections in pool
            key_prefix: Prefix for all Redis keys
        """
        super().__init__(default_ttl, max_size)
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.key_prefix = key_prefix
        self._client: redis.Redis | None = None
        self._connected = False

    def _ensure_connection(self) -> redis.Redis | None:
        """Ensure Redis connection is established."""
        if self._client is not None and self._connected:
            return self._client

        try:
            self._client = redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return self._client
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return None

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Any | None:
        """Get value from Redis cache."""
        client = self._ensure_connection()
        if client is None:
            logger.warning("Redis unavailable, returning None")
            return None

        try:
            redis_key = self._make_key(key)
            data = client.get(redis_key)
            if data is None:
                return None

            # Deserialize
            value = pickle.loads(data)
            return value
        except (RedisError, pickle.PickleError) as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in Redis cache with TTL."""
        client = self._ensure_connection()
        if client is None:
            logger.warning("Redis unavailable, skipping set")
            return

        try:
            redis_key = self._make_key(key)
            ttl = ttl if ttl is not None else self.default_ttl
            ttl_int = int(ttl)

            # Serialize
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # Set with TTL
            client.set(redis_key, data, ex=ttl_int)
        except (RedisError, pickle.PickleError) as e:
            logger.warning(f"Redis set error: {e}")

    def delete(self, key: str) -> None:
        """Delete key from Redis cache."""
        client = self._ensure_connection()
        if client is None:
            return

        try:
            redis_key = self._make_key(key)
            client.delete(redis_key)
        except RedisError as e:
            logger.warning(f"Redis delete error: {e}")

    def clear(self) -> None:
        """Clear all cache entries with our prefix."""
        client = self._ensure_connection()
        if client is None:
            return

        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                if keys:
                    client.delete(*keys)
                if cursor == 0:
                    break
            logger.info(f"Cleared all Redis keys matching pattern: {pattern}")
        except RedisError as e:
            logger.warning(f"Redis clear error: {e}")

    def has(self, key: str) -> bool:
        """Check if key exists in Redis."""
        client = self._ensure_connection()
        if client is None:
            return False

        try:
            redis_key = self._make_key(key)
            return client.exists(redis_key) > 0
        except RedisError as e:
            logger.warning(f"Redis exists error: {e}")
            return False

    def size(self) -> int:
        """Count keys with our prefix."""
        client = self._ensure_connection()
        if client is None:
            return 0

        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0
            count = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except RedisError as e:
            logger.warning(f"Redis size error: {e}")
            return 0

    def keys(self) -> list[str]:
        """List all keys with our prefix (without prefix)."""
        client = self._ensure_connection()
        if client is None:
            return []

        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0
            keys: list[str] = []
            while True:
                cursor, redis_keys = client.scan(cursor, match=pattern, count=100)
                for k in redis_keys:
                    if isinstance(k, bytes):
                        k = k.decode("utf-8")
                    # Strip prefix
                    keys.append(k[len(self.key_prefix) :])
                if cursor == 0:
                    break
            return keys
        except RedisError as e:
            logger.warning(f"Redis keys error: {e}")
            return []

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entry or all entries."""
        if key is None:
            self.clear()
        else:
            self.delete(key)

    def close(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            try:
                self._client.close()
                self._client = None
                self._connected = False
            except RedisError as e:
                logger.warning(f"Redis close error: {e}")
