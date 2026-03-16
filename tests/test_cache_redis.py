"""
Tests for Redis cache backend.

Uses mocking to simulate Redis server interactions.
"""

import pickle
from unittest.mock import MagicMock, patch

import pytest

from router.cache_redis import RedisCache


@pytest.fixture
def redis_cache():
    """Create a RedisCache instance with mocked client."""
    with patch("router.cache_redis.redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_from_url.return_value = mock_client

        cache = RedisCache(
            default_ttl=60.0,
            max_size=1000,
            redis_url="redis://localhost:6379/0",
            max_connections=20,
            key_prefix="test:",
        )
        # Ensure connection is established
        assert cache._ensure_connection() is mock_client
        yield cache, mock_client
        # Cleanup
        cache.close()


def test_redis_cache_set_and_get(redis_cache):
    cache, mock_client = redis_cache

    # Mock get to return pickled data
    test_value = {"key": "value", "number": 42}
    pickled = pickle.dumps(test_value, protocol=pickle.HIGHEST_PROTOCOL)
    mock_client.get.return_value = pickled

    # Set should call set with key and value
    cache.set("mykey", test_value, ttl=120)

    # Verify set called
    mock_client.set.assert_called_once()
    args = mock_client.set.call_args
    assert args[0][0].startswith("test:mykey")  # prefixed key
    assert args[0][1] == pickle.dumps(test_value, protocol=pickle.HIGHEST_PROTOCOL)
    assert args[1]["ex"] == 120

    # Get should return the value
    result = cache.get("mykey")
    assert result == test_value
    mock_client.get.assert_called_with("test:mykey")


def test_redis_cache_get_miss(redis_cache):
    cache, mock_client = redis_cache
    mock_client.get.return_value = None

    result = cache.get("nonexistent")
    assert result is None
    mock_client.get.assert_called_with("test:nonexistent")


def test_redis_cache_delete(redis_cache):
    cache, mock_client = redis_cache

    cache.delete("mykey")
    mock_client.delete.assert_called_once_with("test:mykey")


def test_redis_cache_has(redis_cache):
    cache, mock_client = redis_cache

    mock_client.exists.return_value = 1
    assert cache.has("existing") is True
    mock_client.exists.assert_called_with("test:existing")

    mock_client.exists.return_value = 0
    assert cache.has("missing") is False


def test_redis_cache_clear(redis_cache):
    cache, mock_client = redis_cache

    # Mock scan to return some keys
    mock_client.scan.side_effect = [(0, [b"test:key1", b"test:key2"])]

    cache.clear()

    mock_client.scan.assert_called()
    mock_client.delete.assert_called_once_with(b"test:key1", b"test:key2")


def test_redis_cache_size(redis_cache):
    cache, mock_client = redis_cache

    mock_client.scan.side_effect = [(0, [b"test:a", b"test:b"])]
    size = cache.size()
    assert size == 2


def test_redis_cache_keys(redis_cache):
    cache, mock_client = redis_cache

    mock_client.scan.side_effect = [(0, [b"test:key1", b"test:key2"])]
    keys = cache.keys()
    assert keys == ["key1", "key2"]


def test_redis_cache_invalidate(redis_cache):
    cache, mock_client = redis_cache

    # Test specific key
    cache.invalidate("mykey")
    mock_client.delete.assert_called_with("test:mykey")

    # Reset mock
    mock_client.delete.reset_mock()

    # Test clear all
    cache.invalidate()
    mock_client.scan.assert_called()  # clear() uses scan


def test_redis_cache_close():
    """Test that close closes the client."""
    with patch("router.cache_redis.redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_from_url.return_value = mock_client

        cache = RedisCache()
        client = cache._ensure_connection()
        assert client is mock_client

        cache.close()
        mock_client.close.assert_called_once()


def test_redis_cache_connection_error():
    """Test behavior when Redis connection fails."""
    with patch("router.cache_redis.redis.from_url") as mock_from_url:
        mock_from_url.side_effect = Exception("Connection refused")

        cache = RedisCache()
        client = cache._ensure_connection()
        assert client is None

        # Operations should gracefully handle None
        assert cache.get("key") is None
        cache.set("key", "value")  # Should not raise
        cache.delete("key")  # Should not raise
        assert cache.has("key") is False
        assert cache.size() == 0
        assert cache.keys() == []
