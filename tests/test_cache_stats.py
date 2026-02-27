"""
Tests for enhanced cache statistics and analytics (SmarterRouter 2.1.6+).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from router.cache_stats import CacheAnalytics, TimeSeriesStats
from router.router import SemanticCache


class TestTimeSeriesStats:
    """Tests for TimeSeriesStats class."""

    @pytest.fixture
    def stats(self):
        """Fixture providing TimeSeriesStats instance."""
        return TimeSeriesStats(retention_hours=1)

    @pytest.mark.asyncio
    async def test_record_hit(self, stats):
        """Test recording cache hit events."""
        await stats.record_hit(cache_type="routing", event_type="hit", model="llama3")
        await stats.record_hit(cache_type="response", event_type="miss", model="codellama")
        await stats.record_hit(cache_type="embedding", event_type="similarity_hit", model=None)

        data = await stats.get_stats(window_minutes=60)
        assert data["hits"] == 1
        assert data["misses"] == 1
        assert data["similarity_hits"] == 1
        assert data["total_events"] == 3

    @pytest.mark.asyncio
    async def test_record_eviction(self, stats):
        """Test recording cache eviction events."""
        await stats.record_eviction(cache_type="routing", reason="size", model="llama3")
        await stats.record_eviction(cache_type="response", reason="ttl", model=None)

        data = await stats.get_stats(window_minutes=60)
        assert data["evictions"] == 2
        assert data["eviction_stats"]["size"] == 1
        assert data["eviction_stats"]["ttl"] == 1

    @pytest.mark.asyncio
    async def test_model_stats(self, stats):
        """Test model-specific statistics tracking."""
        # Record hits and misses for llama3
        await stats.record_hit(cache_type="routing", event_type="hit", model="llama3")
        await stats.record_hit(cache_type="routing", event_type="hit", model="llama3")
        await stats.record_hit(cache_type="routing", event_type="miss", model="llama3")
        # Record for another model
        await stats.record_hit(cache_type="routing", event_type="hit", model="codellama")

        data = await stats.get_stats(window_minutes=60)
        model_stats = data["model_stats"]
        assert model_stats["llama3"]["hits"] == 2
        assert model_stats["llama3"]["misses"] == 1
        assert model_stats["codellama"]["hits"] == 1
        assert model_stats["codellama"]["misses"] == 0

    @pytest.mark.asyncio
    async def test_time_window_filtering(self, stats):
        """Test that events outside time window are excluded."""
        # Record an event now
        await stats.record_hit(cache_type="routing", event_type="hit", model="llama3")
        # Simulate old event by directly manipulating events list
        stats.events.append(
            type(
                "CacheEvent",
                (),
                {
                    "timestamp": time.time() - 3600,  # 1 hour ago
                    "cache_type": "routing",
                    "event_type": "miss",
                    "model": "old",
                },
            )
        )

        data = await stats.get_stats(window_minutes=30)  # Only last 30 minutes
        assert data["hits"] == 1
        assert data["misses"] == 0
        assert data["total_events"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, stats):
        """Test automatic cleanup of events older than retention."""
        # Add an old event
        stats.events.append(
            type(
                "CacheEvent",
                (),
                {
                    "timestamp": time.time() - 7200,  # 2 hours ago
                    "cache_type": "routing",
                    "event_type": "hit",
                    "model": "old",
                },
            )
        )
        # Add a recent event
        await stats.record_hit(cache_type="routing", event_type="hit", model="recent")

        # Trigger cleanup (happens every 100 events, but we can call directly)
        stats._cleanup_old_events()

        # Only recent event should remain (retention is 1 hour)
        assert len(stats.events) == 1
        assert stats.events[0].model == "recent"

    @pytest.mark.asyncio
    async def test_get_time_series(self, stats):
        """Test time-series data generation."""
        # Record events at known times (we can't control timestamps exactly,
        # but we can record a few events and ensure buckets are created)
        await stats.record_hit(cache_type="routing", event_type="hit", model="llama3")
        await stats.record_hit(cache_type="routing", event_type="miss", model="llama3")
        await asyncio.sleep(0.1)  # Ensure timestamps differ slightly
        await stats.record_hit(cache_type="response", event_type="hit", model="codellama")

        series = await stats.get_time_series(interval_minutes=1, window_minutes=5)
        assert isinstance(series, list)
        if series:  # Might be empty if window too small
            bucket = series[0]
            assert "start_time" in bucket
            assert "end_time" in bucket
            assert "hits" in bucket
            assert "misses" in bucket
            assert "total" in bucket
            assert "hit_rate" in bucket


class TestCacheAnalytics:
    """Tests for CacheAnalytics class."""

    @pytest.fixture
    def mock_cache(self):
        """Fixture providing mocked SemanticCache instance."""
        cache = MagicMock(spec=SemanticCache)
        cache.cache = {}  # Simulate empty cache initially
        cache.response_cache = {}
        cache.embedding_cache = {}
        cache._routing_lock = asyncio.Lock()
        cache._response_lock = asyncio.Lock()
        cache._embedding_lock = asyncio.Lock()
        cache._eviction_counts = {}
        return cache

    @pytest.fixture
    def analytics(self, mock_cache):
        """Fixture providing CacheAnalytics instance."""
        return CacheAnalytics(mock_cache)

    @pytest.mark.asyncio
    async def test_get_top_prompts_empty(self, analytics, mock_cache):
        """Test top prompts on empty cache."""
        mock_cache.cache = {}
        prompts = await analytics.get_top_prompts(limit=10)
        assert prompts == []

    @pytest.mark.asyncio
    async def test_get_top_prompts_with_entries(self, analytics, mock_cache):
        """Test top prompts with cached entries."""
        # Mock cache entries with access counts

        from router.router import RoutingResult

        result1 = RoutingResult(
            selected_model="llama3",
            confidence=0.9,
            reasoning="test",
        )
        result2 = RoutingResult(
            selected_model="codellama",
            confidence=0.8,
            reasoning="test",
        )

        mock_cache.cache = {
            "hash1": (result1, time.time(), None, None, 5),  # access_count = 5
            "hash2": (result2, time.time(), None, None, 3),  # access_count = 3
        }

        prompts = await analytics.get_top_prompts(limit=2)
        assert len(prompts) == 2
        # Should be sorted by access count descending
        assert prompts[0]["access_count"] == 5
        assert prompts[0]["model"] == "llama3"
        assert prompts[1]["access_count"] == 3
        assert prompts[1]["model"] == "codellama"

    @pytest.mark.asyncio
    async def test_get_model_stats(self, analytics, mock_cache):
        """Test model-specific cache statistics."""
        from router.router import RoutingResult

        result1 = RoutingResult(
            selected_model="llama3",
            confidence=0.9,
            reasoning="test",
        )
        result2 = RoutingResult(
            selected_model="codellama",
            confidence=0.8,
            reasoning="test",
        )
        result3 = RoutingResult(
            selected_model="llama3",
            confidence=0.7,
            reasoning="test",
        )

        mock_cache.cache = {
            "h1": (result1, time.time(), None, None, 2),
            "h2": (result2, time.time(), None, None, 5),
            "h3": (result3, time.time(), None, None, 3),
        }

        stats = await analytics.get_model_stats()
        assert "llama3" in stats
        assert "codellama" in stats
        assert stats["llama3"]["cache_entries"] == 2
        assert stats["llama3"]["total_accesses"] == 5  # 2 + 3
        assert stats["codellama"]["cache_entries"] == 1
        assert stats["codellama"]["total_accesses"] == 5

    @pytest.mark.asyncio
    async def test_get_eviction_stats(self, analytics, mock_cache):
        """Test eviction statistics retrieval."""
        mock_cache._eviction_counts = {
            "size_routing": 5,
            "ttl_response": 3,
            "manual_embedding": 1,
        }

        stats = await analytics.get_eviction_stats()
        assert stats["size_routing"] == 5
        assert stats["ttl_response"] == 3
        assert stats["manual_embedding"] == 1

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, analytics, mock_cache):
        """Test clearing all cache entries."""
        # Populate caches
        mock_cache.cache = {"key1": (MagicMock(), time.time(), None, None, 1)}
        mock_cache.response_cache = {("model1", "hash1"): ("response", time.time())}
        mock_cache.embedding_cache = {"emb1": ([0.1, 0.2], 1.0, time.time())}

        cleared = await analytics.clear_cache()
        assert cleared["routing"] == 1
        assert cleared["response"] == 1
        assert cleared["embedding"] == 1
        assert len(mock_cache.cache) == 0
        assert len(mock_cache.response_cache) == 0
        assert len(mock_cache.embedding_cache) == 0

    @pytest.mark.asyncio
    async def test_clear_cache_by_model(self, analytics, mock_cache):
        """Test clearing cache entries for a specific model."""
        from router.router import RoutingResult

        result1 = RoutingResult(
            selected_model="llama3",
            confidence=0.9,
            reasoning="test",
        )
        result2 = RoutingResult(
            selected_model="codellama",
            confidence=0.8,
            reasoning="test",
        )

        mock_cache.cache = {
            "h1": (result1, time.time(), None, None, 1),
            "h2": (result2, time.time(), None, None, 1),
        }
        mock_cache.response_cache = {
            ("llama3", "p1"): ("resp1", time.time()),
            ("codellama", "p2"): ("resp2", time.time()),
        }

        cleared = await analytics.clear_cache(model="llama3")
        assert cleared["routing"] == 1
        assert cleared["response"] == 1
        assert cleared["embedding"] == 0
        # Only llama3 entries removed
        assert len(mock_cache.cache) == 1
        assert mock_cache.cache["h2"][0].selected_model == "codellama"
        assert len(mock_cache.response_cache) == 1
        assert ("codellama", "p2") in mock_cache.response_cache

    @pytest.mark.asyncio
    async def test_clear_cache_by_age(self, analytics, mock_cache):
        """Test clearing cache entries older than threshold."""
        old_time = time.time() - 7200  # 2 hours ago
        recent_time = time.time() - 300  # 5 minutes ago

        mock_cache.cache = {
            "old": (MagicMock(), old_time, None, None, 1),
            "recent": (MagicMock(), recent_time, None, None, 1),
        }

        cleared = await analytics.clear_cache(older_than_hours=1)  # older than 1 hour
        assert cleared["routing"] == 1
        assert cleared["response"] == 0
        assert cleared["embedding"] == 0
        assert "old" not in mock_cache.cache
        assert "recent" in mock_cache.cache

    @pytest.mark.asyncio
    async def test_warm_cache_placeholder(self, analytics):
        """Test warm cache placeholder (implementation pending)."""
        result = await analytics.warm_cache(prompts=["test prompt"], model="llama3")
        assert result["total_prompts"] == 1
        assert result["model"] == "llama3"
        # Placeholder returns 0 warmed for now
        assert result["warmed"] == 0


class TestSemanticCacheIntegration:
    """Integration tests for cache statistics in SemanticCache."""

    @pytest.fixture
    def semantic_cache(self):
        """Fixture providing SemanticCache with stats enabled."""
        return SemanticCache(
            max_size=10,
            ttl_seconds=3600,
            similarity_threshold=0.8,
            cache_stats_enabled=True,
            cache_stats_retention_hours=1,
        )

    @pytest.mark.asyncio
    async def test_cache_stats_enabled(self, semantic_cache):
        """Test that cache stats components are initialized when enabled."""
        assert semantic_cache.cache_stats_enabled is True
        assert semantic_cache.time_series_stats is not None
        assert semantic_cache.cache_analytics is not None

    @pytest.mark.asyncio
    async def test_cache_stats_disabled(self):
        """Test that cache stats components are None when disabled."""
        cache = SemanticCache(cache_stats_enabled=False)
        assert cache.cache_stats_enabled is False
        assert cache.time_series_stats is None
        assert cache.cache_analytics is None

    @pytest.mark.asyncio
    async def test_record_cache_event_on_hit(self, semantic_cache):
        """Test that cache hits are recorded."""
        # Mock time_series_stats.record_hit
        mock_record = AsyncMock()
        semantic_cache.time_series_stats.record_hit = mock_record

        # Simulate a cache hit via internal method
        await semantic_cache._record_cache_event(
            cache_type="routing",
            event_type="hit",
            model="llama3",
            prompt_hash="abc123",
        )

        mock_record.assert_called_once_with(
            cache_type="routing",
            event_type="hit",
            model="llama3",
            prompt_hash="abc123",
            embedding_dim=None,
        )

    @pytest.mark.asyncio
    async def test_eviction_recording(self, semantic_cache):
        """Test that evictions are recorded."""
        mock_record = AsyncMock()
        semantic_cache.time_series_stats.record_eviction = mock_record

        # Fill cache to force eviction
        semantic_cache.max_size = 2
        for i in range(3):
            prompt = f"prompt{i}"
            result = type(
                "RoutingResult",
                (),
                {"selected_model": "llama3", "confidence": 0.9, "reasoning": ""},
            )
            await semantic_cache.set(prompt, result)

        # Eviction should have been recorded
        assert mock_record.called
        # Check eviction reason is "size"
        call_kwargs = mock_record.call_args[1]
        assert call_kwargs["reason"] == "size"

    @pytest.mark.asyncio
    async def test_get_stats_includes_enhanced_analytics(self, semantic_cache):
        """Test that get_stats includes enhanced analytics when enabled."""
        stats = await semantic_cache.get_stats()
        assert "enhanced" in stats
        enhanced = stats["enhanced"]
        assert "time_series" in enhanced
        assert "top_prompts" in enhanced
        assert "model_stats" in enhanced
        assert "eviction_stats" in enhanced

    @pytest.mark.asyncio
    async def test_evict_oldest_records_manual_eviction(self, semantic_cache):
        """Test that evict_oldest records manual evictions."""
        # Add some entries
        for i in range(3):
            prompt = f"prompt{i}"
            result = type(
                "RoutingResult",
                (),
                {"selected_model": "llama3", "confidence": 0.9, "reasoning": ""},
            )
            await semantic_cache.set(prompt, result)

        # Mock record method to verify manual eviction reason
        mock_record = AsyncMock()
        semantic_cache._record_cache_event = mock_record

        evicted = await semantic_cache.evict_oldest(count=2)
        assert evicted["routing"] == 2

        # Should have recorded evictions with reason "manual"
        assert mock_record.call_count == 2
        for call in mock_record.call_args_list:
            kwargs = call[1]
            assert kwargs["eviction_reason"] == "manual"
            assert kwargs["event_type"] == "eviction"
