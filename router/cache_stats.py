"""
Enhanced cache statistics and analytics for SmarterRouter.

Provides time-series tracking, detailed analytics, and cache management utilities
for the SemanticCache system.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheEvent:
    """A single cache event for time-series tracking."""

    timestamp: float
    cache_type: str  # "routing", "response", "embedding"
    event_type: str  # "hit", "miss", "similarity_hit", "eviction"
    model: str | None = None
    prompt_hash: str | None = None
    eviction_reason: str | None = None  # "ttl", "size", "manual"
    embedding_dim: int | None = None


class TimeSeriesStats:
    """Time-series statistics for cache performance tracking.

    Tracks hit rates, events, and trends over configurable time windows.
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize time-series statistics tracker.

        Args:
            retention_hours: How long to keep historical data (default: 24 hours)
        """
        self.retention_hours = retention_hours
        self.events: list[CacheEvent] = []
        self._lock = asyncio.Lock()
        self._model_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "similarity_hits": 0}
        )
        self._eviction_stats: dict[str, int] = defaultdict(int)

    async def record_hit(
        self,
        cache_type: str,
        event_type: str = "hit",
        model: str | None = None,
        prompt_hash: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Record a cache hit event."""
        async with self._lock:
            event = CacheEvent(
                timestamp=time.time(),
                cache_type=cache_type,
                event_type=event_type,
                model=model,
                prompt_hash=prompt_hash,
                embedding_dim=embedding_dim,
            )
            self.events.append(event)

            # Update model-specific stats if model is provided
            if model:
                if event_type == "hit":
                    self._model_stats[model]["hits"] += 1
                elif event_type == "similarity_hit":
                    self._model_stats[model]["similarity_hits"] += 1
                elif event_type == "miss":
                    self._model_stats[model]["misses"] += 1

            # Clean up old events periodically (every 100 events)
            if len(self.events) % 100 == 0:
                self._cleanup_old_events()

    async def record_eviction(
        self,
        cache_type: str,
        reason: str,
        model: str | None = None,
        prompt_hash: str | None = None,
    ) -> None:
        """Record a cache eviction event."""
        async with self._lock:
            event = CacheEvent(
                timestamp=time.time(),
                cache_type=cache_type,
                event_type="eviction",
                model=model,
                prompt_hash=prompt_hash,
                eviction_reason=reason,
            )
            self.events.append(event)
            self._eviction_stats[reason] += 1

    def _cleanup_old_events(self) -> None:
        """Remove events older than retention period."""
        cutoff = time.time() - (self.retention_hours * 3600)
        self.events = [e for e in self.events if e.timestamp >= cutoff]

    async def get_stats(self, window_minutes: int = 60) -> dict[str, Any]:
        """
        Get statistics for a specific time window.

        Args:
            window_minutes: Time window in minutes (default: 60)

        Returns:
            Dictionary with cache statistics for the window
        """
        async with self._lock:
            self._cleanup_old_events()

            cutoff = time.time() - (window_minutes * 60)
            window_events = [e for e in self.events if e.timestamp >= cutoff]

            # Count events by type
            hits = sum(1 for e in window_events if e.event_type == "hit")
            similarity_hits = sum(1 for e in window_events if e.event_type == "similarity_hit")
            misses = sum(1 for e in window_events if e.event_type == "miss")
            evictions = sum(1 for e in window_events if e.event_type == "eviction")

            total_requests = hits + similarity_hits + misses
            hit_rate = (hits + similarity_hits) / total_requests if total_requests > 0 else 0.0

            # Group by cache type
            by_cache_type: dict[str, dict[str, int]] = defaultdict(
                lambda: {"hits": 0, "similarity_hits": 0, "misses": 0, "evictions": 0}
            )
            for event in window_events:
                if event.event_type == "hit":
                    by_cache_type[event.cache_type]["hits"] += 1
                elif event.event_type == "similarity_hit":
                    by_cache_type[event.cache_type]["similarity_hits"] += 1
                elif event.event_type == "miss":
                    by_cache_type[event.cache_type]["misses"] += 1
                elif event.event_type == "eviction":
                    by_cache_type[event.cache_type]["evictions"] += 1

            return {
                "window_minutes": window_minutes,
                "total_events": len(window_events),
                "hits": hits,
                "similarity_hits": similarity_hits,
                "misses": misses,
                "evictions": evictions,
                "hit_rate": round(hit_rate, 3),
                "by_cache_type": dict(by_cache_type),
                "model_stats": dict(self._model_stats),
                "eviction_stats": dict(self._eviction_stats),
            }

    async def get_time_series(
        self, interval_minutes: int = 5, window_minutes: int = 60
    ) -> list[dict[str, Any]]:
        """
        Get time-series data for visualization.

        Args:
            interval_minutes: Bucket size in minutes (default: 5)
            window_minutes: Total window to cover (default: 60)

        Returns:
            List of buckets with statistics for each time interval
        """
        async with self._lock:
            self._cleanup_old_events()

            cutoff = time.time() - (window_minutes * 60)
            window_events = [e for e in self.events if e.timestamp >= cutoff]

            if not window_events:
                return []

            # Create time buckets
            start_time = min(e.timestamp for e in window_events)
            end_time = max(e.timestamp for e in window_events)
            interval_seconds = interval_minutes * 60

            buckets = []
            current_start = start_time
            while current_start <= end_time:
                current_end = current_start + interval_seconds
                bucket_events = [
                    e for e in window_events if current_start <= e.timestamp < current_end
                ]

                hits = sum(1 for e in bucket_events if e.event_type == "hit")
                similarity_hits = sum(1 for e in bucket_events if e.event_type == "similarity_hit")
                misses = sum(1 for e in bucket_events if e.event_type == "miss")
                total = hits + similarity_hits + misses
                hit_rate = (hits + similarity_hits) / total if total > 0 else 0.0

                buckets.append(
                    {
                        "start_time": current_start,
                        "end_time": current_end,
                        "hits": hits,
                        "similarity_hits": similarity_hits,
                        "misses": misses,
                        "total": total,
                        "hit_rate": round(hit_rate, 3),
                    }
                )

                current_start = current_end

            return buckets


class CacheAnalytics:
    """Advanced cache analytics and management utilities."""

    def __init__(self, semantic_cache):
        """
        Initialize cache analytics.

        Args:
            semantic_cache: Instance of SemanticCache to analyze
        """
        self.semantic_cache = semantic_cache
        self._prompt_frequency: dict[str, int] = defaultdict(int)
        self._model_cache_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"size": 0, "hits": 0, "misses": 0}
        )

    async def get_top_prompts(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get most frequently accessed cached prompts.

        Args:
            limit: Maximum number of prompts to return (default: 50)

        Returns:
            List of prompt statistics sorted by access count
        """
        async with self.semantic_cache._routing_lock:
            prompts = []
            for key, (result, timestamp, _, _, access_count) in self.semantic_cache.cache.items():
                prompts.append(
                    {
                        "hash": key,
                        "model": result.selected_model,
                        "access_count": access_count,
                        "last_accessed": datetime.fromtimestamp(timestamp, UTC).isoformat(),
                        "confidence": result.confidence,
                    }
                )

            # Sort by access count descending
            prompts.sort(key=lambda x: x["access_count"], reverse=True)
            return prompts[:limit]

    async def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get cache statistics per model.

        Returns:
            Dictionary mapping model names to cache statistics
        """
        async with self.semantic_cache._routing_lock:
            stats: dict[str, dict[str, Any]] = {}
            model_cache_counts: dict[str, int] = defaultdict(int)
            model_access_counts: dict[str, int] = defaultdict(int)

            # Count cache entries and access counts per model
            for _key, (result, _, _, _, access_count) in self.semantic_cache.cache.items():
                model = result.selected_model
                model_cache_counts[model] += 1
                model_access_counts[model] += access_count

            # Get total stats for each model
            for model in model_cache_counts:
                stats[model] = {
                    "cache_entries": model_cache_counts[model],
                    "total_accesses": model_access_counts[model],
                    "avg_accesses_per_entry": (
                        model_access_counts[model] / model_cache_counts[model]
                        if model_cache_counts[model] > 0
                        else 0
                    ),
                }

            return stats

    async def get_eviction_stats(self) -> dict[str, int]:
        """
        Get statistics about cache evictions.

        Returns:
            Dictionary mapping eviction reasons to counts
        """
        # Return eviction counts collected by SemanticCache
        if hasattr(self.semantic_cache, "_eviction_counts"):
            return dict(self.semantic_cache._eviction_counts.copy())
        return {}

    async def clear_cache(
        self,
        cache_type: str | None = None,
        model: str | None = None,
        older_than_hours: int | None = None,
    ) -> dict[str, int]:
        """
        Clear cache entries selectively.

        Args:
            cache_type: Type of cache to clear ("routing", "response", "embedding", or None for all)
            model: Only clear entries for this model (if None, clear all models)
            older_than_hours: Only clear entries older than this many hours (if None, all ages)

        Returns:
            Dictionary with counts of cleared entries by cache type
        """
        cleared_counts = {"routing": 0, "response": 0, "embedding": 0}

        # Calculate cutoff time if specified
        cutoff_time = None
        if older_than_hours:
            cutoff_time = time.time() - (older_than_hours * 3600)

        # Clear routing cache
        if cache_type in (None, "routing"):
            async with self.semantic_cache._routing_lock:
                keys_to_remove = []
                for key, (result, timestamp, _, _, _) in self.semantic_cache.cache.items():
                    if model and result.selected_model != model:
                        continue
                    if cutoff_time and timestamp > cutoff_time:
                        continue
                    keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.semantic_cache.cache[key]
                    cleared_counts["routing"] += 1

        # Clear response cache
        if cache_type in (None, "response"):
            async with self.semantic_cache._response_lock:
                keys_to_remove = []
                for key, (_, timestamp) in self.semantic_cache.response_cache.items():
                    if model and key[0] != model:  # key[0] is model name
                        continue
                    if cutoff_time and timestamp > cutoff_time:
                        continue
                    keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.semantic_cache.response_cache[key]
                    cleared_counts["response"] += 1

        # Clear embedding cache
        if cache_type in (None, "embedding"):
            async with self.semantic_cache._embedding_lock:
                keys_to_remove = []
                for key, (_, _, timestamp) in self.semantic_cache.embedding_cache.items():
                    if cutoff_time and timestamp > cutoff_time:
                        continue
                    keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.semantic_cache.embedding_cache[key]
                    cleared_counts["embedding"] += 1

        logger.info(
            f"Cleared cache: {cleared_counts['routing']} routing, "
            f"{cleared_counts['response']} response, {cleared_counts['embedding']} embedding entries"
        )
        return cleared_counts

    async def warm_cache(self, prompts: list[str], model: str | None = None) -> dict[str, Any]:
        """
        Pre-warm cache with specific prompts.

        Args:
            prompts: List of prompts to pre-warm
            model: Model to use for routing (if None, uses default selection)

        Returns:
            Dictionary with warming results
        """
        # This would require the router engine to process prompts
        # For now, return placeholder - will be implemented in integration
        return {
            "total_prompts": len(prompts),
            "warmed": 0,
            "already_cached": 0,
            "errors": 0,
            "model": model,
        }
