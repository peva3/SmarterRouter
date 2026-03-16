import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from router.backends.base import LLMBackend, ModelInfo
from router.benchmark_db import (
    get_benchmarks_for_models,
    invalidate_all_caches,
)
from router.cache_stats import CacheAnalytics, TimeSeriesStats
from router.config import settings
from router.database import get_session
from router.model_filter import filter_model_infos, log_filter_summary
from router.models import ModelFeedback, ModelProfile, RoutingDecision
from router.modality import Modality, get_models_for_modality
from router.persistent_cache import PersistentCacheManager
from router.profiler import profile_all_models
from router.provider_db import get_provider_db

logger = logging.getLogger(__name__)

_PROFILE_CACHE_TTL = 60.0

# Cache for merged benchmarks (local + external) to avoid repeated DB/network calls
_MERGED_BENCHMARKS_CACHE: dict[frozenset, tuple[float, list[dict]]] = {}
_MERGED_BENCHMARKS_CACHE_TTL = 300.0  # 5 minutes

# Cache for prompt analysis to avoid repeated computation
_PROMPT_ANALYSIS_CACHE: dict[str, tuple[float, dict]] = {}
_PROMPT_ANALYSIS_CACHE_TTL = 300.0  # 5 minutes


def get_benchmarks_for_models_with_external(
    model_names: list[str],
) -> list[dict]:
    """Get benchmarks for models from both local router.db and provider.db.

    This function merges benchmark data from:
    1. Local router.db - for Ollama models
    2. provider.db - for external models (OpenAI, Anthropic, etc.)

    Returns a list of benchmark dicts compatible with _calculate_combined_scores.
    """

    # Check merged cache first
    cache_key = frozenset(model_names)
    now = time.monotonic()
    if cache_key in _MERGED_BENCHMARKS_CACHE:
        cached_time, cached_result = _MERGED_BENCHMARKS_CACHE[cache_key]
        if (now - cached_time) < _MERGED_BENCHMARKS_CACHE_TTL:
            logger.debug(f"Using cached merged benchmarks for {len(model_names)} models (age: {now - cached_time:.1f}s)")
            return cached_result
    # Get local benchmarks (from router.db)
    local_benchmarks = get_benchmarks_for_models(model_names)

    # Get external benchmarks (from provider.db) if enabled
    external_benchmarks: dict[str, dict] = {}
    if settings.provider_db_enabled:
        try:
            provider_db = get_provider_db()
            if provider_db.is_available():
                # Get benchmarks for the requested models
                external_benchmarks = provider_db.get_benchmarks_for_models(model_names)

                # Convert to same format as local benchmarks
                # provider.db uses "model_id" instead of "ollama_name"
                for model_id, bench in external_benchmarks.items():
                    bench["ollama_name"] = model_id  # Map to expected key

                if external_benchmarks:
                    logger.debug(
                        f"Found {len(external_benchmarks)} external benchmarks from provider.db"
                    )
        except Exception as e:
            logger.warning(f"Failed to get external benchmarks from provider.db: {e}")

    # Merge: local benchmarks take precedence, then external
    # Create a map of all benchmarks by model name
    merged: dict[str, dict] = {}

    # Add local benchmarks first
    for bench in local_benchmarks:
        name = bench.get("ollama_name")
        if name:
            merged[name] = bench

    # Add external benchmarks for models not in local
    for model_name in model_names:
        if model_name not in merged and model_name in external_benchmarks:
            merged[model_name] = external_benchmarks[model_name]

    # Store in cache before returning
    result = list(merged.values())
    _MERGED_BENCHMARKS_CACHE[cache_key] = (now, result)
    return result


_profiles_cache: list[dict] | None = None
_profiles_cache_time: float = 0.0
_profiles_cache_lock = asyncio.Lock()


@dataclass
class RoutingResult:
    selected_model: str
    confidence: float
    reasoning: str


# Minimum model sizes (in billions) required for task complexity
# If a model is below the minimum for its category/complexity, it gets a severe penalty
CATEGORY_MIN_SIZES = {
    "coding": {"simple": 0, "medium": 4, "hard": 8},
    "reasoning": {"simple": 0, "medium": 4, "hard": 8},
    "creativity": {"simple": 0, "medium": 1, "hard": 4},
    "general": {"simple": 0, "medium": 1, "hard": 4},
}

# Scoring configuration constants to replace magic numbers
# These can be moved to settings in the future
SCORING_CONFIG = {
    # Category boost multipliers
    "dominant_category_boost_with_data": 20.0,
    "dominant_category_boost_with_size": 10.0,
    "dominant_category_boost_default": 1.5,
    "dominant_category_threshold": 0.15,
    # Signal weights
    "benchmark_weight": 1.5,
    "elo_weight": 1.0,
    "name_inference_weight": 0.4,
    "profile_weight": 0.8,
    # Bonus multipliers
    "feedback_bonus_multiplier": 2.0,
    "has_benchmark_bonus": 0.3,
    # Size scoring
    "size_score_very_large": 3.0,
    "size_score_large": 2.0,
    "size_score_medium": 1.0,
    "size_score_tiny_penalty_high": -2.0,
    "size_score_tiny_penalty_low": -0.5,
    # Complexity thresholds
    "complexity_moderate_threshold": 0.3,
    "complexity_high_threshold": 0.5,
    "complexity_low_threshold": 0.15,
    # Complexity-size matching bonuses
    "high_complexity_very_large_bonus": 3.0,
    "high_complexity_large_bonus": 2.0,
    "high_complexity_medium_bonus": 1.0,
    "high_complexity_small_penalty": -3.0,
    "moderate_complexity_very_large_bonus": 2.0,
    "moderate_complexity_large_bonus": 1.2,
    "moderate_complexity_medium_bonus": 0.4,
    "moderate_complexity_small_penalty": -2.0,
    "low_complexity_tiny_bonus": 1.5,
    "low_complexity_small_bonus": 0.8,
    "low_complexity_large_penalty": -1.0,
    "low_complexity_very_large_penalty": -2.0,
    "size_score_multiplier": 0.5,
    "negative_weight_multiplier": 0.01,
    "quality_preference_boost": 0.5,
    "quality_preference_threshold": 0.5,
    "dominant_category_confidence_threshold": 0.5,
}


class SemanticCache:
    """Smart LRU cache for routing decisions with semantic similarity and response caching.

    Optimized with numpy for vectorized similarity search and separate locks for
    routing and response caches to minimize contention.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.85,
        embed_model: str | None = None,
        response_max_size: int = 50,
        embedding_ttl_seconds: int = 86400,  # 24 hours for embeddings,
        persistent_cache_manager: PersistentCacheManager | None = None,
        cache_stats_enabled: bool | None = None,
        cache_stats_retention_hours: int = 24,
    ):
        self.cache: OrderedDict[
            str, tuple[RoutingResult, float, list[float] | None, float | None, int]
        ] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.embed_model = embed_model
        self.recent_selections: list[tuple[str, float]] = []
        self.max_recent = 20
        self._routing_lock = asyncio.Lock()
        self._response_lock = asyncio.Lock()
        self._embedding_lock = asyncio.Lock()

        self.response_cache: OrderedDict[tuple, tuple[str, float]] = OrderedDict()
        self.response_max_size = response_max_size
        self.response_ttl = ttl_seconds

        # Separate embedding cache with longer TTL (24h default)
        # Key: prompt hash, Value: (embedding, magnitude, timestamp)
        self.embedding_cache: OrderedDict[str, tuple[list[float], float, float]] = OrderedDict()
        self.embedding_ttl = embedding_ttl_seconds
        self.embedding_max_size = max_size * 5  # Allow more embeddings than routing decisions

        self.stats = {
            "routing_hits": 0,
            "routing_misses": 0,
            "routing_similarity_hits": 0,
            "response_hits": 0,
            "response_misses": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "adaptive_threshold_adjustments": 0,
        }

        # Enhanced cache statistics (SmarterRouter 2.1.6+)
        if cache_stats_enabled is None:
            cache_stats_enabled = settings.cache_stats_enabled
        self.cache_stats_enabled = cache_stats_enabled

        if self.cache_stats_enabled:
            self.time_series_stats: TimeSeriesStats | None = TimeSeriesStats(retention_hours=cache_stats_retention_hours)
            self.cache_analytics: CacheAnalytics | None = CacheAnalytics(self)
        else:
            self.time_series_stats = None
            self.cache_analytics = None

        self.persistent_cache = persistent_cache_manager
        self._model_frequency: dict[str, int] = {}
        self._access_counts: dict[str, int] = {}  # Track access counts for cached entries
        self._eviction_counts: dict[str, int] = {}  # Track eviction reasons

    async def _record_cache_event(
        self,
        cache_type: str,
        event_type: str,
        model: str | None = None,
        prompt_hash: str | None = None,
        embedding_dim: int | None = None,
        eviction_reason: str | None = None,
    ) -> None:
        """Record a cache event in time-series statistics if enabled."""
        if self.cache_stats_enabled and self.time_series_stats:
            if event_type == "eviction":
                await self.time_series_stats.record_eviction(
                    cache_type=cache_type,
                    reason=eviction_reason or "unknown",
                    model=model,
                    prompt_hash=prompt_hash,
                )
            else:
                await self.time_series_stats.record_hit(
                    cache_type=cache_type,
                    event_type=event_type,
                    model=model,
                    prompt_hash=prompt_hash,
                    embedding_dim=embedding_dim,
                )

        # Track eviction counts for analytics
        if event_type == "eviction":
            key = f"{eviction_reason or 'unknown'}_{cache_type}"
            self._eviction_counts[key] = self._eviction_counts.get(key, 0) + 1

    async def load_from_persistence(self) -> None:
        """Load cache data from persistent storage if enabled."""
        if not self.persistent_cache or not self.persistent_cache.enabled:
            return

        try:
            # Load routing cache with access counts, prioritizing high-access entries
            routing_data = await self.persistent_cache.load_routing_cache()
            async with self._routing_lock:
                self.cache.clear()
                self._access_counts.clear()
                # Sort by access_count descending, limit to max_size (Top-K pre-caching)
                sorted_items = sorted(
                    routing_data.items(),
                    key=lambda x: x[1][4],  # access_count is 5th element in tuple
                    reverse=True,
                )[: self.max_size]
                for cache_key, (
                    result,
                    timestamp,
                    embedding,
                    magnitude,
                    access_count,
                ) in sorted_items:
                    self.cache[cache_key] = (result, timestamp, embedding, magnitude, access_count)
                    self._access_counts[cache_key] = access_count

            # Load response cache, limit to max size
            response_data = await self.persistent_cache.load_response_cache()
            async with self._response_lock:
                self.response_cache.clear()
                # Take only top response_max_size entries (already sorted by access_count)
                response_items = list(response_data.items())[: self.response_max_size]
                for resp_key, (response_text, timestamp) in response_items:
                    self.response_cache[resp_key] = (response_text, timestamp)

            # Load embedding cache, limit to max size
            embedding_data = await self.persistent_cache.load_embedding_cache()
            async with self._embedding_lock:
                self.embedding_cache.clear()
                # Take only top embedding_max_size entries (already sorted by access_count)
                embedding_items = list(embedding_data.items())[: self.embedding_max_size]
                for prompt_hash, (embedding, magnitude, timestamp) in embedding_items:
                    self.embedding_cache[prompt_hash] = (embedding, magnitude, timestamp)

            logger.info(
                f"Loaded from persistent cache: {len(self.cache)} routing (top {len(self.cache)} by access count), "
                f"{len(self.response_cache)} response, {len(self.embedding_cache)} embedding entries (top-K limited)"
            )
        except Exception as e:
            logger.error(f"Failed to load from persistent cache: {e}")

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    def _cosine_similarity(
        self, a: list[float], b: list[float], mag_a: float, mag_b: float
    ) -> float:
        if not a or not b or mag_a == 0 or mag_b == 0:
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        return dot_product / (mag_a * mag_b)

    def _cosine_similarity_batch(
        self, query: list[float], query_mag: float, candidates: list[tuple[str, list[float], float]]
    ) -> list[tuple[str, float]]:
        """Vectorized batch cosine similarity using numpy for better performance.

        Args:
            query: Query embedding vector
            query_mag: Magnitude of query vector
            candidates: List of (key, embedding, magnitude) tuples

        Returns:
            List of (key, similarity) tuples sorted by similarity descending
        """
        if not candidates:
            return []

        try:
            import numpy as np

            query_arr = np.array(query, dtype=np.float32)
            candidate_matrix = np.array([c[1] for c in candidates], dtype=np.float32)
            mags = np.array([c[2] for c in candidates], dtype=np.float32)
            keys = [c[0] for c in candidates]

            dot_products = candidate_matrix @ query_arr
            similarities = dot_products / (mags * query_mag + 1e-10)

            results = [(key, float(sim)) for key, sim in zip(keys, similarities, strict=True)]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except ImportError:
            results = [
                (c[0], self._cosine_similarity(query, c[1], query_mag, c[2])) for c in candidates
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    async def _get_embedding(self, client: LLMBackend, text: str) -> list[float] | None:
        """Get embedding for text, using cache if available.

        Embeddings are cached for 24 hours by default to avoid repeated
        expensive embedding API calls for the same prompts.

        Uses a simple two-phase check pattern:
        1. Check cache under lock, if hit return immediately
        2. If miss, release lock and fetch embedding (allows concurrent fetches)
        3. Re-acquire lock and store result (handles race with other fetchers)
        """
        if not self.embed_model:
            return None

        key = self._hash_prompt(text[:8192])
        current_time = time.time()

        # Phase 1: Check cache under lock
        async with self._embedding_lock:
            if key in self.embedding_cache:
                emb, mag, timestamp = self.embedding_cache[key]
                if current_time - timestamp < self.embedding_ttl:
                    self.embedding_cache.move_to_end(key)
                    self.stats["embedding_cache_hits"] += 1
                    await self._record_cache_event(
                        cache_type="embedding",
                        event_type="hit",
                        model=None,
                        prompt_hash=key,
                        embedding_dim=len(emb) if emb and isinstance(emb, list) else None,
                    )
                    return emb
                else:
                    # TTL expired, record eviction
                    await self._record_cache_event(
                        cache_type="embedding",
                        event_type="eviction",
                        model=None,
                        prompt_hash=key,
                        embedding_dim=len(emb) if emb and isinstance(emb, list) else None,
                        eviction_reason="ttl",
                    )
                    del self.embedding_cache[key]

        # Phase 2: Fetch embedding outside lock (allows concurrent fetches for different keys)
        try:
            result = await client.embed(self.embed_model, text[:8192])
            embeddings = result.get("embeddings") or result.get("embedding")
            if not embeddings:
                return None

            emb = embeddings[0] if isinstance(embeddings[0], list) else embeddings
            if not emb or not isinstance(emb, list):
                return None

            mag = sum(x * x for x in emb) ** 0.5

            # Phase 3: Store under lock (handles race with other fetchers)
            async with self._embedding_lock:
                # Check again - another task may have stored while we were fetching
                if key in self.embedding_cache:
                    # Use cached result from other task
                    cached_emb, cached_mag, cached_ts = self.embedding_cache[key]
                    if current_time - cached_ts < self.embedding_ttl:
                        self.embedding_cache.move_to_end(key)
                        self.stats["embedding_cache_hits"] += 1
                        await self._record_cache_event(
                            cache_type="embedding",
                            event_type="hit",
                            model=None,
                            prompt_hash=key,
                            embedding_dim=len(cached_emb)
                            if cached_emb and isinstance(cached_emb, list)
                            else None,
                        )
                        return cached_emb

                # Store our result
                self.embedding_cache[key] = (emb, mag, current_time)
                self.embedding_cache.move_to_end(key)
                if len(self.embedding_cache) > self.embedding_max_size:
                    old_key, (old_emb, old_mag, old_ts) = self.embedding_cache.popitem(last=False)
                    await self._record_cache_event(
                        cache_type="embedding",
                        event_type="eviction",
                        model=None,
                        prompt_hash=old_key,
                        embedding_dim=len(old_emb)
                        if old_emb and isinstance(old_emb, list)
                        else None,
                        eviction_reason="size",
                    )

            # Save to persistent cache if enabled (outside lock)
            if self.persistent_cache and self.persistent_cache.enabled:
                try:
                    await self.persistent_cache.save_embedding_entry(
                        prompt_hash=key,
                        embedding=emb,
                        magnitude=mag,
                        ttl_seconds=self.embedding_ttl,
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to save embedding cache entry {key[:8]} to persistent storage: {e}"
                    )

            self.stats["embedding_cache_misses"] += 1
            await self._record_cache_event(
                cache_type="embedding",
                event_type="miss",
                model=None,
                prompt_hash=key,
                embedding_dim=len(emb) if emb and isinstance(emb, list) else None,
            )
            return emb
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
            return None

    def _calculate_adaptive_threshold(self, cache_key: str | None = None) -> float:
        """Calculate adaptive similarity threshold based on cache performance and query patterns.

        Args:
            cache_key: Optional cache key of the candidate entry

        Returns:
            Adaptive similarity threshold (0.7 to 0.95)
        """
        base_threshold = self.similarity_threshold

        # Factor 1: Overall cache hit rate
        total_routing = self.stats["routing_hits"] + self.stats["routing_misses"]
        hit_rate_factor = 0.0
        if total_routing > 10:
            hit_rate = self.stats["routing_hits"] / total_routing
            # Low hit rate (< 0.3): lower threshold by up to -0.15
            # High hit rate (> 0.7): raise threshold by up to +0.1
            if hit_rate < 0.3:
                hit_rate_factor = -0.15 * (0.3 - hit_rate) / 0.3
            elif hit_rate > 0.7:
                hit_rate_factor = 0.1 * (hit_rate - 0.7) / 0.3
        else:
            # Not enough data, skip hit rate factor
            hit_rate_factor = 0.0

        # Factor 2: Model frequency (if cache_key provided)
        model_frequency_factor = 0.0
        if cache_key and cache_key in self.cache:
            result, _, _, _, _ = self.cache[cache_key]
            model_name = result.selected_model
            # Get model frequency from recent selections
            model_freq = self._model_frequency.get(model_name, 0)
            total_freq = sum(self._model_frequency.values())
            if total_freq > 0:
                freq_ratio = model_freq / total_freq
                # High frequency models (> 0.5): raise threshold for precision
                # Low frequency models: no adjustment
                if freq_ratio > 0.5:
                    model_frequency_factor = 0.05 * (freq_ratio - 0.5) / 0.5

        # Factor 3: Time of day (encourage exploration during low-traffic times)
        # Not implemented yet

        # Combine factors
        adaptive = base_threshold + hit_rate_factor + model_frequency_factor
        # Clamp to reasonable range
        adaptive = max(0.7, min(0.95, adaptive))

        # Log adjustment if changed significantly
        if abs(adaptive - base_threshold) > 0.05:
            self.stats["adaptive_threshold_adjustments"] += 1
            logger.debug(
                f"Adaptive threshold: {base_threshold:.2f} -> {adaptive:.2f} "
                f"(hit_rate_factor={hit_rate_factor:.3f}, "
                f"model_freq_factor={model_frequency_factor:.3f})"
            )

        return adaptive

    async def get(self, prompt: str, embedding: list[float] | None = None) -> RoutingResult | None:
        key = self._hash_prompt(prompt)
        current_time = time.time()

        async with self._routing_lock:
            if key in self.cache:
                result, timestamp, emb, mag, acc = self.cache[key]
                if current_time - timestamp < self.ttl:
                    # Increment access count
                    acc += 1
                    self.cache[key] = (result, timestamp, emb, mag, acc)
                    self._access_counts[key] = acc
                    self.cache.move_to_end(key)
                    self.stats["routing_hits"] += 1
                    await self._record_cache_event(
                        cache_type="routing",
                        event_type="hit",
                        model=result.selected_model,
                        prompt_hash=key,
                    )
                    logger.debug(f"Cache hit (exact) for prompt hash: {key[:8]}...")
                    return result
                else:
                    # TTL expired, record eviction
                    await self._record_cache_event(
                        cache_type="routing",
                        event_type="eviction",
                        model=result.selected_model,
                        prompt_hash=key,
                        eviction_reason="ttl",
                    )
                    del self.cache[key]
                    if key in self._access_counts:
                        del self._access_counts[key]

            if embedding:
                embedding_mag = sum(x * x for x in embedding) ** 0.5
                candidates = [
                    (cache_key, cache_emb, cache_mag)
                    for cache_key, (_, timestamp, cache_emb, cache_mag, _) in self.cache.items()
                    if cache_emb and cache_mag and current_time - timestamp < self.ttl
                ]

                if candidates:
                    similarities = self._cosine_similarity_batch(
                        embedding, embedding_mag, candidates
                    )

                    for cache_key, similarity in similarities:
                        # Use adaptive threshold for this comparison
                        threshold = self._calculate_adaptive_threshold(cache_key)
                        if similarity >= threshold:
                            result, timestamp, emb, mag, acc = self.cache[cache_key]
                            acc += 1
                            self.cache[cache_key] = (result, timestamp, emb, mag, acc)
                            self._access_counts[cache_key] = acc
                            self.cache.move_to_end(cache_key)
                            self.stats["routing_similarity_hits"] += 1
                            await self._record_cache_event(
                                cache_type="routing",
                                event_type="similarity_hit",
                                model=result.selected_model,
                                prompt_hash=cache_key,
                            )
                            logger.debug(
                                f"Cache hit (similarity={similarity:.2f}, threshold={threshold:.2f}) "
                                f"for prompt hash: {cache_key[:8]}..."
                            )
                            return result

            self.stats["routing_misses"] += 1
            await self._record_cache_event(
                cache_type="routing",
                event_type="miss",
                model=None,
                prompt_hash=None,
            )
            return None

    async def set(
        self,
        prompt: str,
        result: RoutingResult,
        embedding: list[float] | None = None,
    ) -> None:
        key = self._hash_prompt(prompt)
        async with self._routing_lock:
            mag = sum(x * x for x in embedding) ** 0.5 if embedding else None
            self.cache[key] = (result, time.time(), embedding, mag, 1)
            self._access_counts[key] = 1
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                old_key, (old_result, old_ts, old_emb, old_mag, old_acc) = self.cache.popitem(
                    last=False
                )
                await self._record_cache_event(
                    cache_type="routing",
                    event_type="eviction",
                    model=old_result.selected_model,
                    prompt_hash=old_key,
                    eviction_reason="size",
                )
                self._access_counts.pop(old_key, None)

            self.recent_selections.append((result.selected_model, time.time()))
            if len(self.recent_selections) > self.max_recent:
                old = self.recent_selections.pop(0)
                if old[0] in self._model_frequency:
                    self._model_frequency[old[0]] = max(0, self._model_frequency[old[0]] - 1)

            self._model_frequency[result.selected_model] = (
                self._model_frequency.get(result.selected_model, 0) + 1
            )

        # Save to persistent cache if enabled
        if self.persistent_cache and self.persistent_cache.enabled:
            try:
                await self.persistent_cache.save_routing_entry(
                    cache_key=key,
                    result=result,
                    embedding=embedding,
                    embedding_magnitude=mag,
                    ttl_seconds=self.ttl,
                )
            except Exception as e:
                logger.debug(
                    f"Failed to save routing cache entry {key[:8]} to persistent storage: {e}"
                )

        logger.debug(f"Cached routing decision for: {key[:8]}...")

    def _make_response_key(self, model: str, prompt: str, params: dict | None = None) -> tuple:
        """Create cache key including model, prompt, and generation parameters."""
        prompt_hash = self._hash_prompt(prompt)
        if params:
            # Include relevant generation parameters that affect output
            param_tuple = tuple(
                sorted(
                    [
                        (k, v)
                        for k, v in params.items()
                        if v is not None
                        and k
                        in (
                            "temperature",
                            "top_p",
                            "max_tokens",
                            "seed",
                            "presence_penalty",
                            "frequency_penalty",
                        )
                    ]
                )
            )
            return (model, prompt_hash, param_tuple)
        return (model, prompt_hash)

    async def get_response(self, model: str, prompt: str, params: dict | None = None) -> str | None:
        key = self._make_response_key(model, prompt, params)
        current_time = time.time()

        async with self._response_lock:
            if key in self.response_cache:
                response, timestamp = self.response_cache[key]
                if current_time - timestamp < self.response_ttl:
                    self.response_cache.move_to_end(key)
                    self.stats["response_hits"] += 1
                    await self._record_cache_event(
                        cache_type="response",
                        event_type="hit",
                        model=model,
                        prompt_hash=None,
                    )
                    logger.debug(f"Response cache hit for {model}")
                    return response
                else:
                    # TTL expired, record eviction
                    await self._record_cache_event(
                        cache_type="response",
                        event_type="eviction",
                        model=model,
                        prompt_hash=None,
                        eviction_reason="ttl",
                    )
                    del self.response_cache[key]

            self.stats["response_misses"] += 1
            await self._record_cache_event(
                cache_type="response",
                event_type="miss",
                model=model,
                prompt_hash=None,
            )
            return None

    async def set_response(
        self, model: str, prompt: str, response: str, params: dict | None = None
    ) -> None:
        key = self._make_response_key(model, prompt, params)
        async with self._response_lock:
            self.response_cache[key] = (response, time.time())
            self.response_cache.move_to_end(key)
            if len(self.response_cache) > self.response_max_size:
                self.response_cache.popitem(last=False)
        # Save to persistent cache if enabled
        if self.persistent_cache and self.persistent_cache.enabled:
            try:
                await self.persistent_cache.save_response_entry(
                    cache_key=key,
                    response_text=response,
                    ttl_seconds=self.response_ttl,
                )
            except Exception as e:
                logger.debug(
                    f"Failed to save response cache entry for {model} to persistent storage: {e}"
                )

        logger.debug(f"Cached response for {model}")

    async def invalidate_response(self, model: str | None = None) -> int:
        count = 0
        async with self._response_lock:
            if model is None:
                count = len(self.response_cache)
                self.response_cache.clear()
            else:
                keys_to_remove = [k for k in self.response_cache if k[0] == model]
                count = len(keys_to_remove)
                for k in keys_to_remove:
                    del self.response_cache[k]
        return count

    async def get_model_frequency(self, model_name: str) -> float:
        async with self._routing_lock:
            if not self._model_frequency:
                return 0.0
            total = sum(self._model_frequency.values())
            if total == 0:
                return 0.0
            return self._model_frequency.get(model_name, 0) / total

    async def clear(self) -> None:
        """Clear all caches safely using their respective locks."""
        async with self._routing_lock:
            self.cache.clear()
            self.recent_selections.clear()
            self._model_frequency.clear()
        async with self._response_lock:
            self.response_cache.clear()
        async with self._embedding_lock:
            self.embedding_cache.clear()

    async def evict_oldest(self, count: int = 1) -> dict[str, int]:
        """
        Evict the oldest entries from each cache (LRU eviction).

        Args:
            count: Number of oldest entries to evict from each cache

        Returns:
            Dictionary with number of entries evicted per cache type
        """
        if count <= 0:
            return {"routing": 0, "response": 0, "embedding": 0}

        evicted = {"routing": 0, "response": 0, "embedding": 0}

        # Evict from routing cache
        async with self._routing_lock:
            for _ in range(min(count, len(self.cache))):
                key, (result, _, _, _, _) = self.cache.popitem(last=False)
                evicted["routing"] += 1
                await self._record_cache_event(
                    cache_type="routing",
                    event_type="eviction",
                    model=result.selected_model,
                    prompt_hash=key,
                    eviction_reason="manual",
                )

        # Evict from response cache
        async with self._response_lock:
            for _ in range(min(count, len(self.response_cache))):
                key, (response_text, _) = self.response_cache.popitem(last=False)
                evicted["response"] += 1
                await self._record_cache_event(
                    cache_type="response",
                    event_type="eviction",
                    model=key[0] if isinstance(key, tuple) and len(key) > 0 else None,
                    eviction_reason="manual",
                )

        # Evict from embedding cache
        async with self._embedding_lock:
            for _ in range(min(count, len(self.embedding_cache))):
                key, (embedding, magnitude, _) = self.embedding_cache.popitem(last=False)
                evicted["embedding"] += 1
                await self._record_cache_event(
                    cache_type="embedding",
                    event_type="eviction",
                    prompt_hash=key,
                    eviction_reason="manual",
                )

        return evicted

    async def get_stats(self) -> dict[str, Any]:
        async with self._routing_lock:
            routing_stats: dict[str, int | float] = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["routing_hits"],
                "similarity_hits": self.stats["routing_similarity_hits"],
                "misses": self.stats["routing_misses"],
                "adaptive_threshold_adjustments": self.stats["adaptive_threshold_adjustments"],
            }
            total_routing = self.stats["routing_hits"] + self.stats["routing_misses"]
            routing_stats["hit_rate"] = round(
                self.stats["routing_hits"] / total_routing if total_routing > 0 else 0.0, 3
            )

        async with self._response_lock:
            response_stats: dict[str, int | float] = {
                "size": len(self.response_cache),
                "max_size": self.response_max_size,
                "hits": self.stats["response_hits"],
                "misses": self.stats["response_misses"],
            }
            total_response = self.stats["response_hits"] + self.stats["response_misses"]
            response_stats["hit_rate"] = round(
                self.stats["response_hits"] / total_response if total_response > 0 else 0.0, 3
            )

        async with self._embedding_lock:
            embedding_stats: dict[str, int | float] = {
                "size": len(self.embedding_cache),
                "max_size": self.embedding_max_size,
                "hits": self.stats["embedding_cache_hits"],
                "misses": self.stats["embedding_cache_misses"],
            }
            total_embedding = (
                self.stats["embedding_cache_hits"] + self.stats["embedding_cache_misses"]
            )
            embedding_stats["hit_rate"] = round(
                self.stats["embedding_cache_hits"] / total_embedding
                if total_embedding > 0
                else 0.0,
                3,
            )

        result = {
            "routing": routing_stats,
            "response": response_stats,
            "embedding": embedding_stats,
        }

        if self.cache_stats_enabled and self.time_series_stats and self.cache_analytics:
            # Get time-series stats for last hour
            assert self.time_series_stats is not None
            time_series = await self.time_series_stats.get_stats(window_minutes=60)
            analytics: dict[str, Any] = {
                "time_series": time_series,
                "top_prompts": await self.cache_analytics.get_top_prompts(limit=20),
                "model_stats": await self.cache_analytics.get_model_stats(),
                "eviction_stats": await self.cache_analytics.get_eviction_stats(),
            }
            result["enhanced"] = analytics

        return result


class RouterEngine:
    def __init__(
        self,
        client: LLMBackend,
        dispatcher_model: str | None = None,
        cache_enabled: bool = True,
        cache_max_size: int = 500,
        cache_ttl_seconds: int = 3600,
        cache_similarity_threshold: float = 0.85,
        cache_response_max_size: int = 200,
        embed_model: str | None = None,
        vram_manager: Any | None = None,  # VRAMManager, using Any to avoid circular import
        persistent_cache_enabled: bool | None = None,
        persistent_cache_max_age_days: int = 7,
        cache_stats_enabled: bool | None = None,
        cache_stats_retention_hours: int = 24,
    ):
        self.client = client
        self.dispatcher_model = dispatcher_model or settings.router_model
        self.cache_enabled = cache_enabled
        self.embed_model = embed_model
        self.vram_manager = vram_manager
        self.cache_stats_enabled = cache_stats_enabled
        self.cache_stats_retention_hours = cache_stats_retention_hours
        self.semantic_cache: SemanticCache | None

        # Model list caching to reduce backend API calls
        self._models_cache: list[ModelInfo] | None = None
        self._models_cache_time: float = 0.0
        self._models_cache_ttl: float = 10.0

        if cache_enabled:
            # Create persistent cache manager if enabled
            persistent_cache_manager = None
            if persistent_cache_enabled is None:
                # Use settings default if not explicitly set
                persistent_cache_manager = PersistentCacheManager(
                    enabled=settings.persistent_cache_enabled,
                    max_age_days=persistent_cache_max_age_days,
                )
            elif persistent_cache_enabled:
                persistent_cache_manager = PersistentCacheManager(
                    enabled=True,
                    max_age_days=persistent_cache_max_age_days,
                )

            self.semantic_cache = SemanticCache(
                max_size=cache_max_size,
                ttl_seconds=cache_ttl_seconds,
                similarity_threshold=cache_similarity_threshold,
                embed_model=embed_model,
                response_max_size=cache_response_max_size,
                persistent_cache_manager=persistent_cache_manager,
                cache_stats_enabled=cache_stats_enabled,
                cache_stats_retention_hours=cache_stats_retention_hours,
            )
        else:
            self.semantic_cache = None

    async def load_persistent_cache(self) -> None:
        """Load cache data from persistent storage."""
        if self.cache_enabled and self.semantic_cache:
            await self.semantic_cache.load_from_persistence()
            # Expired entries cleanup is handled by background task

    async def warmup_caches(self, model_names: list[str] | None = None) -> None:
        """Pre-warm caches on startup to avoid first-request latency."""
        from router.benchmark_db import get_benchmarks_for_models

        await self._get_all_profiles()
        if model_names:
            # Warm both local and external benchmarks
            get_benchmarks_for_models(model_names)
            get_benchmarks_for_models_with_external(model_names)
        logger.info("Router caches pre-warmed")

    def invalidate_caches(self) -> None:
        """Invalidate all caches (call when models change)."""
        invalidate_all_caches()
        # Also invalidate provider.db cache
        from router.provider_db import invalidate_provider_cache

        invalidate_provider_cache()
        logger.info("Router caches invalidated")

    async def refresh_models(self, cleanup: bool | None = None) -> dict[str, Any]:
        """Refresh model list and update availability.

        Args:
            cleanup: If True, mark missing models as inactive. If None, use settings.model_cleanup_enabled.

        Returns:
            Dictionary with counts and changes.
        """
        if cleanup is None:
            cleanup = settings.model_cleanup_enabled

        # Use cached model list if available
        now = time.monotonic()
        if self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            available_models = self._models_cache
            logger.debug("Using cached model list (age: %.1fs)", now - self._models_cache_time)
        else:
            self._models_cache = None
            self._models_cache_time = 0.0
            # Invalidate model cache on explicit refresh
            available_models = await self.client.list_models()
            self._models_cache = available_models
            self._models_cache_time = now
        available_names = {m.name for m in available_models}

        with get_session() as session:
            # Update last_seen for active models
            # Get all existing profiles in one query
            profiles = session.query(ModelProfile).all()
            existing_names = {p.name for p in profiles}
            changes = {"added": 0, "removed": 0, "updated": 0}

            # Process existing profiles
            for profile in profiles:
                if profile.name in available_names:
                    if not profile.active or profile.last_seen is None:
                        profile.active = True
                        profile.last_seen = datetime.now(UTC)
                        changes["updated"] += 1
                    else:
                        # Update last_seen anyway
                        profile.last_seen = datetime.now(UTC)
                else:
                    if cleanup and profile.active:
                        profile.active = False
                        changes["removed"] += 1

            # Count new models not in profiles
            logger.debug(f"Existing names: {existing_names}, available names: {available_names}")
            new_models = [m for m in available_models if m.name not in existing_names]
            changes["added"] = len(new_models)

            session.commit()

            # Trigger profiling for new models if auto-profile enabled
            if new_models and settings.model_auto_profile_enabled:
                logger.info(f"Auto-profiling {len(new_models)} new models")
                # Note: profile_all_models will only profile new models by default
                await profile_all_models(self.client)

            logger.info(
                f"Model refresh completed: {changes['added']} added, "
                f"{changes['removed']} removed, {changes['updated']} updated"
            )
            return changes

    async def reprofile_models(self, force: bool = False) -> dict[str, Any]:
        """Re-profile all models (or only those needing updates).

        Args:
            force: If True, re-profile all models regardless of last profile time.

        Returns:
            Dictionary with profiling results.
        """
        logger.info(f"Starting model re-profiling (force={force})")
        results = await profile_all_models(self.client, force=force)
        # Update availability (mark all profiled models as active)
        with get_session() as session:
            for result in results:
                profile = session.query(ModelProfile).filter_by(name=result.model_name).first()
                if profile:
                    profile.active = True
                    profile.last_seen = datetime.now(UTC)
            session.commit()
        logger.info(f"Re-profiling completed: {len(results)} models profiled")
        return {"profiled": len(results), "results": [r.model_name for r in results]}

    async def select_model(
        self, prompt: str | list[dict], request_obj: Any = None
    ) -> RoutingResult:
        prompt_str = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)

        # Always attempt cache lookup when enabled - exact hash works without embedding model
        embedding: list[float] | None = None
        if self.cache_enabled and self.semantic_cache:
            if self.embed_model:
                embedding = await self.semantic_cache._get_embedding(self.client, prompt_str)
            cached = await self.semantic_cache.get(prompt_str, embedding)
            if cached:
                return cached

        # Use cached model list if available
        now = time.monotonic()
        if self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            available_models = self._models_cache
            logger.debug("Using cached model list (age: %.1fs)", now - self._models_cache_time)
        else:
            # Invalidate model cache on explicit refresh
            self._models_cache = None
            self._models_cache_time = 0.0
            available_models = await self.client.list_models()
            self._models_cache = available_models
            self._models_cache_time = now

        # Apply model filtering if configured
        include = settings.model_filter_include
        exclude = settings.model_filter_exclude
        if include or exclude:
            original_count = len(available_models)
            available_models = filter_model_infos(available_models, include, exclude)
            excluded_count = original_count - len(available_models)
            log_filter_summary(
                original_count, len(available_models), excluded_count, include, exclude
            )

        if not available_models:
            raise ValueError(
                f"No models available after filtering (include={include}, exclude={exclude})"
            )

        model_names = [m.name for m in available_models]

        # Convert prompt to string for analysis
        if isinstance(prompt, str):
            text_prompt = prompt
        else:
            text_parts: list[str] = []
            for msg in prompt:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
            text_prompt = "\n".join(text_parts)

        if self.dispatcher_model:
            result = await self._llm_dispatch(text_prompt, model_names)
        else:
            result = await self._keyword_dispatch(text_prompt, model_names, request_obj)

        if self.cache_enabled and self.semantic_cache:
            await self.semantic_cache.set(prompt_str, result, embedding)

        return result

    async def _llm_dispatch(self, prompt: str, model_names: list[str]) -> RoutingResult:
        # ... existing implementation ... (no changes needed here for now as it's string based)
        benchmarks = get_benchmarks_for_models_with_external(model_names)

        if not benchmarks:
            logger.warning("No benchmark data, falling back to keyword dispatch")
            return await self._keyword_dispatch(prompt, model_names)

        if not self.dispatcher_model:
            logger.warning("No dispatcher model configured, falling back to keyword dispatch")
            return await self._keyword_dispatch(prompt, model_names)

        context = self._build_dispatch_context(benchmarks)

        dispatch_prompt = f"""You are a model router. Given the user prompt and the available models with their benchmark scores, select the best model.

Available models:
{context}

User prompt: {prompt}

Respond ONLY with a JSON object in this exact format:
{{"model": "model_name", "reasoning": "brief explanation"}}

Select the model that best matches the user's prompt needs."""

        try:
            response = await self.client.chat(
                model=self.dispatcher_model,
                messages=[{"role": "user", "content": dispatch_prompt}],
                temperature=settings.router_temperature,
                max_tokens=settings.router_max_tokens,
            )

            content = response.get("message", {}).get("content", "")
            result = self._parse_llm_response(content, model_names)

            if result:
                return RoutingResult(
                    selected_model=result["model"],
                    confidence=0.9,
                    reasoning=result["reasoning"],
                )

        except Exception as e:
            logger.warning(f"LLM dispatch failed: {e}, falling back to keyword dispatch")

        return await self._keyword_dispatch(prompt, model_names)

    def _build_dispatch_context(self, benchmarks: list[dict]) -> str:
        """Build context string for LLM dispatcher with benchmark data."""
        context_lines = []
        for bm in benchmarks:
            name = bm.get("ollama_name", "unknown")
            elo = bm.get("elo_rating", "N/A")
            reasoning = bm.get("reasoning_score", "N/A")
            coding = bm.get("coding_score", "N/A")
            context_lines.append(f"- {name}: ELO={elo}, Reasoning={reasoning}, Coding={coding}")
        return "\n".join(context_lines) if context_lines else "No benchmark data available"

    def _parse_llm_response(self, content: str, model_names: list[str]) -> dict | None:
        """Parse LLM response to extract model selection."""
        import json

        try:
            data = json.loads(content)
            model = data.get("model", "")
            reasoning = data.get("reasoning", "")

            # Validate model exists
            if model in model_names:
                return {"model": model, "reasoning": reasoning}

            # Try fuzzy match
            for name in model_names:
                if model.lower() in name.lower() or name.lower() in model.lower():
                    return {"model": name, "reasoning": reasoning}

        except json.JSONDecodeError:
            pass

        # Try extracting from text
        for name in model_names:
            if name in content:
                return {"model": name, "reasoning": "Extracted from response"}

        return None

    # ... existing methods ...

    def _get_model_feedback_scores(self) -> dict[str, float]:
        """Get average feedback score for each model."""
        if not settings.feedback_enabled:
            return {}

        try:
            with get_session() as session:
                from sqlalchemy import func

                # Use SQL aggregation for O(1) database load instead of O(N) full table scan
                results = (
                    session.query(
                        ModelFeedback.model_name, func.avg(ModelFeedback.score).label("avg_score")
                    )
                    .group_by(ModelFeedback.model_name)
                    .all()
                )
                return {name: float(avg_score) for name, avg_score in results}
        except Exception as e:
            logger.warning(f"Failed to fetch feedback scores: {e}")
            return {}

    async def _keyword_dispatch(
        self, prompt: str, model_names: list[str], request_obj: Any = None
    ) -> RoutingResult:
        profiles = await self._get_all_profiles()
        benchmarks = get_benchmarks_for_models_with_external(model_names)
        feedback_scores = self._get_model_feedback_scores()

        if not profiles and not benchmarks:
            logger.warning("No profiles or benchmarks found, selecting first available model")
            return RoutingResult(
                selected_model=model_names[0],
                confidence=0.0,
                reasoning="No profiling data available, defaulting to first model",
            )

        # Check prompt analysis cache first
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        now = time.monotonic()

        if prompt_hash in _PROMPT_ANALYSIS_CACHE:
            cached_time, cached_analysis = _PROMPT_ANALYSIS_CACHE[prompt_hash]
            if (now - cached_time) < _PROMPT_ANALYSIS_CACHE_TTL:
                analysis = cached_analysis
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Using cached prompt analysis (age: {now - cached_time:.1f}s)")
            else:
                # Cache expired
                analysis = self._analyze_prompt(prompt, request_obj)
                _PROMPT_ANALYSIS_CACHE[prompt_hash] = (now, analysis)
        else:
            analysis = self._analyze_prompt(prompt, request_obj)
            _PROMPT_ANALYSIS_CACHE[prompt_hash] = (now, analysis)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Prompt analysis: {analysis}")

        # Gather model selection frequencies for diversity penalty if cache enabled
        model_frequencies: dict[str, float] = {}
        if self.semantic_cache:
            freq_tasks = [self.semantic_cache.get_model_frequency(m) for m in model_names]
            freq_results = await asyncio.gather(*freq_tasks)
            model_frequencies = dict(zip(model_names, freq_results, strict=False))

        scores = self._calculate_combined_scores(
            profiles, benchmarks, analysis, model_names, feedback_scores, model_frequencies
        )

        # Log all scores for debugging
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top5 = [
            (
                m,
                round(s["score"], 2),
                round(s.get("base_score", 0), 2),
                s.get("coding", 0),
                s.get("creativity", 0),
            )
            for m, s in sorted_scores[:8]
        ]
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Model scores (top 8): {top5}")
            logger.info("  (format: model, total_score, base_score, coding, creativity)")

        # Determine dominant category (threshold > 0.5) - but exclude complexity!
        task_categories = {k: v for k, v in analysis.items() if k not in ("complexity", "vision", "tools")}
        top_category = max(task_categories.items(), key=lambda x: x[1])
        dominant_category = top_category[0] if top_category[1] > 0.5 else None

        # Determine modality from request object
        modality = Modality.TEXT
        if request_obj:
            from router.modality import ModalityDetector
            modality = ModalityDetector.from_chat_request(request_obj)

        # Build model profiles dict for modality filtering
        model_profiles = {p["name"]: p for p in profiles}

        # Apply modality-based filtering
        modality_candidates = get_models_for_modality(
            model_names, modality, model_profiles
        )
        candidates_filter = set(modality_candidates)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Detected modality: {modality}, filtered {len(model_names)} models to {len(candidates_filter)} candidates")

        if not candidates_filter:
            # Fallback to all models if modality filtering removed everything
            logger.warning(f"Modality filtering ({modality}) removed all models, using fallback")
            candidates_filter = set(model_names)

        # Filter scores dict to only include candidates
        scores = {k: v for k, v in scores.items() if k in candidates_filter}

        if dominant_category:
            # ... existing logic ...
            max_cat_score = max(s[dominant_category] for s in scores.values())
            threshold = max_cat_score * 0.85

            candidates = {m: s for m, s in scores.items() if s[dominant_category] >= threshold}

            best_model_name = max(candidates.items(), key=lambda x: x[1]["score"])[0]
            confidence = candidates[best_model_name]["score"]
            reasoning = f"Selected top {dominant_category} model (score: {candidates[best_model_name][dominant_category]:.2f}) with best overall traits"
        else:
            # Balanced/General task - use overall weighted score
            best_model_name = max(scores.items(), key=lambda x: x[1]["score"])[0]
            confidence = scores[best_model_name]["score"]
            reasoning = self._build_reasoning(analysis, scores[best_model_name])

        # ... rest of method ...
        return RoutingResult(
            selected_model=best_model_name,
            confidence=confidence,
            reasoning=reasoning,
        )

    async def _get_all_profiles(self) -> list[dict]:
        global _profiles_cache, _profiles_cache_time, _profiles_cache_lock
        now = time.monotonic()

        # Fast path: check cache without lock
        if _profiles_cache is not None and (now - _profiles_cache_time) < _PROFILE_CACHE_TTL:
            return _profiles_cache

        # Slow path: acquire lock and check again
        async with _profiles_cache_lock:
            # Double-check after acquiring lock
            if _profiles_cache is not None and (now - _profiles_cache_time) < _PROFILE_CACHE_TTL:
                return _profiles_cache

            # Cache miss, fetch from database
            with get_session() as session:
                profiles = session.execute(select(ModelProfile)).scalars().all()
                _profiles_cache = [
                    {
                        "name": p.name,
                        "reasoning": p.reasoning,
                        "coding": p.coding,
                        "creativity": p.creativity,
                        "speed": p.speed,
                        "avg_response_time_ms": p.avg_response_time_ms,
                        "first_seen": p.first_seen,
                    }
                    for p in profiles
                ]
                _profiles_cache_time = now
                return _profiles_cache

    def _calculate_combined_scores(
        self,
        profiles: list[dict],
        benchmarks: list[dict],
        analysis: dict[str, float],
        model_names: list[str],
        feedback_scores: dict[str, float] | None = None,
        model_frequencies: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        if model_frequencies is None:
            model_frequencies = {}
        if feedback_scores is None:
            feedback_scores = {}
        scores: dict[str, dict[str, float]] = {}

        profile_map = {p["name"]: p for p in profiles}
        benchmark_map = {b["ollama_name"]: b for b in benchmarks}

        # Quality vs Speed Trade-off
        quality_pref = settings.quality_preference
        speed_weight = 1.0 - quality_pref
        quality_weight = (
            quality_pref + SCORING_CONFIG["quality_preference_boost"]
        )  # Boost quality signals if preferred

        # Pre-process benchmark names for faster matching
        processed_benchmarks = {}
        for bm_name, bm in benchmark_map.items():
            bm_base = bm_name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")
            # Store variations for matching
            processed_benchmarks[bm_name] = {
                "benchmark": bm,
                "base": bm_base,
                "parts": {bm_base, bm_base.split("2")[0] if "2" in bm_base else bm_base}
            }

        normalized_benchmark_map = {}
        for name in model_names:
            # Extract base name - handle versions and quantizations
            base = name.split(":")[0].lower().replace("-", "").replace("_", "").replace(".", "")
            base_variations = [base, base.split("2")[0] if "2" in base else base]

            best_match = None
            best_score = 0.0

            # Check for exact matches first
            for bm_data in processed_benchmarks.values():
                bm_base = bm_data["base"]
                if base == bm_base:
                    best_match = bm_data["benchmark"]
                    best_score = 100.0
                    break

            # If no exact match, check partial matches
            if not best_match:
                for bm_data in processed_benchmarks.values():
                    bm_base = bm_data["base"]
                    bm_parts = bm_data["parts"]

                    # Check if any variation matches
                    for var in base_variations:
                        if var in bm_base or bm_base in var:
                            score = len(var) / max(len(var), len(bm_base), 1)
                            if score > best_score:
                                best_match = bm_data["benchmark"]
                                best_score = score
                                break

                    # Check for partial word matches
                    if not best_match:
                        for var in base_variations:
                            for part in [var, var[:4], var[:6]]:
                                if len(part) > 2 and part in bm_parts:
                                    best_match = bm_data["benchmark"]
                                    best_score = 0.5
                                    break
                            if best_match:
                                break

            if best_match:
                normalized_benchmark_map[name] = best_match

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Benchmark matching: {len(normalized_benchmark_map)}/{len(model_names)} models matched"
            )

            # Log benchmark match details for each model
            for name in model_names:
                bench_match = normalized_benchmark_map.get(name)
                if bench_match:
                    logger.info(
                        f"  {name} -> benchmark: reasoning={bench_match.get('reasoning_score')}, coding={bench_match.get('coding_score')}, elo={bench_match.get('elo_rating')}"
                    )
                else:
                    logger.info(f"  {name} -> NO benchmark match")

        # Build model category affinity based on model name patterns
        model_category_affinity = self._build_model_category_affinity(
            model_names, normalized_benchmark_map
        )

        # Determine dominant category (exclude complexity which is a meta-category)
        task_categories = {
            k: v for k, v in analysis.items() if k not in ("complexity", "vision", "tools")
        }
        if task_categories:
            top_category = max(task_categories.items(), key=lambda x: x[1])
            dominant_category = (
                top_category[0]
                if top_category[1] > SCORING_CONFIG["dominant_category_confidence_threshold"]
                else None
            )
        else:
            dominant_category = None

        for model_name in model_names:
            profile = profile_map.get(model_name)
            benchmark = normalized_benchmark_map.get(model_name)
            affinity = model_category_affinity.get(model_name, {})

            base_score = 0.0

            # Map prompt categories to benchmark/profile fields
            category_map = {
                "reasoning": ("reasoning_score", "reasoning"),
                "coding": ("coding_score", "coding"),
                "creativity": ("creativity", None),  # No benchmark creativity, use profile only
                "factual": ("general_score", "factual"),  # Use general_score for factual
            }

            for category, weight in analysis.items():
                # Skip complexity - it's handled separately as a bonus
                if category == "complexity":
                    continue

                # Signal 1: Precise Benchmarks (MMLU, HumanEval, etc.)
                benchmark_score = 0.0
                if benchmark:
                    bm_field, _ = category_map.get(category, (None, None))
                    benchmark_score = benchmark.get(bm_field, 0.0) or 0.0
                    # Convert 0-100 scale to 0.0-1.0
                    if benchmark_score > 1.0:
                        benchmark_score /= 100.0

                # Signal 2: General Quality (ELO / Arena Score)
                elo_signal = 0.0
                if benchmark and benchmark.get("elo_rating"):
                    raw_elo = benchmark["elo_rating"]
                    if raw_elo > 200:  # True ELO
                        elo_signal = max(min((raw_elo - 1000) / 800, 1.5), 0.0)
                    else:  # 0-100 Score
                        elo_signal = raw_elo / 100.0

                # Signal 3: Name-based Inference (Fallback)
                inference_score = affinity.get(category, 0.0)

                # Signal 4: Profile Scores (Runtime profiling data)
                profile_score = 0.0
                if profile:
                    _, profile_field = category_map.get(category, (None, None))
                    if profile_field:
                        profile_score = profile.get(profile_field, 0.0) or 0.0

                # Weighted combination of signals
                combined_cat_score = (
                    (benchmark_score * SCORING_CONFIG["benchmark_weight"] * quality_weight)
                    + (elo_signal * SCORING_CONFIG["elo_weight"] * quality_weight)
                    + (inference_score * SCORING_CONFIG["name_inference_weight"] * quality_weight)
                    + (
                        profile_score * SCORING_CONFIG["profile_weight"] * quality_weight
                    )  # Profile is more reliable than name inference
                )

                # If this is the dominant category, apply the 20x Category-First boost
                # BUT only if we have actual data (benchmark, ELO, or profile)
                has_actual_data = benchmark_score > 0 or elo_signal > 0 or profile_score > 0

                # Check if model meets minimum size for this category + complexity
                complexity_bucket = self._get_complexity_bucket(analysis.get("complexity", 0.0))
                min_size = CATEGORY_MIN_SIZES.get(category, {}).get(complexity_bucket, 0)
                params = self._extract_parameter_count(model_name)
                has_adequate_size = params is not None and params >= min_size

                if (
                    category == dominant_category
                    and combined_cat_score > SCORING_CONFIG["dominant_category_threshold"]
                ):
                    if has_actual_data:
                        combined_cat_score *= SCORING_CONFIG[
                            "dominant_category_boost_with_data"
                        ]  # Strong boost with data
                    elif has_adequate_size:
                        combined_cat_score *= (
                            SCORING_CONFIG[
                                "dominant_category_boost_with_size"
                            ]  # Moderate boost with adequate size but no benchmark
                        )
                    else:
                        combined_cat_score *= SCORING_CONFIG[
                            "dominant_category_boost_default"
                        ]  # Weak boost without data or size

                if weight > 0:
                    base_score += combined_cat_score * weight
                else:
                    base_score += combined_cat_score * SCORING_CONFIG["negative_weight_multiplier"]

            # Bonus factors (speed, size, newness, complexity, feedback)
            bonus_score = 0.0
            params = self._extract_parameter_count(model_name)
            complexity = analysis.get("complexity", 0.0)
            has_benchmark = normalized_benchmark_map.get(model_name) is not None

            # Feedback Bonus
            fb_score = feedback_scores.get(model_name, 0.0)
            if fb_score != 0:
                bonus_score += (
                    fb_score * SCORING_CONFIG["feedback_bonus_multiplier"]
                )  # Significant impact for user preference

            # Bonus for having benchmark data (prefer data-driven over name-based)
            if has_benchmark:
                bonus_score += SCORING_CONFIG["has_benchmark_bonus"] * quality_weight

            # === SIZE/CAPACITY SCORING ===
            # Only apply size bonus when we have benchmark data OR high complexity
            # This prevents size from dominating when we only have runtime profiles (speed-based routing)
            size_score = 0.0
            if params:
                if params >= 30:
                    size_score = (
                        SCORING_CONFIG["size_score_very_large"] * quality_weight
                    )  # Strong preference for very large models
                elif params >= 14:
                    size_score = (
                        SCORING_CONFIG["size_score_large"] * quality_weight
                    )  # Good preference for large models
                elif params >= 7:
                    size_score = (
                        SCORING_CONFIG["size_score_medium"] * quality_weight
                    )  # Medium preference for mid-size models
                elif params >= 3:
                    size_score = 0.0  # Neutral for small models
                else:
                    # Penalize tiny models less if we prefer speed (low quality_pref)
                    penalty = (
                        SCORING_CONFIG["size_score_tiny_penalty_high"]
                        if quality_pref >= SCORING_CONFIG["quality_preference_threshold"]
                        else SCORING_CONFIG["size_score_tiny_penalty_low"]
                    )
                    size_score = penalty

            # Apply size score only for moderate+ complexity tasks
            if complexity >= SCORING_CONFIG["complexity_moderate_threshold"]:
                bonus_score += size_score * SCORING_CONFIG["size_score_multiplier"]

            # Complexity-Size Matching Logic (enhanced)
            if complexity >= SCORING_CONFIG["complexity_high_threshold"]:
                # Very high complexity: STRONGLY prefer larger models
                if params and params >= 30:
                    bonus_score += (
                        SCORING_CONFIG["high_complexity_very_large_bonus"] * quality_weight
                    )
                elif params and params >= 14:
                    bonus_score += SCORING_CONFIG["high_complexity_large_bonus"] * quality_weight
                elif params and params >= 7:
                    bonus_score += SCORING_CONFIG["high_complexity_medium_bonus"] * quality_weight
                elif params and params < 4:
                    bonus_score += SCORING_CONFIG["high_complexity_small_penalty"] * quality_weight
            elif complexity >= SCORING_CONFIG["complexity_moderate_threshold"]:
                # Moderate complexity: Prefer larger models
                if params and params >= 30:
                    bonus_score += (
                        SCORING_CONFIG["moderate_complexity_very_large_bonus"] * quality_weight
                    )
                elif params and params >= 14:
                    bonus_score += (
                        SCORING_CONFIG["moderate_complexity_large_bonus"] * quality_weight
                    )
                elif params and params >= 7:
                    bonus_score += (
                        SCORING_CONFIG["moderate_complexity_medium_bonus"] * quality_weight
                    )
                elif params and params < 4:
                    bonus_score += (
                        SCORING_CONFIG["moderate_complexity_small_penalty"] * quality_weight
                    )
            elif complexity < SCORING_CONFIG["complexity_low_threshold"]:
                # Low complexity: Strong preference for small/fast models
                if params and params <= 4:
                    bonus_score += (
                        SCORING_CONFIG["low_complexity_tiny_bonus"] * speed_weight
                    )  # Strong bonus for tiny models
                elif params and params <= 7:
                    bonus_score += (
                        SCORING_CONFIG["low_complexity_small_bonus"] * speed_weight
                    )  # Good bonus for small models
                elif params and params >= 14:
                    bonus_score += (
                        SCORING_CONFIG["low_complexity_large_penalty"] * speed_weight
                    )  # Penalize large models
                elif params and params >= 30:
                    bonus_score += (
                        SCORING_CONFIG["low_complexity_very_large_penalty"] * speed_weight
                    )  # Strong penalty for very large

            # === CATEGORY-AWARE MINIMUM SIZE REQUIREMENTS ===
            # Apply severe penalty if model is below minimum size for category + complexity
            if dominant_category and params is not None:
                complexity_bucket = self._get_complexity_bucket(complexity)
                min_size = CATEGORY_MIN_SIZES.get(dominant_category, {}).get(complexity_bucket, 0)

                if params < min_size:
                    # Calculate deficit - scales with how far below minimum
                    size_deficit = min_size - params
                    min_size_penalty = -10.0 * size_deficit
                    bonus_score += min_size_penalty
                    logger.debug(
                        f"Min size penalty for {model_name}: {min_size_penalty} (params={params}, min={min_size} for {dominant_category}/{complexity_bucket})"
                    )

            if profile:
                # Speed bonus (only for simple tasks OR if speed is preferred)
                # Boost speed importance if quality_pref is low
                speed_importance = speed_weight * 2.0

                if (complexity < 0.4 or speed_importance > 1.0) and profile.get(
                    "avg_response_time_ms", 0
                ) > 0:
                    # More sensitive time factor (baseline 10s instead of 60s)
                    time_factor = 1.0 - min(profile["avg_response_time_ms"] / 10000.0, 0.8)
                    bonus_score += time_factor * 0.2 * speed_importance

                # Newness bonus
                if settings.prefer_newer_models and profile.get("first_seen"):
                    newness = self._calculate_newness_score(profile["first_seen"])
                    bonus_score += newness * 0.05

            # Diversity Penalty: Reduce score if model has been selected too frequently recently
            # This prevents one model from dominating and encourages exploration
            model_frequency = model_frequencies.get(model_name, 0.0)
            diversity_penalty = 0.0

            if model_frequency > 0.5:
                # Apply multiplicative penalty to base_score
                # freq 0.5 -> 0.65x (35% reduction)
                # freq 0.8 -> 0.44x (56% reduction)
                # freq 1.0 -> 0.30x (70% reduction)
                reduction = (model_frequency - 0.5) * 1.4  # scales from 0 to 0.7
                multiplier = max(0.3, 1.0 - reduction)
                base_score = base_score * multiplier
                # For logging, compute an approximate additive equivalent
                diversity_penalty = (
                    -base_score * (1.0 - multiplier) / (multiplier if multiplier > 0 else 1)
                )
            elif model_frequency > 0:
                # Small frequency gets tiny penalty to nudge exploration
                diversity_penalty = -model_frequency * 0.2
                # No multiplicative penalty for low frequency

            total_score = base_score + bonus_score + diversity_penalty

            # Use the actual scores used for routing in the debug log
            scores[model_name] = {
                "score": total_score,
                "base_score": base_score,
                "bonus": bonus_score,
                "diversity": diversity_penalty,
                "reasoning": affinity.get("reasoning", 0),
                "coding": affinity.get("coding", 0),
                "creativity": affinity.get("creativity", 0),
                "factual": affinity.get("factual", 0),
            }

        # Debug: log all model scores
        logger.info(
            f"Actual routing affinity scores: {[(m, s.get('reasoning', 0), s.get('coding', 0), s.get('creativity', 0)) for m, s in scores.items()]}"
        )

        return scores

    def _extract_parameter_count(self, model_name: str) -> float | None:
        """Extract parameter count in billions from model name."""
        name_lower = model_name.lower()

        # 1. Direct Regex (e.g., "7b", "0.5b", "1.5b") - check BEFORE colon
        # First, get the part before the colon (the model tag)
        model_tag = name_lower.split(":")[0] if ":" in name_lower else name_lower

        match = re.search(r"(\d+(\.\d+)?)\s*b", model_tag)
        if match:
            return float(match.group(1))

        # 2. Known model size mappings for Ollama names
        size_map = {
            "mini": 3.8,  # Phi-3-mini
            "small": 7.0,  # Mistral-small, etc
            "medium": 14.0,  # Phi-3-medium, etc
            "large": 70.0,
            "nemo": 12.0,  # Mistral-Nemo
            "r1": 14.0,  # DeepSeek-R1 (common Ollama default is 14B)
            "gemma3": 1.0,  # Gemma 3 is 1B
            "gemma2": 9.0,  # Gemma 2 is 9B
        }

        for key, size in size_map.items():
            if key in name_lower:
                return size

        # 3. Handle names like "llama3.1" (default is 8b)
        if "llama3" in name_lower or "llama3.1" in name_lower or "llama3.2" in name_lower:
            if ":1b" in name_lower or "1b" in model_tag:
                return 1.0
            if ":3b" in name_lower or "3b" in model_tag:
                return 3.0
            if ":8b" in name_lower or "8b" in model_tag:
                return 8.0
            return 8.0  # default

        if "qwen2.5" in name_lower:
            if ":0.5b" in name_lower or "0.5b" in model_tag:
                return 0.5
            if ":1.5b" in name_lower or "1.5b" in model_tag:
                return 1.5
            if ":7b" in name_lower or "7b" in model_tag:
                return 7.0
            if ":14b" in name_lower or "14b" in model_tag:
                return 14.0
            if ":32b" in name_lower or "32b" in model_tag:
                return 32.0
            if ":72b" in name_lower or "72b" in model_tag:
                return 72.0

        return None

    def _get_complexity_bucket(self, complexity: float) -> str:
        """Determine complexity bucket based on complexity score."""
        if complexity < 0.2:
            return "simple"
        elif complexity < 0.5:
            return "medium"
        else:
            return "hard"

    def _calculate_size_score(self, params: float | None) -> float:
        """Calculate score based on model size (smaller is better)."""
        if params is None:
            return 0.5  # Neutral score if unknown

        # Logarithmic-ish scaling:
        # < 3B -> 1.0
        # 7-8B -> 0.8
        # 13-14B -> 0.6
        # 30B+ -> 0.4
        if params <= 3:
            return 1.0
        elif params <= 8:
            return 0.8
        elif params <= 14:
            return 0.6
        elif params <= 35:
            return 0.4
        else:
            return 0.2

    def _calculate_newness_score(self, first_seen) -> float:
        """Calculate score based on how new the model is to the system."""
        if not first_seen:
            return 0.0

        # Handle both timezone-aware and naive datetimes
        if isinstance(first_seen, datetime):
            if first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=UTC)

        now = datetime.now(UTC)
        age = now - first_seen
        days_old = age.days

        # New models (< 1 day) get boost
        if days_old < 1:
            return 1.0
        elif days_old < 7:
            return 0.8
        elif days_old < 30:
            return 0.5
        else:
            return 0.0

    def _analyze_prompt(self, prompt: str, request_obj: Any = None) -> dict[str, float]:
        prompt_lower = prompt.lower()

        analysis: dict[str, float] = {
            "reasoning": 0.0,
            "coding": 0.0,
            "creativity": 0.0,
            "factual": 0.0,
            "complexity": 0.0,
            "vision": 0.0,  # New
            "tools": 0.0,  # New
        }

        # New: Inspect request object for capabilities
        if request_obj:
            # 1. Vision Detection
            if hasattr(request_obj, "messages"):
                for msg in request_obj.messages:
                    if isinstance(msg.content, list):
                        for part in msg.content:
                            if isinstance(part, dict) and part.get("type") == "image_url":
                                analysis["vision"] = 1.0
                                break

            # 2. Tool Detection
            if hasattr(request_obj, "tools") and request_obj.tools:
                analysis["tools"] = 1.0
                analysis["complexity"] += 0.3  # Tools imply complexity
                analysis["coding"] += 0.2  # Tools often relate to coding/structured output

            # 3. JSON Mode Detection
            if hasattr(request_obj, "response_format") and request_obj.response_format:
                if request_obj.response_format.get("type") == "json_object":
                    analysis["coding"] += 0.3  # JSON mode is coding-adjacent
                    analysis["complexity"] += 0.1

        # ... existing logic ...
        reasoning_keywords = [
            "calculate",
            "logic",
            "solve",
            "reason",
            "prove",
            "math",
            "sequence",
            "pattern",
            "if then",
            "therefore",
            "because",
            "derive",
            "speed",
            "velocity",
            "distance",
            "how much",
            "how many",
            "result",
        ]
        coding_keywords = [
            "code",
            "function",
            "implement",
            "algorithm",
            "program",
            "python",
            "javascript",
            "java",
            "sql",
            "debug",
            "api",
            "class",
            "def ",
            "return",
            "import",
            "write code",
            "bug",
            "fix",
            "script",
            "json",
            "xml",
            "yaml",
            "parse",
            "schema",  # Added data formats
        ]
        # ... rest of keyword analysis ...
        creative_keywords = [
            "story",
            "write",
            "poem",
            "creative",
            "imagine",
            "describe",
            "invent",
            "fantasy",
            "narrative",
            "character",
            "scene",
            "song",
            "haiku",
            "lyrics",
            "joke",
            "humor",
        ]
        factual_keywords = [
            "what is",
            "who is",
            "when did",
            "where is",
            "define",
            "explain",
            "fact",
            "history",
            "capital",
            "year",
            "date",
            "list",
            "tell me about",
            "summary",
            "summarize",
        ]

        for kw in reasoning_keywords:
            if kw in prompt_lower:
                analysis["reasoning"] += 0.3

        for kw in coding_keywords:
            if kw in prompt_lower:
                analysis["coding"] += 0.4

        for kw in creative_keywords:
            if kw in prompt_lower:
                analysis["creativity"] += 0.35

        for kw in factual_keywords:
            if kw in prompt_lower:
                analysis["factual"] += 0.3

        # Complexity Detection (Enhanced for Difficulty Prediction)
        # Length heuristics
        if len(prompt) > 500:
            analysis["complexity"] += 0.2
        if len(prompt) > 1500:
            analysis["complexity"] += 0.3

        # Structure heuristics
        if prompt.count("?") > 2:
            analysis["complexity"] += 0.1
        if prompt.count("\n") > 5:
            analysis["complexity"] += 0.1

        # Keyword-based complexity
        complexity_keywords = [
            "complex",
            "expert",
            "detailed",
            "comprehensive",
            "optimized",
            "architecture",
            "distributed",
            "performance",
            "scalable",
            "deep dive",
            "advanced",
            "professional",
            "senior",
            "production-ready",
            "implement",
            "algorithm",
            "data structure",
            "tree",
            "graph",
            "recursive",
            "unit test",
            "type hint",
            "generics",
            "async",
            "concurrent",
            "nuance",
            "subtle",
            "imply",
            "hidden meaning",
            "step-by-step",
            "reasoning chain",
        ]

        for kw in complexity_keywords:
            if kw in prompt_lower:
                analysis["complexity"] += 0.15  # Incremental boost

        # Additional complexity for coding tasks with multiple requirements
        if analysis["coding"] > 0.5:
            # Count coding-related keywords to gauge complexity
            coding_complexity_indicators = [
                "with",
                "include",
                "and",
                "also",
                "plus",
                "additionally",
                "operations",
                "methods",
                "functions",
                "classes",
                "interface",
                "inheritance",
                "generic",
                "template",
                "exception",
                "handle",
                "error",
                "security",
                "thread",
            ]
            indicator_count = sum(1 for ind in coding_complexity_indicators if ind in prompt_lower)
            if indicator_count >= 3:
                analysis["complexity"] += 0.3
            elif indicator_count >= 2:
                analysis["complexity"] += 0.15

        # Cap complexity at 1.0
        analysis["complexity"] = min(analysis["complexity"], 1.0)

        code_indicators = ["```", "def ", "function ", "const ", "let ", "var ", "class "]
        for ind in code_indicators:
            if ind in prompt:
                analysis["coding"] = 1.0
                break

        if max(analysis.values()) == 0.0:
            analysis["factual"] = 0.5

        return analysis

    def _build_reasoning(
        self,
        analysis: dict[str, float],
        scores: dict[str, float],
    ) -> str:
        top_category = max(analysis.items(), key=lambda x: x[1])
        category_name = top_category[0] if top_category[1] > 0 else "balanced"

        return f"Matched to {category_name} profile (score: {scores['score']:.2f})"

    def _build_model_category_affinity(
        self,
        model_names: list[str],
        benchmark_map: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        """Infers category affinity from model names when benchmarks are missing."""
        affinity: dict[str, dict[str, float]] = {}

        for name in model_names:
            name_lower = name.lower()
            # Start with a "Generalist Floor" - every model has some base capability
            scores = {"coding": 0.1, "reasoning": 0.1, "creativity": 0.1, "factual": 0.1}

            # Specialist Boosts: Only for models that explicitly mention these in their name
            if any(
                kw in name_lower
                for kw in ["coder", "starcoder", "codegeex", "codellama", "deepseek-coder"]
            ):
                scores["coding"] = 0.9
                scores["reasoning"] = 0.5  # Coders are usually good at logic too

            if any(kw in name_lower for kw in ["r1", "math", "logic", "thought", "reasoner"]):
                scores["reasoning"] = 1.0

            if any(kw in name_lower for kw in ["dolphin", "uncensored", "creative", "writer"]):
                scores["creativity"] = 0.8

            # Generalists (Llama, Mistral, Gemma, Phi) are good at everything,
            # especially factual and creative tasks
            if any(kw in name_lower for kw in ["llama", "mistral", "gemma", "phi", "qwen"]):
                scores["factual"] = 0.7
                scores["creativity"] = 0.6 if scores["creativity"] < 0.6 else scores["creativity"]

            affinity[name] = scores

        return affinity

    async def log_decision(
        self,
        prompt: str,
        selected: str,
        confidence: float,
        reasoning: str,
        response_id: str | None = None,
    ) -> None:
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

            with get_session() as session:
                decision = RoutingDecision(
                    prompt_hash=prompt_hash,
                    selected_model=selected,
                    confidence=confidence,
                    reasoning=reasoning,
                    response_id=response_id,
                )
                session.add(decision)
                session.commit()

            # Also cache the routing decision for future similar prompts
            result = RoutingResult(
                selected_model=selected,
                confidence=confidence,
                reasoning=reasoning,
            )
            if self.semantic_cache:
                await self.semantic_cache.set(prompt, result)

        except Exception as e:
            logger.debug(f"Failed to log routing decision: {e}")
