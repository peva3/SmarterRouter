"""
Persistent disk cache manager for Semantic Cache V2.

Provides SQLite-based persistence for routing decisions, LLM responses,
and embeddings across restarts. Integrates with SemanticCache to provide
transparent save/load operations.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select

from router.database import get_session
from router.models import EmbeddingCache, ResponseCache, RoutingCache

if TYPE_CHECKING:
    from router.router import RoutingResult

logger = logging.getLogger(__name__)


class PersistentCacheManager:
    """Manages persistent disk storage for cache data."""

    def __init__(self, enabled: bool = True, max_age_days: int = 7):
        """
        Initialize persistent cache manager.

        Args:
            enabled: Whether persistence is enabled
            max_age_days: Maximum age in days to keep cache entries
        """
        self.enabled = enabled
        self.max_age_days = max_age_days
        self._save_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()
        self._loaded = False

    async def load_routing_cache(
        self,
    ) -> dict[str, tuple[RoutingResult, float, list[float] | None, float | None, int]]:
        """
        Load routing cache from database into memory.

        Returns:
            Dictionary mapping cache_key to (result, timestamp, embedding, magnitude, access_count)
        """
        if not self.enabled:
            return {}

        async with self._load_lock:
            try:
                cache_data = {}
                cutoff_time = datetime.now(UTC) - timedelta(days=self.max_age_days)

                with get_session() as session:
                    # Load non-expired routing cache entries
                    stmt = (
                        select(RoutingCache)
                        .where(
                            (
                                RoutingCache.expires_at.is_(None)
                                | (RoutingCache.expires_at > datetime.now(UTC))
                            )
                            & (RoutingCache.created_at > cutoff_time)
                        )
                        .order_by(
                            RoutingCache.access_count.desc(), RoutingCache.last_accessed.desc()
                        )
                        .limit(1000)  # Load max 1000 entries
                    )
                    results = session.execute(stmt).scalars().all()

                    for entry in results:
                        try:
                            # Convert embedding JSON to list[float]
                            embedding = entry.embedding
                            magnitude = entry.embedding_magnitude

                            # Create RoutingResult
                            result = RoutingResult(
                                selected_model=entry.selected_model,
                                confidence=entry.confidence,
                                reasoning=entry.reasoning or "",
                            )

                            # Use last_accessed as timestamp for LRU
                            timestamp = entry.last_accessed.timestamp()
                            cache_data[entry.cache_key] = (
                                result,
                                timestamp,
                                embedding,
                                magnitude,
                                entry.access_count or 1,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to load routing cache entry {entry.id}: {e}")
                            continue

                    logger.info(f"Loaded {len(cache_data)} routing cache entries from database")
                    return cache_data

            except Exception as e:
                logger.error(f"Failed to load routing cache: {e}")
                return {}

    async def load_response_cache(self) -> dict[tuple, tuple[str, float]]:
        """
        Load response cache from database into memory.

        Returns:
            Dictionary mapping cache_key to (response_text, timestamp)
        """
        if not self.enabled:
            return {}

        async with self._load_lock:
            try:
                cache_data = {}
                cutoff_time = datetime.now(UTC) - timedelta(days=self.max_age_days)

                with get_session() as session:
                    # Load non-expired response cache entries
                    stmt = (
                        select(ResponseCache)
                        .where(
                            (
                                ResponseCache.expires_at.is_(None)
                                | (ResponseCache.expires_at > datetime.now(UTC))
                            )
                            & (ResponseCache.created_at > cutoff_time)
                        )
                        .order_by(
                            ResponseCache.access_count.desc(), ResponseCache.last_accessed.desc()
                        )
                        .limit(500)  # Load max 500 entries
                    )
                    results = session.execute(stmt).scalars().all()

                    for entry in results:
                        try:
                            # Use the cache_key property
                            cache_key = entry.cache_key
                            timestamp = entry.last_accessed.timestamp()
                            cache_data[cache_key] = (entry.response_text, timestamp)
                        except Exception as e:
                            logger.debug(f"Failed to load response cache entry {entry.id}: {e}")
                            continue

                    logger.info(f"Loaded {len(cache_data)} response cache entries from database")
                    return cache_data

            except Exception as e:
                logger.error(f"Failed to load response cache: {e}")
                return {}

    async def load_embedding_cache(self) -> dict[str, tuple[list[float], float, float]]:
        """
        Load embedding cache from database into memory.

        Returns:
            Dictionary mapping prompt_hash to (embedding, magnitude, timestamp)
        """
        if not self.enabled:
            return {}

        async with self._load_lock:
            try:
                cache_data = {}
                cutoff_time = datetime.now(UTC) - timedelta(days=self.max_age_days)

                with get_session() as session:
                    # Load non-expired embedding cache entries
                    stmt = (
                        select(EmbeddingCache)
                        .where(
                            (
                                EmbeddingCache.expires_at.is_(None)
                                | (EmbeddingCache.expires_at > datetime.now(UTC))
                            )
                            & (EmbeddingCache.created_at > cutoff_time)
                        )
                        .order_by(
                            EmbeddingCache.access_count.desc(), EmbeddingCache.last_accessed.desc()
                        )
                        .limit(2500)  # Load max 2500 embeddings
                    )
                    results = session.execute(stmt).scalars().all()

                    for entry in results:
                        try:
                            embedding = entry.embedding
                            magnitude = entry.magnitude
                            timestamp = (
                                entry.last_accessed.timestamp()
                                if entry.last_accessed
                                else entry.created_at.timestamp()
                            )
                            cache_data[entry.prompt_hash] = (embedding, magnitude, timestamp)
                        except Exception as e:
                            logger.debug(f"Failed to load embedding cache entry {entry.id}: {e}")
                            continue

                    logger.info(f"Loaded {len(cache_data)} embedding cache entries from database")
                    return cache_data

            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")
                return {}

    async def save_routing_entry(
        self,
        cache_key: str,
        result: RoutingResult,
        embedding: list[float] | None = None,
        embedding_magnitude: float | None = None,
        ttl_seconds: int = 3600,
    ) -> bool:
        """
        Save a single routing cache entry to database.

        Args:
            cache_key: Unique cache key (prompt hash)
            result: RoutingResult with decision
            embedding: Optional embedding vector
            embedding_magnitude: Optional embedding magnitude
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if saved successfully
        """
        if not self.enabled:
            return False

        async with self._save_lock:
            try:
                expires_at = None
                if ttl_seconds > 0:
                    expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

                with get_session() as session:
                    # Check if entry exists
                    existing = session.execute(
                        select(RoutingCache).where(RoutingCache.cache_key == cache_key)
                    ).scalar_one_or_none()

                    if existing:
                        # Update existing entry
                        existing.selected_model = result.selected_model
                        existing.confidence = result.confidence
                        existing.reasoning = result.reasoning
                        existing.embedding = embedding
                        existing.embedding_magnitude = embedding_magnitude
                        existing.last_accessed = datetime.now(UTC)
                        existing.access_count = (existing.access_count or 0) + 1
                        existing.expires_at = expires_at
                    else:
                        # Create new entry
                        entry = RoutingCache(
                            cache_key=cache_key,
                            selected_model=result.selected_model,
                            confidence=result.confidence,
                            reasoning=result.reasoning,
                            embedding=embedding,
                            embedding_magnitude=embedding_magnitude,
                            last_accessed=datetime.now(UTC),
                            access_count=1,
                            expires_at=expires_at,
                        )
                        session.add(entry)

                    session.commit()
                    return True

            except Exception as e:
                logger.error(f"Failed to save routing cache entry {cache_key[:8]}: {e}")
                return False

    async def save_response_entry(
        self,
        cache_key: tuple,
        response_text: str,
        ttl_seconds: int = 3600,
    ) -> bool:
        """
        Save a single response cache entry to database.

        Args:
            cache_key: Tuple (model_name, prompt_hash[, param_tuple])
            response_text: LLM response text
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if saved successfully
        """
        if not self.enabled:
            return False

        async with self._save_lock:
            try:
                # Parse cache key tuple
                if len(cache_key) == 2:
                    model_name, prompt_hash = cache_key
                    parameters = None
                else:
                    model_name, prompt_hash, param_tuple = cache_key
                    # Convert tuple back to dict
                    parameters = dict(param_tuple) if param_tuple else None

                expires_at = None
                if ttl_seconds > 0:
                    expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

                with get_session() as session:
                    # Check if entry exists
                    existing = session.execute(
                        select(ResponseCache).where(
                            (ResponseCache.model_name == model_name)
                            & (ResponseCache.prompt_hash == prompt_hash)
                            & (ResponseCache.parameters == parameters)
                        )
                    ).scalar_one_or_none()

                    if existing:
                        # Update existing entry
                        existing.response_text = response_text
                        existing.last_accessed = datetime.now(UTC)
                        existing.access_count = (existing.access_count or 0) + 1
                        existing.expires_at = expires_at
                    else:
                        # Create new entry
                        entry = ResponseCache(
                            model_name=model_name,
                            prompt_hash=prompt_hash,
                            parameters=parameters,
                            response_text=response_text,
                            last_accessed=datetime.now(UTC),
                            access_count=1,
                            expires_at=expires_at,
                        )
                        session.add(entry)

                    session.commit()
                    return True

            except Exception as e:
                logger.error(f"Failed to save response cache entry: {e}")
                return False

    async def save_embedding_entry(
        self,
        prompt_hash: str,
        embedding: list[float],
        magnitude: float,
        ttl_seconds: int = 86400,  # 24 hours default
    ) -> bool:
        """
        Save a single embedding cache entry to database.

        Args:
            prompt_hash: Hash of the prompt
            embedding: Embedding vector
            magnitude: Magnitude of embedding vector
            ttl_seconds: Time-to-live in seconds (24h default)

        Returns:
            True if saved successfully
        """
        if not self.enabled:
            return False

        async with self._save_lock:
            try:
                expires_at = None
                if ttl_seconds > 0:
                    expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

                with get_session() as session:
                    # Check if entry exists
                    existing = session.execute(
                        select(EmbeddingCache).where(EmbeddingCache.prompt_hash == prompt_hash)
                    ).scalar_one_or_none()

                    if existing:
                        # Update existing entry
                        existing.embedding = embedding
                        existing.magnitude = magnitude
                        existing.last_accessed = datetime.now(UTC)
                        existing.access_count = (existing.access_count or 0) + 1
                        existing.expires_at = expires_at
                    else:
                        # Create new entry
                        entry = EmbeddingCache(
                            prompt_hash=prompt_hash,
                            embedding=embedding,
                            magnitude=magnitude,
                            last_accessed=datetime.now(UTC),
                            access_count=1,
                            expires_at=expires_at,
                        )
                        session.add(entry)

                    session.commit()
                    return True

            except Exception as e:
                logger.error(f"Failed to save embedding cache entry {prompt_hash[:8]}: {e}")
                return False

    async def delete_expired_entries(self) -> dict[str, int]:
        """
        Delete expired cache entries from database.

        Returns:
            Dictionary with counts of deleted entries per cache type
        """
        if not self.enabled:
            return {"routing": 0, "response": 0, "embedding": 0}

        try:
            counts = {"routing": 0, "response": 0, "embedding": 0}

            with get_session() as session:
                now_dt = datetime.now(UTC)

                # Bulk delete expired routing cache entries
                result = session.execute(
                    delete(RoutingCache).where(
                        RoutingCache.expires_at.is_not(None), RoutingCache.expires_at <= now_dt
                    )
                )
                counts["routing"] = result.rowcount or 0

                # Bulk delete expired response cache entries
                result = session.execute(
                    delete(ResponseCache).where(
                        ResponseCache.expires_at.is_not(None), ResponseCache.expires_at <= now_dt
                    )
                )
                counts["response"] = result.rowcount or 0

                # Bulk delete expired embedding cache entries
                result = session.execute(
                    delete(EmbeddingCache).where(
                        EmbeddingCache.expires_at.is_not(None), EmbeddingCache.expires_at <= now_dt
                    )
                )
                counts["embedding"] = result.rowcount or 0

                session.commit()
                logger.info(
                    f"Cleaned up expired cache entries: "
                    f"{counts['routing']} routing, {counts['response']} response, {counts['embedding']} embedding"
                )
                return counts

        except Exception as e:
            logger.error(f"Failed to delete expired cache entries: {e}")
            return {"routing": 0, "response": 0, "embedding": 0}

    async def clear_all(self) -> bool:
        """Clear all cache entries from database."""
        if not self.enabled:
            return False

        try:
            with get_session() as session:
                session.query(RoutingCache).delete()
                session.query(ResponseCache).delete()
                session.query(EmbeddingCache).delete()
                session.commit()
                logger.info("Cleared all persistent cache entries")
                return True
        except Exception as e:
            logger.error(f"Failed to clear persistent cache: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about persistent cache."""
        if not self.enabled:
            return {"enabled": False, "routing": 0, "response": 0, "embedding": 0}

        try:
            with get_session() as session:
                now_dt = datetime.now(UTC)
                active = RoutingCache.expires_at.is_(None) | (RoutingCache.expires_at > now_dt)
                routing_count = session.scalar(
                    select(func.count()).select_from(RoutingCache).where(active)
                )

                active = ResponseCache.expires_at.is_(None) | (ResponseCache.expires_at > now_dt)
                response_count = session.scalar(
                    select(func.count()).select_from(ResponseCache).where(active)
                )

                active = EmbeddingCache.expires_at.is_(None) | (EmbeddingCache.expires_at > now_dt)
                embedding_count = session.scalar(
                    select(func.count()).select_from(EmbeddingCache).where(active)
                )

                return {
                    "enabled": True,
                    "routing": routing_count,
                    "response": response_count,
                    "embedding": embedding_count,
                    "max_age_days": self.max_age_days,
                }
        except Exception as e:
            logger.error(f"Failed to get persistent cache stats: {e}")
            return {"enabled": False, "error": str(e)}
