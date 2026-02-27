"""
External Provider Database Module

Queries benchmark data from provider.db for external/cloud models
(OpenAI, Anthropic, Google, etc.) that are not running locally via Ollama.

The provider.db is built by smarterrouter-provider project and contains
benchmark scores for 400+ models from OpenRouter.
"""

import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from router.cache import get_cache
from router.config import settings
from router.exceptions import RouterDatabaseError

logger = logging.getLogger(__name__)


# Alias for backward compatibility
ProviderDBError = RouterDatabaseError

_PROVIDER_DB_CACHE_TTL = 60.0
# Unified cache instance
_provider_cache = get_cache("provider_db", default_ttl=_PROVIDER_DB_CACHE_TTL)


class ProviderDB:
    """Interface to provider.db for external model benchmarks."""

    def __init__(self, db_path: str | None = None):
        path = db_path or settings.provider_db_path

        # Allow bypassing validation for tests
        if os.environ.get("ROUTER_TEST_MODE"):
            self.db_path = Path(path)
        else:
            # Security: Validate path is within allowed directory (prevent path traversal)
            resolved = Path(path).resolve()
            allowed_parents = [
                Path("/app/hubrouter/data").resolve(),
                Path("./data").resolve(),
                Path.cwd() / "data",
            ]
            if not any(
                resolved.is_relative_to(parent) for parent in allowed_parents if parent.exists()
            ):
                raise ProviderDBError(
                    f"provider.db path {path} is outside allowed data directories"
                )
            self.db_path = resolved
        self._conn: sqlite3.Connection | None = None

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def is_available(self) -> bool:
        """Check if provider.db is available."""
        return self.db_path.exists()

    def get_stats(self) -> dict[str, Any]:
        """Get provider.db statistics."""
        if not self.is_available():
            return {"available": False, "total_models": 0}

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as total FROM model_benchmarks WHERE archived = 0")
                total = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) as total FROM model_benchmarks WHERE archived = 1")
                archived = cursor.fetchone()[0]

                cursor.execute("SELECT value FROM metadata WHERE key = 'last_build'")
                last_build = cursor.fetchone()[0] if cursor.fetchone() else None

                return {
                    "available": True,
                    "total_models": total,
                    "archived_models": archived,
                    "last_build": last_build,
                }
        except Exception as e:
            logger.warning(f"Failed to get provider.db stats: {e}")
            return {"available": False, "error": str(e)}

    def get_benchmark(self, model_id: str) -> dict[str, Any] | None:
        """Get benchmark for a single model."""
        if not self.is_available():
            return None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM model_benchmarks WHERE model_id = ? AND archived = 0",
                    (model_id,),
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.warning(f"Failed to get benchmark for {model_id}: {e}")
            return None

    def get_benchmarks_for_models(self, model_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Get benchmarks for multiple models.

        Returns dict keyed by model_id with benchmark data.
        """
        if not self.is_available():
            return {}

        if not model_ids:
            return {}

        # Security: Validate model IDs to prevent injection
        # SQLite has a maximum of 999 parameters per query
        if len(model_ids) > 999:
            logger.warning(f"Too many model_ids requested: {len(model_ids)}")
            # Truncate to max allowed
            model_ids = model_ids[:999]
        for model_id in model_ids:
            if not re.match(r"^[a-zA-Z0-9_\-/\.]+$", model_id):
                logger.warning(f"Invalid model_id format: {model_id}")
                return {}

        # Try to get from unified cache first
        cached_all = _provider_cache.get("all_benchmarks")
        if cached_all is not None:
            # Filter requested models
            results = {}
            for model_id in model_ids:
                if model_id in cached_all:
                    results[model_id] = cached_all[model_id]
            if len(results) == len(model_ids):
                # All requested models found in cache
                return results
            # Some models missing from cache, we'll query for missing ones
            missing_models = [m for m in model_ids if m not in results]
        else:
            results = {}
            missing_models = model_ids

        if not missing_models:
            return results

        # Need to query database for missing models
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Chunk queries to avoid SQLite parameter limit (999)
                # and improve performance with smaller queries
                chunk_size = 250
                queried_results = {}

                for i in range(0, len(missing_models), chunk_size):
                    chunk = missing_models[i : i + chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    cursor.execute(
                        f"""SELECT * FROM model_benchmarks
                            WHERE model_id IN ({placeholders}) AND archived = 0""",
                        chunk,
                    )
                    for row in cursor.fetchall():
                        model_id = row["model_id"]
                        queried_results[model_id] = dict(row)

                # Update results with queried data
                results.update(queried_results)

                # Update cache with newly fetched models
                if queried_results:
                    # Get current cache (might have been updated by another thread)
                    current_cache = _provider_cache.get("all_benchmarks")
                    if current_cache is None:
                        current_cache = {}
                    # Merge new data
                    current_cache.update(queried_results)
                    # Store back with TTL
                    _provider_cache.set("all_benchmarks", current_cache)

                return results
        except Exception as e:
            logger.warning(f"Failed to get benchmarks for models: {e}")
            return {}

    def get_all_benchmarks(self) -> list[dict[str, Any]]:
        """Get all active model benchmarks."""
        if not self.is_available():
            return []

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM model_benchmarks WHERE archived = 0")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get all benchmarks: {e}")
            return []

    def resolve_alias(self, alias: str) -> str | None:
        """Resolve a model alias to canonical model_id."""
        if not self.is_available():
            return None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT canonical_id FROM aliases WHERE alias = ?", (alias,))
                row = cursor.fetchone()
                return row["canonical_id"] if row else None
        except Exception as e:
            logger.debug(f"Alias lookup failed for {alias}: {e}")
            return None

    def find_model_by_name(self, name: str) -> dict[str, Any] | None:
        """Find a model by name or alias."""
        if not self.is_available():
            return None

        # Try exact match first
        benchmark = self.get_benchmark(name)
        if benchmark:
            return benchmark

        # Try alias resolution
        canonical_id = self.resolve_alias(name)
        if canonical_id:
            return self.get_benchmark(canonical_id)

        # Try partial match on model names
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM model_benchmarks WHERE model_id LIKE ? AND archived = 0",
                    (f"%{name}%",),
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
        except Exception as e:
            logger.debug(f"Partial match failed for {name}: {e}")

        return None


def get_provider_db() -> ProviderDB:
    """Get the provider.db instance."""
    return ProviderDB()


def invalidate_provider_cache() -> None:
    """Invalidate the provider.db cache."""
    _provider_cache.invalidate("all_benchmarks")
    logger.debug("Provider.db cache invalidated")
