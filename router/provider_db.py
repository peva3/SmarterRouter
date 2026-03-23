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
import time
from contextlib import contextmanager
from datetime import datetime
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


# Slow DB fallback state (process-local)
_provider_db_degraded_until: float = 0.0
_provider_db_last_good_cache_time: float = 0.0


def _get_cached_all_benchmarks_if_fresh(max_age_seconds: int) -> dict[str, dict[str, Any]] | None:
    """Return cached benchmark map if within stale-allowed age."""
    global _provider_db_last_good_cache_time
    cached_all = _provider_cache.get("all_benchmarks")
    if cached_all is None:
        return None

    if max_age_seconds <= 0:
        return cached_all

    age = time.monotonic() - _provider_db_last_good_cache_time
    if _provider_db_last_good_cache_time > 0 and age <= max_age_seconds:
        return cached_all
    return None


def _mark_degraded(window_seconds: int) -> None:
    """Mark provider DB as temporarily degraded for fallback serving."""
    global _provider_db_degraded_until
    _provider_db_degraded_until = time.monotonic() + max(window_seconds, 0)


def _clear_degraded_if_elapsed() -> None:
    """Clear degraded mode when fallback window has elapsed."""
    global _provider_db_degraded_until
    if _provider_db_degraded_until and time.monotonic() > _provider_db_degraded_until:
        _provider_db_degraded_until = 0.0


def _is_stale(last_build: str | None, max_age_hours: int) -> bool:
    """Return True if provider.db build timestamp is older than max_age_hours."""
    if max_age_hours <= 0 or not last_build:
        return False
    try:
        # Expected format from provider metadata: ISO-8601 UTC timestamp
        ts = last_build.replace("Z", "+00:00")
        try:
            built_dt = datetime.fromisoformat(ts)
            built_at = built_dt.timestamp()
        except Exception:
            # Non-ISO metadata formats are treated as non-stale to avoid false blocking
            return False
        age_hours = (time.time() - built_at) / 3600
        return age_hours > max_age_hours
    except Exception:
        return False


class ProviderDB:
    """Interface to provider.db for external model benchmarks."""

    def __init__(self, db_path: str | None = None):
        path = db_path or settings.provider_db_path
        base_dir = Path(__file__).resolve().parents[1]

        # Resolve relative paths against project root for stability across CWDs
        candidate_path = Path(path)
        if not candidate_path.is_absolute():
            candidate_path = base_dir / candidate_path

        # Allow bypassing validation for tests
        if os.environ.get("ROUTER_TEST_MODE"):
            self.db_path = Path(path)
        else:
            # Security: Validate path is within allowed directory (prevent path traversal)
            resolved = candidate_path.resolve()
            allowed_parents = [
                (base_dir / "data").resolve(),
                Path("/app/hubrouter/data").resolve(),
                (Path.cwd() / "data").resolve(),
            ]
            if not any(
                resolved.is_relative_to(parent) for parent in allowed_parents if parent.exists()
            ):
                raise ProviderDBError(
                    f"provider.db path {path} is outside allowed data directories"
                )
            self.db_path = resolved
        self._conn: sqlite3.Connection | None = None
        self._has_archived_column: bool | None = None

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

    def _detect_archived_column(self) -> bool:
        """Detect whether model_benchmarks has an 'archived' column."""
        if self._has_archived_column is not None:
            return self._has_archived_column
        if not self.is_available():
            self._has_archived_column = False
            return self._has_archived_column
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(model_benchmarks)")
                cols = [row[1] for row in cursor.fetchall()]
                self._has_archived_column = "archived" in cols
        except Exception:
            self._has_archived_column = False
        return self._has_archived_column

    def get_stats(self) -> dict[str, Any]:
        """Get provider.db statistics."""
        if not self.is_available():
            return {"available": False, "total_models": 0, "degraded": False, "stale": False}

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if self._detect_archived_column():
                    cursor.execute("SELECT COUNT(*) as total FROM model_benchmarks WHERE archived = 0")
                    total = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) as total FROM model_benchmarks WHERE archived = 1")
                    archived = cursor.fetchone()[0]
                else:
                    cursor.execute("SELECT COUNT(*) as total FROM model_benchmarks")
                    total = cursor.fetchone()[0]
                    archived = 0

                cursor.execute("SELECT value FROM metadata WHERE key = 'last_build'")
                row = cursor.fetchone()
                last_build = row[0] if row else None
                stale = _is_stale(last_build, settings.provider_db_max_age_hours)

                return {
                    "available": True,
                    "total_models": total,
                    "archived_models": archived,
                    "last_build": last_build,
                    "degraded": _provider_db_degraded_until > time.monotonic(),
                    "stale": stale,
                }
        except Exception as e:
            logger.warning(f"Failed to get provider.db stats: {e}")
            return {
                "available": False,
                "error": str(e),
                "degraded": _provider_db_degraded_until > time.monotonic(),
                "stale": False,
            }

    def get_benchmark(self, model_id: str) -> dict[str, Any] | None:
        """Get benchmark for a single model."""
        if not self.is_available():
            return None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if self._detect_archived_column():
                    cursor.execute(
                        "SELECT * FROM model_benchmarks WHERE model_id = ? AND archived = 0",
                        (model_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM model_benchmarks WHERE model_id = ?",
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
        _clear_degraded_if_elapsed()

        if not self.is_available():
            if settings.db_slow_fallback_enabled:
                cached_all = _get_cached_all_benchmarks_if_fresh(settings.db_stale_cache_max_age_seconds)
                if cached_all is not None:
                    return {m: cached_all[m] for m in model_ids if m in cached_all}
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
            if not re.match(r"^[a-zA-Z0-9_\-/\.:]+$", model_id):
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

        # If currently degraded due to recent DB slowness/errors, serve stale cache if available
        if settings.db_slow_fallback_enabled and _provider_db_degraded_until > time.monotonic():
            cached_all = _get_cached_all_benchmarks_if_fresh(settings.db_stale_cache_max_age_seconds)
            if cached_all is not None:
                fallback_results = {m: cached_all[m] for m in model_ids if m in cached_all}
                if fallback_results:
                    logger.debug(
                        "provider.db in degraded mode; serving stale in-memory fallback for %d models",
                        len(fallback_results),
                    )
                    return fallback_results

        # Need to query database for missing models
        try:
            started = time.monotonic()
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Chunk queries to avoid SQLite parameter limit (999)
                # and improve performance with smaller queries
                chunk_size = 250
                queried_results = {}

                archived_clause = " AND archived = 0" if self._detect_archived_column() else ""
                for i in range(0, len(missing_models), chunk_size):
                    chunk = missing_models[i : i + chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    cursor.execute(
                        f"""SELECT * FROM model_benchmarks
                            WHERE model_id IN ({placeholders}){archived_clause}""",
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
                    global _provider_db_last_good_cache_time
                    _provider_db_last_good_cache_time = time.monotonic()

                elapsed_ms = (time.monotonic() - started) * 1000
                if (
                    settings.db_slow_fallback_enabled
                    and elapsed_ms > settings.db_slow_query_threshold_ms
                ):
                    _mark_degraded(settings.db_slow_fallback_window_seconds)
                    logger.warning(
                        "provider.db query slow (%.1fms > %dms); enabling degraded fallback window",
                        elapsed_ms,
                        settings.db_slow_query_threshold_ms,
                    )

                return results
        except Exception as e:
            if settings.db_slow_fallback_enabled:
                _mark_degraded(settings.db_slow_fallback_window_seconds)
                cached_all = _get_cached_all_benchmarks_if_fresh(settings.db_stale_cache_max_age_seconds)
                if cached_all is not None:
                    logger.warning(
                        "provider.db query failed, serving stale fallback cache: %s",
                        e,
                    )
                    return {m: cached_all[m] for m in model_ids if m in cached_all}
            logger.warning(f"Failed to get benchmarks for models: {e}")
            return {}

    def get_all_benchmarks(self) -> list[dict[str, Any]]:
        """Get all active model benchmarks."""
        if not self.is_available():
            return []

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if self._detect_archived_column():
                    cursor.execute("SELECT * FROM model_benchmarks WHERE archived = 0")
                else:
                    cursor.execute("SELECT * FROM model_benchmarks")
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
                if self._detect_archived_column():
                    cursor.execute(
                        "SELECT * FROM model_benchmarks WHERE model_id LIKE ? AND archived = 0",
                        (f"%{name}%",),
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM model_benchmarks WHERE model_id LIKE ?",
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
    global _provider_db_last_good_cache_time, _provider_db_degraded_until
    _provider_cache.invalidate("all_benchmarks")
    _provider_db_last_good_cache_time = 0.0
    _provider_db_degraded_until = 0.0
    logger.debug("Provider.db cache invalidated")
