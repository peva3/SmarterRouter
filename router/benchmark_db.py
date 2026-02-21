import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from router.database import get_session
from router.models import BenchmarkSync, ModelBenchmark

logger = logging.getLogger(__name__)

_BENCHMARK_CACHE_TTL = 60.0
_benchmarks_cache: list[dict] | None = None
_benchmarks_cache_time: float = 0.0
_profiles_cache: list[dict] | None = None
_profiles_cache_time: float = 0.0
_PROFILE_CACHE_TTL = 60.0

_benchmarks_for_models_cache: dict[frozenset, tuple[float, list[dict]]] = {}


def get_benchmark(ollama_name: str) -> ModelBenchmark | None:
    with get_session() as session:
        return session.execute(
            select(ModelBenchmark).where(ModelBenchmark.ollama_name == ollama_name)
        ).scalar_one_or_none()


def get_all_benchmarks() -> list[dict]:
    global _benchmarks_cache, _benchmarks_cache_time
    now = time.monotonic()
    if _benchmarks_cache is not None and (now - _benchmarks_cache_time) < _BENCHMARK_CACHE_TTL:
        return _benchmarks_cache

    with get_session() as session:
        benchmarks = session.execute(select(ModelBenchmark)).scalars().all()
        _benchmarks_cache = [
            {
                "ollama_name": b.ollama_name,
                "reasoning_score": b.reasoning_score,
                "coding_score": b.coding_score,
                "general_score": b.general_score,
                "elo_rating": b.elo_rating,
                "throughput": b.throughput,
                "parameters": b.parameters,
                # Add other fields if needed for logging
                "mmlu": b.mmlu,
                "humaneval": b.humaneval,
            }
            for b in benchmarks
        ]
        _benchmarks_cache_time = now
        return _benchmarks_cache


def get_benchmarks_for_models(model_names: list[str]) -> list[dict]:
    if not model_names:
        return []
    
    cache_key = frozenset(model_names)
    now = time.monotonic()
    if cache_key in _benchmarks_for_models_cache:
        cached_time, cached_result = _benchmarks_for_models_cache[cache_key]
        if (now - cached_time) < _BENCHMARK_CACHE_TTL:
            return cached_result
    
    with get_session() as session:
        benchmarks = (
            session.execute(
                select(ModelBenchmark).where(ModelBenchmark.ollama_name.in_(model_names))
            )
            .scalars()
            .all()
        )
        result = [
            {
                "ollama_name": b.ollama_name,
                "reasoning_score": b.reasoning_score,
                "coding_score": b.coding_score,
                "general_score": b.general_score,
                "elo_rating": b.elo_rating,
            }
            for b in benchmarks
        ]
        _benchmarks_for_models_cache[cache_key] = (now, result)
        return result


def upsert_benchmark(data: dict[str, Any]) -> None:
    bulk_upsert_benchmarks([data])


def bulk_upsert_benchmarks(benchmarks: list[dict[str, Any]]) -> int:
    if not benchmarks:
        return 0

    # Whitelist of allowed ModelBenchmark columns to prevent SQL injection via setattr
    ALLOWED_BENCHMARK_FIELDS = {
        "ollama_name",
        "full_name",
        "parameters",
        "quantization",
        "mmlu",
        "humaneval",
        "math",
        "gpqa",
        "hellaswag",
        "winogrande",
        "truthfulqa",
        "mmlu_pro",
        "reasoning_score",
        "coding_score",
        "general_score",
        "elo_rating",
        "throughput",
        "context_window",
        "vision",
        "tool_calling",
        "extra_data",  # Provider-specific extra data (e.g., ArtificialAnalysis indices)
        "last_updated",
    }

    count = 0
    for data in benchmarks:
        # Filter out None values and non-scalars, and validate keys
        cleaned = {}
        for k, v in data.items():
            if v is None:
                continue
            # Allow dict/list for extra_data only; skip for all other fields
            if isinstance(v, (dict, list)) and k != "extra_data":
                continue
            # Validate key against whitelist
            if k not in ALLOWED_BENCHMARK_FIELDS:
                logger.warning(f"Skipping unknown benchmark field: {k}")
                continue
            cleaned[k] = v
        cleaned["last_updated"] = datetime.now(timezone.utc)

        if not cleaned:
            continue

        with get_session() as session:
            # Try to update existing
            existing = (
                session.query(ModelBenchmark)
                .filter(ModelBenchmark.ollama_name == cleaned.get("ollama_name"))
                .first()
            )

            if existing:
                for k, v in cleaned.items():
                    if k != "ollama_name":
                        # Extra safety: ensure key is in whitelist before setattr
                        if k not in ALLOWED_BENCHMARK_FIELDS:
                            continue
                        new_val = v
                        old_val = getattr(existing, k)
                        
                        # Always update these fields if present
                        if k in ("last_updated", "extra_data"):
                            should_update = new_val is not None
                        # For datetime comparisons, handle naive vs aware
                        elif isinstance(new_val, datetime) and isinstance(old_val, datetime):
                            if new_val.tzinfo is None:
                                new_val = new_val.replace(tzinfo=timezone.utc)
                            if old_val.tzinfo is None:
                                old_val = old_val.replace(tzinfo=timezone.utc)
                            should_update = new_val and (not old_val or new_val > old_val)
                        else:
                            should_update = new_val and (not old_val or new_val > old_val)
                        
                        if should_update:
                            setattr(existing, k, v)
            else:
                # Filter dict one more time to ensure only allowed fields
                safe_data = {k: v for k, v in cleaned.items() if k in ALLOWED_BENCHMARK_FIELDS}
                session.add(ModelBenchmark(**safe_data))

            session.commit()
            count += 1

    return count


def get_last_sync() -> datetime | None:
    with get_session() as session:
        result = session.execute(
            select(BenchmarkSync.last_sync).order_by(BenchmarkSync.id.desc()).limit(1)
        ).scalar_one_or_none()
        return result


def update_sync_status(status: str, models_count: int = 0) -> None:
    with get_session() as session:
        sync = BenchmarkSync(
            last_sync=datetime.now(timezone.utc),
            models_count=models_count,
            status=status,
        )
        session.add(sync)
        session.commit()


def remove_benchmarks_not_in(model_names: list[str]) -> int:
    """Remove benchmarks not in the provided list using ORM delete (SQL injection safe)."""
    with get_session() as session:
        if not model_names:
            return 0

        # Validate all model names are strings and reasonable length
        if not all(isinstance(name, str) and len(name) < 200 for name in model_names):
            logger.warning("Invalid model names provided to remove_benchmarks_not_in")
            return 0

        # Use ORM delete instead of raw SQL for safety
        deleted = (
            session.query(ModelBenchmark)
            .filter(~ModelBenchmark.ollama_name.in_(model_names))
            .delete(synchronize_session=False)
        )

        session.commit()
        return deleted


def invalidate_benchmarks_cache() -> None:
    """Invalidate the benchmarks cache."""
    global _benchmarks_cache, _benchmarks_cache_time, _benchmarks_for_models_cache
    _benchmarks_cache = None
    _benchmarks_cache_time = 0.0
    _benchmarks_for_models_cache = {}
    logger.debug("Benchmarks cache invalidated")


def invalidate_profiles_cache() -> None:
    """Invalidate the profiles cache."""
    global _profiles_cache, _profiles_cache_time
    _profiles_cache = None
    _profiles_cache_time = 0.0
    logger.debug("Profiles cache invalidated")


def invalidate_all_caches() -> None:
    """Invalidate all caches (benchmarks and profiles)."""
    invalidate_benchmarks_cache()
    invalidate_profiles_cache()
    logger.debug("All caches invalidated")
