import logging
import time
from collections import OrderedDict
from datetime import UTC, datetime
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

_BENCHMARKS_FOR_MODELS_CACHE_MAX_SIZE = 512
_benchmarks_for_models_cache: OrderedDict[frozenset, tuple[float, list[dict]]] = OrderedDict()


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
            _benchmarks_for_models_cache.move_to_end(cache_key)
            return cached_result

    try:
        with get_session() as session:
            # Chunk queries to avoid SQLite parameter limit (999)
            # SQLAlchemy's in_() method parameterizes correctly but still hits SQLite limit
            chunk_size = 250
            all_benchmarks: list[ModelBenchmark] = []

            for i in range(0, len(model_names), chunk_size):
                chunk = model_names[i : i + chunk_size]
                benchmarks = (
                    session.execute(
                        select(ModelBenchmark).where(ModelBenchmark.ollama_name.in_(chunk))
                    )
                    .scalars()
                    .all()
                )
                all_benchmarks.extend(benchmarks)

            result = [
                {
                    "ollama_name": b.ollama_name,
                    "reasoning_score": b.reasoning_score,
                    "coding_score": b.coding_score,
                    "general_score": b.general_score,
                    "elo_rating": b.elo_rating,
                }
                for b in all_benchmarks
            ]
            _benchmarks_for_models_cache[cache_key] = (now, result)
            _benchmarks_for_models_cache.move_to_end(cache_key)
            if len(_benchmarks_for_models_cache) > _BENCHMARKS_FOR_MODELS_CACHE_MAX_SIZE:
                _benchmarks_for_models_cache.popitem(last=False)
            return result
    except Exception as e:
        logger.warning(f"Failed to get benchmarks for models: {e}")
        return []


def upsert_benchmark(data: dict[str, Any]) -> None:
    bulk_upsert_benchmarks([data])


def bulk_upsert_benchmarks(benchmarks: list[dict[str, Any]]) -> int:
    """Bulk upsert benchmarks using efficient single-transaction approach.

    Returns the number of benchmarks that were processed.
    """
    if not benchmarks:
        return 0

    allowed_benchmark_fields = {
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
        "extra_data",
        "last_updated",
    }

    processed = []
    for data in benchmarks:
        cleaned = {}
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)) and k != "extra_data":
                continue
            if k not in allowed_benchmark_fields:
                logger.warning(f"Skipping unknown benchmark field: {k}")
                continue
            cleaned[k] = v

        if cleaned and "ollama_name" in cleaned:
            cleaned["last_updated"] = datetime.now(UTC)
            processed.append(cleaned)

    if not processed:
        return 0

    count = 0
    # Single transaction for all benchmarks — commit once at the end
    with get_session() as session:
        for cleaned in processed:
            try:
                existing = (
                    session.query(ModelBenchmark)
                    .filter(ModelBenchmark.ollama_name == cleaned.get("ollama_name"))
                    .first()
                )

                if existing:
                    for k, v in cleaned.items():
                        if k not in ("ollama_name",):
                            setattr(existing, k, v)
                else:
                    safe_data = {k: v for k, v in cleaned.items() if k in allowed_benchmark_fields}
                    session.add(ModelBenchmark(**safe_data))

                count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert benchmark for {cleaned.get('ollama_name')}: {e}")
                # Continue with other benchmarks

        session.commit()

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
            last_sync=datetime.now(UTC),
            models_count=models_count,
            status=status,
        )
        session.add(sync)
        session.commit()


def remove_benchmarks_not_in(model_names: list[str]) -> int:
    """Remove benchmarks not in the provided list using ORM delete (SQL injection safe)."""
    if not model_names:
        return 0

    # Validate all model names are strings and reasonable length
    if not all(isinstance(name, str) and len(name) < 200 for name in model_names):
        logger.warning("Invalid model names provided to remove_benchmarks_not_in")
        return 0

    try:
        with get_session() as session:
            # SQLite has a maximum of 999 parameters per query
            # If we exceed this, we need to use a different approach
            if len(model_names) <= 999:
                # Simple case: single query
                deleted = (
                    session.query(ModelBenchmark)
                    .filter(~ModelBenchmark.ollama_name.in_(model_names))
                    .delete(synchronize_session=False)
                )
            else:
                # Complex case: chunk with AND conditions
                # Delete where ollama_name NOT IN chunk1 AND NOT IN chunk2 AND ...
                from sqlalchemy import and_

                chunk_size = 250
                chunks = [
                    model_names[i : i + chunk_size] for i in range(0, len(model_names), chunk_size)
                ]

                # Build AND conditions
                conditions = [~ModelBenchmark.ollama_name.in_(chunk) for chunk in chunks]
                combined_condition = and_(*conditions) if len(conditions) > 1 else conditions[0]

                deleted = (
                    session.query(ModelBenchmark)
                    .filter(combined_condition)
                    .delete(synchronize_session=False)
                )

            session.commit()
            return deleted
    except Exception as e:
        logger.warning(f"Failed to remove benchmarks: {e}")
        # Rollback happens automatically when exception escapes context manager
        return 0


def invalidate_benchmarks_cache() -> None:
    """Invalidate the benchmarks cache."""
    global _benchmarks_cache, _benchmarks_cache_time, _benchmarks_for_models_cache
    _benchmarks_cache = None
    _benchmarks_cache_time = 0.0
    _benchmarks_for_models_cache = OrderedDict()
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
