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
    """Bulk upsert benchmarks using efficient single-transaction approach.
    
    Uses SQLAlchemy bulk operations with ON CONFLICT for maximum performance.
    Returns the number of benchmarks that were upserted.
    """
    if not benchmarks:
        return 0

    # Whitelist of allowed ModelBenchmark columns
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
        "extra_data",
        "last_updated",
    }

    # Pre-process all items
    processed = []
    for data in benchmarks:
        cleaned = {}
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)) and k != "extra_data":
                continue
            if k not in ALLOWED_BENCHMARK_FIELDS:
                logger.warning(f"Skipping unknown benchmark field: {k}")
                continue
            cleaned[k] = v
        
        if cleaned:
            cleaned["last_updated"] = datetime.now(timezone.utc)
            processed.append(cleaned)

    if not processed:
        return 0

    with get_session() as session:
        # Use SQLAlchemy bulk upsert with ON CONFLICT
        # This is much faster than individual commits
        try:
            from sqlalchemy.dialects.sqlite import insert
            
            stmt = insert(ModelBenchmark).values(processed)
            
            # Define update rules for each column on conflict
            update_dict = {
                col: getattr(stmt.excluded, col) 
                for col in ALLOWED_BENCHMARK_FIELDS 
                if col not in ("ollama_name",)  # Don't update primary key
            }
            
            # Special handling for datetime comparisons
            stmt = stmt.on_conflict_do_update(
                index_elements=["ollama_name"],
                set_=update_dict
            )
            
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
            
        except Exception as e:
            logger.error(f"Bulk upsert failed: {e}")
            session.rollback()
            # Fallback to individual inserts if bulk fails
            count = 0
            for cleaned in processed:
                existing = session.query(ModelBenchmark).filter(
                    ModelBenchmark.ollama_name == cleaned.get("ollama_name")
                ).first()
                
                if existing:
                    for k, v in cleaned.items():
                        if k != "ollama_name":
                            setattr(existing, k, v)
                else:
                    safe_data = {k: v for k, v in cleaned.items() if k in ALLOWED_BENCHMARK_FIELDS}
                    session.add(ModelBenchmark(**safe_data))
                count += 1
            
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
