import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from router.database import get_session
from router.models import BenchmarkSync, ModelBenchmark

logger = logging.getLogger(__name__)


def get_benchmark(ollama_name: str) -> ModelBenchmark | None:
    with get_session() as session:
        return session.execute(
            select(ModelBenchmark).where(ModelBenchmark.ollama_name == ollama_name)
        ).scalar_one_or_none()


def get_all_benchmarks() -> list[dict]:
    with get_session() as session:
        benchmarks = session.execute(select(ModelBenchmark)).scalars().all()
        # Convert to dicts to avoid session detachment
        return [
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


def get_benchmarks_for_models(model_names: list[str]) -> list[ModelBenchmark]:
    with get_session() as session:
        return list(
            session.execute(
                select(ModelBenchmark).where(
                    ModelBenchmark.ollama_name.in_(model_names)
                )
            ).scalars().all()
        )


def upsert_benchmark(data: dict[str, Any]) -> None:
    bulk_upsert_benchmarks([data])


def bulk_upsert_benchmarks(benchmarks: list[dict[str, Any]]) -> int:
    if not benchmarks:
        return 0

    count = 0
    for data in benchmarks:
        # Filter out None values and non-scalars
        cleaned = {}
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                continue
            cleaned[k] = v
        cleaned["last_updated"] = datetime.now(timezone.utc)

        if not cleaned:
            continue

        with get_session() as session:
            # Try to update existing
            existing = session.query(ModelBenchmark).filter(
                ModelBenchmark.ollama_name == cleaned.get("ollama_name")
            ).first()

            if existing:
                for k, v in cleaned.items():
                    if k != "ollama_name":
                        # Only update if the new value is not None/0, or if existing is None/0
                        new_val = v
                        old_val = getattr(existing, k)
                        if new_val and (not old_val or new_val > old_val or k == "last_updated"):
                            setattr(existing, k, v)
            else:
                session.add(ModelBenchmark(**cleaned))

            session.commit()
            count += 1

    return count


def get_last_sync() -> BenchmarkSync | None:
    with get_session() as session:
        return session.execute(
            select(BenchmarkSync).order_by(BenchmarkSync.id.desc()).limit(1)
        ).scalar_one_or_none()


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
        deleted = session.query(ModelBenchmark).filter(
            ~ModelBenchmark.ollama_name.in_(model_names)
        ).delete(synchronize_session=False)
        
        session.commit()
        return deleted
