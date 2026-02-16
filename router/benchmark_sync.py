import logging
import re
from typing import Any

from router.benchmark_db import (
    bulk_upsert_benchmarks,
    remove_benchmarks_not_in,
    update_sync_status,
)
from router.providers.base import BenchmarkProvider
from router.providers.huggingface import HuggingFaceProvider
from router.providers.lmsys import LMSYSProvider
from router.providers.artificial_analysis import ArtificialAnalysisProvider

logger = logging.getLogger(__name__)


async def sync_benchmarks(ollama_models: list[str]) -> tuple[int, list[str]]:
    logger.info("Starting benchmark sync")
    update_sync_status("running", 0)

    # In a real app, inject settings or read from config
    # For now, default to all if not specified, or parse from env
    from router.config import settings
    enabled_sources = [s.strip().lower() for s in settings.benchmark_sources.split(",")]

    providers: list[BenchmarkProvider] = []
    if "huggingface" in enabled_sources:
        providers.append(HuggingFaceProvider())
    if "lmsys" in enabled_sources:
        providers.append(LMSYSProvider())
    if "artificial_analysis" in enabled_sources:
        providers.append(ArtificialAnalysisProvider())

    all_data: dict[str, dict[str, Any]] = {}
    matched_models: set[str] = set()

    for provider in providers:
        try:
            data = await provider.fetch_data(ollama_models)
            logger.info(f"Provider {provider.name} returned {len(data)} records")
            
            for item in data:
                name = item["ollama_name"]
                if name not in all_data:
                    all_data[name] = {}
                
                # Merge data (non-null values overwrite)
                for k, v in item.items():
                    if v is not None:
                        all_data[name][k] = v
                
                matched_models.add(name)
                
        except Exception as e:
            logger.error(f"Provider {provider.name} failed: {e}")

    # Convert merged dict back to list
    final_benchmarks = list(all_data.values())
    
    count = bulk_upsert_benchmarks(final_benchmarks)

    current_ollama_names = [name.split(":")[0].lower() for name in ollama_models]
    # Note: normalize logic duplicates slightly here, ideally centralized
    
    update_sync_status("completed", count)
    logger.info(f"Benchmark sync completed: {count} models synced")

    return count, list(matched_models)
