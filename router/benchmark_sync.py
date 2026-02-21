import asyncio
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
    """Sync benchmarks from all enabled providers in parallel.
    
    Args:
        ollama_models: List of model names to match against benchmarks
        
    Returns:
        Tuple of (count of synced models, list of matched model names)
    """
    logger.info("Starting benchmark sync")
    update_sync_status("running", 0)

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

    async def fetch_provider_data(provider: BenchmarkProvider) -> list[dict[str, Any]]:
        """Fetch data from a single provider with timeout protection."""
        try:
            return await asyncio.wait_for(
                provider.fetch_data(ollama_models),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Provider {provider.name} timed out after 120s")
            return []
        except Exception as e:
            logger.error(f"Provider {provider.name} failed: {e}")
            return []

    results = await asyncio.gather(*[fetch_provider_data(p) for p in providers])

    for provider, data in zip(providers, results):
        logger.info(f"Provider {provider.name} returned {len(data)} records")
        for item in data:
            name = item["ollama_name"]
            if name not in all_data:
                all_data[name] = {}

            for k, v in item.items():
                if v is not None:
                    all_data[name][k] = v

            matched_models.add(name)

    final_benchmarks = list(all_data.values())

    count = bulk_upsert_benchmarks(final_benchmarks)

    update_sync_status("completed", count)
    logger.info(f"Benchmark sync completed: {count} models synced")

    return count, list(matched_models)
