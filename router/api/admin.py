"""Admin API endpoints.

All admin endpoints require authentication via verify_admin_token dependency.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from router.audit import audit_admin_action, get_audit_logs
from router.benchmark_sync import sync_benchmarks
from router.config import Settings, settings
from router.database import get_session
from router.dlq import (
    count_dlq_entries,
    get_dlq_entry,
    list_dlq_entries,
)
from router.lifecycle import retry_dlq_entry
from router.models import (
    BenchmarkSync,
    ModelBenchmark,
    ModelProfile,
)
from router.profiler import profile_all_models
from router.logging_config import sanitize_for_logging
from router.schemas import sanitize_model_name

from router.state import (
    app_state,
    get_available_models_with_cache,
    get_settings,
    rate_limit_request,
    verify_admin_token,
    _log_error_with_context,
)
from router.vram_manager import VRAMManager
from router.vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/admin/profiles")
async def get_profiles(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    limit: int = 100,
    offset: int = 0,
    cursor: str | None = None,
):
    """Get model profiles with pagination.

    Retrieves model profiling data from the database, including reasoning, coding,
    creativity, factual, speed scores, and VRAM requirements. Uses cursor-based
    pagination for efficient navigation through large datasets.

    Args:
        limit (int): Maximum number of profiles to return (default 100, max 1000). Clamped for safety.
        offset (int): Number of profiles to skip when cursor is not provided. Ignored if cursor is set.
        cursor (str | None): Cursor pagination - the `name` of the last profile from previous page.

    Returns:
        dict: A paginated response containing:
            - total (int): Total number of profiles in the database.
            - limit (int): The page size used.
            - offset (int | None): The offset used, or null if cursor pagination.
            - cursor (str | None): The cursor for the next page, if more data exists.
            - next_cursor (str | None): Cursor to fetch the next page.
            - profiles (list[dict]): List of profile objects with keys:
                name (str), reasoning (float), coding (float), creativity (float),
                factual (float), speed (float), avg_response_time_ms (float | None),
                last_profiled (str | None, ISO8601).

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if router engine unavailable.
    """
    await rate_limit_request(request, config, is_admin=True)

    # Clamp limit to prevent memory exhaustion
    limit = min(max(1, limit), 1000)
    offset = max(0, offset)

    with get_session() as session:
        # Get total count for pagination info
        from sqlalchemy import func

        total = session.query(func.count(ModelProfile.id)).scalar()

        # Build query with cursor-based pagination if cursor provided
        query = session.query(ModelProfile).order_by(ModelProfile.name)
        if cursor is not None:
            query = query.filter(ModelProfile.name > cursor)
        else:
            query = query.offset(offset)

        # Fetch one extra to detect if there's a next page
        profiles = query.limit(limit + 1).all()

        # Determine next_cursor and trim to requested limit
        next_cursor = None
        if len(profiles) > limit:
            next_cursor = profiles[limit - 1].name
            profiles = profiles[:limit]

        return {
            "total": total,
            "limit": limit,
            "offset": offset if cursor is None else None,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "profiles": [
                {
                    "name": p.name,
                    "reasoning": p.reasoning,
                    "coding": p.coding,
                    "creativity": p.creativity,
                    "factual": p.factual,
                    "speed": p.speed,
                    "avg_response_time_ms": p.avg_response_time_ms,
                    "last_profiled": p.last_profiled.isoformat() if p.last_profiled else None,
                }
                for p in profiles
            ],
        }


@router.get("/admin/benchmarks")
async def get_benchmarks(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    limit: int = 100,
    offset: int = 0,
    cursor: str | None = None,
):
    """Get benchmark data with pagination.

    Retrieves benchmark scores for models from the database, including MMLU, HumanEval,
    MATH, GPQA, reasoning_score, coding_score, general_score, and parameters. Also
    includes information about the last benchmark sync.

    Args:
        limit (int): Maximum number of benchmarks to return (default 100, max 1000). Clamped for safety.
        offset (int): Number of benchmarks to skip when cursor is not provided. Ignored if cursor is set.
        cursor (str | None): Cursor pagination - the `ollama_name` of the last benchmark from previous page.

    Returns:
        dict: A paginated response containing:
            - total (int): Total number of benchmarks.
            - limit (int): The page size used.
            - offset (int | None): The offset used, or null if cursor pagination.
            - cursor (str | None): The cursor for the next page.
            - next_cursor (str | None): Cursor to fetch the next page.
            - benchmarks (list[dict]): List of benchmark objects with keys:
                ollama_name (str), full_name (str), mmlu (float), humaneval (float),
                math (float), gpqa (float), reasoning_score (float), coding_score (float),
                general_score (float), parameters (str | None), last_updated (str | None, ISO8601).
            - last_sync (str | None): ISO8601 timestamp of last successful sync.
            - sync_status (str | None): Status of the most recent sync.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited.
    """
    await rate_limit_request(request, config, is_admin=True)

    # Clamp limit to prevent memory exhaustion
    limit = min(max(1, limit), 1000)
    offset = max(0, offset)

    with get_session() as session:
        # Get total count for pagination info
        from sqlalchemy import func

        total = session.query(func.count(ModelBenchmark.id)).scalar()

        # Build query with cursor-based pagination if cursor provided
        query = session.query(ModelBenchmark).order_by(ModelBenchmark.ollama_name)
        if cursor is not None:
            query = query.filter(ModelBenchmark.ollama_name > cursor)
        else:
            query = query.offset(offset)

        # Fetch one extra to detect if there's a next page
        benchmarks = query.limit(limit + 1).all()

        # Determine next_cursor and trim to requested limit
        next_cursor = None
        if len(benchmarks) > limit:
            next_cursor = benchmarks[limit - 1].ollama_name
            benchmarks = benchmarks[:limit]

        last_sync = session.execute(
            select(BenchmarkSync).order_by(BenchmarkSync.id.desc()).limit(1)
        ).scalar_one_or_none()

        # Extract sync info while in session
        last_sync_time = (
            last_sync.last_sync.isoformat() if last_sync and last_sync.last_sync else None
        )
        sync_status = last_sync.status if last_sync else None

        return {
            "total": total,
            "limit": limit,
            "offset": offset if cursor is None else None,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "benchmarks": [
                {
                    "ollama_name": b.ollama_name,
                    "full_name": b.full_name,
                    "mmlu": b.mmlu,
                    "humaneval": b.humaneval,
                    "math": b.math,
                    "gpqa": b.gpqa,
                    "reasoning_score": b.reasoning_score,
                    "coding_score": b.coding_score,
                    "general_score": b.general_score,
                    "parameters": b.parameters,
                    "last_updated": b.last_updated.isoformat() if b.last_updated else None,
                }
                for b in benchmarks
            ],
            "last_sync": last_sync_time,
            "sync_status": sync_status,
        }


@router.get("/admin/stats")
async def get_stats(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
):
    """Get router statistics.

    Returns operational metrics including uptime, request counts, error counts,
    per-model request breakdown, cache statistics, and the currently loaded model.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.

    Returns:
        dict: Statistics object containing:
            - uptime_seconds (float): Time since application start.
            - total_requests (int): Total number of requests handled.
            - total_errors (int): Total number of errors encountered.
            - requests_by_model (dict[str, int]): Request count per model.
            - requests_by_category (dict[str, int]): Request count per prompt category.
            - cache (dict): Cache statistics from the semantic cache.
            - current_loaded_model (str | None): Name of the currently loaded model, if any.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if backend unavailable.
    """

    await rate_limit_request(request, config, is_admin=True)

    # Calculate uptime
    uptime_seconds: float = 0.0
    if hasattr(app_state, "start_time"):
        uptime_seconds = (datetime.now(UTC) - app_state.start_time).total_seconds()

    # Get cache stats from router engine
    cache_stats = {}
    if app_state.router_engine and app_state.router_engine.semantic_cache:
        cache_stats = await app_state.router_engine.semantic_cache.get_stats()

    return {
        "uptime_seconds": uptime_seconds,
        "total_requests": getattr(app_state, "total_requests", 0),
        "total_errors": getattr(app_state, "total_errors", 0),
        "requests_by_model": getattr(app_state, "requests_by_model", {}),
        "requests_by_category": getattr(app_state, "requests_by_category", {}),
        "cache": cache_stats,
        "current_loaded_model": app_state.current_loaded_model,
    }


@router.post("/admin/reprofile")
async def reprofile(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    force: bool = False,
):
    """Trigger manual reprofiling of all models.

    Initiates the profiling process for all available models by sending test prompts
    and measuring response times. This operation can be resource-intensive and may
    take several minutes depending on the number of models.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        force (bool): If true, re-profile models even if recently profiled.

    Returns:
        dict: A response containing:
            - profiled (list[str]): Names of models that were successfully profiled.
            - count (int): Number of models profiled.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if backend not available.
    """
    if not app_state.backend:
        return JSONResponse({"error": "Client not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    async with audit_admin_action(request, "reprofile", {"force": force}) as audit:
        logger.info(f"Manual reprofile triggered (force={force})")
        results = await profile_all_models(app_state.backend, force=force)
        audit.set_result(f"Profiled {len(results)} models")

    return {
        "profiled": [r.model_name for r in results],
        "count": len(results),
    }


@router.post("/admin/models/refresh")
async def refresh_models(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    cleanup: bool | None = None,
):
    """Refresh the list of available models from the backend.

    Queries the backend (Ollama, llama.cpp, or OpenAI-compatible) to discover
    currently available models and updates the internal model registry. Optionally
    removes models that are no longer present.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        cleanup (bool | None): If true, remove models that are no longer available.

    Returns:
        dict: A response containing:
            - status (str): Always "success" on success.
            - changes (dict): Summary of changes made during refresh.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if router engine unavailable.
    """
    if not app_state.router_engine:
        return JSONResponse({"error": "Router engine not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    async with audit_admin_action(request, "models_refresh", {"cleanup": cleanup}) as audit:
        logger.info("Manual model refresh triggered")
        changes = await app_state.router_engine.refresh_models(cleanup=cleanup)
        audit.set_result(f"Refresh complete: {changes}")

    return {
        "status": "success",
        "changes": changes,
    }


@router.post("/admin/models/reprofile")
async def reprofile_models(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    force: bool = False,
):
    """Re-profile models via the router engine.

    Uses the RouterEngine's profiling capability to update model performance metrics.
    This differs from the global reprofile endpoint by using the router's own
    profiling logic, which can be more efficient and integrates with the cache.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        force (bool): If true, force reprofiling even if recent data exists.

    Returns:
        dict: A response containing:
            - status (str): Always "success" on success.
            - results (dict): Profiling results details (structure depends on RouterEngine implementation).

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if router engine unavailable.
    """
    if not app_state.router_engine:
        return JSONResponse({"error": "Router engine not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    async with audit_admin_action(request, "models_reprofile", {"force": force}) as audit:
        logger.info(f"Manual model re-profiling triggered (force={force})")
        results = await app_state.router_engine.reprofile_models(force=force)
        audit.set_result(f"Reprofiled: {results}")

    return {
        "status": "success",
        "results": results,
    }


@router.post("/admin/cache/invalidate")
async def invalidate_cache(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    model: str | None = None,
    response_cache_only: bool = False,
):
    """Invalidate semantic cache entries.

    Removes cached responses from the semantic cache. Can target a specific model
    or invalidate the entire cache. Also supports invalidating only response cache entries.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        model (str | None): If provided, only invalidate entries for this model.
        response_cache_only (bool): If true, only invalidate the response cache, not routing cache.

    Returns:
        dict: A response containing:
            - invalidated (int | str): Number of entries invalidated, or "all" if entire cache was cleared.
            - model (str | None): The model that was invalidated, if any.
            - response_cache_only (bool): Whether only response cache was affected.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if cache not initialized.
    """
    if not app_state.router_engine or not app_state.router_engine.semantic_cache:
        return JSONResponse({"error": "Cache not initialized"}, status_code=503)

    async with audit_admin_action(
        request, "cache_invalidate", {"model": model, "response_cache_only": response_cache_only}
    ) as audit:
        cache = app_state.router_engine.semantic_cache
        invalidated: int | str = 0

        if response_cache_only or model:
            invalidated = await cache.invalidate_response(model)
        else:
            await cache.clear()
            invalidated = "all"

        audit.set_result(f"Invalidated: {invalidated}")

    return {
        "invalidated": invalidated,
        "model": model,
        "response_cache_only": response_cache_only,
    }


@router.get("/admin/cache/stats/detailed")
async def get_detailed_cache_stats(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    window_minutes: int = 60,
    interval_minutes: int = 5,
    limit: int = 50,
):
    """Get detailed cache analytics.

    Returns comprehensive statistics about the semantic cache, including time-series
    metrics, top prompts, per-model hit rates, and eviction statistics. This endpoint
    provides deep insight into cache performance and usage patterns.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        window_minutes (int): Time window in minutes for time-series data (default 60).
        interval_minutes (int): Interval bucket size for time-series (default 5).
        limit (int): Maximum number of top prompts to return (default 50).

    Returns:
        dict: A dictionary containing cache analytics. Structure includes:
            - basic (dict): Basic cache stats from get_stats().
            - time_series (list[dict]): Time-series data if time_series_stats enabled.
            - top_prompts (list): Top frequently seen prompts with counts if cache_analytics enabled.
            - model_stats (dict): Per-model cache statistics if cache_analytics enabled.
            - eviction_stats (dict): Eviction counts by reason if cache_analytics enabled.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if cache not initialized.
    """
    if not app_state.router_engine or not app_state.router_engine.semantic_cache:
        return JSONResponse({"error": "Cache not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    cache = app_state.router_engine.semantic_cache
    result: dict[str, Any] = {}

    # Basic cache stats (already enhanced with analytics)
    result["basic"] = await cache.get_stats()

    # Time-series data if cache stats enabled
    if cache.time_series_stats:
        time_series_data = await cache.time_series_stats.get_time_series(
            interval_minutes=interval_minutes,
            window_minutes=window_minutes,
        )
        result["time_series"] = list(time_series_data) if time_series_data else []

    # Advanced analytics if available
    if cache.cache_analytics:
        top_prompts = await cache.cache_analytics.get_top_prompts(limit=limit)
        model_stats = await cache.cache_analytics.get_model_stats()
        eviction_stats = await cache.cache_analytics.get_eviction_stats()

        result["top_prompts"] = list(top_prompts) if top_prompts else []
        result["model_stats"] = dict(model_stats) if model_stats else {}
        result["eviction_stats"] = dict(eviction_stats) if eviction_stats else {}

    return result


@router.post("/admin/cache/clear")
async def clear_cache(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    cache_type: str | None = None,
    model: str | None = None,
    older_than_hours: int | None = None,
):
    """Selectively clear cache entries.

    Provides fine-grained control over clearing semantic cache entries. Can filter by
    cache type (routing, embedding, response), model, and age. This is useful for
    targeted cache invalidation without clearing everything.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        cache_type (str | None): Filter by cache type: 'routing', 'embedding', 'response'. If None, all types.
        model (str | None): If provided, only clear entries associated with this model.
        older_than_hours (int | None): If provided, only clear entries older than this many hours.

    Returns:
        dict: A response containing:
            - cleared (int): Number of cache entries that were cleared.
            - cache_type (str | None): The cache type filter used.
            - model (str | None): The model filter used.
            - older_than_hours (int | None): The age filter used.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if cache not initialized or analytics disabled.
    """
    if not app_state.router_engine or not app_state.router_engine.semantic_cache:
        return JSONResponse({"error": "Cache not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    async with audit_admin_action(
        request, "cache_clear",
        {"cache_type": cache_type, "model": model, "older_than_hours": older_than_hours},
    ) as audit:
        cache = app_state.router_engine.semantic_cache
        if not cache.cache_analytics:
            audit.set_result("Cache analytics not available", status_code=503)
            return JSONResponse(
                {"error": "Cache analytics not available (cache stats disabled?)"},
                status_code=503,
            )

        cleared = await cache.cache_analytics.clear_cache(
            cache_type=cache_type,
            model=model,
            older_than_hours=older_than_hours,
        )
        audit.set_result(f"Cleared {cleared} entries")

    return {
        "cleared": cleared,
        "cache_type": cache_type,
        "model": model,
        "older_than_hours": older_than_hours,
    }


@router.post("/admin/cache/warm")
async def warm_cache(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
):
    """Pre-warm the semantic cache with anticipated prompts.

    Allows administrators to pre-populate the response cache with likely future prompts,
    improving cache hit rates for known workloads. The cache is populated by sending
    the prompts through the router (with model selection) but not actually generating
    responses; instead, a lightweight cache entry is created.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.

    Request Body (JSON):
        - prompts (list[str]): Array of prompt strings to warm the cache with.
        - model (str | None): Optional specific model to use for all prompts; if None, router selects per prompt.

    Returns:
        dict: A response from the underlying cache analytics warm_cache method. Typically includes:
            - warmed (int): Number of prompts that were added to the cache.
            - model (str | None): The model that was used.
            - etc. (depends on implementation).

    Raises:
        HTTPException: 400 if request body invalid or prompts list empty, 401 if admin auth fails,
            429 if rate limited, 503 if cache not initialized or analytics disabled.
    """
    if not app_state.router_engine or not app_state.router_engine.semantic_cache:
        return JSONResponse({"error": "Cache not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    # Parse request body
    try:
        body = await request.json()
        prompts = body.get("prompts", [])
        model = sanitize_model_name(body.get("model"))
    except Exception as e:
        return JSONResponse(
            {"error": f"Invalid request body: {str(e)}"},
            status_code=400,
        )

    if not prompts:
        return JSONResponse(
            {"error": "At least one prompt must be provided in 'prompts' array"},
            status_code=400,
        )

    async with audit_admin_action(
        request, "cache_warm", {"prompt_count": len(prompts), "model": model}
    ) as audit:
        cache = app_state.router_engine.semantic_cache
        if not cache.cache_analytics:
            audit.set_result("Cache analytics not available", status_code=503)
            return JSONResponse(
                {"error": "Cache analytics not available (cache stats disabled?)"},
                status_code=503,
            )

        # Limit number of prompts to prevent abuse
        max_prompts = 1000
        if len(prompts) > max_prompts:
            prompts = prompts[:max_prompts]

        result = await cache.cache_analytics.warm_cache(
            prompts=prompts,
            model=model,
        )
        audit.set_result(f"Warmed cache with {len(prompts)} prompts")

    return result


@router.post("/admin/cache/evict")
async def evict_cache(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    model: str | None = None,
    count: int = 1,
):
    """Manually trigger cache eviction.

    Removes entries from the semantic cache either by LRU (Least Recently Used) policy
    or targeted by model. Useful for freeing cache space without full clear, especially
    when a particular model's entries should be aged out.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        model (str | None): If provided, evict all response cache entries for this model.
        count (int): Number of LRU entries to evict per cache when model is not specified. Clamped to 1-100.

    Returns:
        dict: A response containing:
            - evicted (list[str] | int): If model specified, number of invalidated entries; otherwise list of evicted keys.
            - model (str | None): The model targeted, if any.
            - note (str): Human-readable description of the eviction performed.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if cache not initialized.
    """
    if not app_state.router_engine or not app_state.router_engine.semantic_cache:
        return JSONResponse({"error": "Cache not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    async with audit_admin_action(
        request, "cache_evict", {"model": model, "count": count}
    ) as audit:
        cache = app_state.router_engine.semantic_cache

        # If model specified, evict entries for that model
        if model:
            # Use existing invalidate_response for model-specific eviction
            invalidated = await cache.invalidate_response(model)
            audit.set_result(f"Evicted {invalidated} entries for model {model}")
            return {
                "evicted": invalidated,
                "model": model,
                "note": "Response cache entries evicted for model",
            }

        # Otherwise evict oldest entries from each cache
        if count < 1:
            count = 1
        elif count > 100:
            count = 100  # Limit to prevent abuse

        evicted = await cache.evict_oldest(count=count)
        audit.set_result(f"Evicted oldest {count} entries: {evicted}")

    return {
        "evicted": evicted,
        "model": None,
        "note": f"Evicted oldest {count} entries from each cache",
    }


@router.post("/admin/sync-benchmarks")
async def sync_benchmarks_endpoint(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
):
    """Manually trigger benchmark synchronization.

    Initiates an immediate sync of benchmark data from all configured sources
    (e.g., provider.db download, external benchmark APIs). This operation
    may take some time depending on network and data volume. Typically runs
    in the background, but the endpoint waits for completion before returning.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.

    Returns:
        dict: A response containing:
            - synced (int): Number of benchmark entries that were successfully synchronized.
            - matched_models (int): Number of models matched to local model names.
            - total_models (int): Total number of available models considered during matching.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if backend unavailable.
    """
    if not app_state.backend:
        return JSONResponse({"error": "Backend not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    async with audit_admin_action(request, "sync_benchmarks") as audit:
        models = await get_available_models_with_cache()
        model_names = [m.name for m in models]

        count, matched = await sync_benchmarks(model_names)
        audit.set_result(f"Synced {count} benchmarks, matched {matched} models")

    return {
        "synced": count,
        "matched_models": matched,
        "total_models": len(model_names),
    }


@router.get("/admin/dlq")
async def get_dlq_entries(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List dead letter queue (DLQ) entries.

    Retrieves entries from the dead letter queue, which stores failed background tasks
    that have exceeded their retry limits. Supports optional filtering by status and
    pagination.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        status (str | None): Filter entries by status. One of: 'failed', 'retrying', 'dead', 'resolved'. If None, all statuses.
        limit (int): Maximum number of entries to return (default 50, max 200). Values outside range are clamped.
        offset (int): Number of entries to skip for pagination.

    Returns:
        dict: A response containing:
            - enabled (bool): Whether DLQ is enabled.
            - total (int): Total number of entries matching the filter.
            - limit (int): The page size used.
            - offset (int): The offset used.
            - status (str | None): The status filter applied.
            - entries (list[dict]): List of DLQ entry objects with keys:
                id (int), task_name (str), status (str), attempts (int), max_retries (int),
                error_message (str | None), payload (dict | None), created_at (str | None, ISO8601),
                last_attempt_at (str | None, ISO8601), next_retry_at (str | None, ISO8601),
                resolved_at (str | None, ISO8601).

    Raises:
        HTTPException: 400 if invalid status filter, 401 if admin auth fails, 429 if rate limited.
    """
    await rate_limit_request(request, config, is_admin=True)

    if not settings.dlq_enabled:
        return {"enabled": False, "entries": [], "total": 0}

    if limit < 1:
        limit = 1
    elif limit > 200:
        limit = 200

    if status and status not in {"failed", "retrying", "dead", "resolved"}:
        return JSONResponse(
            {"error": "Invalid status filter. Use failed|retrying|dead|resolved"},
            status_code=400,
        )

    entries = list_dlq_entries(status=status, limit=limit, offset=offset)
    total = count_dlq_entries(status=status)

    return {
        "enabled": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "status": status,
        "entries": [
            {
                "id": e.id,
                "task_name": e.task_name,
                "status": e.status,
                "attempts": e.attempts,
                "max_retries": e.max_retries,
                "error_message": e.error_message,
                "payload": e.payload,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "last_attempt_at": e.last_attempt_at.isoformat() if e.last_attempt_at else None,
                "next_retry_at": e.next_retry_at.isoformat() if e.next_retry_at else None,
                "resolved_at": e.resolved_at.isoformat() if e.resolved_at else None,
            }
            for e in entries
        ],
    }


@router.post("/admin/dlq/retry/{entry_id}")
async def retry_dlq_entry_endpoint(
    entry_id: int,
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
):
    """Manually retry a specific DLQ entry.

    Args:
        entry_id (int): The database ID of the DLQ entry to retry.
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.

    Returns:
        dict: A response containing:
            - entry_id (int): The ID of the DLQ entry that was retried.
            - success (bool): Whether the retry operation succeeded.
            - status (str | None): The new status of the entry after retry attempt (e.g., 'retrying', 'failed').
            - attempts (int | None): The number of attempts after the retry.
            - next_retry_at (str | None): ISO8601 timestamp of next scheduled retry, if any.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if DLQ is disabled.
    """
    await rate_limit_request(request, config, is_admin=True)

    if not settings.dlq_enabled:
        return JSONResponse({"error": "DLQ is disabled"}, status_code=503)

    entry = get_dlq_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "DLQ entry not found"}, status_code=404)

    async with audit_admin_action(request, "dlq_retry", {"entry_id": entry_id}) as audit:
        success = await retry_dlq_entry(entry_id)
        updated = get_dlq_entry(entry_id)
        audit.set_result(
            f"Retry {'succeeded' if success else 'failed'}, status={updated.status if updated else 'unknown'}"
        )

    return {
        "entry_id": entry_id,
        "success": success,
        "status": updated.status if updated else None,
        "attempts": updated.attempts if updated else None,
        "next_retry_at": updated.next_retry_at.isoformat()
        if updated and updated.next_retry_at
        else None,
    }


@router.get("/admin/vram")
async def get_vram_status(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    history_minutes: int = 10,
):
    """Get comprehensive VRAM monitoring data.

    Returns current GPU memory usage, loaded models with their VRAM consumption,
    historical usage trends, and any warnings about high utilization. This endpoint
    is essential for monitoring GPU resource usage and troubleshooting memory issues.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        history_minutes (int): Number of minutes of historical data to return (default 10, max 1440/24h).

    Returns:
        dict: A comprehensive dictionary containing VRAM monitoring data. Structure includes:
            - current (dict | None): Current VRAM snapshot with keys:
                - total_gb (float): Total GPU memory in GB.
                - used_gb (float): Used GPU memory in GB.
                - free_gb (float): Free GPU memory in GB.
                - utilization_pct (float): Percentage of GPU memory in use.
                - timestamp (float): Unix timestamp of the snapshot.
            - budget (dict): VRAM budget configuration and current allocation:
                - max_configured_gb (float): Maximum configured VRAM limit.
                - headroom_gb (float): Fragmentation buffer (typically 1.5GB).
                - available_gb (float | None): Currently available VRAM considering allocations.
                - allocated_gb (float | None): Currently allocated VRAM by models.
                - utilization_pct (float | None): Percentage of allocated vs available.
            - models_loaded (dict[str, float]): Mapping of model name to VRAM usage in GB.
            - history (list[dict]): Time-series history, each entry contains:
                - timestamp (float): Unix timestamp.
                - used_gb (float): Used memory at that time.
                - util_pct (float): Utilization percentage.
                - models (dict[str, float]): Models loaded at that time.
            - warnings (list[str]): List of warning messages about high utilization or thresholds.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited, 503 if VRAM monitoring not enabled.
    """
    if not hasattr(app_state, "vram_monitor") or app_state.vram_monitor is None:
        return {"error": "VRAM monitoring not enabled"}

    monitor: VRAMMonitor = app_state.vram_monitor
    vram_manager: VRAMManager | None = app_state.vram_manager

    current = monitor.get_current()
    history = monitor.get_history(minutes=history_minutes)

    response: dict[str, Any] = {
        "current": None,
        "budget": {
            "max_configured_gb": monitor.total_vram_gb,
            "headroom_gb": VRAMManager.FRAGMENTATION_BUFFER_GB,
            "available_gb": vram_manager.get_available_vram() if vram_manager else None,
            "allocated_gb": vram_manager.get_current_allocated() if vram_manager else None,
            "utilization_pct": round(vram_manager.get_utilization_pct(), 1)
            if vram_manager
            else None,
        },
        "models_loaded": {},
        "history": [],
        "warnings": [],
    }

    if current:
        response["current"] = {
            "total_gb": round(current.total_gb, 2),
            "used_gb": round(current.used_gb, 2),
            "free_gb": round(current.free_gb, 2),
            "utilization_pct": round(current.utilization_pct, 1),
            "timestamp": current.timestamp,
        }
        response["models_loaded"] = {
            model: round(vram, 2) for model, vram in current.per_model_vram_gb.items()
        }

        # Generate warnings
        util = current.utilization_pct
        if util >= 95:
            response["warnings"].append("CRITICAL: VRAM > 95% - immediate unload recommended")
        elif util >= 85:
            response["warnings"].append("WARNING: VRAM > 85% - consider unloading models")
        elif util >= 75:
            response["warnings"].append("NOTICE: VRAM > 75% - monitor usage")

        # Check against configured threshold for proactive unload
        if vram_manager and vram_manager.auto_unload:
            threshold = settings.vram_unload_threshold_pct
            if util >= threshold:
                response["warnings"].append(
                    f"Auto-unload threshold ({threshold}%) reached or exceeded"
                )

    response["history"] = [
        {
            "timestamp": h.timestamp,
            "used_gb": round(h.used_gb, 2),
            "util_pct": round(h.utilization_pct, 1),
            "models": h.models_loaded,
        }
        for h in history
    ]

    return response


@router.get("/admin/explain")
async def explain_routing(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    prompt: str,
    model_override: str | None = None,
):
    """Explain routing decision for a prompt.

    Provides detailed insight into why a particular model was or would be selected
    for a given prompt. Includes scoring breakdown, category analysis, confidence,
    and comparison of all available models. Does not generate a response, making it
    safe to use for debugging and understanding the routing logic.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        prompt (str): The prompt text to analyze for routing decisions.
        model_override (str | None): If provided, forces selection of this model and returns
            explanation based on override rather than full scoring.

    Returns:
        dict: A comprehensive explanation object containing:
            - prompt (str): The original prompt.
            - prompt_preview (str): Truncated/preview version of prompt for display.
            - selected_model (str): The model that would be/was selected.
            - confidence (float): Confidence score (0.0-1.0) in the selection.
            - reasoning (str): Human-readable explanation of the selection.
            - cached (bool): Whether the decision came from routing cache.
            - available_models (int): Count of models considered.
            - analysis (dict): Prompt category analysis with keys:
                - categories (dict[str, float]): Category scores (reasoning, coding, creativity, factual, speed, general).
                - dominant_category (str | None): Category with highest score.
            - scoring_weights (dict): Current routing weights and preferences:
                - quality_preference (float): Quality vs speed tradeoff (0.0-1.0).
                - speed_weight (float): Derived weight for speed (1.0 - quality_preference).
                - quality_weight (float): Derived weight for quality (quality_preference + 0.2).
            - model_scores (list[dict]): Ranked list of all models with detailed scores. Each dict contains:
                - name (str), reasoning (float), coding (float), creativity (float), speed (float),
                - vram_gb (float | None), total_score (float), base_score (float),
                - coding_score (float), creativity_score (float), reasoning_score (float),
                - factual_score (float), speed_score (float), feedback_boost (float),
                - diversity_penalty (float).
            - scoring_breakdown (dict[str, dict]): Raw scoring data by model name.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited,
            400 if model_override is invalid,
            503 if backend or router engine not initialized,
            500 if routing analysis fails.
    """
    if not app_state.backend or not app_state.router_engine:
        return JSONResponse(
            {"error": {"message": "Service not ready", "type": "service_unavailable"}},
            status_code=503,
        )

    try:
        model_override = sanitize_model_name(model_override)
    except ValueError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=400,
        )

    try:
        # Get available models
        available_models = await get_available_models_with_cache()
        if not available_models:
            return JSONResponse(
                {"error": {"message": "No models available", "type": "internal_error"}},
                status_code=500,
            )

        # If model override provided, explain that specific model
        if model_override:
            model_names = [m.name for m in available_models]
            selected_model = None
            for name in model_names:
                if name == model_override or model_override.lower() in name.lower():
                    selected_model = name
                    break

            if not selected_model:
                return JSONResponse(
                    {
                        "error": {
                            "message": f"Model '{model_override}' not found",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=400,
                )

            return {
                "prompt": prompt,
                "selected_model": selected_model,
                "override": True,
                "reasoning": f"Model override specified: {model_override}",
                "confidence": 1.0,
                "scoring_breakdown": None,
            }

        # Otherwise, run the full routing logic to get scoring breakdown
        model_list = [m.name for m in available_models]

        # Check routing cache first
        cached_result = None
        if app_state.router_engine.semantic_cache:
            cached_result = await app_state.router_engine.semantic_cache.get(prompt)

        if cached_result:
            # Still compute scoring breakdown for explanation
            selected_model = cached_result.selected_model
            confidence = cached_result.confidence
            reasoning = cached_result.reasoning
            cached = True
        else:
            # Get routing decision
            result = await app_state.router_engine.select_model(prompt, model_list)

            if result is None:
                return JSONResponse(
                    {"error": {"message": "Could not select model", "type": "internal_error"}},
                    status_code=500,
                )

            selected_model = result.selected_model
            confidence = result.confidence
            reasoning = result.reasoning
            cached = False

        # Enhanced scoring breakdown
        from router.router import get_benchmarks_for_models_with_external

        # Get all components used for scoring
        profiles = await app_state.router_engine._get_all_profiles()
        benchmarks = get_benchmarks_for_models_with_external(model_list)
        analysis = app_state.router_engine._analyze_prompt(prompt, None)
        feedback_scores = app_state.router_engine._get_model_feedback_scores()

        # Get model frequencies for diversity penalty
        model_frequencies: dict[str, float] = {}
        if app_state.router_engine.semantic_cache:
            freq_tasks = [
                app_state.router_engine.semantic_cache.get_model_frequency(m) for m in model_list
            ]
            freq_results = await asyncio.gather(*freq_tasks)
            model_frequencies = dict(zip(model_list, freq_results, strict=False))

        # Calculate combined scores for all models
        scoring_breakdown = app_state.router_engine._calculate_combined_scores(
            profiles, benchmarks, analysis, model_list, feedback_scores, model_frequencies
        )

        # Get profile details for each model
        with get_session() as session:
            db_profiles = (
                session.query(ModelProfile).filter(ModelProfile.name.in_(model_list)).all()
            )
            profile_map = {p.name: p for p in db_profiles}

        # Build detailed model scores
        model_scores = []
        for model_name in model_list:
            profile = profile_map.get(model_name)
            breakdown = scoring_breakdown.get(model_name, {})
            model_scores.append(
                {
                    "name": model_name,
                    "reasoning": profile.reasoning if profile else 0.0,
                    "coding": profile.coding if profile else 0.0,
                    "creativity": profile.creativity if profile else 0.0,
                    "speed": profile.speed if profile else 0.0,
                    "vram_gb": profile.vram_required_gb if profile else None,
                    "total_score": breakdown.get("score", 0.0),
                    "base_score": breakdown.get("base_score", 0.0),
                    "coding_score": breakdown.get("coding", 0.0),
                    "creativity_score": breakdown.get("creativity", 0.0),
                    "reasoning_score": breakdown.get("reasoning", 0.0),
                    "factual_score": breakdown.get("factual", 0.0),
                    "speed_score": breakdown.get("speed", 0.0),
                    "feedback_boost": breakdown.get("feedback_boost", 0.0),
                    "diversity_penalty": breakdown.get("diversity_penalty", 0.0),
                }
            )

        # Sort by total score descending
        model_scores.sort(key=lambda x: x["total_score"], reverse=True)

        return {
            "prompt": prompt,
            "prompt_preview": sanitize_for_logging(prompt)[:100] + "..."
            if len(prompt) > 100
            else prompt,
            "selected_model": selected_model,
            "confidence": confidence,
            "reasoning": reasoning,
            "cached": cached,
            "available_models": len(model_list),
            "analysis": {
                "categories": analysis,
                "dominant_category": max(analysis.items(), key=lambda x: x[1])[0]
                if analysis
                else None,
            },
            "scoring_weights": {
                "quality_preference": settings.quality_preference,
                "speed_weight": 1.0 - settings.quality_preference,
                "quality_weight": settings.quality_preference
                + 0.2,  # SCORING_CONFIG["quality_preference_boost"]
            },
            "model_scores": model_scores,
            "scoring_breakdown": scoring_breakdown,
        }

    except Exception as e:
        _log_error_with_context(
            "Explain routing failed",
            request=request,
            model_name=model_override,
            prompt=prompt,
            exc=e,
            exc_info=True,
        )
        return JSONResponse(
            {"error": {"message": f"Explain routing failed: {str(e)}", "type": "internal_error"}},
            status_code=500,
        )


@router.get("/admin/audit-log")
async def get_admin_audit_log(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    action: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Query admin audit log entries.

    Retrieves a paginated list of admin action audit logs. Audit logs record all
    administrative operations (reprofile, cache operations, benchmark sync, etc.)
    including who performed them, when, from which IP, and with what result.

    Args:
        request (Request): The incoming FastAPI request.
        _ (bool): Admin authentication dependency (verify_admin_token).
        config (Settings): Application configuration dependency.
        action (str | None): Filter by action type (e.g. 'reprofile', 'cache_clear', 'sync_benchmarks', 'dlq_retry').
        limit (int): Maximum number of entries to return (default 50, max 200). Values outside range are clamped.
        offset (int): Number of entries to skip for pagination.

    Returns:
        dict: A paginated response containing:
            - enabled (bool): Whether admin audit logging is enabled.
            - total (int): Total number of entries matching the filter.
            - limit (int): The page size used.
            - offset (int): The offset used.
            - action_filter (str | None): The action filter applied, if any.
            - entries (list[dict]): List of audit log entry objects with keys:
                - id (int): Unique identifier of the audit log entry.
                - timestamp (str | None): ISO8601 timestamp of the action.
                - action (str): Type of action performed.
                - endpoint (str): API endpoint that was called.
                - method (str): HTTP method (GET, POST, etc.).
                - ip_address (str | None): IP address of the admin user.
                - user_agent (str | None): User agent string from the request.
                - parameters (dict | None): Request parameters (query/body) that were used.
                - result_summary (str | None): Summary of the action's result.
                - status_code (int | None): HTTP status code returned.
                - duration_ms (float | None): Duration of the operation in milliseconds.

    Raises:
        HTTPException: 401 if admin authentication fails, 429 if rate limited.
    """
    await rate_limit_request(request, config, is_admin=True)

    if not config.admin_audit_enabled:
        return {"enabled": False, "entries": [], "total": 0}

    limit = min(max(1, limit), 200)
    offset = max(0, offset)

    entries, total = get_audit_logs(action=action, limit=limit, offset=offset)

    return {
        "enabled": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "action_filter": action,
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "action": e.action,
                "endpoint": e.endpoint,
                "method": e.method,
                "ip_address": e.ip_address,
                "user_agent": e.user_agent,
                "parameters": e.parameters,
                "result_summary": e.result_summary,
                "status_code": e.status_code,
                "duration_ms": round(e.duration_ms, 2) if e.duration_ms else None,
            }
            for e in entries
        ],
    }
