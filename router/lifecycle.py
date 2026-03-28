"""
Application lifecycle: startup, shutdown, background tasks.

This module contains all application lifecycle event handlers and background
tasks that were previously in main.py:
- startup_event: Initializes backend, router engine, VRAM monitoring, etc.
- shutdown_event: Cleans up resources on application shutdown
- lifespan: Async context manager wiring startup/shutdown to FastAPI
- background_sync_task: Periodic benchmark sync, provider.db download, model profiling
- download_provider_db: Downloads provider.db from GitHub
- background_cache_cleanup_task: Cleans expired persistent cache entries
- retry_dlq_entry: Retries failed dead letter queue entries
- background_dlq_retry_task: Periodic auto-retry of failed DLQ entries
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI

from router.backends import create_backend
from router.backends.base import supports_unload
from router.benchmark_db import get_last_sync
from router.benchmark_sync import sync_benchmarks
from router.config import init_logging, settings
from router.database import get_session, init_db
from router.dlq import (
    enqueue_failed_task,
    get_dlq_entry,
    get_due_retry_entries,
    mark_retry_failure,
    mark_retry_success,
)
from router.model_filter import filter_model_infos, log_filter_summary
from router.profiler import profile_all_models
from router.router import RouterEngine
from router.state import (
    app_state,
    get_available_models_with_cache,
    get_model_vram_estimate,
)
from router.vram_manager import VRAMManager
from router.vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)


async def startup_event() -> None:
    """Initialize the application during startup.

    This function is called by the FastAPI lifespan context manager when the
    application starts. It performs the following steps:

    1. Initialize logging via init_logging().
    2. Initialize the database via init_db().
    3. Create the LLM backend instance (Ollama, llama.cpp, OpenAI, etc.).
    4. Create the RouterEngine and pre-warm caches with available models.
    5. Initialize VRAM monitoring and management if enabled.
    6. Pre-load the pinned model if configured.
    7. Start background tasks: benchmark sync, cache cleanup, DLQ retry.

    If any step fails, the error is logged but the application continues
    to run, allowing for partial functionality or later restart.

    Raises:
        No exceptions are raised; errors are caught and logged.
    """
    init_logging()
    init_db()
    logger.info("Starting SmarterRouter...")

    # Security warning for production
    if not settings.admin_api_key:
        logger.warning(
            "SECURITY: ROUTER_ADMIN_API_KEY is not set! "
            "Admin endpoints are publicly accessible. "
            "Set this in production to protect sensitive data."
        )

    # Initialize backend
    try:
        app_state.backend = create_backend(settings)
        logger.info(f"Initialized backend: {settings.provider}")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        # Don't crash, allow retry or partial functionality

    # Initialize router engine
    if app_state.backend:
        app_state.router_engine = RouterEngine(
            client=app_state.backend,
            dispatcher_model=settings.router_model,
            cache_enabled=settings.cache_enabled,
            cache_max_size=settings.cache_max_size,
            cache_ttl_seconds=settings.cache_ttl_seconds,
            cache_similarity_threshold=settings.cache_similarity_threshold,
            cache_response_max_size=settings.cache_response_max_size,
            embed_model=settings.embed_model,
            persistent_cache_enabled=settings.persistent_cache_enabled,
            persistent_cache_max_age_days=settings.persistent_cache_max_age_days,
        )

        try:
            available_models = await get_available_models_with_cache()

            # Apply model filtering if configured
            include = settings.model_filter_include
            exclude = settings.model_filter_exclude
            if include or exclude:
                original_count = len(available_models) if available_models else 0
                available_models = filter_model_infos(available_models, include, exclude)
                excluded_count = original_count - (len(available_models) if available_models else 0)
                log_filter_summary(
                    original_count,
                    len(available_models) if available_models else 0,
                    excluded_count,
                    include,
                    exclude,
                )

            model_names = [m.name for m in available_models] if available_models else []
            # Load persistent cache first, then warm up other caches
            await app_state.router_engine.load_persistent_cache()
            await app_state.router_engine.warmup_caches(model_names)
        except Exception as e:
            logger.warning(f"Failed to pre-warm router caches: {e}")

    # Initialize VRAM monitor
    vram_monitor: VRAMMonitor | None = None
    if settings.vram_monitor_enabled:
        vram_monitor = VRAMMonitor(
            interval=settings.vram_monitor_interval,
            total_vram_gb=settings.vram_max_total_gb,
            app_state=app_state,
            log_interval=settings.vram_log_interval,
            apple_unified_memory_gb=settings.apple_unified_memory_gb,
            amd_unified_memory_gb=settings.amd_unified_memory_gb,
        )
        app_state.vram_monitor = vram_monitor
        await vram_monitor.start()

        # Auto-detect total VRAM if not configured
        if settings.vram_max_total_gb is None and vram_monitor.has_gpu:
            current = vram_monitor.get_current()
            if current:
                # Suggest using 90% of total as safe default
                suggested_max = current.total_gb * 0.90
                settings.vram_max_total_gb = suggested_max
                logger.info(
                    f"Auto-detected GPU VRAM: {current.total_gb:.1f}GB. "
                    f"Defaulting ROUTER_VRAM_MAX_TOTAL_GB to {suggested_max:.1f}GB (90%). "
                    f"Adjust this setting if needed."
                )

    # Initialize VRAM manager
    max_vram = settings.vram_max_total_gb or 24.0  # Fallback to 24GB if detection failed
    vram_manager = VRAMManager(
        max_vram_gb=max_vram,
        auto_unload_enabled=settings.vram_auto_unload_enabled,
        unload_strategy=settings.vram_unload_strategy,
        monitor=vram_monitor,
    )
    app_state.vram_manager = vram_manager

    # Connect VRAM manager to router engine and backend
    if app_state.router_engine:
        app_state.router_engine.vram_manager = vram_manager
    if app_state.backend:
        vram_manager.set_backend(app_state.backend)

        # Pre-load pinned model if configured (improves first-response latency)
        if settings.pinned_model:
            logger.info(f"Pre-loading pinned model: {settings.pinned_model}")
            try:
                # Estimate VRAM for pinned model
                vram_gb = get_model_vram_estimate(settings.pinned_model)
                await vram_manager.load_model(settings.pinned_model, vram_gb, pin=True)
                app_state.current_loaded_model = settings.pinned_model
            except Exception as e:
                logger.warning(f"Failed to pre-load pinned model {settings.pinned_model}: {e}")

    # Start background sync task
    # Should run if:
    # 1. We're using Ollama (needs profiling + sync)
    # 2. OR provider.db auto-update is enabled (needs download)
    if settings.provider == "ollama" or (
        settings.provider_db_enabled and settings.provider_db_auto_update_hours > 0
    ):
        task = asyncio.create_task(background_sync_task())
        app_state.background_tasks.add(task)
        task.add_done_callback(app_state.background_tasks.discard)

    # Start periodic persistent cache cleanup task if enabled
    if settings.cache_cleanup_interval_hours > 0:
        task = asyncio.create_task(background_cache_cleanup_task())
        app_state.background_tasks.add(task)
        task.add_done_callback(app_state.background_tasks.discard)

    # Start DLQ auto-retry task
    if settings.dlq_enabled:
        task = asyncio.create_task(background_dlq_retry_task())
        app_state.background_tasks.add(task)
        task.add_done_callback(app_state.background_tasks.discard)


async def shutdown_event() -> None:
    """Perform cleanup tasks during application shutdown.

    This function is called by the FastAPI lifespan context manager when the
    application is shutting down (e.g., on SIGTERM). It performs the following:

    1. Cancel all background tasks and wait up to 5 seconds for them to finish.
    2. Unload the pinned model if backend supports unloading.
    3. Close the backend HTTP client for connection cleanup.
    4. Close distributed cache connections (e.g., Redis).

    Errors during cleanup are logged as warnings but do not prevent shutdown.
    """
    logger.info("Shutting down SmarterRouter...")

    # Cancel all background tasks
    tasks = list(app_state.background_tasks)
    for task in tasks:
        task.cancel()

    # Wait for tasks to complete with timeout (5 seconds)
    if tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
        except TimeoutError:
            logger.warning("Some background tasks did not cancel gracefully within timeout")

    # Unload model if pinned
    if settings.pinned_model and app_state.backend:
        if supports_unload(app_state.backend):
            await app_state.backend.unload_model(settings.pinned_model)

    # Close backend HTTP client for connection cleanup
    if app_state.backend and hasattr(app_state.backend, "close"):
        try:
            await app_state.backend.close()
        except Exception as e:
            logger.warning(f"Error closing backend: {e}")

    # Close distributed cache connections (e.g., Redis)
    try:
        from router.cache import close_all_caches

        close_all_caches()
    except Exception as e:
        logger.warning(f"Error closing cache connections: {e}")


async def background_sync_task() -> None:
    """Background task for periodic benchmark sync, provider.db download, and model profiling.

    Runs in an infinite loop with a sleep interval controlled by
    ROUTER_POLLING_INTERVAL. Performs three main operations:

    1. Download provider.db if enabled and auto-update is configured (with exponential backoff on failures).
    2. Refresh model availability and profiles (if model polling enabled).
    3. Profile new models and sync benchmarks daily.

    All errors are logged and, where applicable, enqueued into the DLQ for retry.
    """
    """Background task to sync benchmarks, download provider.db, and profile new models."""
    provider_db_last_download: float | None = None
    provider_db_failures = 0
    backoff_base = 60  # seconds
    last_model_poll_time: float | None = None

    while True:
        try:
            # 1. Download provider.db if enabled
            if settings.provider_db_enabled and settings.provider_db_auto_update_hours > 0:
                # Calculate backoff if previously failed
                backoff = (
                    min(backoff_base * (2**provider_db_failures), 3600)
                    if provider_db_failures > 0
                    else 0
                )

                hours_since_download = (
                    (time.time() - provider_db_last_download) / 3600
                    if provider_db_last_download
                    else float("inf")
                )

                # Check if it's time to download (considering backoff)
                if (
                    hours_since_download >= settings.provider_db_auto_update_hours
                    and time.time() > (provider_db_last_download or 0) + backoff
                ):
                    logger.info("Downloading latest provider.db...")
                    success = await download_provider_db()
                    if success:
                        provider_db_last_download = time.time()
                        provider_db_failures = 0
                        # Invalidate caches to pick up new benchmarks
                        if app_state.router_engine:
                            app_state.router_engine.invalidate_caches()
                    else:
                        provider_db_failures += 1
                        wait_time = min(backoff_base * (2**provider_db_failures), 3600)
                        enqueue_failed_task(
                            task_name="provider_db_download",
                            error_message=(
                                "Provider.db download failed during background sync "
                                f"(attempt {provider_db_failures})"
                            ),
                            payload={
                                "attempt": provider_db_failures,
                                "backoff_seconds": wait_time,
                            },
                        )
                        logger.warning(
                            f"Provider.db download failed (attempt {provider_db_failures}), backing off for {wait_time}s"
                        )
                        # We don't sleep here, we just won't try again until backoff expires

            # Model Polling & Availability Refresh (SmarterRouter 2.1.6+)
            if settings.model_polling_enabled and app_state.router_engine:
                now = time.time()
                if (
                    last_model_poll_time is None
                    or (now - last_model_poll_time) >= settings.model_polling_interval
                ):
                    logger.debug("Running model availability refresh")
                    try:
                        await app_state.router_engine.refresh_models(
                            cleanup=settings.model_cleanup_enabled
                        )
                    except Exception as e:
                        logger.warning(f"Model refresh failed: {e}")
                    else:
                        last_model_poll_time = now

            if app_state.backend:
                # 2. Sync Benchmarks (once per day or on startup)
                with get_session():
                    last_sync = get_last_sync()
                    should_sync = False
                    if not last_sync:
                        should_sync = True
                    elif (datetime.now(UTC) - last_sync.replace(tzinfo=UTC)).days >= 1:
                        should_sync = True

                if should_sync:
                    logger.info("Starting benchmark sync...")
                    # Get available model names to match against benchmarks
                    models = await get_available_models_with_cache()

                    # Apply model filtering if configured
                    include = settings.model_filter_include
                    exclude = settings.model_filter_exclude
                    if include or exclude:
                        original_count = len(models)
                        models = filter_model_infos(models, include, exclude)
                        excluded_count = original_count - len(models)
                        log_filter_summary(
                            original_count, len(models), excluded_count, include, exclude
                        )

                    model_names = [m.name for m in models]
                    try:
                        await sync_benchmarks(model_names)
                    except Exception as e:
                        enqueue_failed_task(
                            task_name="benchmark_sync",
                            error_message=str(e),
                            payload={"model_count": len(model_names)},
                        )
                        raise
                    # Invalidate router caches after benchmark sync
                    if app_state.router_engine:
                        app_state.router_engine.invalidate_caches()

                # 3. Profile New Models (optional)
                # This will only profile models that haven't been profiled yet
                if settings.model_auto_profile_enabled:
                    try:
                        await profile_all_models(app_state.backend)
                    except ValueError as e:
                        if "No models available after filtering" in str(e):
                            logger.debug(f"No models to profile: {e}")
                        else:
                            enqueue_failed_task(
                                task_name="profile_all_models",
                                error_message=str(e),
                            )
                            raise
                    except Exception as e:
                        enqueue_failed_task(
                            task_name="profile_all_models",
                            error_message=str(e),
                        )
                        raise
                else:
                    logger.debug("Model auto-profiling disabled; skipping profile_all_models")

        except Exception as e:
            enqueue_failed_task(
                task_name="background_sync_task",
                error_message=str(e),
            )
            logger.error(f"Background sync task failed: {e}")

        await asyncio.sleep(settings.polling_interval)


async def download_provider_db() -> bool:
    """Download the latest provider.db from the configured GitHub URL.

    Uses httpx with TLS verification from settings. Performs atomic write via
    a temporary file and validates the SQLite database before replacing the
    existing file.

    Returns:
        True if download and validation succeeded, False otherwise.
    """
    import httpx

    db_path = settings.provider_db_path
    download_url = settings.provider_db_download_url

    try:
        # Create parent directory if needed
        from pathlib import Path

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=60.0, verify=settings.verify_tls) as client:
            response = await client.get(download_url)
            response.raise_for_status()

            # Write to temp file first (offload blocking I/O to thread)
            temp_path = f"{db_path}.tmp"

            def _write_temp() -> None:
                with open(temp_path, "wb") as f:
                    f.write(response.content)

            await asyncio.to_thread(_write_temp)

            # Verify it's a valid SQLite database
            import sqlite3

            with sqlite3.connect(temp_path) as conn:
                conn.execute("SELECT COUNT(*) FROM model_benchmarks")

            # Atomic rename
            import shutil

            shutil.move(temp_path, db_path)

            logger.info("provider.db updated successfully")
            return True

    except Exception as e:
        logger.error(f"Failed to download provider.db: {e}")
        # Clean up temp file if it exists
        try:
            import os

            if os.path.exists(f"{db_path}.tmp"):
                os.remove(f"{db_path}.tmp")
        except Exception:
            pass
        return False


async def background_cache_cleanup_task() -> None:
    """Periodically delete expired entries from the persistent cache.

    Runs forever, sleeping for ROUTER_CACHE_CLEANUP_INTERVAL_HOURS between
    executions. Only active if persistent cache is enabled.
    """
    cleanup_interval = settings.cache_cleanup_interval_hours * 3600.0
    while True:
        try:
            await asyncio.sleep(cleanup_interval)
            # Perform cleanup
            if app_state.router_engine and app_state.router_engine.semantic_cache:
                pc = app_state.router_engine.semantic_cache.persistent_cache
                if pc and pc.enabled:
                    logger.info("Starting persistent cache cleanup...")
                    counts = await pc.delete_expired_entries()
                    logger.info(
                        f"Persistent cache cleanup complete: {counts.get('routing',0)} routing, {counts.get('response',0)} response, {counts.get('embedding',0)} embedding entries removed"
                    )
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
            break
        except Exception as e:
            enqueue_failed_task(
                task_name="background_cache_cleanup",
                error_message=str(e),
            )
            logger.error(f"Cache cleanup task error: {e}")


async def retry_dlq_entry(entry_id: int) -> bool:
    """Retry a specific DLQ entry by task name.

    Supported task names:
    - provider_db_download
    - benchmark_sync
    - profile_all_models
    - background_cache_cleanup

    On success, marks the entry as resolved. On failure, updates the retry
    count and schedules next retry with backoff.

    Args:
        entry_id: The DLQ entry ID to retry.

    Returns:
        True if the retry succeeded, False otherwise.
    """
    """Retry a DLQ entry by task name using best-effort execution."""
    entry = get_dlq_entry(entry_id)
    if not entry:
        return False

    try:
        if entry.task_name == "provider_db_download":
            ok = await download_provider_db()
            if not ok:
                raise RuntimeError("provider.db download failed during retry")
        elif entry.task_name == "benchmark_sync":
            models = await get_available_models_with_cache()
            model_names = [m.name for m in models]
            await sync_benchmarks(model_names)
        elif entry.task_name == "profile_all_models":
            if not app_state.backend:
                raise RuntimeError("Backend not initialized")
            await profile_all_models(app_state.backend)
        elif entry.task_name == "background_cache_cleanup":
            if app_state.router_engine and app_state.router_engine.semantic_cache:
                pc = app_state.router_engine.semantic_cache.persistent_cache
                if pc and pc.enabled:
                    await pc.delete_expired_entries()
        else:
            raise RuntimeError(f"Unknown DLQ task: {entry.task_name}")

        mark_retry_success(entry_id)
        return True
    except Exception as e:
        mark_retry_failure(entry_id, str(e))
        logger.error(f"DLQ retry failed for entry {entry_id}: {e}")
        return False


async def background_dlq_retry_task() -> None:
    """Periodically attempt to retry DLQ entries that are due for retry.

    Sleeps for ROUTER_DLQ_RETRY_BASE_DELAY_SECONDS (or longer if
    configured). Fetches due entries (status='retrying', next_retry_at <= now)
    in batches and retries them via retry_dlq_entry().
    """
    while True:
        try:
            await asyncio.sleep(max(10, settings.dlq_retry_base_delay_seconds))
            if not settings.dlq_enabled:
                continue

            entries = get_due_retry_entries(limit=settings.dlq_auto_retry_batch_size)
            for entry in entries:
                await retry_dlq_entry(entry.id)
        except asyncio.CancelledError:
            logger.info("DLQ retry task cancelled")
            break
        except Exception as e:
            logger.error(f"DLQ retry task error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.

    On startup: calls startup_event() to initialize all components.
    On shutdown: calls shutdown_event() to clean up resources.

    This is attached to the FastAPI app in main.py.
    """
    await startup_event()
    yield
    await shutdown_event()
