"""Health check, root, and Prometheus metrics endpoints.

This module provides basic operational endpoints:
- GET / : Service status and version
- GET /health: Detailed health check with subsystem statuses
- GET /metrics: Prometheus-format metrics for monitoring
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from router.config import Settings, settings
from router.database import get_session
from router.dlq import count_dlq_entries
from router.logging_config import get_request_id
from router.state import app_state, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint returning service status and version.

    Used for simple liveness checks (e.g., Kubernetes readinessProbe can
    check /health for more thorough health status).

    Returns:
        JSON with keys: status, service, version.
    """
    return {
        "status": "running",
        "service": "SmarterRouter",
        "version": "2.2.1",
    }


@router.get("/health")
async def health(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
):
    """Comprehensive health check endpoint.

    Checks multiple subsystems and returns a detailed status report:
    - database: SQLite connectivity
    - backend: LLM backend initialization
    - gpu: VRAM monitoring data or status
    - cache: Redis or memory cache status
    - background_tasks: number of active background tasks
    - dlq: dead letter queue counts (if enabled)

    Args:
        request: Incoming request.
        config: Settings dependency.

    Returns:
        JSON with keys: status (healthy/unhealthy), checks (dict), version, request_id.
    """
    checks: dict[str, Any] = {}
    overall_status = "healthy"

    # 1. Database connectivity
    try:
        with get_session() as session:
            from sqlalchemy import text

            session.execute(text("SELECT 1")).scalar()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        overall_status = "unhealthy"

    # 2. Backend availability
    if app_state.backend:
        checks["backend"] = "initialized"
    else:
        checks["backend"] = "not available"
        overall_status = "unhealthy"

    # 3. GPU monitoring
    if app_state.vram_monitor:
        try:
            metrics = app_state.vram_monitor.get_current()
            if metrics:
                checks["gpu"] = {
                    "total_gb": round(metrics.total_gb, 2),
                    "used_gb": round(metrics.used_gb, 2),
                    "free_gb": round(metrics.free_gb, 2),
                    "vendor": metrics.gpus[0].vendor if metrics.gpus else "unknown",
                }
            else:
                checks["gpu"] = "no data"
        except Exception as e:
            checks["gpu"] = f"error: {str(e)}"
    else:
        checks["gpu"] = "unavailable"

    # 4. Cache (Redis if configured)
    try:
        if config.cache_backend == "redis":
            try:
                import redis as redis_client

                r = redis_client.Redis.from_url(config.redis_url or "redis://localhost:6379/0")
                if r.ping():
                    checks["redis"] = "connected"
                else:
                    checks["redis"] = "ping failed"
                    overall_status = "unhealthy"
            except ImportError:
                checks["redis"] = "redis library not installed"
        else:
            checks["cache"] = "memory (ok)"
    except Exception as e:
        checks["cache"] = f"error: {str(e)}"
        overall_status = "unhealthy"

    # 5. Background tasks
    checks["background_tasks"] = len(app_state.background_tasks)

    # 6. DLQ status
    if settings.dlq_enabled:
        try:
            checks["dlq"] = {
                "failed": count_dlq_entries("failed"),
                "retrying": count_dlq_entries("retrying"),
                "dead": count_dlq_entries("dead"),
            }
        except Exception as e:
            checks["dlq"] = f"error: {str(e)}"

    return {
        "status": overall_status,
        "checks": checks,
        "version": "2.2.1",
        "request_id": get_request_id(),
    }


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint.

    Returns:
        Response with text/plain content type containing metrics in Prometheus format.
    """
    from fastapi.responses import Response

    from router.metrics import generate_metrics

    return Response(content=generate_metrics(), media_type="text/plain; version=0.0.4")
