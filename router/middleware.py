"""
HTTP middleware for SmarterRouter.

This module contains all HTTP middleware functions that process incoming requests
and outgoing responses. Middleware functions include:
- request_size_middleware: Limits request body size
- request_id_middleware: Adds request ID for tracing
- request_timeout_middleware: Enforces overall request timeout
- slow_query_middleware: Logs slow queries with stack traces
- metrics_middleware: Collects Prometheus metrics

The register_middleware function is used to attach all middleware to the FastAPI app.
"""

import asyncio
import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from router.config import settings
from router.logging_config import get_request_id, set_request_id
from router.metrics import ERRORS_TOTAL, REQUEST_DURATION, REQUESTS_TOTAL
from router.state import _log_error_with_context

logger = logging.getLogger(__name__)


async def request_size_middleware(request: Request, call_next):
    """Enforce maximum request body size to prevent memory exhaustion.

    The maximum size is controlled by the ROUTER_MAX_REQUEST_BODY_BYTES
    configuration setting. If the request body exceeds this limit, the
    server responds with HTTP 413 (Payload Too Large).

    Args:
        request: The incoming FastAPI Request.
        call_next: The next middleware or endpoint in the chain.

    Returns:
        The response from the next middleware or endpoint, or a JSON error
        with status 413 if the body is too large.
    """
    max_size = settings.max_request_body_bytes

    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
                if declared_size > max_size:
                    max_mb = max_size / (1024 * 1024)
                    return JSONResponse(
                        {
                            "error": {
                                "message": f"Request body too large (max {max_mb:.0f}MB)",
                                "type": "invalid_request_error",
                            }
                        },
                        status_code=413,
                    )

                # Fast path: declared content length is within limits.
                # Let downstream parse the body normally without buffering here.
                return await call_next(request)
            except ValueError:
                # Invalid content-length header; fall back to explicit body check.
                pass

        body = await request.body()
        if len(body) > max_size:
            max_mb = max_size / (1024 * 1024)
            return JSONResponse(
                {
                    "error": {
                        "message": f"Request body too large (max {max_mb:.0f}MB)",
                        "type": "invalid_request_error",
                    }
                },
                status_code=413,
            )

        # Re-create request with body for next middleware
        async def receive():
            return {"type": "http.request", "body": body}

        request = Request(request.scope, receive, request._send)

    return await call_next(request)


async def request_id_middleware(request: Request, call_next):
    """Assign or propagate a request ID for distributed tracing.

    Checks for an incoming X-Request-ID header. If present, reuses it for
    downstream logging. If not, generates a new UUIDv4. The request ID
    is set in the context for the duration of the request and added to
    the response headers.

    Args:
        request: The incoming FastAPI Request.
        call_next: The next middleware or endpoint in the chain.

    Returns:
        The response with X-Request-ID header added.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def request_timeout_middleware(request: Request, call_next):
    """Enforce an overall request timeout with cancellation.

    If ROUTER_REQUEST_TIMEOUT_ENABLED is true, wraps the entire request
    processing in asyncio.wait_for with the configured timeout. On timeout,
    returns HTTP 504 (Gateway Timeout) after logging the event with context.

    Args:
        request: The incoming FastAPI Request.
        call_next: The next middleware or endpoint in the chain.

    Returns:
        The response if completed within the timeout.

    Raises:
        HTTPException: Not raised directly; returns a 504 JSON response on timeout.
    """
    if not settings.request_timeout_enabled:
        return await call_next(request)

    try:
        timeout_seconds = max(1, int(settings.request_timeout_seconds))
        return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
    except TimeoutError:
        _log_error_with_context(
            f"Request timed out after {settings.request_timeout_seconds}s: "
            f"{request.method} {request.url.path}",
            request=request,
        )
        return JSONResponse(
            {
                "error": {
                    "message": (
                        f"Request timed out after {settings.request_timeout_seconds}s. "
                        "Try a simpler prompt or increase ROUTER_REQUEST_TIMEOUT_SECONDS."
                    ),
                    "type": "timeout_error",
                }
            },
            status_code=504,
        )


async def slow_query_middleware(request: Request, call_next):
    """Log slow requests with stack traces for performance debugging.

    If ROUTER_ENABLE_SLOW_QUERY_LOGGING is true and the request duration
    exceeds ROUTER_SLOW_QUERY_THRESHOLD_MS, logs a warning with the full
    stack trace (excluding this middleware's frames). This helps identify
    performance bottlenecks in development and production.

    Args:
        request: The incoming FastAPI Request.
        call_next: The next middleware or endpoint in the chain.

    Returns:
        The response from the next handler, after potential logging.
    """
    if not settings.enable_slow_query_logging:
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    threshold = settings.slow_query_threshold_ms / 1000.0
    if duration > threshold:
        import traceback

        # Capture stack without this middleware's frames
        stack = "".join(traceback.format_stack()[:-3])
        logger.warning(
            f"Slow query: {request.method} {request.url.path} took {duration:.3f}s "
            f"(threshold {threshold:.3f}s) Request ID: {get_request_id()}\n"
            f"Stack trace:\n{stack}"
        )
    return response


async def metrics_middleware(request: Request, call_next):
    """Collect Prometheus metrics for each request.

    Records:
    - REQUESTS_TOTAL counter (labels: endpoint, method)
    - REQUEST_DURATION histogram (labels: endpoint)
    - ERRORS_TOTAL counter for status >= 400 (labels: endpoint, error_type)

    This middleware should be registered after all other middleware to
    capture metrics from the full request lifecycle.

    Args:
        request: The incoming FastAPI Request.
        call_next: The next middleware or endpoint in the chain.

    Returns:
        The response from the next handler, after metrics have been recorded.
    """
    endpoint = request.url.path
    if endpoint == "/health":
        # Skip metrics collection for hot health probes to reduce overhead.
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    method = request.method
    REQUESTS_TOTAL.labels(endpoint=endpoint, method=method).inc()
    duration = time.time() - start_time
    REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
    status_code = response.status_code
    if status_code >= 400:
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=str(status_code)).inc()
    return response


def register_middleware(app) -> None:
    """Register all HTTP middleware on the FastAPI application.

    This function attaches each middleware function from this module to
    the app using `app.middleware("http")`. The registration order is
    bottom-to-top: the last middleware registered is executed first on
    the request path (metrics_middleware runs outermost, request_size_middleware
    runs innermost).

    Args:
        app: The FastAPI application instance.

    Returns:
        None.
    """
    app.middleware("http")(request_size_middleware)
    app.middleware("http")( request_id_middleware)
    app.middleware("http")(request_timeout_middleware)
    app.middleware("http")(slow_query_middleware)
    app.middleware("http")(metrics_middleware)
