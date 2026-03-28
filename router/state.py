"""
Shared application state, helpers, authentication, and rate limiting.

This module centralizes all global state and helper functions that were
previously scattered throughout main.py. It includes:
- AppState class holding shared application state (backend, router_engine,
  background tasks, VRAM managers, rate limiters, statistics)
- VRAM estimate caching functions to avoid N+1 queries
- Helper functions for logging, IP extraction, and model listing
- Authentication and rate limiting dependencies (verify_admin_token, rate_limit_request)
"""

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from router.backends.base import LLMBackend
from router.config import Settings, settings
from router.database import get_session
from router.logging_config import get_request_id
from router.models import ModelProfile
from router.router import RouterEngine
from router.vram_manager import VRAMManager
from router.vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Cache for VRAM estimates to avoid N+1 queries
_VRAM_ESTIMATE_CACHE: dict[str, tuple[float, float]] = {}  # model_name -> (estimate, timestamp)
_VRAM_CACHE_TTL = 300.0  # 5 minutes


def _prompt_hash_for_logging(prompt: str | None) -> str | None:
    """Generate a short SHA256 hash of a prompt for log correlation.

    This function computes the SHA256 hash of the given prompt string and
    returns the first 16 hexadecimal characters. The hash is used to correlate
    log entries related to the same prompt without storing the raw prompt text,
    which may contain sensitive user data.

    Args:
        prompt: The user prompt string to hash, or None.

    Returns:
        The first 16 characters of the SHA256 hex digest if prompt is provided,
        otherwise None.
    """
    if not prompt:
        return None
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _log_error_with_context(
    message: str,
    *,
    request: Request | None = None,
    model_name: str | None = None,
    prompt: str | None = None,
    exc: Exception | None = None,
    exc_info: bool = False,
) -> None:
    """Log an error with enriched context from request, model, and prompt.

    This function logs an error message at ERROR level, automatically adding
    contextual information such as request ID, client IP, model name, and a
    hash of the prompt. This aids in debugging and tracing issues across
    distributed systems while minimizing sensitive data in logs.

    Args:
        message: The error message to log.
        request: The FastAPI Request object, used to extract client IP.
        model_name: Name of the model being used when the error occurred.
        prompt: The user prompt that triggered the error (only a hash is logged).
        exc: Optional exception instance to include in the log message.
        exc_info: If True, includes the full exception traceback in the log.

    Returns:
        None.

    Note:
        This function does not raise exceptions; any errors in logging are
        handled by the underlying logger.
    """
    if exc is not None:
        message = f"{message}: {exc}"

    user_ip = None
    if request and request.client:
        user_ip = request.client.host

    extra = {
        "request_id": get_request_id(),
        "user_ip": user_ip,
        "model_name": model_name,
        "prompt_hash": _prompt_hash_for_logging(prompt),
    }
    logger.error(message, extra=extra, exc_info=exc_info)


def get_model_vram_estimate(model_name: str) -> float:
    """Estimate VRAM requirement for a given model.

    Checks the model_profiles table in the database for a profiled VRAM value.
    If not found or on DB error, falls back to the default estimate from settings.
    Results are cached in-memory for 5 minutes to reduce database load.

    Args:
        model_name: Name of the model to estimate VRAM for.

    Returns:
        Estimated VRAM in gigabytes (float).

    Note:
        The cache is a simple module-level dictionary with TTL. It is safe for
        concurrent access because the worst that can happen is a cache miss,
        which triggers a fresh DB query.
    """
    now = time.monotonic()

    # Check cache first
    if model_name in _VRAM_ESTIMATE_CACHE:
        estimate, timestamp = _VRAM_ESTIMATE_CACHE[model_name]
        if (now - timestamp) < _VRAM_CACHE_TTL:
            return estimate

    # Cache miss or expired
    try:
        with get_session() as session:
            profile = session.query(ModelProfile).filter_by(name=model_name).first()
            if profile and profile.vram_required_gb:
                estimate = profile.vram_required_gb
                _VRAM_ESTIMATE_CACHE[model_name] = (estimate, now)
                return estimate
    except Exception as e:
        logger.debug(f"Could not fetch VRAM estimate for {model_name}: {e}")

    # Fallback to default
    estimate = settings.vram_default_estimate_gb
    _VRAM_ESTIMATE_CACHE[model_name] = (estimate, now)
    return estimate


def get_model_vram_estimates_batch(model_names: list[str]) -> dict[str, float]:
    """Get VRAM estimates for multiple models in a single database query.

    This function reduces N+1 query problems when needing estimates for many
    models. It first checks the in-memory cache for each model, then fetches
    all uncached models in one batched query (chunked to avoid SQLite parameter
    limits). The cache is updated with fresh values.

    Args:
        model_names: List of model names to estimate.

    Returns:
        A dictionary mapping each model name to its VRAM estimate in GB.
    """
    now = time.monotonic()
    result: dict[str, float] = {}
    uncached_models: list[str] = []

    # Check cache first
    for model_name in model_names:
        if model_name in _VRAM_ESTIMATE_CACHE:
            estimate, timestamp = _VRAM_ESTIMATE_CACHE[model_name]
            if (now - timestamp) < _VRAM_CACHE_TTL:
                result[model_name] = estimate
            else:
                uncached_models.append(model_name)
        else:
            uncached_models.append(model_name)

    # Fetch uncached models in batch (with chunking to avoid SQLite parameter limit)
    if uncached_models:
        try:
            with get_session() as session:
                # Chunk queries to avoid SQLite parameter limit (999)
                chunk_size = 250
                all_profiles = []
                for i in range(0, len(uncached_models), chunk_size):
                    chunk = uncached_models[i : i + chunk_size]
                    profiles_chunk = session.query(ModelProfile).filter(
                        ModelProfile.name.in_(chunk)
                    ).all()
                    all_profiles.extend(profiles_chunk)

                profile_map = {p.name: p for p in all_profiles}
                for model_name in uncached_models:
                    profile = profile_map.get(model_name)
                    if profile and profile.vram_required_gb:
                        estimate = profile.vram_required_gb
                    else:
                        estimate = settings.vram_default_estimate_gb

                    result[model_name] = estimate
                    _VRAM_ESTIMATE_CACHE[model_name] = (estimate, now)
        except Exception as e:
            logger.debug(f"Could not fetch batch VRAM estimates: {e}")
            # Fallback to individual lookups
            for model_name in uncached_models:
                result[model_name] = get_model_vram_estimate(model_name)

    return result


def invalidate_vram_estimate_cache(model_name: str | None = None) -> None:
    """Invalidate the VRAM estimate cache.

    If model_name is provided, removes that single entry from the cache.
    If model_name is None, clears the entire cache.

    Args:
        model_name: Optional model name to invalidate. If None, invalidates all.
    """
    global _VRAM_ESTIMATE_CACHE
    if model_name is None:
        _VRAM_ESTIMATE_CACHE.clear()
        logger.debug("VRAM estimate cache cleared")
    elif model_name in _VRAM_ESTIMATE_CACHE:
        del _VRAM_ESTIMATE_CACHE[model_name]
        logger.debug(f"VRAM estimate cache invalidated for {model_name}")


async def list_models_with_timeout(backend: LLMBackend, timeout: float = 10.0) -> list:
    """
    List models with timeout protection.

    Args:
        backend: The LLM backend to query
        timeout: Maximum time to wait (default 10s)

    Returns:
        List of ModelInfo objects, or empty list on timeout/error
    """
    try:
        return await asyncio.wait_for(backend.list_models(), timeout=timeout)
    except TimeoutError:
        logger.error(f"Timeout ({timeout}s) waiting for backend.list_models()")
        return []
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []


class AppState:
    """Container for shared application state.

    Holds references to all major components that need to be accessed
    across endpoints and background tasks. This includes the LLM backend,
    router engine, VRAM managers, rate limiting structures, and statistics.

    Attributes:
        backend: The active LLM backend instance (Ollama, llama.cpp, OpenAI, etc.)
        router_engine: The RouterEngine for model selection and caching
        background_tasks: Set of asyncio Tasks for background operations (sync, cleanup, DLQ retry)
        current_loaded_model: Name of the currently loaded model in VRAM (if any)
        rate_limiter: In-memory per-IP request timestamps for rate limiting
        rate_limit_lock: Async lock protecting concurrent access to rate_limiter
        vram_monitor: VRAMMonitor instance for GPU memory tracking (if enabled)
        vram_manager: VRAMManager instance for model loading/unloading (if enabled)
        start_time: Timestamp when the application started (for uptime calculation)
        total_requests: Counter of total requests processed (incremented in chat endpoint)
        total_errors: Counter of total errors encountered
        requests_by_model: Mapping of model name to number of requests served
        model_list_cache: Cached list of available models from the backend
        model_list_cache_time: Timestamp of the last model list fetch
        MODEL_LIST_CACHE_TTL: Time-to-live for model list cache in seconds (default 10.0)
        requests_by_category: Mapping of request category (e.g., 'chat', 'embeddings') to count
    """

    def __init__(self):

        self.backend: LLMBackend | None = None
        self.router_engine: RouterEngine | None = None
        self.background_tasks: set[asyncio.Task] = set()
        self.current_loaded_model: str | None = None
        self.rate_limiter: dict[str, list[float]] = {}
        self.rate_limit_lock: asyncio.Lock = asyncio.Lock()

        # VRAM monitoring and management
        self.vram_monitor: VRAMMonitor | None = None
        self.vram_manager: VRAMManager | None = None

        # Health and stats tracking
        self.start_time: datetime = datetime.now(UTC)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.requests_by_model: dict[str, int] = {}
        # Model list caching for performance
        self.model_list_cache: list = []
        self.model_list_cache_time: float = 0.0
        self.MODEL_LIST_CACHE_TTL: float = 30.0
        self.requests_by_category: dict[str, int] = {}


app_state = AppState()


async def get_available_models_with_cache(timeout: float = 10.0) -> list:
    """Get available models with caching to reduce backend API calls.
    Uses a global cache with 30-second TTL.

    Args:
        timeout: Maximum time to wait for backend if cache is stale

    Returns:
        List of ModelInfo objects from cache or fresh backend call
    """
    now = time.time()

    # Check cache first
    if (
        app_state.model_list_cache
        and (now - app_state.model_list_cache_time) < app_state.MODEL_LIST_CACHE_TTL
    ):
        logger.debug("Using cached model list (age: %.1fs)", now - app_state.model_list_cache_time)
        return app_state.model_list_cache

    # Cache miss or stale - fetch fresh models
    logger.debug("Fetching fresh model list from backend (cache expired or empty)")

    if not app_state.backend:
        logger.error("No backend available")
        return []

    models = await list_models_with_timeout(app_state.backend, timeout=timeout)

    # Update cache
    app_state.model_list_cache = models
    app_state.model_list_cache_time = now

    return models


def get_settings() -> Settings:
    """Return the global Settings instance.

    This FastAPI dependency provides access to the application configuration
    loaded from environment variables and defaults.

    Returns:
        The global Settings object.
    """
    return settings


def _get_client_ip(request: Request) -> str:
    """Extract client IP address from the request.

    Checks the X-Forwarded-For header first (used when behind a proxy/load balancer).
    If not present, falls back to the direct client host from request.client.
    Returns "unknown" if neither source is available.

    Args:
        request: The incoming FastAPI Request.

    Returns:
        The client IP address as a string.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # First IP in the chain is the original client
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _ip_in_whitelist(client_ip: str, allowed_ips: list[str]) -> bool:
    """Check if client IP matches any entry in the whitelist.

    Supports exact IP matches (both IPv4 and IPv6) and CIDR notation
    (e.g., '10.0.0.0/8', '192.168.1.0/24', 'fd00::/8'). Invalid client_ip
    or malformed whitelist entries are handled gracefully (treated as non-match
    with warning log).

    Args:
        client_ip: The IP address to check (string).
        allowed_ips: List of allowed IP strings (exact or CIDR).

    Returns:
        True if the client IP is within the whitelist, False otherwise.
    """
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False

    for entry in allowed_ips:
        try:
            if "/" in entry:
                network = ipaddress.ip_network(entry, strict=False)
                if addr in network:
                    return True
            else:
                if addr == ipaddress.ip_address(entry):
                    return True
        except ValueError:
            # Skip malformed entries
            logger.warning(f"Invalid IP whitelist entry: {entry}")
            continue

    return False


async def verify_admin_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    config: Annotated[Settings, Depends(get_settings)],
) -> bool:
    """Verify admin authentication via API key and optional IP whitelist.

    This FastAPI dependency checks:
    1. If ROUTER_ADMIN_API_KEY is configured; if not, returns 401.
    2. If ROUTER_ADMIN_ALLOWED_IPS is set, verifies the client IP (from
       X-Forwarded-For or request.client.host) is in the whitelist. If not,
       returns 403 before checking credentials to avoid information leakage.
    3. If an Authorization header is present with a Bearer token matching
       the configured admin API key.

    Args:
        request: The incoming FastAPI Request.
        credentials: The HTTP Authorization credentials (Bearer token) extracted by FastAPI security.
        config: Settings instance injected by FastAPI Depends.

    Returns:
        True if authentication succeeds.

    Raises:
        HTTPException: With status codes:
            - 401 if admin API key not configured, no credentials provided, or invalid token.
            - 403 if IP whitelist is configured and client IP is not allowed.
    """
    if not config.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key not configured. Set ROUTER_ADMIN_API_KEY environment variable to enable admin endpoints.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check IP whitelist before checking credentials
    if config.admin_allowed_ips:
        client_ip = _get_client_ip(request)
        if not _ip_in_whitelist(client_ip, config.admin_allowed_ips):
            logger.warning(
                f"Admin access denied for IP {client_ip} - not in whitelist"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: IP {client_ip} is not in the admin whitelist",
            )

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(credentials.credentials, config.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


async def rate_limit_request(
    request: Request,
    config: Settings,
    is_admin: bool = False,
    is_chat: bool = False,
) -> None:
    """Enforce per-IP rate limiting using a sliding window algorithm.

    This function checks the current request count for the client IP within the
    last 60 seconds. If the count exceeds the configured limit for the given
    request type (admin, chat, or general), it raises HTTPException 429.
    Otherwise, it records the current timestamp for the IP and returns.

    Args:
        request: The incoming FastAPI Request.
        config: Settings instance containing rate limit configuration.
        is_admin: If True, uses ROUTER_RATE_LIMIT_ADMIN_REQUESTS_PER_MINUTE.
        is_chat: If True, uses ROUTER_RATE_LIMIT_CHAT_REQUESTS_PER_MINUTE.

    Raises:
        HTTPException: With status 429 if rate limit is exceeded.
    """
    if not config.rate_limit_enabled:
        return

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Use lock to prevent race conditions
    async with app_state.rate_limit_lock:
        # Only clean up when list gets too large (>1000 entries) to reduce overhead
        if client_ip in app_state.rate_limiter and len(app_state.rate_limiter[client_ip]) > 1000:
            app_state.rate_limiter[client_ip] = [
                t for t in app_state.rate_limiter[client_ip] if now - t < 60
            ]

        current_requests = len(app_state.rate_limiter.get(client_ip, []))
        if is_admin:
            limit = config.rate_limit_admin_requests_per_minute
        elif is_chat:
            limit = config.rate_limit_chat_requests_per_minute
        else:
            limit = config.rate_limit_requests_per_minute

        if current_requests >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        if client_ip not in app_state.rate_limiter:
            app_state.rate_limiter[client_ip] = []
        app_state.rate_limiter[client_ip].append(now)
