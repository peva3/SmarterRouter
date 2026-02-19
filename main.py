import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select


from router.backends import create_backend
from router.backends.base import LLMBackend, supports_unload
from router.benchmark_db import get_last_sync
from router.benchmark_sync import sync_benchmarks
from router.config import Settings, init_logging, settings
from router.database import get_session, init_db
from router.logging_config import get_request_id, sanitize_for_logging, set_request_id
from router.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    ERRORS_TOTAL,
    MODEL_SELECTIONS_TOTAL,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    VRAM_TOTAL_GB,
    VRAM_USED_GB,
    VRAM_UTILIZATION_PCT,
    gpu_metrics,
)
from router.models import (
    BenchmarkSync,
    ModelBenchmark,
    ModelFeedback,
    ModelProfile,
    RoutingDecision,
)
from router.profiler import ModelProfiler, profile_all_models
from router.router import RouterEngine
from router.vram_monitor import VRAMMonitor
from router.vram_manager import VRAMManager, VRAMExceededError
from router.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    FeedbackRequest,
    sanitize_prompt,
    sanitize_for_logging,
    strip_signature,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingData,
    UsageInfo,
)


def get_model_vram_estimate(model_name: str) -> float:
    """
    Estimate VRAM required for a model.
    Looks up profiled value from DB, falls back to default.
    """
    try:
        with get_session() as session:
            profile = session.query(ModelProfile).filter_by(name=model_name).first()
            if profile and profile.vram_required_gb:
                return profile.vram_required_gb
    except Exception as e:
        logger.debug(f"Could not fetch VRAM estimate for {model_name}: {e}")
    return settings.vram_default_estimate_gb


from router.skills import skills_registry


class AppState:
    def __init__(self):
        from datetime import datetime, timezone

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
        self.start_time: datetime = datetime.now(timezone.utc)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.requests_by_model: dict[str, int] = {}
        self.requests_by_category: dict[str, int] = {}


app_state = AppState()
logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def get_settings() -> Settings:
    return settings


async def verify_admin_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    config: Annotated[Settings, Depends(get_settings)],
) -> bool:
    """Verify admin API key if configured."""
    if not config.admin_api_key:
        return True

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != config.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


async def rate_limit_request(request: Request, config: Settings, is_admin: bool = False) -> None:
    """Simple in-memory rate limiter with thread-safe access."""
    if not config.rate_limit_enabled:
        return

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Use lock to prevent race conditions
    async with app_state.rate_limit_lock:
        # Clean up old requests
        if client_ip in app_state.rate_limiter:
            app_state.rate_limiter[client_ip] = [
                t for t in app_state.rate_limiter[client_ip] if now - t < 60
            ]

        current_requests = len(app_state.rate_limiter.get(client_ip, []))
        limit = (
            config.rate_limit_admin_requests_per_minute
            if is_admin
            else config.rate_limit_requests_per_minute
        )

        if current_requests >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        if client_ip not in app_state.rate_limiter:
            app_state.rate_limiter[client_ip] = []
        app_state.rate_limiter[client_ip].append(now)


async def startup_event():
    init_logging()
    init_db()
    logger.info("Starting SmarterRouter...")

    # Security warning for production
    if not settings.admin_api_key:
        logger.warning(
            "⚠️  SECURITY: ROUTER_ADMIN_API_KEY is not set! "
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
        )

    # Initialize VRAM monitor
    vram_monitor: VRAMMonitor | None = None
    if settings.vram_monitor_enabled:
        vram_monitor = VRAMMonitor(
            interval=settings.vram_monitor_interval,
            total_vram_gb=settings.vram_max_total_gb,
            app_state=app_state,
            log_interval=settings.vram_log_interval,
        )
        app_state.vram_monitor = vram_monitor
        await vram_monitor.start()

        # Auto-detect total VRAM if not configured
        if settings.vram_max_total_gb is None and vram_monitor.has_nvidia:
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
    if settings.provider == "ollama":
        task = asyncio.create_task(background_sync_task())
        app_state.background_tasks.add(task)
        task.add_done_callback(app_state.background_tasks.discard)


async def shutdown_event():
    logger.info("Shutting down SmarterRouter...")
    for task in app_state.background_tasks:
        task.cancel()

    # Unload model if pinned
    if settings.pinned_model and app_state.backend:
        if supports_unload(app_state.backend):
            await app_state.backend.unload_model(settings.pinned_model)


async def background_sync_task():
    """Background task to sync benchmarks and profile new models."""
    while True:
        try:
            if app_state.backend:
                # 1. Sync Benchmarks (once per day or on startup)
                # For simplicity, we run it on startup and then rely on restart
                # But here we can check if it's needed
                with get_session() as session:
                    last_sync = get_last_sync()
                    should_sync = False
                    if not last_sync:
                        should_sync = True
                    elif (
                        datetime.now(timezone.utc) - last_sync.replace(tzinfo=timezone.utc)
                    ).days >= 1:
                        should_sync = True

                if should_sync:
                    logger.info("Starting benchmark sync...")
                    # Get available model names to match against benchmarks
                    models = await app_state.backend.list_models()
                    model_names = [m.name for m in models]
                    await sync_benchmarks(model_names)

                # 2. Profile New Models
                # This will only profile models that haven't been profiled yet
                await profile_all_models(app_state.backend)

        except Exception as e:
            logger.error(f"Background sync task failed: {e}")

        await asyncio.sleep(settings.polling_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(
    title="SmarterRouter",
    description="AI-powered LLM router that intelligently selects the best model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_size_middleware(request: Request, call_next):
    """Limit request body size to prevent memory exhaustion (10MB default)."""
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if len(body) > MAX_SIZE:
            return JSONResponse(
                {"error": {"message": "Request body too large (max 10MB)", "type": "invalid_request_error"}},
                status_code=413
            )
        # Re-create request with body for next middleware
        async def receive():
            return {"type": "http.request", "body": body}
        request = Request(request.scope, receive, request._send)
    
    return await call_next(request)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    endpoint = request.url.path
    method = request.method
    REQUESTS_TOTAL.labels(endpoint=endpoint, method=method).inc()
    duration = time.time() - start_time
    REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
    status = response.status_code
    if status >= 400:
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=str(status)).inc()
    return response


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "SmarterRouter",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from router.metrics import generate_metrics

    return Response(content=generate_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/v1/models")
async def list_models(config: Annotated[Settings, Depends(get_settings)]):
    """List available models. Returns the router itself as a single model for external UIs."""

    return {
        "object": "list",
        "data": [
            {
                "id": config.router_external_model_name,
                "object": "model",
                "created": datetime.now(timezone.utc).timestamp(),
                "owned_by": "local",
                "description": "An intelligent router that selects the best LLM based on prompt analysis and model capabilities.",
                "admin_auth_required": config.admin_api_key is not None,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
):
    # Rate limit check for chat endpoint
    await rate_limit_request(request, config, is_admin=False)

    if not app_state.backend or not app_state.router_engine:
        return JSONResponse(
            {"error": {"message": "Service not ready", "type": "service_unavailable"}},
            status_code=503,
        )

    # Validate Content-Type header
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return JSONResponse(
            {
                "error": {
                    "message": "Content-Type must be application/json",
                    "type": "invalid_request_error",
                }
            },
            status_code=415,
        )

    # Parse and validate request body using Pydantic
    try:
        body = await request.json()
        validated_request = ChatCompletionRequest(**body)
    except Exception as e:
        logger.warning(f"Request validation failed: {e}")
        return JSONResponse(
            {"error": {"message": f"Invalid request: {str(e)}", "type": "invalid_request_error"}},
            status_code=400,
        )

    # Extract and sanitize prompt from last message
    messages = validated_request.messages
    stream = validated_request.stream

    last_message = messages[-1]
    prompt = sanitize_prompt(last_message.content)

    if not prompt:
        return JSONResponse(
            {"error": {"message": "Prompt cannot be empty", "type": "invalid_request_error"}},
            status_code=400,
        )

    # Generate response ID early
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Check for model override query parameter
    model_override = request.query_params.get("model")

    # Track request
    if hasattr(app_state, "total_requests"):
        app_state.total_requests += 1

    try:
        # Model override - skip routing and use specified model
        if model_override:
            available_models = await app_state.backend.list_models()
            model_names = [m.name for m in available_models]

            # Try exact match first, then partial match
            selected_model = None
            for name in model_names:
                if name == model_override:
                    selected_model = name
                    break
                if model_override.lower() in name.lower():
                    selected_model = name
                    break

            if not selected_model:
                return JSONResponse(
                    {
                        "error": {
                            "message": f"Model '{model_override}' not available. Available: {model_names[:5]}...",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=400,
                )

            reasoning = f"User-specified model override: {selected_model}"
            confidence = 1.0
            logger.info(f"Model override: {selected_model}, prompt: {sanitize_for_logging(prompt)}")
        else:
            # Pass full request object for capability detection
            last_content = messages[-1].content
            if last_content is None:
                last_content = ""
            routing_result = await app_state.router_engine.select_model(
                last_content, validated_request
            )
            selected_model = routing_result.selected_model
            reasoning = routing_result.reasoning
            confidence = routing_result.confidence
            # Use sanitized logging
            logger.info(f"Routed to: {selected_model}, prompt: {sanitize_for_logging(prompt)}")
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        models = await app_state.backend.list_models()
        if models:
            selected_model = models[0].name
            reasoning = "Fallback to first available model"
            confidence = 0.0
        else:
            return JSONResponse(
                {"error": {"message": "No models available", "type": "internal_error"}},
                status_code=500,
            )

    # Convert Pydantic models back to dicts for backend compatibility
    # and strip signatures from previous assistant messages to prevent stacking
    def clean_message_content(msg):
        content = msg.content
        if isinstance(content, str) and msg.role == "assistant":
            # Remove any previous signatures from assistant messages
            content = strip_signature(content)
        return content

    messages_dict = [{"role": msg.role, "content": clean_message_content(msg)} for msg in messages]

    # Collect additional parameters for backend
    backend_kwargs = {
        "temperature": validated_request.temperature,
        "top_p": validated_request.top_p,
        "n": validated_request.n,
        "max_tokens": validated_request.max_tokens,
        "presence_penalty": validated_request.presence_penalty,
        "frequency_penalty": validated_request.frequency_penalty,
        "logit_bias": validated_request.logit_bias,
        "user": validated_request.user,
        "seed": validated_request.seed,
        "logprobs": validated_request.logprobs,
        "top_logprobs": validated_request.top_logprobs,
        "tools": skills_registry.list_skills() if validated_request.tools else None,
        "tool_choice": validated_request.tool_choice,
    }
    # Remove None values
    backend_kwargs = {k: v for k, v in backend_kwargs.items() if v is not None}

    if stream:
        # Proactive VRAM management for streaming - unload non-needed models before streaming
        current = app_state.current_loaded_model
        pinned = config.pinned_model

        if current and current != selected_model and current != pinned:
            logger.info(
                f"VRAM management (streaming): unloading {current} to load {selected_model}"
            )
            if supports_unload(app_state.backend):
                await app_state.backend.unload_model(current)

        # Load model via VRAM manager if enabled
        if app_state.vram_manager:
            vram_gb = get_model_vram_estimate(selected_model)
            await app_state.vram_manager.load_model(selected_model, vram_gb)
        else:
            # Traditional: unload current model if different and not pinned
            current = app_state.current_loaded_model
            pinned = config.pinned_model
        if current and current != selected_model and current != pinned:
            logger.info(
                f"VRAM management (streaming): unloading {current} to load {selected_model}"
            )
            if supports_unload(app_state.backend):
                await app_state.backend.unload_model(current)

        # Update current model state
        app_state.current_loaded_model = selected_model

        # Log the decision now that we're committing to it
        await app_state.router_engine.log_decision(
            prompt, selected_model, confidence, reasoning, response_id
        )

        return StreamingResponse(
            stream_chat(
                app_state.backend,
                selected_model,
                messages_dict,
                reasoning,
                config,
                response_id,
                **backend_kwargs,
            ),
            media_type="text/event-stream",
        )

    # Try generation with retries
    response = None
    last_error = None

    # Get available models for fallback
    try:
        available_models = await app_state.backend.list_models()
        fallback_list = [m.name for m in available_models if m.name != selected_model]
        # Put selected_model first in retry list
        fallback_list = [selected_model] + fallback_list
    except Exception:
        fallback_list = [selected_model]

    # Pre-fetch VRAM estimates for all fallback models to avoid N+1 queries
    vram_estimate_map: dict[str, float] = {}
    if app_state.vram_manager:
        with get_session() as session:
            profiles = session.query(ModelProfile).filter(
                ModelProfile.name.in_(fallback_list)
            ).all()
            vram_estimate_map = {
                p.name: (
                    p.vram_required_gb
                    if p.vram_required_gb is not None
                    else config.vram_default_estimate_gb
                )
                for p in profiles
            }

    # If cascading is enabled, we might want to ensure we don't just randomly fallback,
    # but that's handled by the router mostly returning the "best" model first.
    # The simple fallback list is sufficient for reliability.

    final_model = selected_model

    # Check response cache before generation (include generation params in key)
    cache_key_prompt = prompt
    if app_state.router_engine and app_state.router_engine.semantic_cache:
        cached_response = await app_state.router_engine.semantic_cache.get_response(
            selected_model, cache_key_prompt, params=backend_kwargs
        )
        if cached_response:
            logger.info(f"Response cache hit for {selected_model}")
            await app_state.router_engine.log_decision(
                prompt, selected_model, confidence, reasoning, response_id
            )
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": selected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": cached_response},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "router": {"reasoning": reasoning + " [cached]"},
            }

    for try_model in fallback_list:
        try:
            # Load model via VRAM manager if enabled, else fallback to traditional unload
            if app_state.vram_manager:
                # Use pre-fetched VRAM estimate
                vram_gb = vram_estimate_map.get(try_model, config.vram_default_estimate_gb)
                await app_state.vram_manager.load_model(try_model, vram_gb)
            else:
                # Traditional: unload current model if different and not pinned before loading new
                current = app_state.current_loaded_model
                pinned = config.pinned_model
                if current and current != try_model and current != pinned:
                    logger.info(f"VRAM management: unloading {current} to load {try_model}")
                    if supports_unload(app_state.backend):
                        await app_state.backend.unload_model(current)
                    app_state.current_loaded_model = None

            # Generate response
            response = await app_state.backend.chat(
                model=try_model, messages=messages_dict, stream=False, **backend_kwargs
            )
            final_model = try_model
            app_state.current_loaded_model = final_model
            logger.info(f"Generation succeeded with model: {final_model}")

            # Track stats
            if hasattr(app_state, "requests_by_model"):
                app_state.requests_by_model[final_model] = (
                    app_state.requests_by_model.get(final_model, 0) + 1
                )

            # If we fell back, update reasoning
            if final_model != selected_model:
                reasoning += f" (Fallback from {selected_model})"

            break
        except Exception as try_error:
            # If we loaded this model via VRAM manager and it's still loaded, unload it to free VRAM
            if app_state.vram_manager and app_state.vram_manager.is_loaded(try_model):
                await app_state.vram_manager.unload_model(try_model)
            last_error = try_error
            
            # Get VRAM state for error context
            vram_context = ""
            if app_state.vram_manager:
                try:
                    available_vram = app_state.vram_manager.get_available_vram()
                    max_vram = app_state.vram_manager.max_vram
                    vram_context = f" | VRAM: {available_vram:.1f}GB/{max_vram:.1f}GB free"
                except Exception:
                    vram_context = " | VRAM: unknown"
            
            logger.warning(
                f"Model {try_model} failed, trying next: {try_error} | "
                f"Prompt: {sanitize_for_logging(prompt)[:100]}... | "
                f"Response ID: {response_id}{vram_context}",
                exc_info=True
            )
            continue

    if response is None:
        logger.error(f"All models failed. Last error: {last_error}")
        if hasattr(app_state, "total_errors"):
            app_state.total_errors += 1
        return JSONResponse(
            {
                "error": {
                    "message": f"All models failed. Last error: {last_error}",
                    "type": "internal_error",
                }
            },
            status_code=500,
        )

    # Log the initial routing decision
    await app_state.router_engine.log_decision(
        prompt, final_model, confidence, reasoning, response_id
    )

    # === TOOL EXECUTION LOOP ===
    max_tool_calls = 5
    tool_calls_made = 0

    while tool_calls_made < max_tool_calls:
        tool_calls = response.get("message", {}).get("tool_calls")
        if not tool_calls:
            break

        logger.info(f"Model {final_model} requested {len(tool_calls)} tool call(s)")

        # Add assistant message with tool calls to history
        messages_dict.append(response["message"])

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            logger.info(f"Executing tool: {tool_name}({tool_args})")
            tool_result = await skills_registry.execute_skill(tool_name, **tool_args)

            messages_dict.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )

        tool_calls_made += 1

        # Continue conversation with tool results
        response = await app_state.backend.chat(
            model=final_model,
            messages=messages_dict,
            stream=False,
            **backend_kwargs,
        )

    content = response.get("message", {}).get("content", "")

    if config.signature_enabled:
        signature = config.signature_format.format(model=final_model)
        # Strip any existing signature first, then add our own
        content = strip_signature(content)
        content += signature

    # Cache the response (without signature) for future requests (include generation params)
    if app_state.router_engine and app_state.router_engine.semantic_cache:
        content_for_cache = strip_signature(content)
        await app_state.router_engine.semantic_cache.set_response(
            final_model, prompt, content_for_cache, params=backend_kwargs
        )

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": datetime.now(timezone.utc).timestamp(),
        "model": final_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        },
    }


async def stream_chat(
    client: LLMBackend,
    model: str,
    messages: list[dict[str, str]],
    reasoning: str,
    config: Settings,
    chunk_id: str,
    **kwargs: Any,
) -> AsyncIterator[str]:
    created = datetime.now(timezone.utc).timestamp()

    try:
        stream, latency = await client.chat_streaming(model, messages, **kwargs)

        # Initial chunk with metadata
        initial_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': ''},
                    'finish_reason': None,
                }
            ],
            'router': {'reasoning': reasoning},
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        async for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                content_chunk = {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model,
                    'choices': [
                        {'index': 0, 'delta': {'content': content}, 'finish_reason': None}
                    ],
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

            if chunk.get("done", False):
                # Add signature if enabled
                if config.signature_enabled:
                    signature = config.signature_format.format(model=model)
                    signature_chunk = {
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model,
                        'choices': [
                            {
                                'index': 0,
                                'delta': {'content': signature},
                                'finish_reason': 'stop',
                            }
                        ],
                    }
                    yield f"data: {json.dumps(signature_chunk)}\n\n"
                else:
                    done_chunk = {
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model,
                        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                    }
                    yield f"data: {json.dumps(done_chunk)}\n\n"
    except Exception as e:
        logger.error(f"Streaming failed: {e}", exc_info=True)
        error_message = str(e)

        # Provide more helpful error messages for common issues
        if "timeout" in error_message.lower():
            error_message = f"Timeout error: The model took too long to respond. Current timeout: {config.generation_timeout}s. Try increasing ROUTER_GENERATION_TIMEOUT."
        elif "connection" in error_message.lower():
            error_message = "Connection error: Could not connect to the LLM backend. Please check that Ollama is running and accessible."

        error_data = {"error": {"message": error_message, "type": "internal_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/admin/profiles")
async def get_profiles(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
):
    """Get model profiles (requires admin API key if configured)."""
    await rate_limit_request(request, config, is_admin=True)

    with get_session() as session:
        profiles = list(session.query(ModelProfile).all())

        return {
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
            ]
        }


@app.get("/admin/benchmarks")
async def get_benchmarks(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
):
    """Get benchmark data (requires admin API key if configured)."""
    await rate_limit_request(request, config, is_admin=True)

    with get_session() as session:
        benchmarks = list(session.query(ModelBenchmark).all())

        last_sync = session.execute(
            select(BenchmarkSync).order_by(BenchmarkSync.id.desc()).limit(1)
        ).scalar_one_or_none()

        # Extract sync info while in session
        last_sync_time = (
            last_sync.last_sync.isoformat() if last_sync and last_sync.last_sync else None
        )
        sync_status = last_sync.status if last_sync else None

        return {
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


@app.get("/admin/stats")
async def get_stats(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
):
    """Get router statistics (requires admin API key if configured)."""
    from datetime import datetime, timezone

    await rate_limit_request(request, config, is_admin=True)

    # Calculate uptime
    uptime_seconds = 0
    if hasattr(app_state, "start_time"):
        uptime_seconds = (datetime.now(timezone.utc) - app_state.start_time).total_seconds()

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


@app.post("/admin/reprofile")
async def reprofile(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    config: Annotated[Settings, Depends(get_settings)],
    force: bool = False,
):
    """Trigger manual reprofiling (requires admin API key if configured)."""
    if not app_state.backend:
        return JSONResponse({"error": "Client not initialized"}, status_code=503)

    await rate_limit_request(request, config, is_admin=True)

    logger.info(f"Manual reprofile triggered (force={force})")
    results = await profile_all_models(app_state.backend, force=force)

    return {
        "profiled": [r.model_name for r in results],
        "count": len(results),
    }


@app.post("/admin/cache/invalidate")
async def invalidate_cache(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    model: str | None = None,
    response_cache_only: bool = False,
):
    """Invalidate cache entries (requires admin API key if configured)."""
    if not app_state.router_engine or not app_state.router_engine.semantic_cache:
        return JSONResponse({"error": "Cache not initialized"}, status_code=503)

    cache = app_state.router_engine.semantic_cache
    invalidated = 0

    if response_cache_only or model:
        invalidated = await cache.invalidate_response(model)
    else:
        async with cache._lock:
            cache.cache.clear()
            cache.response_cache.clear()
        invalidated = "all"

    return {
        "invalidated": invalidated,
        "model": model,
        "response_cache_only": response_cache_only,
    }


@app.get("/admin/vram")
async def get_vram_status(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    history_minutes: int = 10,
):
    """
    Admin-only endpoint: comprehensive VRAM monitoring data.
    Shows current usage, loaded models, history, and warnings.
    Requires admin API key if configured.
    """
    if not hasattr(app_state, "vram_monitor") or app_state.vram_monitor is None:
        return {"error": "VRAM monitoring not enabled"}

    monitor: VRAMMonitor = app_state.vram_monitor
    vram_manager: VRAMManager | None = app_state.vram_manager

    current = monitor.get_current()
    history = monitor.get_history(minutes=history_minutes)

    response = {
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


@app.get("/admin/explain")
async def explain_routing(
    request: Request,
    _: Annotated[bool, Depends(verify_admin_token)],
    prompt: str,
    model_override: str | None = None,
):
    """
    Explain why a specific model would be selected for a prompt.
    Returns scoring breakdown without generating a response.
    Useful for debugging routing decisions.
    """
    if not app_state.backend or not app_state.router_engine:
        return JSONResponse(
            {"error": {"message": "Service not ready", "type": "service_unavailable"}},
            status_code=503,
        )

    try:
        # Get available models
        available_models = await app_state.backend.list_models()
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
                    {"error": {"message": f"Model '{model_override}' not found", "type": "invalid_request_error"}},
                    status_code=400,
                )
            
            return {
                "prompt": prompt,
                "selected_model": selected_model,
                "override": True,
                "reasoning": f"Model override specified: {model_override}",
                "confidence": 1.0,
                "scores": None,
            }

        # Otherwise, run the full routing logic to get scoring breakdown
        model_list = [m.name for m in available_models]
        
        # Check routing cache first
        cached_result = None
        if app_state.router_engine.semantic_cache:
            cached_result = await app_state.router_engine.semantic_cache.get(prompt)
        
        if cached_result:
            return {
                "prompt": prompt,
                "selected_model": cached_result.selected_model,
                "cached": True,
                "reasoning": cached_result.reasoning,
                "confidence": cached_result.confidence,
                "scores": None,
            }

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
        
        # Get all model scores
        with get_session() as session:
            profiles = session.query(ModelProfile).filter(
                ModelProfile.name.in_(model_list)
            ).all()
            
            model_scores = []
            for profile in profiles:
                model_scores.append({
                    "name": profile.name,
                    "reasoning": profile.reasoning,
                    "coding": profile.coding,
                    "creativity": profile.creativity,
                    "speed": profile.speed,
                    "vram_gb": profile.vram_required_gb,
                })

        return {
            "prompt": prompt,
            "prompt_preview": sanitize_for_logging(prompt)[:100] + "..." if len(prompt) > 100 else prompt,
            "selected_model": selected_model,
            "confidence": confidence,
            "reasoning": reasoning,
            "cached": False,
            "available_models": len(model_list),
            "model_scores": sorted(model_scores, key=lambda x: x["name"]),
        }

    except Exception as e:
        logger.error(f"Explain routing failed: {e}", exc_info=True)
        return JSONResponse(
            {"error": {"message": f"Explain routing failed: {str(e)}", "type": "internal_error"}},
            status_code=500,
        )


@app.get("/v1/skills")
async def list_skills():
    """List available tools/skills."""
    return {"skills": skills_registry.list_skills()}


@app.post("/v1/feedback")
async def feedback(
    request: FeedbackRequest,
    config: Annotated[Settings, Depends(get_settings)],
):
    """Submit user feedback for a routing decision."""
    if not config.feedback_enabled:
        return JSONResponse({"error": "Feedback collection is disabled"}, status_code=403)

    try:
        with get_session() as session:
            # Create feedback entry
            fb = ModelFeedback(
                model_name=request.model_name,
                prompt_hash=None,  # Will be linked if we look up the response_id
                score=request.score,
                comment=request.comment,
                category=request.category,
            )

            # If response_id provided, link to original decision
            if request.response_id:
                decision = session.execute(
                    select(RoutingDecision).where(
                        RoutingDecision.response_id == request.response_id
                    )
                ).scalar_one_or_none()

                if decision:
                    fb.prompt_hash = decision.prompt_hash
                    # Auto-fill model name if not provided
                    if not fb.model_name:
                        fb.model_name = decision.selected_model

            if not fb.model_name:
                return JSONResponse(
                    {"error": "model_name is required if response_id is not found"}, status_code=400
                )

            session.add(fb)
            session.commit()

            return {"status": "success", "id": fb.id}

    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def embeddings(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
):
    """Generate embeddings for the given input."""
    if not app_state.backend:
        return JSONResponse(
            {"error": {"message": "Service not ready", "type": "service_unavailable"}},
            status_code=503,
        )

    # Validate Content-Type
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return JSONResponse(
            {
                "error": {
                    "message": "Content-Type must be application/json",
                    "type": "invalid_request_error",
                }
            },
            status_code=415,
        )

    try:
        body = await request.json()
        validated_request = EmbeddingsRequest(**body)
    except Exception as e:
        logger.warning(f"Embeddings request validation failed: {e}")
        return JSONResponse(
            {"error": {"message": f"Invalid request: {str(e)}", "type": "invalid_request_error"}},
            status_code=400,
        )

    model = validated_request.model
    input_text = validated_request.input

    try:
        # For embeddings, we just forward directly to the backend
        # We don't route yet as embeddings models are usually specific
        result = await app_state.backend.embed(model, input_text)

        # Map response to OpenAI format
        embeddings_list = []

        # Handle Ollama/OpenAI response formats
        if "embeddings" in result:
            # Ollama format
            for i, emb in enumerate(result["embeddings"]):
                embeddings_list.append(EmbeddingData(embedding=emb, index=i))
        elif "data" in result:
            # OpenAI format
            for item in result["data"]:
                embeddings_list.append(
                    EmbeddingData(embedding=item["embedding"], index=item["index"])
                )
        elif "embedding" in result:
            # Single result
            embeddings_list.append(EmbeddingData(embedding=result["embedding"], index=0))

        usage = result.get("usage", {})

        return EmbeddingsResponse(
            data=embeddings_list,
            model=model,
            usage=UsageInfo(
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    except Exception as e:
        logger.error(f"Embeddings failed: {e}")
        return JSONResponse(
            {
                "error": {
                    "message": f"Embeddings generation failed: {str(e)}",
                    "type": "internal_error",
                }
            },
            status_code=500,
        )


def main() -> None:
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
