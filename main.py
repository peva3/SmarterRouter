import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from router.backends import create_backend
from router.backends.base import LLMBackend
from router.benchmark_db import get_last_sync
from router.benchmark_sync import sync_benchmarks
from router.config import Settings, init_logging, settings
from router.database import get_session, init_db
from router.models import BenchmarkSync, ModelBenchmark, ModelProfile
from router.profiler import ModelProfiler, profile_all_models
from router.schemas import (
    ChatCompletionRequest,
    sanitize_for_logging,
    sanitize_prompt,
)
from sqlalchemy import select
from router.router import RouterEngine

logger = logging.getLogger(__name__)

# Security schemes
security = HTTPBearer(auto_error=False)


# Dependency for configuration
def get_settings() -> Settings:
    return settings


# Authentication dependency
async def verify_admin_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    config: Annotated[Settings, Depends(get_settings)]
) -> bool:
    """Verify admin API key if configured."""
    # If no admin API key is configured, allow access (backward compatibility)
    if not config.admin_api_key:
        return True
    
    # If admin API key is configured, require valid token
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != config.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: dict[str, list[float]] = {}
    
    def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < window
            ]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= limit:
            return False
        
        # Record request
        self.requests[key].append(now)
        return True


rate_limiter = RateLimiter()


async def rate_limit_request(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
    is_admin: bool = False
) -> None:
    """Apply rate limiting if enabled."""
    if not config.rate_limit_enabled:
        return
    
    # Get client identifier (IP address or forwarded IP)
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    key = f"{client_ip}:{request.url.path}"
    
    # Determine limit
    if is_admin:
        limit = config.rate_limit_admin_requests_per_minute
    else:
        limit = config.rate_limit_requests_per_minute
    
    if not rate_limiter.is_allowed(key, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {limit} requests per minute."
        )


# Application state
class AppState:
    def __init__(self):
        self.backend: LLMBackend | None = None
        self.router_engine: RouterEngine | None = None
        self.known_models: set[str] = set()
        self.polling_task: asyncio.Task | None = None
        self.benchmark_sync_task: asyncio.Task | None = None
        self.current_loaded_model: str | None = None


app_state = AppState()


async def poll_models(state: AppState) -> None:
    while True:
        try:
            if state.backend:
                models = await state.backend.list_models()
                current_models = {m.name for m in models}

                new_models = current_models - state.known_models
                if new_models:
                    logger.info(f"New models detected: {new_models}")
                    profiler = ModelProfiler(state.backend)
                    for model_name in new_models:
                        try:
                            await profiler.profile_model(model_name)
                        except Exception as e:
                            logger.error(f"Failed to profile {model_name}: {e}")

                state.known_models = current_models

        except Exception as e:
            logger.error(f"Error in model polling: {e}")

        await asyncio.sleep(settings.polling_interval)


async def sync_benchmarks_daily(state: AppState) -> None:
    while True:
        try:
            if state.backend:
                models = await state.backend.list_models()
                model_names = [m.name for m in models]
                logger.info("Running daily benchmark sync...")
                count, matched = await sync_benchmarks(model_names)
                logger.info(f"Benchmark sync completed: {count} models synced")

        except Exception as e:
            logger.error(f"Error in benchmark sync: {e}")

        await asyncio.sleep(86400)


async def startup_event() -> None:
    init_logging()
    init_db()
    logger.info(f"Starting LLM Router Proxy (provider: {settings.provider})")

    app_state.backend = create_backend(settings)
    app_state.router_engine = RouterEngine(
        app_state.backend, dispatcher_model=settings.router_model
    )

    models = await app_state.backend.list_models()
    app_state.known_models = {m.name for m in models}
    logger.info(f"Initial models: {app_state.known_models}")

    if app_state.known_models:
        logger.info("Running initial profiling...")
        await profile_all_models(app_state.backend)

        logger.info("Running initial benchmark sync...")
        try:
            count, matched = await sync_benchmarks(list(app_state.known_models))
            logger.info(f"Benchmark sync: {count} models synced")
        except Exception as e:
            logger.warning(f"Benchmark sync failed: {e}")

    app_state.polling_task = asyncio.create_task(poll_models(app_state))
    logger.info("Model polling started")

    app_state.benchmark_sync_task = asyncio.create_task(sync_benchmarks_daily(app_state))
    logger.info("Benchmark sync task started")


async def shutdown_event() -> None:
    logger.info("Shutting down LLM Router Proxy")

    if app_state.polling_task:
        app_state.polling_task.cancel()
        try:
            await app_state.polling_task
        except asyncio.CancelledError:
            pass

    if app_state.benchmark_sync_task:
        app_state.benchmark_sync_task.cancel()
        try:
            await app_state.benchmark_sync_task
        except asyncio.CancelledError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(
    title="LLM Router Proxy",
    description="AI-powered LLM router that intelligently selects the best model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "LLM Router Proxy",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    if not app_state.backend:
        return JSONResponse({"error": "Client not initialized"}, status_code=503)

    models = await app_state.backend.list_models()

    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": datetime.now(timezone.utc).timestamp(),
                "owned_by": "local",
            }
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
):
    if not app_state.backend or not app_state.router_engine:
        return JSONResponse(
            {"error": {"message": "Service not ready", "type": "service_unavailable"}},
            status_code=503,
        )

    # Validate Content-Type header
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return JSONResponse(
            {"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}},
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

    try:
        routing_result = await app_state.router_engine.select_model(prompt)
        selected_model = routing_result.selected_model
        reasoning = routing_result.reasoning
        # Use sanitized logging
        logger.info(f"Routed to: {selected_model}, prompt: {sanitize_for_logging(prompt)}")
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        models = await app_state.backend.list_models()
        if models:
            selected_model = models[0].name
            reasoning = "Fallback to first available model"
        else:
            return JSONResponse(
                {"error": {"message": "No models available", "type": "internal_error"}},
                status_code=500,
            )

    # Convert Pydantic models back to dicts for backend compatibility
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

    if stream:
        return StreamingResponse(
            stream_chat(app_state.backend, selected_model, messages_dict, reasoning, config),
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
    except:
        fallback_list = [selected_model]
    
    for try_model in fallback_list:
        # Proactive VRAM management
        current = app_state.current_loaded_model
        pinned = config.pinned_model
        
        if current and current != try_model and current != pinned:
            logger.info(f"VRAM management: unloading {current} to load {try_model}")
            await app_state.backend.unload_model(current)
            app_state.current_loaded_model = None
        
        try:
            response = await app_state.backend.chat(
                model=try_model,
                messages=messages_dict,
                stream=False,
            )
            selected_model = try_model
            app_state.current_loaded_model = selected_model
            logger.info(f"Generation succeeded with model: {selected_model}")
            break
        except Exception as try_error:
            last_error = try_error
            logger.warning(f"Model {try_model} failed, trying next: {try_error}")
            continue
    
    if response is None:
        logger.error(f"All models failed. Last error: {last_error}")
        return JSONResponse(
            {"error": {"message": f"All models failed. Last error: {last_error}", "type": "internal_error"}},
            status_code=500,
        )

    content = response.get("message", {}).get("content", "")

    if config.signature_enabled:
        signature = config.signature_format.format(model=selected_model)
        content += signature

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": datetime.now(timezone.utc).timestamp(),
        "model": selected_model,
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
            "total_tokens": response.get("prompt_eval_count", 0)
            + response.get("eval_count", 0),
        },
    }


async def stream_chat(
    client: LLMBackend,
    model: str,
    messages: list[dict[str, str]],
    reasoning: str,
    config: Settings,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = datetime.now(timezone.utc).timestamp()

    try:
        stream_gen, _ = await client.chat_streaming(model, messages)

        first_chunk = True
        async for chunk in stream_gen:
            if chunk.get("done"):
                break

            content = chunk.get("message", {}).get("content", "")

            if first_chunk:
                first_chunk = False
                data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": content,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
            else:
                data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

        if config.signature_enabled:
            signature = config.signature_format.format(model=model)
            data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": signature},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        error_data = {"error": {"message": str(e)}}
        yield f"data: {json.dumps(error_data)}\n\n"


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
        last_sync_time = last_sync.last_sync.isoformat() if last_sync and last_sync.last_sync else None
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


def main() -> None:
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
