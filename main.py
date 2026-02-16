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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select

from router.backends import create_backend
from router.backends.base import LLMBackend
from router.benchmark_db import get_last_sync
from router.benchmark_sync import sync_benchmarks
from router.config import Settings, init_logging, settings
from router.database import get_session, init_db
from router.models import BenchmarkSync, ModelBenchmark, ModelFeedback, ModelProfile, RoutingDecision
from router.profiler import ModelProfiler, profile_all_models
from router.router import RouterEngine
from router.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    FeedbackRequest,
    sanitize_prompt,
    sanitize_for_logging,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingData,
    UsageInfo,
)
from router.skills import skills_registry


class AppState:
    def __init__(self):
        self.backend: LLMBackend | None = None
        self.router_engine: RouterEngine | None = None
        self.background_tasks: set[asyncio.Task] = set()
        self.current_loaded_model: str | None = None
        self.rate_limiter: dict[str, list[float]] = {}


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
    # print(f"DEBUG: admin_key={config.admin_api_key}, creds={credentials.credentials if credentials else 'None'}")
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


async def rate_limit_request(
    request: Request, 
    config: Settings, 
    is_admin: bool = False
) -> None:
    """Simple in-memory rate limiter."""
    if not config.rate_limit_enabled:
        return

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Clean up old requests
    if client_ip in app_state.rate_limiter:
        app_state.rate_limiter[client_ip] = [
            t for t in app_state.rate_limiter[client_ip] 
            if now - t < 60
        ]
    
    current_requests = len(app_state.rate_limiter.get(client_ip, []))
    limit = config.rate_limit_admin_requests_per_minute if is_admin else config.rate_limit_requests_per_minute
    
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
    logger.info("Starting LLM Router Proxy...")

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
            dispatcher_model=settings.router_model
        )

    # Start background sync task
    if settings.provider == "ollama":
        task = asyncio.create_task(background_sync_task())
        app_state.background_tasks.add(task)
        task.add_done_callback(app_state.background_tasks.discard)


async def shutdown_event():
    logger.info("Shutting down LLM Router Proxy...")
    for task in app_state.background_tasks:
        task.cancel()
    
    # Unload model if pinned
    if settings.pinned_model and app_state.backend:
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
                    elif (datetime.now(timezone.utc) - last_sync.replace(tzinfo=timezone.utc)).days >= 1:
                        should_sync = True
                
                if should_sync:
                    logger.info("Starting benchmark sync...")
                    await sync_benchmarks(settings.benchmark_sources.split(","))
                
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
                "admin_auth_required": config.admin_api_key is not None
            }
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

    # Generate response ID early
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    try:
        # Pass full request object for capability detection
        routing_result = await app_state.router_engine.select_model(messages[-1].content, validated_request)
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
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

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
    }
    # Remove None values
    backend_kwargs = {k: v for k, v in backend_kwargs.items() if v is not None}

    if stream:
        # Proactive VRAM management for streaming - unload non-needed models before streaming
        current = app_state.current_loaded_model
        pinned = config.pinned_model
        
        if current and current != selected_model and current != pinned:
            logger.info(f"VRAM management (streaming): unloading {current} to load {selected_model}")
            await app_state.backend.unload_model(current)
        
        # Update current model state
        app_state.current_loaded_model = selected_model
        
        # Log the decision now that we're committing to it
        app_state.router_engine.log_decision(prompt, selected_model, confidence, reasoning, response_id)

        return StreamingResponse(
            stream_chat(app_state.backend, selected_model, messages_dict, reasoning, config, response_id, **backend_kwargs),
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
    
    # If cascading is enabled, we might want to ensure we don't just randomly fallback, 
    # but that's handled by the router mostly returning the "best" model first.
    # The simple fallback list is sufficient for reliability.
    
    final_model = selected_model

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
                **backend_kwargs
            )
            final_model = try_model
            app_state.current_loaded_model = final_model
            logger.info(f"Generation succeeded with model: {final_model}")
            
            # If we fell back, update reasoning
            if final_model != selected_model:
                reasoning += f" (Fallback from {selected_model})"
                
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

    # Log the final decision
    app_state.router_engine.log_decision(prompt, final_model, confidence, reasoning, response_id)

    content = response.get("message", {}).get("content", "")

    if config.signature_enabled:
        signature = config.signature_format.format(model=final_model)
        content += signature

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
    chunk_id: str,
    **kwargs: Any,
) -> AsyncIterator[str]:
    created = datetime.now(timezone.utc).timestamp()
    
    try:
        stream, latency = await client.chat_streaming(model, messages, **kwargs)
        
        # Initial chunk with metadata
        yield f"data: {json.dumps({
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {'role': 'assistant', 'content': ''},
                'finish_reason': None
            }],
            'router': {
                'reasoning': reasoning
            }
        })}\n\n"

        async for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield f"data: {json.dumps({
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': content},
                        'finish_reason': None
                    }]
                })}\n\n"
            
            if chunk.get("done", False):
                # Add signature if enabled
                if config.signature_enabled:
                    signature = config.signature_format.format(model=model)
                    yield f"data: {json.dumps({
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': signature},
                            'finish_reason': 'stop'
                        }]
                    })}\n\n"
                else:
                    yield f"data: {json.dumps({
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model,
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop'
                        }]
                    })}\n\n"
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
                prompt_hash=None, # Will be linked if we look up the response_id
                score=request.score,
                comment=request.comment,
                category=request.category,
            )
            
            # If response_id provided, link to original decision
            if request.response_id:
                decision = session.execute(
                    select(RoutingDecision).where(RoutingDecision.response_id == request.response_id)
                ).scalar_one_or_none()
                
                if decision:
                    fb.prompt_hash = decision.prompt_hash
                    # Auto-fill model name if not provided
                    if not fb.model_name:
                        fb.model_name = decision.selected_model
            
            if not fb.model_name:
                return JSONResponse(
                    {"error": "model_name is required if response_id is not found"}, 
                    status_code=400
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
            {"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}},
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
                embeddings_list.append(EmbeddingData(embedding=item["embedding"], index=item["index"]))
        elif "embedding" in result:
            # Single result
            embeddings_list.append(EmbeddingData(embedding=result["embedding"], index=0))
        
        usage = result.get("usage", {})
        
        return EmbeddingsResponse(
            data=embeddings_list,
            model=model,
            usage=UsageInfo(
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
        )

    except Exception as e:
        logger.error(f"Embeddings failed: {e}")
        return JSONResponse(
            {"error": {"message": f"Embeddings generation failed: {str(e)}", "type": "internal_error"}},
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
