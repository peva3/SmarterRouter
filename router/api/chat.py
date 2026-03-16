"""Chat completions endpoint and streaming.

This module implements the OpenAI-compatible `/v1/chat/completions` endpoint,
which is the primary inference interface for SmarterRouter. It handles:
- Request validation (Content-Type, body parsing)
- Prompt sanitization and security checks (prompt injection detection, content moderation)
- Model selection via RouterEngine (or override via query parameter)
- VRAM management (loading/unloading models)
- Response caching (semantic cache)
- Streaming and non-streaming responses
- Tool execution loop for function calling

The endpoint integrates with the routing engine to select the optimal model based on
prompt analysis, model capabilities, benchmarks, and feedback scores. It supports
fallback cascades if the selected model fails and includes comprehensive error
handling and logging.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router.backends.base import LLMBackend, supports_unload
from router.config import Settings, settings
from router.logging_config import sanitize_for_logging
from router.schemas import (
    ChatCompletionRequest,
    close_unclosed_code_block,
    sanitize_model_name,
    sanitize_prompt,
    strip_signature,
)
from router.skills import skills_registry
from router.state import (
    _log_error_with_context,
    app_state,
    get_available_models_with_cache,
    get_model_vram_estimate,
    get_model_vram_estimates_batch,
    get_settings,
    rate_limit_request,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
):
    """Handle chat completion requests via the OpenAI-compatible API.

    This endpoint accepts a POST request with a JSON body conforming to the
    OpenAI ChatCompletion schema. It performs validation, security checks,
    model selection, and generation (with optional streaming and tool use).

    Args:
        request: The incoming FastAPI Request.
        config: Application settings dependency (injected via get_settings).

    Returns:
        Either a non-streaming JSON response with the completion, or a
        StreamingResponse with server-sent events if `stream=true` was requested.

    Raises:
        HTTPException (via JSONResponse) for various error conditions:
        - 400 Bad Request: invalid body, empty prompt, model not found, etc.
        - 403 Forbidden: content moderation blocked request.
        - 413 Payload Too Large: request body exceeds size limit.
        - 415 Unsupported Media Type: Content-Type not application/json.
        - 429 Too Many Requests: rate limit exceeded.
        - 503 Service Unavailable: backend or router not initialized.
        - 504 Gateway Timeout: request timeout exceeded.
        - 500 Internal Server Error: all models failed or unexpected error.
    """
    # Rate limit check for chat endpoint
    await rate_limit_request(request, config, is_admin=False, is_chat=True)

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
    try:
        model_override = sanitize_model_name(request.query_params.get("model"))
    except ValueError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=400,
        )

    # Track request
    if hasattr(app_state, "total_requests"):
        app_state.total_requests += 1

    # Fetch available models once per request (uses cache)
    try:
        available_models = await get_available_models_with_cache()
        model_names = [m.name for m in available_models]

        # Model override - skip routing and use specified model
        if model_override:
            # Use already fetched model_names for validation
            # (no need to refetch)
            selected_model = None

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
            logger.debug(
                f"Model override: {selected_model}, prompt: {sanitize_for_logging(prompt)}"
            )
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
            logger.debug(f"Routed to: {selected_model}, prompt: {sanitize_for_logging(prompt)}")
    except Exception as e:
        _log_error_with_context("Routing failed", request=request, prompt=prompt, exc=e)
        models = await get_available_models_with_cache()
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
    backend_kwargs: dict[str, Any] = {
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
        "keep_alive": config.model_keep_alive,
    }
    # Remove None values
    backend_kwargs = {k: v for k, v in backend_kwargs.items() if v is not None}

    if stream:
        # Load model via VRAM manager if enabled, else fallback to traditional unload
        if app_state.vram_manager:
            vram_gb = get_model_vram_estimate(selected_model)
            await app_state.vram_manager.load_model(selected_model, vram_gb)
        else:
            # Traditional: unload current model if different and not pinned before loading new
            current = app_state.current_loaded_model
            pinned = config.pinned_model
            if current and current != selected_model and current != pinned:
                logger.info(
                    f"VRAM management (streaming): unloading {current} to load {selected_model}"
                )
                if supports_unload(app_state.backend):
                    await app_state.backend.unload_model(current)
                app_state.current_loaded_model = None

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
        available_models = await get_available_models_with_cache()
        fallback_list = [m.name for m in available_models if m.name != selected_model]
        # Put selected_model first in retry list
        fallback_list = [selected_model] + fallback_list
    except Exception:
        fallback_list = [selected_model]

    # Pre-fetch VRAM estimates for all fallback models to avoid N+1 queries
    vram_estimate_map: dict[str, float] = {}
    if app_state.vram_manager:
        # Use batched VRAM estimates function
        vram_estimate_map = get_model_vram_estimates_batch(fallback_list)

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
                "created": int(datetime.now(UTC).timestamp()),
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
                exc_info=True,
            )
            continue

    if response is None:
        _log_error_with_context(
            "All models failed",
            request=request,
            model_name=selected_model,
            prompt=prompt,
            exc=last_error,
        )
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
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                _log_error_with_context(
                    f"Failed to parse tool arguments for {tool_name}",
                    request=request,
                    model_name=final_model,
                    prompt=prompt,
                    exc=e,
                )
                continue

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
        # Close any unclosed fenced code block before appending signature
        content = close_unclosed_code_block(content)
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
        "created": datetime.now(UTC).timestamp(),
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
    """Stream chat completions using Server-Sent Events (SSE).

    This async generator yields SSE-formatted chunks as the LLM generates tokens.
    It handles the streaming HTTP response for the chat completions endpoint.

    Args:
        client: The LLM backend client to use for generation.
        model: The model name to generate with.
        messages: List of message dictionaries with 'role' and 'content'.
        reasoning: Human-readable explanation of why this model was selected.
        config: Application settings.
        chunk_id: Unique ID for this response (included in each chunk).
        **kwargs: Additional backend-specific parameters (temperature, max_tokens, etc.).

    Yields:
        str: SSE-formatted data lines (e.g., "data: {...}\n\n").

    Errors:
        Any exception during streaming is caught and yields an error chunk.
        The error is also logged with context via `_log_error_with_context`.
    """
    created = datetime.now(UTC).timestamp()

    try:
        stream, latency = await client.chat_streaming(model, messages, **kwargs)

        # Initial chunk with metadata
        initial_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
            "router": {"reasoning": reasoning},
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        accumulated_content = ""

        async for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                accumulated_content += content

                content_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

            if chunk.get("done", False):
                # Use schemas.py function to handle code blocks properly
                closed_content = close_unclosed_code_block(accumulated_content)

                # If content was modified (fence added or removed), emit the difference
                if closed_content != accumulated_content:
                    # Find what was added (closing fence or removal)
                    diff = (
                        closed_content[len(accumulated_content) :]
                        if closed_content.startswith(accumulated_content)
                        else ""
                    )

                    # If we need to add a closing fence (not just remove stray)
                    if diff.strip():
                        fence_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": diff},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(fence_chunk)}\n\n"

                # Add signature if enabled
                if config.signature_enabled:
                    signature = config.signature_format.format(model=model)
                    signature_chunk = {
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
                    yield f"data: {json.dumps(signature_chunk)}\n\n"
                else:
                    done_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(done_chunk)}\n\n"
    except Exception as e:
        prompt_for_hash = None
        if messages:
            last_msg = messages[-1]
            prompt_for_hash = str(last_msg.get("content", ""))
        _log_error_with_context(
            "Streaming failed",
            model_name=model,
            prompt=prompt_for_hash,
            exc=e,
            exc_info=True,
        )
        error_message = str(e)

        # Provide more helpful error messages for common issues
        if "timeout" in error_message.lower():
            error_message = f"Timeout error: The model took too long to respond. Current timeout: {config.generation_timeout}s. Try increasing ROUTER_GENERATION_TIMEOUT."
        elif "connection" in error_message.lower():
            error_message = "Connection error: Could not connect to the LLM backend. Please check that Ollama is running and accessible."

        error_data = {"error": {"message": error_message, "type": "internal_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"

    yield "data: [DONE]\n\n"
