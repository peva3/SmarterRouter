"""Model listing, skills, embeddings, and feedback endpoints."""
import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from router.config import Settings
from router.database import get_session
from router.models import (
    ModelFeedback,
    RoutingDecision,
)
from router.schemas import (
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    FeedbackRequest,
    UsageInfo,
)
from router.modality import Modality, get_models_for_modality
from router.skills import skills_registry
from router.state import (
    _log_error_with_context,
    app_state,
    get_settings,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models")
async def list_models(config: Annotated[Settings, Depends(get_settings)]):
    """List all available LLM models.

    Returns an OpenAI-compatible model list. Always includes the 'router' model,
    which represents the SmarterRouter routing model itself. Depending on configuration,
    may also include external provider models from provider.db and/or all local backend models.

    Args:
        config: Application settings dependency.

    Returns:
        dict: OpenAI-format model list response with `object: "list"` and `data` array.
        Each model entry contains at least:
        - `id`: model name/identifier
        - `object`: "model"
        - `created`: timestamp
        - `owned_by`: provider or "local"
        - `description`: description for router model
        - `admin_auth_required`: boolean if admin API key is configured

    Errors:
        If backend query fails, logs warning and returns only the router model.
    """

    # 1. The main router model
    router_model = {
        "id": config.router_external_model_name,
        "object": "model",
        "created": int(datetime.now(UTC).timestamp()),
        "owned_by": "local",
        "description": "An intelligent router that selects the best LLM based on prompt analysis and model capabilities.",
        "admin_auth_required": config.admin_api_key is not None,
    }

    data = [router_model]

    # 2. External models (if enabled)
    if config.external_providers_enabled and app_state.backend:
        try:
            # If backend is a registry, we can list models efficiently
            # Note: registry.list_models() might be slow if it queries local backend
            # But we want specifically external ones or all?
            # Let's just use what registry returns, which includes provider.db models
            if hasattr(app_state.backend, "list_models"):
                all_models = await app_state.backend.list_models()

                # Filter for external models only (contain "/") to avoid cluttering with local ones
                # unless explicitly requested? For now, just show external ones as they are "new" features
                for m in all_models:
                    if "/" in m.name:  # Heuristic for external models
                        data.append(
                            {
                                "id": m.name,
                                "object": "model",
                                "created": int(datetime.now(UTC).timestamp()),
                                "owned_by": m.name.split("/")[0],
                                "permission": [],
                            }
                        )
        except Exception as e:
            logger.warning(f"Failed to list external models: {e}")

    return {
        "object": "list",
        "data": data,
    }


@router.get("/v1/skills")
async def list_skills():
    """List all registered skills (tools) that the router can execute.

    Skills are user-defined functions that can be called by LLM models during
    tool/function calling. This endpoint returns the list of skill names.

    Returns:
        dict: With key `skills` containing a list of skill names (strings).
    """
    return {"skills": skills_registry.list_skills()}


@router.post("/v1/feedback")
async def feedback(
    request: FeedbackRequest,
    config: Annotated[Settings, Depends(get_settings)],
):
    """Submit user feedback for a routing decision.

    Stores user-provided feedback (score, comment, category) in the database.
    If a `response_id` is provided, links the feedback to the original routing decision.
    Otherwise, requires an explicit `model_name`.

    Args:
        request: FeedbackRequest Pydantic model containing:
            - score: float rating (e.g., 1.0-5.0)
            - comment: optional text feedback
            - category: optional category string
            - model_name: optional model name
            - response_id: optional response ID to link
        config: Application settings.

    Returns:
        dict: `{"status": "success", "id": <feedback_id>}` on success.

    Errors:
        403: Feedback collection disabled via config.
        400: Missing required fields (model_name if response_id not found).
        500: Database error or unexpected exception.
    """
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
        _log_error_with_context(
            "Failed to save feedback",
            model_name=request.model_name,
            prompt=request.response_id,
            exc=e,
        )
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def embeddings(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
):
    """Generate embeddings for the given input(s).

    OpenAI-compatible embeddings endpoint. Accepts a single string or list of strings
    and returns vector embeddings from the configured backend.

    Args:
        request: FastAPI Request.
        config: Application settings.

    Returns:
        EmbeddingsResponse: OpenAI-format response containing:
        - `data`: list of embedding objects with `embedding` vector and `index`
        - `model`: model name used
        - `usage`: token usage info

    Errors:
        503: Backend not initialized.
        415: Content-Type not application/json.
        400: Invalid request body.
        500: Embedding generation failed.
    """
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
        # Get available models and validate the requested model supports embeddings
        if app_state.router_engine:
            available_models = await app_state.router_engine.get_available_models_with_cache()
            model_names = [m.name for m in available_models]
            embedding_candidates = get_models_for_modality(
                model_names, Modality.EMBEDDING
            )

            if model not in embedding_candidates:
                # Model doesn't appear to support embeddings - warn but proceed
                logger.warning(
                    "Requested model %s may not support embeddings. "
                    "Known embedding models: %s",
                    model,
                    embedding_candidates[:5],
                )

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
        prompt_for_hash = input_text if isinstance(input_text, str) else "\n".join(input_text[:3])
        _log_error_with_context(
            "Embeddings failed",
            request=request,
            model_name=model,
            prompt=prompt_for_hash,
            exc=e,
        )
        return JSONResponse(
            {
                "error": {
                    "message": f"Embeddings generation failed: {str(e)}",
                    "type": "internal_error",
                }
            },
            status_code=500,
        )
