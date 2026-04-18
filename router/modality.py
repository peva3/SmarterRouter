"""Modality detection and routing for SmarterRouter.

This module provides lightweight modality detection from request shapes
and modality-aware model filtering for the routing engine.

Supported modalities:
- text: Standard text-based chat/completions
- vision: Image inputs (image_url content parts)
- tool_calling: Function/tool calling capabilities
- embedding: Text embedding generation
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from router.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported model interaction modalities."""

    TEXT = "text"
    VISION = "vision"
    TOOL_CALLING = "tool_calling"
    EMBEDDING = "embedding"

    def __str__(self) -> str:
        return self.value


class ModalityDetector:
    """Detects the modality of incoming requests from request shape.

    Uses lightweight heuristics based on message content, tools, and endpoint.
    """

    @staticmethod
    def from_chat_request(request: ChatCompletionRequest) -> Modality:
        """Detect modality from a chat completion request.

        Detection order:
        1. Check for vision (image_url content parts)
        2. Check for tool_calling (tools present)
        3. Default to text

        Args:
            request: The chat completion request

        Returns:
            Detected modality
        """
        # Check for vision inputs in messages
        if hasattr(request, "messages") and request.messages:
            for msg in request.messages:
                content = getattr(msg, "content", None)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            logger.debug("Detected vision modality from image_url content")
                            return Modality.VISION

        # Check for tool calling
        if hasattr(request, "tools") and request.tools:
            logger.debug("Detected tool_calling modality from tools present")
            return Modality.TOOL_CALLING

        # Default to text
        return Modality.TEXT

    @staticmethod
    def from_embeddings_request() -> Modality:
        """Embeddings requests are always embedding modality.

        Returns:
            Modality.EMBEDDING
        """
        return Modality.EMBEDDING


def get_models_for_modality_OLD(
    available_models: list[str],
    modality,
    model_profiles: dict[str, Any] | None = None,
) -> list[str]:
    """Filter models by modality."""
    candidates: list[str] = []
    for model_name in available_models:
        profile = model_profiles.get(model_name) if model_profiles else None
        if modality == Modality.VISION:
            if _supports_vision(model_name, profile):
                candidates.append(model_name)
        elif modality == Modality.TOOL_CALLING:
            if _supports_tool_calling(model_name, profile):
                candidates.append(model_name)
        elif modality == Modality.EMBEDDING:
            if _supports_embedding(model_name, profile):
                candidates.append(model_name)
        else:
            candidates.append(model_name)
    if not candidates and available_models:
        return available_models
    return candidates

def _supports_vision(model_name: str, profile: Any | None = None) -> bool:
    """Check if a model supports vision capabilities.

    Checks in order:
    1. Profile vision flag
    2. Name-based heuristics
    """
    # Check profile first
    if profile and hasattr(profile, "vision") and profile.vision:
        return True

    # Name-based heuristics
    name_lower = model_name.lower()
    vision_indicators = [
        "llava",
        "pixtral",
        "gpt-4o",
        "gpt-4-vision",
        "claude-3",
        "gemini",
        "qwen-vl",
        "internvl",
        "cogvlm",
        "bakllava",
        "moondream",
        "gemma4",  # Gemma 4 series supports vision
    ]
    return any(ind in name_lower for ind in vision_indicators)


def _supports_tool_calling(model_name: str, profile: Any | None = None) -> bool:
    """Check if a model supports tool calling.

    Checks in order:
    1. Profile tool_calling flag
    2. Name-based heuristics
    """
    # Check profile first
    if profile and hasattr(profile, "tool_calling") and profile.tool_calling:
        return True

    # Name-based heuristics for models known to support tools well
    name_lower = model_name.lower()
    tool_indicators = [
        "gpt-4",
        "claude-3",
        "qwen2.5",
        "mistral-large",
        "command-r",
        "llama3.1",
        "llama3.2",
        "gemma4",  # Gemma 4 series supports tool calling
    ]
    return any(ind in name_lower for ind in tool_indicators)


def _supports_embedding(model_name: str, profile: Any | None = None) -> bool:
    """Check if a model supports embeddings.

    Most modern models support embeddings, but some are specifically
    designed for it.
    """
    # Check profile first
    if profile and hasattr(profile, "supports_embedding"):
        return bool(profile.supports_embedding)

    # Name-based heuristics
    name_lower = model_name.lower()
    embedding_indicators = [
        "embed",
        "nomic",
        "mxbai",
        "bge-",
        "gte-",
        "e5-",
        "text-embedding",
    ]
    return any(ind in name_lower for ind in embedding_indicators)


# Fixed version - the original get_models_for_modality has a mysterious bug
def filter_models_by_modality(
    available_models: list[str],
    modality,
    model_profiles: dict[str, Any] | None = None,
) -> list[str]:
    """Filter models by modality - working version."""
    candidates: list[str] = []
    for model_name in available_models:
        profile = model_profiles.get(model_name) if model_profiles else None
        if modality == Modality.VISION:
            if _supports_vision(model_name, profile):
                candidates.append(model_name)
        elif modality == Modality.TOOL_CALLING:
            if _supports_tool_calling(model_name, profile):
                candidates.append(model_name)
        elif modality == Modality.EMBEDDING:
            if _supports_embedding(model_name, profile):
                candidates.append(model_name)
        else:
            candidates.append(model_name)
    if not candidates and available_models:
        return available_models
    return candidates


# Alias for backward compatibility
get_models_for_modality = filter_models_by_modality


def filter_models_by_modality(
    available_models: list[str],
    modality,
    model_profiles: dict[str, Any] | None = None,
) -> list[str]:
    """Filter models by modality - working version."""
    candidates: list[str] = []
    for model_name in available_models:
        profile = model_profiles.get(model_name) if model_profiles else None
        if modality == Modality.VISION:
            if _supports_vision(model_name, profile):
                candidates.append(model_name)
        elif modality == Modality.TOOL_CALLING:
            if _supports_tool_calling(model_name, profile):
                candidates.append(model_name)
        elif modality == Modality.EMBEDDING:
            if _supports_embedding(model_name, profile):
                candidates.append(model_name)
        else:
            candidates.append(model_name)
    if not candidates and available_models:
        return available_models
    return candidates


# Update alias to point to working version
get_models_for_modality = filter_models_by_modality
