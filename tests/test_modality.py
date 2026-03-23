"""Tests for modality detection and routing.

Covers:
- Modality detection from request shapes
- Model filtering by modality
- Fallback behavior when no modality-specific models found
"""

import pytest

from router.modality import Modality, ModalityDetector, get_models_for_modality
from router.schemas import ChatCompletionRequest, ChatMessage


class TestModalityEnum:
    """Test the Modality enum."""

    def test_modality_values(self):
        """Modality enum has expected values."""
        assert Modality.TEXT.value == "text"
        assert Modality.VISION.value == "vision"
        assert Modality.TOOL_CALLING.value == "tool_calling"
        assert Modality.EMBEDDING.value == "embedding"

    def test_modality_str(self):
        """Modality enum converts to string."""
        assert str(Modality.TEXT) == "text"
        assert str(Modality.VISION) == "vision"


class TestModalityDetector:
    """Test modality detection from request shapes."""

    def test_detects_text_by_default(self):
        """Plain text messages default to TEXT modality."""
        request = ChatCompletionRequest(
            model=None,
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert ModalityDetector.from_chat_request(request) == Modality.TEXT

    def test_detects_vision_from_image_url(self):
        """Image URL content parts trigger VISION modality."""
        request = ChatCompletionRequest(
            model=None,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                    ],
                )
            ],
        )
        assert ModalityDetector.from_chat_request(request) == Modality.VISION

    def test_detects_vision_from_image_only(self):
        """Message with only image triggers VISION modality."""
        request = ChatCompletionRequest(
            model=None,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    ],
                )
            ],
        )
        assert ModalityDetector.from_chat_request(request) == Modality.VISION

    def test_detects_tools_from_tools_present(self):
        """Tools in request trigger TOOL_CALLING modality."""
        request = ChatCompletionRequest(
            model=None,
            messages=[ChatMessage(role="user", content="Get the weather")],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert ModalityDetector.from_chat_request(request) == Modality.TOOL_CALLING

    def test_vision_takes_precedence_over_tools(self):
        """Vision detection takes precedence over tool detection."""
        request = ChatCompletionRequest(
            model=None,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                    ],
                )
            ],
            tools=[{"type": "function", "function": {"name": "analyze_image"}}],
        )
        # Vision should be detected first
        assert ModalityDetector.from_chat_request(request) == Modality.VISION

    def test_handles_empty_messages(self):
        """Empty messages list defaults to TEXT."""
        request = ChatCompletionRequest(
            model=None,
            messages=[ChatMessage(role="user", content="")],
        )
        assert ModalityDetector.from_chat_request(request) == Modality.TEXT

    def test_handles_none_content(self):
        """None content in messages defaults to TEXT."""
        request = ChatCompletionRequest(
            model=None,
            messages=[ChatMessage(role="user", content=None)],
        )
        assert ModalityDetector.from_chat_request(request) == Modality.TEXT


class TestGetModelsForModality:
    """Test modality-based model filtering."""

    def test_text_modality_includes_all_models(self):
        """TEXT modality includes all models."""
        models = ["llama3", "mistral", "gpt-4", "claude-3"]
        candidates = get_models_for_modality(models, Modality.TEXT)
        assert set(candidates) == set(models)

    def test_vision_modality_filters_by_name(self):
        """VISION modality filters using name heuristics."""
        models = [
            "llama3.1",  # no vision
            "llava-phi3",  # vision
            "gpt-4o",  # vision
            "mistral",  # no vision
            "pixtral-12b",  # vision
            "claude-3-opus",  # vision
        ]
        candidates = get_models_for_modality(models, Modality.VISION)
        # Should include vision-capable models
        assert "llava-phi3" in candidates
        assert "gpt-4o" in candidates
        assert "pixtral-12b" in candidates
        assert "claude-3-opus" in candidates

    def test_tool_calling_modality_filters_by_name(self):
        """TOOL_CALLING modality filters using name heuristics."""
        models = [
            "llama3.1",
            "mistral-large",
            "qwen2.5",
            "claude-3-sonnet",
            "command-r",
        ]
        candidates = get_models_for_modality(models, Modality.TOOL_CALLING)
        # Should include tool-capable models
        assert "mistral-large" in candidates
        assert "qwen2.5" in candidates
        assert "claude-3-sonnet" in candidates
        assert "command-r" in candidates

    def test_embedding_modality_filters_by_name(self):
        """EMBEDDING modality filters using name heuristics."""
        models = [
            "llama3",
            "nomic-embed-text",
            "mxbai-embed-large",
            "text-embedding-3-small",
            "mistral",
        ]
        candidates = get_models_for_modality(models, Modality.EMBEDDING)
        # Should include embedding-specific models
        assert "nomic-embed-text" in candidates
        assert "mxbai-embed-large" in candidates
        assert "text-embedding-3-small" in candidates

    def test_fallback_when_no_matching_models(self):
        """Fallback to all models when no modality-specific models found."""
        models = ["llama3", "mistral", "qwen2"]
        # None of these have vision in their names
        candidates = get_models_for_modality(models, Modality.VISION)
        # Should fallback to all models
        assert set(candidates) == set(models)

    def test_empty_model_list_returns_empty(self):
        """Empty input returns empty output."""
        candidates = get_models_for_modality([], Modality.VISION)
        assert candidates == []

    def test_uses_profile_when_available(self):
        """Uses profile vision/tool_calling flags when available."""
        models = ["custom-vision-model", "regular-model"]

        class MockProfile:
            def __init__(self, vision=False, tool_calling=False):
                self.vision = vision
                self.tool_calling = tool_calling

        profiles = {
            "custom-vision-model": MockProfile(vision=True),
            "regular-model": MockProfile(vision=False),
        }

        candidates = get_models_for_modality(
            models, Modality.VISION, model_profiles=profiles
        )

        assert "custom-vision-model" in candidates
        assert "regular-model" not in candidates


class TestModalityIntegration:
    """Integration tests for modality in routing context."""

    def test_vision_request_detected_correctly(self):
        """End-to-end: vision request is detected and filtered."""
        # Simulate a vision request
        request = ChatCompletionRequest(
            model=None,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/photo.jpg"}},
                    ],
                )
            ],
        )

        modality = ModalityDetector.from_chat_request(request)
        assert modality == Modality.VISION

        # Filter available models
        available = [
            "llama3.1",
            "llava-34b",
            "mistral",
            "gpt-4o",
            "claude-3-sonnet",
        ]
        candidates = get_models_for_modality(available, modality)

        # Should prefer vision models
        assert "llava-34b" in candidates
        assert "gpt-4o" in candidates
        assert "claude-3-sonnet" in candidates

    def test_tool_request_detected_correctly(self):
        """End-to-end: tool request is detected and filtered."""
        request = ChatCompletionRequest(
            model=None,
            messages=[ChatMessage(role="user", content="Calculate 2+2")],
            tools=[{"type": "function", "function": {"name": "calculator"}}],
        )

        modality = ModalityDetector.from_chat_request(request)
        assert modality == Modality.TOOL_CALLING

        available = [
            "llama3.1",
            "mistral-large",
            "phi3",
            "qwen2.5",
        ]
        candidates = get_models_for_modality(available, modality)

        # Should prefer tool-capable models
        assert "mistral-large" in candidates
        assert "qwen2.5" in candidates


class TestModalityEdgeCases:
    """Edge cases and boundary conditions."""

    def test_handles_missing_messages_attribute(self):
        """Handles objects without messages attribute gracefully."""
        class FakeRequest:
            pass

        # Should not crash and default to TEXT
        result = ModalityDetector.from_chat_request(FakeRequest())  # type: ignore
        assert result == Modality.TEXT

    def test_handles_none_request_obj(self):
        """Handles None request object."""
        result = ModalityDetector.from_chat_request(None)  # type: ignore
        assert result == Modality.TEXT

    def test_vision_detection_in_multipart_messages(self):
        """Vision detected when multiple messages with mixed content."""
        request = ChatCompletionRequest(
            model=None,
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(
                    role="assistant",
                    content="Hi there!",
                ),
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                    ],
                ),
            ],
        )
        assert ModalityDetector.from_chat_request(request) == Modality.VISION

    def test_text_with_complex_content_structure(self):
        """Text modality when content has text-only multipart."""
        request = ChatCompletionRequest(
            model=None,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Part 1"},
                        {"type": "text", "text": "Part 2"},
                    ],
                )
            ],
        )
        # No image_url parts, so should be TEXT
        assert ModalityDetector.from_chat_request(request) == Modality.TEXT
