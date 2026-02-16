import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    name: str
    size: int | None = None
    modified_at: str | None = None


class LLMBackend(Protocol):
    """Common interface for all LLM backends."""

    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        ...

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        keep_alive: float = -1,
        **kwargs: object,
    ) -> dict:
        """Send a chat completion request."""
        ...

    async def chat_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
        keep_alive: float = -1,
    ) -> tuple[AsyncIterator[dict], float]:
        """Send a streaming chat request. Returns (iterator, latency_ms)."""
        ...

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs: object,
    ) -> dict:
        """Send a generate request (non-chat)."""
        ...

    async def unload_model(self, model_name: str) -> bool:
        """Unload model from VRAM. Return False if not supported."""
        return False

    async def load_model(self, model_name: str, keep_alive: float = -1) -> bool:
        """Explicitly load a model into VRAM. Return False if not supported."""
        return False


def supports_unload(backend: LLMBackend) -> bool:
    """Check if backend supports model unloading."""
    return hasattr(backend, "unload_model") and backend.unload_model != LLMBackend.unload_model
