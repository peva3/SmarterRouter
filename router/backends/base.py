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

    async def unload_model(self, model_name: str) -> bool:
        """Unload model from VRAM. Return False if not supported."""
        return False

    async def load_model(self, model_name: str, keep_alive: float = -1) -> bool:
        """Explicitly load a model into VRAM. Return False if not supported."""
        return False

    async def embed(
        self,
        model: str,
        input_text: str | list[str],
        **kwargs: object,
    ) -> dict:
        """Generate embeddings for the given input."""
        ...

    async def get_model_vram_usage(self, model_name: str) -> float | None:
        """Get VRAM usage for a specific model in GB. Return None if not supported."""
        return None


def supports_unload(backend: LLMBackend) -> bool:
    """Check if backend supports model unloading."""
    from router.backends.ollama import OllamaBackend
    return isinstance(backend, OllamaBackend)
