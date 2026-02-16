import logging
from typing import TYPE_CHECKING

from router.backends.base import LLMBackend

if TYPE_CHECKING:
    from router.config import Settings

logger = logging.getLogger(__name__)


def create_backend(settings: "Settings") -> LLMBackend:
    """Factory function to create the appropriate backend based on settings.

    Supported providers:
    - ollama: Local Ollama instance (default)
    - llama.cpp: llama.cpp server or llama-swap
    - openai: OpenAI-compatible API (OpenAI, Anthropic, local AI, etc.)
    """
    provider = settings.provider.lower()

    match provider:
        case "ollama":
            from router.backends.ollama import OllamaBackend
            return OllamaBackend(
                base_url=settings.ollama_url,
                timeout=settings.profile_timeout,
                generation_timeout=settings.generation_timeout,
            )

        case "llama.cpp" | "llama-cpp" | "llamaswap" | "llama-swap":
            from router.backends.llama_cpp import LlamaCppBackend
            return LlamaCppBackend(
                base_url=settings.llama_cpp_url or settings.openai_base_url or "http://localhost:8080",
                model_prefix=settings.model_prefix,
                timeout=settings.generation_timeout,  # Use longer timeout for generation
            )

        case "openai":
            from router.backends.openai import OpenAIBackend
            return OpenAIBackend(
                base_url=settings.openai_base_url or "https://api.openai.com/v1",
                api_key=settings.openai_api_key or "EMPTY",
                model_prefix=settings.model_prefix,
                timeout=settings.generation_timeout,  # Use longer timeout for external APIs
            )

        case _:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported: ollama, llama.cpp, openai"
            )
