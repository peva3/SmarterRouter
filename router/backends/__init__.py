import logging
from typing import TYPE_CHECKING

from router.backends.base import LLMBackend
from router.encryption import get_encryption_manager

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
                config=settings,
            )

        case "llama.cpp" | "llama-cpp" | "llamaswap" | "llama-swap":
            from router.backends.llama_cpp import LlamaCppBackend

            return LlamaCppBackend(
                base_url=settings.llama_cpp_url
                or settings.openai_base_url
                or "http://localhost:8080",
                model_prefix=settings.model_prefix,
                timeout=settings.generation_timeout,  # Use longer timeout for generation
                config=settings,
            )

        case "openai":
            from router.backends.openai import OpenAIBackend

            manager = get_encryption_manager()
            openai_api_key = manager.maybe_decrypt(settings.openai_api_key)

            return OpenAIBackend(
                base_url=settings.openai_base_url or "https://api.openai.com/v1",
                api_key=openai_api_key or "EMPTY",
                model_prefix=settings.model_prefix,
                timeout=settings.generation_timeout,  # Use longer timeout for external APIs
                config=settings,
            )

        case _:
            raise ValueError(f"Unknown provider: {provider}. Supported: ollama, llama.cpp, openai")
