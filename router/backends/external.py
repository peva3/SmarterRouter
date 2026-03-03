"""
External Provider Backend Factory

Creates and manages LLM backends for external providers:
- OpenAI (openai/gpt-4, openai/gpt-4o, etc.)
- Anthropic (anthropic/claude-3-opus, anthropic/claude-3-sonnet, etc.)
- Google (google/gemini-1.5-pro, etc.)
- Cohere (cohere/command-r-plus, etc.)
- Mistral (mistral/mistral-large, etc.)

Uses the provider prefix in model names to route to the correct backend.
"""

import logging

from router.backends.base import LLMBackend
from router.backends.openai import OpenAIBackend
from router.config import settings

logger = logging.getLogger(__name__)

# Provider configurations
PROVIDER_CONFIGS: dict[str, dict[str, str | None]] = {
    "openai": {
        "default_base_url": "https://api.openai.com/v1",
        "api_key_field": "openai_api_key",
        "base_url_field": "openai_base_url",
        "model_prefix": "",
    },
    "anthropic": {
        "default_base_url": "https://api.anthropic.com/v1",
        "api_key_field": "anthropic_api_key",
        "base_url_field": "anthropic_base_url",
        "model_prefix": "",  # Anthropic uses full model ID in request
    },
    "google": {
        "default_base_url": "https://generativelanguage.googleapis.com/v1",
        "api_key_field": "google_api_key",
        "base_url_field": "google_base_url",
        "model_prefix": "models/",
    },
    "cohere": {
        "default_base_url": "https://api.cohere.ai/v1",
        "api_key_field": "cohere_api_key",
        "base_url_field": "cohere_base_url",
        "model_prefix": "",
    },
    "mistral": {
        "default_base_url": "https://api.mistral.ai/v1",
        "api_key_field": "mistral_api_key",
        "base_url_field": "mistral_base_url",
        "model_prefix": "",
    },
}


class ExternalBackendFactory:
    """Factory for creating external provider backends."""

    def __init__(self):
        self._backends: dict[str, OpenAIBackend] = {}

    def _get_provider_from_model(self, model_name: str) -> str | None:
        """Extract provider prefix from model name.

        Examples:
            "openai/gpt-4" -> "openai"
            "anthropic/claude-3-opus" -> "anthropic"
            "gpt-4" -> None (no prefix)
        """
        if "/" in model_name:
            return model_name.split("/")[0]
        return None

    def get_backend(self, model_name: str) -> LLMBackend | None:
        """Get or create backend for the given model.

        Args:
            model_name: Model name (e.g., "openai/gpt-4" or just "gpt-4")

        Returns:
            LLMBackend instance or None if no external backend configured
        """
        provider = self._get_provider_from_model(model_name)

        # Check if this is an external provider we should handle
        if provider not in PROVIDER_CONFIGS:
            return None

        # Check if provider is enabled
        if provider not in settings.external_providers:
            logger.debug(f"Provider '{provider}' not in enabled external providers")
            return None

        # Return cached backend if available
        if provider in self._backends:
            return self._backends[provider]

        # Create new backend for this provider
        config = PROVIDER_CONFIGS[provider]
        api_key_field = config["api_key_field"]
        base_url_field = config.get("base_url_field")

        api_key: str | None = getattr(settings, api_key_field, None) if api_key_field else None
        base_url: str | None = getattr(settings, base_url_field, None) if base_url_field else None

        # Use default URL if not specified
        if not base_url:
            base_url = config["default_base_url"]

        if not api_key or not base_url:
            logger.warning(f"No API key configured for provider '{provider}'")
            return None

        model_prefix = config.get("model_prefix", "") or ""

        # Create the backend
        backend = OpenAIBackend(
            base_url=base_url,
            api_key=api_key,
            model_prefix=model_prefix,
            timeout=settings.generation_timeout,
        )

        self._backends[provider] = backend
        logger.info(f"Created external backend for provider: {provider}")

        return backend

    def is_external_model(self, model_name: str) -> bool:
        """Check if model name belongs to an external provider."""
        provider = self._get_provider_from_model(model_name)
        return provider in PROVIDER_CONFIGS

    async def close_all(self) -> None:
        """Close all external backend connections."""
        for backend in self._backends.values():
            if hasattr(backend, "close"):
                await backend.close()
        self._backends.clear()


# Global factory instance
_external_backend_factory: ExternalBackendFactory | None = None


def get_external_backend_factory() -> ExternalBackendFactory:
    """Get the global external backend factory instance."""
    global _external_backend_factory
    if _external_backend_factory is None:
        _external_backend_factory = ExternalBackendFactory()
    return _external_backend_factory
