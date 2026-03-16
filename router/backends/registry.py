"""
Backend Registry - Manages multiple LLM backends

Supports:
- Local Ollama instance
- External providers (OpenAI, Anthropic, Google, etc.)
- Automatic model discovery from all configured backends
"""

import logging
import time
from typing import Any

from router.backends import create_backend
from router.backends.base import LLMBackend, ModelInfo
from router.config import settings
from router.provider_db import ProviderDB, get_provider_db

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Manages multiple LLM backends and provides unified interface.

    The registry automatically discovers models from:
    1. Local backend (Ollama) - if configured
    2. External providers (OpenAI, Anthropic, etc.) - via provider.db

    When external_providers_enabled is True, models from provider.db
    are included in model discovery for intelligent routing.
    """

    def __init__(self):
        self._local_backend: LLMBackend | None = None
        self._provider_db: ProviderDB | None = None
        self._initialized = False
        self._external_models_cache: list[ModelInfo] = []
        self._external_models_cache_time: float = 0.0
        self._external_models_cache_ttl: float = 30.0

    def initialize(self) -> None:
        """Initialize all configured backends."""
        if self._initialized:
            return

        # Initialize local backend (Ollama)
        try:
            self._local_backend = create_backend(settings)
            logger.info(f"Initialized local backend: {settings.provider}")
        except Exception as e:
            logger.warning(f"Failed to initialize local backend: {e}")
            self._local_backend = None

        # Initialize provider.db
        if settings.provider_db_enabled:
            self._provider_db = get_provider_db()
            if self._provider_db.is_available():
                stats = self._provider_db.get_stats()
                logger.info(f"Provider.db available: {stats.get('total_models', 0)} models")
            else:
                logger.warning(
                    f"provider.db not found at {settings.provider_db_path}. "
                    "External provider routing disabled."
                )

        self._initialized = True

    @property
    def local_backend(self) -> LLMBackend | None:
        """Get the local backend instance."""
        return self._local_backend

    @property
    def provider_db(self) -> ProviderDB | None:
        """Get the provider.db instance."""
        return self._provider_db

    async def list_models(self) -> list[ModelInfo]:
        """
        List all available models from all backends.

        Returns models from:
        - Local backend (Ollama)
        - External providers (if enabled)
        """
        models: list[ModelInfo] = []

        # Get local models
        if self._local_backend:
            try:
                local_models = await self._local_backend.list_models()
                models.extend(local_models)
                logger.debug(f"Found {len(local_models)} local models")
            except Exception as e:
                logger.warning(f"Failed to list local models: {e}")

        # Get external models from provider.db
        if (
            settings.external_providers_enabled
            and self._provider_db
            and self._provider_db.is_available()
        ):
            try:
                now = time.time()
                if (
                    self._external_models_cache
                    and (now - self._external_models_cache_time) < self._external_models_cache_ttl
                ):
                    models.extend(self._external_models_cache)
                    logger.debug(
                        "Using cached external model list (age: %.1fs)",
                        now - self._external_models_cache_time,
                    )
                else:
                    external_benchmarks = self._provider_db.get_all_benchmarks()
                    external_models: list[ModelInfo] = []
                    # Convert benchmark entries to ModelInfo
                    for bench in external_benchmarks:
                        model_id = bench.get("model_id", "")
                        if model_id:
                            # Create ModelInfo for external models
                            # Note: These don't have size/modified_at from local backend
                            external_models.append(
                                ModelInfo(
                                    name=model_id,
                                    size=None,
                                    modified_at=None,
                                )
                            )

                    self._external_models_cache = external_models
                    self._external_models_cache_time = now
                    models.extend(external_models)
                    logger.debug(
                        "Found %d external models from provider.db",
                        len(external_models),
                    )
            except Exception as e:
                logger.warning(f"Failed to list external models: {e}")

        return models

    def is_local_model(self, model_name: str) -> bool:
        """Check if model exists in local backend."""
        if not self._local_backend:
            return False
        # Simple check - could be enhanced with actual model list
        return True

    def is_external_model(self, model_name: str) -> bool:
        """Check if model exists in provider.db."""
        if not self._provider_db or not self._provider_db.is_available():
            return False

        # Check if model is in provider.db
        benchmark = self._provider_db.get_benchmark(model_name)
        return benchmark is not None

    def get_backend_for_model(self, model_name: str) -> tuple[str, LLMBackend | None]:
        """
        Determine which backend to use for a model.

        Returns:
            Tuple of (backend_type, backend_instance)
            - ("local", backend) - for local Ollama models
            - ("external", None) - for external models (uses OpenAI-compatible API)
            - ("unknown", None) - if model not found
        """
        # Check if it's an external model from provider.db first
        # External models typically have a slash (e.g., "openai/gpt-4")
        if "/" in model_name and self._provider_db and self._provider_db.is_available():
            if self._provider_db.get_benchmark(model_name):
                return ("external", None)

        # Check if it's a local model (no slash or in filter)
        if self._local_backend:
            # Local models typically don't have a slash
            if "/" not in model_name:
                return ("local", self._local_backend)

        # Default to local if we have one
        if self._local_backend:
            return ("local", self._local_backend)

        return ("unknown", None)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Send chat request to appropriate backend."""
        backend_type, backend = self.get_backend_for_model(model)

        if backend_type == "local" and backend:
            return await backend.chat(model, messages, stream=stream, **kwargs)

        if backend_type == "external":
            # Get external backend from factory
            from router.backends.external import get_external_backend_factory

            factory = get_external_backend_factory()
            external_backend = factory.get_backend(model)

            if external_backend:
                return await external_backend.chat(model, messages, stream=stream, **kwargs)

            raise ValueError(
                f"No external backend configured for model '{model}'. "
                "Ensure the provider is in ROUTER_EXTERNAL_PROVIDERS and "
                "the appropriate API key is set."
            )

        raise ValueError(f"Unknown model: {model}")

    async def close(self) -> None:
        """Close all backend connections."""
        if self._local_backend and hasattr(self._local_backend, "close"):
            await self._local_backend.close()

        # Close external backends
        from router.backends.external import get_external_backend_factory

        factory = get_external_backend_factory()
        await factory.close_all()


# Global registry instance
_backend_registry: BackendRegistry | None = None


def get_backend_registry() -> BackendRegistry:
    """Get the global backend registry instance."""
    global _backend_registry
    if _backend_registry is None:
        _backend_registry = BackendRegistry()
        _backend_registry.initialize()
    return _backend_registry
