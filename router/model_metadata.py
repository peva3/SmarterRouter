"""Dynamic model metadata and capability detection.

This module provides a unified system for discovering model capabilities
without hardcoding model names. It queries Ollama's API for model details
and uses intelligent pattern matching for capability inference.

Features:
- Automatic capability detection from Ollama's model manifest
- Fallback pattern matching for models without detailed metadata
- Quantization detection from model tags
- MoE (Mixture of Experts) detection for accurate VRAM estimation
- Extensible via configuration for custom patterns
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx

from router.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Complete model metadata and capabilities."""

    name: str
    """Model name as recognized by Ollama."""

    # Capabilities
    supports_vision: bool = False
    supports_tool_calling: bool = False
    supports_embedding: bool = False
    is_chat_model: bool = True  # Most models support chat

    # Size and architecture
    parameter_count: int | None = None  # e.g., 7_000_000_000 for 7B
    is_moe: bool = False  # Mixture of Experts model
    active_parameters: int | None = None  # For MoE: number of active params
    quantization: str | None = None  # e.g., "q4_k_m", "f16"

    # Source
    details_source: str = "pattern"  # "ollama_api", "pattern", "explicit"
    confidence: float = 1.0  # 0-1 confidence score

    def __str__(self) -> str:
        return f"ModelMetadata({self.name}, vision={self.supports_vision}, tools={self.supports_tool_calling}, moe={self.is_moe})"
    
    def estimate_vram_gb(self) -> float:
        """Estimate VRAM needed for this model in GB.
        
        Uses parameter count, quantization, and MoE architecture
        to provide accurate VRAM estimates.
        
        Returns:
            Estimated VRAM in GB
        """
        if self.parameter_count is None:
            # Default estimate for unknown models
            return 8.0
        
        # Base VRAM from parameters (2 bytes per param for FP16)
        base_vram = (self.parameter_count * 2) / (1024 ** 3)
        
        # Apply quantization factor
        if self.quantization:
            quant_lower = self.quantization.lower()
            if "q4" in quant_lower:
                base_vram *= 0.5  # Q4 is ~50% of FP16 size
            elif "q5" in quant_lower:
                base_vram *= 0.625
            elif "q6" in quant_lower:
                base_vram *= 0.75
            elif "q8" in quant_lower:
                base_vram *= 0.875
            elif "f16" in quant_lower:
                base_vram *= 1.0  # FP16 is baseline
            elif "f32" in quant_lower or "fp32" in quant_lower:
                base_vram *= 2.0  # FP32 is 2x FP16
        
        # MoE models only need VRAM for active parameters
        if self.is_moe and self.active_parameters:
            active_ratio = self.active_parameters / self.parameter_count
            base_vram *= active_ratio
        
        # Add 20% buffer for KV cache and overhead
        return base_vram * 1.2


class ModelMetadataRegistry:
    """Registry for model metadata with automatic discovery and caching.

    Provides:
    - Lazy loading of model metadata from Ollama API
    - Pattern-based fallback detection
    - Configuration-driven custom patterns
    - Thread-safe caching with TTL
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, ModelMetadata] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._last_refresh: float = 0.0
        self._lock = asyncio.Lock()
        self._ollama_url: str | None = None

    async def get_metadata(self, model_name: str) -> ModelMetadata:
        """Get metadata for a model, discovering if needed.

        Args:
            model_name: Model name (e.g., "llama3:70b", "gpt-4o")

        Returns:
            ModelMetadata object with capability flags
        """
        # Check cache first
        if model_name in self._cache:
            timestamp = self._cache_timestamps.get(model_name, 0.0)
            if time.time() - timestamp < self.ttl_seconds:
                return self._cache[model_name]

        # Fetch fresh metadata
        metadata = await self._discover_metadata(model_name)

        # Cache result
        self._cache[model_name] = metadata
        self._cache_timestamps[model_name] = time.time()

        return metadata

    async def refresh_all(self, available_models: list[str]) -> None:
        """Refresh metadata for all available models.

        Args:
            available_models: List of model names to refresh
        """
        async with self._lock:
            # Fetch from Ollama API in batch
            api_metadata = await self._fetch_ollama_metadata()

            # Update cache with API data
            for model_name in available_models:
                # Try to find in API response
                meta = api_metadata.get(model_name)
                if meta:
                    self._cache[model_name] = meta
                    self._cache_timestamps[model_name] = time.time()
                else:
                    # Fall back to pattern detection
                    meta = self._detect_by_patterns(model_name)
                    self._cache[model_name] = meta
                    self._cache_timestamps[model_name] = time.time()

            self._last_refresh = time.time()

    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self._cache.clear()
        self._cache_timestamps.clear()

    async def _discover_metadata(self, model_name: str) -> ModelMetadata:
        """Discover metadata through multiple strategies.

        Order of detection:
        1. Ollama API (if available and model listed)
        2. Pattern-based detection from model name
        3. Explicit configuration overrides

        Args:
            model_name: Model name to detect

        Returns:
            ModelMetadata with detected capabilities
        """
        # 1. Try Ollama API
        if self._ollama_url:
            api_meta = await self._query_ollama_model(model_name)
            if api_meta:
                return api_meta

        # 2. Pattern-based detection
        pattern_meta = self._detect_by_patterns(model_name)
        if pattern_meta.confidence > 0:
            return pattern_meta

        # 3. Default: unknown model, assume basic chat
        return ModelMetadata(
            name=model_name,
            supports_vision=False,
            supports_tool_calling=False,
            supports_embedding=False,
            details_source="default",
            confidence=0.0,
        )

    async def _fetch_ollama_metadata(self) -> dict[str, ModelMetadata]:
        """Fetch all model metadata from Ollama API.

        Returns:
            Dict mapping model names to ModelMetadata
        """
        if not self._ollama_url:
            return {}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(urljoin(self._ollama_url, "/api/tags"))
                response.raise_for_status()
                data = response.json()

                metadata_map: dict[str, ModelMetadata] = {}
                for model_data in data.get("models", []):
                    name = model_data.get("name", "")
                    if not name:
                        continue

                    meta = self._parse_ollama_model_data(model_data)
                    metadata_map[name] = meta

                return metadata_map
        except Exception as e:
            logger.debug(f"Failed to fetch Ollama metadata: {e}")
            return {}

    def _parse_ollama_model_data(self, model_data: dict[str, Any]) -> ModelMetadata:
        """Parse Ollama model data into ModelMetadata.

        Ollama's /api/tags response includes:
        - name: "llama3:70b"
        - size: 1234567890 (bytes)
        - modified_at: timestamp
        - digest: sha256
        - details: {
            "format": "gguf",
            "family": "llama",
            "parameter_size": "70B",
            "quantization_level": "q4_k_m",
            "architecture": "llama"
          }

        Args:
            model_data: Raw model data from Ollama API

        Returns:
            Parsed ModelMetadata
        """
        name = model_data.get("name", "")
        details = model_data.get("details", {})

        # Parse parameter count
        param_size = details.get("parameter_size", "")
        param_count = self._parse_parameter_size(param_size)

        # Detect MoE
        family = details.get("family", "").lower()
        architecture = details.get("architecture", "").lower()
        is_moe = "moe" in family or "mixture" in family or "moe" in architecture

        # Active parameters for MoE (estimate if not explicit)
        active_params = param_count
        if is_moe and param_count:
            # Most MoE models have ~10-30% active params per token
            # e.g., Mixtral 8x7B has 7B active out of 56B total
            active_params = int(param_count * 0.2)  # conservative 20%

        # Capability detection from family/architecture
        supports_vision = self._detect_vision_from_details(details)
        supports_tools = self._detect_tools_from_details(details)

        return ModelMetadata(
            name=name,
            supports_vision=supports_vision,
            supports_tool_calling=supports_tools,
            supports_embedding=False,  # Ollama models rarely are embedding-only
            is_chat_model=True,
            parameter_count=param_count,
            is_moe=is_moe,
            active_parameters=active_params,
            quantization=details.get("quantization_level"),
            details_source="ollama_api",
            confidence=0.9,
        )

    def _parse_parameter_size(self, size_str: str) -> int | None:
        """Parse parameter size string to integer count.

        Examples:
            "7B" -> 7_000_000_000
            "70B" -> 70_000_000_000
            "13B" -> 13_000_000_000
            "8x7B" -> 56_000_000_000 (for MoE)

        Args:
            size_str: Parameter size string (e.g., "7B", "70B")

        Returns:
            Parameter count as integer, or None if unparseable
        """
        if not size_str:
            return None

        size_str = size_str.strip().lower()
        if size_str.endswith("b"):
            try:
                num = float(size_str[:-1])
                return int(num * 1_000_000_000)
            except ValueError:
                return None
        return None

    def _detect_vision_from_details(self, details: dict[str, Any]) -> bool:
        """Detect vision capability from model details."""
        family = details.get("family", "").lower()
        arch = details.get("architecture", "").lower()

        # Known vision families
        vision_families = ["llava", "pixtral", "bakllava", "cogvlm", "moondream", "internvl"]
        if any(vf in family for vf in vision_families):
            return True

        # Check architecture hints
        vision_archs = ["vision", "multimodal", "image"]
        if any(va in arch for va in vision_archs):
            return True

        return False

    def _detect_tools_from_details(self, details: dict[str, Any]) -> bool:
        """Detect tool calling capability from model details."""
        family = details.get("family", "").lower()
        arch = details.get("architecture", "").lower()

        # Most modern chat models support tools
        tool_families = ["gpt-4", "claude-3", "qwen2.5", "mistral-large", "command-r", "llama3.1", "llama3.2", "gemma4"]
        if any(tf in family for tf in tool_families):
            return True

        # Check for tool support in architecture
        if "tool" in arch:
            return True

        return False

    def _detect_by_patterns(self, model_name: str) -> ModelMetadata:
        """Detect capabilities using configurable pattern matching.

        Pattern sources (in order):
        1. Explicitly configured capabilities from settings
        2. Built-in pattern fallbacks (for known model families)

        Args:
            model_name: Model name to analyze

        Returns:
            ModelMetadata with pattern-based detection
        """
        name_lower = model_name.lower()

        # 1. Check explicit configuration from settings
        # This allows users to define capabilities in config
        if hasattr(settings, "model_capability_patterns") and settings.model_capability_patterns:
            for pattern_config in settings.model_capability_patterns:
                pattern = pattern_config.get("pattern", "")
                if pattern and re.search(pattern, model_name, re.IGNORECASE):
                    return ModelMetadata(
                        name=model_name,
                        supports_vision=pattern_config.get("vision", False),
                        supports_tool_calling=pattern_config.get("tool_calling", False),
                        supports_embedding=pattern_config.get("embedding", False),
                        details_source="config",
                        confidence=1.0,
                    )

        # 2. Built-in pattern fallbacks (hardcoded for known families)
        # These serve as defaults and can be overridden by config patterns
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
            "gemma4",
        ]
        if any(ind in name_lower for ind in vision_indicators):
            return ModelMetadata(
                name=model_name,
                supports_vision=True,
                supports_tool_calling=True,  # Vision models usually support tools
                details_source="pattern",
                confidence=0.8,
            )

        tool_indicators = [
            "gpt-4",
            "claude-3",
            "qwen2.5",
            "mistral-large",
            "command-r",
            "llama3.1",
            "llama3.2",
            "gemma4",
        ]
        if any(ind in name_lower for ind in tool_indicators):
            return ModelMetadata(
                name=model_name,
                supports_tool_calling=True,
                details_source="pattern",
                confidence=0.8,
            )

        # Embedding models (rare in Ollama, mostly via external providers)
        embedding_indicators = ["embed", "nomic", "mxbai", "bge-", "gte-", "e5-"]
        if any(ind in name_lower for ind in embedding_indicators):
            return ModelMetadata(
                name=model_name,
                supports_embedding=True,
                is_chat_model=False,
                details_source="pattern",
                confidence=0.7,
            )

        # Default: basic text chat model
        return ModelMetadata(
            name=model_name,
            supports_vision=False,
            supports_tool_calling=False,
            supports_embedding=False,
            details_source="pattern",
            confidence=0.5,
        )

    async def _query_ollama_model(self, model_name: str) -> ModelMetadata | None:
        """Query Ollama API for specific model details.

        Args:
            model_name: Model name to query

        Returns:
            ModelMetadata if found, None otherwise
        """
        if not self._ollama_url:
            return None

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    urljoin(self._ollama_url, "/api/tags"),
                    timeout=5.0,
                )
                response.raise_for_status()
                data = response.json()

                for model_data in data.get("models", []):
                    if model_data.get("name") == model_name:
                        return self._parse_ollama_model_data(model_data)

                return None
        except Exception as e:
            logger.debug(f"Failed to query Ollama for model {model_name}: {e}")
            return None


# Global registry instance
_metadata_registry: ModelMetadataRegistry | None = None


def get_metadata_registry() -> ModelMetadataRegistry:
    """Get or create the global metadata registry.

    Returns:
        ModelMetadataRegistry instance
    """
    if _metadata_registry is None:
        _metadata_registry = ModelMetadataRegistry(ttl_seconds=300)
        # Set Ollama URL from app state if available
        try:
            # Lazy import to avoid circular dependency
            from router.state import app_state

            if app_state and app_state.backend:
                # Extract base URL from backend
                base_url = getattr(app_state.backend, "base_url", None)
                if base_url:
                    _metadata_registry._ollama_url = base_url
        except Exception:
            pass
    return _metadata_registry


# Convenience function

# Convenience function
async def get_model_metadata(model_name: str) -> ModelMetadata:
    """Get metadata for a single model.

    Args:
        model_name: Model name

    Returns:
        ModelMetadata with detected capabilities
    """
    registry = get_metadata_registry()
    return await registry.get_metadata(model_name)


# Synchronous wrapper for compatibility with existing code
def get_model_metadata_sync(model_name: str) -> ModelMetadata:
    """Synchronous wrapper for model metadata lookup.

    Uses asyncio.run() internally. For use in sync contexts only.

    Args:
        model_name: Model name

    Returns:
        ModelMetadata with detected capabilities
    """
    try:
        import asyncio

        return asyncio.run(get_model_metadata(model_name))
    except RuntimeError:
        # Already in event loop, create task
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.create_task(get_model_metadata(model_name))
            # This is risky but we do it for compatibility
            return loop.run_until_complete(task)
        return loop.run_until_complete(get_model_metadata(model_name))
