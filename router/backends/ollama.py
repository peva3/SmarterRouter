import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from router.backends.base import LLMBackend, ModelInfo

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Ollama backend implementation."""

    def __init__(
        self,
        base_url: str,
        model_prefix: str = "",
        timeout: float = 60.0,
        generation_timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_prefix = model_prefix
        self.timeout = timeout  # Short timeout for quick operations (list, etc)
        self.generation_timeout = generation_timeout  # Longer timeout for generation

    def _full_model_name(self, model: str) -> str:
        """Apply model prefix if configured."""
        if self.model_prefix and not model.startswith(self.model_prefix):
            return f"{self.model_prefix}{model}"
        return model

    async def _request(
        self,
        method: str,
        path: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        effective_timeout = timeout if timeout is not None else self.timeout
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

    async def list_models(self) -> list[ModelInfo]:
        try:
            data = await self._request("GET", "/api/tags")
            models = []
            for m in data.get("models", []):
                models.append(
                    ModelInfo(
                        name=m["name"],
                        size=m.get("size", 0),
                        modified_at=m.get("modified_at", ""),
                    )
                )
            return models
        except httpx.HTTPError as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        keep_alive: float = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        full_model = self._full_model_name(model)
        payload: dict[str, Any] = {
            "model": full_model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive,
        }
        payload.update(kwargs)
        return await self._request(
            "POST", "/api/chat", json=payload, timeout=self.generation_timeout
        )

    async def chat_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
        keep_alive: float = -1,
    ) -> tuple[AsyncIterator[dict[str, Any]], float]:
        url = f"{self.base_url}/api/chat"
        full_model = self._full_model_name(model)

        timing = {"start": time.perf_counter(), "first_token": None}
        latency_ms = 0.0

        async def stream_generator() -> AsyncIterator[dict[str, Any]]:
            nonlocal latency_ms
            async with httpx.AsyncClient(timeout=self.generation_timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={
                        "model": full_model,
                        "messages": messages,
                        "stream": True,
                        "keep_alive": keep_alive,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            # Measure time to first token
                            if timing["first_token"] is None:
                                timing["first_token"] = time.perf_counter()
                                latency_ms = (timing["first_token"] - timing["start"]) * 1000
                            yield json.loads(line)

        return stream_generator(), latency_ms

    async def unload_model(self, model_name: str) -> bool:
        """Unload model from VRAM by setting keep_alive to 0."""
        logger.info(f"Attempting to unload model: {model_name}")
        try:
            await self._request(
                "POST",
                "/api/chat",  # Use chat endpoint for unloading (keeps context clean)
                json={"model": model_name, "messages": [], "keep_alive": 0},
            )
            logger.info(f"Sent unload request for {model_name}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"Model {model_name} not found or already unloaded.")
                return True
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error during unload model {model_name}: {e}")
            return False

    async def load_model(self, model_name: str, keep_alive: float = -1) -> bool:
        """Explicitly load a model into VRAM with optional keep_alive duration.

        Args:
            model_name: Name of the model to load
            keep_alive: Duration to keep model in VRAM. -1 = forever, 0 = unload after,
                       positive number = seconds to keep loaded
        """
        logger.info(f"Loading model: {model_name} (keep_alive={keep_alive})")
        try:
            await self._request(
                "POST",
                "/api/chat",
                json={"model": model_name, "messages": [{"role": "user", "content": ""}], "keep_alive": keep_alive},
                timeout=self.generation_timeout,
            )
            logger.info(f"Model {model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    async def embed(
        self,
        model: str,
        input_text: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        full_model = self._full_model_name(model)
        payload: dict[str, Any] = {
            "model": full_model,
            "input": input_text,
        }
        payload.update(kwargs)
        return await self._request(
            "POST", "/api/embed", json=payload, timeout=self.generation_timeout
        )

    async def get_running_models(self) -> dict[str, dict[str, Any]]:
        """Get currently loaded models with their VRAM usage from Ollama.
        
        Returns:
            Dict mapping model names to their info including:
            - vram_bytes: VRAM usage in bytes
            - size_bytes: Model size in bytes
            - digest: Model digest
        """
        try:
            data = await self._request("GET", "/api/ps")
            models = {}
            for m in data.get("models", []):
                name = m.get("name", "")
                if name:
                    models[name] = {
                        "vram_bytes": m.get("size_vram", 0),
                        "size_bytes": m.get("size", 0),
                        "digest": m.get("digest", ""),
                    }
            return models
        except httpx.HTTPError as e:
            logger.error(f"Failed to get running models: {e}")
            return {}

    async def get_model_vram_usage(self, model_name: str) -> float | None:
        """Get VRAM usage for a specific model in GB.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            VRAM usage in GB, or None if model not loaded or error
        """
        running = await self.get_running_models()
        # Try exact match first
        if model_name in running:
            vram_bytes = running[model_name].get("vram_bytes", 0)
            if vram_bytes > 0:
                return vram_bytes / (1024 ** 3)  # Convert bytes to GB
        
        # Try with prefix if configured
        full_name = self._full_model_name(model_name)
        if full_name in running:
            vram_bytes = running[full_name].get("vram_bytes", 0)
            if vram_bytes > 0:
                return vram_bytes / (1024 ** 3)
        
        return None
