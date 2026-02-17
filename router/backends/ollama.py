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

    def __init__(self, base_url: str, timeout: float = 60.0, generation_timeout: float = 120.0):
        self.base_url = base_url
        self.timeout = timeout  # Short timeout for quick operations (list, etc)
        self.generation_timeout = generation_timeout  # Longer timeout for generation

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

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        payload.update(kwargs)
        return await self._request("POST", "/api/generate", json=payload, timeout=self.generation_timeout)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        keep_alive: float = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive,
        }
        payload.update(kwargs)
        return await self._request("POST", "/api/chat", json=payload, timeout=self.generation_timeout)

    async def chat_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
        keep_alive: float = -1,
    ) -> tuple[AsyncIterator[dict[str, Any]], float]:
        url = f"{self.base_url}/api/chat"
        
        # Use a mutable container to track timing inside the generator
        timing = {"start": time.perf_counter(), "first_token": None}
        latency_ms = 0.0

        async def stream_generator() -> AsyncIterator[dict[str, Any]]:
            nonlocal latency_ms
            async with httpx.AsyncClient(timeout=self.generation_timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={
                        "model": model,
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
                "/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
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
                "/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": keep_alive},
                timeout=self.generation_timeout,
            )
            logger.info(f"Model {model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    async def ensure_model_loaded(self, model_name: str, pinned_model: str | None = None) -> bool:
        """Ensure a model is loaded in VRAM, unloading others if necessary.
        
        This is the key method for proactive VRAM management:
        1. First unload any model that's NOT the target model and NOT the pinned model
        2. Then load the target model
        
        Args:
            model_name: The model we want to use
            pinned_model: A model that should be kept in VRAM (e.g., a small fast model)
        
        Returns:
            True if the model is ready for inference
        """
        current = getattr(self, '_current_model', None)
        
        if current == model_name:
            logger.debug(f"Model {model_name} already loaded")
            return True
        
        if current and current != pinned_model:
            logger.info(f"VRAM management: unloading {current} to load {model_name}")
            await self.unload_model(current)
        
        if model_name != pinned_model:
            logger.info(f"VRAM management: loading {model_name}")
            await self.load_model(model_name)
        
        self._current_model = model_name
        return True

    async def embed(
        self,
        model: str,
        input_text: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": input_text,
        }
        payload.update(kwargs)
        return await self._request("POST", "/api/embed", json=payload, timeout=self.generation_timeout)
