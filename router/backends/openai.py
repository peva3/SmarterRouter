import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from router.backends.base import LLMBackend, ModelInfo

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible API backend with persistent HTTP client.

    Supports OpenAI, Anthropic (via compatibility layer), and any
    other OpenAI-compatible API with authentication.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model_prefix: str = "",
        timeout: float = 120.0,
        models_cache_ttl: float = 30.0,  # Cache model list for 30 seconds
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_prefix = model_prefix
        self.timeout = timeout
        self.models_cache_ttl = models_cache_ttl
        self._models_cache: tuple[list[ModelInfo], float] | None = None  # (models, timestamp)
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client for connection reuse."""
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout, connect=10.0),
                        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                    )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client. Call during shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = await self._get_client()
        
        # Create a new client with specific timeout if needed
        if effective_timeout != self.timeout:
            async with httpx.AsyncClient(timeout=effective_timeout) as temp_client:
                response = await temp_client.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    logger.warning(f"Unexpected response type from {method} {url}: {type(data)}")
                    return {}
                return data
        
        response = await client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            logger.warning(f"Unexpected response type from {method} {url}: {type(data)}")
            return {}
        return data

    async def list_models(self) -> list[ModelInfo]:
        """List available models with caching to avoid repeated HTTP requests."""
        now = time.monotonic()
        
        # Check cache
        if self._models_cache is not None:
            models, cached_at = self._models_cache
            if now - cached_at < self.models_cache_ttl:
                return models
        
        # Fetch fresh models
        try:
            data = await self._request("GET", "/models")
            models = []
            for m in data.get("data", []):
                models.append(
                    ModelInfo(
                        name=m.get("id", ""),
                        size=m.get("size"),
                        modified_at=None,
                    )
                )
            
            # Update cache
            self._models_cache = (models, now)
            return models
        except httpx.HTTPError as e:
            # If fetch fails but we have cached data, return it
            if self._models_cache is not None:
                logger.warning(f"Failed to refresh model list: {e}, returning stale cache")
                return self._models_cache[0]
            logger.error(f"Failed to list models: {e}")
            return []

    def invalidate_models_cache(self) -> None:
        """Manually invalidate the models cache."""
        self._models_cache = None

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
        }
        payload.update(kwargs)
        response = await self._request("POST", "/chat/completions", json=payload)
        
        # Transform OpenAI format to Ollama format for consistency
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            usage = response.get("usage", {})
            return {
                "message": {"content": content},
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
            }
        return {"message": {"content": ""}}

    async def chat_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
        keep_alive: float = -1,
    ) -> tuple[AsyncIterator[dict[str, Any]], float]:
        url = f"{self.base_url}/chat/completions"
        full_model = self._full_model_name(model)

        start_time = time.perf_counter()
        first_token_time: float | None = None
        latency_ms = 0.0

        async def stream_generator() -> AsyncIterator[dict[str, Any]]:
            nonlocal latency_ms, first_token_time
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json={
                        "model": full_model,
                        "messages": messages,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        # Handle [DONE] sentinel
                        if line.startswith("data: [DONE]"):
                            yield {"done": True}
                            continue
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                # Measure time to first token
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                    latency_ms = (first_token_time - start_time) * 1000
                                # Normalize to Ollama format (already done in stream)
                                content = ""
                                finish_reason = None
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    content = choice.get("delta", {}).get("content", "")
                                    finish_reason = choice.get("finish_reason")
                                yield {
                                    "message": {"content": content},
                                    "done": finish_reason == "stop",
                                }
                            except json.JSONDecodeError:
                                continue

        return stream_generator(), latency_ms

    async def unload_model(self, model_name: str) -> bool:
        """External APIs don't support model unloading."""
        logger.debug(f"unload_model not supported for external API backend")
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
        return await self._request("POST", "/embeddings", json=payload)
