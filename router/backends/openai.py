import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from router.backends.base import ModelInfo

logger = logging.getLogger(__name__)


class OpenAIBackend:
    """OpenAI-compatible API backend.

    Supports OpenAI, Anthropic (via compatibility layer), and any
    other OpenAI-compatible API with authentication.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model_prefix: str = "",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_prefix = model_prefix
        self.timeout = timeout

    def _full_model_name(self, model: str) -> str:
        """Apply model prefix if configured."""
        if self.model_prefix and not model.startswith(self.model_prefix):
            return f"{self.model_prefix}{model}"
        return model

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method, url, headers=headers, **kwargs
            )
            response.raise_for_status()
            return response.json()

    async def list_models(self) -> list[ModelInfo]:
        try:
            data = await self._request("GET", "/v1/models")
            models = []
            for m in data.get("data", []):
                models.append(
                    ModelInfo(
                        name=m.get("id", ""),
                        size=m.get("size"),
                        modified_at=None,
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        full_model = self._full_model_name(model)
        payload: dict[str, Any] = {
            "model": full_model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(kwargs)
        return await self._request("POST", "/v1/chat/completions", json=payload)

    async def chat_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> tuple[AsyncIterator[dict[str, Any]], float]:
        url = f"{self.base_url}/v1/chat/completions"
        full_model = self._full_model_name(model)
        start_time = time.perf_counter()

        async def stream_generator() -> AsyncIterator[dict[str, Any]]:
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
                        if line.strip() and line.startswith("data: "):
                            yield json.loads(line[6:])

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return stream_generator(), elapsed_ms

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        full_model = self._full_model_name(model)
        payload: dict[str, Any] = {
            "model": full_model,
            "prompt": prompt,
            "stream": stream,
        }
        payload.update(kwargs)
        return await self._request("POST", "/v1/completions", json=payload)

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
        return await self._request("POST", "/v1/embeddings", json=payload)
