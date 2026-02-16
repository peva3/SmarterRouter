import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from router.backends.base import ModelInfo

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """Llama.cpp server backend implementation.

    Compatible with llama.cpp server, llama-swap, and other
    OpenAI-compatible servers that don't require API keys.
    """

    def __init__(
        self,
        base_url: str,
        model_prefix: str = "",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(method, url, **kwargs)
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
        
        timing = {"start": time.perf_counter(), "first_token": None}
        latency_ms = 0.0

        async def stream_generator() -> AsyncIterator[dict[str, Any]]:
            nonlocal latency_ms
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    url,
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
                                if timing["first_token"] is None:
                                    timing["first_token"] = time.perf_counter()
                                    latency_ms = (timing["first_token"] - timing["start"]) * 1000
                                # Normalize to Ollama format
                                content = ""
                                finish_reason = None
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    content = choice.get("delta", {}).get("content", "")
                                    finish_reason = choice.get("finish_reason")
                                yield {
                                    "message": {"content": content},
                                    "done": finish_reason == "stop"
                                }
                            except json.JSONDecodeError:
                                continue

        return stream_generator(), latency_ms

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
        """Llama.cpp doesn't support explicit unloading via API."""
        logger.debug(f"unload_model not supported for llama.cpp backend")
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
