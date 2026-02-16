import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from router.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    name: str
    size: int
    modified_at: str


@dataclass
class GenerateResponse:
    response: str
    done: bool
    context: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = base_url or settings.ollama_url
        self.timeout = timeout

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
    ) -> GenerateResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        payload.update(kwargs)

        try:
            data = await self._request("POST", "/api/generate", json=payload)

            return GenerateResponse(
                response=data.get("response", ""),
                done=data.get("done", True),
                context=data.get("context"),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                eval_count=data.get("eval_count"),
            )
        except httpx.HTTPError as e:
            logger.error(f"Generate failed for model {model}: {e}")
            raise

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(kwargs)

        return await self._request("POST", "/api/chat", json=payload)

    async def chat_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> tuple[AsyncIterator[dict[str, Any]], float]:
        url = f"{self.base_url}/api/chat"
        start_time = time.perf_counter()

        async def stream_generator() -> AsyncIterator[dict[str, Any]]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={"model": model, "messages": messages, "stream": True},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield json.loads(line)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return stream_generator(), elapsed_ms

    async def unload_model(self, model_name: str) -> bool:
        """Explicitly unloads a model from VRAM by setting keep_alive to 0."""
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


async def create_client(timeout: float = 60.0) -> OllamaClient:
    return OllamaClient(timeout=timeout)
