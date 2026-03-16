"""Concurrency stress tests for semantic cache (Item #56)."""

import asyncio

import pytest

from router.backends.base import ModelInfo
from router.router import RouterEngine, RoutingResult


class _DummyBackend:
    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="model-a")]

    async def chat(self, *args, **kwargs) -> dict:
        return {"message": {"content": "ok"}}

    async def chat_streaming(self, *args, **kwargs):
        raise NotImplementedError

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def load_model(self, model_name: str, keep_alive: float = -1, timeout: float | None = None) -> bool:
        return False

    async def embed(self, model: str, input_text, **kwargs) -> dict:
        return {"embedding": [0.1, 0.2, 0.3]}

    async def get_model_vram_usage(self, model_name: str) -> float | None:
        return None

    async def close(self) -> None:
        return None

    def is_external_model(self, model_name: str) -> bool:
        return False


@pytest.fixture
def cache_engine() -> RouterEngine:
    return RouterEngine(client=_DummyBackend(), cache_enabled=True)


@pytest.mark.asyncio
async def test_concurrent_cache_set_get(cache_engine: RouterEngine) -> None:
    if not cache_engine.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    cache = cache_engine.semantic_cache

    async def worker(i: int) -> bool:
        prompt = f"prompt-{i}"
        result = RoutingResult(selected_model="model-a", confidence=0.8, reasoning="stress")
        await cache.set(prompt, result)
        loaded = await cache.get(prompt)
        return loaded is not None and loaded.selected_model == "model-a"

    tasks = [worker(i) for i in range(200)]
    outcomes = await asyncio.gather(*tasks)
    assert all(outcomes)


@pytest.mark.asyncio
async def test_concurrent_response_cache_set_get(cache_engine: RouterEngine) -> None:
    if not cache_engine.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    cache = cache_engine.semantic_cache

    async def worker(i: int) -> bool:
        prompt = f"resp-{i}"
        response = f"value-{i}"
        await cache.set_response("model-a", prompt, response)
        loaded = await cache.get_response("model-a", prompt)
        return loaded == response

    tasks = [worker(i) for i in range(200)]
    outcomes = await asyncio.gather(*tasks)
    assert all(outcomes)
