"""Cache persistence recovery tests (Item #58).

Uses the real RouterEngine/SemanticCache APIs and verifies persistent cache
behavior through reload scenarios.
"""

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
def engine() -> RouterEngine:
    return RouterEngine(client=_DummyBackend(), cache_enabled=True)


@pytest.mark.asyncio
async def test_routing_entry_round_trip_via_persistence(engine: RouterEngine) -> None:
    if not engine.semantic_cache or not engine.semantic_cache.persistent_cache:
        pytest.skip("Persistent cache is not enabled")

    prompt = "cache persistence prompt"
    result = RoutingResult(
        selected_model="model-a",
        confidence=0.9,
        reasoning="test",
    )

    await engine.semantic_cache.set(prompt, result, embedding=[0.1, 0.2, 0.3])

    # Simulate restart/load into a fresh cache object
    other = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    if not other.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    await other.semantic_cache.load_from_persistence()
    loaded = await other.semantic_cache.get(prompt, embedding=[0.1, 0.2, 0.3])

    assert loaded is not None
    assert loaded.selected_model == "model-a"


@pytest.mark.asyncio
async def test_response_entry_round_trip_via_persistence(engine: RouterEngine) -> None:
    if not engine.semantic_cache or not engine.semantic_cache.persistent_cache:
        pytest.skip("Persistent cache is not enabled")

    model = "model-a"
    prompt = "response cache prompt"
    text = "cached response"

    await engine.semantic_cache.set_response(model=model, prompt=prompt, response=text)

    other = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    if not other.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    await other.semantic_cache.load_from_persistence()
    loaded = await other.semantic_cache.get_response(model=model, prompt=prompt)

    assert loaded == text


@pytest.mark.asyncio
async def test_clear_removes_entries_from_memory(engine: RouterEngine) -> None:
    if not engine.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    prompt = "memory clear prompt"
    result = RoutingResult(selected_model="model-a", confidence=0.5, reasoning="test")
    await engine.semantic_cache.set(prompt, result)

    assert await engine.semantic_cache.get(prompt) is not None
    await engine.semantic_cache.clear()
    assert await engine.semantic_cache.get(prompt) is None
