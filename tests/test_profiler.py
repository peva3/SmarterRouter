from unittest.mock import AsyncMock, MagicMock

import pytest

from router.profiler import ModelProfiler
from router.prompts import BENCHMARK_PROMPTS


@pytest.fixture
def mock_client():
    client = AsyncMock()
    return client


@pytest.fixture
def profiler(mock_client):
    return ModelProfiler(mock_client)


@pytest.mark.asyncio
async def test_profile_model_success(profiler, mock_client):
    mock_client.generate = AsyncMock(
        return_value=MagicMock(response="This is a test response", done=True)
    )

    result = await profiler.profile_model("llama3")

    assert result is not None
    assert result.model_name == "llama3"
    assert 0 <= result.reasoning <= 1.0
    assert 0 <= result.coding <= 1.0
    assert 0 <= result.creativity <= 1.0
    assert 0 <= result.speed <= 1.0


@pytest.mark.asyncio
async def test_profile_model_timeout(profiler, mock_client):
    import asyncio

    async def timeout_response(*args, **kwargs):
        await asyncio.sleep(10)
        return MagicMock(response="", done=True)

    mock_client.generate = AsyncMock(side_effect=TimeoutError())

    result = await profiler.profile_model("llama3")

    assert result is not None
    assert result.reasoning == 0.0


def test_category_prompts_exist():
    assert "reasoning" in BENCHMARK_PROMPTS
    assert "coding" in BENCHMARK_PROMPTS
    assert "creativity" in BENCHMARK_PROMPTS

    for category in BENCHMARK_PROMPTS:
        assert len(BENCHMARK_PROMPTS[category]) > 0


@pytest.mark.asyncio
async def test_test_category_multiple_prompts(profiler, mock_client):
    prompts = ["prompt1", "prompt2", "prompt3"]

    mock_client.generate = AsyncMock(
        return_value=MagicMock(response="Response text here", done=True)
    )

    score, avg_time = await profiler._test_category("llama3", "coding", prompts)

    assert mock_client.generate.call_count == 3
    assert score > 0


@pytest.mark.asyncio
async def test_test_category_all_empty_responses(profiler, mock_client):
    prompts = ["prompt1", "prompt2"]

    mock_client.generate = AsyncMock(return_value=MagicMock(response="", done=True))

    score, avg_time = await profiler._test_category("llama3", "coding", prompts)

    assert score == 0.0
