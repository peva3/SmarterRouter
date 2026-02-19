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
    # Mock the chat method to return Ollama-format response
    mock_client.chat = AsyncMock(
        return_value={"message": {"content": "This is a test response"}}
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
        return {"message": {"content": ""}}

    mock_client.chat = AsyncMock(side_effect=asyncio.TimeoutError())

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

    # Mock chat method with Ollama-format response
    mock_client.chat = AsyncMock(
        return_value={"message": {"content": "Response text here"}}
    )

    score, avg_time = await profiler._test_category("llama3", "coding", prompts)

    # Now using parallel processing, so we check chat was called 3 times total
    assert mock_client.chat.call_count == 3
    assert score > 0


@pytest.mark.asyncio
async def test_test_category_all_empty_responses(profiler, mock_client):
    prompts = ["prompt1", "prompt2"]

    mock_client.chat = AsyncMock(return_value={"message": {"content": ""}})

    score, avg_time = await profiler._test_category("llama3", "coding", prompts)

    assert score == 0.0


@pytest.mark.asyncio
async def test_adaptive_timeout_small_model(profiler, mock_client):
    """Test that small models get shorter timeouts."""
    # 1B model should get 0.8x timeout
    small_profiler = ModelProfiler(mock_client, model_name="phi3:1b")
    assert small_profiler.timeout == small_profiler.base_timeout * 0.8


@pytest.mark.asyncio
async def test_adaptive_timeout_medium_model(profiler, mock_client):
    """Test that medium models (7B) get moderate timeouts."""
    # 7B model should get 1.1x timeout
    medium_profiler = ModelProfiler(mock_client, model_name="llama3.1:8b")
    assert medium_profiler.timeout == medium_profiler.base_timeout * 1.1


@pytest.mark.asyncio
async def test_adaptive_timeout_medium_large_model(profiler, mock_client):
    """Test that medium-large models (14B) get longer timeouts."""
    # 14B model should get 1.4x timeout
    medium_large_profiler = ModelProfiler(mock_client, model_name="qwen:14b")
    assert medium_large_profiler.timeout == medium_large_profiler.base_timeout * 1.4


@pytest.mark.asyncio
async def test_adaptive_timeout_large_model(profiler, mock_client):
    """Test that large models (30B+) get even longer timeouts."""
    # 30B model should get 1.8x timeout
    large_profiler = ModelProfiler(mock_client, model_name="qwen:30b")
    assert large_profiler.timeout == large_profiler.base_timeout * 1.8


@pytest.mark.asyncio
async def test_adaptive_timeout_very_large_model(profiler, mock_client):
    """Test that very large models (70B+) get the longest timeouts."""
    # 70B model should get 2.5x timeout
    very_large_profiler = ModelProfiler(mock_client, model_name="llama3:70b")
    assert very_large_profiler.timeout == very_large_profiler.base_timeout * 2.5


@pytest.mark.asyncio
async def test_profile_model_vram_via_ollama_api(profiler, mock_client):
    """Test that VRAM is measured via Ollama API when available."""
    from router.judge import JudgeClient
    
    # Mock client.chat for successful responses (need >50 chars to pass screening)
    long_response = "This is a test response that is long enough to pass the screening heuristic. It needs to be at least fifty characters long."
    mock_client.chat = AsyncMock(
        return_value={"message": {"content": long_response}}
    )
    
    # Mock Ollama API VRAM method
    mock_client.get_model_vram_usage = AsyncMock(return_value=2.5)
    
    # Mock judge to avoid API calls
    profiler.judge = MagicMock()
    profiler.judge.score_responses_batch = AsyncMock(return_value=[0.8, 0.8, 0.8, 0.8, 0.8])
    
    result = await profiler.profile_model("llama3.2:1b")
    
    assert result is not None
    # Verify Ollama API was called for VRAM
    mock_client.get_model_vram_usage.assert_called_once_with("llama3.2:1b")


@pytest.mark.asyncio
async def test_profile_model_vram_fallback_to_nvidia_smi(profiler, mock_client):
    """Test VRAM fallback to nvidia-smi when Ollama API returns None."""
    from router.judge import JudgeClient
    import router.profiler as profiler_module
    
    # Mock client.chat for successful responses (need >50 chars to pass screening)
    long_response = "This is a test response that is long enough to pass the screening heuristic. It needs to be at least fifty characters long."
    mock_client.chat = AsyncMock(
        return_value={"message": {"content": long_response}}
    )
    
    # Mock Ollama API to return None (model not found in running models)
    mock_client.get_model_vram_usage = AsyncMock(return_value=None)
    
    # Mock nvidia-smi measurement
    profiler._measure_vram_gb_async = AsyncMock(return_value=8.0)
    
    # Mock judge
    profiler.judge = MagicMock()
    profiler.judge.score_responses_batch = AsyncMock(return_value=[0.8, 0.8, 0.8, 0.8, 0.8])
    
    result = await profiler.profile_model("llama3.2:1b")
    
    assert result is not None
    # Verify Ollama API was attempted
    mock_client.get_model_vram_usage.assert_called_once_with("llama3.2:1b")
